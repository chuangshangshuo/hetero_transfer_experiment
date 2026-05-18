from __future__ import annotations

import copy
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.train.train_transfer import (
    build_metapath_adjacency,
    choose_threshold_by_youden,
    evaluate_binary,
    load_encoder_state,
    make_finetune_model,
)
from src.train.utils import GraphBundle, resolve_device, set_random_seed


def finite_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if np.unique(y_true).size < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def sample_fewshot_support(
    split_frame: pd.DataFrame,
    shots_per_class: int,
    seed: int,
    source: str = "target_adapt_unlabeled",
) -> pd.DataFrame:
    pool = split_frame[split_frame["split"] == source].copy()
    if pool["label"].nunique() < 2:
        raise ValueError("Few-shot support requires a binary target adaptation pool")
    rng = np.random.default_rng(seed * 7919 + int(shots_per_class))
    selected_parts = []
    for label, part in pool.groupby("label"):
        take = min(int(shots_per_class), len(part))
        chosen = rng.choice(part.index.to_numpy(), size=take, replace=False)
        selected_parts.append(pool.loc[chosen])
    support = pd.concat(selected_parts, ignore_index=False).copy()
    support["split"] = "target_fewshot_train"
    support["fewshot_shots_per_class"] = int(shots_per_class)
    support["fewshot_support_source"] = source
    return support


def build_fewshot_split(split_frame: pd.DataFrame, shots_per_class: int, seed: int) -> pd.DataFrame:
    support = sample_fewshot_support(split_frame, shots_per_class=shots_per_class, seed=seed)
    support_nodes = set(support["node_id"].astype(str))
    adjusted = split_frame.copy()
    keep_mask = ~(
        adjusted["split"].eq("target_adapt_unlabeled")
        & adjusted["node_id"].astype(str).isin(support_nodes)
    )
    adjusted = adjusted[keep_mask].copy()
    adjusted = pd.concat([adjusted, support], ignore_index=True)
    adjusted["target_labels_used_for_training"] = adjusted["split"].eq("target_fewshot_train")
    return adjusted.reset_index(drop=True)


def build_label_vector(bundle: GraphBundle, split_frame: pd.DataFrame) -> torch.Tensor:
    labels = torch.full((len(bundle.website_frame),), -1, dtype=torch.long)
    for _, row in split_frame.iterrows():
        labels[int(row["graph_node_index"])] = int(row["label"])
    return labels


def index_tensor(split_frame: pd.DataFrame, split_name: str, device: torch.device) -> torch.Tensor:
    return torch.tensor(
        split_frame.loc[split_frame["split"] == split_name, "graph_node_index"].to_numpy(dtype=np.int64),
        dtype=torch.long,
        device=device,
    )


def run_fewshot_finetune(
    bundle: GraphBundle,
    graph_data: Any,
    split_frame: pd.DataFrame,
    seed: int,
    shots_per_class: int,
    smoke_test: bool = False,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    set_random_seed(seed)
    fewshot_split = build_fewshot_split(split_frame, shots_per_class=shots_per_class, seed=seed)
    encoder_state, encoder_checkpoint = load_encoder_state(bundle, seed)
    device = resolve_device(bundle.config)
    data = copy.deepcopy(graph_data).to(device)
    metapath_adjacency = build_metapath_adjacency(bundle, graph_data, device)
    labels = build_label_vector(bundle, fewshot_split).to(device)
    source_train_idx = index_tensor(fewshot_split, "source_train", device)
    fewshot_idx = index_tensor(fewshot_split, "target_fewshot_train", device)
    source_val_idx = index_tensor(fewshot_split, "source_val", device)
    target_test_idx = index_tensor(fewshot_split, "target_test", device)
    train_idx = torch.cat([source_train_idx, fewshot_idx], dim=0)
    model = make_finetune_model(bundle, encoder_state, device)
    optimizer = torch.optim.Adam(
        [
            {"params": model.encoder_parameters(), "lr": float(bundle.config["finetune"]["encoder_lr"])},
            {"params": model.head_parameters(), "lr": float(bundle.config["finetune"]["classifier_lr"])},
        ],
        weight_decay=float(bundle.config["finetune"]["weight_decay"]),
    )

    max_epochs = 5 if smoke_test else int(bundle.config["finetune"].get("max_epochs", 200))
    patience = max_epochs if smoke_test else int(bundle.config["finetune"].get("patience", 25))
    min_epochs = 0 if smoke_test else int(bundle.config["finetune"].get("min_epochs_before_early_stop", 50))
    fewshot_weight = float(bundle.config.get("week9_fewshot", {}).get("target_loss_weight", 1.0))
    best_state: dict[str, Any] | None = None
    best_score = float("-inf")
    best_epoch = 0
    history_rows: list[dict[str, Any]] = []
    started = time.time()

    for epoch in range(1, max_epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits, _, _ = model(data, metapath_adjacency)
        source_loss = F.cross_entropy(logits[source_train_idx], labels[source_train_idx])
        target_loss = F.cross_entropy(logits[fewshot_idx], labels[fewshot_idx])
        loss = source_loss + fewshot_weight * target_loss
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            eval_logits, _, _ = model(data, metapath_adjacency)
            probabilities = torch.softmax(eval_logits, dim=1)[:, 1].detach().cpu().numpy()
        source_val_scores = probabilities[source_val_idx.detach().cpu().numpy()]
        target_test_scores = probabilities[target_test_idx.detach().cpu().numpy()]
        source_val_labels = fewshot_split.loc[fewshot_split["split"] == "source_val", "label"].to_numpy(dtype=int)
        target_test_labels = fewshot_split.loc[fewshot_split["split"] == "target_test", "label"].to_numpy(dtype=int)
        source_val_auc = finite_auc(source_val_labels, source_val_scores)
        target_auc = finite_auc(target_test_labels, target_test_scores)
        score = source_val_auc if math.isfinite(source_val_auc) else -float(loss.item())
        history_rows.append(
            {
                "epoch": epoch,
                "loss": float(loss.item()),
                "source_loss": float(source_loss.item()),
                "target_fewshot_loss": float(target_loss.item()),
                "source_val_auc": float(source_val_auc),
                "target_test_auc": float(target_auc),
                "runtime_s": float(time.time() - started),
            }
        )
        if math.isfinite(score) and score >= best_score:
            best_score = float(score)
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
        elif epoch >= min_epochs and best_state is not None and epoch - best_epoch >= patience:
            break

    if best_state is None:
        best_state = copy.deepcopy(model.state_dict())
        best_epoch = max_epochs
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits, _, _ = model(data, metapath_adjacency)
        probabilities_all = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()

    source_val = fewshot_split[fewshot_split["split"] == "source_val"].copy()
    target_test = fewshot_split[fewshot_split["split"] == "target_test"].copy()
    val_scores = probabilities_all[source_val["graph_node_index"].to_numpy(dtype=int)]
    threshold = choose_threshold_by_youden(source_val["label"].to_numpy(dtype=int), val_scores)
    target_scores = probabilities_all[target_test["graph_node_index"].to_numpy(dtype=int)]
    target_metrics = evaluate_binary(target_test["label"].to_numpy(dtype=int), target_scores, threshold)
    prediction_frame = fewshot_split.copy()
    prediction_frame["pred_prob_illegal"] = prediction_frame["graph_node_index"].map(
        lambda idx: float(probabilities_all[int(idx)])
    )
    prediction_frame["pred_label"] = (prediction_frame["pred_prob_illegal"] >= threshold).astype(int)
    prediction_frame["model"] = "week9_fewshot_finetune"
    metrics = {
        "transfer_id": str(split_frame["transfer_id"].dropna().iloc[0]),
        "method": "fewshot_finetune",
        "seed": int(seed),
        "shots_per_class": int(shots_per_class),
        "target_support_size": int(len(fewshot_idx)),
        "target_support_label_0": int((fewshot_split.loc[fewshot_split["split"] == "target_fewshot_train", "label"] == 0).sum()),
        "target_support_label_1": int((fewshot_split.loc[fewshot_split["split"] == "target_fewshot_train", "label"] == 1).sum()),
        "target_test_auc": target_metrics["roc_auc"],
        "target_pr_auc": target_metrics["pr_auc"],
        "target_balanced_acc": target_metrics["balanced_accuracy"],
        "target_macro_f1": target_metrics["macro_f1"],
        "target_illegal_consistency": target_metrics["illegal_consistency"],
        "target_licensed_consistency": target_metrics["licensed_consistency"],
        "target_mean_pred_illegal": target_metrics["mean_pred_illegal"],
        "best_epoch": int(best_epoch),
        "best_source_val_score": float(best_score),
        "trained_epochs": int(len(history_rows)),
        "threshold": float(threshold),
        "encoder_checkpoint": encoder_checkpoint,
        "is_posthoc": True,
        "runtime_s": float(time.time() - started),
    }
    return metrics, prediction_frame, pd.DataFrame(history_rows)
