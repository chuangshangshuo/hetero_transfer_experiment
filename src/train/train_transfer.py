"""Week-7 transfer runner: source_only and StruRW edge-reweighted training."""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.hetero_full_with_heco import HeCoFineTuneClassifier
from src.models.view_generators import build_metapath_artifacts
from src.train.train_contrastive import make_heco_model
from src.train.utils import (
    GraphBundle,
    build_primary_task_frame,
    choose_threshold_by_youden,
    compute_sample_counts,
    load_graph_bundle,
    make_common_manifest,
    metric_columns,
    record_run_manifest,
    resolve_device,
    resolve_workspace_path,
    save_dataframe,
    set_random_seed,
    summarise_runs,
    utc_now_iso,
)
from src.transfer.dann import HeCoDANNClassifier, lambda_schedule
from src.transfer.strurw_reweight import StruRWReweighter


def add_transfer_groups(frame: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    """Add transfer groups."""
    grouped = frame.copy().reset_index(drop=True)
    split_group = pd.Series([""] * len(grouped), index=grouped.index, dtype="object")
    split_group_source = pd.Series([""] * len(grouped), index=grouped.index, dtype="object")
    for column in group_columns:
        if column not in grouped.columns:
            continue
        values = grouped[column].fillna("").astype(str).str.strip()
        needs_value = split_group.eq("") & values.ne("")
        split_group.loc[needs_value] = column + "::" + values.loc[needs_value]
        split_group_source.loc[needs_value] = column
    if split_group.eq("").any():
        missing = grouped.loc[split_group.eq(""), "node_id"].head(10).tolist()
        raise ValueError(f"Unable to construct transfer split groups for node_ids: {missing}")
    grouped["split_group"] = split_group
    grouped["split_group_source"] = split_group_source
    return grouped


def distribution_score(frame: pd.DataFrame, candidate_idx: np.ndarray, target_fraction: float) -> float:
    """Distribution score."""
    candidate = frame.iloc[candidate_idx]
    score = abs((len(candidate) / max(1, len(frame))) - target_fraction)
    for column in ["label", "sample_tier"]:
        overall = frame[column].value_counts(normalize=True)
        selected = candidate[column].value_counts(normalize=True)
        keys = overall.index.union(selected.index)
        score += float((overall.reindex(keys, fill_value=0) - selected.reindex(keys, fill_value=0)).abs().sum())
    return score


def best_group_split(
    frame: pd.DataFrame,
    test_fraction: float,
    seed: int,
    attempts: int,
    require_two_classes_when_available: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Best group split."""
    if len(frame) < 2:
        raise ValueError("Cannot split a transfer frame with fewer than two rows")
    groups = frame["split_group"].to_numpy()
    best_train_idx: np.ndarray | None = None
    best_test_idx: np.ndarray | None = None
    best_score = float("inf")
    strict_possible = frame["label"].nunique() > 1 and require_two_classes_when_available
    fallback: tuple[np.ndarray, np.ndarray, float] | None = None
    for attempt in range(max(1, attempts)):
        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=test_fraction,
            random_state=(seed * 1009) + attempt,
        )
        train_idx, test_idx = next(splitter.split(frame.index.to_numpy(), groups=groups))
        score = distribution_score(frame, test_idx, test_fraction)
        if fallback is None or score < fallback[2]:
            fallback = (train_idx, test_idx, score)
        if strict_possible:
            if frame.iloc[train_idx]["label"].nunique() < 2 or frame.iloc[test_idx]["label"].nunique() < 2:
                continue
        if score < best_score:
            best_score = score
            best_train_idx = train_idx
            best_test_idx = test_idx
    if best_train_idx is None or best_test_idx is None:
        if fallback is None:
            raise RuntimeError("GroupShuffleSplit failed to produce a candidate split")
        best_train_idx, best_test_idx, _ = fallback
    return best_train_idx, best_test_idx


def build_transfer_split(bundle: GraphBundle, transfer_id: str, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build transfer split."""
    config = bundle.config
    pair_config = config["transfer_pairs"][transfer_id]
    primary = build_primary_task_frame(bundle)
    source_regions = set(pair_config["source_regions"])
    target_regions = set(pair_config["target_regions"])
    source = primary[primary["jurisdiction"].isin(source_regions)].copy().reset_index(drop=True)
    target = primary[primary["jurisdiction"].isin(target_regions)].copy().reset_index(drop=True)
    if source["label"].nunique() < 2:
        raise ValueError(f"{transfer_id} source does not contain both binary classes")
    if target.empty:
        raise ValueError(f"{transfer_id} target has no primary-label rows")

    split_config = config["splits"]["transfer"]
    group_columns = list(split_config["group_columns"])
    attempts = int(split_config.get("balance_attempts", 30))
    source = add_transfer_groups(source, group_columns)
    target = add_transfer_groups(target, group_columns)
    source_train_idx, source_val_idx = best_group_split(
        source,
        test_fraction=float(split_config["source_val_fraction"]),
        seed=seed,
        attempts=attempts,
        require_two_classes_when_available=True,
    )
    target_adapt_idx, target_test_idx = best_group_split(
        target,
        test_fraction=float(split_config["target_test_fraction"]),
        seed=seed + 211,
        attempts=attempts,
        require_two_classes_when_available=True,
    )

    source_split = source.copy()
    source_split["split"] = "unassigned"
    source_split.loc[source_train_idx, "split"] = "source_train"
    source_split.loc[source_val_idx, "split"] = "source_val"
    source_split["domain_role"] = "source"

    target_split = target.copy()
    target_split["split"] = "unassigned"
    target_split.loc[target_adapt_idx, "split"] = "target_adapt_unlabeled"
    target_split.loc[target_test_idx, "split"] = "target_test"
    target_split["domain_role"] = "target"

    split_frame = pd.concat([source_split, target_split], ignore_index=True)
    split_frame["transfer_id"] = transfer_id
    split_frame["seed"] = seed
    split_frame["target_labels_used_for_training"] = False
    split_frame["primary_metric_configured"] = pair_config["primary_metric"]

    audit = make_transfer_split_audit(split_frame, transfer_id, seed)
    return split_frame, audit


def make_transfer_split_audit(split_frame: pd.DataFrame, transfer_id: str, seed: int) -> pd.DataFrame:
    """Construct transfer split audit."""
    rows: list[dict[str, Any]] = []
    for split_name, part in split_frame.groupby("split", dropna=False):
        rows.append(
            {
                "transfer_id": transfer_id,
                "seed": seed,
                "split": split_name,
                "domain_role": ",".join(sorted(part["domain_role"].dropna().unique())),
                "num_rows": int(len(part)),
                "num_groups": int(part["split_group"].nunique()),
                "licensed_baseline": int((part["sample_tier"] == "licensed_baseline").sum()),
                "illegal_single": int((part["sample_tier"] == "illegal_confirmed_official_single").sum()),
                "illegal_cross_verified": int(
                    (part["sample_tier"] == "illegal_confirmed_official_cross_verified").sum()
                ),
                "label_0": int((part["label"] == 0).sum()),
                "label_1": int((part["label"] == 1).sum()),
            }
        )
    for role in ["source", "target"]:
        role_frame = split_frame[split_frame["domain_role"] == role]
        leaking = (
            role_frame.groupby("split_group")["split"]
            .nunique()
            .reset_index(name="num_splits")
            .query("num_splits > 1")
        )
        rows.append(
            {
                "transfer_id": transfer_id,
                "seed": seed,
                "split": f"__{role}_group_overlap_audit__",
                "domain_role": role,
                "num_rows": int(len(leaking)),
                "num_groups": int(len(leaking)),
                "licensed_baseline": 0,
                "illegal_single": 0,
                "illegal_cross_verified": 0,
                "label_0": 0,
                "label_1": 0,
            }
        )
    return pd.DataFrame(rows)


def build_label_vector(bundle: GraphBundle, split_frame: pd.DataFrame, source_only: bool = False) -> torch.Tensor:
    """Build label vector."""
    labels = torch.full((len(bundle.website_frame),), -1, dtype=torch.long)
    if source_only:
        split_frame = split_frame[split_frame["domain_role"] == "source"]
    for _, row in split_frame.iterrows():
        labels[int(row["graph_node_index"])] = int(row["label"])
    return labels


def index_tensor(split_frame: pd.DataFrame, split_name: str, device: torch.device) -> torch.Tensor:
    """Index tensor."""
    return torch.tensor(
        split_frame.loc[split_frame["split"] == split_name, "graph_node_index"].to_numpy(dtype=np.int64),
        dtype=torch.long,
        device=device,
    )


def build_metapath_adjacency(bundle: GraphBundle, graph_data: Any, device: torch.device) -> dict[str, torch.Tensor]:
    """Build metapath adjacency."""
    heco_config = bundle.config["heco"]
    artifacts = build_metapath_artifacts(
        data=graph_data.cpu(),
        metapath_relations=dict(heco_config["metapaths"]),
        top_k=int(heco_config["top_k_positive"]),
        min_metapath_support=int(heco_config["min_metapath_support"]),
    )
    return {name: adjacency.to(device) for name, adjacency in artifacts.adjacency.items()}


def load_encoder_state(bundle: GraphBundle, seed: int) -> tuple[dict[str, torch.Tensor], str]:
    """Load encoder state."""
    for template in bundle.config["heco"]["encoder_checkpoint_templates"]:
        checkpoint_path = resolve_workspace_path(bundle.config, template.format(seed=seed))
        if checkpoint_path.exists():
            encoder = make_heco_model(bundle)
            state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            encoder.load_state_dict(state_dict)
            return copy.deepcopy(encoder.state_dict()), str(checkpoint_path)
    raise FileNotFoundError(f"No HeCo tau0.7 encoder checkpoint found for seed {seed}")


def make_finetune_model(
    bundle: GraphBundle,
    encoder_state: dict[str, torch.Tensor],
    device: torch.device,
) -> HeCoFineTuneClassifier:
    """Construct finetune model."""
    encoder = make_heco_model(bundle).to(device)
    encoder.load_state_dict(copy.deepcopy(encoder_state))
    return HeCoFineTuneClassifier(
        heco_encoder=encoder,
        head_type=str(bundle.config["finetune"]["head_type"]),
        hidden_dim=int(bundle.config["heco"]["hidden_dim"]),
        dropout=float(bundle.config["finetune"]["dropout"]),
    ).to(device)


def make_dann_model(
    bundle: GraphBundle,
    encoder_state: dict[str, torch.Tensor],
    device: torch.device,
) -> HeCoDANNClassifier:
    """Construct DANN model."""
    encoder = make_heco_model(bundle).to(device)
    encoder.load_state_dict(copy.deepcopy(encoder_state))
    return HeCoDANNClassifier(
        heco_encoder=encoder,
        hidden_dim=int(bundle.config["heco"]["hidden_dim"]),
        dropout=float(bundle.config["finetune"]["dropout"]),
        head_type=str(bundle.config["finetune"]["head_type"]),
    ).to(device)


def finite_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Finite AUC."""
    if np.unique(y_true).size < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def evaluate_binary(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> dict[str, float]:
    """Evaluate binary."""
    y_true = y_true.astype(int)
    y_score = y_score.astype(float)
    y_pred = (y_score >= threshold).astype(int)
    unique = np.unique(y_true)
    roc_auc = finite_auc(y_true, y_score)
    pr_auc = float("nan") if unique.size < 2 else float(average_precision_score(y_true, y_score))
    if unique.size < 2:
        balanced_accuracy = float(np.mean(y_pred == y_true)) if y_true.size else float("nan")
    else:
        recall_0 = np.mean(y_pred[y_true == 0] == 0) if np.any(y_true == 0) else np.nan
        recall_1 = np.mean(y_pred[y_true == 1] == 1) if np.any(y_true == 1) else np.nan
        balanced_accuracy = float(np.nanmean([recall_0, recall_1]))
    licensed_consistency = float(np.mean(y_pred[y_true == 0] == 0)) if np.any(y_true == 0) else float("nan")
    illegal_consistency = float(np.mean(y_pred[y_true == 1] == 1)) if np.any(y_true == 1) else float("nan")
    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "balanced_accuracy": balanced_accuracy,
        "macro_f1": float(f1_score(y_true, y_pred, labels=[0, 1], average="macro", zero_division=0)),
        "licensed_consistency": licensed_consistency,
        "illegal_consistency": illegal_consistency,
        "mean_pred_illegal": float(np.mean(y_score)) if y_score.size else float("nan"),
    }


def train_heco_transfer_model(
    bundle: GraphBundle,
    graph_data: Any,
    split_frame: pd.DataFrame,
    encoder_state: dict[str, torch.Tensor],
    seed: int,
    method: str,
    smoke_test: bool,
    purpose: str = "classifier",
    max_epochs_override: int | None = None,
    min_epochs_override: int | None = None,
    patience_override: int | None = None,
    force_full_epochs: bool = False,
    select_last_state: bool = False,
) -> tuple[dict[str, Any], pd.DataFrame, np.ndarray]:
    """Train HeCo transfer model."""
    set_random_seed(seed)
    device = resolve_device(bundle.config)
    data = copy.deepcopy(graph_data).to(device)
    metapath_adjacency = build_metapath_adjacency(bundle, graph_data, device)
    labels = build_label_vector(bundle, split_frame).to(device)
    source_train_idx = index_tensor(split_frame, "source_train", device)
    source_val_idx = index_tensor(split_frame, "source_val", device)
    target_adapt_idx = index_tensor(split_frame, "target_adapt_unlabeled", device)
    target_test_idx = index_tensor(split_frame, "target_test", device)

    if method == "dann":
        model = make_dann_model(bundle, encoder_state, device)
        optimizer = torch.optim.Adam(
            [
                {"params": model.encoder_parameters(), "lr": float(bundle.config["finetune"]["encoder_lr"])},
                {"params": model.head_parameters(), "lr": float(bundle.config["finetune"]["classifier_lr"])},
                {"params": model.domain_parameters(), "lr": float(bundle.config["finetune"]["classifier_lr"])},
            ],
            weight_decay=float(bundle.config["finetune"]["weight_decay"]),
        )
    elif method == "source_only":
        model = make_finetune_model(bundle, encoder_state, device)
        optimizer = torch.optim.Adam(
            [
                {"params": model.encoder_parameters(), "lr": float(bundle.config["finetune"]["encoder_lr"])},
                {"params": model.head_parameters(), "lr": float(bundle.config["finetune"]["classifier_lr"])},
            ],
            weight_decay=float(bundle.config["finetune"]["weight_decay"]),
        )
    else:
        raise ValueError(f"Unsupported train method core: {method}")

    max_epochs_config = (
        bundle.config["dann"].get("max_epochs", bundle.config["finetune"]["max_epochs"])
        if method == "dann"
        else bundle.config["finetune"]["max_epochs"]
    )
    min_epochs_config = (
        bundle.config["dann"].get(
            "min_epochs_before_early_stop",
            bundle.config["finetune"]["min_epochs_before_early_stop"],
        )
        if method == "dann"
        else bundle.config["finetune"]["min_epochs_before_early_stop"]
    )
    if max_epochs_override is not None:
        max_epochs_config = max_epochs_override
    if min_epochs_override is not None:
        min_epochs_config = min_epochs_override
    max_epochs = 5 if smoke_test else int(max_epochs_config)
    patience_config = bundle.config["finetune"]["patience"] if patience_override is None else patience_override
    patience = max_epochs if smoke_test else int(patience_config)
    min_epochs = 0 if smoke_test else int(min_epochs_config)
    dann_warmup_epochs = int(bundle.config["dann"]["lambda_warmup_epochs"]) if method == "dann" else 0
    best_state_min_epoch = 1
    if method == "dann" and bool(bundle.config["dann"].get("mask_warmup_best_state", True)):
        best_state_min_epoch = max(1, dann_warmup_epochs)
    best_state: dict[str, Any] | None = None
    last_state: dict[str, Any] | None = None
    best_score = float("-inf")
    best_epoch = 0
    last_epoch = 0
    history_rows: list[dict[str, Any]] = []
    started = time.time()

    for epoch in range(1, max_epochs + 1):
        model.train()
        optimizer.zero_grad()
        lambda_value = 0.0
        domain_loss = torch.tensor(0.0, device=device)
        domain_acc = float("nan")
        if method == "dann":
            lambda_value = lambda_schedule(
                epoch=epoch,
                warmup_epochs=dann_warmup_epochs,
                lambda_max=float(bundle.config["dann"]["lambda_max"]),
                schedule=str(bundle.config["dann"]["lambda_schedule"]),
            )
            logits, domain_logits_all, _ = model(data, metapath_adjacency, reverse_scale=lambda_value)
            domain_indices = torch.cat([source_train_idx, target_adapt_idx], dim=0)
            domain_labels = torch.cat(
                [
                    torch.zeros(source_train_idx.shape[0], dtype=torch.long, device=device),
                    torch.ones(target_adapt_idx.shape[0], dtype=torch.long, device=device),
                ],
                dim=0,
            )
            domain_loss = F.cross_entropy(domain_logits_all[domain_indices], domain_labels)
            domain_pred = domain_logits_all[domain_indices].argmax(dim=1)
            domain_acc = float((domain_pred == domain_labels).float().mean().detach().cpu().item())
        else:
            logits, _, _ = model(data, metapath_adjacency)
        classification_loss = F.cross_entropy(logits[source_train_idx], labels[source_train_idx])
        total_loss = classification_loss + (
            float(bundle.config["dann"]["domain_loss_weight"]) * domain_loss if method == "dann" else 0.0
        )
        total_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            if method == "dann":
                eval_logits, _, _ = model(data, metapath_adjacency, reverse_scale=0.0)
            else:
                eval_logits, _, _ = model(data, metapath_adjacency)
            probabilities = torch.softmax(eval_logits, dim=1)[:, 1].detach().cpu().numpy()
            source_val_scores = probabilities[source_val_idx.detach().cpu().numpy()]
            target_test_scores = probabilities[target_test_idx.detach().cpu().numpy()]

        source_val_labels = split_frame.loc[split_frame["split"] == "source_val", "label"].to_numpy(dtype=int)
        target_test_labels = split_frame.loc[split_frame["split"] == "target_test", "label"].to_numpy(dtype=int)
        source_val_auc = finite_auc(source_val_labels, source_val_scores)
        target_test_auc = finite_auc(target_test_labels, target_test_scores)
        if math.isfinite(source_val_auc):
            early_stop_score = source_val_auc
        else:
            early_stop_score = -float(F.cross_entropy(eval_logits[source_val_idx], labels[source_val_idx]).item())
        history_rows.append(
            {
                "epoch": epoch,
                "phase": method,
                "purpose": purpose,
                "loss": float(total_loss.item()),
                "classification_loss": float(classification_loss.item()),
                "domain_loss": float(domain_loss.item()),
                "lambda": float(lambda_value),
                "domain_acc": domain_acc,
                "source_val_roc_auc": float(source_val_auc),
                "target_test_roc_auc": float(target_test_auc),
                "best_state_eligible": bool(epoch >= best_state_min_epoch),
                "runtime_s": float(time.time() - started),
            }
        )
        last_state = copy.deepcopy(model.state_dict())
        last_epoch = epoch
        eligible_for_best = epoch >= best_state_min_epoch
        if eligible_for_best and math.isfinite(early_stop_score) and early_stop_score >= best_score:
            best_score = float(early_stop_score)
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
        elif (
            not force_full_epochs
            and epoch >= min_epochs
            and best_state is not None
            and epoch - best_epoch >= patience
        ):
            break

    selection_policy = "best_source_val"
    if method == "dann" and best_state_min_epoch > 1:
        selection_policy = "best_source_val_after_dann_warmup"
    if select_last_state:
        if last_state is None:
            raise RuntimeError(f"{method} did not produce a last model state")
        best_state = last_state
        best_epoch = last_epoch
        selection_policy = "last_epoch"
    if best_state is None:
        if last_state is None:
            raise RuntimeError(f"{method} did not produce a best model state")
        best_state = last_state
        best_epoch = last_epoch
        selection_policy = "fallback_last_epoch"
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        if method == "dann":
            logits, _, _ = model(data, metapath_adjacency, reverse_scale=0.0)
        else:
            logits, _, _ = model(data, metapath_adjacency)
        probabilities_all = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
    result = {
        "best_epoch": int(best_epoch),
        "best_source_val_score": float(best_score),
        "selection_policy": selection_policy,
        "best_state_min_epoch": int(best_state_min_epoch),
        "trained_epochs": int(last_epoch),
        "force_full_epochs": bool(force_full_epochs),
        "purpose": purpose,
        "runtime_s": float(time.time() - started),
    }
    return result, pd.DataFrame(history_rows), probabilities_all


def make_probability_frame(split_frame: pd.DataFrame, probabilities_all: np.ndarray, threshold: float, method: str) -> pd.DataFrame:
    """Construct probability frame."""
    frame = split_frame.copy()
    frame["pred_prob_illegal"] = frame["graph_node_index"].map(lambda idx: float(probabilities_all[int(idx)]))
    frame["pred_label"] = (frame["pred_prob_illegal"] >= threshold).astype(int)
    frame["method"] = method
    return frame


def summarize_run_metrics(
    bundle: GraphBundle,
    split_frame: pd.DataFrame,
    probabilities_all: np.ndarray,
    method: str,
    seed: int,
    transfer_id: str,
    train_info: dict[str, Any],
    domain_acc_final: float,
    pseudo_label_quality: float,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Summarise run metrics."""
    source_val = split_frame[split_frame["split"] == "source_val"].copy()
    source_val_scores = probabilities_all[source_val["graph_node_index"].to_numpy(dtype=int)]
    threshold = choose_threshold_by_youden(source_val["label"].to_numpy(dtype=int), source_val_scores)
    prediction_frame = make_probability_frame(split_frame, probabilities_all, threshold, method)
    source_train = prediction_frame[prediction_frame["split"] == "source_train"]
    source_val = prediction_frame[prediction_frame["split"] == "source_val"]
    target_test = prediction_frame[prediction_frame["split"] == "target_test"]

    source_train_metrics = evaluate_binary(
        source_train["label"].to_numpy(dtype=int),
        source_train["pred_prob_illegal"].to_numpy(dtype=float),
        threshold,
    )
    source_val_metrics = evaluate_binary(
        source_val["label"].to_numpy(dtype=int),
        source_val["pred_prob_illegal"].to_numpy(dtype=float),
        threshold,
    )
    target_metrics = evaluate_binary(
        target_test["label"].to_numpy(dtype=int),
        target_test["pred_prob_illegal"].to_numpy(dtype=float),
        threshold,
    )
    pair_config = bundle.config["transfer_pairs"][transfer_id]
    configured_primary = str(pair_config["primary_metric"])
    primary_metric_direction = str(pair_config.get("primary_metric_direction", "higher_is_better"))
    if configured_primary == "illegal_recall_at_youden":
        primary_value = target_metrics["illegal_consistency"]
        effective_primary = "illegal_recall_at_youden"
    elif configured_primary == "mean_pred_illegal":
        primary_value = target_metrics["mean_pred_illegal"]
        effective_primary = "mean_pred_illegal_lower_is_better"
    elif configured_primary == "balanced_accuracy":
        primary_value = target_metrics["balanced_accuracy"]
        effective_primary = "balanced_accuracy"
    elif math.isfinite(target_metrics["roc_auc"]):
        primary_value = target_metrics["roc_auc"]
        effective_primary = "roc_auc"
    else:
        primary_value = target_metrics["balanced_accuracy"]
        effective_primary = "balanced_accuracy_fallback_due_to_one_class_target"

    target_label_count = int(target_test["label"].nunique())
    row = {
        "created_at_utc": utc_now_iso(),
        "transfer_id": transfer_id,
        "method": method,
        "seed": seed,
        "source_size": int((split_frame["domain_role"] == "source").sum()),
        "target_size": int((split_frame["domain_role"] == "target").sum()),
        "source_train_size": int(len(source_train)),
        "source_val_size": int(len(source_val)),
        "target_test_size": int(len(target_test)),
        "source_train_auc": source_train_metrics["roc_auc"],
        "source_val_auc": source_val_metrics["roc_auc"],
        "target_test_auc": target_metrics["roc_auc"],
        "target_pr_auc": target_metrics["pr_auc"],
        "target_balanced_acc": target_metrics["balanced_accuracy"],
        "target_macro_f1": target_metrics["macro_f1"],
        "target_licensed_consistency": target_metrics["licensed_consistency"],
        "target_illegal_consistency": target_metrics["illegal_consistency"],
        "target_illegal_recall_at_youden": target_metrics["illegal_consistency"],
        "target_mean_pred_illegal": target_metrics["mean_pred_illegal"],
        "target_label_class_count": target_label_count,
        "target_has_complete_binary_labels": bool(target_label_count == 2),
        "configured_primary_metric": configured_primary,
        "effective_primary_metric": effective_primary,
        "primary_metric_direction": primary_metric_direction,
        "primary_metric_value": primary_value,
        "threshold": threshold,
        "domain_acc_final": domain_acc_final,
        "pseudo_label_quality": pseudo_label_quality,
        **train_info,
    }
    return row, prediction_frame


def build_pseudo_labels(
    bundle: GraphBundle,
    split_frame: pd.DataFrame,
    probabilities_all: np.ndarray,
    transfer_id: str,
    method: str,
    seed: int,
    iteration: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Build pseudo labels."""
    threshold = float(bundle.config["strurw"]["pseudo_confidence_threshold"])
    strategy = str(bundle.config["strurw"].get("pseudo_label_strategy", "confidence_threshold"))
    entropy_top_fraction = float(bundle.config["strurw"].get("entropy_top_fraction", 0.30))
    labels = torch.full((len(bundle.website_frame),), -1, dtype=torch.long)
    confident_mask = torch.zeros((len(bundle.website_frame),), dtype=torch.bool)
    target_adapt = split_frame[split_frame["split"] == "target_adapt_unlabeled"].copy()
    node_indices = target_adapt["graph_node_index"].to_numpy(dtype=int)
    scores = probabilities_all[node_indices].astype(np.float64)
    confidence = np.maximum(scores, 1.0 - scores)
    clipped_scores = np.clip(scores, 1e-7, 1.0 - 1e-7)
    entropy = -(
        clipped_scores * np.log(clipped_scores)
        + (1.0 - clipped_scores) * np.log(1.0 - clipped_scores)
    ) / np.log(2.0)
    pseudo = (scores >= 0.5).astype(int)
    if strategy == "entropy_top_fraction":
        confident = np.zeros_like(scores, dtype=bool)
        if scores.size:
            k = max(1, int(math.ceil(scores.size * entropy_top_fraction)))
            selected = np.argsort(entropy)[:k]
            confident[selected] = True
    elif strategy == "confidence_threshold":
        confident = confidence >= threshold
    else:
        raise ValueError(f"Unsupported StruRW pseudo-label strategy: {strategy}")
    for node_idx, pseudo_label, is_confident in zip(node_indices, pseudo, confident):
        if bool(is_confident):
            labels[int(node_idx)] = int(pseudo_label)
            confident_mask[int(node_idx)] = True
    if confident.any():
        true_labels = target_adapt["label"].to_numpy(dtype=int)[confident]
        pseudo_quality = float(np.mean(pseudo[confident] == true_labels))
        pseudo_positive_rate = float(np.mean(pseudo[confident]))
        mean_confidence = float(np.mean(confidence[confident]))
    else:
        pseudo_quality = float("nan")
        pseudo_positive_rate = float("nan")
        mean_confidence = float("nan")
    audit = {
        "transfer_id": transfer_id,
        "method": method,
        "seed": seed,
        "iteration": iteration,
        "target_adapt_size": int(len(target_adapt)),
        "num_confident_pseudo_labels": int(confident.sum()),
        "pseudo_label_coverage": float(confident.mean()) if len(confident) else float("nan"),
        "pseudo_label_quality": pseudo_quality,
        "pseudo_positive_rate": pseudo_positive_rate,
        "mean_confidence": mean_confidence,
        "mean_entropy": float(np.mean(entropy[confident])) if confident.any() else float("nan"),
        "confidence_threshold": threshold,
        "selection_strategy": strategy,
        "entropy_top_fraction": entropy_top_fraction if strategy == "entropy_top_fraction" else float("nan"),
    }
    return labels, confident_mask, audit


def build_source_label_inputs(bundle: GraphBundle, split_frame: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor]:
    """Build source label inputs."""
    labels = torch.full((len(bundle.website_frame),), -1, dtype=torch.long)
    source_mask = torch.zeros((len(bundle.website_frame),), dtype=torch.bool)
    source_frame = split_frame[split_frame["domain_role"] == "source"]
    for _, row in source_frame.iterrows():
        node_idx = int(row["graph_node_index"])
        labels[node_idx] = int(row["label"])
        source_mask[node_idx] = True
    return labels, source_mask


def run_strurw_family(
    bundle: GraphBundle,
    split_frame: pd.DataFrame,
    encoder_state: dict[str, torch.Tensor],
    seed: int,
    transfer_id: str,
    method: str,
    smoke_test: bool,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run StruRW family."""
    current_graph = copy.deepcopy(bundle.graph_data.cpu())
    source_labels, source_mask = build_source_label_inputs(bundle, split_frame)
    reweighter = StruRWReweighter(
        relations=list(bundle.config["strurw"]["metapath_relations"]),
        edge_weight_clip=tuple(bundle.config["strurw"]["edge_weight_clip"]),
        laplace_alpha=float(bundle.config["strurw"]["laplace_alpha"]),
    )
    all_history: list[pd.DataFrame] = []
    pseudo_rows: list[dict[str, Any]] = []
    edge_audits: list[pd.DataFrame] = []
    csbm_audits: list[pd.DataFrame] = []
    final_probabilities: np.ndarray | None = None
    final_train_info: dict[str, Any] = {}

    iterations = 1 if smoke_test else int(bundle.config["strurw"]["pseudo_label_iterations"])
    for iteration in range(1, iterations + 1):
        pseudo_config = bundle.config["strurw"]
        train_info, history, probabilities = train_heco_transfer_model(
            bundle=bundle,
            graph_data=current_graph,
            split_frame=split_frame,
            encoder_state=encoder_state,
            seed=seed + iteration - 1,
            method="source_only",
            smoke_test=smoke_test,
            purpose="pseudo_labeler",
            max_epochs_override=int(pseudo_config.get("pseudo_labeler_max_epochs", bundle.config["finetune"]["max_epochs"])),
            min_epochs_override=int(
                pseudo_config.get(
                    "pseudo_labeler_min_epochs_before_early_stop",
                    bundle.config["finetune"]["min_epochs_before_early_stop"],
                )
            ),
            patience_override=int(pseudo_config.get("pseudo_labeler_patience", bundle.config["finetune"]["patience"])),
            select_last_state=bool(pseudo_config.get("pseudo_labeler_select_last_state", True)),
        )
        history["strurw_iteration"] = iteration
        history["phase"] = f"strurw_prefit_{iteration}"
        history["purpose"] = "pseudo_labeler"
        all_history.append(history)
        pseudo_labels, confident_mask, pseudo_audit = build_pseudo_labels(
            bundle=bundle,
            split_frame=split_frame,
            probabilities_all=probabilities,
            transfer_id=transfer_id,
            method=method,
            seed=seed,
            iteration=iteration,
        )
        pseudo_rows.append(pseudo_audit)
        result = reweighter.reweight(
            data=current_graph,
            source_labels=source_labels,
            source_mask=source_mask,
            target_pseudo_labels=pseudo_labels,
            target_confident_mask=confident_mask,
            iteration=iteration,
        )
        edge_audit = result.edge_audit.copy()
        edge_audit["transfer_id"] = transfer_id
        edge_audit["method"] = method
        edge_audit["seed"] = seed
        csbm_audit = result.csbm_audit.copy()
        csbm_audit["transfer_id"] = transfer_id
        csbm_audit["method"] = method
        csbm_audit["seed"] = seed
        edge_audits.append(edge_audit)
        csbm_audits.append(csbm_audit)
        current_graph = result.graph_data

    final_core = "dann" if method == "dann_strurw" else "source_only"
    final_config = bundle.config["strurw"]
    final_train_info, final_history, final_probabilities = train_heco_transfer_model(
        bundle=bundle,
        graph_data=current_graph,
        split_frame=split_frame,
        encoder_state=encoder_state,
        seed=seed,
        method=final_core,
        smoke_test=smoke_test,
        purpose="final_classifier",
        max_epochs_override=int(final_config.get("final_max_epochs", bundle.config["finetune"]["max_epochs"])),
        min_epochs_override=int(
            final_config.get(
                "final_min_epochs_before_early_stop",
                bundle.config["finetune"]["min_epochs_before_early_stop"],
            )
        ),
        patience_override=int(final_config.get("final_patience", bundle.config["finetune"]["patience"])),
        force_full_epochs=bool(final_config.get("final_force_full_epochs", True)),
        select_last_state=bool(final_config.get("final_select_last_state_for_source_only", True))
        if final_core == "source_only"
        else False,
    )
    final_history["strurw_iteration"] = iterations
    final_history["phase"] = method
    final_history["purpose"] = "final_classifier"
    all_history.append(final_history)
    if final_probabilities is None:
        raise RuntimeError("StruRW did not produce final probabilities")
    domain_acc_final = (
        float(final_history["domain_acc"].dropna().iloc[-1])
        if "domain_acc" in final_history and final_history["domain_acc"].dropna().shape[0]
        else float("nan")
    )
    pseudo_quality = pseudo_rows[-1]["pseudo_label_quality"] if pseudo_rows else float("nan")
    metrics, prediction_frame = summarize_run_metrics(
        bundle=bundle,
        split_frame=split_frame,
        probabilities_all=final_probabilities,
        method=method,
        seed=seed,
        transfer_id=transfer_id,
        train_info=final_train_info,
        domain_acc_final=domain_acc_final,
        pseudo_label_quality=float(pseudo_quality),
    )
    history_frame = pd.concat(all_history, ignore_index=True)
    pseudo_frame = pd.DataFrame(pseudo_rows)
    edge_frame = pd.concat(edge_audits, ignore_index=True) if edge_audits else pd.DataFrame()
    return metrics, prediction_frame, history_frame, pseudo_frame, edge_frame


def run_single_experiment(
    bundle: GraphBundle,
    split_frame: pd.DataFrame,
    encoder_state: dict[str, torch.Tensor],
    seed: int,
    transfer_id: str,
    method: str,
    smoke_test: bool,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run single experiment."""
    if method in {"strurw", "dann_strurw"}:
        return run_strurw_family(bundle, split_frame, encoder_state, seed, transfer_id, method, smoke_test)
    core_method = "dann" if method == "dann" else "source_only"
    train_info, history, probabilities = train_heco_transfer_model(
        bundle=bundle,
        graph_data=bundle.graph_data.cpu(),
        split_frame=split_frame,
        encoder_state=encoder_state,
        seed=seed,
        method=core_method,
        smoke_test=smoke_test,
    )
    domain_acc_final = (
        float(history["domain_acc"].dropna().iloc[-1])
        if "domain_acc" in history and history["domain_acc"].dropna().shape[0]
        else float("nan")
    )
    metrics, prediction_frame = summarize_run_metrics(
        bundle=bundle,
        split_frame=split_frame,
        probabilities_all=probabilities,
        method=method,
        seed=seed,
        transfer_id=transfer_id,
        train_info=train_info,
        domain_acc_final=domain_acc_final,
        pseudo_label_quality=float("nan"),
    )
    return metrics, prediction_frame, history, pd.DataFrame(), pd.DataFrame()


def save_run_artifacts(
    bundle: GraphBundle,
    transfer_id: str,
    method: str,
    seed: int,
    split_frame: pd.DataFrame,
    split_path: Path,
    metrics: dict[str, Any],
    history: pd.DataFrame,
    encoder_checkpoint: str,
) -> None:
    """Save run artifacts."""
    suffix = f"{transfer_id}__{method}__seed{seed}"
    save_dataframe(history, bundle.output_paths["logs"] / f"{suffix}_history.csv")
    manifest = make_common_manifest(
        bundle=bundle,
        task=transfer_id,
        model=method,
        split_name=split_path.name,
        sample_counts=compute_sample_counts(split_frame),
        seed=seed,
    )
    manifest["encoder_checkpoint"] = encoder_checkpoint
    manifest["target_label_boundary"] = {
        "target_has_complete_binary_labels": bool(metrics["target_has_complete_binary_labels"]),
        "effective_primary_metric": metrics["effective_primary_metric"],
    }
    manifest["metrics"] = metrics
    record_run_manifest(bundle.output_paths["runs"] / f"{suffix}.json", manifest)


def plot_transfer_boxplot(summary: pd.DataFrame, output_path: Path) -> None:
    """Plot transfer boxplot."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    transfer_ids = list(dict.fromkeys(summary["transfer_id"].tolist()))
    method_order = ["source_only", "dann", "strurw", "dann_strurw"]
    methods = [method for method in method_order if method in set(summary["method"])]
    if not methods:
        return
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
    axes_flat = axes.flatten()
    colors = ["#2563eb", "#dc2626", "#059669", "#7c3aed"]
    for axis, transfer_id in zip(axes_flat, transfer_ids):
        transfer_frame = summary[summary["transfer_id"] == transfer_id]
        values = [
            transfer_frame.loc[transfer_frame["method"] == method, "primary_metric_value"].dropna().to_numpy()
            for method in methods
        ]
        axis.boxplot(values, tick_labels=methods, patch_artist=True)
        for patch, color in zip(axis.patches, colors[: len(methods)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.45)
        axis.set_title(transfer_id)
        axis.set_ylim(0.0, 1.02)
        axis.tick_params(axis="x", rotation=20)
        axis.grid(axis="y", alpha=0.2)
        metric_names = sorted(transfer_frame["effective_primary_metric"].dropna().unique())
        axis.text(0.02, 0.04, " / ".join(metric_names), transform=axis.transAxes, fontsize=8)
    for axis in axes_flat[len(transfer_ids) :]:
        axis.axis("off")
    fig.suptitle("Week 7 Transfer Primary Metric by Method")
    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def plot_t3_dann_diagnostics(logs_dir: Path, output_path: Path) -> None:
    """Plot t3 DANN diagnostics."""
    path = logs_dir / "T3_DiagnoseFrance__dann__seed42_history.csv"
    if not path.exists():
        return
    history = pd.read_csv(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].plot(history["epoch"], history["domain_acc"], color="#dc2626")
    axes[0].axhline(0.5, color="gray", linestyle="--", linewidth=1)
    axes[0].set_title("Domain Acc")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylim(0.0, 1.05)
    axes[1].plot(history["epoch"], history["source_val_roc_auc"], label="source val", color="#2563eb")
    axes[1].plot(history["epoch"], history["target_test_roc_auc"], label="target test", color="#059669")
    axes[1].set_title("ROC-AUC Diagnostics")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].legend()
    axes[2].plot(history["epoch"], history["lambda"], color="#7c3aed")
    axes[2].set_title("DANN Lambda")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylim(0.0, 1.05)
    for axis in axes:
        axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def merge_existing_frame(path: Path, new_frame: pd.DataFrame, key_columns: list[str]) -> pd.DataFrame:
    """Merge existing frame."""
    if path.exists() and path.stat().st_size > 0:
        try:
            existing = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            existing = pd.DataFrame()
        merged = pd.concat([existing, new_frame], ignore_index=True, sort=False)
    else:
        merged = new_frame.copy()
    present_keys = [column for column in key_columns if column in merged.columns]
    if present_keys:
        merged = merged.drop_duplicates(subset=present_keys, keep="last")
    return merged.reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run Week 7 transfer experiments.")
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "week7_transfer.yaml"),
        help="Path to the Week 7 transfer config.",
    )
    parser.add_argument("--methods", nargs="+", choices=["source_only", "dann", "strurw", "dann_strurw"])
    parser.add_argument("--transfer-ids", nargs="+")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument(
        "--formal-output-for-smoke",
        action="store_true",
        help="Write smoke-test artifacts to the configured output tree instead of output/week7_smoke.",
    )
    return parser.parse_args()


def main() -> None:
    """Command-line entry point."""
    args = parse_args()
    bundle = load_graph_bundle(args.config)
    if args.smoke_test and not args.formal_output_for_smoke:
        smoke_root = resolve_workspace_path(bundle.config, "output/week7_smoke")
        remapped_paths: dict[str, Path] = {}
        for name in bundle.output_paths:
            if name == "root":
                remapped_paths[name] = smoke_root
            else:
                remapped_paths[name] = smoke_root / name
            remapped_paths[name].mkdir(parents=True, exist_ok=True)
        bundle.output_paths = remapped_paths
    transfer_ids = args.transfer_ids or list(bundle.config["transfer_pairs"].keys())
    methods = args.methods or list(bundle.config["methods"])
    seeds = args.seeds or [int(seed) for seed in bundle.config["seeds"]]
    if args.smoke_test:
        seeds = seeds[:1]
        if args.transfer_ids is None:
            transfer_ids = ["T3_DiagnoseFrance", "T2_ON"]

    raw_rows: list[dict[str, Any]] = []
    split_audits: list[pd.DataFrame] = []
    pseudo_audits: list[pd.DataFrame] = []
    strurw_audits: list[pd.DataFrame] = []
    split_cache: dict[tuple[str, int], tuple[pd.DataFrame, Path]] = {}

    for transfer_id in transfer_ids:
        if transfer_id not in bundle.config["transfer_pairs"]:
            raise ValueError(f"Unknown transfer_id: {transfer_id}")
        for seed in seeds:
            split_frame, split_audit = build_transfer_split(bundle, transfer_id, int(seed))
            split_path = bundle.output_paths["splits"] / f"{transfer_id}__seed{seed}.csv"
            save_dataframe(split_frame, split_path)
            split_cache[(transfer_id, int(seed))] = (split_frame, split_path)
            split_audits.append(split_audit)
            encoder_state, encoder_checkpoint = load_encoder_state(bundle, int(seed))
            for method in methods:
                started = time.time()
                metrics, prediction_frame, history, pseudo_frame, strurw_frame = run_single_experiment(
                    bundle=bundle,
                    split_frame=split_frame,
                    encoder_state=encoder_state,
                    seed=int(seed),
                    transfer_id=transfer_id,
                    method=method,
                    smoke_test=args.smoke_test,
                )
                metrics["wall_runtime_s"] = float(time.time() - started)
                raw_rows.append(metrics)
                save_run_artifacts(
                    bundle=bundle,
                    transfer_id=transfer_id,
                    method=method,
                    seed=int(seed),
                    split_frame=prediction_frame,
                    split_path=split_path,
                    metrics=metrics,
                    history=history,
                    encoder_checkpoint=encoder_checkpoint,
                )
                if not pseudo_frame.empty:
                    pseudo_audits.append(pseudo_frame)
                if not strurw_frame.empty:
                    strurw_audits.append(strurw_frame)

    raw_frame_new = pd.DataFrame(raw_rows).sort_values(["transfer_id", "method", "seed"]).reset_index(drop=True)
    raw_path = bundle.output_paths["metrics"] / "transfer_summary.csv"
    raw_frame = merge_existing_frame(
        raw_path,
        raw_frame_new,
        ["transfer_id", "method", "seed"],
    ).sort_values(["transfer_id", "method", "seed"]).reset_index(drop=True)
    summary_metric_columns = [
        "primary_metric_value",
        "target_test_auc",
        "target_pr_auc",
        "target_balanced_acc",
        "target_licensed_consistency",
        "target_illegal_consistency",
        "target_illegal_recall_at_youden",
        "target_mean_pred_illegal",
    ]
    method_summary = summarise_runs(raw_frame, ["transfer_id", "method"], summary_metric_columns)
    save_dataframe(raw_frame, raw_path)
    save_dataframe(method_summary, bundle.output_paths["metrics"] / "transfer_method_summary.csv")
    split_audit_path = bundle.output_paths["audits"] / "transfer_split_audit.csv"
    split_audit_frame = merge_existing_frame(
        split_audit_path,
        pd.concat(split_audits, ignore_index=True),
        ["transfer_id", "seed", "split", "domain_role"],
    )
    save_dataframe(split_audit_frame, split_audit_path)
    pseudo_path = bundle.output_paths["audits"] / "pseudo_label_quality.csv"
    pseudo_frame = pd.concat(pseudo_audits, ignore_index=True) if pseudo_audits else pd.DataFrame()
    if not pseudo_frame.empty:
        pseudo_frame = merge_existing_frame(
            pseudo_path,
            pseudo_frame,
            ["transfer_id", "method", "seed", "iteration"],
        )
    elif pseudo_path.exists() and pseudo_path.stat().st_size > 0:
        try:
            pseudo_frame = pd.read_csv(pseudo_path)
        except pd.errors.EmptyDataError:
            pseudo_frame = pd.DataFrame()
    save_dataframe(
        pseudo_frame,
        pseudo_path,
    )
    strurw_path = bundle.output_paths["audits"] / "strurw_edge_weight_audit.csv"
    strurw_frame = pd.concat(strurw_audits, ignore_index=True, sort=False) if strurw_audits else pd.DataFrame()
    if not strurw_frame.empty:
        strurw_frame = merge_existing_frame(
            strurw_path,
            strurw_frame,
            ["transfer_id", "method", "seed", "iteration", "relation"],
        )
    elif strurw_path.exists() and strurw_path.stat().st_size > 0:
        try:
            strurw_frame = pd.read_csv(strurw_path)
        except pd.errors.EmptyDataError:
            strurw_frame = pd.DataFrame()
    save_dataframe(
        strurw_frame,
        strurw_path,
    )
    plot_transfer_boxplot(raw_frame, bundle.output_paths["plots"] / "week7_transfer_boxplot_grid.png")
    plot_t3_dann_diagnostics(bundle.output_paths["logs"], bundle.output_paths["plots"] / "week7_t3_dann_diagnostics.png")

    acceptance = {
        "created_at_utc": utc_now_iso(),
        "smoke_test": bool(args.smoke_test),
        "num_runs": int(len(raw_frame)),
        "transfer_ids": transfer_ids,
        "methods": methods,
        "seeds": seeds,
        "one_class_target_runs": int((~raw_frame["target_has_complete_binary_labels"]).sum()),
        "output_metrics": str(bundle.output_paths["metrics"] / "transfer_summary.csv"),
    }
    record_run_manifest(bundle.output_paths["runs"] / "week7_transfer_run_manifest.json", acceptance)


if __name__ == "__main__":
    main()
