"""Evaluate a trained model across pooled/per-jurisdiction/cross-verified scenarios."""
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
from sklearn.manifold import TSNE
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.family_extractor import extract_denmark_families
from src.explain.structure_vs_lexical import apply_edge_mode, apply_feature_mode, make_transform_audit
from src.models.hetero_full_with_heco import HeCoFineTuneClassifier
from src.train.train_contrastive import make_heco_model
from src.train.train_transfer import (
    build_metapath_adjacency,
    evaluate_binary,
    load_encoder_state,
)
from src.train.utils import (
    GraphBundle,
    build_primary_task_frame,
    choose_threshold_by_youden,
    compute_sample_counts,
    load_graph_bundle,
    make_common_manifest,
    record_run_manifest,
    resolve_device,
    save_dataframe,
    set_random_seed,
    summarise_runs,
    utc_now_iso,
)


def redirect_output_root(bundle: GraphBundle, root_name: str) -> None:
    """Redirect output root."""
    workspace = Path(bundle.config["workspace_root"])
    for key in list(bundle.output_paths.keys()):
        subdir = key if key != "root" else ""
        new_path = workspace / "output" / root_name / subdir
        new_path.mkdir(parents=True, exist_ok=True)
        bundle.output_paths[key] = new_path
        bundle.config["output"][key] = str(Path("output") / root_name / subdir) if subdir else str(Path("output") / root_name)


def add_week8_groups(frame: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    """Add week8 groups."""
    grouped = frame.copy().reset_index(drop=True)
    split_group = pd.Series([""] * len(grouped), index=grouped.index, dtype="object")
    for column in group_columns:
        if column not in grouped.columns:
            continue
        values = grouped[column].fillna("").astype(str).str.strip()
        needs_value = split_group.eq("") & values.ne("")
        split_group.loc[needs_value] = column + "::" + values.loc[needs_value]
    if split_group.eq("").any():
        missing = grouped.loc[split_group.eq(""), "node_id"].head(10).tolist()
        raise ValueError(f"Unable to construct split groups for node_ids: {missing}")
    grouped["split_group"] = split_group
    return grouped


def _split_score(frame: pd.DataFrame, candidate_idx: np.ndarray, target_fraction: float) -> float:
    """Split score."""
    if len(frame) == 0:
        return float("inf")
    candidate = frame.iloc[candidate_idx]
    score = abs((len(candidate) / max(1, len(frame))) - target_fraction)
    if "label" in frame.columns:
        overall = frame["label"].value_counts(normalize=True)
        selected = candidate["label"].value_counts(normalize=True)
        keys = overall.index.union(selected.index)
        score += float((overall.reindex(keys, fill_value=0) - selected.reindex(keys, fill_value=0)).abs().sum())
    return score


def best_group_shuffle(
    frame: pd.DataFrame,
    test_fraction: float,
    seed: int,
    attempts: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """Best group shuffle."""
    if len(frame) < 2:
        raise ValueError("Cannot split a frame with fewer than two rows")
    groups = frame["split_group"].to_numpy()
    if len(set(groups.tolist())) < 2:
        rng = np.random.default_rng(seed)
        indices = np.arange(len(frame))
        rng.shuffle(indices)
        test_size = max(1, int(round(len(indices) * test_fraction)))
        return indices[test_size:], indices[:test_size]

    best: tuple[np.ndarray, np.ndarray, float] | None = None
    for attempt in range(max(1, attempts)):
        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=float(test_fraction),
            random_state=(seed * 1009) + attempt,
        )
        train_idx, test_idx = next(splitter.split(frame.index.to_numpy(), groups=groups))
        score = _split_score(frame, test_idx, test_fraction)
        if best is None or score < best[2]:
            best = (train_idx, test_idx, score)
    if best is None:
        raise RuntimeError("GroupShuffleSplit failed")
    return best[0], best[1]


def build_labels(bundle: GraphBundle, split_frame: pd.DataFrame) -> torch.Tensor:
    """Build labels."""
    labels = torch.full((len(bundle.website_frame),), -1, dtype=torch.long)
    for _, row in split_frame.iterrows():
        labels[int(row["graph_node_index"])] = int(row["label"])
    return labels


def make_finetune_model(bundle: GraphBundle, encoder_state: dict[str, torch.Tensor], device: torch.device) -> HeCoFineTuneClassifier:
    """Construct finetune model."""
    encoder = make_heco_model(bundle).to(device)
    encoder.load_state_dict(copy.deepcopy(encoder_state))
    return HeCoFineTuneClassifier(
        heco_encoder=encoder,
        head_type=str(bundle.config["finetune"]["head_type"]),
        hidden_dim=int(bundle.config["heco"]["hidden_dim"]),
        dropout=float(bundle.config["finetune"]["dropout"]),
    ).to(device)


def run_finetune_split(
    bundle: GraphBundle,
    graph_data: Any,
    split_frame: pd.DataFrame,
    encoder_state: dict[str, torch.Tensor],
    seed: int,
    smoke_test: bool,
    task_name: str,
    model_name: str = "heco_disc_lr_finetune",
    feature_mode: str = "full",
    edge_mode: str = "full",
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, dict[str, torch.Tensor]]:
    """Run finetune split."""
    set_random_seed(seed)
    device = resolve_device(bundle.config)
    transformed_graph = apply_feature_mode(graph_data, bundle.feature_schema, feature_mode)
    transformed_graph = apply_edge_mode(transformed_graph, edge_mode)
    data = copy.deepcopy(transformed_graph).to(device)
    metapath_adjacency = build_metapath_adjacency(bundle, transformed_graph, device)
    labels = build_labels(bundle, split_frame).to(device)
    train_idx = torch.tensor(
        split_frame.loc[split_frame["split"] == "train", "graph_node_index"].to_numpy(dtype=np.int64),
        dtype=torch.long,
        device=device,
    )
    val_idx = torch.tensor(
        split_frame.loc[split_frame["split"] == "val", "graph_node_index"].to_numpy(dtype=np.int64),
        dtype=torch.long,
        device=device,
    )
    test_idx = torch.tensor(
        split_frame.loc[split_frame["split"] == "test", "graph_node_index"].to_numpy(dtype=np.int64),
        dtype=torch.long,
        device=device,
    )
    model = make_finetune_model(bundle, encoder_state, device)
    optimizer = torch.optim.Adam(
        [
            {"params": model.encoder_parameters(), "lr": float(bundle.config["finetune"]["encoder_lr"])},
            {"params": model.head_parameters(), "lr": float(bundle.config["finetune"]["classifier_lr"])},
        ],
        weight_decay=float(bundle.config["finetune"]["weight_decay"]),
    )

    max_epochs = 5 if smoke_test else int(bundle.config["finetune"]["max_epochs"])
    patience = max_epochs if smoke_test else int(bundle.config["finetune"]["patience"])
    min_epochs = 0 if smoke_test else int(bundle.config["finetune"]["min_epochs_before_early_stop"])
    best_state: dict[str, Any] | None = None
    best_score = float("-inf")
    best_epoch = 0
    history_rows: list[dict[str, Any]] = []
    started = time.time()

    for epoch in range(1, max_epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits, _, _ = model(data, metapath_adjacency)
        loss = F.cross_entropy(logits[train_idx], labels[train_idx])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            eval_logits, _, _ = model(data, metapath_adjacency)
            probabilities = torch.softmax(eval_logits, dim=1)[:, 1].detach().cpu().numpy()
        val_scores = probabilities[val_idx.detach().cpu().numpy()]
        test_scores = probabilities[test_idx.detach().cpu().numpy()]
        val_labels = split_frame.loc[split_frame["split"] == "val", "label"].to_numpy(dtype=int)
        test_labels = split_frame.loc[split_frame["split"] == "test", "label"].to_numpy(dtype=int)
        val_auc = evaluate_binary(val_labels, val_scores, 0.5)["roc_auc"]
        test_auc = evaluate_binary(test_labels, test_scores, 0.5)["roc_auc"]
        early_stop_score = val_auc if math.isfinite(val_auc) else -float(loss.item())
        history_rows.append(
            {
                "epoch": epoch,
                "loss": float(loss.item()),
                "val_roc_auc": float(val_auc),
                "test_roc_auc": float(test_auc),
                "runtime_s": float(time.time() - started),
            }
        )
        if math.isfinite(early_stop_score) and early_stop_score >= best_score:
            best_score = float(early_stop_score)
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
        logits, embeddings, aux = model(data, metapath_adjacency)
        probabilities_all = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
        embedding_payload = {
            "fused_embedding": embeddings.detach().cpu(),
            "schema_embedding": aux["schema_embedding"].detach().cpu(),
            "metapath_embedding": aux["metapath_embedding"].detach().cpu(),
        }
        if "attention_weights" in aux:
            embedding_payload["attention_weights"] = aux["attention_weights"].detach().cpu()
        embedding_payload["probabilities_all"] = torch.tensor(probabilities_all)

    val_frame = split_frame[split_frame["split"] == "val"]
    test_frame = split_frame[split_frame["split"] == "test"]
    val_scores = probabilities_all[val_frame["graph_node_index"].to_numpy(dtype=int)]
    test_scores = probabilities_all[test_frame["graph_node_index"].to_numpy(dtype=int)]
    threshold = choose_threshold_by_youden(val_frame["label"].to_numpy(dtype=int), val_scores)
    test_metrics = evaluate_binary(test_frame["label"].to_numpy(dtype=int), test_scores, threshold)
    val_metrics = evaluate_binary(val_frame["label"].to_numpy(dtype=int), val_scores, threshold)

    predictions = split_frame.copy()
    predictions["pred_prob_illegal"] = predictions["graph_node_index"].map(
        lambda idx: float(probabilities_all[int(idx)])
    )
    predictions["pred_label"] = (predictions["pred_prob_illegal"] >= threshold).astype(int)
    predictions["model"] = model_name
    predictions["task"] = task_name

    metrics = {
        "created_at_utc": utc_now_iso(),
        "task": task_name,
        "model": model_name,
        "seed": int(seed),
        "best_epoch": int(best_epoch),
        "best_val_score": float(best_score),
        "trained_epochs": int(len(history_rows)),
        "threshold": float(threshold),
        "val_roc_auc": val_metrics["roc_auc"],
        "val_balanced_accuracy": val_metrics["balanced_accuracy"],
        "test_roc_auc": test_metrics["roc_auc"],
        "test_pr_auc": test_metrics["pr_auc"],
        "test_balanced_accuracy": test_metrics["balanced_accuracy"],
        "test_macro_f1": test_metrics["macro_f1"],
        "test_licensed_consistency": test_metrics["licensed_consistency"],
        "test_illegal_consistency": test_metrics["illegal_consistency"],
        "test_mean_pred_illegal": test_metrics["mean_pred_illegal"],
        "runtime_s": float(time.time() - started),
        "feature_mode": feature_mode,
        "edge_mode": edge_mode,
    }
    embedding_payload["transform_audit"] = make_transform_audit(bundle, f"{feature_mode}__{edge_mode}", transformed_graph)
    return metrics, predictions, pd.DataFrame(history_rows), embedding_payload


def recall_at_fpr(y_true: np.ndarray, y_score: np.ndarray, max_fpr: float = 0.10) -> float:
    """Recall at fpr."""
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    if y_true.size == 0 or not np.any(y_true == 1) or not np.any(y_true == 0):
        return float("nan")
    thresholds = np.unique(y_score)[::-1]
    best_recall = 0.0
    for threshold in thresholds:
        pred = (y_score >= threshold).astype(int)
        fp = float(np.sum((pred == 1) & (y_true == 0)))
        tn = float(np.sum((pred == 0) & (y_true == 0)))
        tp = float(np.sum((pred == 1) & (y_true == 1)))
        fn = float(np.sum((pred == 0) & (y_true == 1)))
        fpr = fp / max(fp + tn, 1.0)
        recall = tp / max(tp + fn, 1.0)
        if fpr <= max_fpr:
            best_recall = max(best_recall, recall)
    return float(best_recall)


def build_e3_lofo_split(bundle: GraphBundle, assignments: pd.DataFrame, family_id: str, seed: int) -> pd.DataFrame:
    """Build E3 leave-one-family-out split."""
    primary = build_primary_task_frame(bundle).copy()
    family_cols = ["node_id", "week8_family_id", "week8_family_source", "week8_family_size"]
    primary = primary.merge(assignments[family_cols], on="node_id", how="left")
    primary["week8_family_id"] = primary["week8_family_id"].fillna("")
    denmark = primary[primary["jurisdiction"] == "Denmark"].copy()
    heldout_pos = denmark[denmark["week8_family_id"] == family_id].copy()
    remaining_pos = denmark[denmark["week8_family_id"] != family_id].copy()
    negatives = primary[primary["label"] == 0].copy()
    if heldout_pos.empty:
        raise ValueError(f"No held-out positives for family {family_id}")
    if remaining_pos.empty or negatives.empty:
        raise ValueError("E3 requires remaining Denmark positives and licensed negatives")

    group_cols = list(bundle.config["splits"]["E3_lofo"]["group_columns"])
    remaining_pos = add_week8_groups(remaining_pos, group_cols)
    negatives = add_week8_groups(negatives, group_cols[1:])
    attempts = int(bundle.config["splits"]["pooled_primary"].get("balance_attempts", 20))
    pos_train_idx, pos_val_idx = best_group_shuffle(
        remaining_pos,
        test_fraction=float(bundle.config["splits"]["E3_lofo"]["val_fraction_within_train"]),
        seed=seed + 100,
        attempts=attempts,
    )
    neg_train_idx, neg_temp_idx = best_group_shuffle(
        negatives,
        test_fraction=(
            float(bundle.config["splits"]["pooled_primary"]["val_fraction"])
            + float(bundle.config["splits"]["pooled_primary"]["test_fraction"])
        ),
        seed=seed + 200,
        attempts=attempts,
    )
    neg_temp = negatives.iloc[neg_temp_idx].reset_index(drop=True)
    neg_val_rel, neg_test_rel = best_group_shuffle(
        neg_temp,
        test_fraction=0.5,
        seed=seed + 201,
        attempts=attempts,
    )
    neg_val_idx = neg_temp_idx[neg_val_rel]
    neg_test_idx = neg_temp_idx[neg_test_rel]

    parts = []
    for split_name, frame in [
        ("train", remaining_pos.iloc[pos_train_idx]),
        ("val", remaining_pos.iloc[pos_val_idx]),
        ("train", negatives.iloc[neg_train_idx]),
        ("val", negatives.iloc[neg_val_idx]),
        ("test", negatives.iloc[neg_test_idx]),
        ("test", heldout_pos),
    ]:
        part = frame.copy()
        part["split"] = split_name
        parts.append(part)
    split_frame = pd.concat(parts, ignore_index=True)
    split_frame["task"] = "E3_lofo"
    split_frame["seed"] = int(seed)
    split_frame["family_id"] = family_id
    split_frame["heldout_family"] = family_id
    split_frame["heldout_positive"] = split_frame["node_id"].isin(set(heldout_pos["node_id"]))
    if "split_group" not in split_frame.columns:
        split_frame = add_week8_groups(split_frame, group_cols)
    split_frame["split_group"] = split_frame["split_group"].fillna(
        split_frame["root_domain"].map(lambda value: f"root_domain::{value}")
    )
    return split_frame.reset_index(drop=True)


EUROPE_LICENSED_JURISDICTIONS = {
    "Belgium",
    "France",
    "Germany",
    "Italy",
    "Netherlands",
    "Spain",
    "Sweden",
    "United Kingdom",
}


def build_e3_w85_split(
    bundle: GraphBundle,
    assignments: pd.DataFrame,
    family_id: str,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the W8.5 corrected LOFO split plus a hard-negative eval frame."""
    primary = build_primary_task_frame(bundle).copy()
    family_cols = ["node_id", "week8_family_id", "week8_family_source", "week8_family_size"]
    primary = primary.merge(assignments[family_cols], on="node_id", how="left")
    primary["week8_family_id"] = primary["week8_family_id"].fillna("")
    denmark = primary[primary["jurisdiction"] == "Denmark"].copy()
    heldout_pos = denmark[denmark["week8_family_id"] == family_id].copy()
    other_illegal_pool = denmark[denmark["week8_family_id"] != family_id].copy()
    licensed_pool = primary[
        (primary["label"] == 0)
        & (primary["jurisdiction"].isin(EUROPE_LICENSED_JURISDICTIONS))
    ].copy()
    if heldout_pos.empty:
        raise ValueError(f"No held-out positives for family {family_id}")
    n_pos = int(len(heldout_pos))
    if len(licensed_pool) < n_pos:
        raise ValueError(f"Not enough non-Denmark European licensed controls for {family_id}")
    if len(other_illegal_pool) < n_pos:
        raise ValueError(f"Not enough hard-negative Denmark illegal controls for {family_id}")

    rng = np.random.default_rng((seed * 1009) + sum(ord(ch) for ch in family_id))
    licensed_test_ids = set(rng.choice(licensed_pool["node_id"].to_numpy(), size=n_pos, replace=False).tolist())
    hard_negative_ids = set(rng.choice(other_illegal_pool["node_id"].to_numpy(), size=n_pos, replace=False).tolist())
    test_licensed = licensed_pool[licensed_pool["node_id"].isin(licensed_test_ids)].copy()
    hard_negative = other_illegal_pool[other_illegal_pool["node_id"].isin(hard_negative_ids)].copy()
    remaining_pos = other_illegal_pool[~other_illegal_pool["node_id"].isin(hard_negative_ids)].copy()
    licensed_train_pool = licensed_pool[~licensed_pool["node_id"].isin(licensed_test_ids)].copy()
    if remaining_pos.empty or licensed_train_pool.empty:
        raise ValueError(f"W8.5 E3 split for {family_id} has empty training components")

    group_cols = list(bundle.config["splits"]["E3_lofo"]["group_columns"])
    remaining_pos = add_week8_groups(remaining_pos, group_cols)
    licensed_train_pool = add_week8_groups(licensed_train_pool, group_cols[1:])
    attempts = int(bundle.config["splits"]["pooled_primary"].get("balance_attempts", 20))
    pos_train_idx, pos_val_idx = best_group_shuffle(
        remaining_pos,
        test_fraction=float(bundle.config["splits"]["E3_lofo"]["val_fraction_within_train"]),
        seed=seed + 510,
        attempts=attempts,
    )
    neg_train_idx, neg_val_idx = best_group_shuffle(
        licensed_train_pool,
        test_fraction=float(bundle.config["splits"]["E3_lofo"]["val_fraction_within_train"]),
        seed=seed + 610,
        attempts=attempts,
    )

    parts = []
    for split_name, frame in [
        ("train", remaining_pos.iloc[pos_train_idx]),
        ("val", remaining_pos.iloc[pos_val_idx]),
        ("train", licensed_train_pool.iloc[neg_train_idx]),
        ("val", licensed_train_pool.iloc[neg_val_idx]),
        ("test", test_licensed),
        ("test", heldout_pos),
    ]:
        part = frame.copy()
        part["split"] = split_name
        parts.append(part)
    split_frame = pd.concat(parts, ignore_index=True)
    split_frame["task"] = "E3_w85_balanced_licensed"
    split_frame["seed"] = int(seed)
    split_frame["family_id"] = family_id
    split_frame["heldout_family"] = family_id
    split_frame["heldout_positive"] = split_frame["node_id"].isin(set(heldout_pos["node_id"]))
    split_frame["test_form"] = "balanced_licensed_negative"
    split_frame["split_group"] = split_frame["split_group"].fillna(
        split_frame["root_domain"].map(lambda value: f"root_domain::{value}")
    )

    hard_pos = heldout_pos.copy()
    hard_pos["family_eval_label"] = 1
    hard_pos["hard_eval_role"] = "heldout_family_illegal"
    hard_neg = hard_negative.copy()
    hard_neg["family_eval_label"] = 0
    hard_neg["hard_eval_role"] = "other_denmark_illegal"
    hard_eval = pd.concat([hard_pos, hard_neg], ignore_index=True)
    hard_eval["task"] = "E3_w85_hard_illegal_negative"
    hard_eval["seed"] = int(seed)
    hard_eval["family_id"] = family_id
    hard_eval["heldout_family"] = family_id
    hard_eval["test_form"] = "hard_illegal_negative"
    return split_frame.reset_index(drop=True), hard_eval.reset_index(drop=True)


def build_e3_random_split(
    bundle: GraphBundle,
    heldout_size: int,
    seed: int,
    permutation_id: int,
    reference_family: str,
) -> pd.DataFrame:
    """Build E3 random split."""
    primary = build_primary_task_frame(bundle).copy()
    denmark = primary[primary["jurisdiction"] == "Denmark"].copy().reset_index(drop=True)
    negatives = primary[primary["label"] == 0].copy()
    if heldout_size < 1 or heldout_size >= len(denmark):
        raise ValueError(f"Invalid random heldout size for E3 permutation: {heldout_size}")
    rng = np.random.default_rng((seed * 100_003) + int(permutation_id))
    heldout_positions = rng.choice(np.arange(len(denmark)), size=int(heldout_size), replace=False)
    heldout_pos = denmark.iloc[heldout_positions].copy()
    remaining_pos = denmark.drop(index=heldout_positions).reset_index(drop=True)
    heldout_id = f"perm_{permutation_id:03d}_size{heldout_size}"

    for frame in [heldout_pos, remaining_pos]:
        frame["week8_family_id"] = heldout_id
    negatives["week8_family_id"] = ""

    group_cols = list(bundle.config["splits"]["E3_lofo"]["group_columns"])
    remaining_pos = add_week8_groups(remaining_pos, group_cols)
    negatives = add_week8_groups(negatives, group_cols[1:])
    attempts = int(bundle.config["splits"]["pooled_primary"].get("balance_attempts", 20))
    pos_train_idx, pos_val_idx = best_group_shuffle(
        remaining_pos,
        test_fraction=float(bundle.config["splits"]["E3_lofo"]["val_fraction_within_train"]),
        seed=seed + 300 + permutation_id,
        attempts=attempts,
    )
    neg_train_idx, neg_temp_idx = best_group_shuffle(
        negatives,
        test_fraction=(
            float(bundle.config["splits"]["pooled_primary"]["val_fraction"])
            + float(bundle.config["splits"]["pooled_primary"]["test_fraction"])
        ),
        seed=seed + 400 + permutation_id,
        attempts=attempts,
    )
    neg_temp = negatives.iloc[neg_temp_idx].reset_index(drop=True)
    neg_val_rel, neg_test_rel = best_group_shuffle(
        neg_temp,
        test_fraction=0.5,
        seed=seed + 401 + permutation_id,
        attempts=attempts,
    )
    neg_val_idx = neg_temp_idx[neg_val_rel]
    neg_test_idx = neg_temp_idx[neg_test_rel]

    parts = []
    for split_name, frame in [
        ("train", remaining_pos.iloc[pos_train_idx]),
        ("val", remaining_pos.iloc[pos_val_idx]),
        ("train", negatives.iloc[neg_train_idx]),
        ("val", negatives.iloc[neg_val_idx]),
        ("test", negatives.iloc[neg_test_idx]),
        ("test", heldout_pos),
    ]:
        part = frame.copy()
        part["split"] = split_name
        parts.append(part)
    split_frame = pd.concat(parts, ignore_index=True)
    split_frame["task"] = "E3_permutation"
    split_frame["seed"] = int(seed)
    split_frame["family_id"] = heldout_id
    split_frame["heldout_family"] = heldout_id
    split_frame["reference_family"] = reference_family
    split_frame["permutation_id"] = int(permutation_id)
    split_frame["heldout_positive"] = split_frame["node_id"].isin(set(heldout_pos["node_id"]))
    split_frame["split_group"] = split_frame["split_group"].fillna(
        split_frame["root_domain"].map(lambda value: f"root_domain::{value}")
    )
    return split_frame.reset_index(drop=True)


def save_week8_run(
    bundle: GraphBundle,
    suffix: str,
    split_frame: pd.DataFrame,
    metrics: dict[str, Any],
    history: pd.DataFrame,
    predictions: pd.DataFrame,
    encoder_checkpoint: str,
) -> None:
    """Save week8 run."""
    save_dataframe(split_frame, bundle.output_paths["splits"] / f"{suffix}_split.csv")
    save_dataframe(history, bundle.output_paths["logs"] / f"{suffix}_history.csv")
    save_dataframe(predictions, bundle.output_paths["predictions"] / f"{suffix}_predictions.csv")
    manifest = make_common_manifest(
        bundle=bundle,
        task=str(metrics["task"]),
        model=str(metrics["model"]),
        split_name=f"{suffix}_split.csv",
        sample_counts=compute_sample_counts(split_frame),
        seed=int(metrics["seed"]),
    )
    manifest["encoder_checkpoint"] = encoder_checkpoint
    manifest["metrics"] = metrics
    record_run_manifest(bundle.output_paths["runs"] / f"{suffix}.json", manifest)


def plot_e3_summary(summary: pd.DataFrame, output_path: Path) -> None:
    """Plot E3 summary."""
    if summary.empty:
        return
    plot_frame = summary.sort_values("test_roc_auc_mean", ascending=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(9, max(4, 0.35 * len(plot_frame))))
    axis.barh(plot_frame["family_id"], plot_frame["test_roc_auc_mean"], color="#2563eb", alpha=0.75)
    axis.axvline(0.85, color="#dc2626", linestyle="--", linewidth=1)
    axis.set_xlabel("LOFO ROC-AUC")
    axis.set_ylabel("Held-out family")
    axis.set_title("Week 8 E3 Denmark Family LOFO")
    axis.set_xlim(0.0, 1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def run_e3(bundle: GraphBundle, smoke_test: bool, only_family: str | None = None) -> None:
    """Run E3."""
    if smoke_test:
        redirect_output_root(bundle, "week8_smoke")
    assignments = extract_denmark_families(bundle)
    save_dataframe(assignments, bundle.output_paths["audits"] / "E3_dk_family_assignments.csv")
    eligible = (
        assignments[assignments["week8_lofo_eligible"]]
        .groupby("week8_family_id")
        .size()
        .sort_values(ascending=False)
    )
    families = eligible.index.tolist()
    if only_family is not None:
        families = [only_family]
    elif smoke_test:
        families = ["tsars"] if "tsars" in families else families[:1]

    seeds = [int(seed) for seed in bundle.config["seeds"]]
    if smoke_test:
        seeds = seeds[:1]

    raw_rows: list[dict[str, Any]] = []
    for seed in seeds:
        encoder_state, encoder_checkpoint = load_encoder_state(bundle, seed)
        for family_id in families:
            split_frame = build_e3_lofo_split(bundle, assignments, family_id, seed)
            metrics, predictions, history, _ = run_finetune_split(
                bundle=bundle,
                graph_data=bundle.graph_data.cpu(),
                split_frame=split_frame,
                encoder_state=encoder_state,
                seed=seed,
                smoke_test=smoke_test,
                task_name="E3_lofo",
            )
            heldout = split_frame[split_frame["heldout_positive"]]
            metrics.update(
                {
                    "experiment": "E3_lofo",
                    "family_id": family_id,
                    "heldout_positive_count": int(len(heldout)),
                    "heldout_family_size": int(assignments.loc[assignments["week8_family_id"] == family_id, "week8_family_size"].iloc[0]),
                    "is_tsars": bool(family_id == "tsars"),
                }
            )
            suffix = f"E3_lofo_{family_id.replace(':', '_').replace('/', '_')}__seed{seed}"
            save_week8_run(bundle, suffix, split_frame, metrics, history, predictions, encoder_checkpoint)
            raw_rows.append(metrics)

    raw = pd.DataFrame(raw_rows)
    save_dataframe(raw, bundle.output_paths["metrics"] / "E3_lofo_family_sensitivity_raw.csv")
    if not raw.empty:
        summary = summarise_runs(
            raw,
            ["experiment", "family_id"],
            ["test_roc_auc", "test_balanced_accuracy", "test_illegal_consistency", "test_mean_pred_illegal"],
        )
    else:
        summary = pd.DataFrame()
    save_dataframe(summary, bundle.output_paths["metrics"] / "E3_lofo_family_sensitivity.csv")
    plot_e3_summary(summary, bundle.output_paths["plots"] / "E3_family_vs_random.png")


def _append_hard_eval_scores(hard_eval: pd.DataFrame, probabilities_all: np.ndarray) -> pd.DataFrame:
    """Append hard eval scores."""
    frame = hard_eval.copy()
    frame["pred_prob_illegal"] = frame["graph_node_index"].map(lambda idx: float(probabilities_all[int(idx)]))
    return frame


def _hard_eval_metrics(hard_eval_scores: pd.DataFrame, seed: int, family_id: str) -> dict[str, Any]:
    """Helper: hard eval metrics."""
    y_true = hard_eval_scores["family_eval_label"].to_numpy(dtype=int)
    y_score = hard_eval_scores["pred_prob_illegal"].to_numpy(dtype=float)
    metrics = evaluate_binary(y_true, y_score, threshold=0.5)
    pos = hard_eval_scores[hard_eval_scores["family_eval_label"] == 1]["pred_prob_illegal"].to_numpy(dtype=float)
    neg = hard_eval_scores[hard_eval_scores["family_eval_label"] == 0]["pred_prob_illegal"].to_numpy(dtype=float)
    return {
        "created_at_utc": utc_now_iso(),
        "task": "E3_w85_hard_illegal_negative",
        "model": "heco_disc_lr_finetune",
        "seed": int(seed),
        "experiment": "E3_w85_lofo",
        "family_id": family_id,
        "test_form": "hard_illegal_negative",
        "test_roc_auc": metrics["roc_auc"],
        "test_pr_auc": metrics["pr_auc"],
        "test_balanced_accuracy": metrics["balanced_accuracy"],
        "test_macro_f1": metrics["macro_f1"],
        "recall_at_fpr_0_10": recall_at_fpr(y_true, y_score, max_fpr=0.10),
        "positive_score_mean": float(np.mean(pos)) if pos.size else float("nan"),
        "negative_score_mean": float(np.mean(neg)) if neg.size else float("nan"),
        "score_gap_pos_minus_neg": float(np.mean(pos) - np.mean(neg)) if pos.size and neg.size else float("nan"),
        "heldout_positive_count": int(pos.size),
        "negative_count": int(neg.size),
    }


def run_e3_w85(bundle: GraphBundle, smoke_test: bool, only_family: str | None = None) -> None:
    """Run E3 w85."""
    redirect_output_root(bundle, "week85_smoke" if smoke_test else "week85")
    assignments = extract_denmark_families(bundle)
    save_dataframe(assignments, bundle.output_paths["audits"] / "E3_w85_dk_family_assignments.csv")
    eligible = (
        assignments[assignments["week8_lofo_eligible"]]
        .groupby("week8_family_id")
        .size()
        .sort_values(ascending=False)
    )
    families = eligible.index.tolist()
    if only_family is not None:
        families = [only_family]
    elif smoke_test:
        families = ["tsars"] if "tsars" in families else families[:1]

    seeds = [int(seed) for seed in bundle.config["seeds"]]
    if smoke_test:
        seeds = seeds[:1]

    raw_rows: list[dict[str, Any]] = []
    for seed in seeds:
        encoder_state, encoder_checkpoint = load_encoder_state(bundle, seed)
        for family_id in families:
            split_frame, hard_eval = build_e3_w85_split(bundle, assignments, family_id, seed)
            metrics, predictions, history, payload = run_finetune_split(
                bundle=bundle,
                graph_data=bundle.graph_data.cpu(),
                split_frame=split_frame,
                encoder_state=encoder_state,
                seed=seed,
                smoke_test=smoke_test,
                task_name="E3_w85_balanced_licensed",
            )
            balanced_test = predictions[predictions["split"] == "test"].copy()
            y_true = balanced_test["label"].to_numpy(dtype=int)
            y_score = balanced_test["pred_prob_illegal"].to_numpy(dtype=float)
            pos = balanced_test[balanced_test["label"] == 1]["pred_prob_illegal"].to_numpy(dtype=float)
            neg = balanced_test[balanced_test["label"] == 0]["pred_prob_illegal"].to_numpy(dtype=float)
            metrics.update(
                {
                    "experiment": "E3_w85_lofo",
                    "family_id": family_id,
                    "test_form": "balanced_licensed_negative",
                    "recall_at_fpr_0_10": recall_at_fpr(y_true, y_score, max_fpr=0.10),
                    "positive_score_mean": float(np.mean(pos)) if pos.size else float("nan"),
                    "negative_score_mean": float(np.mean(neg)) if neg.size else float("nan"),
                    "score_gap_pos_minus_neg": float(np.mean(pos) - np.mean(neg)) if pos.size and neg.size else float("nan"),
                    "heldout_positive_count": int(pos.size),
                    "negative_count": int(neg.size),
                    "encoder_checkpoint": encoder_checkpoint,
                }
            )
            suffix = f"E3W85_balanced_{family_id.replace(':', '_').replace('/', '_')}__seed{seed}"
            save_week8_run(bundle, suffix, split_frame, metrics, history, predictions, encoder_checkpoint)
            raw_rows.append(metrics)

            probabilities_all = payload["probabilities_all"].numpy()
            hard_scores = _append_hard_eval_scores(hard_eval, probabilities_all)
            hard_metrics = _hard_eval_metrics(hard_scores, seed, family_id)
            hard_metrics["encoder_checkpoint"] = encoder_checkpoint
            hard_suffix = f"E3W85_hard_{family_id.replace(':', '_').replace('/', '_')}__seed{seed}"
            save_dataframe(hard_scores, bundle.output_paths["predictions"] / f"{hard_suffix}_predictions.csv")
            manifest = make_common_manifest(
                bundle=bundle,
                task="E3_w85_hard_illegal_negative",
                model="heco_disc_lr_finetune",
                split_name=f"{hard_suffix}_hard_eval.csv",
                sample_counts={"hard_eval_rows": int(len(hard_scores))},
                seed=seed,
            )
            manifest["encoder_checkpoint"] = encoder_checkpoint
            manifest["metrics"] = hard_metrics
            record_run_manifest(bundle.output_paths["runs"] / f"{hard_suffix}.json", manifest)
            raw_rows.append(hard_metrics)

    raw = pd.DataFrame(raw_rows)
    save_dataframe(raw, bundle.output_paths["metrics"] / "E3_w85_lofo_raw.csv")
    summary = (
        summarise_runs(
            raw,
            ["experiment", "test_form", "family_id"],
            ["test_roc_auc", "test_balanced_accuracy", "recall_at_fpr_0_10", "score_gap_pos_minus_neg"],
        )
        if not raw.empty
        else pd.DataFrame()
    )
    save_dataframe(summary, bundle.output_paths["metrics"] / "E3_w85_lofo_summary.csv")


def run_e3_permutation(
    bundle: GraphBundle,
    smoke_test: bool,
    seed_filter: int | None = None,
    permutation_start: int = 0,
    permutation_count: int | None = None,
) -> None:
    """Run E3 permutation."""
    if smoke_test:
        redirect_output_root(bundle, "week8_smoke")
    assignments_path = bundle.output_paths["audits"] / "E3_dk_family_assignments.csv"
    if assignments_path.exists():
        assignments = pd.read_csv(assignments_path)
    else:
        assignments = extract_denmark_families(bundle)
        save_dataframe(assignments, assignments_path)
    eligible = (
        assignments[assignments["week8_lofo_eligible"]]
        .groupby("week8_family_id")
        .size()
        .sort_values(ascending=False)
    )
    if eligible.empty:
        raise ValueError("No eligible families available for E3 permutation")
    references = [(str(family), int(size)) for family, size in eligible.items()]
    seeds = [int(seed) for seed in bundle.config["seeds"]]
    if seed_filter is not None:
        seeds = [int(seed_filter)]
    if smoke_test:
        seeds = seeds[:1]
    total_perm = int(bundle.config["E3_lofo"].get("permutation_n", 100))
    if smoke_test:
        total_perm = min(total_perm, 2)
    start = max(0, int(permutation_start))
    stop = total_perm if permutation_count is None else min(total_perm, start + int(permutation_count))
    rows: list[dict[str, Any]] = []
    existing_path = bundle.output_paths["audits"] / "E3_permutation_aucs.csv"
    if existing_path.exists() and not smoke_test:
        existing = pd.read_csv(existing_path)
        done = set(zip(existing["seed"].astype(int), existing["permutation_id"].astype(int)))
    else:
        existing = pd.DataFrame()
        done = set()

    for seed in seeds:
        encoder_state, encoder_checkpoint = load_encoder_state(bundle, seed)
        for permutation_id in range(start, stop):
            if (seed, permutation_id) in done:
                continue
            reference_family, heldout_size = references[permutation_id % len(references)]
            split_frame = build_e3_random_split(
                bundle=bundle,
                heldout_size=heldout_size,
                seed=seed,
                permutation_id=permutation_id,
                reference_family=reference_family,
            )
            metrics, predictions, history, _ = run_finetune_split(
                bundle=bundle,
                graph_data=bundle.graph_data.cpu(),
                split_frame=split_frame,
                encoder_state=encoder_state,
                seed=seed,
                smoke_test=smoke_test,
                task_name="E3_permutation",
            )
            metrics.update(
                {
                    "experiment": "E3_permutation",
                    "permutation_id": int(permutation_id),
                    "reference_family": reference_family,
                    "heldout_positive_count": int(heldout_size),
                    "encoder_checkpoint": encoder_checkpoint,
                }
            )
            suffix = f"E3_perm_{permutation_id:03d}__seed{seed}"
            save_dataframe(history, bundle.output_paths["logs"] / f"{suffix}_history.csv")
            manifest = make_common_manifest(
                bundle=bundle,
                task="E3_permutation",
                model="heco_disc_lr_finetune",
                split_name=f"{suffix}_split_not_saved.csv",
                sample_counts=compute_sample_counts(split_frame),
                seed=seed,
            )
            manifest["encoder_checkpoint"] = encoder_checkpoint
            manifest["metrics"] = metrics
            record_run_manifest(bundle.output_paths["runs"] / f"{suffix}.json", manifest)
            rows.append(metrics)
            combined = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True)
            save_dataframe(combined, existing_path)

    combined = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True)
    save_dataframe(combined, existing_path)


def build_e4_scenario_frame(bundle: GraphBundle, scenario_id: str) -> pd.DataFrame:
    """Build E4 scenario frame."""
    scenario = bundle.config["E4_control_groups"]["scenarios"][scenario_id]
    frame = bundle.website_frame.copy()
    if bool(bundle.config["runtime"].get("exclude_isolated_websites", True)):
        frame = frame[frame["exclude_from_training_default"] != 1].copy()
    positive_tiers = set(scenario["positive_tiers"])
    negative_tiers = set(scenario["negative_tiers"])
    frame = frame[frame["sample_tier"].isin(positive_tiers.union(negative_tiers))].copy()
    frame["label"] = frame["sample_tier"].map(lambda tier: 1 if tier in positive_tiers else 0).astype(int)
    frame["task"] = scenario_id
    return frame.reset_index(drop=True)


def build_pooled_split(frame: pd.DataFrame, bundle: GraphBundle, seed: int, task_name: str) -> pd.DataFrame:
    """Build pooled split."""
    split_config = bundle.config["splits"]["pooled_primary"]
    grouped = add_week8_groups(frame, list(split_config["group_columns"]))
    train_idx, temp_idx = best_group_shuffle(
        grouped,
        test_fraction=float(split_config["val_fraction"]) + float(split_config["test_fraction"]),
        seed=seed,
        attempts=int(split_config.get("balance_attempts", 20)),
    )
    temp = grouped.iloc[temp_idx].reset_index(drop=True)
    val_rel, test_rel = best_group_shuffle(
        temp,
        test_fraction=float(split_config["test_fraction"])
        / (float(split_config["val_fraction"]) + float(split_config["test_fraction"])),
        seed=seed + 17,
        attempts=int(split_config.get("balance_attempts", 20)),
    )
    split_frame = grouped.copy()
    split_frame["split"] = "unassigned"
    split_frame.loc[train_idx, "split"] = "train"
    split_frame.loc[temp_idx[val_rel], "split"] = "val"
    split_frame.loc[temp_idx[test_rel], "split"] = "test"
    split_frame["seed"] = int(seed)
    split_frame["task"] = task_name
    return split_frame.reset_index(drop=True)


def extract_frozen_embeddings(bundle: GraphBundle, encoder_state: dict[str, torch.Tensor]) -> np.ndarray:
    """Extract frozen embeddings."""
    device = resolve_device(bundle.config)
    data = copy.deepcopy(bundle.graph_data.cpu()).to(device)
    metapath_adjacency = build_metapath_adjacency(bundle, bundle.graph_data.cpu(), device)
    encoder = make_heco_model(bundle).to(device)
    encoder.load_state_dict(copy.deepcopy(encoder_state))
    encoder.eval()
    with torch.no_grad():
        output = encoder(data, metapath_adjacency)
    return output.combined_embedding.detach().cpu().numpy()


def compute_e4_distances(bundle: GraphBundle, seed: int, embedding: np.ndarray) -> dict[str, Any]:
    """Compute E4 distances."""
    frame = bundle.website_frame.copy()
    if bool(bundle.config["runtime"].get("exclude_isolated_websites", True)):
        frame = frame[frame["exclude_from_training_default"] != 1].copy()
    idx = frame["graph_node_index"].to_numpy(dtype=int)
    tiers = frame["sample_tier"].astype(str)
    illegal = idx[tiers.str.startswith("illegal_confirmed_").to_numpy()]
    licensed = idx[(tiers == "licensed_baseline").to_numpy()]
    control = idx[(tiers == "control_legal_commercial").to_numpy()]

    def centroid_distance(a: np.ndarray, b: np.ndarray) -> float:
        """Centroid distance."""
        return float(np.linalg.norm(embedding[a].mean(axis=0) - embedding[b].mean(axis=0)))

    def spread(indices: np.ndarray) -> float:
        """Spread."""
        centered = embedding[indices] - embedding[indices].mean(axis=0, keepdims=True)
        return float(np.sqrt((centered * centered).sum(axis=1)).mean())

    sigma = float(np.mean([spread(illegal), spread(licensed), spread(control)]))
    d_ill_lic = centroid_distance(illegal, licensed)
    d_ill_ctrl = centroid_distance(illegal, control)
    d_lic_ctrl = centroid_distance(licensed, control)
    return {
        "seed": int(seed),
        "d_illegal_licensed": d_ill_lic,
        "d_illegal_control": d_ill_ctrl,
        "d_licensed_control": d_lic_ctrl,
        "sigma_mean": sigma,
        "H_E4b_pass": bool(d_ill_ctrl > d_ill_lic + 0.5 * sigma),
        "H_E4c_pass": bool(d_lic_ctrl > d_ill_lic),
    }


def save_e4_tsne(bundle: GraphBundle, seed: int, embedding: np.ndarray) -> None:
    """Save E4 tsne."""
    frame = bundle.website_frame.copy()
    tiers = {
        "illegal": frame["sample_tier"].astype(str).str.startswith("illegal_confirmed_"),
        "licensed": frame["sample_tier"].eq("licensed_baseline"),
        "control": frame["sample_tier"].eq("control_legal_commercial"),
    }
    mask = tiers["illegal"] | tiers["licensed"] | tiers["control"]
    plot_frame = frame.loc[mask, ["graph_node_index", "node_id", "root_domain", "jurisdiction", "sample_tier"]].copy()
    indices = plot_frame["graph_node_index"].to_numpy(dtype=int)
    coords = TSNE(
        n_components=2,
        perplexity=min(30, max(5, len(indices) // 8)),
        random_state=seed,
        init="pca",
        learning_rate="auto",
    ).fit_transform(embedding[indices])
    plot_frame["tsne_x"] = coords[:, 0]
    plot_frame["tsne_y"] = coords[:, 1]
    save_dataframe(plot_frame, bundle.output_paths["embeddings"] / f"E4_tsne_three_class__seed{seed}.csv")
    color_map = {
        "licensed_baseline": "#2563eb",
        "control_legal_commercial": "#059669",
        "illegal_confirmed_official_single": "#dc2626",
        "illegal_confirmed_official_cross_verified": "#7c2d12",
    }
    fig, axis = plt.subplots(figsize=(8, 6))
    for tier, part in plot_frame.groupby("sample_tier"):
        axis.scatter(part["tsne_x"], part["tsne_y"], s=18, alpha=0.72, label=tier, color=color_map.get(tier, "#6b7280"))
    axis.set_title("Week 8 E4 Three-Class HeCo Embedding")
    axis.set_xticks([])
    axis.set_yticks([])
    axis.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(bundle.output_paths["plots"] / "E4_tsne_three_class.png", dpi=170)
    plt.close(fig)


def run_e4(bundle: GraphBundle, smoke_test: bool) -> None:
    """Run E4."""
    if smoke_test:
        redirect_output_root(bundle, "week8_smoke")
    seeds = [int(seed) for seed in bundle.config["seeds"]]
    scenarios = list(bundle.config["E4_control_groups"]["scenarios"].keys())
    if smoke_test:
        seeds = seeds[:1]
        scenarios = scenarios[:1]
    raw_rows: list[dict[str, Any]] = []
    distance_rows: list[dict[str, Any]] = []
    for seed in seeds:
        encoder_state, encoder_checkpoint = load_encoder_state(bundle, seed)
        embedding = extract_frozen_embeddings(bundle, encoder_state)
        distance_rows.append(compute_e4_distances(bundle, seed, embedding))
        if seed == 42 and not smoke_test:
            save_e4_tsne(bundle, seed, embedding)
        for scenario_id in scenarios:
            scenario_frame = build_e4_scenario_frame(bundle, scenario_id)
            split_frame = build_pooled_split(scenario_frame, bundle, seed, f"E4_{scenario_id}")
            metrics, predictions, history, _ = run_finetune_split(
                bundle=bundle,
                graph_data=bundle.graph_data.cpu(),
                split_frame=split_frame,
                encoder_state=encoder_state,
                seed=seed,
                smoke_test=smoke_test,
                task_name=f"E4_{scenario_id}",
            )
            metrics.update({"experiment": "E4_control_groups", "scenario": scenario_id})
            suffix = f"E4_{scenario_id}__seed{seed}"
            save_week8_run(bundle, suffix, split_frame, metrics, history, predictions, encoder_checkpoint)
            raw_rows.append(metrics)
    raw = pd.DataFrame(raw_rows)
    save_dataframe(raw, bundle.output_paths["metrics"] / "E4_three_scenario_raw_runs.csv")
    summary = summarise_runs(raw, ["experiment", "scenario"], ["test_roc_auc", "test_balanced_accuracy", "test_macro_f1"]) if not raw.empty else pd.DataFrame()
    save_dataframe(summary, bundle.output_paths["metrics"] / "E4_three_scenario_summary.csv")
    save_dataframe(pd.DataFrame(distance_rows), bundle.output_paths["metrics"] / "E4_embedding_distances.csv")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run Week 8 multi-scenario experiments E3/E4.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "week8.yaml"))
    parser.add_argument("--experiment", choices=["E3", "E3_W85", "E3_PERM", "E4"], default="E3")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--family", default=None, help="Optional E3 held-out family id.")
    parser.add_argument("--seed", type=int, default=None, help="Optional seed filter for E3_PERM.")
    parser.add_argument("--permutation-start", type=int, default=0)
    parser.add_argument("--permutation-count", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    """Command-line entry point."""
    args = parse_args()
    bundle = load_graph_bundle(args.config)
    if args.experiment == "E3":
        run_e3(bundle, smoke_test=args.smoke_test, only_family=args.family)
    elif args.experiment == "E3_W85":
        run_e3_w85(bundle, smoke_test=args.smoke_test, only_family=args.family)
    elif args.experiment == "E3_PERM":
        run_e3_permutation(
            bundle,
            smoke_test=args.smoke_test,
            seed_filter=args.seed,
            permutation_start=args.permutation_start,
            permutation_count=args.permutation_count,
        )
    elif args.experiment == "E4":
        run_e4(bundle, smoke_test=args.smoke_test)
    else:
        raise ValueError(args.experiment)


if __name__ == "__main__":
    main()
