from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, train_test_split


PRIMARY_NEGATIVE_TIER = "licensed_baseline"


@dataclass
class GraphBundle:
    config: dict[str, Any]
    config_path: Path
    config_hash: str
    graph_data: Any
    graph_metadata: dict[str, Any]
    feature_schema: dict[str, Any]
    website_frame: pd.DataFrame
    output_paths: dict[str, Path]


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def save_yaml(path: Path, payload: dict[str, Any]) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def hash_dict(payload: dict[str, Any]) -> str:
    dump = yaml.safe_dump(payload, sort_keys=True, allow_unicode=True)
    return hashlib.sha256(dump.encode("utf-8")).hexdigest()


def resolve_workspace_path(config: dict[str, Any], relative_path: str) -> Path:
    return Path(config["workspace_root"]) / relative_path


def build_output_paths(config: dict[str, Any]) -> dict[str, Path]:
    output_paths = {
        name: resolve_workspace_path(config, rel_path)
        for name, rel_path in config["output"].items()
    }
    for path in output_paths.values():
        ensure_directory(path)
    return output_paths


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(config: dict[str, Any]) -> torch.device:
    requested = config["runtime"].get("device", "auto")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def load_graph_bundle(config_path: Path | str) -> GraphBundle:
    config_path = Path(config_path)
    config = load_yaml(config_path)
    output_paths = build_output_paths(config)
    graph_path = resolve_workspace_path(config, config["data"]["graph_pt"])
    metadata_path = resolve_workspace_path(config, config["data"]["graph_metadata"])
    registry_path = resolve_workspace_path(config, config["data"]["master_site_registry"])
    feature_schema_path = resolve_workspace_path(config, config["data"]["node_feature_schema"])

    graph_data = torch.load(graph_path, map_location="cpu", weights_only=False)
    graph_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    feature_schema = json.loads(feature_schema_path.read_text(encoding="utf-8"))
    registry = pd.read_csv(registry_path)

    graph_node_ids = pd.DataFrame(
        {
            "node_id": list(graph_data["Website"].node_id),
            "graph_node_index": np.arange(graph_data["Website"].x.shape[0], dtype=int),
        }
    )
    website_frame = graph_node_ids.merge(registry, on="node_id", how="left", validate="1:1")
    required_columns = [
        "node_index",
        "sample_tier",
        "jurisdiction",
        "exclude_from_training_default",
        "feat_domain_len_z",
        "feat_root_len_z",
        "feat_domain_segment_count_z",
        "feat_digit_count_z",
        "feat_hyphen_count_z",
        "feat_is_dot_com",
        "feat_is_local_tld",
    ]
    missing_mask = website_frame[required_columns].isna().any(axis=1)
    if missing_mask.any():
        missing = website_frame.loc[missing_mask, "node_id"].tolist()[:10]
        raise ValueError(
            "Website registry merge produced missing required training values "
            f"for node_ids: {missing}"
        )

    website_frame["exclude_from_training_default"] = (
        pd.to_numeric(website_frame["exclude_from_training_default"], errors="coerce")
        .fillna(0)
        .astype(int)
    )
    website_frame["is_isolated"] = (
        pd.to_numeric(website_frame["is_isolated"], errors="coerce").fillna(0).astype(int)
    )
    website_frame["label_binary"] = website_frame["sample_tier"].map(
        lambda tier: 0
        if tier == PRIMARY_NEGATIVE_TIER
        else 1
        if str(tier).startswith("illegal_confirmed_")
        else np.nan
    )

    bundle = GraphBundle(
        config=config,
        config_path=config_path,
        config_hash=hash_dict(config),
        graph_data=graph_data,
        graph_metadata=graph_metadata,
        feature_schema=feature_schema,
        website_frame=website_frame.sort_values("graph_node_index").reset_index(drop=True),
        output_paths=output_paths,
    )
    validate_graph_bundle(bundle)
    return bundle


def validate_graph_bundle(bundle: GraphBundle) -> None:
    graph = bundle.graph_data
    expected_node_types = {"Website", "IP", "Certificate", "NameServer", "Registrar", "ExternalReference"}
    expected_edge_types = {
        ("Website", "hosted_on", "IP"),
        ("Website", "uses_cert", "Certificate"),
        ("Website", "uses_ns", "NameServer"),
        ("Website", "registered_via", "Registrar"),
        ("Website", "referenced_by", "ExternalReference"),
        ("IP", "rev_hosted_on", "Website"),
        ("Certificate", "rev_uses_cert", "Website"),
        ("NameServer", "rev_uses_ns", "Website"),
        ("Registrar", "rev_registered_via", "Website"),
        ("ExternalReference", "rev_referenced_by", "Website"),
    }
    if set(graph.node_types) != expected_node_types:
        raise ValueError(f"Unexpected node types: {graph.node_types}")
    if set(graph.edge_types) != expected_edge_types:
        raise ValueError(f"Unexpected edge types: {graph.edge_types}")

    for edge_type in graph.edge_types:
        store = graph[edge_type]
        if not hasattr(store, "edge_weight"):
            raise ValueError(f"Missing edge_weight for edge type {edge_type}")
        if store.edge_weight.shape[0] != store.edge_index.shape[1]:
            raise ValueError(f"edge_weight length mismatch for edge type {edge_type}")

    if bundle.config["runtime"].get("validate_no_all_zero_feature_columns", False):
        for node_type in graph.node_types:
            features = graph[node_type].x
            zero_columns = (features.abs().sum(dim=0) == 0).nonzero(as_tuple=False).flatten().tolist()
            if zero_columns:
                raise ValueError(f"Detected all-zero columns in {node_type}: {zero_columns}")

    website_feature_dim = bundle.feature_schema["Website"]["feature_dim"]
    if tuple(graph["Website"].x.shape) != (len(bundle.website_frame), website_feature_dim):
        raise ValueError("Website feature matrix does not match registry or feature schema")


def build_primary_task_frame(bundle: GraphBundle) -> pd.DataFrame:
    config = bundle.config
    label_space = set(config["tasks"]["primary_label_space"])
    frame = bundle.website_frame.copy()
    frame = frame[frame["sample_tier"].isin(label_space)].copy()
    if config["runtime"].get("exclude_isolated_websites", True):
        frame = frame[frame["exclude_from_training_default"] != 1].copy()
    frame["label"] = frame["label_binary"].astype(int)
    return frame.reset_index(drop=True)


def build_same_region_task_frame(bundle: GraphBundle, jurisdiction: str) -> pd.DataFrame:
    frame = build_primary_task_frame(bundle)
    frame = frame[frame["jurisdiction"] == jurisdiction].copy()
    return frame.reset_index(drop=True)


def build_transfer_source_target_frames(bundle: GraphBundle) -> tuple[pd.DataFrame, pd.DataFrame]:
    primary = build_primary_task_frame(bundle)
    transfer_config = bundle.config["tasks"]["transfer_pooled_to_france"]
    target_jurisdiction = transfer_config["target_jurisdiction"]
    source_excluded = transfer_config["source_excluded_jurisdiction"]
    source = primary[primary["jurisdiction"] != source_excluded].copy().reset_index(drop=True)
    target = primary[primary["jurisdiction"] == target_jurisdiction].copy().reset_index(drop=True)
    return source, target


def add_split_group(
    frame: pd.DataFrame,
    config: dict[str, Any],
    split_section: str = "pooled_primary",
) -> pd.DataFrame:
    split_config = config["splits"][split_section]
    group_columns = split_config.get("group_columns")
    if not group_columns:
        group_columns = [
            split_config.get("group_column", "operator_or_case"),
            split_config.get("group_fallback_column", "root_domain"),
        ]
    grouped = frame.copy()
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
        raise ValueError(f"Unable to construct split groups for node_ids: {missing}")
    grouped["split_group"] = split_group
    grouped["split_group_source"] = split_group_source
    return grouped


def _distribution_score(
    frame: pd.DataFrame,
    candidate_idx: np.ndarray,
    target_fraction: float,
) -> float:
    candidate = frame.iloc[candidate_idx]
    score = abs((len(candidate) / len(frame)) - target_fraction)
    for column in ["label", "sample_tier"]:
        overall = frame[column].value_counts(normalize=True)
        selected = candidate[column].value_counts(normalize=True)
        all_keys = overall.index.union(selected.index)
        score += float((overall.reindex(all_keys, fill_value=0) - selected.reindex(all_keys, fill_value=0)).abs().sum())
    return score


def _best_group_shuffle_split(
    frame: pd.DataFrame,
    test_fraction: float,
    seed: int,
    attempts: int,
) -> tuple[np.ndarray, np.ndarray]:
    groups = frame["split_group"].to_numpy()
    best_train_idx: np.ndarray | None = None
    best_test_idx: np.ndarray | None = None
    best_score = float("inf")
    for attempt in range(max(1, attempts)):
        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=test_fraction,
            random_state=(seed * 1009) + attempt,
        )
        train_idx, test_idx = next(splitter.split(frame.index.to_numpy(), groups=groups))
        score = _distribution_score(frame, test_idx, test_fraction)
        if score < best_score:
            best_score = score
            best_train_idx = train_idx
            best_test_idx = test_idx
    if best_train_idx is None or best_test_idx is None:
        raise RuntimeError("GroupShuffleSplit failed to produce a candidate split")
    return best_train_idx, best_test_idx


def make_pooled_primary_split(frame: pd.DataFrame, seed: int, config: dict[str, Any]) -> pd.DataFrame:
    split_config = config["splits"]["pooled_primary"]
    train_size = split_config["train_fraction"]
    val_size = split_config["val_fraction"]
    test_size = split_config["test_fraction"]
    if not math.isclose(train_size + val_size + test_size, 1.0, rel_tol=1e-6):
        raise ValueError("Pooled primary split fractions must sum to 1.0")

    grouped_frame = add_split_group(frame, config, "pooled_primary").reset_index(drop=True)
    attempts = int(split_config.get("balance_attempts", 20))
    train_pos, temp_pos = _best_group_shuffle_split(
        grouped_frame,
        test_fraction=(1.0 - train_size),
        seed=seed,
        attempts=attempts,
    )
    temp = grouped_frame.iloc[temp_pos].reset_index(drop=True)
    val_ratio_within_temp = val_size / (val_size + test_size)
    val_pos_in_temp, test_pos_in_temp = _best_group_shuffle_split(
        temp,
        test_fraction=(1.0 - val_ratio_within_temp),
        seed=seed + 17,
        attempts=attempts,
    )

    split_frame = grouped_frame.copy()
    split_frame["split"] = "unassigned"
    split_frame.loc[train_pos, "split"] = "train"
    temp_original_pos = temp_pos
    val_original_pos = temp_original_pos[val_pos_in_temp]
    test_original_pos = temp_original_pos[test_pos_in_temp]
    split_frame.loc[val_original_pos, "split"] = "val"
    split_frame.loc[test_original_pos, "split"] = "test"
    split_frame["seed"] = seed
    split_frame["task"] = "pooled_primary"
    return split_frame


def make_same_region_folds(frame: pd.DataFrame, jurisdiction: str, config: dict[str, Any]) -> pd.DataFrame:
    split_config = config["splits"]["same_region_sensitivity"]
    grouped_frame = add_split_group(frame, config, "same_region_sensitivity").reset_index(drop=True)
    splitter = GroupKFold(n_splits=split_config["n_splits"])
    fold_frames: list[pd.DataFrame] = []
    for fold_id, (train_val_idx, test_idx) in enumerate(
        splitter.split(
            grouped_frame.index.to_numpy(),
            grouped_frame["label"].to_numpy(),
            groups=grouped_frame["split_group"].to_numpy(),
        ),
        start=1,
    ):
        train_val = grouped_frame.iloc[train_val_idx].reset_index(drop=False).rename(columns={"index": "original_pos"})
        val_relative_train_idx, val_relative_idx = _best_group_shuffle_split(
            train_val,
            test_fraction=split_config["val_fraction_within_train"],
            seed=int(split_config["random_state"]) + fold_id,
            attempts=int(split_config.get("balance_attempts", 20)),
        )
        train_original_pos = train_val.iloc[val_relative_train_idx]["original_pos"].to_numpy()
        val_original_pos = train_val.iloc[val_relative_idx]["original_pos"].to_numpy()
        test_original_pos = test_idx
        split_frame = grouped_frame.copy()
        split_frame["split"] = "unassigned"
        split_frame.loc[train_original_pos, "split"] = "train"
        split_frame.loc[val_original_pos, "split"] = "val"
        split_frame.loc[test_original_pos, "split"] = "test"
        split_frame["task"] = "same_region_sensitivity"
        split_frame["jurisdiction_eval"] = jurisdiction
        split_frame["fold"] = fold_id
        fold_frames.append(split_frame)
    return pd.concat(fold_frames, ignore_index=True)


def make_split_audit(
    split_frame: pd.DataFrame,
    task: str,
    seed: int | None = None,
    jurisdiction: str | None = None,
    fold: int | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for split_name, split_part in split_frame.groupby("split", dropna=False):
        row: dict[str, Any] = {
            "task": task,
            "seed": seed,
            "jurisdiction": jurisdiction,
            "fold": fold,
            "split": split_name,
            "num_rows": int(len(split_part)),
            "num_groups": int(split_part["split_group"].nunique()) if "split_group" in split_part else 0,
            "licensed_baseline": int((split_part["sample_tier"] == "licensed_baseline").sum()),
            "illegal_single": int((split_part["sample_tier"] == "illegal_confirmed_official_single").sum()),
            "illegal_cross_verified": int(
                (split_part["sample_tier"] == "illegal_confirmed_official_cross_verified").sum()
            ),
            "label_0": int((split_part["label"] == 0).sum()),
            "label_1": int((split_part["label"] == 1).sum()),
        }
        rows.append(row)

    leaking_groups = (
        split_frame.groupby("split_group")["split"]
        .nunique()
        .reset_index(name="num_splits")
        .query("num_splits > 1")
    )
    rows.append(
        {
            "task": task,
            "seed": seed,
            "jurisdiction": jurisdiction,
            "fold": fold,
            "split": "__group_overlap_audit__",
            "num_rows": int(len(leaking_groups)),
            "num_groups": int(len(leaking_groups)),
            "licensed_baseline": 0,
            "illegal_single": 0,
            "illegal_cross_verified": 0,
            "label_0": 0,
            "label_1": 0,
        }
    )
    return pd.DataFrame(rows)


def make_transfer_france_split(
    source_frame: pd.DataFrame,
    target_frame: pd.DataFrame,
    seed: int,
    config: dict[str, Any],
) -> pd.DataFrame:
    split_config = config["splits"]["transfer_pooled_to_france"]
    target_adapt = split_config["target_adapt_fraction"]
    target_val = split_config["target_val_fraction"]
    target_test = split_config["target_test_fraction"]
    if not math.isclose(target_adapt + target_val + target_test, 1.0, rel_tol=1e-6):
        raise ValueError("Transfer target split fractions must sum to 1.0")

    target_idx = target_frame.index.to_numpy()
    non_test_idx, test_idx = train_test_split(
        target_idx,
        test_size=target_test,
        random_state=seed,
        shuffle=True,
        stratify=target_frame["label"],
    )
    non_test_frame = target_frame.loc[non_test_idx]
    adapt_ratio_within_non_test = target_adapt / (target_adapt + target_val)
    adapt_idx, val_idx = train_test_split(
        non_test_frame.index.to_numpy(),
        test_size=(1.0 - adapt_ratio_within_non_test),
        random_state=seed,
        shuffle=True,
        stratify=non_test_frame["label"],
    )

    source_split = source_frame.copy()
    source_split["split"] = "source_train"
    source_split["task"] = "transfer_pooled_to_france"
    source_split["seed"] = seed
    source_split["domain_role"] = "source"

    target_split = target_frame.copy()
    target_split["split"] = "unassigned"
    target_split.loc[adapt_idx, "split"] = "target_adapt_unlabeled"
    target_split.loc[val_idx, "split"] = "target_val"
    target_split.loc[test_idx, "split"] = "target_test"
    target_split["task"] = "transfer_pooled_to_france"
    target_split["seed"] = seed
    target_split["domain_role"] = "target"

    return pd.concat([source_split, target_split], ignore_index=True)


def save_dataframe(frame: pd.DataFrame, path: Path) -> None:
    ensure_parent(path)
    frame.to_csv(path, index=False)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_parent(path)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def choose_threshold_by_youden(y_true: np.ndarray, y_score: np.ndarray) -> float:
    unique_classes = np.unique(y_true)
    if unique_classes.size < 2:
        return 0.5
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    if thresholds.size == 0:
        return 0.5
    youden = tpr - fpr
    best_idx = int(np.argmax(youden))
    threshold = float(thresholds[best_idx])
    if not np.isfinite(threshold):
        return 0.5
    return threshold


def safe_metric(metric_fn, y_true: np.ndarray, y_score: np.ndarray) -> float:
    unique_classes = np.unique(y_true)
    if unique_classes.size < 2:
        return float("nan")
    return float(metric_fn(y_true, y_score))


def compute_binary_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    y_pred = (y_score >= threshold).astype(int)
    metrics = {
        "roc_auc": safe_metric(roc_auc_score, y_true, y_score),
        "pr_auc": safe_metric(average_precision_score, y_true, y_score),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "threshold": float(threshold),
    }
    return metrics


def summarise_runs(
    frame: pd.DataFrame,
    group_columns: Iterable[str],
    metric_columns: Iterable[str],
) -> pd.DataFrame:
    grouped = frame.groupby(list(group_columns), dropna=False)
    records: list[dict[str, Any]] = []
    for group_key, group_frame in grouped:
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        record = {column: value for column, value in zip(group_columns, group_key)}
        record["num_runs"] = int(len(group_frame))
        for metric in metric_columns:
            record[f"{metric}_mean"] = float(group_frame[metric].mean())
            record[f"{metric}_std"] = float(group_frame[metric].std(ddof=0))
        records.append(record)
    return pd.DataFrame(records)


def build_prediction_frame(
    split_frame: pd.DataFrame,
    probabilities: dict[int, float],
    threshold: float,
    model_name: str,
    task_name: str,
    seed: int | None = None,
    fold: int | None = None,
) -> pd.DataFrame:
    frame = split_frame.copy()
    frame["pred_prob_illegal"] = frame["graph_node_index"].map(probabilities).astype(float)
    frame["pred_label"] = (frame["pred_prob_illegal"] >= threshold).astype(int)
    frame["model"] = model_name
    frame["task"] = task_name
    if seed is not None:
        frame["seed"] = seed
    if fold is not None:
        frame["fold"] = fold
    return frame


def record_run_manifest(path: Path, payload: dict[str, Any]) -> None:
    ensure_parent(path)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def metric_columns() -> list[str]:
    return ["roc_auc", "pr_auc", "balanced_accuracy", "macro_f1", "threshold"]


def make_common_manifest(
    bundle: GraphBundle,
    task: str,
    model: str,
    split_name: str,
    sample_counts: dict[str, int],
    seed: int | None = None,
    fold: int | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "created_at_utc": utc_now_iso(),
        "graph_version": bundle.graph_metadata.get("graph_version"),
        "graph_schema": bundle.graph_metadata.get("schema"),
        "config_path": str(bundle.config_path),
        "config_hash": bundle.config_hash,
        "task": task,
        "model": model,
        "split_name": split_name,
        "sample_counts": sample_counts,
    }
    if seed is not None:
        payload["seed"] = seed
    if fold is not None:
        payload["fold"] = fold
    return payload


def compute_sample_counts(frame: pd.DataFrame, split_column: str = "split") -> dict[str, int]:
    return {str(name): int(count) for name, count in frame.groupby(split_column).size().items()}


def plot_mean_roc_pr(prediction_frame: pd.DataFrame, output_path: Path) -> None:
    ensure_parent(output_path)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    roc_axis, pr_axis = axes
    roc_grid = np.linspace(0.0, 1.0, 101)
    pr_grid = np.linspace(0.0, 1.0, 101)

    for model_name, model_frame in prediction_frame.groupby("model"):
        roc_curves: list[np.ndarray] = []
        pr_curves: list[np.ndarray] = []
        for _, seed_frame in model_frame.groupby("seed"):
            y_true = seed_frame["label"].to_numpy()
            y_score = seed_frame["pred_prob_illegal"].to_numpy()
            if np.unique(y_true).size < 2:
                continue
            fpr, tpr, _ = roc_curve(y_true, y_score)
            precision, recall, _ = precision_recall_curve(y_true, y_score)
            roc_curves.append(np.interp(roc_grid, fpr, tpr))
            pr_curves.append(np.interp(pr_grid, recall[::-1], precision[::-1]))
        if not roc_curves:
            continue
        roc_mean = np.mean(roc_curves, axis=0)
        pr_mean = np.mean(pr_curves, axis=0)
        roc_axis.plot(roc_grid, roc_mean, label=model_name)
        pr_axis.plot(pr_grid, pr_mean, label=model_name)

    roc_axis.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    roc_axis.set_title("Week 4 Pooled Primary ROC")
    roc_axis.set_xlabel("False Positive Rate")
    roc_axis.set_ylabel("True Positive Rate")
    pr_axis.set_title("Week 4 Pooled Primary PR")
    pr_axis.set_xlabel("Recall")
    pr_axis.set_ylabel("Precision")
    for axis in axes:
        axis.legend()
        axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_transfer_comparison(summary_frame: pd.DataFrame, output_path: Path) -> None:
    ensure_parent(output_path)
    fig, axis = plt.subplots(figsize=(7, 5))
    ordered = summary_frame.sort_values("model").reset_index(drop=True)
    axis.bar(
        ordered["model"],
        ordered["roc_auc_mean"],
        yerr=ordered["roc_auc_std"],
        color=["#3b82f6", "#ef4444"][: len(ordered)],
        capsize=6,
    )
    axis.set_ylim(0.0, 1.0)
    axis.set_title("Week 4 France Transfer ROC-AUC")
    axis.set_ylabel("ROC-AUC")
    axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
