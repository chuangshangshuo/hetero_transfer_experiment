from __future__ import annotations

import argparse
import copy
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
import yaml
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.explain.structure_vs_lexical import apply_feature_mode
from src.train.train_transfer import (
    build_metapath_adjacency,
    choose_threshold_by_youden,
    load_encoder_state,
    make_finetune_model,
)
from src.train.utils import (
    GraphBundle,
    load_graph_bundle,
    make_common_manifest,
    record_run_manifest,
    save_dataframe,
    set_random_seed,
    utc_now_iso,
)


TARGETS = ["T2_PH", "T3_DiagnoseFrance"]
TARGET_TYPES = {
    "T2_PH": "primary hard-transfer target",
    "T3_DiagnoseFrance": "saturated / shortcut-sensitive control target",
}
TRAINING_MODE_BASELINE = "source_only_baseline_from_corrected_E5"
TRAINING_MODE_FEWSHOT = "heco_encoder_checkpoint_source_plus_balanced_target_joint_finetune"
OVERCLAIMS = {
    "T2_PH": "Do not claim few-shot universally improves transfer; this is a hard but recoverable target-specific result.",
    "T3_DiagnoseFrance": "Do not use high T3 AUC as structural-generalization evidence; Week 8 shows lexical/ccTLD assistance.",
}
SHOTS = [0, 1, 3, 5, 10]
EXPERIMENT_SEEDS = [0, 1, 2, 3, 4]
CHECKPOINT_SEEDS = [42, 43, 44, 45, 46]
FEATURE_CONDITIONS = {
    "full": "full",
    "no-ccTLD": "no_cctld",
    "no-website-lexical": "zero_website_lexical",
    "graph-only": "zero_website_all",
}
BASELINE_CONFIG = {
    "full": "C1_full",
    "no-ccTLD": "C2_no_cctld",
    "no-website-lexical": "C3_no_website_lexical",
    "graph-only": "C4_graph_only",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Week 9 few-shot target calibration.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "week8.yaml"))
    parser.add_argument("--week9-config", default=str(ROOT / "configs" / "week9_fewshot.yaml"))
    parser.add_argument("--targets", nargs="+", default=None)
    parser.add_argument("--shots", nargs="+", type=int, default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--feature-conditions", nargs="+", default=None)
    parser.add_argument("--no-merge-existing", action="store_true")
    parser.add_argument("--rebuild-from-artifacts", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def load_week9_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    if not config_path.exists():
        return {}
    loaded = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Week 9 config must be a mapping: {config_path}")
    return loaded


def load_week7_split(transfer_id: str, checkpoint_seed: int) -> pd.DataFrame:
    path = ROOT / "output" / "week7" / "splits" / f"{transfer_id}__seed{checkpoint_seed}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing Week 7 split: {path}")
    return pd.read_csv(path)


def source_baselines() -> pd.DataFrame:
    path = ROOT / "output" / "week8_patch" / "metrics" / "E5_structure_vs_lexical.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing corrected E5 baseline table: {path}")
    frame = pd.read_csv(path)
    return frame[
        frame["transfer_id"].isin(TARGETS)
        & frame["config_id"].isin(BASELINE_CONFIG.values())
        & frame["method"].eq("source_only")
    ].copy()


def baseline_for(
    baselines: pd.DataFrame,
    target_id: str,
    feature_condition: str,
    checkpoint_seed: int,
) -> pd.Series:
    config_id = BASELINE_CONFIG[feature_condition]
    subset = baselines[
        (baselines["transfer_id"] == target_id)
        & (baselines["config_id"] == config_id)
        & (baselines["seed"].astype(int) == int(checkpoint_seed))
    ]
    if subset.empty:
        raise ValueError(f"Missing baseline for {target_id} {feature_condition} seed {checkpoint_seed}")
    return subset.iloc[0]


def select_balanced_support(
    split_frame: pd.DataFrame,
    requested_shot: int,
    experiment_seed: int,
    target_id: str,
    feature_condition: str,
    checkpoint_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    pool = split_frame[split_frame["split"] == "target_adapt_unlabeled"].copy()
    counts = pool.groupby("label").size().to_dict()
    if set(counts) != {0, 1}:
        raise ValueError(f"{target_id} target adaptation pool is not binary")
    actual = min(int(requested_shot), int(counts.get(0, 0)), int(counts.get(1, 0)))
    if actual == 0:
        support = pool.iloc[0:0].copy()
    else:
        rng = np.random.default_rng((experiment_seed + 1) * 100003 + requested_shot * 997 + checkpoint_seed)
        parts = []
        for label in [0, 1]:
            part = pool[pool["label"] == label]
            chosen = rng.choice(part.index.to_numpy(), size=actual, replace=False)
            parts.append(pool.loc[chosen])
        support = pd.concat(parts, ignore_index=False).sort_index().copy()
    support_nodes = set(support["node_id"].astype(str))
    adjusted = split_frame[
        ~(
            split_frame["split"].eq("target_adapt_unlabeled")
            & split_frame["node_id"].astype(str).isin(support_nodes)
        )
    ].copy()
    if not support.empty:
        support = support.copy()
        support["split"] = "target_fewshot_train"
        support["target_labels_used_for_training"] = True
        adjusted = pd.concat([adjusted, support], ignore_index=True)
    adjusted["target_labels_used_for_training"] = adjusted["split"].eq("target_fewshot_train")
    support_audit = support.copy()
    if support_audit.empty:
        support_audit = pd.DataFrame(
            columns=[
                "target_id",
                "feature_condition",
                "shot",
                "actual_shot_per_class",
                "seed",
                "checkpoint_seed",
                "node_id",
                "graph_node_index",
                "label",
                "split_group",
            ]
        )
    else:
        support_audit.insert(0, "target_id", target_id)
        support_audit.insert(1, "feature_condition", feature_condition)
        support_audit.insert(2, "shot", int(requested_shot))
        support_audit.insert(3, "actual_shot_per_class", int(actual))
        if "seed" in support_audit.columns:
            support_audit = support_audit.rename(columns={"seed": "week7_split_seed"})
        support_audit.insert(4, "seed", int(experiment_seed))
        support_audit.insert(5, "checkpoint_seed", int(checkpoint_seed))
    return adjusted.reset_index(drop=True), support_audit, int(actual)


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


def compute_binary_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> dict[str, float]:
    y_pred = (y_score >= threshold).astype(int)
    if np.unique(y_true).size < 2:
        return {
            "auc": float("nan"),
            "pr_auc": float("nan"),
            "f1": float("nan"),
            "balanced_accuracy": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
        }
    return {
        "auc": float(roc_auc_score(y_true, y_score)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
    }


def run_fewshot_train(
    bundle: GraphBundle,
    graph_data: Any,
    split_frame: pd.DataFrame,
    target_id: str,
    feature_condition: str,
    requested_shot: int,
    actual_shot: int,
    experiment_seed: int,
    checkpoint_seed: int,
    source_auc: float,
    smoke_test: bool,
    split_file: str,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    set_random_seed(experiment_seed)
    encoder_state, encoder_checkpoint = load_encoder_state(bundle, checkpoint_seed)
    device = torch.device("cuda" if torch.cuda.is_available() and str(bundle.config["runtime"].get("device", "auto")) != "cpu" else "cpu")
    data = copy.deepcopy(graph_data).to(device)
    metapath_adjacency = build_metapath_adjacency(bundle, graph_data, device)
    labels = build_label_vector(bundle, split_frame).to(device)
    source_train_idx = index_tensor(split_frame, "source_train", device)
    fewshot_idx = index_tensor(split_frame, "target_fewshot_train", device)
    source_val_idx = index_tensor(split_frame, "source_val", device)
    target_test_idx = index_tensor(split_frame, "target_test", device)
    train_idx = torch.cat([source_train_idx, fewshot_idx], dim=0)
    if len(fewshot_idx) == 0:
        raise ValueError("run_fewshot_train requires shot > 0")

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
    target_weight = float(bundle.config.get("week9_fewshot", {}).get("target_loss_weight", 1.0))
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
        loss = source_loss + target_weight * target_loss
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            eval_logits, _, _ = model(data, metapath_adjacency)
            probs = torch.softmax(eval_logits, dim=1)[:, 1].detach().cpu().numpy()
        val = split_frame[split_frame["split"] == "source_val"]
        val_y = val["label"].to_numpy(dtype=int)
        val_s = probs[val["graph_node_index"].to_numpy(dtype=int)]
        val_auc = float(roc_auc_score(val_y, val_s)) if np.unique(val_y).size == 2 else float("nan")
        target = split_frame[split_frame["split"] == "target_test"]
        target_y = target["label"].to_numpy(dtype=int)
        target_s = probs[target["graph_node_index"].to_numpy(dtype=int)]
        target_auc = float(roc_auc_score(target_y, target_s)) if np.unique(target_y).size == 2 else float("nan")
        score = val_auc if math.isfinite(val_auc) else -float(loss.item())
        history_rows.append(
            {
                "epoch": int(epoch),
                "loss": float(loss.item()),
                "source_loss": float(source_loss.item()),
                "target_fewshot_loss": float(target_loss.item()),
                "source_val_auc": val_auc,
                "target_test_auc": target_auc,
                "runtime_s": float(time.time() - started),
            }
        )
        if math.isfinite(score) and score >= best_score:
            best_score = score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
        elif epoch >= min_epochs and best_state is not None and epoch - best_epoch >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits, _, _ = model(data, metapath_adjacency)
        probabilities = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()

    source_val = split_frame[split_frame["split"] == "source_val"]
    val_scores = probabilities[source_val["graph_node_index"].to_numpy(dtype=int)]
    threshold = choose_threshold_by_youden(source_val["label"].to_numpy(dtype=int), val_scores)
    target_test = split_frame[split_frame["split"] == "target_test"].copy()
    target_scores = probabilities[target_test["graph_node_index"].to_numpy(dtype=int)]
    metrics = compute_binary_metrics(target_test["label"].to_numpy(dtype=int), target_scores, threshold)
    prediction_frame = split_frame.copy()
    prediction_frame["pred_prob_illegal"] = prediction_frame["graph_node_index"].map(lambda idx: float(probabilities[int(idx)]))
    prediction_frame["pred_label"] = (prediction_frame["pred_prob_illegal"] >= threshold).astype(int)
    prediction_frame["target_id"] = target_id
    prediction_frame["feature_condition"] = feature_condition
    prediction_frame["shot"] = int(requested_shot)
    prediction_frame["actual_shot_per_class"] = int(actual_shot)
    prediction_frame["seed"] = int(experiment_seed)
    prediction_frame["checkpoint_seed"] = int(checkpoint_seed)

    row = {
        "created_at_utc": utc_now_iso(),
        "target_id": target_id,
        "shot": int(requested_shot),
        "actual_shot_per_class": int(actual_shot),
        "seed": int(experiment_seed),
        "checkpoint_seed": int(checkpoint_seed),
        "feature_condition": feature_condition,
        "source_only_auc": float(source_auc),
        "fewshot_auc": metrics["auc"],
        "delta_auc": metrics["auc"] - float(source_auc),
        "pr_auc": metrics["pr_auc"],
        "f1": metrics["f1"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "n_target_train": int(len(fewshot_idx)),
        "n_target_test": int(len(target_test)),
        "best_epoch": int(best_epoch),
        "trained_epochs": int(len(history_rows)),
        "threshold": float(threshold),
        "encoder_checkpoint": encoder_checkpoint,
        "training_mode": TRAINING_MODE_FEWSHOT,
        "training_strategy": TRAINING_MODE_FEWSHOT,
        "split_file": split_file,
        "is_posthoc": True,
        "runtime_s": float(time.time() - started),
    }
    return row, prediction_frame, pd.DataFrame(history_rows)


def baseline_row(
    baseline: pd.Series,
    target_id: str,
    feature_condition: str,
    shot: int,
    experiment_seed: int,
    checkpoint_seed: int,
    split_file: str,
) -> dict[str, Any]:
    auc = float(baseline["target_test_auc"])
    return {
        "created_at_utc": utc_now_iso(),
        "target_id": target_id,
        "shot": int(shot),
        "actual_shot_per_class": 0,
        "seed": int(experiment_seed),
        "checkpoint_seed": int(checkpoint_seed),
        "feature_condition": feature_condition,
        "source_only_auc": auc,
        "fewshot_auc": auc,
        "delta_auc": 0.0,
        "pr_auc": float(baseline["target_pr_auc"]),
        "f1": float(baseline["target_macro_f1"]),
        "balanced_accuracy": float(baseline["target_balanced_acc"]),
        "precision": float("nan"),
        "recall": float(baseline["target_illegal_recall_at_youden"]),
        "n_target_train": 0,
        "n_target_test": int(baseline["target_test_size"]),
        "best_epoch": int(float(baseline["best_epoch"])),
        "trained_epochs": int(float(baseline["trained_epochs"])),
        "threshold": float(baseline["threshold"]),
        "encoder_checkpoint": str(baseline["encoder_checkpoint"]),
        "training_mode": TRAINING_MODE_BASELINE,
        "training_strategy": TRAINING_MODE_BASELINE,
        "split_file": split_file,
        "is_posthoc": True,
        "runtime_s": float(baseline["runtime_s"]),
    }


def ci95(values: pd.Series) -> tuple[float, float]:
    vals = pd.to_numeric(values, errors="coerce").dropna()
    if vals.empty:
        return float("nan"), float("nan")
    mean = float(vals.mean())
    if len(vals) < 2:
        return mean, mean
    half = 1.96 * float(vals.std(ddof=1)) / math.sqrt(len(vals))
    return mean - half, mean + half


def stability_label(mean_delta: float, std_auc: float, success_rate_delta_ge_005: float) -> str:
    if abs(mean_delta) < 0.02:
        return "no meaningful improvement"
    if mean_delta > 0.10 and success_rate_delta_ge_005 >= 0.8:
        return "stable improvement"
    if mean_delta > 0.05 and success_rate_delta_ge_005 >= 0.6:
        return "moderate improvement"
    if std_auc > 0.10 or success_rate_delta_ge_005 < 0.6:
        return "unstable/sample-sensitive"
    return "limited improvement"


def wilcoxon_p(deltas: pd.Series) -> float:
    vals = pd.to_numeric(deltas, errors="coerce").dropna()
    if len(vals) < 5 or np.allclose(vals, 0.0):
        return float("nan")
    try:
        from scipy.stats import wilcoxon

        return float(wilcoxon(vals, zero_method="wilcox", alternative="two-sided").pvalue)
    except Exception:
        return float("nan")


def build_stability(raw: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (target_id, shot, condition, training_mode), part in raw.groupby(
        ["target_id", "shot", "feature_condition", "training_mode"]
    ):
        auc = pd.to_numeric(part["fewshot_auc"], errors="coerce")
        delta = pd.to_numeric(part["delta_auc"], errors="coerce")
        lo, hi = ci95(auc)
        success_any = float((delta > 0).mean())
        success_005 = float((delta >= 0.05).mean())
        mean_delta = float(delta.mean())
        std_auc = float(auc.std(ddof=1)) if len(auc) > 1 else 0.0
        rows.append(
            {
                "target_id": target_id,
                "shot": int(shot),
                "feature_condition": condition,
                "training_mode": training_mode,
                "n_seeds": int(part["seed"].nunique()),
                "mean_auc": float(auc.mean()),
                "std_auc": std_auc,
                "min_auc": float(auc.min()),
                "max_auc": float(auc.max()),
                "median_auc": float(auc.median()),
                "ci95_low": lo,
                "ci95_high": hi,
                "mean_delta_auc": mean_delta,
                "std_delta_auc": float(delta.std(ddof=1)) if len(delta) > 1 else 0.0,
                "success_rate_positive": success_any,
                "success_rate_delta_ge_005": success_005,
                "success_rate": success_any,
                "success_rate_delta_ge_0_05": success_005,
                "wilcoxon_p_vs_0": wilcoxon_p(delta) if int(shot) in {5, 10} else float("nan"),
                "stability_label": stability_label(mean_delta, std_auc, success_005),
                "is_posthoc": True,
            }
        )
    return pd.DataFrame(rows)


def shortcut_label(full_auc: float, condition_auc: float, full_delta: float, condition_delta: float) -> str:
    auc_gap = full_auc - condition_auc
    delta_gap = full_delta - condition_delta
    if auc_gap >= 0.05 or delta_gap >= 0.05:
        return "partly shortcut-dependent"
    if abs(auc_gap) < 0.02 and abs(delta_gap) < 0.02:
        return "not clearly shortcut-dependent"
    return "mixed shortcut sensitivity"


def build_shortcut_summary(stability: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    lookup = {
        (str(row.target_id), int(row.shot), str(row.feature_condition), str(row.training_mode)): row
        for row in stability.itertuples(index=False)
    }
    for row in stability.itertuples(index=False):
        full = lookup.get((str(row.target_id), int(row.shot), "full", str(row.training_mode)))
        if full is not None and str(row.feature_condition) != "full":
            label = shortcut_label(
                float(full.mean_auc),
                float(row.mean_auc),
                float(full.mean_delta_auc),
                float(row.mean_delta_auc),
            )
        elif str(row.feature_condition) == "full":
            no_cctld = lookup.get((str(row.target_id), int(row.shot), "no-ccTLD", str(row.training_mode)))
            label = (
                shortcut_label(
                    float(row.mean_auc),
                    float(no_cctld.mean_auc),
                    float(row.mean_delta_auc),
                    float(no_cctld.mean_delta_auc),
                )
                if no_cctld is not None
                else "shortcut condition not available"
            )
        else:
            label = "shortcut condition not available"
        if str(row.target_id) == "T2_PH":
            claim = (
                "Few-shot calibration substantially improves a hard but recoverable target."
                if float(row.mean_delta_auc) > 0.10 and float(row.success_rate_delta_ge_005) >= 0.8
                else "Few-shot calibration has potential for T2_PH but remains sample-sensitive."
            )
        else:
            claim = "Few-shot provides little additional benefit for saturated / shortcut-sensitive target."
        rows.append(
            {
                "target_id": row.target_id,
                "shot": int(row.shot),
                "feature_condition": row.feature_condition,
                "training_mode": row.training_mode,
                "mean_auc": float(row.mean_auc),
                "std_auc": float(row.std_auc),
                "mean_delta_vs_source_only": float(row.mean_delta_auc),
                "success_rate_positive": float(row.success_rate_positive),
                "success_rate_delta_ge_005": float(row.success_rate_delta_ge_005),
                "success_rate": float(row.success_rate_positive),
                "shortcut_dependency_label": label,
                "paper_safe_claim": claim,
                "is_posthoc": True,
            }
        )
    return pd.DataFrame(rows)


def build_rollup(stability: pd.DataFrame, shortcut_summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for target_id in TARGETS:
        full = stability[(stability["target_id"] == target_id) & (stability["feature_condition"] == "full")].copy()
        if full.empty:
            continue
        source = full[full["shot"] == 0]
        candidates = full[full["shot"] > 0].sort_values(["mean_auc", "mean_delta_auc"], ascending=False)
        if source.empty or candidates.empty:
            continue
        best = candidates.iloc[0]
        shortcut = shortcut_summary[
            (shortcut_summary["target_id"] == target_id)
            & (shortcut_summary["shot"] == int(best["shot"]))
            & (shortcut_summary["feature_condition"] == "full")
        ]
        shortcut_result = str(shortcut["shortcut_dependency_label"].iloc[0]) if not shortcut.empty else ""
        mean_delta = float(best["mean_delta_auc"])
        success_positive = float(best["success_rate_positive"])
        success_005 = float(best["success_rate_delta_ge_005"])
        if target_id == "T2_PH" and mean_delta > 0.10 and success_005 >= 0.8:
            status = "recoverable_target_supported"
            claim = "Few-shot calibration substantially improves a hard but recoverable target."
        elif target_id == "T2_PH":
            status = "potential_but_sample_sensitive"
            claim = "Few-shot calibration has potential but is sample-sensitive."
        elif abs(mean_delta) < 0.02:
            status = "saturated_no_clear_gain"
            claim = "Few-shot provides little additional benefit for saturated / shortcut-sensitive target."
        else:
            status = "target_dependent_mixed"
            claim = "Few-shot calibration is target-dependent and should not be generalized."
        rows.append(
            {
                "target_id": target_id,
                "target_type": TARGET_TYPES[target_id],
                "source_only_auc": float(source["mean_auc"].iloc[0]),
                "best_shot": int(best["shot"]),
                "best_mean_auc": float(best["mean_auc"]),
                "best_std_auc": float(best["std_auc"]),
                "best_mean_delta_auc": mean_delta,
                "success_rate_positive": success_positive,
                "success_rate_delta_ge_005": success_005,
                "success_rate": success_positive,
                "shortcut_condition_result": shortcut_result,
                "final_status": status,
                "paper_safe_claim": claim,
                "overclaim_to_avoid": OVERCLAIMS[target_id],
                "is_posthoc": True,
            }
        )
    return pd.DataFrame(rows)


def build_acceptance_v2(rollup: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in rollup.itertuples(index=False):
        final_status = str(row.final_status)
        if final_status == "recoverable_target_supported":
            status = "posthoc_support"
        elif final_status == "saturated_no_clear_gain":
            status = "no_clear_gain"
        else:
            status = final_status
        target = "T3" if str(row.target_id) == "T3_DiagnoseFrance" else str(row.target_id)
        rows.append(
            {
                "target": target,
                "best_shot": int(row.best_shot),
                "best_auc": float(row.best_mean_auc),
                "delta": float(row.best_mean_delta_auc),
                "status": status,
                "source_file": "output/week9/metrics/week9_fewshot_rollup.csv",
                "is_posthoc": True,
            }
        )
    return pd.DataFrame(rows)


def mark_legacy_preliminary_outputs(metrics_dir: Path) -> None:
    legacy_files = [
        "fewshot_raw_runs.csv",
        "fewshot_summary.csv",
        "fewshot_acceptance_audit.csv",
    ]
    superseded_by = (
        "output/week9/metrics/W9_E1_fewshot_curve.csv;"
        "output/week9/metrics/W9_E2_fewshot_seed_stability.csv;"
        "output/week9/metrics/W9_E3_shortcut_aware_summary.csv;"
        "output/week9/metrics/week9_fewshot_rollup.csv"
    )
    for name in legacy_files:
        path = metrics_dir / name
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        frame["week9_result_status"] = "legacy_preformal"
        frame["superseded_by"] = superseded_by
        frame["paper_use"] = "legacy_only_do_not_use_as_authoritative"
        frame["legacy_marked_at_utc"] = utc_now_iso()
        save_dataframe(frame, path)


def condition_from_path_token(token: str) -> str:
    return {
        "full": "full",
        "no_ccTLD": "no-ccTLD",
        "no_website_lexical": "no-website-lexical",
        "graph_only": "graph-only",
    }.get(token, token.replace("_", "-"))


def split_identity(split_path: Path) -> tuple[str, str, int, int]:
    stem = split_path.stem
    if not stem.endswith("_split"):
        raise ValueError(f"Unexpected Week 9 split filename: {split_path.name}")
    stem = stem.removesuffix("_split")
    parts = stem.split("__")
    if len(parts) != 4 or not parts[0].startswith("W9_"):
        raise ValueError(f"Unexpected Week 9 split filename: {split_path.name}")
    target_id = parts[0].removeprefix("W9_")
    condition = condition_from_path_token(parts[1])
    shot = int(parts[2].removeprefix("k"))
    seed = int(parts[3].removeprefix("seed"))
    return target_id, condition, shot, seed


def write_week9_outputs(raw: pd.DataFrame, support_all: pd.DataFrame, out: Path) -> pd.DataFrame:
    save_dataframe(raw, out / "metrics" / "W9_E1_fewshot_curve.csv")
    save_dataframe(support_all, out / "audits" / "W9_fewshot_support_samples.csv")
    stability = build_stability(raw)
    save_dataframe(stability, out / "metrics" / "W9_E2_fewshot_seed_stability.csv")
    shortcut = build_shortcut_summary(stability)
    save_dataframe(raw, out / "metrics" / "W9_E3_shortcut_aware_fewshot.csv")
    save_dataframe(shortcut, out / "metrics" / "W9_E3_shortcut_aware_summary.csv")
    rollup = build_rollup(stability, shortcut)
    save_dataframe(rollup, out / "metrics" / "week9_fewshot_rollup.csv")
    acceptance_v2 = build_acceptance_v2(rollup)
    save_dataframe(acceptance_v2, out / "metrics" / "fewshot_acceptance_audit_v2.csv")
    mark_legacy_preliminary_outputs(out / "metrics")
    plot_target_curve(stability, "T2_PH", out / "plots" / "W9_T2_PH_shot_curve.png")
    plot_target_curve(stability, "T3_DiagnoseFrance", out / "plots" / "W9_T3_France_shot_curve.png")
    plot_delta(stability, out / "plots" / "W9_delta_auc_by_shot.png")
    plot_shortcut(stability, out / "plots" / "W9_shortcut_condition_comparison.png")
    write_docs(rollup, stability, shortcut)
    return rollup


def rebuild_from_artifacts() -> None:
    out = ROOT / "output" / "week9"
    for sub in ["metrics", "plots", "logs", "runs", "predictions", "audits", "splits"]:
        (out / sub).mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for manifest_path in sorted((out / "runs").glob("W9_*.json")):
        import json

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        metrics = manifest.get("metrics")
        if isinstance(metrics, dict):
            rows.append(metrics)

    baselines = source_baselines()
    for split_path in sorted((out / "splits").glob("W9_*__k0__seed*_split.csv")):
        target_id, condition, shot, seed = split_identity(split_path)
        checkpoint_seed = CHECKPOINT_SEEDS[seed % len(CHECKPOINT_SEEDS)]
        baseline = baseline_for(baselines, target_id, condition, checkpoint_seed)
        split_file = str(split_path.relative_to(ROOT))
        rows.append(baseline_row(baseline, target_id, condition, shot, seed, checkpoint_seed, split_file))

    raw = pd.DataFrame(rows)
    raw = raw.drop_duplicates(
        subset=["target_id", "shot", "seed", "checkpoint_seed", "feature_condition", "training_mode"],
        keep="last",
    )
    raw = raw.sort_values(["target_id", "feature_condition", "training_mode", "shot", "seed"]).reset_index(drop=True)
    support_parts: list[pd.DataFrame] = []
    for split_path in sorted((out / "splits").glob("W9_*_split.csv")):
        split = pd.read_csv(split_path)
        support = split[split["split"].eq("target_fewshot_train")].copy()
        if not support.empty:
            support_parts.append(support)
    support_all = pd.concat(support_parts, ignore_index=True, sort=False) if support_parts else pd.DataFrame()
    support_all = support_all.drop_duplicates(
        subset=["target_id", "feature_condition", "shot", "seed", "checkpoint_seed", "node_id"],
        keep="last",
    )
    rollup = write_week9_outputs(raw, support_all, out)
    print(rollup.to_string(index=False))


def plot_target_curve(stability: pd.DataFrame, target_id: str, output_path: Path) -> None:
    target_frame = stability[stability["target_id"] == target_id]
    if target_frame.empty:
        return
    fig, axis = plt.subplots(figsize=(7, 4.5))
    for condition, part in target_frame.groupby("feature_condition"):
        part = part.sort_values("shot")
        axis.errorbar(part["shot"], part["mean_auc"], yerr=part["std_auc"], marker="o", capsize=3, label=condition)
    axis.set_title(f"Week 9 Few-Shot Calibration: {target_id}")
    axis.set_xlabel("Target shots per class")
    axis.set_ylabel("Target ROC-AUC")
    axis.set_ylim(0.0, 1.02)
    axis.grid(alpha=0.2)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def plot_delta(stability: pd.DataFrame, output_path: Path) -> None:
    if stability.empty:
        return
    fig, axis = plt.subplots(figsize=(8, 4.8))
    for key, part in stability.groupby(["target_id", "feature_condition"]):
        target_id, condition = key
        part = part.sort_values("shot")
        axis.plot(part["shot"], part["mean_delta_auc"], marker="o", label=f"{target_id} / {condition}")
    axis.axhline(0.0, color="black", linewidth=1)
    axis.axhline(0.05, color="gray", linewidth=1, linestyle="--")
    axis.set_title("Week 9 Delta AUC by Shot")
    axis.set_xlabel("Target shots per class")
    axis.set_ylabel("Mean delta vs source-only")
    axis.grid(alpha=0.2)
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def plot_shortcut(stability: pd.DataFrame, output_path: Path) -> None:
    focus = stability[stability["shot"].isin([5, 10])].copy()
    if focus.empty:
        return
    labels = [f"{r.target_id}\n{k}" for k, r in enumerate(focus.itertuples(index=False), start=1)]
    fig, axis = plt.subplots(figsize=(9, 4.8))
    axis.bar(range(len(focus)), focus["mean_auc"], color=["#4c78a8" if c == "full" else "#f58518" for c in focus["feature_condition"]])
    axis.set_xticks(range(len(focus)))
    axis.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    axis.set_ylabel("Mean ROC-AUC")
    axis.set_title("Shortcut Condition Comparison (5/10-shot)")
    for idx, row in enumerate(focus.itertuples(index=False)):
        axis.text(idx, float(row.mean_auc) + 0.015, str(row.feature_condition), ha="center", va="bottom", fontsize=7)
    axis.set_ylim(0.0, 1.08)
    axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    text = frame.copy()
    for column in text.columns:
        if pd.api.types.is_float_dtype(text[column]):
            text[column] = text[column].map(lambda value: "" if pd.isna(value) else f"{float(value):.4f}")
        else:
            text[column] = text[column].map(lambda value: "" if pd.isna(value) else str(value))
    headers = list(text.columns)
    rows = text.values.tolist()
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def merge_existing_frame(
    new_frame: pd.DataFrame,
    path: Path,
    key_columns: list[str],
    enabled: bool,
) -> pd.DataFrame:
    if not enabled or not path.exists():
        return new_frame.copy()
    existing = pd.read_csv(path)
    if existing.empty:
        return new_frame.copy()
    combined = pd.concat([existing, new_frame], ignore_index=True, sort=False)
    available_keys = [column for column in key_columns if column in combined.columns]
    if available_keys:
        combined = combined.drop_duplicates(subset=available_keys, keep="last")
    sort_columns = [
        column
        for column in ["target_id", "feature_condition", "training_mode", "shot", "seed", "checkpoint_seed", "node_id"]
        if column in combined.columns
    ]
    if sort_columns:
        combined = combined.sort_values(sort_columns).reset_index(drop=True)
    return combined


def condition_matrix_text(stability: pd.DataFrame) -> str:
    if stability.empty:
        return "- feature conditions: none\n"
    lines = []
    for target_id, part in stability.groupby("target_id"):
        conditions = ", ".join(sorted(str(value) for value in part["feature_condition"].dropna().unique()))
        lines.append(f"- {target_id}: {conditions}")
    return "\n".join(lines) + "\n"


def feature_condition_interpretation(stability: pd.DataFrame) -> str:
    t2 = stability[
        (stability["target_id"] == "T2_PH")
        & (stability["shot"] == 10)
        & (stability["feature_condition"].isin(["full", "no-ccTLD", "no-website-lexical", "graph-only"]))
    ].copy()
    if t2.empty:
        return ""
    parts = []
    for condition in ["full", "no-ccTLD", "no-website-lexical", "graph-only"]:
        row = t2[t2["feature_condition"] == condition]
        if row.empty:
            continue
        parts.append(
            f"{condition}: mean AUC {float(row['mean_auc'].iloc[0]):.4f}, "
            f"delta {float(row['mean_delta_auc'].iloc[0]):+.4f}, "
            f"success(delta>=0.05) {float(row['success_rate_delta_ge_005'].iloc[0]):.2f}"
        )
    if not parts:
        return ""
    return (
        "For T2_PH, the expanded feature-condition audit shows that requested 10-shot calibration "
        "remains strong even after removing Website lexical features or using the graph-only condition: "
        + "; ".join(parts)
        + ". This supports a stronger recoverable-target claim than the full/no-ccTLD matrix alone, "
        "but it should still be written as target calibration evidence rather than proof of broad pure "
        "structural generalization.\n"
    )


def write_docs(rollup: pd.DataFrame, stability: pd.DataFrame, shortcut: pd.DataFrame) -> None:
    docs = ROOT / "docs"
    paper = ROOT / "paper" / "draft"
    docs.mkdir(parents=True, exist_ok=True)
    paper.mkdir(parents=True, exist_ok=True)
    rollup_md = markdown_table(rollup)
    stability_focus = stability[(stability["shot"].isin([0, 5, 10])) & (stability["feature_condition"].isin(["full", "no-ccTLD"]))]
    stability_md = markdown_table(
        stability_focus[
            [
                "target_id",
                "shot",
                "feature_condition",
                "training_mode",
                "mean_auc",
                "std_auc",
                "mean_delta_auc",
                "success_rate_positive",
                "success_rate_delta_ge_005",
                "stability_label",
            ]
        ]
    )
    shortcut_md = markdown_table(
        shortcut[
            [
                "target_id",
                "shot",
                "feature_condition",
                "training_mode",
                "mean_delta_vs_source_only",
                "success_rate_delta_ge_005",
                "shortcut_dependency_label",
                "paper_safe_claim",
            ]
        ]
    )
    condition_lines = condition_matrix_text(stability)
    feature_interpretation = feature_condition_interpretation(stability)
    (docs / "week9_fewshot_experiment_design.md").write_text(
        "# Week 9 Few-Shot Experiment Design\n\n"
        "Title: Few-shot Target Calibration for Recoverable Transfer.\n\n"
        "Week 9 tests whether a small number of balanced target labels can repair hard transfer targets. "
        "It is not framed as universal few-shot improvement. The primary target is T2_PH; T3_DiagnoseFrance "
        "is a saturated / shortcut-sensitive control. T1_Nordic and T2_ON are one-class targets and are not "
        "included in the standard ROC-AUC main experiment.\n\n"
        "Main questions:\n\n"
        "- Does few-shot target calibration improve source-only transfer?\n"
        "- Is improvement concentrated on the hard transfer target?\n"
        "- Is improvement stable across random seeds?\n"
        "- Does improvement depend on ccTLD / Website lexical shortcut information?\n\n"
        "Implemented P0/P1 matrix:\n\n"
        "- targets: T2_PH, T3_DiagnoseFrance\n"
        "- shots: 0, 1, 3, 5, 10\n"
        "- seeds: 0, 1, 2, 3, 4, mapped to existing split/checkpoint seeds 42, 43, 44, 45, 46\n"
        "- base feature conditions in `configs/week9_fewshot.yaml`: full, no-ccTLD\n"
        "- training fallback: saved HeCo encoder checkpoint plus source_train and balanced target support joint fine-tuning\n\n"
        "Formal result feature-condition coverage:\n\n"
        f"{condition_lines}\n"
        "All outputs are post-hoc calibration evidence and should not be treated as preregistered Week 8 evidence.\n",
        encoding="utf-8",
    )
    (docs / "week9_fewshot_protocol.md").write_text(
        "# Week 9 Few-Shot Protocol\n\n"
        "The protocol uses existing Week 7 target adaptation pools and target test splits. Few-shot samples "
        "are drawn only from `target_adapt_unlabeled`; selected support nodes are removed from that pool and "
        "never appear in `target_test`.\n\n"
        "Sampling is class-balanced. If the requested shot count exceeds the minority-class adaptation pool, "
        "the actual balanced support size is capped and recorded in `actual_shot_per_class`. For T2_PH, "
        "requested 10-shot is capped at 8 per class in the current Week 7 splits.\n\n"
        "Training mode: saved HeCo encoder checkpoint plus source_train and balanced target support joint "
        "fine-tuning. This is the documented fallback because Week 7 did not save source-only classifier "
        "checkpoint files. The 0-shot rows are source-only baselines from corrected E5 for the matching "
        "feature condition, split seed, and checkpoint seed.\n\n"
        "Every result row records target_id, shot, actual_shot_per_class, seed, checkpoint_seed, feature_condition, "
        "source_only_auc, fewshot_auc, delta_auc, target metrics, and target train/test counts. Selected target "
        "support sample ids are saved in `output/week9/audits/W9_fewshot_support_samples.csv`, and every "
        "run writes its formal split under `output/week9/splits/`.\n",
        encoding="utf-8",
    )
    (paper / "sec_week9_fewshot_calibration.md").write_text(
        "# Week 9 Few-Shot Target Calibration\n\n"
        "Week 9 evaluates few-shot target calibration for recoverable transfer. The result should not be "
        "read as universal few-shot improvement. Instead, few-shot calibration is target-dependent: it may "
        "repair a hard but recoverable target, while saturated or shortcut-sensitive targets may show little "
        "additional gain.\n\n"
        "## Rollup\n\n"
        f"{rollup_md}\n\n"
        "## Seed Stability\n\n"
        f"{stability_md}\n\n"
        "## Shortcut-Aware Interpretation\n\n"
        f"{shortcut_md}\n\n"
        "## Feature-Condition Expansion\n\n"
        f"{feature_interpretation}\n"
        "Paper-safe interpretation: T2_PH is the key hard-transfer target. If its few-shot improvement is "
        "stable, the safe claim is that few-shot calibration substantially improves a hard but recoverable "
        "target. T3 France is already saturated and Week 8 E5 showed lexical/ccTLD assistance, so little "
        "additional few-shot gain should be interpreted as target saturation rather than structural proof.\n",
        encoding="utf-8",
    )


def run(args: argparse.Namespace) -> None:
    if args.rebuild_from_artifacts:
        rebuild_from_artifacts()
        return
    week9_config = load_week9_config(args.week9_config)
    bundle = load_graph_bundle(args.config)
    bundle.config.setdefault("week9_fewshot", {})
    bundle.config["week9_fewshot"].setdefault("target_loss_weight", 1.0)
    baselines = source_baselines()
    out = ROOT / "output" / "week9"
    for sub in ["metrics", "plots", "logs", "runs", "predictions", "audits", "splits"]:
        (out / sub).mkdir(parents=True, exist_ok=True)

    configured_targets = args.targets or week9_config.get("targets") or TARGETS
    configured_shots = args.shots or week9_config.get("shots") or SHOTS
    configured_seeds = args.seeds or week9_config.get("seeds") or EXPERIMENT_SEEDS
    configured_conditions = args.feature_conditions or week9_config.get("feature_conditions") or ["full", "no-ccTLD"]
    targets = list(configured_targets[:1] if args.smoke_test else configured_targets)
    shots = [int(value) for value in (configured_shots[:2] if args.smoke_test else configured_shots)]
    seeds = [int(value) for value in (configured_seeds[:1] if args.smoke_test else configured_seeds)]
    conditions = list(configured_conditions[:1] if args.smoke_test else configured_conditions)
    unknown_conditions = sorted(set(conditions) - set(FEATURE_CONDITIONS))
    if unknown_conditions:
        raise ValueError(f"Unknown Week 9 feature conditions: {unknown_conditions}")
    seed_map = {seed: CHECKPOINT_SEEDS[idx % len(CHECKPOINT_SEEDS)] for idx, seed in enumerate(seeds)}

    raw_rows: list[dict[str, Any]] = []
    support_rows: list[pd.DataFrame] = []
    for target_id in targets:
        for condition in conditions:
            feature_mode = FEATURE_CONDITIONS[condition]
            graph_data = apply_feature_mode(bundle.graph_data, bundle.feature_schema, feature_mode)
            for experiment_seed in seeds:
                checkpoint_seed = seed_map[int(experiment_seed)]
                split = load_week7_split(target_id, checkpoint_seed)
                baseline = baseline_for(baselines, target_id, condition, checkpoint_seed)
                source_auc = float(baseline["target_test_auc"])
                for shot in shots:
                    suffix = f"W9_{target_id}__{condition.replace('-', '_')}__k{int(shot)}__seed{int(experiment_seed)}"
                    split_path = out / "splits" / f"{suffix}_split.csv"
                    split_file = str(split_path.relative_to(ROOT))
                    if int(shot) == 0:
                        baseline_split = split.copy()
                        baseline_split["target_labels_used_for_training"] = False
                        baseline_split["target_id"] = target_id
                        baseline_split["feature_condition"] = condition
                        baseline_split["shot"] = 0
                        baseline_split["actual_shot_per_class"] = 0
                        baseline_split["seed"] = int(experiment_seed)
                        baseline_split["checkpoint_seed"] = checkpoint_seed
                        save_dataframe(baseline_split, split_path)
                        raw_rows.append(
                            baseline_row(
                                baseline,
                                target_id,
                                condition,
                                int(shot),
                                int(experiment_seed),
                                checkpoint_seed,
                                split_file,
                            )
                        )
                        continue
                    fewshot_split, support, actual_shot = select_balanced_support(
                        split,
                        int(shot),
                        int(experiment_seed),
                        target_id,
                        condition,
                        checkpoint_seed,
                    )
                    fewshot_split["target_id"] = target_id
                    fewshot_split["feature_condition"] = condition
                    fewshot_split["shot"] = int(shot)
                    fewshot_split["actual_shot_per_class"] = actual_shot
                    fewshot_split["seed"] = int(experiment_seed)
                    fewshot_split["checkpoint_seed"] = checkpoint_seed
                    save_dataframe(fewshot_split, split_path)
                    support_rows.append(support)
                    row, predictions, history = run_fewshot_train(
                        bundle=bundle,
                        graph_data=graph_data,
                        split_frame=fewshot_split,
                        target_id=target_id,
                        feature_condition=condition,
                        requested_shot=int(shot),
                        actual_shot=actual_shot,
                        experiment_seed=int(experiment_seed),
                        checkpoint_seed=checkpoint_seed,
                        source_auc=source_auc,
                        smoke_test=bool(args.smoke_test),
                        split_file=split_file,
                    )
                    save_dataframe(predictions, out / "predictions" / f"{suffix}_predictions.csv")
                    save_dataframe(history, out / "logs" / f"{suffix}_history.csv")
                    manifest = make_common_manifest(
                        bundle=bundle,
                        task="W9_fewshot_target_calibration",
                        model="heco_checkpoint_fewshot_calibration",
                        split_name=f"{target_id}__seed{checkpoint_seed}.csv",
                        sample_counts={
                            "target_fewshot_train": int(row["n_target_train"]),
                            "target_test": int(row["n_target_test"]),
                        },
                        seed=int(experiment_seed),
                    )
                    manifest["checkpoint_seed"] = checkpoint_seed
                    manifest["shot"] = int(shot)
                    manifest["actual_shot_per_class"] = actual_shot
                    manifest["feature_condition"] = condition
                    manifest["metrics"] = row
                    manifest["is_posthoc"] = True
                    record_run_manifest(out / "runs" / f"{suffix}.json", manifest)
                    raw_rows.append(row)
                    save_dataframe(pd.DataFrame(raw_rows), out / "metrics" / "W9_E1_fewshot_curve.partial.csv")

    raw_new = pd.DataFrame(raw_rows)
    merge_existing = (not args.no_merge_existing) and (not args.smoke_test)
    raw = merge_existing_frame(
        raw_new,
        out / "metrics" / "W9_E1_fewshot_curve.csv",
        ["target_id", "shot", "seed", "checkpoint_seed", "feature_condition", "training_mode"],
        merge_existing,
    )
    save_dataframe(raw, out / "metrics" / "W9_E1_fewshot_curve.csv")
    support_new = pd.concat(support_rows, ignore_index=True, sort=False) if support_rows else pd.DataFrame()
    support_all = merge_existing_frame(
        support_new,
        out / "audits" / "W9_fewshot_support_samples.csv",
        ["target_id", "feature_condition", "shot", "seed", "checkpoint_seed", "node_id"],
        merge_existing,
    )
    rollup = write_week9_outputs(raw, support_all, out)
    print(rollup.to_string(index=False))


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
