"""Week-11 P3: full few-shot calibration matrix (4 targets x 5 shots x 5 seeds)."""
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
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.train.train_transfer import (
    build_label_vector,
    build_metapath_adjacency,
    choose_threshold_by_youden,
    evaluate_binary,
    load_encoder_state,
    make_finetune_model,
)
from src.train.utils import load_graph_bundle, set_random_seed, utc_now_iso


def load_config(path: Path) -> dict[str, Any]:
    """Load config."""
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def ensure_output_dirs(config: dict[str, Any]) -> dict[str, Path]:
    """Ensure output directories."""
    root = Path(config["workspace_root"])
    paths = {name: root / rel for name, rel in config["output"].items()}
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def index_tensor(split_frame: pd.DataFrame, split_name: str, device: torch.device) -> torch.Tensor:
    """Index tensor."""
    return torch.tensor(
        split_frame.loc[split_frame["split"] == split_name, "graph_node_index"].to_numpy(dtype=np.int64),
        dtype=torch.long,
        device=device,
    )


def select_oneclass_support(
    split_frame: pd.DataFrame,
    transfer_id: str,
    shot: int,
    seed: int,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Select oneclass support."""
    proto = config["P3_full_fewshot"]["one_class_protocols"][transfer_id]
    target_label = int(proto["target_support_label"])
    pool = split_frame[
        split_frame["split"].eq("target_adapt_unlabeled")
        & split_frame["label"].astype(int).eq(target_label)
    ].copy()
    priority = proto.get("target_support_tier_priority")
    if priority:
        priority_pool = pool[pool["sample_tier"].astype(str).eq(str(priority))].copy()
        if len(priority_pool) >= min(int(shot), len(pool)):
            pool = priority_pool
    actual = min(int(shot), len(pool))
    if actual == 0:
        support = pool.iloc[0:0].copy()
    else:
        rng = np.random.default_rng(seed * 8191 + int(shot))
        chosen = rng.choice(pool.index.to_numpy(), size=actual, replace=False)
        support = pool.loc[chosen].copy()
    adjusted = split_frame.copy()
    if not support.empty:
        support_nodes = set(support["node_id"].astype(str))
        adjusted = adjusted[
            ~(
                adjusted["split"].eq("target_adapt_unlabeled")
                & adjusted["node_id"].astype(str).isin(support_nodes)
            )
        ].copy()
        support["split"] = "target_fewshot_train"
        support["target_labels_used_for_training"] = True
        adjusted = pd.concat([adjusted, support], ignore_index=True)
    adjusted["target_labels_used_for_training"] = adjusted["split"].eq("target_fewshot_train")
    support_audit = support.copy()
    if "seed" in support_audit.columns:
        support_audit = support_audit.rename(columns={"seed": "week7_split_seed"})
    support_audit.insert(0, "target_id", transfer_id)
    support_audit.insert(1, "shot", int(shot))
    support_audit.insert(2, "seed", int(seed))
    support_audit.insert(3, "actual_target_support", int(actual))
    return adjusted.reset_index(drop=True), support_audit, int(actual)


def run_oneclass_fewshot(
    bundle,
    graph_data: Any,
    split_frame: pd.DataFrame,
    transfer_id: str,
    seed: int,
    shot: int,
    config: dict[str, Any],
    output_paths: dict[str, Path],
    smoke_test: bool = False,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run oneclass few-shot."""
    started = time.time()
    set_random_seed(seed)
    fewshot_split, support_audit, actual = select_oneclass_support(split_frame, transfer_id, shot, seed, config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = copy.deepcopy(graph_data).to(device)
    metapath_adjacency = build_metapath_adjacency(bundle, graph_data, device)
    encoder_state, encoder_checkpoint = load_encoder_state(bundle, seed)
    model = make_finetune_model(bundle, encoder_state, device)
    labels = build_label_vector(bundle, fewshot_split).to(device)
    source_train_idx = index_tensor(fewshot_split, "source_train", device)
    source_val_idx = index_tensor(fewshot_split, "source_val", device)
    fewshot_idx = index_tensor(fewshot_split, "target_fewshot_train", device)
    target_test_idx = index_tensor(fewshot_split, "target_test", device)
    p3 = config["P3_full_fewshot"]
    optimizer = torch.optim.Adam(
        [
            {"params": model.encoder_parameters(), "lr": float(bundle.config["finetune"]["encoder_lr"])},
            {"params": model.head_parameters(), "lr": float(bundle.config["finetune"]["classifier_lr"])},
        ],
        weight_decay=float(bundle.config["finetune"]["weight_decay"]),
    )
    max_epochs = 5 if smoke_test else int(p3["max_epochs"])
    min_epochs = 0 if smoke_test else int(p3["min_epochs_before_early_stop"])
    patience = max_epochs if smoke_test else int(p3["patience"])
    target_loss_weight = float(p3["target_loss_weight"])
    best_state: dict[str, Any] | None = None
    best_score = float("-inf")
    best_epoch = 0
    history_rows: list[dict[str, Any]] = []
    for epoch in range(1, max_epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits, _, _ = model(data, metapath_adjacency)
        source_loss = F.cross_entropy(logits[source_train_idx], labels[source_train_idx])
        if fewshot_idx.numel() > 0:
            target_loss = F.cross_entropy(logits[fewshot_idx], labels[fewshot_idx])
        else:
            target_loss = torch.tensor(0.0, device=device)
        loss = source_loss + target_loss_weight * target_loss
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            eval_logits, _, _ = model(data, metapath_adjacency)
            probs = torch.softmax(eval_logits, dim=1)[:, 1].detach().cpu().numpy()
        source_val = fewshot_split[fewshot_split["split"] == "source_val"]
        source_val_scores = probs[source_val["graph_node_index"].to_numpy(dtype=int)]
        source_val_labels = source_val["label"].to_numpy(dtype=int)
        val_auc = float("nan")
        if np.unique(source_val_labels).size == 2:
            val_auc = evaluate_binary(source_val_labels, source_val_scores, 0.5)["roc_auc"]
        score = val_auc if math.isfinite(val_auc) else -float(loss.detach().cpu().item())
        history_rows.append(
            {
                "epoch": epoch,
                "transfer_id": transfer_id,
                "seed": seed,
                "shot": shot,
                "loss": float(loss.detach().cpu().item()),
                "source_loss": float(source_loss.detach().cpu().item()),
                "target_fewshot_loss": float(target_loss.detach().cpu().item()),
                "source_val_auc": float(val_auc),
                "runtime_s": float(time.time() - started),
            }
        )
        if math.isfinite(score) and score >= best_score:
            best_score = float(score)
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
        elif epoch >= min_epochs and best_state is not None and epoch - best_epoch >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        final_logits, _, _ = model(data, metapath_adjacency)
        probs_all = torch.softmax(final_logits, dim=1)[:, 1].detach().cpu().numpy()
    source_val = fewshot_split[fewshot_split["split"] == "source_val"].copy()
    threshold = choose_threshold_by_youden(
        source_val["label"].to_numpy(dtype=int),
        probs_all[source_val["graph_node_index"].to_numpy(dtype=int)],
    )
    target = fewshot_split[fewshot_split["split"] == "target_test"].copy()
    target_scores = probs_all[target["graph_node_index"].to_numpy(dtype=int)]
    metrics = evaluate_binary(target["label"].to_numpy(dtype=int), target_scores, threshold)
    primary_metric = config["P3_full_fewshot"]["one_class_protocols"][transfer_id]["target_metric"]
    if primary_metric == "illegal_recall_at_youden":
        primary_value = metrics["illegal_consistency"]
        direction = "higher_is_better"
    elif primary_metric == "mean_pred_illegal":
        primary_value = metrics["mean_pred_illegal"]
        direction = "lower_is_better"
    else:
        primary_value = metrics.get(primary_metric, float("nan"))
        direction = "higher_is_better"
    prediction_frame = fewshot_split.copy()
    prediction_frame["pred_prob_illegal"] = prediction_frame["graph_node_index"].map(lambda idx: float(probs_all[int(idx)]))
    prediction_frame["pred_label"] = (prediction_frame["pred_prob_illegal"] >= threshold).astype(int)
    prediction_frame["method"] = "week11_oneclass_fewshot"
    row = {
        "created_at_utc": utc_now_iso(),
        "target_id": transfer_id,
        "shot": int(shot),
        "seed": int(seed),
        "checkpoint_seed": int(seed),
        "feature_condition": "full",
        "training_mode": "source_plus_oneclass_target_support_joint_finetune",
        "metric_family": "one_class_boundary",
        "target_primary_metric": primary_metric,
        "primary_metric_direction": direction,
        "primary_metric_value": float(primary_value),
        "target_test_auc": metrics["roc_auc"],
        "pr_auc": metrics["pr_auc"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "f1": metrics["macro_f1"],
        "target_licensed_consistency": metrics["licensed_consistency"],
        "target_illegal_consistency": metrics["illegal_consistency"],
        "target_mean_pred_illegal": metrics["mean_pred_illegal"],
        "actual_target_support": int(actual),
        "n_target_train": int(len(fewshot_idx)),
        "n_target_test": int(len(target)),
        "threshold": float(threshold),
        "best_epoch": int(best_epoch),
        "trained_epochs": int(len(history_rows)),
        "encoder_checkpoint": encoder_checkpoint,
        "is_prerun_frozen": True,
        "is_posthoc": False,
        "paper_safe_claim": "One-class few-shot is a boundary calibration check, not a standard ROC-AUC claim.",
        "paper_forbidden_claim": config["P3_full_fewshot"]["one_class_protocols"][transfer_id]["forbidden_claim"],
        "runtime_s": float(time.time() - started),
    }
    return row, prediction_frame, pd.DataFrame(history_rows), support_audit


def build_week9_full_rows(config: dict[str, Any]) -> pd.DataFrame:
    """Build week9 full rows."""
    w9 = pd.read_csv(ROOT / config["inputs"]["week9_fewshot_curve"])
    w9 = w9[
        w9["target_id"].isin(["T2_PH", "T3_DiagnoseFrance"])
        & w9["feature_condition"].eq("full")
    ].copy()
    rows = []
    for _, row in w9.iterrows():
        rows.append(
            {
                "created_at_utc": row.get("created_at_utc", ""),
                "target_id": row["target_id"],
                "shot": int(row["shot"]),
                "seed": int(row.get("checkpoint_seed", row["seed"])),
                "checkpoint_seed": int(row.get("checkpoint_seed", row["seed"])),
                "feature_condition": "full",
                "training_mode": row["training_mode"],
                "metric_family": "binary_auc",
                "target_primary_metric": "roc_auc",
                "primary_metric_direction": "higher_is_better",
                "primary_metric_value": float(row["fewshot_auc"]),
                "source_only_baseline": float(row["source_only_auc"]),
                "delta_vs_source_only": float(row["delta_auc"]),
                "target_test_auc": float(row["fewshot_auc"]),
                "pr_auc": float(row["pr_auc"]),
                "balanced_accuracy": float(row["balanced_accuracy"]),
                "f1": float(row["f1"]),
                "target_licensed_consistency": np.nan,
                "target_illegal_consistency": float(row["recall"]),
                "target_mean_pred_illegal": np.nan,
                "actual_target_support": int(row.get("actual_shot_per_class", row["shot"])) * 2,
                "n_target_train": int(row["n_target_train"]),
                "n_target_test": int(row["n_target_test"]),
                "threshold": np.nan,
                "best_epoch": np.nan,
                "trained_epochs": np.nan,
                "encoder_checkpoint": row.get("encoder_checkpoint", ""),
                "is_prerun_frozen": False,
                "is_posthoc": True,
                "paper_safe_claim": "Week 9 binary few-shot remains target-dependent; T2_PH improves and T3 is saturated.",
                "paper_forbidden_claim": "Do not claim few-shot universally improves transfer.",
                "runtime_s": np.nan,
            }
        )
    return pd.DataFrame(rows)


def add_source_baselines(frame: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Add source baselines."""
    week7 = pd.read_csv(ROOT / config["inputs"]["week7_transfer_summary"])
    baselines = week7[week7["method"].eq("source_only")][["transfer_id", "seed", "primary_metric_value"]]
    keyed = {(row.transfer_id, int(row.seed)): float(row.primary_metric_value) for row in baselines.itertuples()}
    values = []
    deltas = []
    for row in frame.itertuples():
        baseline = keyed.get((row.target_id, int(row.seed)))
        if baseline is None:
            baseline = float(getattr(row, "source_only_baseline", np.nan))
        values.append(baseline)
        direction = getattr(row, "primary_metric_direction", "higher_is_better")
        current = float(row.primary_metric_value)
        if not math.isfinite(float(baseline)) or not math.isfinite(current):
            deltas.append(float("nan"))
        elif direction == "lower_is_better":
            deltas.append(float(baseline - current))
        else:
            deltas.append(float(current - baseline))
    frame = frame.copy()
    frame["source_only_baseline"] = values
    frame["delta_vs_source_only"] = deltas
    return frame


def plot_curves(frame: pd.DataFrame, output_path: Path) -> None:
    """Plot curves."""
    targets = ["T1_Nordic", "T2_PH", "T2_ON", "T3_DiagnoseFrance"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, target in zip(axes.flatten(), targets):
        part = frame[frame["target_id"].eq(target)].copy()
        if part.empty:
            ax.axis("off")
            continue
        agg = part.groupby("shot")["primary_metric_value"].agg(["mean", "std"]).reset_index()
        ax.errorbar(agg["shot"], agg["mean"], yerr=agg["std"].fillna(0.0), marker="o", capsize=3)
        metric = part["target_primary_metric"].iloc[0]
        ax.set_title(f"{target}: {metric}")
        ax.set_xlabel("shot")
        ax.set_ylabel("primary metric")
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Week 11 P3 full transfer few-shot matrix.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "week11.yaml"))
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Command-line entry point."""
    args = parse_args()
    config = load_config(Path(args.config))
    output_paths = ensure_output_dirs(config)
    bundle = load_graph_bundle(ROOT / config["inputs"]["week7_config"])
    graph_data = bundle.graph_data
    p3 = config["P3_full_fewshot"]
    transfer_ids = list(p3["transfers"])
    seeds = [int(seed) for seed in config["seeds"]]
    shots = [int(shot) for shot in p3["shots_per_class"]]
    if args.smoke_test:
        transfer_ids = transfer_ids[:1]
        seeds = seeds[:1]
        shots = shots[:2]
    rows: list[dict[str, Any]] = []
    supports: list[pd.DataFrame] = []
    for transfer_id in transfer_ids:
        for seed in seeds:
            split_frame = pd.read_csv(ROOT / "output" / "week7" / "splits" / f"{transfer_id}__seed{seed}.csv")
            for shot in shots:
                if shot == 0:
                    continue
                row, preds, history, support = run_oneclass_fewshot(
                    bundle,
                    graph_data,
                    split_frame,
                    transfer_id,
                    seed,
                    shot,
                    config,
                    output_paths,
                    smoke_test=args.smoke_test,
                )
                suffix = f"P3_fewshot_{transfer_id}__seed{seed}__shots{shot}"
                preds.to_csv(output_paths["runs"] / f"{suffix}_predictions.csv", index=False, encoding="utf-8")
                history.to_csv(output_paths["logs"] / f"{suffix}_history.csv", index=False, encoding="utf-8")
                split_frame_out = preds[["node_id", "graph_node_index", "label", "split", "target_labels_used_for_training"]].copy()
                split_frame_out.to_csv(output_paths["audits"] / f"{suffix}_split.csv", index=False, encoding="utf-8")
                (output_paths["runs"] / f"{suffix}.json").write_text(
                    json.dumps({"created_at_utc": utc_now_iso(), "metrics": row}, indent=2),
                    encoding="utf-8",
                )
                rows.append(row)
                supports.append(support)

    new_frame = pd.DataFrame(rows)
    zero_rows = []
    week7 = pd.read_csv(ROOT / config["inputs"]["week7_transfer_summary"])
    for transfer_id in transfer_ids:
        subset = week7[(week7["transfer_id"].eq(transfer_id)) & week7["method"].eq("source_only")]
        for _, row in subset.iterrows():
            primary_metric = p3["one_class_protocols"][transfer_id]["target_metric"]
            direction = "lower_is_better" if primary_metric == "mean_pred_illegal" else "higher_is_better"
            zero_rows.append(
                {
                    "created_at_utc": row["created_at_utc"],
                    "target_id": transfer_id,
                    "shot": 0,
                    "seed": int(row["seed"]),
                    "checkpoint_seed": int(row["seed"]),
                    "feature_condition": "full",
                    "training_mode": "source_only_baseline",
                    "metric_family": "one_class_boundary",
                    "target_primary_metric": primary_metric,
                    "primary_metric_direction": direction,
                    "primary_metric_value": float(row["primary_metric_value"]),
                    "target_test_auc": row["target_test_auc"],
                    "pr_auc": row["target_pr_auc"],
                    "balanced_accuracy": row["target_balanced_acc"],
                    "f1": row["target_macro_f1"],
                    "target_licensed_consistency": row["target_licensed_consistency"],
                    "target_illegal_consistency": row["target_illegal_consistency"],
                    "target_mean_pred_illegal": row["target_mean_pred_illegal"],
                    "actual_target_support": 0,
                    "n_target_train": 0,
                    "n_target_test": int(row["target_test_size"]),
                    "threshold": row["threshold"],
                    "best_epoch": row["best_epoch"],
                    "trained_epochs": row["trained_epochs"],
                    "encoder_checkpoint": "",
                    "is_prerun_frozen": False,
                    "is_posthoc": True,
                    "paper_safe_claim": "This is the Week 7 source-only one-class boundary baseline.",
                    "paper_forbidden_claim": p3["one_class_protocols"][transfer_id]["forbidden_claim"],
                    "runtime_s": row["runtime_s"],
                }
            )
    combined = pd.concat([pd.DataFrame(zero_rows), new_frame, build_week9_full_rows(config)], ignore_index=True, sort=False)
    combined = add_source_baselines(combined, config)
    combined.to_csv(output_paths["metrics"] / "P3_full_fewshot_matrix.csv", index=False, encoding="utf-8")
    if supports:
        pd.concat(supports, ignore_index=True, sort=False).to_csv(
            output_paths["audits"] / "P3_oneclass_support_samples.csv",
            index=False,
            encoding="utf-8",
        )
    plot_curves(combined, output_paths["plots"] / "P3_fewshot_curves.png")
    print(f"P3 rows={len(combined)} new_oneclass_rows={len(new_frame)}")


if __name__ == "__main__":
    main()
