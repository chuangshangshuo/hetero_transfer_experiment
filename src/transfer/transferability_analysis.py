from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.train.multi_scenario_eval import extract_frozen_embeddings
from src.train.train_transfer import (
    add_transfer_groups,
    best_group_split,
    load_encoder_state,
    summarize_run_metrics,
    train_heco_transfer_model,
)
from src.train.utils import (
    compute_sample_counts,
    build_primary_task_frame,
    load_graph_bundle,
    make_common_manifest,
    record_run_manifest,
    save_dataframe,
)
from src.transfer.logme_score import compute_logme


def compute_h_divergence(emb_s: np.ndarray, emb_t: np.ndarray, seed: int = 42) -> float:
    x = np.vstack([emb_s, emb_t])
    y = np.concatenate([np.zeros(len(emb_s), dtype=int), np.ones(len(emb_t), dtype=int)])
    if min(np.bincount(y)) < 3:
        return float("nan")
    cv = StratifiedKFold(n_splits=min(5, int(min(np.bincount(y)))), shuffle=True, random_state=seed)
    clf = LogisticRegression(max_iter=1000, solver="liblinear", random_state=seed)
    accuracy = float(cross_val_score(clf, x, y, cv=cv, scoring="accuracy").mean())
    error = 1.0 - accuracy
    return float(2.0 * (1.0 - 2.0 * error))


def compute_sliced_wasserstein(
    emb_s: np.ndarray,
    emb_t: np.ndarray,
    seed: int = 42,
    n_projections: int = 64,
) -> float:
    rng = np.random.default_rng(seed)
    dim = emb_s.shape[1]
    distances = []
    for _ in range(int(n_projections)):
        direction = rng.normal(size=dim)
        direction = direction / max(np.linalg.norm(direction), 1e-12)
        proj_s = np.sort(emb_s @ direction)
        proj_t = np.sort(emb_t @ direction)
        q = np.linspace(0.0, 1.0, num=max(len(proj_s), len(proj_t)))
        s_quant = np.quantile(proj_s, q)
        t_quant = np.quantile(proj_t, q)
        distances.append(float(np.mean(np.abs(s_quant - t_quant))))
    return float(np.mean(distances))


def valid_region_pairs(task_frame: pd.DataFrame, min_per_class: int) -> list[tuple[str, str]]:
    regions = sorted(task_frame["jurisdiction"].dropna().unique().tolist())
    valid_regions = []
    for region in regions:
        part = task_frame[task_frame["jurisdiction"] == region]
        counts = part["label"].value_counts()
        if counts.get(0, 0) >= min_per_class and counts.get(1, 0) >= min_per_class:
            valid_regions.append(region)
    return [(source, target) for source, target in itertools.permutations(valid_regions, 2)]


def compute_pair_scores(bundle: Any, smoke_test: bool = False) -> pd.DataFrame:
    task_frame = build_primary_task_frame(bundle)
    min_per_class = int(bundle.config["E2_logme"].get("min_per_class_per_region", 2))
    pairs = valid_region_pairs(task_frame, min_per_class=min_per_class)
    if smoke_test:
        pairs = pairs[:2]
    rows: list[dict[str, Any]] = []
    seeds = [int(seed) for seed in bundle.config["seeds"]]
    if smoke_test:
        seeds = seeds[:1]
    for seed in seeds:
        encoder_state, encoder_checkpoint = load_encoder_state(bundle, seed)
        embedding = extract_frozen_embeddings(bundle, encoder_state)
        np.savez_compressed(
            bundle.output_paths["embeddings"] / f"E2_embeddings_seed{seed}.npz",
            X=embedding,
            node_id=bundle.website_frame["node_id"].to_numpy(),
        )
        for source_region, target_region in pairs:
            source = task_frame[task_frame["jurisdiction"] == source_region]
            target = task_frame[task_frame["jurisdiction"] == target_region]
            source_idx = source["graph_node_index"].to_numpy(dtype=int)
            target_idx = target["graph_node_index"].to_numpy(dtype=int)
            x_s = embedding[source_idx]
            y_s = source["label"].to_numpy(dtype=int)
            x_t = embedding[target_idx]
            y_t = target["label"].to_numpy(dtype=int)
            logme_source = compute_logme(x_s, y_s)
            logme_cross = compute_logme(np.vstack([x_s, x_t]), np.concatenate([y_s, y_t]))
            h_div = compute_h_divergence(x_s, x_t, seed=seed)
            wasserstein = compute_sliced_wasserstein(x_s, x_t, seed=seed)
            rows.append(
                {
                    "seed": seed,
                    "source_region": source_region,
                    "target_region": target_region,
                    "source_size": int(len(source)),
                    "target_size": int(len(target)),
                    "logme": logme_source,
                    "logme_cross": logme_cross,
                    "h_divergence": h_div,
                    "wasserstein": wasserstein,
                    "encoder_checkpoint": encoder_checkpoint,
                }
            )
    return pd.DataFrame(rows)


def _safe_slug(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in str(value)).strip("_")


def build_single_source_transfer_split(
    bundle: Any,
    source_region: str,
    target_region: str,
    seed: int,
) -> pd.DataFrame:
    primary = build_primary_task_frame(bundle)
    source = primary[primary["jurisdiction"] == source_region].copy().reset_index(drop=True)
    target = primary[primary["jurisdiction"] == target_region].copy().reset_index(drop=True)
    if source["label"].nunique() < 2 or target["label"].nunique() < 2:
        raise ValueError(f"{source_region}->{target_region} is not a complete binary transfer pair")
    split_config = bundle.config["splits"]["transfer"]
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
    split_frame["transfer_id"] = f"E2_{_safe_slug(source_region)}_to_{_safe_slug(target_region)}"
    split_frame["seed"] = int(seed)
    split_frame["source_region"] = source_region
    split_frame["target_region"] = target_region
    split_frame["target_labels_used_for_training"] = False
    split_frame["primary_metric_configured"] = "roc_auc"
    return split_frame


def run_extra_transfer_alignment(bundle: Any, pair_scores: pd.DataFrame, smoke_test: bool = False) -> pd.DataFrame:
    if pair_scores.empty:
        return pd.DataFrame()
    pairs = (
        pair_scores[["source_region", "target_region"]]
        .drop_duplicates()
        .sort_values(["source_region", "target_region"])
        .itertuples(index=False, name=None)
    )
    pair_list = list(pairs)
    if smoke_test:
        pair_list = pair_list[:1]
    seeds = [int(seed) for seed in bundle.config["seeds"]]
    if smoke_test:
        seeds = seeds[:1]
    rows: list[dict[str, Any]] = []
    existing_path = bundle.output_paths["metrics"] / "E2_realized_transfer_auc.csv"
    if existing_path.exists() and not smoke_test:
        existing = pd.read_csv(existing_path)
        done = set(
            zip(
                existing["source_region"].astype(str),
                existing["target_region"].astype(str),
                existing["seed"].astype(int),
            )
        )
    else:
        existing = pd.DataFrame()
        done = set()
    for seed in seeds:
        encoder_state, encoder_checkpoint = load_encoder_state(bundle, seed)
        for source_region, target_region in pair_list:
            if (source_region, target_region, seed) in done:
                continue
            transfer_id = f"E2_{_safe_slug(source_region)}_to_{_safe_slug(target_region)}"
            bundle.config["transfer_pairs"][transfer_id] = {
                "source_regions": [source_region],
                "target_regions": [target_region],
                "primary_metric": "roc_auc",
                "primary_metric_direction": "higher_is_better",
                "incomplete_target_policy": "standard_binary_eval",
            }
            split_frame = build_single_source_transfer_split(bundle, source_region, target_region, seed)
            train_info, history, probabilities = train_heco_transfer_model(
                bundle=bundle,
                graph_data=bundle.graph_data.cpu(),
                split_frame=split_frame,
                encoder_state=encoder_state,
                seed=seed,
                method="source_only",
                smoke_test=smoke_test,
                purpose="E2_transfer_alignment",
            )
            metrics, predictions = summarize_run_metrics(
                bundle=bundle,
                split_frame=split_frame,
                probabilities_all=probabilities,
                method="source_only",
                seed=seed,
                transfer_id=transfer_id,
                train_info=train_info,
                domain_acc_final=float("nan"),
                pseudo_label_quality=float("nan"),
            )
            metrics.update(
                {
                    "experiment": "E2_transfer_alignment",
                    "source_region": source_region,
                    "target_region": target_region,
                    "target_auc": metrics["target_test_auc"],
                    "encoder_checkpoint": encoder_checkpoint,
                }
            )
            suffix = f"E2_{_safe_slug(source_region)}_to_{_safe_slug(target_region)}__seed{seed}"
            save_dataframe(split_frame, bundle.output_paths["splits"] / f"{suffix}_split.csv")
            save_dataframe(history, bundle.output_paths["logs"] / f"{suffix}_history.csv")
            save_dataframe(predictions, bundle.output_paths["predictions"] / f"{suffix}_predictions.csv")
            manifest = make_common_manifest(
                bundle=bundle,
                task=transfer_id,
                model="source_only",
                split_name=f"{suffix}_split.csv",
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
    return combined


def correlate_with_realized_auc(pair_scores: pd.DataFrame, realized: pd.DataFrame) -> pd.DataFrame:
    if realized.empty:
        return pd.DataFrame()
    aligned = pair_scores.merge(realized, on=["source_region", "target_region", "seed"], how="inner")
    if aligned.empty:
        return pd.DataFrame()
    metrics = [
        ("logme_cross", False),
        ("h_divergence", True),
        ("wasserstein", True),
    ]
    rows = []
    for metric, invert in metrics:
        values = -aligned[metric] if invert else aligned[metric]
        rho, rho_p = spearmanr(values, aligned["target_auc"], nan_policy="omit")
        tau, tau_p = kendalltau(values, aligned["target_auc"], nan_policy="omit")
        rows.append(
            {
                "metric": metric,
                "aligned_pairs": int(len(aligned)),
                "spearman_rho": float(rho),
                "spearman_p": float(rho_p),
                "kendall_tau": float(tau),
                "kendall_p": float(tau_p),
            }
        )
    return pd.DataFrame(rows)


def plot_logme_vs_auc(aligned: pd.DataFrame, output_path: Path) -> None:
    if aligned.empty or "target_auc" not in aligned.columns:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(6, 5))
    axis.scatter(aligned["logme_cross"], aligned["target_auc"], color="#2563eb", alpha=0.75)
    axis.set_xlabel("LogME cross score")
    axis.set_ylabel("Realized target AUC")
    axis.set_title("Week 8 E2 LogME vs Realized Transfer")
    axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Week 8 E2 transferability scoring.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "week8.yaml"))
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--skip-extra-transfers", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle = load_graph_bundle(args.config)
    if args.smoke_test:
        from src.train.multi_scenario_eval import redirect_output_root

        redirect_output_root(bundle, "week8_smoke")
    pair_scores = compute_pair_scores(bundle, smoke_test=args.smoke_test)
    save_dataframe(pair_scores, bundle.output_paths["metrics"] / "E2_pair_scores.csv")
    realized = pd.DataFrame()
    if not args.skip_extra_transfers:
        realized = run_extra_transfer_alignment(bundle, pair_scores, smoke_test=args.smoke_test)
    save_dataframe(realized, bundle.output_paths["metrics"] / "E2_realized_transfer_auc.csv")
    aligned = pair_scores.merge(
        realized[["source_region", "target_region", "seed", "target_auc"]] if not realized.empty else realized,
        on=["source_region", "target_region", "seed"],
        how="inner",
    ) if not realized.empty else pd.DataFrame()
    save_dataframe(aligned, bundle.output_paths["metrics"] / "E2_pair_scores_aligned.csv")
    metrics = correlate_with_realized_auc(pair_scores, realized)
    save_dataframe(metrics, bundle.output_paths["metrics"] / "E2_transferability_metrics.csv")
    plot_logme_vs_auc(aligned, bundle.output_paths["plots"] / "E2_logme_vs_auc.png")


if __name__ == "__main__":
    main()
