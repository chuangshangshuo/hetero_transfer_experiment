from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.train.utils import (
    build_primary_task_frame,
    choose_threshold_by_youden,
    compute_binary_metrics,
    load_graph_bundle,
    make_pooled_primary_split,
    metric_columns,
    plot_mean_roc_pr,
    save_dataframe,
    summarise_runs,
)


POOLED_MODELS = ["logistic_regression", "mlp", "hetero_gnn"]
CROSS_VERIFIED_TIER = "illegal_confirmed_official_cross_verified"


def split_audit(split_frame: pd.DataFrame, seed: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for split_name, split_part in split_frame.groupby("split"):
        row = {
            "seed": seed,
            "split": split_name,
            "num_rows": int(len(split_part)),
            "num_groups": int(split_part["split_group"].nunique()),
            "licensed_baseline": int((split_part["sample_tier"] == "licensed_baseline").sum()),
            "illegal_single": int((split_part["sample_tier"] == "illegal_confirmed_official_single").sum()),
            "illegal_cross_verified": int((split_part["sample_tier"] == CROSS_VERIFIED_TIER).sum()),
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
            "seed": seed,
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


def reaggregate_model_seed(
    split_frame: pd.DataFrame,
    prediction_frame: pd.DataFrame,
    model_name: str,
    seed: int,
) -> tuple[dict[str, object], pd.DataFrame]:
    score_columns = ["node_id", "graph_node_index", "split", "pred_prob_illegal"]
    scored = prediction_frame[score_columns].copy()
    scored = scored.rename(columns={"split": "original_prediction_split"})
    scored["graph_node_index"] = scored["graph_node_index"].astype(int)
    merged = split_frame.drop(columns=["pred_prob_illegal"], errors="ignore").merge(
        scored,
        on=["node_id", "graph_node_index"],
        how="left",
        validate="1:1",
    )
    if merged["pred_prob_illegal"].isna().any():
        missing = merged.loc[merged["pred_prob_illegal"].isna(), "node_id"].head(10).tolist()
        raise ValueError(f"Missing prediction scores for {model_name} seed {seed}: {missing}")

    val_frame = merged[merged["split"] == "val"].copy()
    test_frame = merged[merged["split"] == "test"].copy()
    threshold = choose_threshold_by_youden(
        val_frame["label"].to_numpy(dtype=int),
        val_frame["pred_prob_illegal"].to_numpy(dtype=float),
    )
    metrics = compute_binary_metrics(
        test_frame["label"].to_numpy(dtype=int),
        test_frame["pred_prob_illegal"].to_numpy(dtype=float),
        threshold=threshold,
    )
    metrics_row: dict[str, object] = {
        "task": "pooled_primary_group_aware_reaggregated",
        "model": model_name,
        "seed": seed,
        **metrics,
        "val_roc_auc": compute_binary_metrics(
            val_frame["label"].to_numpy(dtype=int),
            val_frame["pred_prob_illegal"].to_numpy(dtype=float),
            threshold=threshold,
        )["roc_auc"],
        "note": "Existing Week 4 predictions reaggregated on regenerated operator_or_case group-aware splits; models were not retrained.",
    }
    merged["pred_label"] = (merged["pred_prob_illegal"] >= threshold).astype(int)
    merged["model"] = model_name
    merged["task"] = "pooled_primary_group_aware_reaggregated"
    merged["seed"] = seed
    return metrics_row, merged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenerate pooled primary group-aware splits and reaggregate existing Week 4 prediction scores."
    )
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "week4_experiments.yaml"),
        help="Path to the Week 4 experiment config.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle = load_graph_bundle(args.config)
    task_frame = build_primary_task_frame(bundle)
    raw_metrics: list[dict[str, object]] = []
    corrected_predictions: list[pd.DataFrame] = []
    audits: list[pd.DataFrame] = []
    overlap_audits: list[pd.DataFrame] = []

    for seed in bundle.config["seeds"]:
        split_frame = make_pooled_primary_split(task_frame, int(seed), bundle.config)
        split_path = bundle.output_paths["splits"] / f"pooled_primary__seed{seed}.csv"
        save_dataframe(split_frame, split_path)
        audits.append(split_audit(split_frame, int(seed)))

        for model_name in POOLED_MODELS:
            prediction_path = bundle.output_paths["predictions"] / f"pooled_primary__{model_name}__seed{seed}.csv"
            prediction_frame = pd.read_csv(prediction_path)
            metrics_row, merged_predictions = reaggregate_model_seed(
                split_frame=split_frame,
                prediction_frame=prediction_frame,
                model_name=model_name,
                seed=int(seed),
            )
            raw_metrics.append(metrics_row)
            corrected_predictions.append(merged_predictions[merged_predictions["split"] == "test"].copy())
            overlap_audit = (
                merged_predictions.groupby(["model", "seed", "split", "original_prediction_split"])
                .size()
                .reset_index(name="num_rows")
            )
            overlap_audits.append(overlap_audit)
            save_dataframe(
                merged_predictions,
                bundle.output_paths["predictions"]
                / f"pooled_primary_group_aware_reaggregated__{model_name}__seed{seed}.csv",
            )

    raw_metrics_frame = pd.DataFrame(raw_metrics).sort_values(["model", "seed"]).reset_index(drop=True)
    summary_frame = summarise_runs(
        raw_metrics_frame,
        ["task", "model"],
        metric_columns(),
    )
    save_dataframe(
        raw_metrics_frame,
        bundle.output_paths["metrics"] / "pooled_primary_group_aware_reaggregated_raw_runs.csv",
    )
    save_dataframe(
        summary_frame,
        bundle.output_paths["metrics"] / "pooled_primary_group_aware_reaggregated_summary.csv",
    )
    save_dataframe(
        pd.concat(audits, ignore_index=True),
        bundle.output_paths["audits"] / "pooled_primary_group_aware_split_audit.csv",
    )
    save_dataframe(
        pd.concat(overlap_audits, ignore_index=True),
        bundle.output_paths["audits"] / "pooled_primary_group_aware_prediction_overlap_audit.csv",
    )
    plot_mean_roc_pr(
        pd.concat(corrected_predictions, ignore_index=True),
        bundle.output_paths["plots"] / "pooled_primary_group_aware_reaggregated_roc_pr.png",
    )


if __name__ == "__main__":
    main()
