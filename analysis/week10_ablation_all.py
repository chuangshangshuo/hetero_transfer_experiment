"""Week-10 consolidated ablation table (table6) builder."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.week10_common import DEFAULT_CONFIG, ensure_output_dirs, input_path, load_config, output_path, read_input, save_csv


def add_representation_rows(config: dict, rows: list[dict]) -> None:
    """Add representation rows."""
    w4 = read_input(config, "week4", "pooled_primary_summary")
    for _, row in w4.iterrows():
        rows.append(
            {
                "ablation_group": "representation_learning",
                "ablation_item": row["model"],
                "target_or_scope": "pooled_primary",
                "metric": "roc_auc_mean",
                "value": float(row["roc_auc_mean"]),
                "std": float(row["roc_auc_std"]),
                "status": "baseline_context",
                "is_posthoc": False,
                "paper_safe_claim": "Representation baselines provide context for model integration.",
                "paper_forbidden_claim": "Do not claim one model universally dominates from pooled results alone.",
                "source_file": str(input_path(config, "week4", "pooled_primary_summary").relative_to(ROOT)),
            }
        )
    w5 = read_input(config, "week5", "heco_probe_summary")
    for _, row in w5.iterrows():
        rows.append(
            {
                "ablation_group": "representation_learning",
                "ablation_item": row["model"],
                "target_or_scope": "heco_probe",
                "metric": "roc_auc_mean",
                "value": float(row["roc_auc_mean"]),
                "std": float(row["roc_auc_std"]),
                "status": "pretraining_context",
                "is_posthoc": False,
                "paper_safe_claim": "HeCo probing/fine-tuning is useful but must be compared against split variance.",
                "paper_forbidden_claim": "Do not claim HeCo universally outperforms HeteroGNN.",
                "source_file": str(input_path(config, "week5", "heco_probe_summary").relative_to(ROOT)),
            }
        )
    head = read_input(config, "week6", "head_ablation_summary")
    for _, row in head.iterrows():
        rows.append(
            {
                "ablation_group": "representation_learning",
                "ablation_item": f"head_{row['head_type']}",
                "target_or_scope": "pooled_primary",
                "metric": "roc_auc_mean",
                "value": float(row["roc_auc_mean"]),
                "std": float(row["roc_auc_std"]),
                "status": "head_ablation_context",
                "is_posthoc": True,
                "paper_safe_claim": "Head choice affects stability and should be reported as an integration boundary.",
                "paper_forbidden_claim": "Do not claim a head ablation proves general model superiority.",
                "source_file": str(input_path(config, "week6", "head_ablation_summary").relative_to(ROOT)),
            }
        )


def add_transfer_rows(config: dict, rows: list[dict]) -> None:
    """Add transfer rows."""
    transfer = read_input(config, "week7", "transfer_method_summary")
    for _, row in transfer.iterrows():
        rows.append(
            {
                "ablation_group": "transfer_method",
                "ablation_item": row["method"],
                "target_or_scope": row["transfer_id"],
                "metric": "primary_metric_value_mean",
                "value": float(row["primary_metric_value_mean"]),
                "std": float(row["primary_metric_value_std"]),
                "status": "target_dependent_transfer",
                "is_posthoc": True,
                "paper_safe_claim": "Transfer methods are target-dependent and must be read against source-only.",
                "paper_forbidden_claim": "Do not claim DANN/StruRW solve transfer.",
                "source_file": str(input_path(config, "week7", "transfer_method_summary").relative_to(ROOT)),
            }
        )
    rows.append(
        {
            "ablation_group": "transfer_method",
            "ablation_item": "IRM",
            "target_or_scope": "not_available",
            "metric": "not_run",
            "value": float("nan"),
            "std": float("nan"),
            "status": "not_implemented_not_citable",
            "is_posthoc": True,
            "paper_safe_claim": "IRM is not available as current evidence.",
            "paper_forbidden_claim": "Do not cite IRM as implemented or validated.",
            "source_file": "src/transfer/irm_penalty.py",
        }
    )


def add_feature_rows(config: dict, rows: list[dict]) -> None:
    """Add feature rows."""
    feature = read_input(config, "week8_patch", "corrected_e5_summary")
    for _, row in feature.iterrows():
        rows.append(
            {
                "ablation_group": "feature_bucket",
                "ablation_item": row["config_id"],
                "target_or_scope": row["transfer_id"],
                "metric": "primary_metric_value_mean",
                "value": float(row["primary_metric_value_mean"]),
                "std": float(row["primary_metric_value_std"]),
                "status": "feature_bucket_diagnostic",
                "is_posthoc": True,
                "paper_safe_claim": "Feature bucket ablation diagnoses shortcut and structure boundaries.",
                "paper_forbidden_claim": "Do not infer pure structural transfer from full-feature AUC alone.",
                "source_file": str(input_path(config, "week8_patch", "corrected_e5_summary").relative_to(ROOT)),
            }
        )


def add_fewshot_rows(config: dict, rows: list[dict]) -> None:
    """Add few-shot rows."""
    few = read_input(config, "week9", "fewshot_stability")
    for _, row in few.iterrows():
        rows.append(
            {
                "ablation_group": "few_shot",
                "ablation_item": f"{int(row['shot'])}-shot/{row['feature_condition']}",
                "target_or_scope": row["target_id"],
                "metric": "mean_auc",
                "value": float(row["mean_auc"]),
                "std": float(row["std_auc"]),
                "status": row["stability_label"],
                "is_posthoc": True,
                "paper_safe_claim": "Few-shot calibration is target-dependent; T2_PH improves, saturated T3 has little gain.",
                "paper_forbidden_claim": "Do not claim few-shot universally improves transfer.",
                "source_file": str(input_path(config, "week9", "fewshot_stability").relative_to(ROOT)),
            }
        )


def add_family_rows(config: dict, rows: list[dict]) -> None:
    """Add family rows."""
    hard = read_input(config, "week9", "hard_negative_summary")
    for _, row in hard.iterrows():
        rows.append(
            {
                "ablation_group": "family_hard_negative",
                "ablation_item": row["family_id"],
                "target_or_scope": "Denmark_illegal_family",
                "metric": "hard_negative_auc_mean",
                "value": float(row["hard_negative_auc_mean"]),
                "std": float(row["hard_negative_auc_std"]),
                "status": "hard_negative_pass" if float(row["hard_negative_auc_mean"]) >= 0.80 else "hard_negative_fail",
                "is_posthoc": True,
                "paper_safe_claim": "Hard-negative family audit bounds illegal-family generalization.",
                "paper_forbidden_claim": "Do not claim illegal-family attribution success.",
                "source_file": str(input_path(config, "week9", "hard_negative_summary").relative_to(ROOT)),
            }
        )


def build_ablation_all(config: dict) -> pd.DataFrame:
    """Build ablation all."""
    rows: list[dict] = []
    add_representation_rows(config, rows)
    add_transfer_rows(config, rows)
    add_feature_rows(config, rows)
    add_fewshot_rows(config, rows)
    add_family_rows(config, rows)
    return pd.DataFrame(rows)


def main() -> None:
    """Command-line entry point."""
    config = load_config(DEFAULT_CONFIG)
    ensure_output_dirs(config)
    frame = build_ablation_all(config)
    save_csv(frame, output_path(config, "metrics", "table6_ablation_all.csv"))
    save_csv(frame, output_path(config, "tables", "table_ablation_all.csv"))
    print(frame.groupby("ablation_group").size().to_string())


if __name__ == "__main__":
    main()
