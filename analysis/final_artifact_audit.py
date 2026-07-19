"""Audit released artifacts: row counts, post-hoc flags, and headline metrics."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "final_artifact_audit.csv"


def rel(path: str) -> Path:
    """Rel."""
    return ROOT / path


def read_csv(path: str) -> pd.DataFrame | None:
    """Read CSV."""
    full = rel(path)
    if not full.exists():
        return None
    return pd.read_csv(full)


def add(rows: list[dict[str, Any]], check: str, status: str, observed: Any, expected: Any, detail: str = "") -> None:
    """Add."""
    rows.append(
        {
            "check": check,
            "status": status,
            "observed": observed,
            "expected": expected,
            "detail": detail,
        }
    )


def check_csv_rows(rows: list[dict[str, Any]], path: str, expected_rows: int) -> None:
    """Validate CSV rows."""
    frame = read_csv(path)
    if frame is None:
        add(rows, f"{path}:exists", "fail", "missing", "present")
        return
    observed = len(frame)
    status = "pass" if observed == expected_rows else "fail"
    add(rows, f"{path}:row_count", status, observed, expected_rows)


def check_dir_file_count(rows: list[dict[str, Any]], path: str, expected_files: int, pattern: str = "*") -> None:
    """Validate directory file count."""
    full = rel(path)
    if not full.exists():
        add(rows, f"{path}:exists", "fail", "missing", "present")
        return
    observed = len([item for item in full.glob(pattern) if item.is_file()])
    status = "pass" if observed == expected_files else "fail"
    detail = f"pattern={pattern}" if pattern != "*" else ""
    add(rows, f"{path}:file_count", status, observed, expected_files, detail)


def check_posthoc(
    rows: list[dict[str, Any]],
    path: str,
    should_all_true: bool = True,
    column: str = "is_posthoc",
) -> None:
    """Validate post-hoc."""
    frame = read_csv(path)
    if frame is None:
        add(rows, f"{path}:posthoc_column", "fail", "missing_file", f"{column} column")
        return
    if column not in frame.columns:
        add(rows, f"{path}:posthoc_column", "fail", "missing_column", f"{column} column")
        return
    values = frame[column].astype(str).str.lower()
    if should_all_true:
        ok = values.eq("true").all()
        observed = f"{int(values.eq('true').sum())}/{len(values)} true"
        add(rows, f"{path}:{column}_true", "pass" if ok else "fail", observed, "all true")
    else:
        observed = ",".join(sorted(values.unique()))
        add(rows, f"{path}:{column}_present", "pass", observed, "column present")


def check_column_all_value(
    rows: list[dict[str, Any]],
    path: str,
    column: str,
    expected_value: str,
) -> None:
    """Validate column all value."""
    frame = read_csv(path)
    if frame is None:
        add(rows, f"{path}:{column}", "fail", "missing_file", expected_value)
        return
    if column not in frame.columns:
        add(rows, f"{path}:{column}", "fail", "missing_column", expected_value)
        return
    values = frame[column].astype(str).str.lower()
    expected = str(expected_value).lower()
    ok = values.eq(expected).all()
    observed = f"{int(values.eq(expected).sum())}/{len(values)} {expected}"
    add(rows, f"{path}:{column}_all_{expected}", "pass" if ok else "fail", observed, f"all {expected}")


def check_legacy_marker(rows: list[dict[str, Any]], path: str) -> None:
    """Validate legacy marker."""
    frame = read_csv(path)
    if frame is None:
        add(rows, f"{path}:legacy_marker", "fail", "missing_file", "legacy marker")
        return
    required = {"week9_result_status", "superseded_by", "paper_use"}
    if not required.issubset(frame.columns):
        add(rows, f"{path}:legacy_marker", "fail", "missing_column", ",".join(sorted(required)))
        return
    statuses = frame["week9_result_status"].astype(str)
    paper_use = frame["paper_use"].astype(str)
    ok = statuses.eq("legacy_preformal").all() and paper_use.eq("legacy_only_do_not_use_as_authoritative").all()
    observed = f"legacy={int(statuses.eq('legacy_preformal').sum())}/{len(frame)}"
    add(rows, f"{path}:legacy_marker", "pass" if ok else "fail", observed, "all legacy_preformal")


def check_claim_columns(rows: list[dict[str, Any]], path: str) -> None:
    """Validate claim columns."""
    frame = read_csv(path)
    required = {"status", "is_posthoc", "paper_safe_claim", "paper_forbidden_claim"}
    if frame is None:
        add(rows, f"{path}:claim_columns", "fail", "missing_file", ",".join(sorted(required)))
        return
    if not required.issubset(frame.columns):
        add(rows, f"{path}:claim_columns", "fail", ",".join(frame.columns), ",".join(sorted(required)))
        return
    nonempty = frame[list(required)].astype(str).apply(lambda col: col.str.len().gt(0)).all().all()
    add(rows, f"{path}:claim_columns", "pass" if bool(nonempty) else "fail", f"rows={len(frame)}", "required columns non-empty")


def first_value(frame: pd.DataFrame, mask: pd.Series, column: str) -> float | None:
    """First value."""
    subset = frame[mask]
    if subset.empty or column not in subset.columns:
        return None
    try:
        return float(subset[column].iloc[0])
    except Exception:
        return None


def check_metric_close(
    rows: list[dict[str, Any]],
    path: str,
    check: str,
    mask_column: str,
    mask_value: str,
    value_column: str,
    expected: float,
    tolerance: float = 1.0e-4,
) -> None:
    """Validate metric close."""
    frame = read_csv(path)
    if frame is None:
        add(rows, check, "fail", "missing_file", expected, path)
        return
    if mask_column not in frame.columns or value_column not in frame.columns:
        add(rows, check, "fail", "missing_column", expected, f"{mask_column}/{value_column}")
        return
    value = first_value(frame, frame[mask_column].astype(str) == mask_value, value_column)
    if value is None:
        add(rows, check, "fail", "missing_metric", expected, f"{mask_column}={mask_value}")
        return
    status = "pass" if abs(value - expected) <= tolerance else "fail"
    add(rows, check, status, f"{value:.6f}", f"{expected:.6f}", path)


def check_status_counts(rows: list[dict[str, Any]]) -> None:
    """Validate status counts."""
    frame = read_csv("output/week8/metrics/week8_acceptance_audit.csv")
    if frame is None or "status" not in frame.columns:
        add(rows, "week8_acceptance_status_counts", "fail", "missing", "5 pass / 10 fail")
        return
    counts = frame["status"].value_counts().to_dict()
    observed = f"pass={counts.get('pass', 0)}; fail={counts.get('fail', 0)}"
    status = "pass" if counts.get("pass", 0) == 5 and counts.get("fail", 0) == 10 else "fail"
    add(rows, "week8_acceptance_status_counts", status, observed, "pass=5; fail=10")


def check_final_rollup_posthoc(rows: list[dict[str, Any]]) -> None:
    """Validate final rollup post-hoc."""
    frame = read_csv("output/week10/metrics/final_hypothesis_rollup.csv")
    if frame is None:
        add(rows, "final_rollup_posthoc_marking", "fail", "missing_file", "posthoc rows marked")
        return
    required = {"evidence_layer", "patch", "is_posthoc"}
    if not required.issubset(frame.columns):
        add(rows, "final_rollup_posthoc_marking", "fail", "missing_column", ",".join(sorted(required)))
        return
    is_posthoc = frame["is_posthoc"].astype(str).str.lower().eq("true")
    ok = bool(is_posthoc.all())
    detail = f"posthoc_true={int(is_posthoc.sum())}; total={len(frame)}"
    add(rows, "final_rollup_posthoc_marking", "pass" if ok else "fail", detail, "all final-rollup rows marked posthoc")


def main() -> None:
    """Command-line entry point."""
    rows: list[dict[str, Any]] = []
    expected_csvs = {
        "output/week7/metrics/transfer_summary.csv": 80,
        "output/week7/metrics/week7_acceptance_audit.csv": 4,
        "output/week8/metrics/week8_acceptance_audit.csv": 15,
        "output/week8/metrics/E1_edge_ablation_raw_runs.csv": 120,
        "output/week8/metrics/E2_pair_scores.csv": 30,
        "output/week8/metrics/E2_realized_transfer_auc.csv": 30,
        "output/week8/metrics/E3_lofo_family_sensitivity_raw.csv": 35,
        "output/week8/audits/E3_permutation_aucs.csv": 500,
        "output/week8/metrics/E4_three_scenario_raw_runs.csv": 15,
        "output/week8/metrics/E5_structure_vs_lexical.csv": 100,
        "output/week85/metrics/E3_w85_lofo_raw.csv": 70,
        "output/week85/metrics/E2_w85_expanded_pair_scores.csv": 150,
        "output/week85/metrics/week85_acceptance_audit.csv": 12,
        "output/week8_patch/metrics/E5_structure_vs_lexical.csv": 180,
        "output/week8_patch/metrics/E5_corrected_acceptance_audit.csv": 6,
        "output/week6_head_ablation/metrics/head_ablation_raw_runs.csv": 30,
        "output/week6_head_ablation/metrics/head_ablation_summary.csv": 6,
        "output/week9/metrics/fewshot_raw_runs.csv": 30,
        "output/week9/metrics/fewshot_summary.csv": 6,
        "output/week9/metrics/fewshot_acceptance_audit.csv": 2,
        "output/week9/metrics/W9_E1_fewshot_curve.csv": 150,
        "output/week9/metrics/W9_E2_fewshot_seed_stability.csv": 30,
        "output/week9/metrics/W9_E3_shortcut_aware_fewshot.csv": 150,
        "output/week9/metrics/W9_E3_shortcut_aware_summary.csv": 30,
        "output/week9/metrics/week9_fewshot_rollup.csv": 2,
        "output/week9/metrics/fewshot_acceptance_audit_v2.csv": 2,
        "output/week9/metrics/hard_negative_family_raw.csv": 35,
        "output/week9/metrics/hard_negative_family_acceptance.csv": 2,
        "output/week10/metrics/final_rq_evidence_table.csv": 21,
        "output/week10/metrics/final_hypothesis_rollup.csv": 31,
        "output/week10/metrics/feature_bucket_transfer_summary.csv": 28,
        "output/week10/metrics/table6_ablation_all.csv": 101,
        "output/week10/metrics/fewshot_final_summary.csv": 30,
        "output/week10/metrics/hard_negative_family_final_summary.csv": 7,
        "output/week10/metrics/deviation_case_index.csv": 3,
        "output/week10/metrics/week10_acceptance_audit.csv": 31,
        "output/week10/audits/posthoc_flag_audit.csv": 5,
        "output/week10/audits/metric_consistency_audit.csv": 3,
        "output/week10/audits/table_figure_crosswalk.csv": 9,
        "output/week10/tables/table_rq1_main_results.csv": 3,
        "output/week10/tables/table_rq2_regional_heterogeneity.csv": 11,
        "output/week10/tables/table_rq3_transfer_boundary.csv": 5,
        "output/week10/tables/table_rq4_fewshot.csv": 30,
        "output/week10/tables/table_ablation_all.csv": 101,
        "output/week10/tables/table_family_boundary.csv": 7,
        "output/week7/metrics/week7_transfer_summary.csv": 4,
        "output/week7/metrics/week7_transfer_taxonomy.csv": 4,
        "output/week7/metrics/week7_to_week8_bridge.csv": 4,
        "output/week8/metrics/E3_lofo_setting_comparison.csv": 2,
        "output/week8/metrics/E3_hard_negative_family_summary.csv": 7,
        "output/week8/metrics/E5_structure_vs_lexical_corrected_summary.csv": 36,
        "output/week8/metrics/week8_diagnostic_rollup.csv": 5,
        "output/week11/metrics/P1_family_metric_lofo.csv": 35,
        "output/week11/metrics/P1_alpha_ablation.csv": 40,
        "output/week11/metrics/P1_family_centroid_distance.csv": 105,
        "output/week11/metrics/P2_method_comparison.csv": 125,
        "output/week11/metrics/P2_eerm_K_sensitivity.csv": 4,
        "output/week11/metrics/P3_full_fewshot_matrix.csv": 100,
        "output/week11/metrics/P4_w7_method_pairwise.csv": 24,
        "output/week11/metrics/P4_w9_fewshot_significance.csv": 8,
        "output/week11/metrics/P4_w11_overall_significance.csv": 5,
        "output/week11/metrics/week11_acceptance_audit.csv": 15,
        "output/week11/metrics/week11_score_estimate.csv": 1,
        "output/week11/tables/table_week11_hypothesis_audit.csv": 15,
        "output/week11/tables/table_week11_patch_summary.csv": 5,
    }
    for path, expected_rows in expected_csvs.items():
        check_csv_rows(rows, path, expected_rows)

    expected_dirs = [
        ("output/week7/runs", 80, "*__seed*.json"),
        ("output/week7/logs", 80, "*"),
        ("output/week8/runs", 800, "*"),
        ("output/week8/logs", 800, "*"),
        ("output/week85/runs", 70, "*"),
        ("output/week85/predictions", 70, "*"),
        ("output/week8_patch/runs", 180, "*"),
        ("output/week8_patch/logs", 180, "*"),
        ("output/week8_patch/predictions", 180, "*"),
        ("output/week6_head_ablation/runs", 30, "*"),
        ("output/week9/runs", 30, "fewshot_*.json"),
        ("output/week9/logs", 30, "fewshot_*_history.csv"),
        ("output/week9/runs", 120, "W9_*.json"),
        ("output/week9/logs", 120, "W9_*_history.csv"),
        ("output/week9/predictions", 120, "W9_*_predictions.csv"),
        ("output/week9/splits", 150, "W9_*_split.csv"),
        ("output/week10/figures", 4, "fig*.pdf"),
        ("output/week11/logs", 160, "*_history.csv"),
        ("output/week11/runs", 160, "*.json"),
        ("output/week11/plots", 4, "P*.png"),
        ("output/week11/audits", 40, "P3_*_split.csv"),
    ]
    for path, expected_files, pattern in expected_dirs:
        check_dir_file_count(rows, path, expected_files, pattern)

    check_posthoc(rows, "output/week8_patch/metrics/E5_corrected_acceptance_audit.csv")
    check_posthoc(rows, "output/week9/metrics/fewshot_acceptance_audit.csv")
    check_posthoc(rows, "output/week9/metrics/W9_E1_fewshot_curve.csv")
    check_posthoc(rows, "output/week9/metrics/W9_E2_fewshot_seed_stability.csv")
    check_posthoc(rows, "output/week9/metrics/W9_E3_shortcut_aware_summary.csv")
    check_posthoc(rows, "output/week9/metrics/week9_fewshot_rollup.csv")
    check_posthoc(rows, "output/week9/metrics/fewshot_acceptance_audit_v2.csv")
    check_legacy_marker(rows, "output/week9/metrics/fewshot_raw_runs.csv")
    check_legacy_marker(rows, "output/week9/metrics/fewshot_summary.csv")
    check_legacy_marker(rows, "output/week9/metrics/fewshot_acceptance_audit.csv")
    check_posthoc(rows, "output/week9/metrics/hard_negative_family_acceptance.csv")
    check_posthoc(rows, "output/week10/metrics/fewshot_final_summary.csv")
    check_posthoc(rows, "output/week10/metrics/hard_negative_family_final_summary.csv")
    check_posthoc(rows, "output/week10/metrics/deviation_case_index.csv")
    check_posthoc(rows, "output/week10/metrics/final_rq_evidence_table.csv", should_all_true=False)
    check_posthoc(rows, "output/week10/metrics/final_hypothesis_rollup.csv", should_all_true=False)
    check_claim_columns(rows, "output/week10/metrics/final_rq_evidence_table.csv")
    check_claim_columns(rows, "output/week10/metrics/final_hypothesis_rollup.csv")
    check_claim_columns(rows, "output/week10/metrics/table6_ablation_all.csv")
    check_claim_columns(rows, "output/week11/metrics/week11_acceptance_audit.csv")
    check_column_all_value(rows, "output/week11/metrics/P1_family_metric_lofo.csv", "is_prerun_frozen", "true")
    check_column_all_value(rows, "output/week11/metrics/P1_family_metric_lofo.csv", "is_posthoc", "false")
    check_column_all_value(rows, "output/week11/metrics/P2_eerm_K_sensitivity.csv", "is_prerun_frozen", "true")
    check_column_all_value(rows, "output/week11/metrics/P3_full_fewshot_matrix.csv", "feature_condition", "full")
    check_column_all_value(rows, "output/week11/metrics/week11_acceptance_audit.csv", "is_prerun_frozen", "true")
    check_column_all_value(rows, "output/week11/metrics/week11_acceptance_audit.csv", "is_posthoc", "false")
    check_posthoc(rows, "output/week7/metrics/week7_transfer_summary.csv", column="is_posthoc_diagnostic")
    check_posthoc(rows, "output/week7/metrics/week7_transfer_taxonomy.csv", column="is_posthoc_diagnostic")
    check_posthoc(rows, "output/week7/metrics/week7_to_week8_bridge.csv", column="is_posthoc_diagnostic")
    check_posthoc(rows, "output/week8/metrics/E3_lofo_setting_comparison.csv", column="is_posthoc_diagnostic")
    check_posthoc(rows, "output/week8/metrics/E3_hard_negative_family_summary.csv", column="is_posthoc_diagnostic")
    check_posthoc(rows, "output/week8/metrics/E5_structure_vs_lexical_corrected_summary.csv", column="is_posthoc_diagnostic")
    check_posthoc(rows, "output/week8/metrics/week8_diagnostic_rollup.csv", column="is_posthoc_diagnostic")
    check_final_rollup_posthoc(rows)
    check_status_counts(rows)

    check_metric_close(
        rows,
        "output/week8_patch/metrics/E5_corrected_acceptance_audit.csv",
        "corrected_e5_lex_only_minus_graph_only",
        "hypothesis",
        "W8P_E5_lex_only_vs_graph_only",
        "observed_value",
        0.2395692007797271,
    )
    check_metric_close(
        rows,
        "output/week6_head_ablation/metrics/head_ablation_summary.csv",
        "head_ablation_fixed_mean_auc",
        "head_type",
        "fixed_mean",
        "roc_auc_mean",
        0.9735092576177454,
    )
    check_metric_close(
        rows,
        "output/week9/metrics/fewshot_acceptance_audit.csv",
        "fewshot_t2ph_delta",
        "transfer_id",
        "T2_PH",
        "delta_vs_source_only",
        0.28333333333333344,
    )
    check_metric_close(
        rows,
        "output/week9/metrics/week9_fewshot_rollup.csv",
        "week9_formal_t2ph_delta",
        "target_id",
        "T2_PH",
        "best_mean_delta_auc",
        0.3125,
    )
    check_metric_close(
        rows,
        "output/week9/metrics/week9_fewshot_rollup.csv",
        "week9_formal_t3_delta",
        "target_id",
        "T3_DiagnoseFrance",
        "best_mean_delta_auc",
        0.009641975308641876,
    )
    check_metric_close(
        rows,
        "output/week9/metrics/hard_negative_family_acceptance.csv",
        "hard_negative_family_mean_auc",
        "hypothesis",
        "hard_negative_family_mean_auc",
        "observed_value",
        0.621849647266314,
    )
    check_metric_close(
        rows,
        "output/week9/metrics/hard_negative_family_acceptance.csv",
        "hard_negative_tsars_auc",
        "hypothesis",
        "hard_negative_tsars_auc",
        "observed_value",
        0.3864197530864197,
    )

    audit = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(OUT, index=False, encoding="utf-8")
    print(audit["status"].value_counts().to_string())
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
