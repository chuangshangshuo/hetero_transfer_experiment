"""Week-10 acceptance audit against pre-declared hypotheses."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.week10_common import DEFAULT_CONFIG, ensure_output_dirs, load_config, output_path, read_csv, save_csv


def add(rows: list[dict], check: str, status: str, observed: str, expected: str, detail: str = "") -> None:
    """Add."""
    rows.append({"check": check, "status": status, "observed": observed, "expected": expected, "detail": detail})


def file_exists(rows: list[dict], path: str) -> None:
    """File exists."""
    full = ROOT / path
    add(rows, f"{path}:exists", "pass" if full.exists() else "fail", "present" if full.exists() else "missing", "present")


def posthoc_audit() -> pd.DataFrame:
    """Post-hoc audit."""
    files = [
        "output/week10/metrics/final_rq_evidence_table.csv",
        "output/week10/metrics/final_hypothesis_rollup.csv",
        "output/week10/metrics/fewshot_final_summary.csv",
        "output/week10/metrics/hard_negative_family_final_summary.csv",
        "output/week10/metrics/deviation_case_index.csv",
    ]
    rows = []
    for path in files:
        frame = read_csv(path)
        if frame.empty:
            rows.append({"file": path, "status": "fail", "detail": "missing_or_empty"})
            continue
        if "is_posthoc" not in frame.columns:
            rows.append({"file": path, "status": "fail", "detail": "missing is_posthoc"})
            continue
        values = frame["is_posthoc"].astype(str).str.lower()
        rows.append({"file": path, "status": "pass", "detail": f"posthoc_true={int(values.eq('true').sum())}; rows={len(frame)}"})
    return pd.DataFrame(rows)


def metric_consistency_audit() -> pd.DataFrame:
    """Metric consistency audit."""
    rows = []
    rollup = read_csv("output/week9/metrics/week9_fewshot_rollup.csv")
    acc_v2 = read_csv("output/week9/metrics/fewshot_acceptance_audit_v2.csv")
    few10 = read_csv("output/week10/metrics/fewshot_final_summary.csv")
    if not rollup.empty and not acc_v2.empty:
        t2_roll = float(rollup.loc[rollup["target_id"].eq("T2_PH"), "best_mean_delta_auc"].iloc[0])
        t2_acc = float(acc_v2.loc[acc_v2["target"].eq("T2_PH"), "delta"].iloc[0])
        rows.append({"check": "week9_rollup_vs_acceptance_v2_t2_delta", "status": "pass" if abs(t2_roll - t2_acc) < 1e-9 else "fail", "observed": t2_acc, "expected": t2_roll})
        t3_roll = float(rollup.loc[rollup["target_id"].eq("T3_DiagnoseFrance"), "best_mean_delta_auc"].iloc[0])
        t3_acc = float(acc_v2.loc[acc_v2["target"].eq("T3"), "delta"].iloc[0])
        rows.append({"check": "week9_rollup_vs_acceptance_v2_t3_delta", "status": "pass" if abs(t3_roll - t3_acc) < 1e-9 else "fail", "observed": t3_acc, "expected": t3_roll})
    if not few10.empty:
        source_files = set(few10.get("source_file", pd.Series(dtype=str)).astype(str))
        ok = any("W9_E2_fewshot_seed_stability.csv" in value for value in source_files)
        rows.append({"check": "week10_fewshot_uses_w9_stability", "status": "pass" if ok else "fail", "observed": ",".join(sorted(source_files)), "expected": "W9_E2_fewshot_seed_stability.csv"})
    return pd.DataFrame(rows)


def crosswalk() -> pd.DataFrame:
    """Crosswalk."""
    rows = [
        ("fig7_feature_bucket_transfer.pdf", "feature_bucket_transfer_summary.csv", "RQ2 regional heterogeneity"),
        ("fig8_fewshot_curves.pdf", "fewshot_final_summary.csv", "RQ4 few-shot calibration"),
        ("fig9_deviation_case.pdf", "deviation_case_index.csv", "case-level failure/recovery explanation"),
        ("fig10_hard_negative_family.pdf", "hard_negative_family_final_summary.csv", "family hard-negative boundary"),
        ("table6_ablation_all.csv", "table6_ablation_all.csv", "ablation summary"),
        ("table_rq1_main_results.csv", "final_rq_evidence_table.csv", "RQ1 main results"),
        ("table_rq2_regional_heterogeneity.csv", "feature_bucket_transfer_summary.csv", "RQ2 regional heterogeneity"),
        ("table_rq3_transfer_boundary.csv", "final_rq_evidence_table.csv", "RQ3 transfer boundary"),
        ("table_rq4_fewshot.csv", "fewshot_final_summary.csv", "RQ4 few-shot"),
    ]
    out = []
    for artifact, source, role in rows:
        artifact_path = ROOT / "output" / "week10" / ("figures" if artifact.endswith(".pdf") else "tables") / artifact
        if artifact == "table6_ablation_all.csv":
            artifact_path = ROOT / "output" / "week10" / "metrics" / artifact
        source_path = ROOT / "output" / "week10" / "metrics" / source
        out.append(
            {
                "artifact": str(artifact_path.relative_to(ROOT)),
                "source_table": str(source_path.relative_to(ROOT)),
                "paper_role": role,
                "artifact_exists": artifact_path.exists(),
                "source_exists": source_path.exists(),
                "status": "pass" if artifact_path.exists() and source_path.exists() else "fail",
            }
        )
    return pd.DataFrame(out)


def build_acceptance() -> pd.DataFrame:
    """Build acceptance."""
    rows: list[dict] = []
    required = [
        "output/week10/metrics/final_rq_evidence_table.csv",
        "output/week10/metrics/final_hypothesis_rollup.csv",
        "output/week10/metrics/feature_bucket_transfer_summary.csv",
        "output/week10/metrics/table6_ablation_all.csv",
        "output/week10/metrics/fewshot_final_summary.csv",
        "output/week10/metrics/hard_negative_family_final_summary.csv",
        "output/week10/metrics/deviation_case_index.csv",
        "output/week10/figures/fig7_feature_bucket_transfer.pdf",
        "output/week10/figures/fig8_fewshot_curves.pdf",
        "output/week10/figures/fig9_deviation_case.pdf",
        "output/week10/figures/fig10_hard_negative_family.pdf",
    ]
    for path in required:
        file_exists(rows, path)
    rq = read_csv("output/week10/metrics/final_rq_evidence_table.csv")
    if not rq.empty:
        required_cols = {"status", "is_posthoc", "paper_safe_claim", "paper_forbidden_claim"}
        add(rows, "final_rq_required_claim_columns", "pass" if required_cols.issubset(rq.columns) else "fail", ",".join(rq.columns), ",".join(sorted(required_cols)))
        forbidden_nonempty = rq["paper_forbidden_claim"].astype(str).str.len().gt(0).all() if "paper_forbidden_claim" in rq.columns else False
        add(rows, "forbidden_claims_nonempty", "pass" if forbidden_nonempty else "fail", str(forbidden_nonempty), "true")
    rollup = read_csv("output/week9/metrics/week9_fewshot_rollup.csv")
    acc_v2 = read_csv("output/week9/metrics/fewshot_acceptance_audit_v2.csv")
    add(rows, "week9_rollup_authoritative_available", "pass" if not rollup.empty and not acc_v2.empty else "fail", f"rollup_rows={len(rollup)}; acc_v2_rows={len(acc_v2)}", "non-empty")
    return pd.DataFrame(rows)


def main() -> None:
    """Command-line entry point."""
    config = load_config(DEFAULT_CONFIG)
    ensure_output_dirs(config)
    posthoc = posthoc_audit()
    metric = metric_consistency_audit()
    cw = crosswalk()
    acceptance = build_acceptance()
    save_csv(posthoc, output_path(config, "audits", "posthoc_flag_audit.csv"))
    save_csv(metric, output_path(config, "audits", "metric_consistency_audit.csv"))
    save_csv(cw, output_path(config, "audits", "table_figure_crosswalk.csv"))
    combined = pd.concat(
        [
            acceptance,
            posthoc.rename(columns={"file": "check", "detail": "observed"}).assign(expected="posthoc column present"),
            metric.rename(columns={"check": "check", "observed": "observed", "expected": "expected"}),
            cw.rename(columns={"artifact": "check"}).assign(observed=lambda x: x["status"], expected="pass"),
        ],
        ignore_index=True,
        sort=False,
    )
    save_csv(combined, output_path(config, "metrics", "week10_acceptance_audit.csv"))
    print(combined["status"].value_counts().to_string())


if __name__ == "__main__":
    main()
