from __future__ import annotations

import math
import re
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
DESKTOP_ZIP = Path(r"<user_home>\Desktop\files.zip")


def load_config() -> dict[str, Any]:
    return yaml.safe_load((ROOT / "configs" / "week11.yaml").read_text(encoding="utf-8"))


def output_dirs(config: dict[str, Any]) -> dict[str, Path]:
    root = Path(config["workspace_root"])
    paths = {name: root / rel for name, rel in config["output"].items()}
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def copy_predeclared_docs(paths: dict[str, Path]) -> None:
    if not DESKTOP_ZIP.exists():
        return
    with zipfile.ZipFile(DESKTOP_ZIP) as zf:
        for name in zf.namelist():
            text = zf.read(name).decode("utf-8", errors="replace")
            if "predeclared" in name.lower():
                out_name = "predeclared_hypotheses.md"
            elif "验收" in name:
                out_name = "Week11_acceptance_roadmap.md"
            else:
                out_name = "Week11_experiment_design.md"
            (paths["audits"] / out_name).write_text(text, encoding="utf-8")


def add(
    rows: list[dict[str, Any]],
    patch: str,
    hypothesis: str,
    observed: Any,
    threshold: Any,
    status: str,
    safe: str,
    forbidden: str,
    metric: str = "",
) -> None:
    rows.append(
        {
            "patch": patch,
            "hypothesis": hypothesis,
            "metric": metric,
            "observed_value": observed,
            "threshold": threshold,
            "status": status,
            "is_prerun_frozen": True,
            "is_posthoc": False,
            "paper_safe_claim": safe,
            "paper_forbidden_claim": forbidden,
        }
    )


def ge(value: float, threshold: float) -> str:
    return "pass" if math.isfinite(value) and value >= threshold else "fail"


def le(value: float, threshold: float) -> str:
    return "pass" if math.isfinite(value) and value <= threshold else "fail"


def p1_rows(config: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    p1_path = ROOT / config["output"]["metrics"] / "P1_family_metric_lofo.csv"
    ab_path = ROOT / config["output"]["metrics"] / "P1_alpha_ablation.csv"
    baseline_path = ROOT / config["inputs"]["week9_hard_negative_summary"]
    safe = "Family metric learning is interpreted as a targeted family-boundary calibration patch."
    forbidden = "Do not claim illegal-family attribution success from P1 alone."
    if not p1_path.exists() or not ab_path.exists():
        for hyp in ["H_P1a", "H_P1b", "H_P1c", "H_P1d"]:
            add(rows, "P1_family_metric", hyp, "missing", "see config", "fail", safe, forbidden)
        return
    p1 = pd.read_csv(p1_path)
    ab = pd.read_csv(ab_path)
    baseline = pd.read_csv(baseline_path)
    base_mean = float(baseline["hard_negative_auc_mean"].mean())
    mean_auc = float(p1["hard_negative_auc"].mean())
    tsars_auc = float(p1[p1["family_id"].eq("tsars")]["hard_negative_auc"].mean())
    delta = mean_auc - base_mean
    a2 = ab[ab["method"].eq("A2_main")]["hard_negative_auc"].mean()
    a4 = ab[ab["method"].eq("A4_shuffled_family")]["hard_negative_auc"].mean()
    gap = float(a2 - a4)
    add(rows, "P1_family_metric", "H_P1a", mean_auc, ">=0.75", ge(mean_auc, 0.75), safe, forbidden, "mean_hard_negative_auc")
    add(rows, "P1_family_metric", "H_P1b", tsars_auc, ">=0.70", ge(tsars_auc, 0.70), safe, forbidden, "tsars_hard_negative_auc")
    add(rows, "P1_family_metric", "H_P1c", delta, ">=+0.10 vs W9 baseline", ge(delta, 0.10), safe, forbidden, "mean_delta_vs_w9")
    add(rows, "P1_family_metric", "H_P1d", gap, ">=+0.05 A2-A4", ge(gap, 0.05), safe, forbidden, "A2_minus_A4_gap")


def source_baseline_map(config: dict[str, Any]) -> dict[tuple[str, int], float]:
    w7 = pd.read_csv(ROOT / config["inputs"]["week7_transfer_summary"])
    subset = w7[w7["method"].eq("source_only")]
    return {(row.transfer_id, int(row.seed)): float(row.primary_metric_value) for row in subset.itertuples()}


def p2_rows(config: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    path = ROOT / config["output"]["metrics"] / "P2_method_comparison.csv"
    sens_path = ROOT / config["output"]["metrics"] / "P2_eerm_K_sensitivity.csv"
    safe = "IRM/EERM results are method-boundary evidence in small-sample heterogeneous transfer."
    forbidden = "Do not claim IRM/EERM solves transfer."
    if not path.exists():
        for hyp in ["H_P2a", "H_P2b", "H_P2c"]:
            add(rows, "P2_invariance_methods", hyp, "missing", "see config", "fail", safe, forbidden)
        return
    frame = pd.read_csv(path)
    baseline = source_baseline_map(config)
    improvements = []
    new = frame[frame.get("result_origin", "").astype(str).eq("week11_P2")].copy()
    for row in new.itertuples():
        value = float(getattr(row, "primary_metric_value", np.nan))
        base = baseline.get((row.transfer_id, int(row.seed)))
        if base is None or not math.isfinite(value):
            continue
        if row.transfer_id == "T2_ON":
            improvements.append(base - value)
        else:
            improvements.append(value - base)
    max_improvement = float(np.nanmax(improvements)) if improvements else float("nan")
    add(rows, "P2_invariance_methods", "H_P2a", max_improvement, ">=+0.05", ge(max_improvement, 0.05), safe, forbidden, "max_improvement_vs_source_only")
    add(rows, "P2_invariance_methods", "H_P2b", "informative_by_design", "any result is informative", "pass", safe, forbidden, "method_boundary_documented")
    if sens_path.exists():
        sens = pd.read_csv(sens_path)
        max_range = float(sens["absolute_range"].max()) if not sens.empty else float("nan")
    else:
        max_range = float("nan")
    add(rows, "P2_invariance_methods", "H_P2c", max_range, "<=0.10", le(max_range, 0.10), safe, forbidden, "max_eerm_k_range")


def p3_rows(config: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    path = ROOT / config["output"]["metrics"] / "P3_full_fewshot_matrix.csv"
    safe = "Full few-shot matrix is interpreted target-by-target, including one-class boundary targets."
    forbidden = "Do not force ROC-AUC for one-class targets or claim universal few-shot improvement."
    if not path.exists():
        for hyp in ["H_P3a", "H_P3b", "H_P3c"]:
            add(rows, "P3_full_fewshot", hyp, "missing", "see config", "fail", safe, forbidden)
        return
    frame = pd.read_csv(path)
    t1 = float(frame[(frame["target_id"].eq("T1_Nordic")) & (frame["shot"].astype(int).eq(5))]["primary_metric_value"].mean())
    t2on = float(frame[(frame["target_id"].eq("T2_ON")) & (frame["shot"].astype(int).eq(5))]["primary_metric_value"].mean())
    recoverable = 0
    if math.isfinite(t1) and t1 >= 0.50:
        recoverable += 1
    if math.isfinite(t2on) and t2on < 0.30:
        recoverable += 1
    t2ph_best = frame[frame["target_id"].eq("T2_PH")]["delta_vs_source_only"].max()
    if math.isfinite(float(t2ph_best)) and float(t2ph_best) >= 0.05:
        recoverable += 1
    t3_best = frame[frame["target_id"].eq("T3_DiagnoseFrance")]["delta_vs_source_only"].max()
    if math.isfinite(float(t3_best)) and float(t3_best) >= 0.05:
        recoverable += 1
    add(rows, "P3_full_fewshot", "H_P3a", t1, ">=0.50", ge(t1, 0.50), safe, forbidden, "T1_5shot_illegal_recall")
    add(rows, "P3_full_fewshot", "H_P3b", t2on, "<0.30", le(t2on, 0.30), safe, forbidden, "T2_ON_5shot_mean_pred_illegal")
    add(rows, "P3_full_fewshot", "H_P3c", recoverable, ">=3/4 recoverable", "pass" if recoverable >= 3 else "fail", safe, forbidden, "recoverable_transfer_count")


def p4_rows(config: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    w7_path = ROOT / config["output"]["metrics"] / "P4_w7_method_pairwise.csv"
    w9_path = ROOT / config["output"]["metrics"] / "P4_w9_fewshot_significance.csv"
    safe = "Significance tests add uncertainty estimates and do not override effect-size interpretation."
    forbidden = "Do not use p-values to overclaim broad transfer success."
    if w7_path.exists():
        w7 = pd.read_csv(w7_path)
        count = int(w7["holm_reject_bootstrap"].sum())
        add(rows, "P4_significance", "H_P4a", count, ">=1/24 Holm-significant", "pass" if count >= 1 else "fail", safe, forbidden, "w7_significant_pair_count")
    else:
        add(rows, "P4_significance", "H_P4a", "missing", ">=1/24", "fail", safe, forbidden)
    if w9_path.exists():
        w9 = pd.read_csv(w9_path)
        count = int(w9["holm_reject_bootstrap"].sum())
        add(rows, "P4_significance", "H_P4b", count, ">=4/8 Holm-significant", "pass" if count >= 4 else "fail", safe, forbidden, "w9_significant_pair_count")
    else:
        add(rows, "P4_significance", "H_P4b", "missing", ">=4/8", "fail", safe, forbidden)


def count_chars(path: Path) -> int:
    if not path.exists():
        return 0
    text = path.read_text(encoding="utf-8")
    return len(re.sub(r"\s+", "", text))


def p5_rows(config: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    safe = "The writing patch turns negative and diagnostic results into an explicit evidence-boundary framework."
    forbidden = "Do not rewrite the discussion as model-wide success."
    draft = ROOT / "paper" / "draft"
    sec5 = draft / "sec5_honest_evidence_boundary_framework.md"
    section5_chars = count_chars(sec5)
    rq_files = [
        draft / "sec3_methods_deepened_week11.md",
        draft / "sec4_1_rq1_family_metric_week11.md",
        draft / "sec4_3_rq3_invariance_methods_week11.md",
        draft / "sec4_4_rq4_full_fewshot_week11.md",
        sec5,
    ]
    total = sum(count_chars(path) for path in rq_files)
    cites = sec5.exists() and "docs/evidence_boundary_table.md" in sec5.read_text(encoding="utf-8")
    add(rows, "P5_writing", "H_P5a", section5_chars, ">=3500 chars", ge(section5_chars, 3500), safe, forbidden, "section5_chars")
    add(rows, "P5_writing", "H_P5b", total, ">=8000 chars", ge(total, 8000), safe, forbidden, "rq_plus_section5_chars")
    add(rows, "P5_writing", "H_P5c", int(cites), "explicit citation", "pass" if cites else "fail", safe, forbidden, "section5_cites_evidence_boundary_table")


def score_estimate(audit: pd.DataFrame) -> pd.DataFrame:
    pass_count = int(audit["status"].eq("pass").sum())
    total = int(len(audit))
    key = audit[audit["hypothesis"].isin(["H_P1a", "H_P1b", "H_P1c", "H_P3a"])]
    key_pass = int(key["status"].eq("pass").sum())
    if pass_count == total:
        score = "9.0/10 theoretical upper bound"
    elif key_pass >= 3 and pass_count >= 11:
        score = "8.5-8.7/10 likely"
    elif pass_count >= 8:
        score = "around 8.0/10 bounded improvement"
    else:
        score = "7.7-8.0/10, mainly P4/P5 and negative-result value"
    return pd.DataFrame(
        [
            {
                "hypotheses_passed": pass_count,
                "hypotheses_total": total,
                "key_hypotheses_passed": key_pass,
                "estimated_project_score_band": score,
                "paper_safe_claim": "Score band is a planning heuristic, not a statistical result.",
            }
        ]
    )


def main() -> None:
    config = load_config()
    paths = output_dirs(config)
    copy_predeclared_docs(paths)
    rows: list[dict[str, Any]] = []
    p1_rows(config, rows)
    p2_rows(config, rows)
    p3_rows(config, rows)
    p4_rows(config, rows)
    p5_rows(config, rows)
    audit = pd.DataFrame(rows)
    audit.to_csv(paths["metrics"] / "week11_acceptance_audit.csv", index=False, encoding="utf-8")
    audit.to_csv(paths["tables"] / "table_week11_hypothesis_audit.csv", index=False, encoding="utf-8")
    summary = (
        audit.groupby("patch")
        .agg(
            num_hypotheses=("hypothesis", "count"),
            num_pass=("status", lambda s: int((s == "pass").sum())),
            num_fail=("status", lambda s: int((s == "fail").sum())),
        )
        .reset_index()
    )
    summary["is_prerun_frozen"] = True
    summary.to_csv(paths["tables"] / "table_week11_patch_summary.csv", index=False, encoding="utf-8")
    score_estimate(audit).to_csv(paths["metrics"] / "week11_score_estimate.csv", index=False, encoding="utf-8")
    print(audit["status"].value_counts().to_string())


if __name__ == "__main__":
    main()
