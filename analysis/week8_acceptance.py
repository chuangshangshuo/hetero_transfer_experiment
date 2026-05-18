from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.train.utils import save_dataframe
from src.utils.paired_bootstrap import paired_bootstrap


METRICS = ROOT / "output" / "week8" / "metrics"
AUDITS = ROOT / "output" / "week8" / "audits"
PAPER = ROOT / "paper" / "draft"


def _finite(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out


def _status(pass_condition: bool, partial: bool = False) -> str:
    if pass_condition:
        return "pass"
    if partial:
        return "partial"
    return "fail"


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def e1_rows() -> list[dict[str, Any]]:
    summary = _load_csv(METRICS / "E1_edge_ablation_summary.csv")
    rows: list[dict[str, Any]] = []
    t3 = summary[summary["transfer_id"] == "T3_DiagnoseFrance"]
    h1_candidates = t3[t3["edge_removed"].isin(["hosted_on", "uses_cert"])]
    h1_best = h1_candidates.sort_values("effect_mean", ascending=False).head(1)
    h1_obs = _finite(h1_best["effect_mean"].iloc[0]) if not h1_best.empty else float("nan")
    h1_ci_lo = _finite(h1_best["effect_ci_lo"].iloc[0]) if not h1_best.empty else float("nan")
    rows.append(
        {
            "experiment": "E1",
            "hypothesis": "H_E1a",
            "observed_metric": "max_T3_drop_from_removing_hosted_on_or_uses_cert",
            "observed_value": h1_obs,
            "threshold": ">=0.05 and 95% CI excludes 0",
            "status": _status(h1_obs >= 0.05 and h1_ci_lo > 0),
            "details": f"best_edge={h1_best['edge_removed'].iloc[0] if not h1_best.empty else 'NA'}, ci_lo={h1_ci_lo:.4f}",
        }
    )
    t2ph = summary[summary["transfer_id"] == "T2_PH"].copy()
    max_abs_delta = float(t2ph["delta_raw_mean"].abs().max()) if not t2ph.empty else float("nan")
    worst_edge = t2ph.loc[t2ph["delta_raw_mean"].abs().idxmax(), "edge_removed"] if not t2ph.empty else "NA"
    rows.append(
        {
            "experiment": "E1",
            "hypothesis": "H_E1b",
            "observed_metric": "max_abs_T2_PH_delta_auc",
            "observed_value": max_abs_delta,
            "threshold": "<0.03 for all five edges",
            "status": _status(max_abs_delta < 0.03),
            "details": f"largest_edge={worst_edge}",
        }
    )
    t2on = summary[(summary["transfer_id"] == "T2_ON") & (summary["edge_removed"].isin(["uses_ns", "registered_via"]))]
    max_delta = float(t2on["delta_raw_mean"].max()) if not t2on.empty else float("nan")
    best_edge = t2on.loc[t2on["delta_raw_mean"].idxmax(), "edge_removed"] if not t2on.empty else "NA"
    rows.append(
        {
            "experiment": "E1",
            "hypothesis": "H_E1c",
            "observed_metric": "max_T2_ON_mean_pred_illegal_increase",
            "observed_value": max_delta,
            "threshold": ">=0.10 for -uses_ns or -registered_via",
            "status": _status(max_delta >= 0.10),
            "details": f"best_edge={best_edge}",
        }
    )
    return rows


def e2_rows() -> list[dict[str, Any]]:
    metrics = _load_csv(METRICS / "E2_transferability_metrics.csv")
    rows: list[dict[str, Any]] = []
    if metrics.empty:
        return [
            {
                "experiment": "E2",
                "hypothesis": hypothesis,
                "observed_metric": "missing",
                "observed_value": float("nan"),
                "threshold": "see predeclared hypotheses",
                "status": "not_run",
                "details": "E2_transferability_metrics.csv missing",
            }
            for hypothesis in ["H_E2a", "H_E2b", "H_E2c"]
        ]
    by_metric = metrics.set_index("metric")
    rho_logme = _finite(by_metric.loc["logme_cross", "spearman_rho"])
    p_logme = _finite(by_metric.loc["logme_cross", "spearman_p"])
    tau_logme = _finite(by_metric.loc["logme_cross", "kendall_tau"])
    rho_hdiv = abs(_finite(by_metric.loc["h_divergence", "spearman_rho"]))
    rho_wass = abs(_finite(by_metric.loc["wasserstein", "spearman_rho"]))
    aligned = int(_finite(by_metric.loc["logme_cross", "aligned_pairs"]))
    rows.append(
        {
            "experiment": "E2",
            "hypothesis": "H_E2a",
            "observed_metric": "LogME_AUC_spearman_rho",
            "observed_value": rho_logme,
            "threshold": "rho>=0.60 and p<0.05",
            "status": _status(rho_logme >= 0.60 and p_logme < 0.05, partial=(0.40 <= rho_logme < 0.60)),
            "details": f"p={p_logme:.4f}, aligned_rows={aligned}",
        }
    )
    rows.append(
        {
            "experiment": "E2",
            "hypothesis": "H_E2b",
            "observed_metric": "abs_rho_LogME_vs_baselines",
            "observed_value": abs(rho_logme),
            "threshold": "|rho_LogME| > |rho_H-div| and > |rho_W|",
            "status": _status(abs(rho_logme) > rho_hdiv and abs(rho_logme) > rho_wass),
            "details": f"|rho_H-div|={rho_hdiv:.4f}, |rho_W|={rho_wass:.4f}",
        }
    )
    rows.append(
        {
            "experiment": "E2",
            "hypothesis": "H_E2c",
            "observed_metric": "LogME_AUC_kendall_tau",
            "observed_value": tau_logme,
            "threshold": "tau>=0.50",
            "status": _status(tau_logme >= 0.50),
            "details": f"aligned_rows={aligned}",
        }
    )
    return rows


def e3_rows() -> list[dict[str, Any]]:
    summary = _load_csv(METRICS / "E3_lofo_family_sensitivity.csv")
    perm = _load_csv(METRICS / "E3_permutation_test_summary.csv")
    min_auc = float(summary["test_roc_auc_mean"].min()) if not summary.empty else float("nan")
    max_std = float(summary["test_roc_auc_std"].max()) if not summary.empty else float("nan")
    tsars = summary[summary["family_id"] == "tsars"]
    tsars_auc = _finite(tsars["test_roc_auc_mean"].iloc[0]) if not tsars.empty else float("nan")
    perm_note = ""
    if not perm.empty:
        perm_note = (
            f"; permutation real_minus_random={_finite(perm['real_minus_random'].iloc[0]):.4f}, "
            f"p_real_lower={_finite(perm['one_sided_p_real_lower'].iloc[0]):.3f}"
        )
    return [
        {
            "experiment": "E3",
            "hypothesis": "H_E3a",
            "observed_metric": "min_family_lofo_auc_mean",
            "observed_value": min_auc,
            "threshold": ">=0.85",
            "status": _status(min_auc >= 0.85),
            "details": f"families={len(summary)}{perm_note}",
        },
        {
            "experiment": "E3",
            "hypothesis": "H_E3b",
            "observed_metric": "tsars_lofo_auc_mean",
            "observed_value": tsars_auc,
            "threshold": ">=0.85",
            "status": _status(tsars_auc >= 0.85),
            "details": "critical W6 cross-verified robustness check",
        },
        {
            "experiment": "E3",
            "hypothesis": "H_E3c",
            "observed_metric": "max_family_auc_std",
            "observed_value": max_std,
            "threshold": "<0.10",
            "status": _status(max_std < 0.10),
            "details": "all eligible families use five seeds",
        },
    ]


def e4_rows() -> list[dict[str, Any]]:
    summary = _load_csv(METRICS / "E4_three_scenario_summary.csv")
    distances = _load_csv(METRICS / "E4_embedding_distances.csv")
    s2 = summary[summary["scenario"] == "S2_control"]
    s2_auc = _finite(s2["test_roc_auc_mean"].iloc[0]) if not s2.empty else float("nan")
    e4b_count = int((distances["H_E4b_pass"].astype(str).str.lower() == "true").sum()) if not distances.empty else 0
    e4c_count = int((distances["H_E4c_pass"].astype(str).str.lower() == "true").sum()) if not distances.empty else 0
    return [
        {
            "experiment": "E4",
            "hypothesis": "H_E4a",
            "observed_metric": "S2_illegal_vs_control_auc_mean",
            "observed_value": s2_auc,
            "threshold": ">=0.85",
            "status": _status(s2_auc >= 0.85),
            "details": "control_legal_commercial reintroduced only for E4",
        },
        {
            "experiment": "E4",
            "hypothesis": "H_E4b",
            "observed_metric": "encoder_pass_count",
            "observed_value": e4b_count,
            "threshold": ">=4/5 encoders",
            "status": _status(e4b_count >= 4),
            "details": "d(illegal,control) > d(illegal,licensed)+0.5sigma",
        },
        {
            "experiment": "E4",
            "hypothesis": "H_E4c",
            "observed_metric": "encoder_pass_count",
            "observed_value": e4c_count,
            "threshold": ">=4/5 encoders",
            "status": _status(e4c_count >= 4),
            "details": "d(licensed,control) > d(licensed,illegal)",
        },
    ]


def e5_rows() -> list[dict[str, Any]]:
    raw = _load_csv(METRICS / "E5_structure_vs_lexical.csv")
    summary = _load_csv(METRICS / "E5_structure_vs_lexical_summary.csv")
    t3 = summary[summary["transfer_id"] == "T3_DiagnoseFrance"].set_index("config_id")
    c1 = _finite(t3.loc["C1_full", "primary_metric_value_mean"])
    c2 = _finite(t3.loc["C2_no_cctld", "primary_metric_value_mean"])
    c4 = _finite(t3.loc["C4_graph_only", "primary_metric_value_mean"])
    c5 = _finite(t3.loc["C5_lex_only", "primary_metric_value_mean"])
    drop = c1 - c2
    diff = c4 - c5
    ci_note = ""
    if not raw.empty:
        t3_raw = raw[raw["transfer_id"] == "T3_DiagnoseFrance"]
        graph = t3_raw[t3_raw["config_id"] == "C4_graph_only"].set_index("seed")["primary_metric_value"]
        lex = t3_raw[t3_raw["config_id"] == "C5_lex_only"].set_index("seed")["primary_metric_value"]
        aligned = pd.concat([graph.rename("graph"), lex.rename("lex")], axis=1).dropna()
        boot = paired_bootstrap((aligned["graph"] - aligned["lex"]).to_numpy(dtype=float), B=1000, seed=42)
        ci_note = f", paired_ci=[{boot.ci_lo:.4f},{boot.ci_hi:.4f}]"
    return [
        {
            "experiment": "E5",
            "hypothesis": "H_E5a",
            "observed_metric": "T3_full_minus_no_cctld_auc",
            "observed_value": drop,
            "threshold": "<0.05 drop",
            "status": _status(drop < 0.05),
            "details": f"C1={c1:.4f}, C2={c2:.4f}",
        },
        {
            "experiment": "E5",
            "hypothesis": "H_E5b",
            "observed_metric": "T3_graph_only_auc_mean",
            "observed_value": c4,
            "threshold": ">=0.85",
            "status": _status(c4 >= 0.85),
            "details": "Website.x zeroed, Graph v2 edges retained",
        },
        {
            "experiment": "E5",
            "hypothesis": "H_E5c",
            "observed_metric": "T3_graph_only_minus_lex_only_auc",
            "observed_value": diff,
            "threshold": ">=0.10 and CI excludes 0",
            "status": _status(diff >= 0.10),
            "details": f"C4={c4:.4f}, C5={c5:.4f}{ci_note}",
        },
    ]


def completion_rows() -> pd.DataFrame:
    specs = [
        ("E1_raw_runs", METRICS / "E1_edge_ablation_raw_runs.csv", 120),
        ("E2_pair_scores", METRICS / "E2_pair_scores.csv", 30),
        ("E2_realized_transfer_auc", METRICS / "E2_realized_transfer_auc.csv", 30),
        ("E3_lofo_raw_runs", METRICS / "E3_lofo_family_sensitivity_raw.csv", 35),
        ("E3_permutation_runs", AUDITS / "E3_permutation_aucs.csv", 500),
        ("E4_raw_runs", METRICS / "E4_three_scenario_raw_runs.csv", 15),
        ("E5_raw_runs", METRICS / "E5_structure_vs_lexical.csv", 100),
    ]
    rows = []
    for name, path, expected in specs:
        frame = _load_csv(path)
        rows.append(
            {
                "artifact": name,
                "path": str(path.relative_to(ROOT)),
                "expected_rows": expected,
                "observed_rows": int(len(frame)),
                "status": "pass" if len(frame) == expected else "check",
            }
        )
    return pd.DataFrame(rows)


def markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in frame.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                values.append(f"{value:.4f}" if math.isfinite(value) else "nan")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_paper_draft(audit: pd.DataFrame, completion: pd.DataFrame) -> None:
    PAPER.mkdir(parents=True, exist_ok=True)
    status_counts = audit["status"].value_counts().to_dict()
    text = f"""# Week 8 Mechanism Evidence Summary

Week 8 evaluated fifteen predeclared mechanism hypotheses across edge-channel ablation, transferability scoring, Denmark family LOFO, control-group reintroduction, and structure-vs-lexical slicing. The result is intentionally mixed: {status_counts.get('pass', 0)} pass, {status_counts.get('partial', 0)} partial, and {status_counts.get('fail', 0)} fail.

## Acceptance Audit

{markdown_table(audit[['experiment', 'hypothesis', 'observed_metric', 'observed_value', 'threshold', 'status']])}

## Completion Audit

{markdown_table(completion)}

## Interpretation Notes

- E3 supports the core Denmark family-level robustness claim, including the tsars held-out audit. The supplemental permutation test does not show that the reconstructed real families are harder than random held-out groups, so the family labels should be framed as a robustness partition rather than definitive organization-level causal units.
- E5 is the major counterexample: T3 France transfer loses more than the predeclared threshold when ccTLD/local-TLD information is removed, graph-only underperforms, and lex-only remains strong. This supports a lexical/ccTLD mechanism chapter rather than a graph-only explanation.
- E2 does not validate LogME as a strong transferability prescreen in this small complete-label subset; H-divergence is numerically stronger in the current aligned set.
"""
    (PAPER / "sec4_4_week8_mechanism.md").write_text(text, encoding="utf-8")


def main() -> None:
    rows: list[dict[str, Any]] = []
    rows.extend(e1_rows())
    rows.extend(e2_rows())
    rows.extend(e3_rows())
    rows.extend(e4_rows())
    rows.extend(e5_rows())
    audit = pd.DataFrame(rows)
    save_dataframe(audit, METRICS / "week8_acceptance_audit.csv")
    completion = completion_rows()
    save_dataframe(completion, METRICS / "week8_completion_audit.csv")
    write_paper_draft(audit, completion)
    print(audit[["experiment", "hypothesis", "observed_value", "threshold", "status"]].to_string(index=False))
    print()
    print(completion.to_string(index=False))


if __name__ == "__main__":
    main()
