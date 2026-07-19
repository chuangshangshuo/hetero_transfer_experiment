"""Bridge tables revising Week-7 interpretations with Week-8 diagnostics."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


TRANSFER_DOMAINS = {
    "T1_Nordic": {
        "source_domain": "Sweden + Spain + Italy",
        "target_domain": "Denmark",
        "primary_metric": "illegal_recall_at_youden",
    },
    "T2_PH": {
        "source_domain": "European pooled",
        "target_domain": "Philippines",
        "primary_metric": "roc_auc",
    },
    "T2_ON": {
        "source_domain": "European pooled",
        "target_domain": "Ontario",
        "primary_metric": "mean_pred_illegal",
    },
    "T3_DiagnoseFrance": {
        "source_domain": "Spain + Italy",
        "target_domain": "France",
        "primary_metric": "roc_auc",
    },
}

BINARY_AUC_TRANSFERS = {"T2_PH", "T3_DiagnoseFrance"}

E5_CONDITIONS = {
    "C1_full": "full",
    "C2_no_cctld": "no_cctld",
    "C3_no_website_lexical": "no_website_lexical",
    "C4_graph_only": "graph_only",
    "C5_lex_only": "lex_only",
    "C6_no_edges": "no_edges",
    "C7_no_cert_edge": "no_cert_edge",
    "C8_no_registrar_edge": "no_registrar_edge",
    "C9_infra_edges_only": "infra_edges_only",
}


def read_csv(path: str) -> pd.DataFrame:
    """Read CSV."""
    full = ROOT / path
    return pd.read_csv(full) if full.exists() else pd.DataFrame()


def write_csv(frame: pd.DataFrame, path: str) -> None:
    """Write CSV."""
    full = ROOT / path
    full.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(full, index=False, encoding="utf-8")


def first_float(frame: pd.DataFrame, column: str) -> float | None:
    """First float."""
    if frame.empty or column not in frame.columns:
        return None
    value = frame[column].iloc[0]
    if pd.isna(value):
        return None
    try:
        return float(value)
    except Exception:
        return None


def source_only_row(method_summary: pd.DataFrame, transfer_id: str) -> pd.DataFrame:
    """Source only row."""
    return method_summary[
        (method_summary["transfer_id"] == transfer_id) & (method_summary["method"] == "source_only")
    ]


def best_week7_method(method_summary: pd.DataFrame, transfer_id: str) -> tuple[str | None, float | None]:
    """Best week7 method."""
    subset = method_summary[method_summary["transfer_id"] == transfer_id].copy()
    if subset.empty:
        return None, None
    metric = TRANSFER_DOMAINS[transfer_id]["primary_metric"]
    column = "primary_metric_value_mean"
    subset[column] = pd.to_numeric(subset[column], errors="coerce")
    if metric == "mean_pred_illegal":
        best = subset.sort_values(column, ascending=True).head(1)
    else:
        best = subset.sort_values(column, ascending=False).head(1)
    return str(best["method"].iloc[0]), float(best[column].iloc[0])


def build_week7_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build week7 tables."""
    method_summary = read_csv("output/week7/metrics/transfer_method_summary.csv")
    fewshot = read_csv("output/week9/metrics/fewshot_acceptance_audit.csv")

    taxonomy_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for transfer_id, domains in TRANSFER_DOMAINS.items():
        src = source_only_row(method_summary, transfer_id)
        auc = first_float(src, "target_test_auc_mean") if transfer_id in BINARY_AUC_TRANSFERS else None
        source_primary = first_float(src, "primary_metric_value_mean")
        best_method, best_value = best_week7_method(method_summary, transfer_id)
        few = fewshot[fewshot["transfer_id"] == transfer_id] if not fewshot.empty else pd.DataFrame()
        few_auc = first_float(few, "best_fewshot_auc")
        few_delta = first_float(few, "delta_vs_source_only")

        if transfer_id == "T1_Nordic":
            diagnostic = (
                "One-class Denmark target; source-only illegal_recall_at_youden=0.2065 and best Week7 recall=0.2710. "
                "ROC-AUC is undefined."
            )
            transfer_class = "hard transfer"
            safe = (
                "Denmark transfer should be reported as low-recall hard transfer under one-class target labels, "
                "not as evidence for or against broad structural generalization."
            )
        elif transfer_id == "T2_PH":
            diagnostic = (
                "Week7 source-only AUC=0.6417 and best adaptation AUC=0.7458 stay below 0.80; "
                "Week9 post-hoc few-shot reaches AUC=0.9250 at 5 shots per class."
            )
            transfer_class = "few-shot recoverable hard transfer"
            safe = (
                "Philippines is a hard transfer target that becomes recoverable with small target supervision; "
                "Week7 alone does not establish robust unsupervised transfer."
            )
        elif transfer_id == "T2_ON":
            diagnostic = (
                "Ontario target is one-class licensed; ROC-AUC is undefined. Lower mean_pred_illegal is a boundary "
                "metric, not a binary transfer score."
            )
            transfer_class = "unreliable transfer boundary"
            safe = (
                "Ontario should be used only as licensed-consistency/anomaly-style boundary evidence, not as a "
                "standard transfer success."
            )
        else:
            diagnostic = (
                "Week7 source-only AUC=0.9861 is high, but corrected E5 shows no-ccTLD drop=0.0815, "
                "graph-only AUC=0.7134, and lex-only AUC=0.9530."
            )
            transfer_class = "shortcut-sensitive easy transfer"
            safe = (
                "France has high transfer performance, but Week8 E5 shows substantial lexical/ccTLD assistance; "
                "it is not pure structural transfer evidence."
            )

        taxonomy_rows.append(
            {
                "transfer_id": transfer_id,
                "source_domain": domains["source_domain"],
                "target_domain": domains["target_domain"],
                "source_only_auc": auc,
                "source_only_primary_metric": source_primary,
                "primary_metric": domains["primary_metric"],
                "fewshot_best_auc": few_auc,
                "fewshot_delta_vs_source_only": few_delta,
                "diagnostic_evidence": diagnostic,
                "transfer_class": transfer_class,
                "paper_safe_interpretation": safe,
                "is_posthoc_diagnostic": True,
            }
        )
        summary_rows.append(
            {
                "transfer_id": transfer_id,
                "source_domain": domains["source_domain"],
                "target_domain": domains["target_domain"],
                "primary_metric": domains["primary_metric"],
                "source_only_auc": auc,
                "source_only_primary_metric": source_primary,
                "best_week7_method": best_method,
                "best_week7_primary_metric": best_value,
                "fewshot_best_auc": few_auc,
                "transfer_class": transfer_class,
                "summary_interpretation": safe,
                "is_posthoc_diagnostic": True,
            }
        )

    bridge = pd.DataFrame(
        [
            {
                "week7_observation": "T3 France source-only AUC is very high (0.9861).",
                "initial_interpretation_risk": "Mistaking high AUC for structural generalization.",
                "week8_diagnostic_check": "Corrected E5: no-ccTLD drop=0.0815, graph-only AUC=0.7134, lex-only AUC=0.9530.",
                "revised_interpretation": "T3 is shortcut-sensitive and substantially lexical/ccTLD-assisted.",
                "is_posthoc_diagnostic": True,
            },
            {
                "week7_observation": "T2_PH source-only/adaptation AUC values remain below 0.80.",
                "initial_interpretation_risk": "Treating low unsupervised transfer as total non-transferability.",
                "week8_diagnostic_check": "Week9 few-shot: k=5 target shots per class reaches AUC=0.9250.",
                "revised_interpretation": "T2_PH is hard but few-shot recoverable.",
                "is_posthoc_diagnostic": True,
            },
            {
                "week7_observation": "T1_Nordic has one-class target labels and low illegal recall.",
                "initial_interpretation_risk": "Forcing invalid ROC-AUC or calling all methods failed.",
                "week8_diagnostic_check": "One-class metric boundary plus hard-negative family audits.",
                "revised_interpretation": "T1 is hard transfer evidence under one-class labels, not structural proof.",
                "is_posthoc_diagnostic": True,
            },
            {
                "week7_observation": "T2_ON has no illegal target labels and low mean_pred_illegal for StruRW variants.",
                "initial_interpretation_risk": "Calling anomaly-style low illegal probability a standard transfer victory.",
                "week8_diagnostic_check": "One-class target policy and edge/channel diagnostics.",
                "revised_interpretation": "T2_ON is licensed-consistency boundary evidence only.",
                "is_posthoc_diagnostic": True,
            },
        ]
    )
    return pd.DataFrame(summary_rows), pd.DataFrame(taxonomy_rows), bridge


def build_e3_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build E3 tables."""
    original = read_csv("output/week8/metrics/E3_lofo_family_sensitivity.csv")
    w85 = read_csv("output/week85/metrics/E3_w85_lofo_summary.csv")
    licensed_vals = pd.to_numeric(original["test_roc_auc_mean"], errors="coerce") if not original.empty else pd.Series(dtype=float)
    hard = w85[w85["test_form"] == "hard_illegal_negative"].copy() if not w85.empty else pd.DataFrame()
    hard_vals = pd.to_numeric(hard["test_roc_auc_mean"], errors="coerce") if not hard.empty else pd.Series(dtype=float)

    comparison = pd.DataFrame(
        [
            {
                "evaluation_setting": "licensed-negative LOFO",
                "positive_class": "held-out Denmark illegal family websites",
                "negative_class": "licensed baseline websites",
                "mean_auc": float(licensed_vals.mean()),
                "min_auc": float(licensed_vals.min()),
                "max_auc": float(licensed_vals.max()),
                "what_it_tests": "illegal-vs-licensed separability under held-out positive family partitions",
                "what_it_cannot_test": "family-level discrimination among illegal websites",
                "conclusion": "High scores indicate illegal-vs-licensed separability, not strong unseen-family generalization.",
                "source_artifact": "output/week8/metrics/E3_lofo_family_sensitivity.csv",
                "is_posthoc_diagnostic": True,
            },
            {
                "evaluation_setting": "hard illegal-negative LOFO",
                "positive_class": "held-out Denmark illegal family websites",
                "negative_class": "other Denmark illegal-family websites",
                "mean_auc": float(hard_vals.mean()),
                "min_auc": float(hard_vals.min()),
                "max_auc": float(hard_vals.max()),
                "what_it_tests": "family-level discrimination among illegal websites",
                "what_it_cannot_test": "general illegal-vs-licensed detection",
                "conclusion": "Current hard-negative results do not strongly support unseen illegal-family generalization.",
                "source_artifact": "output/week85/metrics/E3_w85_lofo_summary.csv",
                "is_posthoc_diagnostic": True,
            },
        ]
    )
    hard_summary = hard.rename(
        columns={
            "family_id": "family_id",
            "test_roc_auc_mean": "mean_auc",
            "test_roc_auc_std": "std_auc",
            "recall_at_fpr_0_10_mean": "recall_at_fpr_0_10",
            "score_gap_pos_minus_neg_mean": "score_gap_pos_minus_neg",
        }
    )
    keep = [
        "family_id",
        "num_runs",
        "mean_auc",
        "std_auc",
        "recall_at_fpr_0_10",
        "score_gap_pos_minus_neg",
    ]
    hard_summary = hard_summary[keep].copy()
    hard_summary["threshold"] = 0.80
    hard_summary["status"] = pd.to_numeric(hard_summary["mean_auc"], errors="coerce").apply(
        lambda value: "supported" if value >= 0.80 else "failed"
    )
    hard_summary["paper_safe_interpretation"] = hard_summary.apply(
        lambda row: (
            "This family clears the exploratory hard-negative threshold."
            if row["status"] == "supported"
            else "This family does not support strong family-level discrimination among illegal sites."
        ),
        axis=1,
    )
    hard_summary["is_posthoc_diagnostic"] = True
    return comparison, hard_summary


def e5_interpretation(transfer_id: str, condition: str, value: float | None, full_value: float | None) -> str:
    """E5 interpretation."""
    if transfer_id == "T3_DiagnoseFrance":
        if condition == "full":
            return "High T3 performance baseline; requires diagnostic attribution."
        if condition == "no_cctld":
            return "Removing ccTLD/local-TLD features causes a material performance drop."
        if condition == "graph_only":
            return "Graph-only AUC is lower than lex-only; not pure structural evidence."
        if condition == "lex_only":
            return "Lex-only is strong, indicating substantial lexical/ccTLD assistance."
        if condition == "no_website_lexical":
            return "Removing Website lexical information substantially weakens T3."
        return "Condition contributes to the corrected E5 shortcut/feature-boundary diagnosis."
    if transfer_id in BINARY_AUC_TRANSFERS:
        return "Binary target AUC diagnostic condition; interpret relative to full model."
    return "One-class target metric condition; ROC-AUC is undefined and this is not binary transfer evidence."


def build_e5_table() -> pd.DataFrame:
    """Build E5 table."""
    e5 = read_csv("output/week8_patch/metrics/E5_structure_vs_lexical_summary.csv")
    rows: list[dict[str, Any]] = []
    for transfer_id, group in e5.groupby("transfer_id"):
        full = group[group["config_id"] == "C1_full"]
        full_value = first_float(full, "primary_metric_value_mean")
        for _, row in group.sort_values("config_id").iterrows():
            config = str(row["config_id"])
            condition = E5_CONDITIONS.get(config, config)
            value = float(row["primary_metric_value_mean"])
            auc = value if transfer_id in BINARY_AUC_TRANSFERS else None
            delta = value - full_value if full_value is not None else None
            rows.append(
                {
                    "transfer_id": transfer_id,
                    "condition": condition,
                    "auc": auc,
                    "metric_value": value,
                    "delta_vs_full": delta,
                    "interpretation": e5_interpretation(transfer_id, condition, value, full_value),
                    "is_posthoc_diagnostic": True,
                }
            )
    return pd.DataFrame(rows)


def build_week8_rollup() -> pd.DataFrame:
    """Build week8 rollup."""
    return pd.DataFrame(
        [
            {
                "experiment_id": "E1",
                "original_claim": "Specific graph edge channels would reveal stable structural transfer mechanisms.",
                "diagnostic_result": "T3 shows a uses_cert edge effect, but T2_PH and T2_ON effects are directionally mixed and target-dependent.",
                "final_status": "mixed",
                "paper_safe_claim": "Edge channels are diagnostically useful but not uniformly beneficial across targets.",
                "overclaim_to_avoid": "Do not claim edge structure generally explains transfer success.",
                "is_posthoc_diagnostic": True,
            },
            {
                "experiment_id": "E2",
                "original_claim": "LogME would predict transferability.",
                "diagnostic_result": "Original LogME Spearman rho=0.2754; expanded W8.5 rho=0.0733 and does not beat distance baselines.",
                "final_status": "failed",
                "paper_safe_claim": "LogME is not reliable as a transferability proxy in this setting.",
                "overclaim_to_avoid": "Do not use LogME as a validated prescreen.",
                "is_posthoc_diagnostic": True,
            },
            {
                "experiment_id": "E3",
                "original_claim": "Family LOFO validates unseen illegal-family generalization.",
                "diagnostic_result": "Licensed-negative LOFO is high, but hard illegal-negative mean AUC=0.6218 and tsars=0.3864.",
                "final_status": "limited",
                "paper_safe_claim": "Illegal-vs-licensed separability is strong; family-level generalization among illegal sites is limited.",
                "overclaim_to_avoid": "Do not claim strong unseen-family generalization from licensed-negative LOFO.",
                "is_posthoc_diagnostic": True,
            },
            {
                "experiment_id": "E4",
                "original_claim": "Embedding distances would explain transfer/control-group behavior.",
                "diagnostic_result": "Illegal-vs-control classification is high, but embedding geometry hypotheses H_E4b/H_E4c fail.",
                "final_status": "limited",
                "paper_safe_claim": "Control-group separability is supported, but embedding-distance explanation is insufficient.",
                "overclaim_to_avoid": "Do not claim embedding distance explains the transfer mechanism.",
                "is_posthoc_diagnostic": True,
            },
            {
                "experiment_id": "E5",
                "original_claim": "T3 transfer would be primarily graph-structural rather than lexical.",
                "diagnostic_result": "Corrected E5: graph-only AUC=0.7134, lex-only AUC=0.9530, no-ccTLD drop=0.0815.",
                "final_status": "failed",
                "paper_safe_claim": "T3 France transfer is substantially lexical/ccTLD-assisted.",
                "overclaim_to_avoid": "Do not present T3 France as pure structural transfer evidence.",
                "is_posthoc_diagnostic": True,
            },
        ]
    )


def main() -> None:
    """Command-line entry point."""
    week7_summary, week7_taxonomy, bridge = build_week7_tables()
    e3_comparison, e3_hard = build_e3_tables()
    e5_corrected = build_e5_table()
    rollup = build_week8_rollup()

    write_csv(week7_summary, "output/week7/metrics/week7_transfer_summary.csv")
    write_csv(week7_taxonomy, "output/week7/metrics/week7_transfer_taxonomy.csv")
    write_csv(bridge, "output/week7/metrics/week7_to_week8_bridge.csv")
    write_csv(e3_comparison, "output/week8/metrics/E3_lofo_setting_comparison.csv")
    write_csv(e3_hard, "output/week8/metrics/E3_hard_negative_family_summary.csv")
    write_csv(e5_corrected, "output/week8/metrics/E5_structure_vs_lexical_corrected_summary.csv")
    write_csv(rollup, "output/week8/metrics/week8_diagnostic_rollup.csv")

    print("wrote Week7 interpretation tables:")
    print(week7_taxonomy[["transfer_id", "transfer_class"]].to_string(index=False))
    print("wrote Week8 diagnostic rollup:")
    print(rollup[["experiment_id", "final_status"]].to_string(index=False))


if __name__ == "__main__":
    main()
