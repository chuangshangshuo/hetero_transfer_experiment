"""Week-10 final RQ evidence tables and hypothesis rollup."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.week10_common import (
    DEFAULT_CONFIG,
    ensure_output_dirs,
    first_float,
    first_text,
    input_path,
    load_config,
    markdown_table,
    output_path,
    read_input,
    save_csv,
    with_claim_columns,
)


def build_rq1(config: dict[str, Any]) -> pd.DataFrame:
    """Build rq1."""
    cv = read_input(config, "week6", "cross_verified")
    head = read_input(config, "week6", "head_ablation_summary")
    hard = read_input(config, "week9", "hard_negative_summary")
    pooled = cv[cv["scope"].eq("pooled_5_seed")] if "scope" in cv.columns else cv
    best_head = head.sort_values("roc_auc_mean", ascending=False).head(1) if not head.empty else pd.DataFrame()
    rows = [
        with_claim_columns(
            {
                "rq": "RQ1",
                "evidence_item": "cross_verified_vs_licensed",
                "metric": "roc_auc",
                "value": first_float(pooled, "roc_auc"),
                "supporting_file": str(input_path(config, "week6", "cross_verified").relative_to(ROOT)),
                "interpretation": "Cross-verified illegal sites are highly separable from licensed controls.",
            },
            "support_illegal_vs_licensed",
            False,
            "The model separates illegal websites from licensed controls in the cross-verified evidence layer.",
            "Do not claim the model performs reliable illegal-family attribution.",
        ),
        with_claim_columns(
            {
                "rq": "RQ1",
                "evidence_item": "hard_negative_family_boundary",
                "metric": "mean_hard_negative_auc",
                "value": float(pd.to_numeric(hard.get("hard_negative_auc_mean", pd.Series(dtype=float)), errors="coerce").mean()),
                "supporting_file": str(input_path(config, "week9", "hard_negative_summary").relative_to(ROOT)),
                "interpretation": "Hard illegal-vs-illegal family discrimination is weak and family-dependent.",
            },
            "family_boundary_failed",
            True,
            "Illegal-vs-licensed separability is strong, but family-level discrimination among illegal sites remains limited.",
            "Do not claim successful illegal-family attribution or robust unseen-family recognition.",
        ),
        with_claim_columns(
            {
                "rq": "RQ1",
                "evidence_item": "head_ablation_best",
                "metric": "roc_auc_mean",
                "value": first_float(best_head, "roc_auc_mean"),
                "supporting_file": str(input_path(config, "week6", "head_ablation_summary").relative_to(ROOT)),
                "interpretation": f"Best head is {first_text(best_head, 'head_type', 'unknown')}; head choice affects stability.",
            },
            "mixed_model_integration",
            True,
            "Head ablation suggests stability depends on downstream head design.",
            "Do not claim HeCo universally outperforms HeteroGNN or that one head solves all settings.",
        ),
    ]
    return pd.DataFrame(rows)


def build_rq2(config: dict[str, Any]) -> pd.DataFrame:
    """Build rq2."""
    e5 = read_input(config, "week8_patch", "corrected_e5_acceptance")
    summary = read_input(config, "week8_patch", "corrected_e5_summary")
    t3 = summary[summary.get("transfer_id", "").eq("T3_DiagnoseFrance")] if not summary.empty else pd.DataFrame()
    rows = [
        with_claim_columns(
            {
                "rq": "RQ2",
                "evidence_item": "T3_ccTLD_sensitivity",
                "metric": "full_minus_no_cctld_auc",
                "value": first_float(e5[e5["hypothesis"].eq("W8P_E5_ccTLD_sensitivity")], "observed_value") if not e5.empty else float("nan"),
                "supporting_file": str(input_path(config, "week8_patch", "corrected_e5_acceptance").relative_to(ROOT)),
                "interpretation": "Removing ccTLD/local-TLD information materially changes T3 France performance.",
            },
            "regional_shortcut_boundary",
            True,
            "T3 France transfer is substantially assisted by ccTLD/local lexical boundary information.",
            "Do not present T3 France as pure structural transfer.",
        ),
        with_claim_columns(
            {
                "rq": "RQ2",
                "evidence_item": "T3_lexical_vs_graph",
                "metric": "lex_only_minus_graph_only_auc",
                "value": first_float(e5[e5["hypothesis"].eq("W8P_E5_lex_only_vs_graph_only")], "observed_value") if not e5.empty else float("nan"),
                "supporting_file": str(input_path(config, "week8_patch", "corrected_e5_acceptance").relative_to(ROOT)),
                "interpretation": "Lexical-only T3 performance exceeds graph-only performance.",
            },
            "regional_lexical_boundary",
            True,
            "Regional heterogeneity includes lexical/ccTLD shortcuts that must be separated from structural evidence.",
            "Do not claim graph-only regional generalization from high full-feature T3 AUC.",
        ),
    ]
    if not t3.empty:
        for _, row in t3.iterrows():
            rows.append(
                with_claim_columns(
                    {
                        "rq": "RQ2",
                        "evidence_item": f"T3_{row['config_id']}",
                        "metric": "primary_metric_value_mean",
                        "value": float(row["primary_metric_value_mean"]),
                        "supporting_file": str(input_path(config, "week8_patch", "corrected_e5_summary").relative_to(ROOT)),
                        "interpretation": f"T3 feature bucket {row['config_id']} diagnostic performance.",
                    },
                    "diagnostic_context",
                    True,
                    "Feature-bucket diagnostics expose which regional signals support or weaken transfer.",
                    "Do not turn a single feature bucket into a universal mechanism claim.",
                )
            )
    return pd.DataFrame(rows)


def build_rq3(config: dict[str, Any]) -> pd.DataFrame:
    """Build rq3."""
    transfer = read_input(config, "week7", "transfer_summary")
    diagnostic = read_input(config, "week8", "diagnostic_rollup")
    rows: list[dict[str, Any]] = []
    for _, row in transfer.iterrows():
        rows.append(
            with_claim_columns(
                {
                    "rq": "RQ3",
                    "evidence_item": str(row["transfer_id"]),
                    "metric": str(row["primary_metric"]),
                    "value": row.get("source_only_primary_metric", row.get("source_only_auc", "")),
                    "supporting_file": str(input_path(config, "week7", "transfer_summary").relative_to(ROOT)),
                    "interpretation": row.get("summary_interpretation", ""),
                },
                str(row.get("transfer_class", "target_dependent_transfer")),
                True,
                "Transfer is target-dependent and must be interpreted with target-specific metric boundaries.",
                "Do not claim DANN/StruRW solve cross-domain transfer or force ROC-AUC onto one-class targets.",
            )
        )
    logme = diagnostic[diagnostic.get("experiment_id", "").eq("E2")] if not diagnostic.empty else pd.DataFrame()
    rows.append(
        with_claim_columns(
            {
                "rq": "RQ3",
                "evidence_item": "LogME_transferability",
                "metric": "diagnostic_status",
                "value": first_text(logme, "final_status", "failed"),
                "supporting_file": str(input_path(config, "week8", "diagnostic_rollup").relative_to(ROOT)),
                "interpretation": first_text(logme, "diagnostic_result", "LogME failed as a transferability proxy."),
            },
            "transferability_proxy_failed",
            True,
            "LogME is not reliable as a transferability prescreen in this setting.",
            "Do not claim LogME reliably predicts transfer success.",
        )
    )
    return pd.DataFrame(rows)


def build_rq4(config: dict[str, Any]) -> pd.DataFrame:
    """Build rq4."""
    few = read_input(config, "week9", "fewshot_rollup")
    rows: list[dict[str, Any]] = []
    for _, row in few.iterrows():
        status = "posthoc_support" if str(row["target_id"]) == "T2_PH" else "no_clear_gain"
        forbidden = (
            "Do not claim few-shot universally improves transfer."
            if str(row["target_id"]) == "T2_PH"
            else "Do not write saturated T3 as a few-shot success or structural proof."
        )
        rows.append(
            with_claim_columns(
                {
                    "rq": "RQ4",
                    "evidence_item": str(row["target_id"]),
                    "metric": "best_mean_delta_auc",
                    "value": float(row["best_mean_delta_auc"]),
                    "best_shot": int(row["best_shot"]),
                    "best_mean_auc": float(row["best_mean_auc"]),
                    "supporting_file": str(input_path(config, "week9", "fewshot_rollup").relative_to(ROOT)),
                    "interpretation": row["paper_safe_claim"],
                },
                status,
                True,
                row["paper_safe_claim"],
                forbidden,
            )
        )
    return pd.DataFrame(rows)


def build_final_rq_evidence(config: dict[str, Any]) -> pd.DataFrame:
    """Build final rq evidence."""
    return pd.concat(
        [build_rq1(config), build_rq2(config), build_rq3(config), build_rq4(config)],
        ignore_index=True,
        sort=False,
    )


def build_hypothesis_rollup(config: dict[str, Any]) -> pd.DataFrame:
    """Build hypothesis rollup."""
    sources = [
        ("week7_transfer", "week7", "acceptance_audit", True),
        ("week8_diagnostic", "week8", "diagnostic_rollup", True),
        ("week8_patch_e5", "week8_patch", "corrected_e5_acceptance", True),
        ("week85_patch", "week85", "acceptance_audit", True),
        ("week9_fewshot", "week9", "fewshot_acceptance_v2", True),
        ("week9_hard_negative", "week9", "hard_negative_acceptance", True),
    ]
    frames = []
    for layer, group, key, default_posthoc in sources:
        frame = read_input(config, group, key)
        if frame.empty:
            continue
        frame = frame.copy()
        frame["evidence_layer"] = layer
        if "is_posthoc" not in frame.columns:
            frame["is_posthoc"] = default_posthoc
        frame["is_posthoc"] = frame["is_posthoc"].astype(str).str.lower().isin(["true", "1", "yes"])
        if "status" not in frame.columns:
            frame["status"] = frame.get("final_status", "diagnostic_context")
        if "paper_safe_claim" not in frame.columns:
            frame["paper_safe_claim"] = frame.get("paper_safe_claim", "Diagnostic evidence should be interpreted within its stated boundary.")
        if "paper_forbidden_claim" not in frame.columns:
            frame["paper_forbidden_claim"] = frame.get(
                "overclaim_to_avoid",
                "Do not overgeneralize diagnostic or post-hoc evidence.",
            )
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def write_paper_stub(rq: pd.DataFrame, hyp: pd.DataFrame) -> None:
    """Write paper stub."""
    paper_dir = ROOT / "paper" / "draft"
    paper_dir.mkdir(parents=True, exist_ok=True)
    cols = ["rq", "evidence_item", "metric", "value", "status", "is_posthoc", "paper_safe_claim", "paper_forbidden_claim"]
    hyp_cols = [col for col in ["evidence_layer", "experiment_id", "hypothesis", "status", "is_posthoc", "paper_safe_claim", "paper_forbidden_claim"] if col in hyp.columns]
    text = "# Final RQ Evidence Tables\n\n"
    text += "These tables freeze the Week10 interpretation layer. They summarize evidence boundaries rather than claiming comprehensive success.\n\n"
    text += "## RQ Evidence\n\n" + markdown_table(rq[[col for col in cols if col in rq.columns]]) + "\n\n"
    text += "## Hypothesis Rollup\n\n" + markdown_table(hyp[hyp_cols]) + "\n"
    (paper_dir / "sec5_final_rq_tables.md").write_text(text, encoding="utf-8")


def main() -> None:
    """Command-line entry point."""
    config = load_config(DEFAULT_CONFIG)
    ensure_output_dirs(config)
    rq = build_final_rq_evidence(config)
    hyp = build_hypothesis_rollup(config)
    save_csv(rq, output_path(config, "metrics", "final_rq_evidence_table.csv"))
    save_csv(hyp, output_path(config, "metrics", "final_hypothesis_rollup.csv"))
    save_csv(build_rq1(config), output_path(config, "tables", "table_rq1_main_results.csv"))
    save_csv(build_rq2(config), output_path(config, "tables", "table_rq2_regional_heterogeneity.csv"))
    save_csv(build_rq3(config), output_path(config, "tables", "table_rq3_transfer_boundary.csv"))
    save_csv(build_rq4(config), output_path(config, "tables", "table_rq4_fewshot.csv"))
    write_paper_stub(rq, hyp)
    print(rq[["rq", "evidence_item", "status", "is_posthoc"]].to_string(index=False))


if __name__ == "__main__":
    main()
