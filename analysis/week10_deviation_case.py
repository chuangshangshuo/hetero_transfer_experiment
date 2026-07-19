"""Week-10 deviation-case selection and export."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.week10_common import DEFAULT_CONFIG, ensure_output_dirs, input_path, load_config, output_path, read_input, save_csv
from src.explain.deviation_case_viz import plot_deviation_cases


def build_case_index(config: dict) -> pd.DataFrame:
    """Build case index."""
    e5 = read_input(config, "week8_patch", "corrected_e5_acceptance")
    few = read_input(config, "week9", "fewshot_stability")
    hard = read_input(config, "week9", "hard_negative_summary")

    cc_drop = e5[e5["hypothesis"].eq("W8P_E5_ccTLD_sensitivity")] if not e5.empty else pd.DataFrame()
    t2_10 = few[
        few["target_id"].eq("T2_PH")
        & few["feature_condition"].eq("full")
        & few["shot"].astype(int).eq(10)
    ] if not few.empty else pd.DataFrame()
    tsars = hard[hard["family_id"].eq("tsars")] if not hard.empty else pd.DataFrame()
    hard_mean = float(pd.to_numeric(hard.get("hard_negative_auc_mean", pd.Series(dtype=float)), errors="coerce").mean()) if not hard.empty else float("nan")

    rows = [
        {
            "case_id": "case_france_cctld_shortcut",
            "case_label": "France ccTLD shortcut",
            "case_type": "regional_shortcut",
            "metric": "T3 full-minus-no-ccTLD AUC",
            "value": float(cc_drop["observed_value"].iloc[0]) if not cc_drop.empty else float("nan"),
            "reference_value": 0.05,
            "evidence_source": str(input_path(config, "week8_patch", "corrected_e5_acceptance").relative_to(ROOT)),
            "interpretation": "T3 France performance is materially affected by ccTLD/local lexical features.",
            "status": "shortcut_boundary",
            "is_posthoc": True,
            "paper_safe_claim": "T3 France is shortcut-sensitive and cannot be used as pure structural-transfer evidence.",
            "paper_forbidden_claim": "Do not claim T3 France proves structural generalization.",
        },
        {
            "case_id": "case_ph_fewshot_recovery",
            "case_label": "Philippines few-shot recovery",
            "case_type": "fewshot_recovery",
            "metric": "T2_PH 10-shot delta AUC",
            "value": float(t2_10["mean_delta_auc"].iloc[0]) if not t2_10.empty else float("nan"),
            "reference_value": 0.05,
            "evidence_source": str(input_path(config, "week9", "fewshot_stability").relative_to(ROOT)),
            "interpretation": "T2_PH weak source-only transfer is recoverable with balanced target support.",
            "status": "recoverable_target",
            "is_posthoc": True,
            "paper_safe_claim": "Few-shot calibration repairs T2_PH in a target-dependent way.",
            "paper_forbidden_claim": "Do not claim few-shot universally improves transfer.",
        },
        {
            "case_id": "case_family_hard_negative_failure",
            "case_label": "Family hard-negative failure",
            "case_type": "family_boundary_failure",
            "metric": "tsars hard-negative AUC",
            "value": float(tsars["hard_negative_auc_mean"].iloc[0]) if not tsars.empty else float("nan"),
            "reference_value": hard_mean,
            "evidence_source": str(input_path(config, "week9", "hard_negative_summary").relative_to(ROOT)),
            "interpretation": "The model does not reliably distinguish held-out illegal families from other illegal families.",
            "status": "family_boundary_failed",
            "is_posthoc": True,
            "paper_safe_claim": "Illegal-vs-licensed detection is easier than illegal-family attribution.",
            "paper_forbidden_claim": "Do not claim illegal-family attribution success.",
        },
    ]
    return pd.DataFrame(rows)


def main() -> None:
    """Command-line entry point."""
    config = load_config(DEFAULT_CONFIG)
    ensure_output_dirs(config)
    frame = build_case_index(config)
    save_csv(frame, output_path(config, "metrics", "deviation_case_index.csv"))
    plot_deviation_cases(frame, output_path(config, "figures", "fig9_deviation_case.pdf"))
    print(frame[["case_id", "metric", "value", "status"]].to_string(index=False))


if __name__ == "__main__":
    main()
