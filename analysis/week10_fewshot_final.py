from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.week10_common import DEFAULT_CONFIG, ensure_output_dirs, input_path, load_config, output_path, read_input, save_csv


def build_fewshot_final(config: dict) -> pd.DataFrame:
    stability = read_input(config, "week9", "fewshot_stability")
    rollup = read_input(config, "week9", "fewshot_rollup")
    if stability.empty:
        return pd.DataFrame()
    rows = []
    for _, row in stability.iterrows():
        target_id = row["target_id"]
        claim = (
            "Few-shot calibration substantially improves the hard T2_PH target."
            if target_id == "T2_PH" and float(row["mean_delta_auc"]) > 0.10 and float(row["success_rate_delta_ge_005"]) >= 0.8
            else "Few-shot provides little additional benefit for saturated T3 France."
            if target_id == "T3_DiagnoseFrance"
            else "Few-shot calibration remains target-dependent."
        )
        forbidden = (
            "Do not claim few-shot universally improves transfer."
            if target_id == "T2_PH"
            else "Do not write T3 France as a few-shot success or structural proof."
        )
        rows.append(
            {
                "target_id": target_id,
                "shot": int(row["shot"]),
                "feature_condition": row["feature_condition"],
                "training_mode": row["training_mode"],
                "mean_auc": float(row["mean_auc"]),
                "std_auc": float(row["std_auc"]),
                "mean_delta_auc": float(row["mean_delta_auc"]),
                "success_rate_delta_ge_005": float(row["success_rate_delta_ge_005"]),
                "stability_label": row["stability_label"],
                "status": "posthoc_support" if target_id == "T2_PH" and float(row["mean_delta_auc"]) > 0.10 else "no_clear_gain" if target_id == "T3_DiagnoseFrance" else "diagnostic_context",
                "is_posthoc": True,
                "paper_safe_claim": claim,
                "paper_forbidden_claim": forbidden,
                "source_file": str(input_path(config, "week9", "fewshot_stability").relative_to(ROOT)),
            }
        )
    frame = pd.DataFrame(rows)
    if not rollup.empty:
        frame = frame.merge(
            rollup[["target_id", "best_shot", "best_mean_auc", "best_mean_delta_auc", "final_status"]],
            on="target_id",
            how="left",
        )
    return frame


def plot_fewshot(frame: pd.DataFrame, output: Path) -> None:
    if frame.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for axis, target_id in zip(axes, ["T2_PH", "T3_DiagnoseFrance"]):
        part = frame[(frame["target_id"].eq(target_id)) & (frame["feature_condition"].isin(["full", "no-ccTLD", "no-website-lexical", "graph-only"]))].copy()
        for condition, cond_frame in part.groupby("feature_condition"):
            cond_frame = cond_frame.sort_values("shot")
            axis.errorbar(cond_frame["shot"], cond_frame["mean_auc"], yerr=cond_frame["std_auc"], marker="o", capsize=3, label=condition)
        axis.set_title(target_id)
        axis.set_xlabel("shots per class")
        axis.set_ylabel("ROC-AUC")
        axis.set_ylim(0, 1.05)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7)
    fig.suptitle("Figure 8. Few-shot Target Calibration Curves")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)


def main() -> None:
    config = load_config(DEFAULT_CONFIG)
    ensure_output_dirs(config)
    frame = build_fewshot_final(config)
    save_csv(frame, output_path(config, "metrics", "fewshot_final_summary.csv"))
    save_csv(frame, output_path(config, "tables", "table_rq4_fewshot.csv"))
    plot_fewshot(frame, output_path(config, "figures", "fig8_fewshot_curves.pdf"))
    print(frame[["target_id", "shot", "feature_condition", "mean_auc", "mean_delta_auc", "status"]].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
