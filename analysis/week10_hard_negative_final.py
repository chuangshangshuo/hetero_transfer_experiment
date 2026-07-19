"""Week-10 final family hard-negative summary."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.week10_common import DEFAULT_CONFIG, ensure_output_dirs, input_path, load_config, output_path, read_input, save_csv


def build_hard_negative_final(config: dict) -> pd.DataFrame:
    """Build hard negative final."""
    summary = read_input(config, "week9", "hard_negative_summary")
    raw = read_input(config, "week9", "hard_negative_raw")
    if summary.empty:
        return pd.DataFrame()
    rows = []
    for _, row in summary.iterrows():
        auc = float(row["hard_negative_auc_mean"])
        status = "hard_negative_pass" if auc >= 0.80 else "hard_negative_fail"
        rows.append(
            {
                "family_id": row["family_id"],
                "num_runs": int(row["num_runs"]),
                "hard_negative_auc_mean": auc,
                "hard_negative_auc_std": float(row["hard_negative_auc_std"]),
                "recall_at_fpr_0_10_mean": float(row["recall_at_fpr_0_10_mean"]),
                "score_gap_mean": float(row["score_gap_mean"]),
                "heldout_positive_count_mean": float(row["heldout_positive_count_mean"]),
                "status": status,
                "is_posthoc": True,
                "paper_safe_claim": "The model separates illegal from licensed controls, but hard illegal-vs-illegal family discrimination is not stable.",
                "paper_forbidden_claim": "Do not claim illegal-family attribution success or robust unseen-family recognition.",
                "source_file": str(input_path(config, "week9", "hard_negative_summary").relative_to(ROOT)),
            }
        )
    frame = pd.DataFrame(rows).sort_values("hard_negative_auc_mean", ascending=False).reset_index(drop=True)
    frame["overall_mean_auc"] = frame["hard_negative_auc_mean"].mean()
    frame["overall_status"] = "family_boundary_failed" if frame["overall_mean_auc"].iloc[0] < 0.80 else "family_boundary_supported"
    if not raw.empty:
        frame["raw_num_rows"] = len(raw)
    return frame


def plot_hard_negative(frame: pd.DataFrame, output: Path) -> None:
    """Plot hard negative."""
    if frame.empty:
        return
    fig, axis = plt.subplots(figsize=(8.5, 4.6))
    colors = ["#4c78a8" if value >= 0.80 else "#e45756" for value in frame["hard_negative_auc_mean"]]
    axis.bar(frame["family_id"], frame["hard_negative_auc_mean"], yerr=frame["hard_negative_auc_std"], color=colors, capsize=3)
    axis.axhline(0.80, color="black", linestyle="--", linewidth=1, label="family-recognition threshold")
    axis.axhline(float(frame["overall_mean_auc"].iloc[0]), color="#f58518", linestyle=":", linewidth=1.5, label="mean")
    axis.set_ylabel("Hard-negative AUC")
    axis.set_xlabel("held-out illegal family")
    axis.set_ylim(0, 1.05)
    axis.set_title("Figure 10. Hard Illegal-Negative Family Boundary")
    axis.tick_params(axis="x", rotation=30)
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)


def main() -> None:
    """Command-line entry point."""
    config = load_config(DEFAULT_CONFIG)
    ensure_output_dirs(config)
    frame = build_hard_negative_final(config)
    save_csv(frame, output_path(config, "metrics", "hard_negative_family_final_summary.csv"))
    save_csv(frame, output_path(config, "tables", "table_family_boundary.csv"))
    plot_hard_negative(frame, output_path(config, "figures", "fig10_hard_negative_family.pdf"))
    print(frame[["family_id", "hard_negative_auc_mean", "status", "overall_status"]].to_string(index=False))


if __name__ == "__main__":
    main()
