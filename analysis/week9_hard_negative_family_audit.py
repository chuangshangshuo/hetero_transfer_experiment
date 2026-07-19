"""Week-9 family hard-negative audit aggregation."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.train.utils import save_dataframe


OUT = ROOT / "output" / "week9"


def main() -> None:
    """Command-line entry point."""
    source = ROOT / "output" / "week85" / "metrics" / "E3_w85_lofo_raw.csv"
    if not source.exists():
        raise FileNotFoundError(source)
    OUT.joinpath("metrics").mkdir(parents=True, exist_ok=True)
    OUT.joinpath("plots").mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(source)
    hard = raw[raw["test_form"] == "hard_illegal_negative"].copy()
    hard["is_posthoc"] = True
    save_dataframe(hard, OUT / "metrics" / "hard_negative_family_raw.csv")
    summary = (
        hard.groupby("family_id", dropna=False)
        .agg(
            num_runs=("seed", "count"),
            hard_negative_auc_mean=("test_roc_auc", "mean"),
            hard_negative_auc_std=("test_roc_auc", lambda x: float(x.std(ddof=0))),
            recall_at_fpr_0_10_mean=("recall_at_fpr_0_10", "mean"),
            score_gap_mean=("score_gap_pos_minus_neg", "mean"),
            heldout_positive_count_mean=("heldout_positive_count", "mean"),
        )
        .reset_index()
        .sort_values("hard_negative_auc_mean", ascending=False)
    )
    save_dataframe(summary, OUT / "metrics" / "hard_negative_family_summary.csv")
    mean_auc = float(hard["test_roc_auc"].mean())
    tsars = summary[summary["family_id"] == "tsars"]
    tsars_auc = float(tsars["hard_negative_auc_mean"].iloc[0]) if not tsars.empty else float("nan")
    acceptance = pd.DataFrame(
        [
            {
                "experiment": "Week9_hard_negative_family",
                "hypothesis": "hard_negative_family_mean_auc",
                "observed_metric": "mean hard-negative family AUC",
                "observed_value": mean_auc,
                "threshold": ">=0.80",
                "status": "pass" if mean_auc >= 0.80 else "fail",
                "is_posthoc": True,
            },
            {
                "experiment": "Week9_hard_negative_family",
                "hypothesis": "hard_negative_tsars_auc",
                "observed_metric": "tsars hard-negative AUC",
                "observed_value": tsars_auc,
                "threshold": ">=0.80",
                "status": "pass" if tsars_auc >= 0.80 else "fail",
                "is_posthoc": True,
            },
        ]
    )
    save_dataframe(acceptance, OUT / "metrics" / "hard_negative_family_acceptance.csv")

    plot_frame = summary.sort_values("hard_negative_auc_mean", ascending=True)
    fig, axis = plt.subplots(figsize=(8, 4.5))
    axis.barh(plot_frame["family_id"], plot_frame["hard_negative_auc_mean"], color="#dc2626", alpha=0.75)
    axis.axvline(0.80, color="#111827", linestyle="--", linewidth=1)
    axis.set_xlim(0.0, 1.02)
    axis.set_xlabel("Hard-negative family AUC")
    axis.set_title("Week 9 Hard-Negative Family Audit")
    fig.tight_layout()
    fig.savefig(OUT / "plots" / "hard_negative_family_auc.png", dpi=170)
    plt.close(fig)
    print(acceptance.to_string(index=False))


if __name__ == "__main__":
    main()
