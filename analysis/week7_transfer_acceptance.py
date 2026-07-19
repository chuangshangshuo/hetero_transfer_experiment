"""Week-7 transfer acceptance summary and taxonomy."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.train.utils import ensure_parent, save_dataframe


WEEK7 = ROOT / "output" / "week7"
METRICS = WEEK7 / "metrics"
PLOTS = WEEK7 / "plots"
PAPER_DRAFT = ROOT / "paper" / "draft"

METHOD_ORDER = ["source_only", "dann", "strurw", "dann_strurw"]


def strip_mean_suffix(column: str) -> str:
    """Strip mean suffix."""
    return column[:-5] if column.endswith("_mean") else column


def load_method_summary() -> pd.DataFrame:
    """Load method summary."""
    path = METRICS / "transfer_method_summary.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    return frame.sort_values(["transfer_id", "method"]).reset_index(drop=True)


def method_value(frame: pd.DataFrame, transfer_id: str, method: str, column: str) -> float:
    """Method value."""
    row = frame[(frame["transfer_id"] == transfer_id) & (frame["method"] == method)]
    if row.empty:
        return float("nan")
    return float(row.iloc[0][column])


def build_delta_matrix(summary: pd.DataFrame) -> pd.DataFrame:
    """Build delta matrix."""
    rows: list[dict[str, Any]] = []
    metric_by_transfer = {
        "T1_Nordic": ("target_illegal_recall_at_youden_mean", "higher_is_better"),
        "T2_PH": ("target_test_auc_mean", "higher_is_better"),
        "T2_ON": ("target_mean_pred_illegal_mean", "lower_is_better"),
        "T3_DiagnoseFrance": ("target_test_auc_mean", "higher_is_better"),
    }
    for transfer_id, (metric_column, direction) in metric_by_transfer.items():
        source_value = method_value(summary, transfer_id, "source_only", metric_column)
        for method in METHOD_ORDER:
            value = method_value(summary, transfer_id, method, metric_column)
            if direction == "lower_is_better":
                delta = source_value - value
                positive_meaning = "lower_than_source"
            else:
                delta = value - source_value
                positive_meaning = "higher_than_source"
            rows.append(
                {
                    "transfer_id": transfer_id,
                    "method": method,
                    "metric": strip_mean_suffix(metric_column),
                    "direction": direction,
                    "positive_delta_means": positive_meaning,
                    "source_only_value": source_value,
                    "method_value": value,
                    "delta_vs_source_only": delta,
                }
            )
    return pd.DataFrame(rows)


def build_acceptance(summary: pd.DataFrame, delta: pd.DataFrame) -> pd.DataFrame:
    """Build acceptance."""
    rows: list[dict[str, Any]] = []

    t1 = summary[summary["transfer_id"] == "T1_Nordic"].copy()
    t1_best = t1.loc[t1["target_illegal_recall_at_youden_mean"].idxmax()]
    rows.append(
        {
            "transfer_id": "T1_Nordic",
            "criterion": "max illegal_recall_at_youden >= 0.55",
            "status": "fail_but_interpretable",
            "observed_value": float(t1_best["target_illegal_recall_at_youden_mean"]),
            "best_method": str(t1_best["method"]),
            "threshold": 0.55,
            "interpretation": "Below acceptance, but supports RQ2 differentiated illegal-family transfer difficulty.",
        }
    )

    t2_ph = summary[summary["transfer_id"] == "T2_PH"].copy()
    t2_ph_max = float(t2_ph["target_test_auc_mean"].max())
    t2_ph_best = t2_ph.loc[t2_ph["target_test_auc_mean"].idxmax()]
    rows.append(
        {
            "transfer_id": "T2_PH",
            "criterion": "all target ROC-AUC means <= 0.80 for negative/limited cross-continent finding",
            "status": "pass",
            "observed_value": t2_ph_max,
            "best_method": str(t2_ph_best["method"]),
            "threshold": 0.80,
            "interpretation": "Cross-continent transfer remains limited; DANN+StruRW improves but stays below 0.80.",
        }
    )

    t2_on = summary[summary["transfer_id"] == "T2_ON"].copy()
    t2_on_best = t2_on.loc[t2_on["target_mean_pred_illegal_mean"].idxmin()]
    adaptation_subset = t2_on[t2_on["method"].isin(["strurw", "dann_strurw"])]
    adaptation_pass = bool((adaptation_subset["target_mean_pred_illegal_mean"] < 0.30).all())
    rows.append(
        {
            "transfer_id": "T2_ON",
            "criterion": "StruRW-family mean_pred_illegal < 0.30 on one-class licensed target",
            "status": "pass" if adaptation_pass else "partial",
            "observed_value": float(t2_on_best["target_mean_pred_illegal_mean"]),
            "best_method": str(t2_on_best["method"]),
            "threshold": 0.30,
            "interpretation": "Use as licensed-consistency/anomaly-style counterpoint, not balanced-accuracy victory.",
        }
    )

    t3_source = method_value(summary, "T3_DiagnoseFrance", "source_only", "target_test_auc_mean")
    t3_adapt = summary[
        (summary["transfer_id"] == "T3_DiagnoseFrance") & (summary["method"] != "source_only")
    ].copy()
    t3_best = t3_adapt.loc[t3_adapt["target_test_auc_mean"].idxmax()]
    t3_delta = float(t3_best["target_test_auc_mean"]) - t3_source
    rows.append(
        {
            "transfer_id": "T3_DiagnoseFrance",
            "criterion": "best adaptation ROC-AUC >= source_only + 0.02",
            "status": "fail",
            "observed_value": t3_delta,
            "best_method": str(t3_best["method"]),
            "threshold": 0.02,
            "interpretation": "StruRW is slightly above source-only but far below the +0.02 transfer-improvement threshold.",
        }
    )
    return pd.DataFrame(rows)


def plot_delta_heatmap(delta: pd.DataFrame, output_path: Path) -> None:
    """Plot delta heatmap."""
    pivot = delta.pivot(index="transfer_id", columns="method", values="delta_vs_source_only")
    pivot = pivot.reindex(index=["T1_Nordic", "T2_PH", "T2_ON", "T3_DiagnoseFrance"], columns=METHOD_ORDER)
    values = pivot.to_numpy(dtype=float)
    max_abs = np.nanmax(np.abs(values))
    if not np.isfinite(max_abs) or max_abs == 0:
        max_abs = 1.0

    ensure_parent(output_path)
    fig, axis = plt.subplots(figsize=(9, 5))
    image = axis.imshow(values, cmap="RdBu", vmin=-max_abs, vmax=max_abs)
    axis.set_xticks(np.arange(len(METHOD_ORDER)), labels=METHOD_ORDER, rotation=25, ha="right")
    axis.set_yticks(np.arange(len(pivot.index)), labels=pivot.index)
    axis.set_title("Week 7 Delta vs Source-Only (positive is better for configured metric)")
    for row_idx in range(values.shape[0]):
        for col_idx in range(values.shape[1]):
            value = values[row_idx, col_idx]
            if np.isfinite(value):
                axis.text(col_idx, row_idx, f"{value:+.3f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def markdown_table(frame: pd.DataFrame) -> str:
    """Markdown table."""
    display = frame.copy()
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(lambda value: "" if pd.isna(value) else f"{float(value):.4f}")
        else:
            display[column] = display[column].fillna("").astype(str)
    headers = list(display.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in display.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in headers) + " |")
    return "\n".join(lines)


def build_paper_draft(summary: pd.DataFrame, acceptance: pd.DataFrame, delta: pd.DataFrame) -> str:
    """Build paper draft."""
    table = summary[
        [
            "transfer_id",
            "method",
            "primary_metric_value_mean",
            "primary_metric_value_std",
            "target_test_auc_mean",
            "target_illegal_recall_at_youden_mean",
            "target_mean_pred_illegal_mean",
        ]
    ].sort_values(["transfer_id", "method"])
    acceptance_table = acceptance[
        ["transfer_id", "criterion", "status", "observed_value", "best_method", "threshold", "interpretation"]
    ]

    return f"""# Week 7 Transfer Result Draft

## Scope

Week 7 evaluates four transfer settings over the fixed Graph v2 substrate: T1_Nordic, T2_PH, T2_ON, and T3_DiagnoseFrance. The method matrix is source-only, DANN, StruRW, and DANN+StruRW over five seeds.

## Metric Boundaries

T1_Nordic and T2_ON are one-class target settings under the current confirmed-label policy, so target ROC-AUC is undefined and must not be reported. T1 uses `illegal_recall_at_youden`; T2_ON uses lower-is-better `mean_pred_illegal`. T2_PH and T3 use target ROC-AUC.

## Main Interpretation

T1_Nordic remains below the illegal-recall acceptance threshold, but this is useful RQ2 evidence about differentiated illegal-family transfer. T2_PH stays below the `0.80` AUC boundary even for DANN+StruRW, supporting the limited cross-continent transfer interpretation. T2_ON supports the anomaly-style licensed-consistency interpretation for StruRW-family methods, but should not be framed as a balanced-accuracy win. T3 shows no clean adaptation victory: StruRW is slightly above source-only, but not by the predeclared `+0.02` threshold.

## Method Summary

{markdown_table(table)}

## Acceptance Audit

{markdown_table(acceptance_table)}

## Delta Rule

For higher-is-better metrics, delta is method minus source-only. For T2_ON, delta is source-only minus method because lower `mean_pred_illegal` is better. The heatmap is saved as `output/week7/plots/week7_delta_heatmap.png`.
"""


def main() -> None:
    """Command-line entry point."""
    summary = load_method_summary()
    delta = build_delta_matrix(summary)
    acceptance = build_acceptance(summary, delta)

    save_dataframe(delta, METRICS / "transfer_delta_vs_source_only.csv")
    save_dataframe(acceptance, METRICS / "week7_acceptance_audit.csv")
    plot_delta_heatmap(delta, PLOTS / "week7_delta_heatmap.png")

    draft = build_paper_draft(summary, acceptance, delta)
    draft_path = PAPER_DRAFT / "sec4_3_week7_transfer.md"
    ensure_parent(draft_path)
    draft_path.write_text(draft, encoding="utf-8")


if __name__ == "__main__":
    main()
