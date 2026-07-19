"""Visualise deviation cases selected by the Week-10 diagnostics."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_deviation_cases(case_index: pd.DataFrame, output_path: str | Path) -> None:
    """Plot deviation cases."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if case_index.empty:
        return
    fig, axes = plt.subplots(1, len(case_index), figsize=(12, 3.8), sharey=False)
    if len(case_index) == 1:
        axes = [axes]
    for axis, (_, row) in zip(axes, case_index.iterrows()):
        value = float(row["value"])
        reference = float(row.get("reference_value", 0.0))
        axis.bar(["value", "reference"], [value, reference], color=["#4c78a8", "#f58518"])
        axis.set_title(str(row["case_label"]), fontsize=10)
        axis.set_ylabel(str(row["metric"]))
        axis.grid(axis="y", alpha=0.25)
        axis.text(0, value, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
        axis.text(1, reference, f"{reference:.3f}", ha="center", va="bottom", fontsize=8)
    fig.suptitle("Figure 9. Deviation Cases: Shortcut, Recovery, and Boundary Failure")
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)
