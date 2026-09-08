"""Regenerate figure 5 with the corrected T3_FR source-domain label.

Only the T3_FR panel title changes: it previously read "瑞典+意大利→法国" while
configs/week7_transfer.yaml sets T3_DiagnoseFrance.source_regions = [Spain, Italy].
Curves, shaded ±1 SD bands and thresholds are recomputed from
results/week11/metrics/P3_full_fewshot_matrix.csv (feature_condition == "full").
"""
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(r"D:\hetero_transfer_experiment_release\hetero_transfer_experiment_release")
MATRIX = REPO / "results" / "week11" / "metrics" / "P3_full_fewshot_matrix.csv"
OUT = Path(__file__).resolve().parent / "fig5_out"
SHOTS = [0, 1, 3, 5, 10]

PANELS = [
    ("T2_PH", "T2_PH:8国合并→菲律宾", "AUC", 0.80, "验收阈值0.80"),
    ("T3_DiagnoseFrance", "T3_FR:西班牙+意大利→法国", "AUC", None, None),
    ("T1_Nordic", "T1_DK:瑞典+西班牙+意大利→丹麦", "约登阈值下非法召回率", 0.50, "预声明阈值0.50"),
    ("T2_ON", "T2_ON:8国合并→安大略", "1−非法概率均值", 0.70, "验收阈值0.70"),
]

plt.rcParams.update({
    "font.family": ["Microsoft YaHei", "SimHei", "sans-serif"],
    "axes.unicode_minus": False,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.size": 7,
})


def load() -> dict:
    """Return {(target_id, shot): (mean, sd)} in the plotted (higher-is-better) direction."""
    frame = pd.read_csv(MATRIX)
    frame = frame[frame["feature_condition"] == "full"]
    buckets = defaultdict(list)
    for _, r in frame.iterrows():
        value = 1.0 - float(r["target_mean_pred_illegal"]) if r["target_id"] == "T2_ON" else float(r["primary_metric_value"])
        buckets[(r["target_id"], int(r["shot"]))].append(value)
    return {k: (statistics.mean(v), statistics.stdev(v) if len(v) > 1 else 0.0) for k, v in buckets.items()}


def main() -> None:
    """Render figure 5 into SVG, PDF, EPS, 600 dpi PNG and 600 dpi TIFF."""
    data = load()
    OUT.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(6.85, 4.96))
    x = np.arange(len(SHOTS))

    for axis, (target, title, ylabel, threshold, threshold_label) in zip(axes.flatten(), PANELS):
        means = np.array([data[(target, s)][0] for s in SHOTS])
        sds = np.array([data[(target, s)][1] for s in SHOTS])
        axis.fill_between(x, means - sds, means + sds, color="#c7c7c7", alpha=0.85, linewidth=0)
        axis.plot(x, means, color="black", linewidth=1.2, marker="o", markersize=4.5,
                  markerfacecolor="white", markeredgecolor="black", markeredgewidth=1.0, zorder=3)
        for xi, m in zip(x, means):
            axis.annotate(f"{m:.3f}", (xi, m), textcoords="offset points", xytext=(0, 7),
                          ha="center", fontsize=6.8, zorder=4)
        if threshold is not None:
            axis.axhline(threshold, color="black", linestyle="--", linewidth=0.9)
            axis.text(len(SHOTS) - 1, threshold, threshold_label, fontsize=6.8,
                      ha="right", va="bottom")
        axis.set_title(title, fontsize=8)
        axis.set_xticks(x)
        axis.set_xticklabels([str(s) for s in SHOTS], fontsize=8)
        axis.set_xlabel("目标域校准样本数(shot)", fontsize=8)
        axis.set_ylabel(ylabel, fontsize=8)
        axis.grid(True, linestyle=":", linewidth=0.5, alpha=0.5)
        axis.set_axisbelow(True)
        axis.tick_params(labelsize=7)
        for spine in axis.spines.values():
            spine.set_linewidth(0.8)

    fig.tight_layout()
    stem = "Fig5_图5_少样本校准曲线"
    fig.savefig(OUT / f"{stem}.svg", format="svg", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.pdf", format="pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.eps", format="eps", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.png", format="png", dpi=600, bbox_inches="tight")
    plt.close(fig)

    from PIL import Image
    im = Image.open(OUT / f"{stem}.png").convert("RGB")
    im.save(OUT / f"{stem}.tif", format="TIFF", compression="tiff_lzw", dpi=(600, 600))

    for f in sorted(OUT.iterdir()):
        print(f"  {f.name:<44}{f.stat().st_size:>12,} B")


if __name__ == "__main__":
    main()
