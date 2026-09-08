"""Regenerate figure 6(a) with the corrected C4 legend label.

Only the legend text changes: the previous label read "仅图结构(零特征)", which is
inaccurate -- configs/week8.yaml C4_graph_only zeroes the 7 Website feature columns
only and retains all edges plus the 47 feature dimensions carried by IP / Certificate
/ NameServer / Registrar / ExternalReference nodes.

Values come from results/week8/metrics/E5_structure_vs_lexical_summary.csv; the T2_ON
column is converted to 1 - mean_pred_illegal so every panel reads higher-is-better.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(r"D:\hetero_transfer_experiment_release\hetero_transfer_experiment_release")
SUMMARY = REPO / "results" / "week8" / "metrics" / "E5_structure_vs_lexical_summary.csv"
OUT = Path(__file__).resolve().parent / "fig6a_out"

TRANSFERS = [
    ("T3_DiagnoseFrance", "T3_FR(法国)", False),
    ("T2_PH", "T2_PH(菲律宾)", False),
    ("T2_ON", "T2_ON(安大略)", True),   # invert: reported metric is mean_pred_illegal
    ("T1_Nordic", "T1_DK(丹麦)", False),
]
SERIES = [
    ("C1_full", "全配置(图结构+词法)", "#3f3f3f", ""),
    ("C5_lex_only", "仅词法(无图结构)", "#9a9a9a", "//"),
    ("C4_graph_only", "零Website特征(保留全部边)", "#dcdcdc", ".."),
]

plt.rcParams.update({
    "font.family": ["Microsoft YaHei", "SimHei", "sans-serif"],
    "axes.unicode_minus": False,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.size": 7.5,
})


def load() -> dict:
    """Read the E5 summary and return {(transfer, config): (mean, std)} in plot direction."""
    frame = pd.read_csv(SUMMARY)
    out = {}
    for _, r in frame.iterrows():
        tid, cid = r["transfer_id"], r["config_id"]
        mean = float(r["primary_metric_value_mean"])
        std = float(r["primary_metric_value_std"])
        invert = next((inv for t, _, inv in TRANSFERS if t == tid), False)
        out[(tid, cid)] = (1.0 - mean if invert else mean, std)
    return out


def main() -> None:
    """Render figure 6(a) into SVG, PDF, EPS, 600 dpi PNG and 600 dpi TIFF."""
    data = load()
    OUT.mkdir(parents=True, exist_ok=True)
    n_groups = len(TRANSFERS)
    width = 0.26
    x = np.arange(n_groups)

    fig, ax = plt.subplots(figsize=(5.98, 3.05))
    for i, (cid, label, color, hatch) in enumerate(SERIES):
        means = [data[(t, cid)][0] for t, _, _ in TRANSFERS]
        stds = [data[(t, cid)][1] for t, _, _ in TRANSFERS]
        offset = (i - 1) * width
        ax.bar(
            x + offset, means, width, yerr=stds, capsize=2.5,
            label=label, color=color, hatch=hatch,
            edgecolor="black", linewidth=0.7,
            error_kw={"elinewidth": 0.7, "capthick": 0.7, "ecolor": "black"},
        )
        for xi, m in zip(x + offset, means):
            ax.text(xi, m + 0.012, f"{m:.2f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, lbl, _ in TRANSFERS], fontsize=7.5)
    ax.set_ylabel("主指标值", fontsize=8)
    ax.set_xlabel("迁移情境(T2_ON已换算为1−非法概率均值;全部指标越高越好)", fontsize=8)
    ax.set_ylim(0.0, 1.32)
    ax.set_yticks(np.arange(0.0, 1.21, 0.2))
    ax.tick_params(axis="both", labelsize=7.5)
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.5)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
    ax.legend(loc="upper right", fontsize=7.5, frameon=True, edgecolor="black", framealpha=1.0)
    fig.tight_layout()

    stem = "Fig6a_图6a_结构与词法主导性"
    fig.savefig(OUT / f"{stem}.svg", format="svg", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.pdf", format="pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.eps", format="eps", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.png", format="png", dpi=600, bbox_inches="tight")
    plt.close(fig)

    png = OUT / f"{stem}.png"
    try:
        from PIL import Image
        Image.open(png).save(OUT / f"{stem}.tif", format="TIFF", compression="tiff_lzw", dpi=(600, 600))
    except Exception as exc:  # noqa: BLE001 - TIFF is a convenience export
        print("TIFF export skipped:", exc)

    for f in sorted(OUT.iterdir()):
        print(f"  {f.name:<50}{f.stat().st_size:>10,} B")


if __name__ == "__main__":
    main()
