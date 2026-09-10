"""Regenerate figure 6(b) without the retracted significance labels.

The previous render annotated each bar with p<0.001 / p=0.028. Section 4.4 of the manuscript
withdraws those bootstrap p-values as an invalid significance procedure (percentile inversion
collapses to zero whenever the five seed-level differences share a sign), so the figure must not
keep asserting significance on its own. The annotations are replaced by the per-seed sign counts
and 95% bootstrap intervals, which are the descriptive evidence the text actually relies on, and
the axis label now states the effect definition (full minus ablated) explicitly.
"""
import collections
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(r"D:\hetero_transfer_experiment_release\hetero_transfer_experiment_release")
SUMMARY = REPO / "results" / "week8" / "metrics" / "E1_edge_ablation_summary.csv"
RAW = REPO / "results" / "week8" / "metrics" / "E1_edge_ablation_raw_runs.csv"
OUT = Path(__file__).resolve().parent / "fig6b_out"
STEM = "Fig6b_图6b_边通道消融效应"

# rows in the order the previous figure used (most positive effect at the top)
ROWS = [
    ("T3_DiagnoseFrance", "uses_cert", "T3_FR(法国)"),
    ("T2_PH", "uses_cert", "T2_PH(菲律宾)"),
    ("T2_ON", "hosted_on", "T2_ON(安大略)"),
    ("T2_ON", "registered_via", "T2_ON(安大略)"),
    ("T1_Nordic", "registered_via", "T1_DK(丹麦)"),
    ("T2_PH", "registered_via", "T2_PH(菲律宾)"),
]
DIRECTION = {"T1_Nordic": 1, "T2_PH": 1, "T3_DiagnoseFrance": 1, "T2_ON": -1}

plt.rcParams.update({
    "font.family": ["Microsoft YaHei", "SimHei", "sans-serif"],
    "axes.unicode_minus": False,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.size": 7,
})


def seed_signs() -> dict:
    """Return {(transfer, edge): (n_pos, n_neg, n_zero)} for effect = full - ablated."""
    raw = pd.read_csv(RAW)
    base, abl = {}, collections.defaultdict(dict)
    for _, r in raw.iterrows():
        key = (r["transfer_id"], int(r["seed"]))
        value = float(r["primary_metric_value"])
        if r["edge_removed"] == "none":
            base[key] = value
        else:
            abl[(r["transfer_id"], r["edge_removed"])][int(r["seed"])] = value
    out = {}
    for (tid, edge), per_seed in abl.items():
        diffs = [DIRECTION[tid] * (base[(tid, s)] - per_seed[s]) for s in sorted(per_seed)]
        pos = sum(1 for d in diffs if d > 1e-12)
        neg = sum(1 for d in diffs if d < -1e-12)
        out[(tid, edge)] = (pos, neg, len(diffs) - pos - neg)
    return out


def main() -> None:
    """Render figure 6(b) into SVG, PDF, EPS, 600 dpi PNG and 600 dpi TIFF."""
    summary = pd.read_csv(SUMMARY).set_index(["transfer_id", "edge_removed"])
    signs = seed_signs()
    OUT.mkdir(parents=True, exist_ok=True)

    labels, effects, lo, hi, notes = [], [], [], [], []
    for tid, edge, disp in ROWS:
        row = summary.loc[(tid, edge)]
        eff = float(row["effect_mean"])
        effects.append(eff)
        lo.append(eff - float(row["effect_ci_lo"]))
        hi.append(float(row["effect_ci_hi"]) - eff)
        labels.append(f"{disp}\n移除 {edge}")
        pos, neg, zero = signs[(tid, edge)]
        parts = []
        if pos:
            parts.append(f"{pos}正")
        if neg:
            parts.append(f"{neg}负")
        if zero:
            parts.append(f"{zero}零差")
        notes.append("/".join(parts))

    y = np.arange(len(ROWS))[::-1]
    fig, ax = plt.subplots(figsize=(6.81, 3.06))
    for i, (yi, eff) in enumerate(zip(y, effects)):
        positive = eff > 0
        ax.barh(yi, eff, height=0.62,
                color="#3f3f3f" if positive else "#c9c9c9",
                hatch="" if positive else "//",
                edgecolor="black", linewidth=0.8,
                xerr=[[lo[i]], [hi[i]]], capsize=2.5,
                error_kw={"elinewidth": 0.7, "capthick": 0.7, "ecolor": "black"})
        pad = 0.004 if positive else -0.004
        ax.text(eff + (hi[i] if positive else -lo[i]) + pad, yi, notes[i],
                va="center", ha="left" if positive else "right", fontsize=7)

    ax.axvline(0.0, color="black", linewidth=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.set_ylabel("情境×边类型", fontsize=8)
    ax.set_xlabel("效应值 = 全配置 − 移除该边类型后（按各情境指标方向换算；>0该边有正贡献，<0为负迁移）",
                  fontsize=7.5)
    ax.set_xlim(-0.21, 0.21)
    ax.grid(axis="x", linestyle=":", linewidth=0.5, alpha=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=7)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
    ax.set_title("误差线为95%自助置信区间；标注为5个随机种子的效应符号计数（不作显著性主张）",
                 fontsize=7.5, pad=6)
    fig.tight_layout()

    fig.savefig(OUT / f"{STEM}.svg", format="svg", bbox_inches="tight")
    fig.savefig(OUT / f"{STEM}.pdf", format="pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{STEM}.eps", format="eps", bbox_inches="tight")
    fig.savefig(OUT / f"{STEM}.png", format="png", dpi=600, bbox_inches="tight")
    plt.close(fig)

    from PIL import Image
    Image.open(OUT / f"{STEM}.png").convert("RGB").save(
        OUT / f"{STEM}.tif", format="TIFF", compression="tiff_lzw", dpi=(600, 600))

    for f in sorted(OUT.iterdir()):
        print(f"  {f.name:<44}{f.stat().st_size:>12,} B")
    print("\n逐行标注：")
    for (tid, edge, disp), eff, note in zip(ROWS, effects, notes):
        print(f"  {disp:<14}移除 {edge:<15}效应={eff:+.4f}  {note}")


if __name__ == "__main__":
    main()
