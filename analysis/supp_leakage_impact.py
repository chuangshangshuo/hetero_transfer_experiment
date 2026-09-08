"""M11 impact bound: recompute every transfer metric on the leak-free subset of target_test.

The temperature that produced the shared HeCo encoder was chosen on the pooled-primary validation
split. Some target_test samples of each transfer scenario also appear in that split. This script
recomputes each run's primary metric after removing exactly those samples, so the effect of the
overlap on the reported numbers can be bounded instead of argued about.
"""
# NOTE: this script was run against the controlled research workspace, whose sample-level
# split files and model checkpoints are not part of the public release. WS and the output
# directory below are the paths used for the recorded run; set them to your own copies to
# re-execute. Only aggregate outputs are published, under reports/rerun_2026-09-08/.

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

WS = Path(r"D:\codex\hetero_transfer_experiment")
SCRATCH = Path(r"<scratch_dir>")
RERUN = SCRATCH / "rerun_out"
PRED = RERUN / "predictions"
OUT = SCRATCH / "m11_out"

DIRECTION = {
    "T1_Nordic": "illegal_recall_at_youden",
    "T2_ON": "mean_pred_illegal",
    "T2_PH": "roc_auc",
    "T3_DiagnoseFrance": "roc_auc",
}


def primary(kind: str, labels: np.ndarray, scores: np.ndarray, threshold: float) -> float:
    """Compute the scenario's primary metric on an arbitrary subset."""
    if kind == "roc_auc":
        return float(roc_auc_score(labels, scores)) if np.unique(labels).size == 2 else float("nan")
    if kind == "mean_pred_illegal":
        return float(scores.mean())
    illegal = scores[labels == 1]
    return float((illegal >= threshold).mean()) if illegal.size else float("nan")


def main() -> None:
    """Write the full-vs-isolated comparison table and print the summary."""
    OUT.mkdir(parents=True, exist_ok=True)
    sel = pd.read_csv(WS / "output" / "week5" / "splits" / "heco_pooled_primary__seed42.csv",
                      usecols=["node_id", "split"])
    sel_val = set(sel.loc[sel["split"] == "val", "node_id"].astype(str))

    summary = pd.read_csv(RERUN / "rerun_transfer_summary.csv")
    rows = []
    for _, r in summary.iterrows():
        tid, method, seed = r["transfer_id"], r["method"], int(r["seed"])
        f = PRED / f"{tid}__{method}__seed{seed}.csv"
        if not f.exists():
            continue
        pred = pd.read_csv(f)
        test = pred[pred["split"] == "target_test"].copy()
        test["node_id"] = test["node_id"].astype(str)
        kind = DIRECTION[tid]
        thr = float(r["threshold"]) if pd.notna(r.get("threshold")) else 0.5

        full = primary(kind, test["label"].to_numpy(int), test["pred_prob_illegal"].to_numpy(float), thr)
        keep = ~test["node_id"].isin(sel_val)
        iso = test[keep]
        iso_v = primary(kind, iso["label"].to_numpy(int), iso["pred_prob_illegal"].to_numpy(float), thr)
        rows.append({
            "transfer_id": tid, "method": method, "seed": seed, "metric": kind,
            "n_target_test": len(test), "n_removed_overlap": int((~keep).sum()), "n_isolated": len(iso),
            "primary_full": full, "primary_isolated": iso_v, "delta": iso_v - full,
            "rerun_primary_reported": float(r["primary_metric_value"]),
            "is_posthoc": True,
        })

    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "m11_leakage_impact.csv", index=False, encoding="utf-8")
    print(f"写出 {OUT / 'm11_leakage_impact.csv'} （{len(frame)} 行）\n")

    chk = (frame["primary_full"] - frame["rerun_primary_reported"]).abs().max()
    print(f"自检：按预测重算的 primary 与运行时记录的最大差 = {chk:.2e}\n")

    print("按情境×方法（5 种子均值）：完整 target_test vs 剔除温度选优验证集重叠后")
    g = frame.groupby(["transfer_id", "method"]).agg(
        removed=("n_removed_overlap", "mean"), n_iso=("n_isolated", "mean"),
        full=("primary_full", "mean"), iso=("primary_isolated", "mean"), delta=("delta", "mean"),
    ).reset_index()
    for _, r in g.iterrows():
        print(f"  {r['transfer_id']:<20}{r['method']:<14}剔除{r['removed']:.1f}条 "
              f"完整={r['full']:.4f}  隔离={r['iso']:.4f}  差={r['delta']:+.4f}")

    print(f"\n全部 {len(frame)} 次运行：|差| 最大={frame['delta'].abs().max():.4f} "
          f"平均={frame['delta'].abs().mean():.4f} 中位={frame['delta'].abs().median():.4f}")

    # does the direction-corrected delta-vs-source_only table change?
    print("\n方法相对 source_only 的差值（按全文方向换算），完整 vs 隔离：")
    piv = frame.pivot_table(index=["transfer_id", "seed"], columns="method",
                            values=["primary_full", "primary_isolated"])
    out = []
    for tid in DIRECTION:
        sign = -1.0 if tid == "T2_ON" else 1.0
        for m in ["dann", "strurw", "dann_strurw"]:
            for col, name in [("primary_full", "完整"), ("primary_isolated", "隔离")]:
                try:
                    d = sign * (piv[(col, m)] - piv[(col, "source_only")]).xs(tid).mean()
                except KeyError:
                    continue
                out.append({"transfer_id": tid, "method": m, "which": name, "delta_vs_source_only": d})
    od = pd.DataFrame(out).pivot_table(index=["transfer_id", "method"], columns="which",
                                       values="delta_vs_source_only").reset_index()
    od["变化"] = od["隔离"] - od["完整"]
    od["完整>+0.05"] = od["完整"] > 0.05
    od["隔离>+0.05"] = od["隔离"] > 0.05
    print(od.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
    od.to_csv(OUT / "m11_delta_table_full_vs_isolated.csv", index=False, encoding="utf-8")
    n_full = int(od["完整>+0.05"].sum())
    n_iso = int(od["隔离>+0.05"].sum())
    print(f"\n超过 +0.05 的方法对：完整 {n_full} 项 → 隔离 {n_iso} 项")


if __name__ == "__main__":
    main()
