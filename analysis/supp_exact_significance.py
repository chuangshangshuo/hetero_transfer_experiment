"""Supplementary exact significance tests at the true n=5 seed resolution.

Motivation (reviewer memo A3): with only 5 paired seed-level deltas, a
B=10,000 paired bootstrap overstates p-value resolution. This script reports
exact finite-sample tests on the same released seed-level data:

- exact sign-flip permutation test on the mean delta (2^5 = 32 assignments;
  the smallest attainable one-sided p is 1/32 ~= 0.031, two-sided 2/32 = 0.0625)
- exact two-sided sign test (binomial on delta signs)
- exact Wilcoxon signed-rank (scipy, exact mode for n<=25)

Families tested, mirroring the released P4 tables:
- W7: 24 method pairs over 4 transfers (results/week7/metrics/transfer_summary.csv)
- W9/P3: k-shot vs 0-shot per target, k in {1,3,5,10}
  (results/week11/metrics/P3_full_fewshot_matrix.csv)

Outputs CSVs next to this repo under reports/supplementary_exact_significance/.
"""
from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "reports" / "supplementary_exact_significance"

LOWER_IS_BETTER = {"T2_ON": True}  # mean_pred_illegal; all other primary metrics are higher-is-better


def exact_signflip_pvalues(deltas: np.ndarray) -> tuple[float, float]:
    """Exact sign-flip permutation p-values (two-sided, one-sided toward observed sign)."""
    d = deltas[np.isfinite(deltas)]
    n = d.size
    if n == 0:
        return float("nan"), float("nan")
    obs = d.mean()
    means = np.array([np.mean(d * np.array(signs)) for signs in itertools.product((1.0, -1.0), repeat=n)])
    eps = 1e-12
    two = float(np.mean(np.abs(means) >= abs(obs) - eps))
    if obs >= 0:
        one = float(np.mean(means >= obs - eps))
    else:
        one = float(np.mean(means <= obs + eps))
    return two, one


def exact_sign_test(deltas: np.ndarray) -> float:
    """Exact sign test."""
    d = deltas[np.isfinite(deltas)]
    nz = d[d != 0]
    if nz.size == 0:
        return 1.0
    k = int((nz > 0).sum())
    return float(binomtest(k, nz.size, 0.5, alternative="two-sided").pvalue)


def exact_wilcoxon(deltas: np.ndarray) -> float:
    """Exact wilcoxon."""
    d = deltas[np.isfinite(deltas)]
    if d.size < 2 or np.allclose(d, 0):
        return 1.0
    try:
        return float(wilcoxon(d, zero_method="zsplit", mode="exact").pvalue)
    except (ValueError, TypeError):
        return float(wilcoxon(d, zero_method="zsplit").pvalue)


def holm(pvals: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Holm."""
    p = np.asarray(pvals, dtype=float)
    m = np.isfinite(p).sum()
    order = np.argsort(np.where(np.isfinite(p), p, np.inf))
    reject = np.zeros(p.size, dtype=bool)
    for rank, idx in enumerate(order):
        if not np.isfinite(p[idx]):
            break
        if p[idx] <= alpha / (m - rank):
            reject[idx] = True
        else:
            break
    return reject


def pair_rows_w7() -> pd.DataFrame:
    """Pair rows Week-7."""
    raw = pd.read_csv(ROOT / "results" / "week7" / "metrics" / "transfer_summary.csv")
    rows = []
    for transfer_id, part in raw.groupby("transfer_id"):
        methods = sorted(part["method"].dropna().unique().tolist())
        for a, b in itertools.combinations(methods, 2):
            fa = part[part["method"].eq(a)].sort_values("seed")
            fb = part[part["method"].eq(b)].sort_values("seed")
            paired = fa[["seed", "primary_metric_value"]].merge(
                fb[["seed", "primary_metric_value"]], on="seed", suffixes=("_a", "_b")
            )
            d = (paired["primary_metric_value_a"] - paired["primary_metric_value_b"]).to_numpy(float)
            two, one = exact_signflip_pvalues(d)
            lower_better = LOWER_IS_BETTER.get(transfer_id, False)
            delta = float(np.nanmean(d))
            winner = (a if ((delta < 0) if lower_better else (delta > 0)) else b) if delta != 0 else "tie"
            rows.append(
                {
                    "comparison_family": "W7_method_pairwise",
                    "transfer_id": transfer_id,
                    "method_a": a,
                    "method_b": b,
                    "n_pairs": int(len(paired)),
                    "mean_delta_a_minus_b": delta,
                    "direction_aware_winner": winner,
                    "perm_p_two_sided": two,
                    "perm_p_one_sided": one,
                    "sign_test_p_two_sided": exact_sign_test(d),
                    "wilcoxon_exact_p": exact_wilcoxon(d),
                }
            )
    frame = pd.DataFrame(rows)
    frame["holm_reject_perm_two_sided"] = holm(frame["perm_p_two_sided"].to_numpy())
    frame["holm_reject_wilcoxon"] = holm(frame["wilcoxon_exact_p"].to_numpy())
    return frame


def pair_rows_fewshot() -> pd.DataFrame:
    """Pair rows few-shot."""
    p3 = pd.read_csv(ROOT / "results" / "week11" / "metrics" / "P3_full_fewshot_matrix.csv")
    rows = []
    for target, part in p3.groupby("target_id"):
        base = part[part["shot"] == 0].sort_values("seed")[["seed", "primary_metric_value"]]
        for shot in (1, 3, 5, 10):
            cur = part[part["shot"] == shot].sort_values("seed")[["seed", "primary_metric_value"]]
            paired = cur.merge(base, on="seed", suffixes=("_shot", "_0"))
            d = (paired["primary_metric_value_shot"] - paired["primary_metric_value_0"]).to_numpy(float)
            two, one = exact_signflip_pvalues(d)
            rows.append(
                {
                    "comparison_family": "P3_fewshot_vs_0shot",
                    "target_id": target,
                    "shot": shot,
                    "n_pairs": int(len(paired)),
                    "mean_delta_vs_0shot": float(np.nanmean(d)),
                    "perm_p_two_sided": two,
                    "perm_p_one_sided": one,
                    "sign_test_p_two_sided": exact_sign_test(d),
                    "wilcoxon_exact_p": exact_wilcoxon(d),
                }
            )
    frame = pd.DataFrame(rows)
    frame["holm_reject_perm_two_sided"] = holm(frame["perm_p_two_sided"].to_numpy())
    frame["holm_reject_wilcoxon"] = holm(frame["wilcoxon_exact_p"].to_numpy())
    return frame


def main() -> None:
    """Command-line entry point."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    w7 = pair_rows_w7()
    fs = pair_rows_fewshot()
    w7.to_csv(OUT_DIR / "supp_exact_w7_method_pairwise.csv", index=False, encoding="utf-8")
    fs.to_csv(OUT_DIR / "supp_exact_fewshot_vs_0shot.csv", index=False, encoding="utf-8")

    print("=== W7 method pairs (24), exact tests at n=5 ===")
    print(f"min attainable perm p: one-sided 1/32={1/32:.4f}, two-sided 2/32={2/32:.4f}")
    print(f"Holm-rejected (perm two-sided): {int(w7['holm_reject_perm_two_sided'].sum())}/{len(w7)}")
    print(f"Holm-rejected (wilcoxon exact): {int(w7['holm_reject_wilcoxon'].sum())}/{len(w7)}")
    at_floor = w7[w7["perm_p_two_sided"] <= 2 / 32 + 1e-9]
    print(f"pairs at the two-sided exact floor (p=0.0625): {len(at_floor)}")
    for _, r in at_floor.iterrows():
        print(
            f"  {r['transfer_id']}: {r['method_a']} vs {r['method_b']} "
            f"delta={r['mean_delta_a_minus_b']:+.3f} winner={r['direction_aware_winner']} "
            f"perm2s={r['perm_p_two_sided']:.4f} perm1s={r['perm_p_one_sided']:.4f}"
        )
    dann_on = w7[(w7["transfer_id"] == "T2_ON") & (w7["method_a"] == "dann") & (w7["method_b"] == "source_only")]
    if len(dann_on):
        r = dann_on.iloc[0]
        print(
            f"headline pair T2_ON dann vs source_only: delta={r['mean_delta_a_minus_b']:+.4f} "
            f"perm2s={r['perm_p_two_sided']:.4f} perm1s={r['perm_p_one_sided']:.4f} "
            f"sign={r['sign_test_p_two_sided']:.4f} wilcoxon={r['wilcoxon_exact_p']:.4f}"
        )

    print()
    print("=== P3 few-shot vs 0-shot (16 tests), exact tests at n=5 ===")
    print(f"Holm-rejected (perm two-sided): {int(fs['holm_reject_perm_two_sided'].sum())}/{len(fs)}")
    show = fs[fs["perm_p_two_sided"] <= 0.0625 + 1e-9]
    print(f"tests at/below two-sided floor: {len(show)}")
    for _, r in show.iterrows():
        print(
            f"  {r['target_id']} {r['shot']}-shot: delta={r['mean_delta_vs_0shot']:+.3f} "
            f"perm2s={r['perm_p_two_sided']:.4f} perm1s={r['perm_p_one_sided']:.4f}"
        )


if __name__ == "__main__":
    main()
