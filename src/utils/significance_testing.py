"""Paired bootstrap p-value and Holm-Bonferroni correction used by the paper."""
from __future__ import annotations

import numpy as np


def paired_bootstrap_pvalue(
    a: np.ndarray,
    b: np.ndarray,
    B: int = 10000,
    seed: int = 42,
) -> dict[str, float]:
    """Paired bootstrap for mean(a-b), with a two-sided sign p-value."""

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    a = a[mask]
    b = b[mask]
    if a.size == 0:
        return {"mean_delta": np.nan, "ci95_low": np.nan, "ci95_high": np.nan, "bootstrap_p": np.nan}
    rng = np.random.default_rng(seed)
    diffs = a - b
    obs = float(diffs.mean())
    boot = np.empty(int(B), dtype=float)
    for i in range(int(B)):
        idx = rng.choice(diffs.size, diffs.size, replace=True)
        boot[i] = diffs[idx].mean()
    ci_low, ci_high = np.percentile(boot, [2.5, 97.5])
    if obs >= 0:
        p = 2.0 * min(float((boot <= 0).mean()), 0.5)
    else:
        p = 2.0 * min(float((boot >= 0).mean()), 0.5)
    return {
        "mean_delta": obs,
        "ci95_low": float(ci_low),
        "ci95_high": float(ci_high),
        "bootstrap_p": float(p),
    }


def holm_bonferroni(p_values: list[float] | np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Return Holm-Bonferroni rejection decisions preserving input order."""

    p = np.asarray(p_values, dtype=float)
    reject = np.zeros(p.shape[0], dtype=bool)
    finite_order = [idx for idx in np.argsort(np.nan_to_num(p, nan=np.inf)) if np.isfinite(p[idx])]
    m = len(finite_order)
    for rank, idx in enumerate(finite_order):
        threshold = float(alpha) / max(1, m - rank)
        if p[idx] <= threshold:
            reject[idx] = True
        else:
            break
    return reject
