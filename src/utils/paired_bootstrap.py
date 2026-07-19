"""Paired bootstrap over same-seed delta vectors (CI and sign p-values)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BootstrapResult:
    """Immutable result of a paired bootstrap run."""
    mean: float
    ci_lo: float
    ci_hi: float
    p_value_two_sided: float
    p_value_positive: float
    p_value_negative: float
    n: int
    B: int


def _clean(values: np.ndarray | list[float]) -> np.ndarray:
    """Drop non-finite entries and return a float64 array."""
    array = np.asarray(values, dtype=np.float64)
    return array[np.isfinite(array)]


def paired_bootstrap(
    paired_deltas: np.ndarray | list[float],
    B: int = 1000,
    seed: int = 42,
    ci: float = 0.95,
) -> BootstrapResult:
    """Bootstrap a same-seed paired delta vector.

    Positive deltas should be defined by the caller so that "larger is stronger
    evidence" for the hypothesis being tested.
    """
    deltas = _clean(paired_deltas)
    n = int(deltas.shape[0])
    if n == 0:
        nan = float("nan")
        return BootstrapResult(nan, nan, nan, nan, nan, nan, 0, int(B))

    rng = np.random.default_rng(seed)
    samples = np.empty(int(B), dtype=np.float64)
    for idx in range(int(B)):
        draw = rng.integers(0, n, size=n)
        samples[idx] = float(deltas[draw].mean())

    alpha = (1.0 - float(ci)) / 2.0
    ci_lo = float(np.quantile(samples, alpha))
    ci_hi = float(np.quantile(samples, 1.0 - alpha))
    p_positive = float(np.mean(samples <= 0.0))
    p_negative = float(np.mean(samples >= 0.0))
    p_two = float(min(1.0, 2.0 * min(p_positive, p_negative)))
    return BootstrapResult(
        mean=float(deltas.mean()),
        ci_lo=ci_lo,
        ci_hi=ci_hi,
        p_value_two_sided=p_two,
        p_value_positive=p_positive,
        p_value_negative=p_negative,
        n=n,
        B=int(B),
    )


def paired_bootstrap_ci(
    paired_deltas: np.ndarray | list[float],
    B: int = 1000,
    seed: int = 42,
    ci: float = 0.95,
) -> tuple[float, float]:
    """Convenience wrapper returning only the bootstrap confidence interval."""
    result = paired_bootstrap(paired_deltas, B=B, seed=seed, ci=ci)
    return result.ci_lo, result.ci_hi
