"""Unit tests for src/utils/paired_bootstrap.py."""
from __future__ import annotations

import numpy as np

from src.utils.paired_bootstrap import BootstrapResult, paired_bootstrap, paired_bootstrap_ci


def test_ci_brackets_mean_for_random_deltas():
    rng = np.random.default_rng(3)
    deltas = rng.normal(loc=0.2, scale=0.1, size=30)
    res = paired_bootstrap(deltas, B=500, seed=1)
    assert res.ci_lo <= res.mean <= res.ci_hi
    assert 0.0 <= res.p_value_two_sided <= 1.0
    assert res.n == 30


def test_empty_input_returns_nan_result_with_zero_n():
    res = paired_bootstrap([], B=50)
    assert res.n == 0
    assert np.isnan(res.mean)


def test_deterministic_for_fixed_seed():
    deltas = [0.1, -0.2, 0.05, 0.3]
    assert paired_bootstrap(deltas, B=300, seed=9) == paired_bootstrap(deltas, B=300, seed=9)


def test_ci_helper_matches_full_result():
    deltas = [0.1, 0.2, 0.3, 0.15]
    lo, hi = paired_bootstrap_ci(deltas, B=200, seed=5)
    full = paired_bootstrap(deltas, B=200, seed=5)
    assert (lo, hi) == (full.ci_lo, full.ci_hi)
    assert isinstance(full, BootstrapResult)
