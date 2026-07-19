"""Unit tests for src/utils/significance_testing.py (paired bootstrap + Holm)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.utils.significance_testing import holm_bonferroni, paired_bootstrap_pvalue

from conftest import REPO_ROOT


class TestHolmBonferroni:
    def test_all_rejected_when_sequentially_below_thresholds(self):
        # thresholds for m=4: 0.05/4, 0.05/3, 0.05/2, 0.05/1 after sorting
        reject = holm_bonferroni([0.001, 0.01, 0.02, 0.04], alpha=0.05)
        assert reject.tolist() == [True, True, True, True]

    def test_sequential_stop_rejects_nothing_when_smallest_fails(self):
        # smallest p (0.03) > 0.05/2 -> stop immediately, nothing rejected
        reject = holm_bonferroni([0.03, 0.04], alpha=0.05)
        assert reject.tolist() == [False, False]

    def test_input_order_is_preserved(self):
        reject = holm_bonferroni([0.9, 0.0001, 0.9], alpha=0.05)
        assert reject.tolist() == [False, True, False]

    def test_nan_entries_are_never_rejected_and_shrink_m(self):
        # NaN is excluded from the family: m=1 so 0.04 <= 0.05 is rejected
        reject = holm_bonferroni([np.nan, 0.04], alpha=0.05)
        assert reject.tolist() == [False, True]


class TestPairedBootstrapPvalue:
    def test_identical_arrays_give_pvalue_one(self):
        a = np.array([0.5, 0.6, 0.7, 0.8, 0.9])
        out = paired_bootstrap_pvalue(a, a.copy(), B=200, seed=0)
        assert out["mean_delta"] == pytest.approx(0.0)
        assert out["bootstrap_p"] == pytest.approx(1.0)

    def test_constant_positive_shift_gives_zero_pvalue_and_tight_ci(self):
        b = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        a = b + 1.0
        out = paired_bootstrap_pvalue(a, b, B=200, seed=0)
        assert out["mean_delta"] == pytest.approx(1.0)
        assert out["ci95_low"] == pytest.approx(1.0)
        assert out["ci95_high"] == pytest.approx(1.0)
        assert out["bootstrap_p"] == pytest.approx(0.0)

    def test_deterministic_for_fixed_seed(self):
        rng = np.random.default_rng(7)
        a, b = rng.normal(size=8), rng.normal(size=8)
        o1 = paired_bootstrap_pvalue(a, b, B=500, seed=42)
        o2 = paired_bootstrap_pvalue(a, b, B=500, seed=42)
        assert o1 == o2

    def test_empty_and_nonfinite_inputs_return_nan(self):
        out = paired_bootstrap_pvalue(np.array([np.nan]), np.array([np.nan]), B=10)
        assert np.isnan(out["mean_delta"]) and np.isnan(out["bootstrap_p"])


class TestReleasedHeadlineReproduction:
    """Regression: recompute the paper's T2_ON DANN-vs-source_only bootstrap p."""

    def test_t2on_dann_vs_source_only_matches_released_p4(self):
        path = REPO_ROOT / "results" / "week7" / "metrics" / "transfer_summary.csv"
        frame = pd.read_csv(path)
        part = frame[frame["transfer_id"] == "T2_ON"]
        dann = part[part["method"] == "dann"].sort_values("seed")["primary_metric_value"].to_numpy()
        src = part[part["method"] == "source_only"].sort_values("seed")["primary_metric_value"].to_numpy()
        out = paired_bootstrap_pvalue(dann, src, B=10000, seed=42)
        assert out["mean_delta"] == pytest.approx(0.0569, abs=1e-3)
        assert out["bootstrap_p"] == pytest.approx(0.0012, abs=1e-4)
