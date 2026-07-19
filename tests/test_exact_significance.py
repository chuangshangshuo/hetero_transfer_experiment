"""Unit tests for analysis/supp_exact_significance.py (exact n=5 tests)."""
from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from conftest import REPO_ROOT

spec = importlib.util.spec_from_file_location(
    "supp_exact_significance", REPO_ROOT / "analysis" / "supp_exact_significance.py"
)
supp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(supp)


class TestExactSignflip:
    def test_all_same_sign_hits_the_exact_floor(self):
        # n=5, all positive: one-sided 1/32, two-sided 2/32
        two, one = supp.exact_signflip_pvalues(np.array([0.1, 0.2, 0.15, 0.05, 0.3]))
        assert one == pytest.approx(1 / 32)
        assert two == pytest.approx(2 / 32)

    def test_perfectly_symmetric_deltas_are_not_significant(self):
        two, _ = supp.exact_signflip_pvalues(np.array([0.5, -0.5]))
        assert two == pytest.approx(1.0)

    def test_empty_input_returns_nan(self):
        two, one = supp.exact_signflip_pvalues(np.array([np.nan]))
        assert np.isnan(two) and np.isnan(one)

    def test_two_sided_floor_beats_uncorrected_alpha(self):
        """Core paper claim: at n=5 the two-sided floor (0.0625) exceeds 0.05."""
        two, _ = supp.exact_signflip_pvalues(np.ones(5))
        assert two > 0.05


class TestExactSignTest:
    def test_all_positive_n5(self):
        assert supp.exact_sign_test(np.ones(5)) == pytest.approx(0.0625)

    def test_four_of_five_positive(self):
        p = supp.exact_sign_test(np.array([1.0, 1.0, 1.0, 1.0, -1.0]))
        assert p == pytest.approx(0.375)

    def test_all_zero_deltas_give_p_one(self):
        assert supp.exact_sign_test(np.zeros(4)) == pytest.approx(1.0)


class TestHolmConsistency:
    def test_matches_src_utils_holm(self):
        from src.utils.significance_testing import holm_bonferroni

        pvals = np.array([0.001, 0.0625, 0.0625, 0.5, np.nan])
        assert supp.holm(pvals).tolist() == holm_bonferroni(pvals).tolist()


class TestReleasedDataProperties:
    def test_w7_pairs_none_survive_holm(self):
        """Regression for the supplementary headline: 0/24 Holm-significant."""
        frame = supp.pair_rows_w7()
        assert len(frame) == 24
        assert int(frame["holm_reject_perm_two_sided"].sum()) == 0
        floor = frame[frame["perm_p_two_sided"] <= 2 / 32 + 1e-12]
        assert len(floor) == 6
