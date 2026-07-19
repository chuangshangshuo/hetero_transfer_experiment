"""Unit tests for src/transfer/logme_score.py."""
from __future__ import annotations

import numpy as np
import pytest

from src.transfer.logme_score import compute_logme


def _toy_features(seed: int = 0, n: int = 120, d: int = 8):
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 2, size=n)
    centers = np.where(labels[:, None] == 1, 2.0, -2.0)
    features = centers + rng.normal(scale=0.5, size=(n, d))
    return features, labels


def test_separable_features_score_higher_than_shuffled_labels():
    features, labels = _toy_features()
    rng = np.random.default_rng(1)
    shuffled = rng.permutation(labels)
    assert compute_logme(features, labels) > compute_logme(features, shuffled)


def test_single_class_returns_nan():
    features, _ = _toy_features()
    assert np.isnan(compute_logme(features, np.zeros(features.shape[0], dtype=int)))


def test_shape_mismatch_raises():
    features, labels = _toy_features()
    with pytest.raises(ValueError):
        compute_logme(features[:-1], labels)


def test_non_2d_features_raise():
    with pytest.raises(ValueError):
        compute_logme(np.zeros(5), np.zeros(5, dtype=int))
