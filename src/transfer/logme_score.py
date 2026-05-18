from __future__ import annotations

import numpy as np


def _as_2d_float(features: np.ndarray) -> np.ndarray:
    array = np.asarray(features, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"features must be 2D, got shape {array.shape}")
    return array


def _max_evidence(u: np.ndarray, singular_values: np.ndarray, y: np.ndarray, max_iter: int = 50) -> float:
    n = y.shape[0]
    rank = singular_values.shape[0]
    sigma2 = np.square(singular_values)
    z = u.T @ y
    alpha = 1.0
    beta = 1.0 / max(float(np.var(y)), 1e-6)
    y_norm2 = float(y @ y)
    for _ in range(max_iter):
        denom = alpha + beta * sigma2
        gamma = float(np.sum((beta * sigma2) / denom))
        m = (beta / denom) * z
        m_norm2 = float(m @ m)
        residual = y_norm2 - float(np.sum((beta * sigma2 / denom) * np.square(z)))
        residual = max(residual, 1e-8)
        new_alpha = max(gamma / max(m_norm2, 1e-8), 1e-8)
        new_beta = max((n - gamma) / residual, 1e-8)
        if abs(new_alpha - alpha) / max(alpha, 1e-8) < 1e-4 and abs(new_beta - beta) / max(beta, 1e-8) < 1e-4:
            alpha, beta = float(new_alpha), float(new_beta)
            break
        alpha, beta = float(new_alpha), float(new_beta)

    alpha = max(float(alpha), 1e-8)
    beta = max(float(beta), 1e-8)
    denom = np.maximum(alpha + beta * sigma2, 1e-12)
    evidence = 0.5 * (
        rank * np.log(alpha)
        + n * np.log(beta)
        - np.sum(np.log(denom))
        - n * np.log(2.0 * np.pi)
        - beta * y_norm2
        + np.sum((beta * beta * sigma2 / denom) * np.square(z))
    )
    return float(evidence / max(n, 1))


def compute_logme(features: np.ndarray, labels: np.ndarray) -> float:
    """Compute a compact LogME-style evidence score for classification labels.

    Larger values indicate that the provided features support the labels with
    less task-specific fitting. Labels must be integer-coded classes.
    """
    x = _as_2d_float(features)
    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    if x.shape[0] != y.shape[0]:
        raise ValueError("features and labels have incompatible first dimensions")
    classes = np.unique(y)
    if classes.shape[0] < 2:
        return float("nan")
    x = x - x.mean(axis=0, keepdims=True)
    u, singular_values, _ = np.linalg.svd(x, full_matrices=False)
    scores = []
    for klass in classes:
        target = (y == klass).astype(np.float64)
        target = target - target.mean()
        scores.append(_max_evidence(u, singular_values, target))
    return float(np.mean(scores))
