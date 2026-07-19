"""Batch sampler guaranteeing same-family positives for SupCon training."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class NWayKShotBatch:
    """N Way K Shot Batch."""
    indices: np.ndarray
    selected_families: list[str]
    licensed_count: int


def sample_n_way_k_shot(
    frame: pd.DataFrame,
    family_column: str,
    label_column: str = "label",
    n_way: int = 4,
    k_shot: int = 2,
    licensed_in_batch: int = 8,
    seed: int = 42,
) -> NWayKShotBatch:
    """Build a small family-aware batch from a labelled frame.

    The current Week 11 runner uses full-batch optimization for stability, but
    this sampler is kept as the auditable implementation of the preregistered
    N-way/K-shot protocol.
    """

    rng = np.random.default_rng(seed)
    illegal = frame[(frame[label_column] == 1) & frame[family_column].notna()].copy()
    licensed = frame[frame[label_column] == 0].copy()
    families = sorted(f for f, part in illegal.groupby(family_column) if len(part) >= 2)
    if not families:
        raise ValueError("No family has at least two samples for N-way/K-shot sampling")
    take_families = list(rng.choice(families, size=min(n_way, len(families)), replace=False))
    selected_indices: list[int] = []
    for family in take_families:
        part = illegal[illegal[family_column] == family]
        take = min(k_shot, len(part))
        selected_indices.extend(rng.choice(part.index.to_numpy(), size=take, replace=False).tolist())
    if licensed_in_batch > 0 and not licensed.empty:
        take_licensed = min(licensed_in_batch, len(licensed))
        selected_indices.extend(rng.choice(licensed.index.to_numpy(), size=take_licensed, replace=False).tolist())
    return NWayKShotBatch(
        indices=np.array(selected_indices, dtype=int),
        selected_families=take_families,
        licensed_count=min(licensed_in_batch, len(licensed)),
    )
