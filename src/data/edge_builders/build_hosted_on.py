"""Build Website->IP hosted_on edges."""
from __future__ import annotations

import pandas as pd

from ..builder_common import aggregate_edge_pairs


def build_hosted_on_edges(raw_df: pd.DataFrame, node_index: dict[str, dict[str, int]]) -> pd.DataFrame:
    """Build hosted on edges."""
    return aggregate_edge_pairs(raw_df, "hosted_on", node_index["Website"], node_index["IP"])
