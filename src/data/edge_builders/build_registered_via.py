"""Build Website->Registrar registered_via edges."""
from __future__ import annotations

import pandas as pd

from ..builder_common import aggregate_edge_pairs


def build_registered_via_edges(raw_df: pd.DataFrame, node_index: dict[str, dict[str, int]]) -> pd.DataFrame:
    """Build registered via edges."""
    return aggregate_edge_pairs(raw_df, "registered_via", node_index["Website"], node_index["Registrar"])
