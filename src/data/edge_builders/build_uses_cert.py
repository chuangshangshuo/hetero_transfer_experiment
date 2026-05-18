from __future__ import annotations

import pandas as pd

from ..builder_common import aggregate_edge_pairs


def build_uses_cert_edges(raw_df: pd.DataFrame, node_index: dict[str, dict[str, int]]) -> pd.DataFrame:
    return aggregate_edge_pairs(raw_df, "uses_cert", node_index["Website"], node_index["Certificate"])
