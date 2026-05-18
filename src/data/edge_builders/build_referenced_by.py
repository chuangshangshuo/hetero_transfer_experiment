from __future__ import annotations

import pandas as pd

from ..builder_common import aggregate_edge_pairs


def build_referenced_by_edges(
    raw_df: pd.DataFrame,
    node_index: dict[str, dict[str, int]],
    *,
    hub_degree_threshold: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    grouped = aggregate_edge_pairs(raw_df, "referenced_by", node_index["Website"], node_index["ExternalReference"])
    if grouped.empty:
        empty_hub = pd.DataFrame(columns=["dst_id", "dst_degree", "hub_flag"])
        return grouped, empty_hub

    dst_degree_map = grouped.groupby("dst_id")["src_id"].nunique().astype(int).to_dict()
    grouped["dst_degree"] = grouped["dst_id"].map(dst_degree_map).astype(int)
    grouped["hub_flag"] = grouped["dst_degree"].gt(hub_degree_threshold).astype(int)
    grouped["edge_weight"] = grouped["edge_weight_base_log1p"] / grouped["dst_degree"].astype(float)

    grouped = grouped[
        [
            "edge_id",
            "src_id",
            "dst_id",
            "src_index",
            "dst_index",
            "edge_count_raw",
            "edge_weight_base_log1p",
            "edge_weight",
            "dst_degree",
            "hub_flag",
            "source_dataset_set",
        ]
    ].copy()

    hub_audit = (
        grouped[["dst_id", "dst_degree", "hub_flag"]]
        .drop_duplicates()
        .sort_values(["hub_flag", "dst_degree", "dst_id"], ascending=[False, False, True])
        .reset_index(drop=True)
    )
    return grouped, hub_audit
