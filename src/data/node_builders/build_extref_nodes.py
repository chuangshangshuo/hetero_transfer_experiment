from __future__ import annotations

from typing import Any

import pandas as pd

from ..builder_common import clean_value, maybe_add_feature_column


def build_extref_nodes(extref_nodes: pd.DataFrame, *, drop_all_zero_columns: bool) -> tuple[pd.DataFrame, dict[str, Any]]:
    df = extref_nodes.copy()
    if df.empty:
        return df, {"active_feature_columns": [], "dropped_columns": []}

    df["external_ref_type"] = df.get("external_ref_type", "").map(clean_value)
    active_columns: list[str] = []
    dropped_columns: list[dict[str, Any]] = []

    maybe_add_feature_column(
        df,
        "feat_is_search_discovery",
        (df["external_ref_type"] == "search_discovery").astype(float).to_numpy(),
        active_columns,
        dropped_columns,
        source_column="external_ref_type",
        feature_kind="binary",
        drop_all_zero=drop_all_zero_columns,
    )
    maybe_add_feature_column(
        df,
        "feat_is_archive_history",
        (df["external_ref_type"] == "archive_history").astype(float).to_numpy(),
        active_columns,
        dropped_columns,
        source_column="external_ref_type",
        feature_kind="binary",
        drop_all_zero=drop_all_zero_columns,
    )

    builder_state = {
        "candidate_feature_columns": ["feat_is_search_discovery", "feat_is_archive_history"],
        "active_feature_columns": active_columns,
        "dropped_columns": dropped_columns,
    }
    return df, builder_state
