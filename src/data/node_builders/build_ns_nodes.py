from __future__ import annotations

from typing import Any

import pandas as pd

from ..builder_common import clean_value, maybe_add_feature_column, zscore


def build_ns_nodes(
    ns_nodes: pd.DataFrame,
    *,
    zscore_epsilon: float,
    drop_all_zero_columns: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    df = ns_nodes.copy()
    if df.empty:
        return df, {"active_feature_columns": [], "dropped_columns": [], "scalers": {}}

    df["nameserver"] = df.get("nameserver", "").map(clean_value)
    df["provider_hint"] = df.get("provider_hint", "").map(clean_value)
    df["nameserver_len"] = df["nameserver"].map(len).astype(float)
    df["nameserver_segment_count"] = df["nameserver"].map(lambda value: value.count(".") + 1 if value else 0).astype(float)
    df["known_provider_hint"] = (df["provider_hint"] != "other").astype(float)

    active_columns: list[str] = []
    dropped_columns: list[dict[str, Any]] = []
    scaler_state: dict[str, Any] = {}

    for source_col, feature_name in {
        "nameserver_len": "feat_nameserver_len_z",
        "nameserver_segment_count": "feat_nameserver_segment_count_z",
    }.items():
        values, scaler = zscore(df[source_col], zscore_epsilon)
        scaler_state[feature_name] = {"source_column": source_col, **scaler}
        maybe_add_feature_column(
            df,
            feature_name,
            values,
            active_columns,
            dropped_columns,
            source_column=source_col,
            feature_kind="zscore",
            drop_all_zero=drop_all_zero_columns,
        )

    maybe_add_feature_column(
        df,
        "feat_known_provider_hint",
        df["known_provider_hint"].to_numpy(),
        active_columns,
        dropped_columns,
        source_column="provider_hint",
        feature_kind="binary",
        drop_all_zero=drop_all_zero_columns,
    )

    builder_state = {
        "candidate_feature_columns": [
            "feat_nameserver_len_z",
            "feat_nameserver_segment_count_z",
            "feat_known_provider_hint",
        ],
        "active_feature_columns": active_columns,
        "dropped_columns": dropped_columns,
        "scalers": scaler_state,
    }
    return df, builder_state
