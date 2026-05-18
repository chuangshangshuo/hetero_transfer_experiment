from __future__ import annotations

from typing import Any

import pandas as pd

from ..builder_common import clean_value, maybe_add_feature_column, zscore


def build_registrar_nodes(
    registrar_nodes: pd.DataFrame,
    *,
    zscore_epsilon: float,
    drop_all_zero_columns: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    df = registrar_nodes.copy()
    if df.empty:
        return df, {"active_feature_columns": [], "dropped_columns": [], "scalers": {}}

    df["registrar"] = df.get("registrar", "").map(clean_value)
    df["registrar_len"] = df["registrar"].map(len).astype(float)
    df["is_numeric"] = df["registrar"].map(lambda value: float(value.isdigit())).astype(float)

    active_columns: list[str] = []
    dropped_columns: list[dict[str, Any]] = []
    scaler_state: dict[str, Any] = {}

    values, scaler = zscore(df["registrar_len"], zscore_epsilon)
    scaler_state["feat_registrar_len_z"] = {"source_column": "registrar_len", **scaler}
    maybe_add_feature_column(
        df,
        "feat_registrar_len_z",
        values,
        active_columns,
        dropped_columns,
        source_column="registrar_len",
        feature_kind="zscore",
        drop_all_zero=drop_all_zero_columns,
    )
    maybe_add_feature_column(
        df,
        "feat_is_numeric",
        df["is_numeric"].to_numpy(),
        active_columns,
        dropped_columns,
        source_column="registrar",
        feature_kind="binary",
        drop_all_zero=drop_all_zero_columns,
    )

    builder_state = {
        "candidate_feature_columns": ["feat_registrar_len_z", "feat_is_numeric"],
        "active_feature_columns": active_columns,
        "dropped_columns": dropped_columns,
        "scalers": scaler_state,
    }
    return df, builder_state
