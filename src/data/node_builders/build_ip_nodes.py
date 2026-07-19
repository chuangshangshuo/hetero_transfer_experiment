"""Build IP nodes with ASN one-hot plus summary statistics features."""
from __future__ import annotations

from typing import Any

import pandas as pd

from ..builder_common import clean_value, maybe_add_feature_column, stable_top_categories


def build_ip_nodes(
    ip_nodes: pd.DataFrame,
    *,
    asn_vocab_size: int,
    drop_all_zero_columns: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build ip nodes."""
    df = ip_nodes.copy()
    if df.empty:
        return df, {"active_feature_columns": [], "dropped_columns": [], "asn_vocab": []}

    df["asn_number"] = df.get("asn_number", "").map(clean_value)
    df["asn_org"] = df.get("asn_org", "").map(clean_value)
    df["is_private"] = df.get("is_private", "0").map(clean_value)
    df["is_global"] = df.get("is_global", "0").map(clean_value)

    asn_vocab = stable_top_categories(df["asn_number"], asn_vocab_size)
    active_columns: list[str] = []
    dropped_columns: list[dict[str, Any]] = []

    maybe_add_feature_column(
        df,
        "feat_is_private",
        (df["is_private"] == "1").astype(float).to_numpy(),
        active_columns,
        dropped_columns,
        source_column="is_private",
        feature_kind="binary",
        drop_all_zero=drop_all_zero_columns,
    )
    maybe_add_feature_column(
        df,
        "feat_is_global",
        (df["is_global"] == "1").astype(float).to_numpy(),
        active_columns,
        dropped_columns,
        source_column="is_global",
        feature_kind="binary",
        drop_all_zero=drop_all_zero_columns,
    )
    maybe_add_feature_column(
        df,
        "feat_asn_missing",
        (df["asn_number"] == "").astype(float).to_numpy(),
        active_columns,
        dropped_columns,
        source_column="asn_number",
        feature_kind="binary",
        drop_all_zero=drop_all_zero_columns,
    )
    for asn in asn_vocab:
        maybe_add_feature_column(
            df,
            f"feat_asn_{asn}",
            (df["asn_number"] == asn).astype(float).to_numpy(),
            active_columns,
            dropped_columns,
            source_column="asn_number",
            feature_kind="one_hot",
            drop_all_zero=drop_all_zero_columns,
        )
    maybe_add_feature_column(
        df,
        "feat_asn_other",
        ((df["asn_number"] != "") & (~df["asn_number"].isin(asn_vocab))).astype(float).to_numpy(),
        active_columns,
        dropped_columns,
        source_column="asn_number",
        feature_kind="binary",
        drop_all_zero=drop_all_zero_columns,
    )

    builder_state = {
        "candidate_feature_columns": ["feat_is_private", "feat_is_global", "feat_asn_missing"]
        + [f"feat_asn_{asn}" for asn in asn_vocab]
        + ["feat_asn_other"],
        "active_feature_columns": active_columns,
        "dropped_columns": dropped_columns,
        "asn_vocab": asn_vocab,
    }
    return df, builder_state
