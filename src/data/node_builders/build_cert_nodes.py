from __future__ import annotations

import re
from typing import Any

import pandas as pd

from ..builder_common import clean_value, maybe_add_feature_column, normalize_text_token, stable_top_categories


def _issuer_family(row: pd.Series) -> str:
    for col in ("tls_issuer_org", "region_tls_issuer", "tls_issuer_cn"):
        value = normalize_text_token(row.get(col, ""))
        if value:
            return value
    return ""


def _feature_safe_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value).strip("_")


def build_cert_nodes(
    cert_nodes: pd.DataFrame,
    *,
    issuer_family_vocab_size: int,
    drop_all_zero_columns: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    df = cert_nodes.copy()
    if df.empty:
        return df, {"active_feature_columns": [], "dropped_columns": [], "issuer_family_vocab": []}

    for col in [
        "tls_issuer_org",
        "region_tls_issuer",
        "tls_issuer_cn",
        "tls_common_name",
        "tls_san_values",
        "tls_san_count",
        "tls_wildcard_san_count",
        "tls_subject_cn",
    ]:
        df[col] = df.get(col, "").map(clean_value)

    df["issuer_family"] = df.apply(_issuer_family, axis=1)
    df["has_wildcard_san"] = pd.to_numeric(df["tls_wildcard_san_count"], errors="coerce").fillna(0.0).gt(0).astype(float)
    df["has_multi_san"] = pd.to_numeric(df["tls_san_count"], errors="coerce").fillna(0.0).gt(1).astype(float)

    issuer_vocab = stable_top_categories(df["issuer_family"], issuer_family_vocab_size)
    active_columns: list[str] = []
    dropped_columns: list[dict[str, Any]] = []

    for issuer in issuer_vocab:
        maybe_add_feature_column(
            df,
            f"feat_issuer_family_{_feature_safe_name(issuer)}",
            (df["issuer_family"] == issuer).astype(float).to_numpy(),
            active_columns,
            dropped_columns,
            source_column="issuer_family",
            feature_kind="one_hot",
            drop_all_zero=drop_all_zero_columns,
        )
    maybe_add_feature_column(
        df,
        "feat_issuer_family_other",
        ((df["issuer_family"] != "") & (~df["issuer_family"].isin(issuer_vocab))).astype(float).to_numpy(),
        active_columns,
        dropped_columns,
        source_column="issuer_family",
        feature_kind="binary",
        drop_all_zero=drop_all_zero_columns,
    )
    maybe_add_feature_column(
        df,
        "feat_issuer_family_missing",
        (df["issuer_family"] == "").astype(float).to_numpy(),
        active_columns,
        dropped_columns,
        source_column="issuer_family",
        feature_kind="binary",
        drop_all_zero=drop_all_zero_columns,
    )
    maybe_add_feature_column(
        df,
        "feat_has_wildcard_san",
        df["has_wildcard_san"].to_numpy(),
        active_columns,
        dropped_columns,
        source_column="tls_wildcard_san_count",
        feature_kind="binary",
        drop_all_zero=drop_all_zero_columns,
    )
    maybe_add_feature_column(
        df,
        "feat_has_multi_san",
        df["has_multi_san"].to_numpy(),
        active_columns,
        dropped_columns,
        source_column="tls_san_count",
        feature_kind="binary",
        drop_all_zero=drop_all_zero_columns,
    )

    builder_state = {
        "candidate_feature_columns": [f"feat_issuer_family_{_feature_safe_name(issuer)}" for issuer in issuer_vocab]
        + ["feat_issuer_family_other", "feat_issuer_family_missing", "feat_has_wildcard_san", "feat_has_multi_san"],
        "active_feature_columns": active_columns,
        "dropped_columns": dropped_columns,
        "issuer_family_vocab": issuer_vocab,
    }
    return df, builder_state
