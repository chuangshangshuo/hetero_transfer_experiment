"""Build (deprecated) Website->Website redirects_to edges; kept for audits."""
from __future__ import annotations

import pandas as pd

from ..builder_common import coarse_redirect_site_with_port, normalize_domain, registrable_domain


def build_redirect_audit(
    redirect_rows: pd.DataFrame,
    source_to_website: dict[str, str],
    domain_to_website: dict[str, str],
) -> pd.DataFrame:
    """Build redirect audit."""
    if redirect_rows.empty:
        return pd.DataFrame(
            [
                {"stage": "raw_hops_total", "count": 0, "definition": "all redirect hops"},
                {"stage": "raw_cross_domain_hops", "count": 0, "definition": "coarse host-level cross-domain hops including port pseudo-differences"},
                {"stage": "normalized_cross_site_hops", "count": 0, "definition": "registrable-domain cross-site hops after port stripping and ccTLD normalization"},
                {"stage": "mapped_to_graph_pairs", "count": 0, "definition": "cross-site pairs that map to distinct Website nodes in current vocab"},
            ]
        )

    work = redirect_rows.copy()
    work["site_id"] = work["site_id"].astype(str)
    work["from_host_coarse"] = work["from_url"].map(coarse_redirect_site_with_port)
    work["to_host_coarse"] = work["to_url"].map(coarse_redirect_site_with_port)
    work["from_site_normalized"] = work["from_url"].map(registrable_domain)
    work["to_site_normalized"] = work["to_url"].map(registrable_domain)
    work["to_host_exact"] = work["to_url"].map(normalize_domain)

    coarse_cross = work[
        (work["from_host_coarse"] != "")
        & (work["to_host_coarse"] != "")
        & (work["from_host_coarse"] != work["to_host_coarse"])
    ].copy()
    normalized_cross = work[
        (work["from_site_normalized"] != "")
        & (work["to_site_normalized"] != "")
        & (work["from_site_normalized"] != work["to_site_normalized"])
    ].copy()

    mapped_pairs: set[tuple[str, str]] = set()
    for _, row in normalized_cross.iterrows():
        src_id = source_to_website.get(str(row["site_id"]).strip(), "")
        dst_id = domain_to_website.get(str(row["to_host_exact"]).strip(), "")
        if src_id and dst_id and src_id != dst_id:
            mapped_pairs.add((src_id, dst_id))

    return pd.DataFrame(
        [
            {
                "stage": "raw_hops_total",
                "count": int(len(work)),
                "definition": "all redirect hops in phase1 redirect_relation.csv",
            },
            {
                "stage": "raw_cross_domain_hops",
                "count": int(len(coarse_cross)),
                "definition": "coarse host-level cross-domain hops including :443 pseudo-differences",
            },
            {
                "stage": "normalized_cross_site_hops",
                "count": int(len(normalized_cross)),
                "definition": "registrable-domain cross-site hops after port stripping and composite ccTLD normalization",
            },
            {
                "stage": "mapped_to_graph_pairs",
                "count": int(len(mapped_pairs)),
                "definition": "normalized cross-site pairs mapped to distinct Website nodes in current vocab",
            },
        ]
    )
