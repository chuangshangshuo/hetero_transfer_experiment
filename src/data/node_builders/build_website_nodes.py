from __future__ import annotations

from typing import Any

import pandas as pd

from ..builder_common import SOURCE_ROOT, clean_value, domain_features, maybe_add_feature_column, normalize_domain, read_csv, zscore


def infer_bucket(row: dict[str, str]) -> str:
    sample_bucket = clean_value(row.get("sample_bucket"))
    sample_type = clean_value(row.get("sample_type"))
    sample_tier = clean_value(row.get("sample_tier"))
    cohort = clean_value(row.get("cohort"))
    combined = " ".join([sample_bucket, sample_type, sample_tier, cohort]).lower()
    if "illegal" in combined:
        return "illegal_confirmed"
    if "gray" in combined:
        return "gray_candidate"
    if "licensed" in combined or "gambling" in combined:
        return "licensed_baseline"
    if "control" in combined:
        return "control_legal_commercial"
    return sample_bucket or sample_type or "unknown"


def bucket_priority(bucket: str, tier: str) -> int:
    combined = f"{bucket} {tier}".lower()
    if "illegal_confirmed_official_cross_verified" in combined:
        return 110
    if "illegal" in combined:
        return 100
    if "licensed" in combined:
        return 80
    if "gray" in combined:
        return 70
    if "control" in combined:
        return 50
    return 10


def dataset_priority(dataset: str) -> int:
    order = {
        "region_expansion_site_profile": 50,
        "phase1_site_profile": 40,
        "phase2_illegal_tls_detail": 35,
        "phase2_illegal_catalog": 30,
        "phase2_licensed_catalog": 20,
        "phase2_gray_catalog": 15,
    }
    return order.get(dataset, 0)


def _append_site_profile(frames: list[pd.DataFrame], path, dataset: str) -> None:
    df = read_csv(path)
    if df.empty:
        return
    frames.append(
        pd.DataFrame(
            {
                "source_dataset": dataset,
                "site_id": df.get("site_id", ""),
                "entity_id": df.get("entity_id", ""),
                "root_domain": df.get("root_domain", "").map(normalize_domain),
                "jurisdiction": df.get("jurisdiction", df.get("region", "")),
                "sample_type": df.get("sample_type", ""),
                "sample_bucket": df.get("sample_bucket", ""),
                "sample_tier": df.get("sample_tier", ""),
                "cohort": df.get("cohort", ""),
                "brand": df.get("brand", df.get("canonical_name", "")),
                "operator_or_case": df.get("operator_or_case", df.get("linked_operator", "")),
                "regulatory_status": df.get("regulatory_status", ""),
                "source_page": df.get("source_page", df.get("official_basis_url", "")),
                "landing_url": df.get("landing_url", df.get("observed_url", "")),
                "page_probe_status": df.get("page_probe_status", ""),
                "collected_at": df.get("collected_at", ""),
            }
        )
    )


def _append_catalog(frames: list[pd.DataFrame], path, dataset: str) -> None:
    df = read_csv(path)
    if df.empty:
        return
    frames.append(
        pd.DataFrame(
            {
                "source_dataset": dataset,
                "site_id": df.get("sample_id", ""),
                "entity_id": df.get("entity_id", ""),
                "root_domain": df.get("root_domain", "").map(normalize_domain),
                "jurisdiction": df.get("region", ""),
                "sample_type": df.get("sample_bucket", ""),
                "sample_bucket": df.get("sample_bucket", ""),
                "sample_tier": df.get("sample_tier", ""),
                "cohort": df.get("analysis_role", ""),
                "brand": df.get("linked_brand", df.get("canonical_name", "")),
                "operator_or_case": df.get("linked_operator", df.get("canonical_name", "")),
                "regulatory_status": df.get("official_confirmed_tier", ""),
                "source_page": df.get("official_basis_url", ""),
                "landing_url": df.get("observed_url", ""),
                "page_probe_status": "",
                "collected_at": df.get("collected_at", ""),
            }
        )
    )


def load_site_rows() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    _append_site_profile(frames, SOURCE_ROOT / "phase1" / "core" / "site_profile.csv", "phase1_site_profile")
    _append_site_profile(
        frames,
        SOURCE_ROOT / "phase2" / "region_expansion" / "region_expansion_site_profile.csv",
        "region_expansion_site_profile",
    )
    _append_catalog(frames, SOURCE_ROOT / "phase2" / "evidence" / "licensed_baseline_catalog.csv", "phase2_licensed_catalog")
    _append_catalog(frames, SOURCE_ROOT / "phase2" / "evidence" / "illegal_confirmed_catalog.csv", "phase2_illegal_catalog")
    _append_catalog(frames, SOURCE_ROOT / "phase2" / "evidence" / "gray_candidate_catalog.csv", "phase2_gray_catalog")

    tls = read_csv(SOURCE_ROOT / "phase2" / "followup" / "illegal_tls_certificate_detail.csv")
    if not tls.empty:
        frames.append(
            pd.DataFrame(
                {
                    "source_dataset": "phase2_illegal_tls_detail",
                    "site_id": tls.get("site_id", ""),
                    "entity_id": "",
                    "root_domain": tls.get("root_domain", "").map(normalize_domain),
                    "jurisdiction": tls.get("jurisdiction", ""),
                    "sample_type": tls.get("sample_bucket", ""),
                    "sample_bucket": tls.get("sample_bucket", ""),
                    "sample_tier": "illegal_confirmed_official_single",
                    "cohort": "illegal",
                    "brand": tls.get("root_domain", ""),
                    "operator_or_case": "",
                    "regulatory_status": "illegal_confirmed",
                    "source_page": "",
                    "landing_url": "",
                    "page_probe_status": "",
                    "collected_at": tls.get("observed_at", ""),
                }
            )
        )

    if not frames:
        return pd.DataFrame()

    site_rows = pd.concat(frames, ignore_index=True)
    site_rows["root_domain"] = site_rows["root_domain"].map(normalize_domain)
    site_rows = site_rows[site_rows["root_domain"] != ""].copy()
    for col in site_rows.columns:
        site_rows[col] = site_rows[col].map(clean_value)

    site_rows["sample_tier_source"] = "observed"
    licensed_phase1 = (site_rows["source_dataset"] == "phase1_site_profile") & (site_rows["sample_type"] == "licensed_gambling")
    control_phase1 = (site_rows["source_dataset"] == "phase1_site_profile") & (
        site_rows["sample_type"] == "control_legal_commercial"
    )
    site_rows.loc[licensed_phase1 & (site_rows["sample_tier"] == ""), "sample_tier"] = "licensed_baseline"
    site_rows.loc[licensed_phase1 & (site_rows["sample_tier_source"] == "observed"), "sample_tier_source"] = "backfilled_phase1"
    site_rows.loc[control_phase1 & (site_rows["sample_tier"] == ""), "sample_tier"] = "control_legal_commercial"
    site_rows.loc[control_phase1 & (site_rows["sample_tier_source"] == "observed"), "sample_tier_source"] = "backfilled_phase1"

    site_rows["inferred_bucket"] = site_rows.apply(lambda r: infer_bucket(r.to_dict()), axis=1)
    site_rows["priority"] = site_rows.apply(
        lambda r: bucket_priority(r["inferred_bucket"], r["sample_tier"]) + dataset_priority(r["source_dataset"]),
        axis=1,
    )
    return site_rows


def build_website_nodes(
    site_rows: pd.DataFrame,
    *,
    zscore_epsilon: float,
    drop_all_zero_columns: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, str], dict[str, str], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    duplicate_rows: list[dict[str, Any]] = []
    source_to_website: dict[str, str] = {}

    for idx, (domain, group) in enumerate(sorted(site_rows.groupby("root_domain"), key=lambda x: x[0]), start=1):
        group = group.sort_values(["priority", "source_dataset", "site_id"], ascending=[False, True, True])
        canon = group.iloc[0].to_dict()
        website_id = f"W_{idx:06d}"
        source_site_ids = sorted({clean_value(v) for v in group["site_id"].tolist() if clean_value(v)})
        source_entity_ids = sorted({clean_value(v) for v in group["entity_id"].tolist() if clean_value(v)})
        source_datasets = sorted({clean_value(v) for v in group["source_dataset"].tolist() if clean_value(v)})
        jurisdictions = sorted({clean_value(v) for v in group["jurisdiction"].tolist() if clean_value(v)})
        buckets = sorted({clean_value(v) for v in group["inferred_bucket"].tolist() if clean_value(v)})
        tiers = sorted({clean_value(v) for v in group["sample_tier"].tolist() if clean_value(v)})
        tier_sources = sorted({clean_value(v) for v in group["sample_tier_source"].tolist() if clean_value(v)})

        for sid in source_site_ids:
            source_to_website[sid] = website_id
        for eid in source_entity_ids:
            source_to_website[eid] = website_id
        source_to_website[f"domain::{domain.replace('.', '_')}"] = website_id
        source_to_website[f"domain::{domain}"] = website_id

        bucket = infer_bucket(canon)
        is_illegal = int("illegal" in bucket.lower())
        is_gray = int("gray" in bucket.lower())
        is_licensed = int("licensed" in bucket.lower())
        is_control = int("control" in bucket.lower())
        feats = domain_features(domain)
        records.append(
            {
                "node_id": website_id,
                "website_id": website_id,
                "root_domain": domain,
                "canonical_source_dataset": canon["source_dataset"],
                "source_site_ids": "|".join(source_site_ids),
                "source_entity_ids": "|".join(source_entity_ids),
                "source_datasets": "|".join(source_datasets),
                "jurisdiction": canon["jurisdiction"],
                "jurisdiction_set": "|".join(jurisdictions),
                "sample_type": canon["sample_type"],
                "sample_bucket": bucket,
                "sample_tier": canon["sample_tier"],
                "sample_tier_source": canon["sample_tier_source"],
                "sample_bucket_set": "|".join(buckets),
                "sample_tier_set": "|".join(tiers),
                "sample_tier_source_set": "|".join(tier_sources),
                "cohort": canon["cohort"],
                "brand": canon["brand"],
                "operator_or_case": canon["operator_or_case"],
                "regulatory_status": canon["regulatory_status"],
                "landing_url": canon["landing_url"],
                "source_page": canon["source_page"],
                "label_illegal": is_illegal,
                "label_licensed": is_licensed,
                "label_gray": is_gray,
                "label_control": is_control,
                **feats,
            }
        )
        if len(group) > 1 or len(buckets) > 1 or len(jurisdictions) > 1:
            duplicate_rows.append(
                {
                    "root_domain": domain,
                    "canonical_website_id": website_id,
                    "row_count": len(group),
                    "source_site_ids": "|".join(source_site_ids),
                    "source_entity_ids": "|".join(source_entity_ids),
                    "source_datasets": "|".join(source_datasets),
                    "jurisdiction_set": "|".join(jurisdictions),
                    "sample_bucket_set": "|".join(buckets),
                    "sample_tier_set": "|".join(tiers),
                    "sample_tier_source_set": "|".join(tier_sources),
                    "canonical_bucket": bucket,
                    "audit_note": "root-domain dedup retained highest-priority neutral tier; no rows deleted from audit trail",
                }
            )

    master = pd.DataFrame(records).sort_values("root_domain").reset_index(drop=True)
    duplicate_audit = pd.DataFrame(duplicate_rows)
    active_columns: list[str] = []
    dropped_columns: list[dict[str, Any]] = []
    scaler_state: dict[str, Any] = {}

    continuous_specs = {
        "domain_len": "feat_domain_len_z",
        "root_len": "feat_root_len_z",
        "domain_segment_count": "feat_domain_segment_count_z",
        "digit_count": "feat_digit_count_z",
        "hyphen_count": "feat_hyphen_count_z",
    }
    for source_col, feature_name in continuous_specs.items():
        values, scaler = zscore(master[source_col], zscore_epsilon)
        scaler_state[feature_name] = {"source_column": source_col, **scaler}
        maybe_add_feature_column(
            master,
            feature_name,
            values,
            active_columns,
            dropped_columns,
            source_column=source_col,
            feature_kind="zscore",
            drop_all_zero=drop_all_zero_columns,
        )

    binary_specs = {
        "is_dot_com": "feat_is_dot_com",
        "is_local_tld": "feat_is_local_tld",
    }
    for source_col, feature_name in binary_specs.items():
        maybe_add_feature_column(
            master,
            feature_name,
            master[source_col].astype(float).to_numpy(),
            active_columns,
            dropped_columns,
            source_column=source_col,
            feature_kind="binary",
            drop_all_zero=drop_all_zero_columns,
        )

    if any(not col.startswith("feat_") for col in active_columns):
        raise ValueError(f"Website feature columns must start with feat_: {active_columns}")
    if any("label_" in col for col in active_columns):
        raise ValueError(f"Website feature columns must not contain label_: {active_columns}")

    domain_to_website = dict(zip(master["root_domain"], master["node_id"]))
    tier_backfill_audit = master[
        (master["canonical_source_dataset"] == "phase1_site_profile") & (master["sample_tier_source"] == "backfilled_phase1")
    ].copy()
    builder_state = {
        "candidate_feature_columns": list(continuous_specs.values()) + list(binary_specs.values()),
        "active_feature_columns": active_columns,
        "dropped_columns": dropped_columns,
        "scalers": scaler_state,
        "tier_backfill_counts": {
            "licensed_baseline": int((tier_backfill_audit["sample_tier"] == "licensed_baseline").sum()),
            "control_legal_commercial": int((tier_backfill_audit["sample_tier"] == "control_legal_commercial").sum()),
        },
    }
    return master, duplicate_audit, tier_backfill_audit, source_to_website, domain_to_website, builder_state
