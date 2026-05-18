from __future__ import annotations

import json
from pathlib import Path


REQUIRED_FILES = [
    "configs/default_v2.yaml",
    "src/data/SCHEMA_V2.md",
    "docs/compliance/README_promotion_collection_archived.md",
    "data/graphs/hetero_graph_v2_metadata.json",
    "data/versions/feature_builder_state.json",
]


FORBIDDEN_ACTIVE_FILES = [
    "configs/default.yaml",
    "src/data/collect_promotion_osint.py",
    "src/data/feature_extractors/pa.py",
    "configs/promotion_osint.json",
    "data/processed/edges/redirects_to.csv",
]


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    missing = [path for path in REQUIRED_FILES if not (root / path).exists()]
    forbidden = [path for path in FORBIDDEN_ACTIVE_FILES if (root / path).exists()]
    if missing:
        raise SystemExit(f"Missing required files: {missing}")
    if forbidden:
        raise SystemExit(f"Forbidden active files remain in Graph v2 workspace: {forbidden}")

    metadata_path = root / "data/graphs/hetero_graph_v2_metadata.json"
    builder_state_path = root / "data/versions/feature_builder_state.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    builder_state = json.loads(builder_state_path.read_text(encoding="utf-8"))

    expected_schema = "v2_6_nodes_5_relations_10_materialized_edges"
    if metadata.get("schema") != expected_schema:
        raise SystemExit(f"Unexpected graph schema: {metadata.get('schema')}")
    if metadata.get("promotion_account_status") != "archived_excluded":
        raise SystemExit("PromotionAccount status is not archived_excluded.")

    redirect_audit = metadata.get("redirect_audit", {})
    expected_redirect = {
        "raw_hops_total": 166,
        "raw_cross_domain_hops": 11,
        "normalized_cross_site_hops": 7,
        "mapped_to_graph_pairs": 0,
    }
    if redirect_audit != expected_redirect:
        raise SystemExit(f"Redirect audit mismatch: {redirect_audit}")

    website_features = builder_state["node_feature_builders"]["Website"]["active_feature_columns"]
    if len(website_features) != 7:
        raise SystemExit(f"Website feature count mismatch: {len(website_features)}")
    if any("label_" in column for column in website_features):
        raise SystemExit(f"Website features still contain label leakage: {website_features}")

    edge_type_triples = builder_state.get("edge_type_triples", {})
    if len(edge_type_triples) != 10:
        raise SystemExit(f"Expected 10 materialized edge types, found {len(edge_type_triples)}")

    print("sanity_check_passed: Graph v2 workspace and outputs match the expected PA-free schema.")
