from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.data.builder_common import (
        AUDIT_ROOT,
        EDGE_ROOT,
        GRAPH_ROOT,
        MAXMIND_DB,
        NODE_ROOT,
        OUT_ROOT,
        SOURCE_ROOT,
        VERSION_ROOT,
        NodeStore,
        archive_file,
        assert_no_all_zero_columns,
        clean_value,
        join_unique,
        load_maxmind_reader,
        lookup_asn,
        parse_as_bool,
        parse_san_values,
        read_csv,
        sha256_file,
        write_csv,
        write_json,
    )
    from src.data.edge_builders.build_hosted_on import build_hosted_on_edges
    from src.data.edge_builders.build_redirects_to import build_redirect_audit
    from src.data.edge_builders.build_referenced_by import build_referenced_by_edges
    from src.data.edge_builders.build_registered_via import build_registered_via_edges
    from src.data.edge_builders.build_uses_cert import build_uses_cert_edges
    from src.data.edge_builders.build_uses_ns import build_uses_ns_edges
    from src.data.node_builders.build_cert_nodes import build_cert_nodes
    from src.data.node_builders.build_extref_nodes import build_extref_nodes
    from src.data.node_builders.build_ip_nodes import build_ip_nodes
    from src.data.node_builders.build_ns_nodes import build_ns_nodes
    from src.data.node_builders.build_registrar_nodes import build_registrar_nodes
    from src.data.node_builders.build_website_nodes import build_website_nodes, load_site_rows
else:
    from .builder_common import (
        AUDIT_ROOT,
        EDGE_ROOT,
        GRAPH_ROOT,
        MAXMIND_DB,
        NODE_ROOT,
        OUT_ROOT,
        SOURCE_ROOT,
        VERSION_ROOT,
        NodeStore,
        archive_file,
        assert_no_all_zero_columns,
        clean_value,
        join_unique,
        load_maxmind_reader,
        lookup_asn,
        parse_as_bool,
        parse_san_values,
        read_csv,
        sha256_file,
        write_csv,
        write_json,
    )
    from .edge_builders.build_hosted_on import build_hosted_on_edges
    from .edge_builders.build_redirects_to import build_redirect_audit
    from .edge_builders.build_referenced_by import build_referenced_by_edges
    from .edge_builders.build_registered_via import build_registered_via_edges
    from .edge_builders.build_uses_cert import build_uses_cert_edges
    from .edge_builders.build_uses_ns import build_uses_ns_edges
    from .node_builders.build_cert_nodes import build_cert_nodes
    from .node_builders.build_extref_nodes import build_extref_nodes
    from .node_builders.build_ip_nodes import build_ip_nodes
    from .node_builders.build_ns_nodes import build_ns_nodes
    from .node_builders.build_registrar_nodes import build_registrar_nodes
    from .node_builders.build_website_nodes import build_website_nodes, load_site_rows


NODE_ORDER = ["Website", "IP", "Certificate", "NameServer", "Registrar", "ExternalReference"]
FORWARD_EDGE_TYPES = ["hosted_on", "uses_cert", "uses_ns", "registered_via", "referenced_by"]
FORWARD_EDGE_TRIPLES = {
    "hosted_on": ("Website", "hosted_on", "IP"),
    "uses_cert": ("Website", "uses_cert", "Certificate"),
    "uses_ns": ("Website", "uses_ns", "NameServer"),
    "registered_via": ("Website", "registered_via", "Registrar"),
    "referenced_by": ("Website", "referenced_by", "ExternalReference"),
}
REVERSE_EDGE_TRIPLES = {
    f"rev_{edge_type}": (dst, f"rev_{edge_type}", src)
    for edge_type, (src, _, dst) in FORWARD_EDGE_TRIPLES.items()
}

CONFIG = {
    "features": {
        "asn_vocab_size": 16,
        "issuer_family_vocab_size": 16,
        "zscore_epsilon": 1e-8,
        "drop_all_zero_columns": True,
    },
    "edges": {
        "use_edge_weight": True,
        "edge_weight_fn": "log1p",
        "extref_hub_policy": "inverse_dst_degree",
        "extref_hub_degree_threshold": 20,
        "reverse_edge_policy": "explicit_materialized",
    },
    "dedup": {
        "strict_pair_uniqueness": True,
        "forbid_raw_duplicates_in_training": True,
    },
    "redirects": {
        "include_in_graph": False,
        "audit_only": True,
    },
    "training": {
        "exclude_isolated_websites": True,
        "validate_no_all_zero_feature_columns": True,
    },
}


def provider_hint(nameserver: str) -> str:
    ns = clean_value(nameserver).lower()
    if "cloudflare" in ns:
        return "cloudflare"
    if "akamai" in ns:
        return "akamai"
    if "nsone" in ns:
        return "nsone"
    if "awsdns" in ns or "amazon" in ns:
        return "aws"
    if "google" in ns:
        return "google"
    return "other"


def collect_raw_graph_inputs(source_to_website: dict[str, str]) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame], dict[str, Any]]:
    reader = load_maxmind_reader()
    stores = {
        "IP": NodeStore("IP", "ip"),
        "Certificate": NodeStore("CERT", "cert_hash"),
        "NameServer": NodeStore("NS", "nameserver"),
        "Registrar": NodeStore("REG", "registrar"),
        "ExternalReference": NodeStore("EXT", "external_ref"),
    }
    edge_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    tls_site_metadata: dict[str, dict[str, str]] = defaultdict(dict)
    asn_hit_count = 0

    def add_edge(edge_type: str, src: str, dst: str, source_dataset: str, source_row: int) -> None:
        if not src or not dst:
            return
        edge_rows[edge_type].append(
            {
                "src_id": src,
                "dst_id": dst,
                "source_dataset": source_dataset,
                "source_row": source_row,
            }
        )

    def add_ip(value: str) -> str:
        nonlocal asn_hit_count
        ip_value = clean_value(value)
        asn, org, is_private, is_global = lookup_asn(reader, ip_value)
        if asn:
            asn_hit_count += 1
        return stores["IP"].get(ip_value, asn_number=asn, asn_org=org, is_private=is_private, is_global=is_global)

    infra_sources = [
        (SOURCE_ROOT / "phase1" / "core" / "infra_relation.csv", "phase1_infra_relation"),
        (SOURCE_ROOT / "phase2" / "followup" / "illegal_infra_relation.csv", "phase2_illegal_infra_relation"),
        (SOURCE_ROOT / "phase2" / "region_expansion" / "region_expansion_infra_relation.csv", "region_expansion_infra_relation"),
    ]
    for path, dataset in infra_sources:
        df = read_csv(path)
        if df.empty:
            continue
        for row_idx, row in df.iterrows():
            site_id = clean_value(row.get("site_id"))
            website_id = source_to_website.get(site_id, "")
            if not website_id:
                continue
            status = clean_value(row.get("retrieval_status")).lower()
            if status.startswith("failed"):
                continue
            infra_type = clean_value(row.get("infra_type")).lower()
            value = clean_value(row.get("infra_value"))
            if not value:
                continue
            if infra_type == "tls_issuer":
                tls_site_metadata[site_id]["region_tls_issuer"] = value
                continue
            if infra_type == "ip":
                add_edge("hosted_on", website_id, add_ip(value), dataset, row_idx)
            elif infra_type == "tls_cert":
                cert_id = stores["Certificate"].get(
                    value,
                    source_hint=dataset,
                    tls_issuer_org="",
                    region_tls_issuer="",
                    tls_issuer_cn="",
                    tls_common_name="",
                    tls_san_values="",
                    tls_san_count="",
                    tls_wildcard_san_count="",
                    tls_subject_cn="",
                )
                add_edge("uses_cert", website_id, cert_id, dataset, row_idx)
            elif infra_type == "ns":
                ns_value = clean_value(value).rstrip(".").lower()
                ns_id = stores["NameServer"].get(ns_value, provider_hint=provider_hint(ns_value))
                add_edge("uses_ns", website_id, ns_id, dataset, row_idx)
            elif infra_type == "registrar":
                registrar_id = stores["Registrar"].get(value, is_numeric=str(value).isdigit())
                add_edge("registered_via", website_id, registrar_id, dataset, row_idx)

    tls_followup = read_csv(SOURCE_ROOT / "phase2" / "followup" / "illegal_tls_certificate_detail.csv")
    for row_idx, row in tls_followup.iterrows():
        website_id = source_to_website.get(clean_value(row.get("site_id")), "")
        cert_hash = clean_value(row.get("tls_fingerprint"))
        if not website_id or not cert_hash:
            continue
        sans = parse_san_values(row.get("tls_san_values"))
        wildcard_count = sum(1 for value in sans if clean_value(value).startswith("*."))
        cert_id = stores["Certificate"].get(
            cert_hash,
            source_hint="phase2_illegal_tls_detail",
            tls_issuer_org="",
            region_tls_issuer="",
            tls_issuer_cn="",
            tls_common_name=clean_value(row.get("tls_common_name")),
            tls_san_values="|".join(sans),
            tls_san_count=str(len(sans)),
            tls_wildcard_san_count=str(wildcard_count),
            tls_subject_cn=clean_value(row.get("tls_common_name")),
        )
        add_edge("uses_cert", website_id, cert_id, "phase2_illegal_tls_detail", row_idx)

    tls_region = read_csv(SOURCE_ROOT / "phase2" / "region_expansion" / "region_expansion_tls_detail.csv")
    for row_idx, row in tls_region.iterrows():
        site_id = clean_value(row.get("site_id"))
        website_id = source_to_website.get(site_id, "")
        if not website_id:
            continue
        if parse_as_bool(row.get("dns_success")) and clean_value(row.get("tls_target_ip")):
            add_edge("hosted_on", website_id, add_ip(clean_value(row.get("tls_target_ip"))), "region_expansion_tls_detail", row_idx)
        if parse_as_bool(row.get("tls_success")) and clean_value(row.get("tls_cert_hash")):
            site_meta = tls_site_metadata.get(site_id, {})
            cert_id = stores["Certificate"].get(
                clean_value(row.get("tls_cert_hash")),
                source_hint="region_expansion_tls_detail",
                tls_issuer_org=clean_value(row.get("tls_issuer_org")),
                region_tls_issuer=clean_value(site_meta.get("region_tls_issuer", "")),
                tls_issuer_cn=clean_value(row.get("tls_issuer_cn")),
                tls_common_name=clean_value(row.get("tls_subject_cn")),
                tls_san_values="",
                tls_san_count=clean_value(row.get("tls_san_count")),
                tls_wildcard_san_count=clean_value(row.get("tls_wildcard_san_count")),
                tls_subject_cn=clean_value(row.get("tls_subject_cn")),
            )
            add_edge("uses_cert", website_id, cert_id, "region_expansion_tls_detail", row_idx)

    search_refs = read_csv(SOURCE_ROOT / "phase2" / "evidence" / "search_discovery_registry.csv")
    for row_idx, row in search_refs.iterrows():
        source_id = clean_value(row.get("source_id")) or f"search_row_{row_idx}"
        ext_id = stores["ExternalReference"].get(
            f"search::{source_id}",
            external_ref_type="search_discovery",
            source_cluster=clean_value(row.get("source_cluster")),
            page_domain=clean_value(row.get("page_domain")),
            page_url=clean_value(row.get("page_url")),
        )
        candidates = [clean_value(row.get("derived_sample_id"))]
        derived_ids = clean_value(row.get("derived_sample_ids"))
        if derived_ids:
            candidates.extend(derived_ids.split("|"))
        for sample_id in candidates:
            website_id = source_to_website.get(clean_value(sample_id), "")
            if website_id:
                add_edge("referenced_by", website_id, ext_id, "search_discovery_registry", row_idx)
                break

    history_refs = read_csv(SOURCE_ROOT / "phase2" / "evidence" / "history_osint_evidence.csv")
    for row_idx, row in history_refs.iterrows():
        sample_id = clean_value(row.get("sample_id"))
        website_id = source_to_website.get(sample_id, "") or source_to_website.get(clean_value(row.get("entity_id")), "")
        archived_url = clean_value(row.get("archived_url")) or f"history_row_{row_idx}"
        ext_id = stores["ExternalReference"].get(
            f"history::{archived_url}",
            external_ref_type="archive_history",
            source_cluster=clean_value(row.get("source_cluster")),
            page_domain="",
            page_url=archived_url,
            snapshot_date=clean_value(row.get("snapshot_date")),
        )
        if website_id:
            add_edge("referenced_by", website_id, ext_id, "history_osint_evidence", row_idx)

    if reader is not None:
        reader.close()

    node_frames = {name: store.to_frame() for name, store in stores.items()}
    edge_frames = {name: pd.DataFrame(rows) for name, rows in edge_rows.items()}
    return node_frames, edge_frames, {"maxmind_asn_hits": asn_hit_count}


def build_node_tables(
    raw_node_frames: dict[str, pd.DataFrame],
    website_nodes: pd.DataFrame,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    zscore_epsilon = CONFIG["features"]["zscore_epsilon"]
    drop_all_zero = CONFIG["features"]["drop_all_zero_columns"]

    node_frames: dict[str, pd.DataFrame] = {"Website": website_nodes.copy().sort_values("node_id").reset_index(drop=True)}
    website_state = {
        "active_feature_columns": [col for col in node_frames["Website"].columns if col.startswith("feat_")],
        "dropped_columns": [],
    }
    if not website_state["active_feature_columns"]:
        raise ValueError("Website node builder did not emit active feat_* columns.")

    ip_nodes, ip_state = build_ip_nodes(
        raw_node_frames.get("IP", pd.DataFrame()), asn_vocab_size=CONFIG["features"]["asn_vocab_size"], drop_all_zero_columns=drop_all_zero
    )
    cert_nodes, cert_state = build_cert_nodes(
        raw_node_frames.get("Certificate", pd.DataFrame()),
        issuer_family_vocab_size=CONFIG["features"]["issuer_family_vocab_size"],
        drop_all_zero_columns=drop_all_zero,
    )
    ns_nodes, ns_state = build_ns_nodes(
        raw_node_frames.get("NameServer", pd.DataFrame()), zscore_epsilon=zscore_epsilon, drop_all_zero_columns=drop_all_zero
    )
    registrar_nodes, registrar_state = build_registrar_nodes(
        raw_node_frames.get("Registrar", pd.DataFrame()), zscore_epsilon=zscore_epsilon, drop_all_zero_columns=drop_all_zero
    )
    extref_nodes, extref_state = build_extref_nodes(
        raw_node_frames.get("ExternalReference", pd.DataFrame()), drop_all_zero_columns=drop_all_zero
    )

    node_frames["IP"] = ip_nodes
    node_frames["Certificate"] = cert_nodes
    node_frames["NameServer"] = ns_nodes
    node_frames["Registrar"] = registrar_nodes
    node_frames["ExternalReference"] = extref_nodes

    for node_type in NODE_ORDER:
        frame = node_frames[node_type].copy().reset_index(drop=True)
        frame.insert(0, "node_index", range(len(frame)))
        node_frames[node_type] = frame

    feature_builder_state = {
        "graph_version": "v2",
        "zero_column_policy": "drop_and_fail_fast",
        "node_feature_builders": {
            "Website": website_state,
            "IP": ip_state,
            "Certificate": cert_state,
            "NameServer": ns_state,
            "Registrar": registrar_state,
            "ExternalReference": extref_state,
        },
        "edge_type_triples": {
            edge_type: list(triple) for edge_type, triple in {**FORWARD_EDGE_TRIPLES, **REVERSE_EDGE_TRIPLES}.items()
        },
        "extref_hub_policy": {
            "name": CONFIG["edges"]["extref_hub_policy"],
            "degree_threshold": CONFIG["edges"]["extref_hub_degree_threshold"],
        },
    }
    return node_frames, feature_builder_state


def build_forward_edges(
    raw_edge_frames: dict[str, pd.DataFrame],
    node_index: dict[str, dict[str, int]],
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    forward_edges = {
        "hosted_on": build_hosted_on_edges(raw_edge_frames.get("hosted_on", pd.DataFrame()), node_index),
        "uses_cert": build_uses_cert_edges(raw_edge_frames.get("uses_cert", pd.DataFrame()), node_index),
        "uses_ns": build_uses_ns_edges(raw_edge_frames.get("uses_ns", pd.DataFrame()), node_index),
        "registered_via": build_registered_via_edges(raw_edge_frames.get("registered_via", pd.DataFrame()), node_index),
    }
    referenced_by, hub_audit = build_referenced_by_edges(
        raw_edge_frames.get("referenced_by", pd.DataFrame()),
        node_index,
        hub_degree_threshold=CONFIG["edges"]["extref_hub_degree_threshold"],
    )
    forward_edges["referenced_by"] = referenced_by
    return forward_edges, hub_audit


def annotate_website_isolation(
    website_nodes: pd.DataFrame,
    forward_edges: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    connected_websites = set()
    for edge_type in FORWARD_EDGE_TYPES:
        df = forward_edges.get(edge_type, pd.DataFrame())
        if not df.empty:
            connected_websites.update(df["src_id"].tolist())
    website_nodes = website_nodes.copy()
    website_nodes["is_isolated"] = (~website_nodes["node_id"].isin(connected_websites)).astype(int)
    website_nodes["exclude_from_training_default"] = website_nodes["is_isolated"]
    isolated_audit = website_nodes[website_nodes["is_isolated"] == 1][
        [
            "node_id",
            "website_id",
            "root_domain",
            "jurisdiction",
            "sample_bucket",
            "sample_tier",
            "label_illegal",
            "label_licensed",
            "label_gray",
            "label_control",
        ]
    ].copy()
    return website_nodes, isolated_audit


def build_edge_dedup_summary(raw_edge_frames: dict[str, pd.DataFrame], forward_edges: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for edge_type in FORWARD_EDGE_TYPES:
        raw_df = raw_edge_frames.get(edge_type, pd.DataFrame())
        dedup_df = forward_edges.get(edge_type, pd.DataFrame())
        edge_count_sum = int(dedup_df["edge_count_raw"].sum()) if not dedup_df.empty else 0
        rows.append(
            {
                "edge_type": edge_type,
                "raw_rows": int(len(raw_df)),
                "unique_pairs": int(len(dedup_df)),
                "duplicates_removed": int(len(raw_df) - len(dedup_df)),
                "sum_edge_count_raw": edge_count_sum,
                "edge_weight_fn": "log1p/inverse_dst_degree" if edge_type == "referenced_by" else CONFIG["edges"]["edge_weight_fn"],
            }
        )
    return pd.DataFrame(rows)


def build_graph_arrays(
    node_frames: dict[str, pd.DataFrame],
    forward_edges: dict[str, pd.DataFrame],
    feature_builder_state: dict[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, Any], dict[str, int]]:
    arrays: dict[str, np.ndarray] = {}
    feature_schema: dict[str, Any] = {}
    materialized_edge_counts: dict[str, int] = {}

    for node_type in NODE_ORDER:
        df = node_frames[node_type]
        feature_columns = feature_builder_state["node_feature_builders"][node_type]["active_feature_columns"]
        matrix = df[feature_columns].to_numpy(dtype=np.float32)
        if CONFIG["training"]["validate_no_all_zero_feature_columns"]:
            assert_no_all_zero_columns(matrix, feature_columns, node_type)
        arrays[f"x__{node_type}"] = matrix
        arrays[f"node_id__{node_type}"] = df["node_id"].astype(str).to_numpy()
        feature_schema[node_type] = {
            "feature_columns": feature_columns,
            "feature_dim": len(feature_columns),
            "dropped_columns": feature_builder_state["node_feature_builders"][node_type].get("dropped_columns", []),
        }

    for edge_type in FORWARD_EDGE_TYPES:
        df = forward_edges.get(edge_type, pd.DataFrame())
        if df.empty:
            edge_index = np.zeros((2, 0), dtype=np.int64)
            edge_weight = np.zeros((0,), dtype=np.float32)
        else:
            edge_index = df[["src_index", "dst_index"]].to_numpy(dtype=np.int64).T
            edge_weight = df["edge_weight"].to_numpy(dtype=np.float32)
        arrays[f"edge_index__{edge_type}"] = edge_index
        arrays[f"edge_weight__{edge_type}"] = edge_weight
        materialized_edge_counts[edge_type] = int(edge_index.shape[1])

        rev_edge_type = f"rev_{edge_type}"
        arrays[f"edge_index__{rev_edge_type}"] = edge_index[::-1, :].copy()
        arrays[f"edge_weight__{rev_edge_type}"] = edge_weight.copy()
        materialized_edge_counts[rev_edge_type] = int(edge_index.shape[1])

    return arrays, feature_schema, materialized_edge_counts


def export_graph_package(
    arrays: dict[str, np.ndarray],
    node_frames: dict[str, pd.DataFrame],
    feature_builder_state: dict[str, Any],
    feature_schema: dict[str, Any],
    metadata: dict[str, Any],
) -> tuple[Path, Path | None]:
    GRAPH_ROOT.mkdir(parents=True, exist_ok=True)
    VERSION_ROOT.mkdir(parents=True, exist_ok=True)

    builder_state_path = VERSION_ROOT / "feature_builder_state.json"
    write_json(feature_builder_state, builder_state_path)

    npz_path = GRAPH_ROOT / "hetero_graph_v2.npz"
    np.savez_compressed(npz_path, **arrays)

    pt_path: Path | None = None
    pt_status = {"available": False, "reason": "torch/torch_geometric not installed in active runtime"}
    try:
        import torch
        from torch_geometric.data import HeteroData

        data = HeteroData()
        for node_type in NODE_ORDER:
            data[node_type].x = torch.tensor(arrays[f"x__{node_type}"], dtype=torch.float32)
            data[node_type].node_id = arrays[f"node_id__{node_type}"].astype(str).tolist()

        for edge_type, triple in feature_builder_state["edge_type_triples"].items():
            edge_index_key = f"edge_index__{edge_type}"
            edge_weight_key = f"edge_weight__{edge_type}"
            data[tuple(triple)].edge_index = torch.tensor(arrays[edge_index_key], dtype=torch.long)
            data[tuple(triple)].edge_weight = torch.tensor(arrays[edge_weight_key], dtype=torch.float32)

        pt_path = GRAPH_ROOT / "hetero_graph_v2.pt"
        torch.save(data, pt_path)
        marker = GRAPH_ROOT / "hetero_graph_v2.pt.NOT_GENERATED.txt"
        if marker.exists():
            marker.unlink()
        pt_status = {"available": True, "path": str(pt_path)}
    except Exception as exc:
        marker = GRAPH_ROOT / "hetero_graph_v2.pt.NOT_GENERATED.txt"
        marker.write_text(
            "PyG export was skipped because the active runtime could not complete torch_geometric export.\n"
            f"Graph package: {npz_path.name}\n"
            f"Reason: {type(exc).__name__}: {exc}\n",
            encoding="utf-8",
        )

    write_json(feature_schema, OUT_ROOT / "node_feature_schema.json")

    metadata["builder_state_path"] = str(builder_state_path)
    metadata["feature_schema"] = feature_schema
    metadata["graph_npz"] = str(npz_path)
    metadata["pyg_export"] = pt_status
    metadata_path = GRAPH_ROOT / "hetero_graph_v2_metadata.json"
    write_json(metadata, metadata_path)
    return npz_path, pt_path


def archive_previous_outputs() -> None:
    graph_archive = GRAPH_ROOT / "archived"
    version_archive = VERSION_ROOT / "archived"
    edge_archive = EDGE_ROOT / "archived"
    for path in GRAPH_ROOT.glob("hetero_graph_v1*"):
        archive_file(path, graph_archive)
    for path in VERSION_ROOT.glob("hetero_graph_v1*"):
        archive_file(path, version_archive)
    archive_file(EDGE_ROOT / "redirects_to.csv", edge_archive)


def main() -> None:
    for path in [NODE_ROOT, EDGE_ROOT, GRAPH_ROOT, VERSION_ROOT, AUDIT_ROOT]:
        path.mkdir(parents=True, exist_ok=True)

    site_rows = load_site_rows()
    if site_rows.empty:
        raise SystemExit("No site rows found. Check source_materials/data.")

    website_nodes, duplicate_audit, tier_backfill_audit, source_to_website, domain_to_website, website_builder_state = build_website_nodes(
        site_rows,
        zscore_epsilon=CONFIG["features"]["zscore_epsilon"],
        drop_all_zero_columns=CONFIG["features"]["drop_all_zero_columns"],
    )
    raw_node_frames, raw_edge_frames, aux_counts = collect_raw_graph_inputs(source_to_website)
    node_frames, feature_builder_state = build_node_tables(raw_node_frames, website_nodes)
    feature_builder_state["node_feature_builders"]["Website"] = website_builder_state

    node_index = {
        node_type: dict(zip(frame["node_id"], frame["node_index"]))
        for node_type, frame in node_frames.items()
        if not frame.empty
    }
    forward_edges, hub_audit = build_forward_edges(raw_edge_frames, node_index)
    website_with_isolation, isolated_audit = annotate_website_isolation(node_frames["Website"], forward_edges)
    node_frames["Website"] = website_with_isolation
    node_frames["Website"] = node_frames["Website"].sort_values("node_index").reset_index(drop=True)

    if not hub_audit.empty:
        hub_audit = hub_audit.merge(
            node_frames["ExternalReference"][["node_id", "external_ref", "external_ref_type", "source_cluster", "page_url"]],
            left_on="dst_id",
            right_on="node_id",
            how="left",
        ).drop(columns=["node_id"])

    redirect_rows = read_csv(SOURCE_ROOT / "phase1" / "core" / "redirect_relation.csv")
    redirect_audit = build_redirect_audit(redirect_rows, source_to_website, domain_to_website)
    edge_dedup_summary = build_edge_dedup_summary(raw_edge_frames, forward_edges)

    write_csv(site_rows, OUT_ROOT / "site_registry_input_rows_audit.csv")
    write_csv(tier_backfill_audit, AUDIT_ROOT / "website_tier_backfill_audit.csv")
    write_csv(duplicate_audit, OUT_ROOT / "duplicate_audit.csv")
    write_csv(redirect_audit, AUDIT_ROOT / "redirect_audit.csv")
    write_csv(edge_dedup_summary, AUDIT_ROOT / "edge_dedup_summary.csv")
    write_csv(isolated_audit, AUDIT_ROOT / "isolated_website_nodes.csv")
    write_csv(hub_audit, AUDIT_ROOT / "extref_hub_audit.csv")

    for node_type in NODE_ORDER:
        write_csv(node_frames[node_type], NODE_ROOT / f"{node_type}.csv")
    write_csv(node_frames["Website"], OUT_ROOT / "master_site_registry.csv")

    for edge_type in FORWARD_EDGE_TYPES:
        write_csv(forward_edges[edge_type], EDGE_ROOT / f"{edge_type}.csv")

    arrays, feature_schema, materialized_edge_counts = build_graph_arrays(node_frames, forward_edges, feature_builder_state)

    node_counts = {node_type: int(len(node_frames[node_type])) for node_type in NODE_ORDER}
    forward_edge_counts = {edge_type: int(len(forward_edges[edge_type])) for edge_type in FORWARD_EDGE_TYPES}
    raw_edge_counts = {edge_type: int(len(raw_edge_frames.get(edge_type, pd.DataFrame()))) for edge_type in FORWARD_EDGE_TYPES}
    write_csv(pd.DataFrame([{"node_type": k, "count": v} for k, v in node_counts.items()]), OUT_ROOT / "node_stats.csv")
    write_csv(pd.DataFrame([{"edge_type": k, "count": v} for k, v in forward_edge_counts.items()]), OUT_ROOT / "edge_stats.csv")

    edge_schema = {
        "semantic_relations": {
            edge_type: {
                "src": triple[0],
                "dst": triple[2],
                "materialized_reverse": f"rev_{edge_type}",
            }
            for edge_type, triple in FORWARD_EDGE_TRIPLES.items()
        },
        "pyg_materialized_reverse_edges": {
            edge_type: {"src": triple[0], "dst": triple[2], "reverse_of": edge_type.replace("rev_", "", 1)}
            for edge_type, triple in REVERSE_EDGE_TRIPLES.items()
        },
        "redirect_policy": {
            "include_in_graph": False,
            "audit_only": True,
        },
    }
    write_json(edge_schema, OUT_ROOT / "edge_schema.json")

    source_inputs = {
        str(path.relative_to(OUT_ROOT.parents[0])): int(len(read_csv(path)))
        for path in [
            SOURCE_ROOT / "phase1" / "core" / "site_profile.csv",
            SOURCE_ROOT / "phase1" / "core" / "infra_relation.csv",
            SOURCE_ROOT / "phase1" / "core" / "redirect_relation.csv",
            SOURCE_ROOT / "phase2" / "followup" / "illegal_infra_relation.csv",
            SOURCE_ROOT / "phase2" / "followup" / "illegal_tls_certificate_detail.csv",
            SOURCE_ROOT / "phase2" / "region_expansion" / "region_expansion_site_profile.csv",
            SOURCE_ROOT / "phase2" / "region_expansion" / "region_expansion_infra_relation.csv",
            SOURCE_ROOT / "phase2" / "region_expansion" / "region_expansion_tls_detail.csv",
            SOURCE_ROOT / "phase2" / "evidence" / "search_discovery_registry.csv",
            SOURCE_ROOT / "phase2" / "evidence" / "history_osint_evidence.csv",
        ]
        if path.exists()
    }
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "graph_version": "v2",
        "schema": "v2_6_nodes_5_relations_10_materialized_edges",
        "promotion_account_status": "archived_excluded",
        "node_counts": node_counts,
        "forward_unique_edge_counts": forward_edge_counts,
        "forward_raw_edge_counts": raw_edge_counts,
        "materialized_edge_counts": materialized_edge_counts,
        "duplicate_root_domain_groups": int(len(duplicate_audit)),
        "source_inputs": source_inputs,
        "maxmind": {
            "db_path": str(MAXMIND_DB),
            "db_sha256": sha256_file(MAXMIND_DB) if MAXMIND_DB.exists() else "",
            "asn_hits": aux_counts.get("maxmind_asn_hits", 0),
        },
        "redirect_audit": {
            row["stage"]: int(row["count"]) for row in redirect_audit.to_dict(orient="records")
        },
        "isolated_websites": {
            "count": int(len(isolated_audit)),
            "training_default_excluded": True,
        },
        "config": CONFIG,
    }
    npz_path, pt_path = export_graph_package(arrays, node_frames, feature_builder_state, feature_schema, metadata)

    checksum_targets = [npz_path, GRAPH_ROOT / "hetero_graph_v2_metadata.json", VERSION_ROOT / "feature_builder_state.json"]
    if pt_path is not None:
        checksum_targets.append(pt_path)
    checksum_payload = "\n".join(f"{path.name}:{sha256_file(path)}" for path in checksum_targets)
    checksum_path = VERSION_ROOT / "hetero_graph_v2.sha256"
    checksum_path.write_text(checksum_payload + "\n", encoding="ascii")

    archive_previous_outputs()

    version_log = VERSION_ROOT / "version_log.md"
    with version_log.open("a", encoding="utf-8") as fh:
        fh.write(
            f"- {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Graph v2 build: "
            f"nodes={sum(node_counts.values())}, semantic_edges={sum(forward_edge_counts.values())}, "
            f"pyg_edges={sum(materialized_edge_counts.values())}, npz={npz_path.name}, pt={'yes' if pt_path else 'no'}\n"
        )

    print("hetero_graph_v2_build_complete")
    print(f"node_counts={json.dumps(node_counts, ensure_ascii=False)}")
    print(f"forward_edge_counts={json.dumps(forward_edge_counts, ensure_ascii=False)}")
    print(f"materialized_edge_counts={json.dumps(materialized_edge_counts, ensure_ascii=False)}")
    print(f"graph_npz={npz_path}")
    print(f"pyg_export={'generated' if pt_path else 'skipped_missing_torch_or_pyg'}")
    print(f"checksum={checksum_path}")


if __name__ == "__main__":
    main()
