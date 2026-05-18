# Week 3 Graph Build Summary

Build date: 2026-04-24

Workspace: `<workspace_root>`

## Objective

Rebuild the PA-free heterogeneous graph as Graph v2, removing label leakage, deleting `redirects_to` from the formal graph, deduplicating all active edges, materializing reverse PyG edges, and aligning all outputs to the `6 nodes / 5 semantic relations / 10 PyG edge types` schema.

## Inputs

- `source_materials/data/phase1/core/site_profile.csv`
- `source_materials/data/phase1/core/infra_relation.csv`
- `source_materials/data/phase1/core/redirect_relation.csv`
- `source_materials/data/phase2/followup/illegal_infra_relation.csv`
- `source_materials/data/phase2/followup/illegal_tls_certificate_detail.csv`
- `source_materials/data/phase2/region_expansion/region_expansion_site_profile.csv`
- `source_materials/data/phase2/region_expansion/region_expansion_infra_relation.csv`
- `source_materials/data/phase2/region_expansion/region_expansion_tls_detail.csv`
- `source_materials/data/phase2/evidence/search_discovery_registry.csv`
- `source_materials/data/phase2/evidence/history_osint_evidence.csv`

PromotionAccount archive files are not active inputs.

## Output Scale

Node counts:

- Website: 623
- IP: 1325
- Certificate: 510
- NameServer: 959
- Registrar: 79
- ExternalReference: 53

Forward unique edge counts:

- hosted_on: 1516
- uses_cert: 519
- uses_ns: 1662
- registered_via: 252
- referenced_by: 57

Materialized PyG edge counts:

- hosted_on / rev_hosted_on: 1516 each
- uses_cert / rev_uses_cert: 519 each
- uses_ns / rev_uses_ns: 1662 each
- registered_via / rev_registered_via: 252 each
- referenced_by / rev_referenced_by: 57 each

## Key Validation Points

- `Website.x = (623, 7)` with no `label_*` feature columns.
- All active forward edges are unique `(src,dst)` pairs and retain `edge_count_raw` plus `edge_weight`.
- `redirects_to` is audit-only: `166 -> 11 -> 7 -> 0`.
- `referenced_by` uses `log1p(edge_count_raw) / dst_degree` to reduce hub dominance.
- `38` isolated Website nodes remain in the graph but are marked `exclude_from_training_default = 1`.
- Graph v1 outputs were archived instead of overwritten.

## Main Artifacts

- `data/master_site_registry.csv`
- `data/processed/nodes/*.csv`
- `data/processed/edges/*.csv`
- `data/audits/redirect_audit.csv`
- `data/audits/edge_dedup_summary.csv`
- `data/audits/isolated_website_nodes.csv`
- `data/audits/extref_hub_audit.csv`
- `data/audits/website_tier_backfill_audit.csv`
- `data/graphs/hetero_graph_v2.npz`
- `data/graphs/hetero_graph_v2.pt`
- `data/graphs/hetero_graph_v2_metadata.json`
- `data/versions/feature_builder_state.json`
- `data/versions/hetero_graph_v2.sha256`

## Validation

- `src/data/build_hetero_graph.py` completed successfully.
- `src/data/convert_npz_to_pyg.py` uses `feature_builder_state.json` to attach `edge_weight` to the correct PyG edge triples.
- `data/graphs/hetero_graph_v2_metadata.json` records the final schema and audit counts.
- Native PyG export succeeded with explicit reverse edges.
