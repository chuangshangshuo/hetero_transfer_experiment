# Workspace Manifest v2

Generated: 2026-04-24

## Workspace

`<workspace_root>`

## Active Graph Definition

Nodes:

- `Website`
- `IP`
- `Certificate`
- `NameServer`
- `Registrar`
- `ExternalReference`

Semantic relations:

- `hosted_on`
- `uses_cert`
- `uses_ns`
- `registered_via`
- `referenced_by`

PyG materialized reverse edges:

- `rev_hosted_on`
- `rev_uses_cert`
- `rev_uses_ns`
- `rev_registered_via`
- `rev_referenced_by`

Redirect data is audit-only and does not enter the active graph.

## Key Build Outputs

- `data/graphs/hetero_graph_v2.npz`
- `data/graphs/hetero_graph_v2.pt`
- `data/graphs/hetero_graph_v2_metadata.json`
- `data/versions/feature_builder_state.json`
- `data/versions/hetero_graph_v2.sha256`

Legacy `hetero_graph_v1*` files are archived under `data/graphs/archived/` and `data/versions/archived/`.

## Paper Interpretation Boundary

Final paper-safe Week 7/8 interpretation should follow `docs/evidence_boundary_table.md` and the new `sec_week7/8` draft files. Older `paper/draft/sec4_3_week7_transfer.md` and `paper/draft/sec4_4_week8_mechanism.md` are historical drafts and are marked superseded for interpretation.
