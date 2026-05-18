# Data Version Log

## hetero_graph_v1

- status: planned
- schema version: v2
- created: pending
- expected node count: 3260 +/- 5%
- expected edge count: 5063
- node types: Website, IP, Certificate, NameServer, Registrar, ExternalReference
- edge types: hosted_on, uses_cert, uses_ns, registered_via, redirects_to, referenced_by
- excluded main-graph dimensions: PromotionAccount, App, Brand, ASN
- PA archive: `<workspace_parent>\phase2\data\promotion_collection_archived`
- MaxMind input: `data/external/maxmind/GeoLite2-ASN.mmdb`
- expected graph output: `data/graphs/hetero_graph_v1.pt`
- expected checksum: `data/versions/hetero_graph_v1.sha256`

## Notes

Record the exact MaxMind checksum and source data snapshot timestamps before training Week 4 baselines.
- 2026-04-23 17:31:01 Week3 graph build: nodes=3549, edges=5834, websites=623, fallback_npz=hetero_graph_v1_fallback.npz, pyg_export=no
- 2026-04-23 20:21:48 Week3 graph build: nodes=3549, edges=5834, websites=623, fallback_npz=hetero_graph_v1_fallback.npz, pyg_export=no
- 2026-04-23 20:32:05 Week3 graph build: nodes=3549, edges=5834, websites=623, fallback_npz=hetero_graph_v1_fallback.npz, pyg_export=no
- 2026-04-23 20:55:46 Week3 graph build: nodes=3549, edges=5834, websites=623, fallback_npz=hetero_graph_v1_fallback.npz, pyg_export=no
- 2026-04-23 20:57:07 Week3 graph build: nodes=3549, edges=5834, websites=623, fallback_npz=hetero_graph_v1_fallback.npz, pyg_export=yes
- 2026-04-23 20:58:44 Week3 graph build: nodes=3549, edges=5834, websites=623, fallback_npz=hetero_graph_v1_fallback.npz, pyg_export=yes
- 2026-04-23 21:00:26 Week3 graph build: nodes=3549, edges=5834, websites=623, fallback_npz=hetero_graph_v1_fallback.npz, pyg_export=yes
- 2026-04-24 09:06:35 Graph v2 build: nodes=3549, semantic_edges=4006, pyg_edges=8012, npz=hetero_graph_v2.npz, pt=no
- 2026-04-24 09:08:48 Graph v2 build: nodes=3549, semantic_edges=4006, pyg_edges=8012, npz=hetero_graph_v2.npz, pt=yes
- 2026-04-24 09:15:42 Graph v2 build: nodes=3549, semantic_edges=4006, pyg_edges=8012, npz=hetero_graph_v2.npz, pt=yes
