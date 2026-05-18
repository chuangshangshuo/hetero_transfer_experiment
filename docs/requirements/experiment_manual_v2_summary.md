# 实验执行手册 v2 摘要（Graph v2）

## 核心决策

- PromotionAccount 正式归档，不进入主图。
- 主图改为 `6` 类节点、`5` 条语义边。
- `redirects_to` 改为 audit-only，不进入正式图建模。
- 训练产物显式物化 `5` 条反向边，因此 `.pt/.npz` 共 `10` 个 edge types。

## 6 类节点

- Website
- IP
- Certificate
- NameServer
- Registrar
- ExternalReference

## 5 条语义边

- `hosted_on`
- `uses_cert`
- `uses_ns`
- `registered_via`
- `referenced_by`

## 关键 Week 3 口径

- Website.x 仅 7 维结构特征，不允许任何 `label_*` 特征入模。
- 所有 active edge 都先按 `(src,dst)` 去重，再保留 `edge_count_raw` 和 `edge_weight`。
- `referenced_by` 采用 `log1p(edge_count_raw) / dst_degree`。
- redirect 审计固定为 `166 -> 11 -> 7 -> 0`。
- 孤立 Website 节点 38 个，默认训练剔除。

## Week 3 主产物

- `data/graphs/hetero_graph_v2.pt`
- `data/graphs/hetero_graph_v2.npz`
- `data/graphs/hetero_graph_v2_metadata.json`
- `data/versions/feature_builder_state.json`
- `data/versions/hetero_graph_v2.sha256`

## 验收基线

- Website: 623
- Website.x: `(623, 7)`
- hosted_on: 1516
- uses_cert: 519
- uses_ns: 1662
- registered_via: 252
- referenced_by: 57

## Week 4 Addendum

- Week 4 graph model is the `HeteroConv baseline`; it is not `SeHGNN` or `HINormer`.
- Week 5 HeCo contrastive learning must be implemented as an incremental method over this baseline, so the paper ablation table can use the Week 4 result as the `Baseline` row.
- Pooled primary splits are now group-aware by `operator_or_case`; blank `operator_or_case` values fall back to `root_domain`.
- Group-aware split audit output: `output/week4/audits/pooled_primary_group_aware_split_audit.csv`.
- Reaggregated group-aware score output: `output/week4/metrics/pooled_primary_group_aware_reaggregated_summary.csv`.
- Reaggregated group-aware metrics reuse existing Week 4 prediction scores and do not represent retrained group-aware models.
- Philippines same-region HeteroConv acceptance threshold is `ROC-AUC >= 0.80`; the older phase2 `0.88` threshold is not used for Graph v2 / v4-tier Week 4 acceptance.
- `illegal_confirmed_official_cross_verified` primary-layer evaluation should be handled as a dedicated Denmark same-region experiment rather than over-interpreting pooled split counts.
