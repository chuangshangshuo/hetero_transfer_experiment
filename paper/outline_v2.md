# Paper Outline v2

This outline is aligned to Graph v2 and the completed Week 4/Week 5/Week 6/Week 7 experiments.

## Section 3.1 Graph Schema

Use the final Graph v2 wording:

- `6` semantic node types:
  - Website
  - IP
  - Certificate
  - NameServer
  - Registrar
  - ExternalReference
- `5` semantic relations:
  - `hosted_on`
  - `uses_cert`
  - `uses_ns`
  - `registered_via`
  - `referenced_by`
- `10` materialized PyG edge types:
  - the `5` forward semantic relations above
  - `5` explicit `rev_*` reverse edges for message passing

Explicitly state that:

- `redirects_to` is not part of the formal graph
- redirect evidence is audit-only through `redirect_audit.csv`
- PromotionAccount is archived and excluded from the active graph

## Section 3.3 Metapath and Training Objective

Core metapaths:

- `M1 = W-IP-W`
- `M2 = W-Cert-W`
- `M3 = W-NS-W`
- `M4 = W-ExtRef-W`

Extended metapaths for later ablation only:

- `M5 = W-Registrar-W`
- `M6 = W-IP-NS-IP-W`

Do not mention:

- `W-redirect-W`
- `M7`
- any `6 edges` wording

## Week 4 Experimental Scope

Week 4 should be described as the baseline-launch stage rather than the full transfer-method matrix.

Formal Week 4 outputs:

- pooled primary baseline
- same-region sensitivity baseline
- first transfer experiment: `Pooled excluding France -> France`

Week 4 model set:

- Website-only `Logistic Regression`
- Website-only `2-layer MLP`
- weighted supervised `HeteroConv baseline`
- transfer comparison: `source_only` vs `DANN`

Do not describe the Week 4 graph model as `SeHGNN` or `HINormer`. Those files remain method placeholders; the Week 4 graph baseline is the ablation-table `Baseline` row.

Week 5 should add HeCo contrastive learning incrementally on top of this HeteroConv baseline so that the paper can compare `Baseline` versus `Baseline + HeCo` cleanly.

## Week 5 HeCo Contrastive Scope

Week 5 implements a HeCo-style pretraining module with two views:

- schema view: weighted HeteroConv over the five Graph v2 semantic relations and their reverse edges
- metapath view: aggregation over `W-IP-W`, `W-Cert-W`, `W-NS-W`, and `W-ExtRef-W`

Week 5 design choices:

- positives are mined from top-k metapath co-occurrence, without using labels
- temperature grid is `{0.1, 0.3, 0.5, 0.7}`
- hard negative ratio is `30%`
- pretraining uses `200` epochs
- downstream evaluation includes frozen linear probing and full encoder fine-tuning

Week 5 headline results:

- best temperature by frozen validation ROC-AUC: `tau = 0.7`
- frozen encoder linear probe ROC-AUC: `0.9180`
- full fine-tune best ROC-AUC: `0.9897` at `tau = 0.3`
- t-SNE audit includes Denmark `tsars`, `icecasino`, and `verdecasino` family points

## Week 6 RQ1 Integration Scope

Week 6 integrates HeCo pretraining with supervised fine-tuning for the RQ1 main-result layer.

Week 6 model/result rows:

- `HeCo + Disc-LR finetune (5 seed)` reaches ROC-AUC `0.9322 ± 0.0716`
- `HeCo + uniform-LR finetune fallback (5 seed)` reaches ROC-AUC `0.9337 ± 0.0740`
- `Cross-verified primary layer evaluation` reaches pooled ROC-AUC `0.9798`

Interpretation boundary:

- The cross-verified evidence layer passes the predeclared RQ1 target.
- The integrated HeCo finetune model does not pass the predeclared mean/std/delta acceptance targets because seed46 is an outlier.
- The paper should not claim Week 6 finetuning as a locked improvement over HeteroGNN-only until the variance issue is resolved.

## Result Tables

Table candidates for the paper draft:

- Table 1: Graph v2 schema and node/edge counts
- Table 2: RQ1 ablation table from `output/week6/metrics/ablation_table.csv`
- Table 3: same-region sensitivity results for Belgium, France, and Philippines
- Table 4: France transfer comparison (`source_only` vs `DANN`)
- Table 5: group-aware pooled split audit and reaggregated score audit
- Table 6: Week 5 HeCo temperature grid and probe results
- Table 7: Week 6 acceptance audit from `output/week6/metrics/week6_acceptance_audit.csv`
- Table 8: Week 7 transfer method summary from `output/week7/metrics/transfer_method_summary.csv`
- Table 9: Week 7 pseudo-label quality and StruRW edge-weight audit summaries
- Table 10: Week 7 acceptance audit from `output/week7/metrics/week7_acceptance_audit.csv`
- Table 11: Week 7 direction-aware method deltas from `output/week7/metrics/transfer_delta_vs_source_only.csv`
- Table 12: Week 8 prerun mechanism acceptance audit from `output/week8/metrics/week8_acceptance_audit.csv`
- Table 13: Week 8.5 correction audit from `output/week85/metrics/week85_acceptance_audit.csv`
- Table 14: Corrected E5 diagnostics from `output/week8_patch/metrics/E5_corrected_acceptance_audit.csv`
- Table 15: Week 6 head-ablation summary from `output/week6_head_ablation/metrics/head_ablation_summary.csv`
- Table 16: Week 9 few-shot and hard-negative audit summaries from `output/week9/metrics/`
- Table 17: Final RQ evidence table from `output/week10/metrics/final_rq_evidence_table.csv`

Figure candidates:

- Figure 1: Graph v2 schema diagram (`6 node types / 5 semantic relations`)
- Figure 2: pooled primary ROC/PR summary
- Figure 3: France transfer comparison bar chart
- Figure 4: RQ1 pooled AUC bar chart from `figs/fig4_pooled_auc.pdf`
- Figure 5: HeCo fine-tune t-SNE by sample tier
- Figure 6: HeCo t-SNE Denmark family audit
- Figure 7: Week 7 transfer boxplot grid from `output/week7/plots/week7_transfer_boxplot_grid.png`
- Figure 8: Week 7 T3 DANN diagnostics from `output/week7/plots/week7_t3_dann_diagnostics.png`
- Figure 9: Week 7 direction-aware delta heatmap from `output/week7/plots/week7_delta_heatmap.png`
- Figure 10: Week 8 E1 edge ablation heatmap from `output/week8/plots/E1_edge_ablation_heatmap.png`
- Figure 11: Week 8 E3 family/random audit from `output/week8/plots/E3_family_vs_random.png`
- Figure 12: Week 10 feature-bucket transfer diagnostics from `output/week10/figures/fig7_feature_bucket_transfer.pdf`
- Figure 13: Week 10 few-shot calibration curves from `output/week10/figures/fig8_fewshot_curves.pdf`
- Figure 14: Week 10 deviation cases from `output/week10/figures/fig9_deviation_case.pdf`
- Figure 15: Week 10 hard-negative family boundary from `output/week10/figures/fig10_hard_negative_family.pdf`

## Week 7 Transfer Scope

Week 7 runs the full transfer matrix:

- T1_Nordic: Sweden + Spain + Italy -> Denmark
- T2_PH: European pooled -> Philippines
- T2_ON: European pooled -> Ontario
- T3_DiagnoseFrance: Spain + Italy -> France

Week 7 methods:

- `source_only`
- `dann`
- `strurw`
- `dann_strurw`

Week 7 result stance:

- T1_Nordic and T2_ON have one-class targets under the current confirmed-label policy, so ROC-AUC is undefined and must not be reported as a binary transfer score.
- T1_Nordic should use `illegal_recall_at_youden` as the main metric with threshold `0.55`; values around `0.30` are RQ2 evidence about differentiated transfer, not a generic all-method failure.
- T2_ON should use lower-is-better `mean_pred_illegal` with threshold `<0.30`; do not describe high one-class balanced accuracy as a StruRW victory.
- T2_PH remains a cross-continent stress test; after the final selection fix, DANN+StruRW reaches `0.7458` ROC-AUC but remains below `0.80`, so it fits the planned negative/limited-transfer interpretation.
- T3_DiagnoseFrance after the final selection fix: source-only `0.9861`, DANN `0.9748`, StruRW `0.9905`, and DANN+StruRW `0.9751`; StruRW is slightly above source-only, but not by the predeclared `+0.02` threshold.
- StruRW should be interpreted through the pseudo-label quality audit and the entropy top-30% selection rule.

## Week 8 Mechanism and Week 8.5 Correction Scope

Week 8 should be written as the prerun mechanism-evidence layer:

- E1 edge-channel ablation
- E2 LogME transferability prescreen
- E3 Denmark family LOFO
- E4 control-group reintroduction
- E5 structure-vs-lexical slicing

Week 8 headline stance:

- The formal Week 8 package completed `800` runs.
- The prerun acceptance table has `5` passes and `10` failures.
- E5 is the central negative/mechanistic finding: T3 France transfer is strongly mediated by Website lexical/ccTLD information, not graph-only structure.
- The original E3 LOFO pass must now be described as invalid for family-level generalization because its licensed-negative test design mostly repeated illegal-vs-licensed discrimination.

Week 8.5 is a separate correction layer, not a rewrite of Week 8:

- balanced licensed-negative E3 remains strong at family AUC `1.0000`
- hard illegal-negative family recognition fails with mean AUC `0.6219` and tsars AUC `0.3864`
- expanded one-class-aware E2 still does not rescue LogME, with Spearman rho `0.0733`
- P3/P4 are explicitly post-hoc interpretation revisions

## Phase 23 Integration Scope

Phase 23 should be used to support the final paper framing:

- Corrected E5 under `output/week8_patch/` supports a lexical/ccTLD boundary interpretation: lex-only beats graph-only on T3 by `0.2396`.
- Week 6 head ablation shows the original attention pooling head is a variance source; `fixed_mean` reaches mean ROC-AUC `0.9735` with std `0.0163`.
- Week 9 few-shot provides post-hoc, target-dependent RQ4 evidence only in the hard Philippines target: formal T2_PH requested 10-shot improves by `+0.3125`, including graph-only/no-website-lexical diagnostic conditions, while T3 shows no clear gain because source-only is already saturated.
- Week 9 hard-negative family audit keeps the family-discrimination claim bounded and negative.
- Week 10 final tables should be treated as the authoritative RQ rollup for current drafting.

## Method and Claim Boundaries

Keep these wording constraints fixed:

- Week 4 does not claim completion of the full `T1/T2/T3/T4` transfer matrix
- Week 4 does not claim few-shot France results
- Week 4 does not describe `Spain -> France` as the formal first transfer experiment
- Week 4 uses `Pooled excluding France -> France` as the first transfer setting because it matches the Graph v2 label availability
- Week 4 group-aware pooled metrics from `pooled_primary_summary.csv` are retrained group-aware model metrics
- The Philippines same-region HeteroConv acceptance threshold is `ROC-AUC >= 0.80`, not the older phase2 `0.88` threshold
- The `illegal_confirmed_official_cross_verified` primary-layer evaluation should be handled as a dedicated Denmark same-region experiment in a later section, because the pooled task mixes cross-verified and single-source illegal tiers
- Week 5 HeCo positives are metapath-derived and should not be described as label-derived hard positives
- Week 6 should be described as mixed: the evidence-layer RQ1 result passes, while the integrated HeCo finetune model still needs variance follow-up
- Week 7 should not force ROC-AUC on one-class transfer targets; use licensed/illegal consistency and mean predicted-illegal probability for those boundary cases
- Week 7 should not claim DANN or StruRW as generally superior; the completed matrix supports context-dependent and partly negative transfer findings
- Week 8 original E3 should not be presented as valid family-level generalization after the Week 8.5 defect audit
- Week 8.5 and Phase 23 post-hoc results must stay explicitly labeled as correction/diagnostic evidence
- RQ1 should be described as mixed: illegal-vs-licensed separability is strong, but illegal-family discrimination remains weak
- RQ3 should be described as mixed/negative-transfer-aware, not as a general transferability victory
