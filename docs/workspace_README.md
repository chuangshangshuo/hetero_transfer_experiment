# Hetero Transfer Experiment

This is the canonical workspace for the PA-free heterogeneous transfer experiments. The active graph is now Graph v2: `6` node types, `5` semantic relations, and `10` explicitly materialized PyG edge types with reverse edges.

## Active Schema

Semantic node types:

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

Core metapaths:

- `M1 = W-IP-W`
- `M2 = W-Cert-W`
- `M3 = W-NS-W`
- `M4 = W-ExtRef-W`

Extended metapaths:

- `M5 = W-Registrar-W`
- `M6 = W-IP-NS-IP-W`

`redirects_to` and `M7` are removed from the formal graph and preserved only through `data/audits/redirect_audit.csv`.

## Graph v2 Build Status

Actual node counts:

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

Forward raw edge counts before dedup:

- hosted_on: 1889
- uses_cert: 942
- uses_ns: 1690
- registered_via: 253
- referenced_by: 929

Key audits:

- Redirect audit: `166 -> 11 -> 7 -> 0`
- Isolated Website nodes: `38` (`Italy 26`, `France 12`)
- Phase1 `sample_tier` backfill: `licensed_baseline 85`, `control_legal_commercial 34`
- Website feature shape: `(623, 7)`

## Key Outputs

- `data/master_site_registry.csv`
- `data/duplicate_audit.csv`
- `data/node_feature_schema.json`
- `data/edge_schema.json`
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

Legacy `hetero_graph_v1*` outputs are preserved under `data/graphs/archived/` and `data/versions/archived/`.

## Commands

Run commands from the repository root unless a Windows local path is shown as an example. Full training requires the `hetero-transfer-v2` conda environment, PyTorch, PyG, the Graph v2 data files, and the relevant saved HeCo checkpoints under `output/week5/checkpoints/` or `output/week6/checkpoints/`.

Relative-path smoke and audit commands:

```powershell
python src\data\sanity_check.py
python analysis\final_artifact_audit.py
```

Relative-path Graph v2 rebuild commands:

```powershell
python src\data\build_hetero_graph.py
python src\data\convert_npz_to_pyg.py
powershell -ExecutionPolicy Bypass -File scripts\setup\run_week3_build_conda.ps1
```

Relative-path Week 4-7 rerun commands:

```powershell
python src\train\train_single.py
python src\train\train_transfer_dann.py
python src\train\train_single.py --tasks pooled_primary --smoke-test
python src\train\train_transfer_dann.py --smoke-test
python src\train\reaggregate_week4_group_split.py
python src\train\train_contrastive.py
python src\train\train_contrastive.py --smoke-test
python src\train\train_full.py
python src\train\train_full.py --optimizer-strategy uniform
python src\train\train_full.py --head-ablation
python analysis\rq1_cross_verified_eval.py
python analysis\ablation_week6.py
python src\train\train_transfer.py --methods source_only
python src\train\train_transfer.py --methods dann
python src\train\train_transfer.py --methods strurw dann_strurw
```

Relative-path Week 8-10 analysis commands:

```powershell
python analysis\week8_acceptance.py
python analysis\week85_patch.py
python analysis\week8_patch_corrected_e5.py
python analysis\week9_fewshot_calibration.py --rebuild-from-artifacts
python analysis\week9_hard_negative_family_audit.py
python analysis\week10_final_rq_tables.py
python analysis\feature_bucket_transfer.py
python analysis\week10_ablation_all.py
python analysis\week10_fewshot_final.py
python analysis\week10_hard_negative_final.py
python analysis\week10_deviation_case.py
python analysis\week10_acceptance_audit.py
python analysis\final_artifact_audit.py
```

Windows local conda examples:

```powershell
& "<user_home>\anaconda3\Scripts\conda.exe" run --no-capture-output -n hetero-transfer-v2 python <workspace_root>\src\data\sanity_check.py
& "<user_home>\anaconda3\Scripts\conda.exe" run --no-capture-output -n hetero-transfer-v2 python <workspace_root>\src\train\train_full.py
& "<user_home>\anaconda3\Scripts\conda.exe" run --no-capture-output -n hetero-transfer-v2 python <workspace_root>\src\train\train_transfer.py --methods source_only
& "<user_home>\anaconda3\Scripts\conda.exe" run --no-capture-output -n hetero-transfer-v2 python <workspace_root>\analysis\final_artifact_audit.py
```

Standalone placeholders are intentionally not rerun commands. `src\models\sehgnn.py`, `src\models\hinormer.py`, `src\transfer\irm_penalty.py`, `src\transfer\eerm_virtual_env.py`, and `src\train\train_transfer_strurw.py` are marked placeholders and do not support current reported results.

## Week 4 Outputs

Week 4 artifacts are written to `output/week4/`:

- `metrics/pooled_primary_summary.csv`
- `metrics/same_region_sensitivity_summary.csv`
- `metrics/transfer_france_summary.csv`
- `audits/pooled_primary_group_aware_split_audit.csv`
- `predictions/*.csv`
- `runs/*.json`
- `logs/*_history.csv`
- `plots/pooled_primary_roc_pr.png`
- `plots/transfer_france_source_only_vs_dann.png`

Week 4 scope is fixed as:

- pooled primary supervised task: `licensed_baseline` vs `illegal_confirmed_official_single + illegal_confirmed_official_cross_verified`
- same-region sensitivity only for `Belgium`, `France`, and `Philippines`
- first transfer experiment: `Pooled excluding France -> France`
- first transfer comparison: `source_only` vs `DANN`

Current Week 4 summary:

- group-aware retrained pooled primary mean ROC-AUC: `hetero_gnn = 0.9277`, `logistic_regression = 0.8938`, `mlp = 0.8876`
- same-region strongest logistic baseline after GroupKFold: `Belgium = 0.9649`, `France = 0.9822`, `Philippines = 0.9583`
- Philippines HeteroConv under GroupKFold is `ROC-AUC = 0.7889`; treat this as borderline/needs follow-up rather than a clean pass.
- France transfer mean ROC-AUC: `source_only = 0.9547`, `dann = 0.9337`

The pooled primary splits are now group-aware retraining splits using `operator_or_case -> brand -> root_domain` grouping and `balance_attempts = 20`. The pooled split audit records zero group overlap across train/val/test for all five seeds.

## Week 5 Outputs

Week 5 HeCo artifacts are written to `output/week5/`:

- `metrics/heco_temperature_grid_summary.csv`
- `metrics/heco_probe_raw_runs.csv`
- `metrics/heco_probe_summary.csv`
- `runs/week5_heco_acceptance.json`
- `audits/heco_positive_pair_audit__tau*.csv`
- `embeddings/heco_embeddings__tau0.7__seed42.csv`
- `embeddings/heco_tsne__tau0.7__seed42.csv`
- `plots/heco_tsne_sample_tier.png`
- `plots/heco_tsne_denmark_families.png`

Current Week 5 summary:

- best temperature by frozen linear validation ROC-AUC: `tau = 0.7`
- frozen encoder linear probe ROC-AUC: `0.9180`, passing the `0.90` manual threshold
- full fine-tune best ROC-AUC: `0.9897` at `tau = 0.3`
- metapath top-k positives: `2594` positive pairs, average `4.16` positive targets per Website
- Denmark family t-SNE audit includes `tsars = 9`, `icecasino = 6`, and `verdecasino = 4`

## Week 6 Outputs

Week 6 HeCo finetune and RQ1 artifacts are written to `output/week6/`, `figs/`, and `paper/draft/`:

- `metrics/pooled_primary_finetune_summary.csv`
- `metrics/pooled_primary_finetune_uniform_lr_summary.csv`
- `metrics/cross_verified_pooled_eval.csv`
- `metrics/ablation_table.csv`
- `metrics/week6_acceptance_audit.csv`
- `plots/heco_finetune_tsne_sample_tier.png`
- `figs/fig4_pooled_auc.pdf`
- `paper/draft/sec4_2_rq1.md`

Current Week 6 summary:

- Disc-LR HeCo finetune mean ROC-AUC: `0.9322`, std `0.0716`
- uniform-LR fallback mean ROC-AUC: `0.9337`, std `0.0740`
- HeteroGNN-only group-aware baseline mean ROC-AUC: `0.9277`
- cross-verified vs licensed pooled RQ1 AUC: `0.9798`
- acceptance audit: model-level Week 6 mean/std/delta targets are not met; cross-verified evidence-layer target is met

Week 6 should be reported as a mixed integration result. The RQ1 cross-verified evidence layer is strong, but the HeCo finetune model has seed variance driven by seed46 and should not be claimed as a locked improvement over HeteroGNN-only yet.

## Week 7 Outputs

Week 7 transfer artifacts are written to `output/week7/`:

- `metrics/transfer_summary.csv`
- `metrics/transfer_method_summary.csv`
- `metrics/week7_acceptance_audit.csv`
- `metrics/transfer_delta_vs_source_only.csv`
- `audits/pseudo_label_quality.csv`
- `audits/strurw_edge_weight_audit.csv`
- `plots/week7_transfer_boxplot_grid.png`
- `plots/week7_t3_dann_diagnostics.png`
- `plots/week7_delta_heatmap.png`
- `runs/*.json`
- `logs/*.csv`
- `paper/draft/sec4_3_week7_transfer.md`

Current Week 7 summary:

- full matrix completed: `4 transfer pairs x 4 methods x 5 seeds = 80` runs
- T1_Nordic and T2_ON targets are one-class under the current confirmed-label policy, so ROC-AUC is undefined for those targets
- T1_Nordic primary metric is `illegal_recall_at_youden`: `source_only = 0.2065`, `dann = 0.2710`, `strurw = 0.1419`, `dann_strurw = 0.2581`
- T2_ON primary metric is lower-is-better `mean_pred_illegal`: `source_only = 0.3028`, `dann = 0.3596`, `strurw = 0.0003`, `dann_strurw = 0.0243`
- T2_PH ROC-AUC: `source_only = 0.6417`, `dann = 0.5375`, `strurw = 0.6458`, `dann_strurw = 0.7458`
- T3_DiagnoseFrance ROC-AUC after StruRW/DANN selection fix: `source_only = 0.9861`, `dann = 0.9748`, `strurw = 0.9905`, `dann_strurw = 0.9751`

Week 7 should be reported as mixed and boundary-aware: T1 is a differentiated illegal-recall evidence setting that remains below the `0.55` target, T2_ON is an anomaly-style licensed-consistency counterpoint to T2_PH, and T3 no longer has the StruRW/DANN+StruRW final-stage selection artifact but still does not meet the `source_only + 0.02` transfer-improvement threshold.

## Week 8 Outputs

Week 8 mechanism-evidence artifacts are written to `output/week8/`:

- `metrics/week8_acceptance_audit.csv`
- `metrics/week8_completion_audit.csv`
- `metrics/E1_edge_ablation_summary.csv`
- `metrics/E2_transferability_metrics.csv`
- `metrics/E3_lofo_family_sensitivity.csv`
- `metrics/E4_three_scenario_summary.csv`
- `metrics/E5_structure_vs_lexical_summary.csv`
- `audits/predeclared_hypotheses.md`
- `audits/predeclared_freeze_manifest.json`
- `paper/draft/sec4_4_week8_mechanism.md`

Current Week 8 summary:

- formal completion is `800` run manifests and `800` logs
- prerun hypothesis status is `5` pass and `10` fail
- E1 H_E1a passes for T3 edge sensitivity, but H_E1b/H_E1c fail
- E2 LogME fails as a transferability prescreen: Spearman rho `0.2754`
- original E3 LOFO passed under the prerun licensed-negative design, but this was later downgraded by Week 8.5 because the test task mostly repeated illegal-vs-licensed discrimination
- E4 H_E4a passes, while embedding-distance geometry H_E4b/H_E4c fails
- E5 shows T3 is not graph-only: no-ccTLD drop is `0.0815`, graph-only AUC is `0.7134`, and lex-only AUC is `0.9530`

## Week 8.5 Patch Outputs

Week 8.5 correction artifacts are written to `output/week85/` and leave `output/week8/` unchanged:

- `metrics/E3_w85_lofo_raw.csv`
- `metrics/E3_w85_lofo_summary.csv`
- `metrics/E2_w85_expanded_pair_scores.csv`
- `metrics/E2_w85_transferability_metrics.csv`
- `metrics/week85_acceptance_audit.csv`
- `paper/draft/sec4_4_week85_patch.md`

Current Week 8.5 summary:

- balanced licensed-negative E3 passes with family mean AUC `1.0000`
- hard illegal-negative family recognition fails: mean AUC `0.6219`, tsars AUC `0.3864`
- expanded one-class-aware E2 still fails: LogME Spearman rho `0.0733`
- P3/P4 are post-hoc interpretation revisions and must not be merged into the prerun Week 8 acceptance table

## Phase 23 Patch Outputs

Phase 23 adds security/config cleanup, corrected E5 diagnostics, head ablation, few-shot RQ4 probes, formal hard-negative family auditing, and final RQ tables:

- `output/week8_patch/metrics/E5_corrected_acceptance_audit.csv`
- `output/week6_head_ablation/metrics/head_ablation_summary.csv`
- `output/week9/metrics/W9_E1_fewshot_curve.csv`
- `output/week9/metrics/week9_fewshot_rollup.csv`
- `output/week9/metrics/fewshot_acceptance_audit_v2.csv`
- `output/week9/metrics/hard_negative_family_acceptance.csv`
- `output/week10/metrics/final_rq_evidence_table.csv`
- `output/week10/metrics/final_hypothesis_rollup.csv`
- `output/week10/metrics/week10_acceptance_audit.csv`
- `paper/draft/sec5_final_rq_tables.md`

Current Phase 23 summary:

- corrected E5 supports a lexical/ccTLD boundary interpretation for T3: lex-only minus graph-only is `0.2396`
- Week 6 head ablation shows `fixed_mean` is the strongest/stablest head at mean ROC-AUC `0.9735`, std `0.0163`; the original attention head is `0.9322`, std `0.0716`
- Week 9 few-shot formal rollup is target-dependent: T2_PH improves from `0.6417` to `0.9542` at requested 10-shot calibration, while T3 is already saturated and shows no clear gain
- T2_PH few-shot recovery also remains strong under no-ccTLD, no-website-lexical, and graph-only conditions, but this is still target calibration evidence rather than broad pure structural generalization
- Week 9 formal hard-negative audit repeats the family-recognition boundary: mean hard-negative AUC `0.6218`, tsars `0.3864`
- Week 10 final tables freeze the paper-safe frame: structural commonality, regional bias, evidence hierarchy, and few-shot local calibration jointly define the final interpretation

## Interpretation Boundary Revisions

The current paper interpretation should use these boundary documents:

Final paper-safe Week 7/8 interpretation should follow `docs/evidence_boundary_table.md` and the new `sec_week7/8` draft files.

- `docs/week7_transfer_interpretation.md`: Week 7 is transfer performance analysis, not direct structural-generalization proof.
- `docs/week8_diagnostic_revision.md`: Week 8 is diagnostic analysis of transfer boundaries, not clean hypothesis validation.
- `docs/evidence_boundary_table.md`: paper-safe claim wording and overclaims to avoid.

Derived interpretation tables are saved without overwriting original experiment outputs:

- `output/week7/metrics/week7_transfer_summary.csv`
- `output/week7/metrics/week7_transfer_taxonomy.csv`
- `output/week7/metrics/week7_to_week8_bridge.csv`
- `output/week8/metrics/E3_lofo_setting_comparison.csv`
- `output/week8/metrics/E3_hard_negative_family_summary.csv`
- `output/week8/metrics/E5_structure_vs_lexical_corrected_summary.csv`
- `output/week8/metrics/week8_diagnostic_rollup.csv`

## Environment

Recommended local environment:

- conda env: `hetero-transfer-v2`
- Python: `3.10`
- PyTorch: `2.5.1`
- PyG: `2.6.1`
- MaxMind GeoLite2-ASN: local-only dependency, not redistributed in git

Environment records:

- `configs/environment_pyg_cpu.yml`
- `configs/environment_pyg_cpu_export.yml`
- `configs/requirements-week3.lock.txt`
- `data/versions/pyg_environment_verification.json`

## Workspace Policy

All subsequent experiment implementation and generated outputs should be placed under:

- `<workspace_root>`

The git handoff mirror remains:

- `<release_root>\project\hetero_transfer_experiment`
- `<release_root>\data\hetero_transfer_experiment`

## Compliance Boundary

- No illegal landing pages are visited by the graph builder.
- No login, registration, recharge, CAPTCHA bypass, or geoblock bypass is performed.
- PromotionAccount remains archived and excluded from the active graph.
- ExternalReference uses only public search-discovery and archive-history evidence.
