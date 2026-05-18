# Task Plan

## Goal
Implement the "第二阶段 v4 收紧版" on top of the existing phase2 evidence library and anomaly outputs, without changing phase1 matched-main, page-error isolation, strong-edge auditing, or the phase1 main/appendix boundaries.

## Constraints
- Do not change phase1 modeling framework or rerun phase1 as the main output.
- Do not keep using "illegal is denser" as the default hypothesis.
- Keep `gray_candidate` candidate-only; never auto-upgrade from similarity or anomaly score alone.
- `illegal_confirmed` must keep official regulatory basis.
- No login, registration, recharge, or bypassing of geoblock/captcha/anti-bot.
- Keep entity granularity boundaries: `domain / service_name / operator / brand` must not be merged into one supervised label space.

## Phases
- [x] Phase 1: Inspect the current phase2 upgrade pipeline and identify where to tighten cross-source strong, gray role, discovery role, and anomaly-first outputs.
- [x] Phase 2: Tighten sample-tier logic into baseline / gray minimum / gray strong / illegal single / illegal cross-verified with explicit analysis roles and strong subtypes.
- [x] Phase 3: Upgrade sample-quality outputs with OSINT strength profiles, discovery diversity summaries, and gray-to-confirmed relation support summaries.
- [x] Phase 4: Promote anomaly-over-density outputs and baseline deviation decomposition to the v4 main result layer.
- [x] Phase 5: Re-run the pipeline, validate CSV outputs, generate the v4 report docx, and render-check it with soffice.

## Acceptance Focus
- Second stage no longer defaults to the "illegal is denser" hypothesis.
- `gray_candidate_strong` no longer carries the main structural conclusion role; `illegal_confirmed_official_cross_verified` becomes the primary structural analysis layer.
- All samples expose `source_clusters`, `source_cluster_count`, `independent_source_cluster_count`, `cross_source_minimum_passed`, `cross_source_strong_passed`, `cross_source_strong_subtype`, and `strong_pass_confidence`.
- `cross_source_strong` is tightened so that weak discovery-led stacking is downgraded to `weak_strong_borderline`.
- Discovery is fixed as a candidate-discovery / diversity layer, not a main fused graph layer.
- Main second-stage outputs prioritize anomaly/deviation summaries over density/modularity.

## Phase 7: Experimental Report PPT
- [x] Traverse project outputs, reports, CSV tables, and existing figures for phase1 / phase2 experimental evidence.
- [x] Extract key experiment objectives, methods, sample scale, validation logic, metrics, evidence tiers, anomaly results, and limitations.
- [x] Build a detailed, image-rich editable PowerPoint report with native charts, tables, diagrams, and project figures.
- [x] Render/inspect slide previews and export the final PPTX.

## Phase 6: RLS / Column-Hiding Deadlock Analysis
- [x] Inspect `死锁问题扩展.docx`, explain why "方案二" still fails, and redesign a safe access pattern where ordinary users cannot see `username` but can still see only their own row.
 
## Phase 8: Six-Region Neutral Stability Expansion
- [x] Add an isolated six-region expansion pipeline without overwriting phase1 matched-main or phase2 v4 outputs.
- [x] Freeze official-source and control-seed sampling into region expansion CSVs for France, Germany, Italy, Belgium, the United Kingdom, and Ontario.
- [x] Generate neutral experiment 1 / experiment 2 direction audits, leave-one-region-out checks, bootstrap support summaries, collection-failure audit, and conclusion-fit audit.
- [x] Generate and render-check `第一阶段扩大稳健性实验报告_六地扩展.docx`.
- [x] Sync the new script, data outputs, and report into the Git-style handoff folder.

## Phase 9: Six-Region Infrastructure Enhancement
- [x] Add DNS/TLS/RDAP-only enhancement script for the frozen six-region expansion sample.
- [x] Generate infrastructure relation, TLS detail, domain-audit, feature-matrix, coverage, experiment 1 infra, France TLD-bias, four-dimensional deviation, pooled LVI, and Italy ADM-source audit CSVs.
- [x] Keep Italy licensed candidates as sensitivity-only because an official machine-readable ADM authorized-domain source was not verified.
- [x] Generate standalone infrastructure-enhancement report and integrate section 8.7 into the comprehensive phase1 report.
- [x] Render DOCX reports to PDF and verify no `???` replacement artifacts in extracted PDF text.
- [x] Sync enhanced outputs and README into `<workspace_root>\git`.

## Phase 10: PA Removal and v2 Hetero Transfer Workspace
- [x] Archive PA collection script and outputs to `<workspace_root>\phase2\data\promotion_collection_archived`.
- [x] Write `README_archived.md` with collection date, 0.83% initial hit rate, 0 high-confidence hits, removal decision, and ExtRef migration rationale.
- [x] Create `<workspace_root>\hetero_transfer_experiment_v2` as a clean PA-free workspace.
- [x] Configure v2 schema as 6 nodes / 5 semantic relations / 10 materialized PyG edge types.
- [x] Initialize Week 3-10 script tree and paper/output directories.
- [x] Add MaxMind GeoLite2-ASN setup helper and PyG CUDA 11.8 environment files.
- [x] Add paper outline updates for §3.1, §3.3, and §5.5.
- [x] Run sanity check confirming no active PA collector/config remains in v2.
 
## Phase 11: Week 3 Heterogeneous Graph Build
- [x] Download MaxMind GeoLite2-ASN locally without writing the license key into tracked files.
- [x] Implement `src/data/build_hetero_graph.py` as the Graph v2 orchestrator with node/edge builders.
- [x] Build `master_site_registry.csv` with root-domain deduplication and duplicate audit traceability.
- [x] Build Website, IP, Certificate, NameServer, Registrar, and ExternalReference node tables.
- [x] Build deduplicated `hosted_on`, `uses_cert`, `uses_ns`, `registered_via`, and `referenced_by` edge tables with `edge_count_raw` and `edge_weight`.
- [x] Move `redirects_to` out of the active graph and preserve it via `redirect_audit.csv` only.
- [x] Generate Graph v2 artifacts: `hetero_graph_v2.npz`, `hetero_graph_v2.pt`, `hetero_graph_v2_metadata.json`, and `hetero_graph_v2.sha256`.
- [x] Verify PA remains archived/excluded and sanity check passes.
- [x] Sync Week 3 project code and graph outputs into `<workspace_root>\git`.
- [x] Install or activate a PyG runtime and convert fallback NPZ to native `HeteroData` `.pt`.
- [x] Freeze Week 3 final acceptance baselines: Website.x `(623, 7)`, forward unique edge counts `1516 / 519 / 1662 / 252 / 57`, redirect audit `166 -> 11 -> 7 -> 0`, isolated Website nodes `38`.
- [x] Archive legacy Graph v1 outputs under `data/graphs/archived/` and `data/versions/archived/`.
- [ ] Start Week 4 baseline model scripts using the generated graph package.

### Week 3 Final Outputs
- `data/graphs/hetero_graph_v2.npz`
- `data/graphs/hetero_graph_v2.pt`
- `data/graphs/hetero_graph_v2_metadata.json`
- `data/versions/feature_builder_state.json`
- `data/versions/hetero_graph_v2.sha256`
- `data/node_feature_schema.json`
- `data/edge_schema.json`
- `data/audits/redirect_audit.csv`
- `data/audits/edge_dedup_summary.csv`
- `data/audits/isolated_website_nodes.csv`
- `data/audits/extref_hub_audit.csv`
- `data/audits/website_tier_backfill_audit.csv`

## Phase 12: Canonical Workspace Sync
- [x] Promote `<workspace_root>\hetero_transfer_experiment` as the canonical workspace for subsequent experiments.
- [x] Sync PA-free v2 code, configs, docs, data outputs, source materials, and local lightweight dependencies into the canonical workspace.
- [x] Archive stale PA active files from the old workspace under `source_materials/archive/legacy_pa_active_files_20260423`.
- [x] Update README and Week 3 summary paths to use `<workspace_root>\hetero_transfer_experiment`.
- [x] Re-run sanity check and graph build from the canonical workspace.
- [x] Prepare git handoff mirror from the canonical workspace.

## Phase 13: Conda PyTorch/PyG Environment
- [x] Locate Anaconda and inspect existing environments.
- [x] Detect local hardware/CUDA status and choose CPU PyTorch/PyG for this machine.
- [x] Create conda env `hetero-transfer-v2` with Python 3.10.
- [x] Install PyTorch 2.5.1 CPU, PyG 2.6.1, and PyG CPU extension wheels.
- [x] Install supporting scientific/data dependencies.
- [x] Generate native PyG `data/graphs/hetero_graph_v2.pt`.
- [x] Add environment export, pip lock, verification JSON, and one-click conda build script.
- [x] Verify `.pt` load, graph dimensions, environment imports, graph build, and sanity check.

## Phase 14: Week 4 Baseline Training Launch
- [x] Freeze Graph v2 as the canonical Week 4 input package for all baseline work.
- [x] Freeze Week 4 task definitions and split protocol around the current Graph v2 registry and label availability.
- [x] Build a reusable Graph v2 training interface that loads `.pt` plus `master_site_registry.csv`, excludes default isolated websites, and fail-fast checks all-zero feature columns before training.
- [x] Implement Baseline 0 as a non-graph sanity check: Website-only logistic regression / MLP using the 7 structural Website features.
- [x] Implement Baseline 1 as the first graph baseline: edge-weight-aware heterogeneous message passing over Graph v2 with explicit reverse edges.
- [x] Materialize reproducible split artifacts and run manifests for the pooled primary task plus same-region sensitivity tasks.
- [x] Run one-seed smoke tests and then 5-seed formal baseline runs for pooled licensed-vs-illegal classification.
- [x] Emit Week 4 result artifacts, including metrics tables, seed manifests, prediction dumps, and training logs, under a dedicated Week 4 output subtree.
- [x] Update README and `paper/outline_v2.md` so the written method description matches Graph v2 and the Week 4 baseline scope.
- [x] Run the first transfer experiment as `Pooled excluding France -> France`, with matched `source_only` and `DANN` runs on the same target split family.
- [x] Correct pooled primary split generation to group-aware `GroupShuffleSplit` using `operator_or_case -> brand -> root_domain` grouping.
- [x] Set `balance_attempts = 20` and regenerate the five pooled primary split artifacts.
- [x] Retrain the 5 seed x 3 model pooled primary runs on the new group-aware splits.
- [x] Correct same-region sensitivity to group-aware `GroupKFold` with group-aware validation split.
- [x] Lower the Week 4 Philippines HeteroConv acceptance threshold to `ROC-AUC >= 0.80` for the Graph v2 / v4-tier sample system.
- [x] Clarify that Week 4's graph result is the `HeteroConv baseline`, not SeHGNN or HINormer.

### Week 4 Scope Notes
- Week 4 is the baseline-launch week, not the full transfer-learning week.
- The primary supervised task should be pooled `licensed_baseline` versus `illegal_confirmed_official_single + illegal_confirmed_official_cross_verified`.
- `gray_candidate_*` and `control_legal_commercial` should stay out of the primary training label space and be reserved for auxiliary audits or later sensitivity checks.
- Same-region binary sensitivity baselines are currently viable for Belgium, France, and the Philippines.
- Denmark, Italy, Ontario, the United Kingdom, and Germany are better treated as transfer or one-sided evaluation settings in later weeks because their current label composition is not balanced for the primary same-region binary task.

### Week 4 Acceptance Focus
- A first runnable, reproducible Graph v2 baseline must exist end-to-end in the canonical workspace.
- Every training run must be traceable to saved split files, seed values, graph version `v2`, and the default isolated-node policy.
- The first formal metric package should prioritize pooled licensed-vs-illegal classification before any transfer or few-shot claims.
- Philippines same-region HeteroConv is borderline after group-aware GroupKFold at `0.7889`, so it should be reported as needing follow-up rather than a clean pass.
- `illegal_confirmed_official_cross_verified` should get a dedicated Denmark same-region evaluation later instead of being over-claimed from pooled splits alone.

### Week 4 Final Outputs
- `configs/week4_experiments.yaml`
- `src/train/train_single.py`
- `src/train/train_transfer_dann.py`
- `src/train/utils.py`
- `src/models/hetero_full.py`
- `src/transfer/dann_head.py`
- `output/week4/metrics/pooled_primary_summary.csv`
- `output/week4/audits/pooled_primary_group_aware_split_audit.csv`
- `output/week4/audits/same_region_group_aware_split_audit.csv`
- `output/week4/metrics/same_region_sensitivity_summary.csv`
- `output/week4/metrics/transfer_france_summary.csv`
- `output/week4/plots/pooled_primary_roc_pr.png`
- `output/week4/plots/transfer_france_source_only_vs_dann.png`

## Phase 15: Week 5 HeCo Contrastive Pretraining
- [x] Add `configs/week5_heco.yaml` as the single source of truth for HeCo temperature grid, positives, hard negatives, pretraining epochs, probe settings, and output paths.
- [x] Implement `src/models/projection_heads.py` with a 2-layer MLP projection head.
- [x] Implement `src/models/view_generators.py` with the schema-view encoder and metapath-view encoder.
- [x] Implement `src/models/heco_head.py` with the HeCo model wrapper and co-contrastive loss.
- [x] Implement `src/train/train_contrastive.py` for pretraining, frozen linear probing, full fine-tuning, t-SNE plots, and acceptance output.
- [x] Use top-k metapath co-occurrence positives rather than label-derived positives.
- [x] Run a smoke test and the full temperature grid `{0.1, 0.3, 0.5, 0.7}`.
- [x] Verify the frozen encoder linear probe passes the manual `ROC-AUC >= 0.90` threshold.
- [x] Generate t-SNE outputs for sample-tier separation and Denmark family clustering.
- [x] Sync Week 5 code, docs, and outputs into `<workspace_root>\git`.

### Week 5 Final Outputs
- `configs/week5_heco.yaml`
- `src/models/heco_head.py`
- `src/models/view_generators.py`
- `src/models/projection_heads.py`
- `src/train/train_contrastive.py`
- `output/week5/metrics/heco_temperature_grid_summary.csv`
- `output/week5/metrics/heco_probe_raw_runs.csv`
- `output/week5/metrics/heco_probe_summary.csv`
- `output/week5/runs/week5_heco_acceptance.json`
- `output/week5/plots/heco_tsne_sample_tier.png`
- `output/week5/plots/heco_tsne_denmark_families.png`

## Phase 16: Week 6 HeCo Finetune and RQ1 Freeze
- [x] Add `configs/week6_heco_finetune.yaml` for selected temperature, per-seed checkpoints, fine-tune settings, acceptance thresholds, and output paths.
- [x] Implement `src/models/hetero_full_with_heco.py` with schema/metapath attention pooling and discriminative-LR parameter groups.
- [x] Implement `src/train/train_full.py` for per-seed HeCo checkpoint loading/pretraining, Disc-LR fine-tuning, uniform-LR fallback, prediction dumps, histories, checkpoints, and t-SNE export.
- [x] Reuse the Week 5 seed42 tau0.7 encoder and generate tau0.7 encoders for seeds 43-46.
- [x] Run five-seed Disc-LR fine-tuning and uniform-LR fallback.
- [x] Implement and run `analysis/rq1_cross_verified_eval.py`.
- [x] Implement and run `analysis/ablation_week6.py`.
- [x] Generate Table 2 source data, Figure 4, RQ1 paper draft, and Week 6 acceptance audit.
- [x] Record that HeCo fine-tune model-level acceptance targets are not met under the current group-aware split family.
- [x] Record that the cross-verified evidence-layer RQ1 target is met.

### Week 6 Final Outputs
- `configs/week6_heco_finetune.yaml`
- `src/models/hetero_full_with_heco.py`
- `src/train/train_full.py`
- `analysis/ablation_week6.py`
- `analysis/rq1_cross_verified_eval.py`
- `output/week6/metrics/pooled_primary_finetune_summary.csv`
- `output/week6/metrics/pooled_primary_finetune_uniform_lr_summary.csv`
- `output/week6/metrics/cross_verified_pooled_eval.csv`
- `output/week6/metrics/ablation_table.csv`
- `output/week6/metrics/week6_acceptance_audit.csv`
- `figs/fig4_pooled_auc.pdf`
- `paper/draft/sec4_2_rq1.md`

### Week 6 Acceptance Notes
- Disc-LR HeCo finetune mean ROC-AUC is `0.9322` with std `0.0716`, so it misses the predeclared `>=0.95` mean and `<=0.04` std targets.
- Uniform-LR fallback mean ROC-AUC is `0.9337` with std `0.0740`, so it does not solve the variance issue.
- Cross-verified vs licensed pooled RQ1 AUC is `0.9798`, so the evidence-layer RQ1 target passes.
- Week 6 should be written as a mixed integration result rather than a clean final model victory.

## Phase 17: Week 7 Transfer Experiment Matrix
- [x] Restore Week 3-6 context and read `<user_home>\Desktop\d.txt`.
- [x] Check for existing Week 7 files or unfinished outputs.
- [x] Check Week 6 / Week 5 HeCo checkpoint availability and conda PyTorch/PyG imports.
- [x] Add `configs/week7_transfer.yaml` for the 4 transfer pairs, 4 methods, 5 seeds, output paths, and target-label boundary policy.
- [x] Implement StruRW CSBM-style source edge reweighting with pseudo-label confidence threshold `0.7`, weight clip `[0.1, 10.0]`, and 2 iterations.
- [x] Implement a Week 7 DANN path by reusing the HeCo encoder/fine-tune stack and the existing gradient reversal domain discriminator.
- [x] Add unified `src/train/train_transfer.py` for `source_only`, `dann`, `strurw`, and `dann_strurw`.
- [x] Run static checks and Week 7 smoke tests before formal runs.
- [x] Run staged Week 7 experiments in order: source_only, DANN, then StruRW / DANN+StruRW.
- [x] Emit Week 7 metrics, audits, plots, per-run manifests, and training logs under `output/week7/`.
- [x] Record any incomplete-target-label cases as method boundaries or negative findings rather than forcing invalid ROC-AUC.

### Week 7 Acceptance Focus
- Preserve Graph v2: no label-derived Website features, no active `redirects_to`, and PA remains archived/excluded.
- Keep `gray_candidate_*` out of confirmed-illegal labels and out of primary supervised training.
- Treat target domains with one observed class honestly: ROC-AUC is undefined for Ontario and any other one-class target; report balanced accuracy / licensed or illegal consistency / anomaly-style scores instead.
- Run and interpret transfer methods against the source-only baseline rather than assuming DANN or StruRW should always improve.

### Week 7 Final Outputs
- `configs/week7_transfer.yaml`
- `src/transfer/dann.py`
- `src/transfer/strurw_reweight.py`
- `src/train/train_transfer.py`
- `output/week7/metrics/transfer_summary.csv`
- `output/week7/metrics/transfer_method_summary.csv`
- `output/week7/audits/pseudo_label_quality.csv`
- `output/week7/audits/strurw_edge_weight_audit.csv`
- `output/week7/plots/week7_transfer_boxplot_grid.png`
- `output/week7/plots/week7_t3_dann_diagnostics.png`
- `output/week7/runs/*.json`
- `output/week7/logs/*.csv`

### Week 7 Result Notes
- All `80` formal runs completed.
- T1_Nordic and T2_ON target sets are one-class under the current confirmed-label policy, so their ROC-AUC is intentionally undefined.
- T2_PH remains a negative transfer setting: source-only ROC-AUC `0.6542`, DANN `0.5167`, StruRW `0.6292`, DANN+StruRW `0.5917`.
- T3_DiagnoseFrance is mixed: source-only ROC-AUC `0.8510`, DANN `0.8642`, StruRW `0.8702`, DANN+StruRW `0.7686`; DANN and StruRW improve slightly but do not clearly pass the predeclared `+0.02` delta threshold.
- StruRW pseudo-label quality is uneven, especially T1_Nordic `~0.082`, T2_PH `~0.626`, and T3 `~0.544`, so Week 7 should not be written as a clean StruRW win.

## Phase 18: Week 7 Remediation Pass
- [x] Fix transfer early stopping so equal validation scores refresh the best epoch and no run can stop before `min_epochs_before_early_stop`.
- [x] Reframe T1_Nordic primary metric as `illegal_recall_at_youden` with threshold `0.55`, treating low recall as RQ2 evidence rather than a generic transfer failure.
- [x] Reframe T2_ON primary metric as `mean_pred_illegal` with lower-is-better threshold `<0.30`, avoiding balanced-accuracy victory language for one-class Ontario.
- [x] Tune DANN schedule to warmup `20`, max epochs `200`, and min epochs `80`.
- [x] Replace StruRW fixed confidence threshold selection with entropy-based top-30% pseudo-label selection.
- [x] Run smoke tests and rerun the affected Week 7 matrix.
- [x] Refresh metrics, plots, docs, findings, and progress notes.

### Week 7 Remediation Result Notes
- Formal matrix remains complete at `80` runs.
- T1_Nordic primary metric is `illegal_recall_at_youden`: source_only `0.2065`, DANN `0.2387`, StruRW `0.1032`, DANN+StruRW `0.3484`.
- T2_ON primary metric is lower-is-better `mean_pred_illegal`: source_only `0.3028`, DANN `0.3596`, StruRW `0.0004`, DANN+StruRW `0.0258`.
- T2_PH ROC-AUC after remediation: source_only `0.6417`, DANN `0.5375`, StruRW `0.6750`, DANN+StruRW `0.7333`.
- T3_DiagnoseFrance ROC-AUC after remediation: source_only `0.9861`, DANN `0.9748`, StruRW `0.9404`, DANN+StruRW `0.8851`; early stopping was material, but adaptation methods should not be claimed as beating the remediated source-only baseline.

## Phase 19: Week 7 StruRW/DANN Selection Fix
- [x] Split StruRW prefit stages into pseudo-labeler-only runs and make the final stage the only formal classifier training stage.
- [x] Force StruRW final classifier training to run the full configured `200` epochs.
- [x] Prevent DANN best-state selection from choosing warmup epochs before adversarial training starts.
- [x] Run targeted T3 seed42 validation for DANN, StruRW, and DANN+StruRW.
- [x] Rerun affected Week 7 methods if the targeted validation passes.
- [x] Refresh findings/progress and final summaries.

### Week 7 Selection-Fix Result Notes
- T3 seed42 targeted validation confirmed the intended behavior:
  - StruRW prefit stages are `pseudo_labeler` runs of `80` epochs each.
  - StruRW final stage is a `final_classifier` run of `200` epochs with `selection_policy = last_epoch`.
  - DANN and DANN+StruRW best-state selection starts at epoch `20`, after warmup begins.
- Affected methods were rerun across the full Week 7 matrix: DANN, StruRW, and DANN+StruRW.
- Final T1_Nordic `illegal_recall_at_youden`: source_only `0.2065`, DANN `0.2710`, StruRW `0.1419`, DANN+StruRW `0.2581`; still below the `0.55` target but useful as RQ2 differentiated-transfer evidence.
- Final T2_PH ROC-AUC: source_only `0.6417`, DANN `0.5375`, StruRW `0.6458`, DANN+StruRW `0.7458`; all remain below `0.80`.
- Final T2_ON lower-is-better `mean_pred_illegal`: source_only `0.3028`, DANN `0.3596`, StruRW `0.0003`, DANN+StruRW `0.0243`.
- Final T3 ROC-AUC: source_only `0.9861`, DANN `0.9748`, StruRW `0.9905`, DANN+StruRW `0.9751`; StruRW does not clear the predeclared `source_only + 0.02` threshold.

## Phase 20: Week 7 Acceptance and Paper Draft Artifacts
- [x] Add a reproducible Week 7 acceptance/delta analysis script.
- [x] Generate `week7_acceptance_audit.csv`.
- [x] Generate `transfer_delta_vs_source_only.csv`.
- [x] Generate `week7_delta_heatmap.png`.
- [x] Generate paper draft section `paper/draft/sec4_3_week7_transfer.md`.
- [x] Update README, paper outline, findings, and progress with the final Week 7 acceptance state.

### Week 7 Analysis-Layer Outputs
- `analysis/week7_transfer_acceptance.py`
- `output/week7/metrics/week7_acceptance_audit.csv`
- `output/week7/metrics/transfer_delta_vs_source_only.csv`
- `output/week7/plots/week7_delta_heatmap.png`
- `paper/draft/sec4_3_week7_transfer.md`

## Phase 21: Week 8 Mechanism Evidence Layer
- [x] Restore Week 8 source-of-truth documents from the Desktop: design document, acceptance summary, and predeclared hypotheses.
- [x] Re-check the experiment environment, Week 7 artifacts, Week 6/5 HeCo checkpoint availability, and existing Week 8 placeholders before edits.
- [x] Freeze `configs/week8.yaml` and `output/week8/audits/predeclared_hypotheses.md` before formal Week 8 runs.
- [x] Implement shared Week 8 utilities for paired bootstrap, graph/feature ablations, family extraction, transferability scores, and control-group evaluation.
- [x] Run Week 8 smoke tests, prioritizing E3 family LOFO per the design document's Day 1-2 risk order.
- [x] Run staged Week 8 formal experiments: E3, E1, E5, E4, then E2.
  - [x] E3 formal core: 7 eligible Denmark families x 5 seeds = 35 LOFO runs.
  - [x] E3 permutation baseline: 100 random held-out controls x 5 seeds = 500 runs.
  - [x] E1 edge-channel ablation formal matrix: 120 runs.
  - [x] E5 structure-vs-lexical formal matrix: 100 runs.
  - [x] E4 control-group formal scenarios: 15 runs.
  - [x] E2 LogME transferability scoring and actual complete-label transfer alignment: 30 realized transfer runs over 6 valid ordered region pairs.
- [x] Generate Week 8 metrics, audits, plots, run manifests, and paper-ready summaries under `output/week8/`.
- [x] Record pass/fail outcomes against the prerun-frozen hypotheses without post-hoc threshold changes.

### Week 8 Acceptance Focus
- Freeze Graph v2 and W6 HeCo encoder checkpoints; do not change schema, Website feature definitions, active edge set, or pretraining.
- Use W7 v2 transfer splits/metrics for E1 and E5; T1 and T2_ON remain one-class target boundary cases.
- Keep `gray_candidate_*` out of confirmed-illegal labels; only E4 reintroduces `control_legal_commercial` as an explicit control group.
- Use paired bootstrap for seed-paired comparisons and report failed hypotheses as findings rather than tuning thresholds after the fact.

### Week 8 E3 Core Result Notes
- Denmark family extraction produced 7 LOFO-eligible families with size >= 2: `ggbet`, `tsars`, `icecasino`, `verdecasino`, `bcdot`, `bcgame`, and `freshbet`.
- E3 formal core completed `35` runs with matching logs, splits, run manifests, and predictions.
- Five-seed LOFO AUC means are all high: minimum `0.9928` for `bcdot`, `tsars = 0.9989`, and `ggbet/verdecasino = 1.0000`.
- H_E3a and H_E3b are provisionally supported by the core LOFO runs; H_E3c is also provisionally supported because family-level five-seed AUC std is below `0.10` for all eligible families.
- The permutation baseline is complete; it does not show real reconstructed families are significantly harder than random held-out groups (`real_minus_random = -0.0005`, `p_real_lower = 0.160`), so E3 should be framed as a strong robustness partition with a negative supplemental family-label-difficulty audit.

### Week 8 Final Acceptance Notes
- Week 8 produced `800` run manifests and `800` training/history logs under `output/week8/`.
- The completion audit passes for all actually valid matrices:
  - E1: `120/120`
  - E2 pair scores: `30/30`
  - E2 realized transfer alignment: `30/30`
  - E3 core: `35/35`
  - E3 permutation: `500/500`
  - E4: `15/15`
  - E5: `100/100`
- The design document's `105` E2 extra-run estimate was not forced because the current Graph v2 primary label space has only `6` ordered single-source region pairs with complete binary labels; this is recorded as a label-availability boundary rather than filled with invalid AUC.
- Final hypothesis status from `output/week8/metrics/week8_acceptance_audit.csv`: `5` pass and `10` fail.
- Passing hypotheses:
  - E1 H_E1a
  - E3 H_E3a/H_E3b/H_E3c
  - E4 H_E4a
- Failing hypotheses:
  - E1 H_E1b/H_E1c
  - E2 H_E2a/H_E2b/H_E2c
  - E4 H_E4b/H_E4c
  - E5 H_E5a/H_E5b/H_E5c
- The major interpretive finding is E5: T3 France transfer is not graph-only. Removing ccTLD/local-TLD information drops T3 AUC by `0.0815`, graph-only AUC is `0.7134`, and lex-only AUC is `0.9530`.

## Phase 22: Week 8.5 Patch Audit
- [x] Record the methodological defect in the original E3 LOFO: test negatives were licensed controls, so the task mostly repeated illegal-vs-licensed discrimination rather than family-level generalization.
- [x] Implement revised E3 LOFO with balanced licensed negatives: test positives are held-out family illegal sites, test negatives are equal-count held-out non-Denmark licensed sites, and those licensed negatives are removed from training.
- [x] Implement hard-negative E3 LOFO: evaluate held-out family illegal sites against other Denmark illegal-family sites, using the trained illegal-vs-licensed model scores as a family-discrimination audit.
- [x] Run revised E3 across 7 families x 5 seeds x 2 test forms = 70 formal runs/evaluations under `output/week85/`.
- [x] Extend E2 transferability pairing without GNN retraining by allowing one-class targets and using target-boundary pseudo metrics aligned with Week 7 semantics.
- [x] Generate W8.5 post-hoc acceptance audit for P1-P4, explicitly marking P3/P4 as post-hoc design-direction revisions rather than changes to the prerun Week 8 audit.
- [x] Update paper draft and project findings so original Week 8 E3 pass is downgraded to "invalid original LOFO; replaced by W8.5 balanced/hard audit."

### Week 8.5 Acceptance Focus
- Keep original Week 8 artifacts unchanged for auditability.
- Treat P1 and P2 as new evidence.
- Treat P3 and P4 as post-hoc interpretive revisions, not as retroactive changes to prerun thresholds.
- Do not force ROC-AUC for one-class targets; use `illegal_recall_at_youden`, `1 - mean_pred_illegal`, or other declared pseudo metrics when labels are incomplete.

### Week 8.5 Patch Results
- P1 revised E3:
  - Balanced licensed-negative LOFO passes: all family mean AUC values are `1.0000`.
  - Hard illegal-negative family audit fails as a family-recognition claim: mean hard-negative AUC is `0.6219`; tsars hard-negative AUC is `0.3864`.
  - Correct interpretation: the model robustly recognizes held-out family sites as illegal versus licensed controls, but does not consistently distinguish one illegal family from other illegal families.
- P2 expanded E2:
  - Expanded to `150` rows: `3` complete-label source regions x `10` targets x `5` seeds.
  - Target pseudo metrics are split as `30` binary AUC rows, `45` illegal-recall one-class rows, and `75` one-minus-mean-illegal-probability rows.
  - LogME still fails: Spearman rho is `0.0733`, below `0.60`, and it does not beat the distance baselines.
- P3 post-hoc E4:
  - H_E4b' direction-only passes at `4/5` seeds.
  - H_E4c' direction-only fails at `0/5` seeds.
- P4 post-hoc E1/E5:
  - H_E5a' passes: T3 full-minus-no-ccTLD drop is `0.0815`.
  - H_E5b' passes: T3 lex-only AUC is `0.9530`.
  - H_E5c' passes under transfer-specific dominance checks.
  - H_E1b' passes: T2_PH has a significant negative-transfer edge effect (`registered_via`, effect `-0.0833`).
  - H_E1c' passes: T2_ON has `2` significant edge effects under the post-hoc criterion.

## Phase 23: Security Patch, Corrected E5, Week9/10 Integration
- [x] Restore current Week 8 / Week 8.5 state and register the new prioritized patch plan.
- [x] P0 security and config repair:
  - [x] Redact real API token from `.claude/settings.local.json`.
  - [x] Add `.env.example`.
  - [x] Add `.gitignore` entries for local env, Claude local settings, key files, and large checkpoints.
  - [x] Correct `configs/week8.yaml` E5 config names and expanded modes without deleting old Week 8 outputs.
- [x] P1 corrected E5:
  - [x] Implement corrected feature and edge modes.
  - [x] Save per-run manifest, prediction, and audit files.
  - [x] Run the corrected E5 matrix to `output/week8_patch/`.
- [x] P2 Week 6 head ablation to `output/week6_head_ablation/`.
- [x] P3 Few-shot RQ4 to `output/week9/`.
- [x] P4 Formal hard-negative family audit outputs to `output/week9/`.
- [x] P5 Final RQ tables to `output/week10/`.
- [x] P6 README and paper draft updates.

### Phase 23 Guardrails
- Do not delete or overwrite old Week 8 / Week 8.5 results.
- New post-hoc outputs must be marked `is_posthoc=True`.
- New results must save manifests.
- New splits must include group-overlap audits.
- Preserve Graph v2 data semantics: no label-derived Website features, no active `redirects_to`, and no PA restoration.

### Phase 23 Result Notes
- P0 completed: local secret material was redacted, `.env.example` and `.gitignore` were added, and corrected E5 config names/modes are present in `configs/week8.yaml`.
- P1 corrected E5 completed under `output/week8_patch/` without overwriting `output/week8/`; all corrected E5 audit rows are marked `is_posthoc=True`.
- Corrected E5 supports the T3 lexical/ccTLD boundary interpretation: full-minus-no-ccTLD is `0.0815`, no-Website-lexical/graph-only AUC is `0.7134`, and lex-only minus graph-only is `0.2396`.
- P2 head ablation completed under `output/week6_head_ablation/`; `fixed_mean` is best at mean ROC-AUC `0.9735`, std `0.0163`, while the original attention head remains `0.9322`, std `0.0716`.
- P3 few-shot RQ4 completed under `output/week9/`; T2_PH improves from Week7 source-only `0.6417` to best few-shot `0.9250` at 5 shots per class, while T3 has no clear gain because source-only is already `0.9861`.
- P4 formal hard-negative family audit completed under `output/week9/`; mean hard-negative family AUC is `0.6218` and tsars is `0.3864`, so family-level illegal-site discrimination remains a negative/bounded claim.
- P5 final RQ tables completed under `output/week10/`; RQ1/RQ3 are mixed, RQ2 supports a lexical-boundary interpretation, RQ4 is target-dependent, and the model-integration story now includes head-ablation evidence.
- P6 completed by updating `README.md`, `paper/outline_v2.md`, and `paper/draft/sec5_final_rq_tables.md`.

## Phase 24: Final Cleanup and Reproducibility Audit
- [x] Scan all zero-byte Python files and replace them with explicit package-marker or placeholder docstrings.
- [x] Mark SeHGNN, HINormer, IRM, EERM, standalone StruRW, and related empty modules as not implemented and not citable for current results.
- [x] Lower over-strong wording in README, paper outline, and final RQ tables.
- [x] Add relative-path rerun commands while retaining Windows local conda examples.
- [x] Add `docs/reproducibility_checklist.md`.
- [x] Add `analysis/final_artifact_audit.py`.
- [x] Generate `output/final_artifact_audit.csv`.
- [x] Verify no zero-byte Python files remain.
- [x] Run Python compile checks and Graph v2 sanity check.

### Phase 24 Result Notes
- No existing experiment metrics or training outputs were changed.
- `output/final_artifact_audit.csv` contains `46` checks, all passing.
- `paper/draft/sec5_final_rq_tables.md` now starts with an interpretation guardrail: the tables do not prove broad cross-region generalization, and post-hoc rows are correction/diagnostic evidence rather than preregistered Week 8 confirmations.
- Current final RQ stance:
  - RQ1: mixed support.
  - RQ2: lexical/ccTLD boundary evidence.
  - RQ3: transferability unreliable.
  - RQ4: target-dependent few-shot support.
- Remaining references to SeHGNN/HINormer in README, paper outline, and task plan are negative/clarifying references, not implementation claims.

## Phase 25: Week 7 / Week 8 Interpretation Boundary Revision
- [x] Reframe Week 7 as transfer performance analysis rather than direct structural-generalization proof.
- [x] Create `docs/week7_transfer_interpretation.md`.
- [x] Generate `output/week7/metrics/week7_transfer_summary.csv`.
- [x] Generate `output/week7/metrics/week7_transfer_taxonomy.csv`.
- [x] Generate `output/week7/metrics/week7_to_week8_bridge.csv`.
- [x] Create `paper/draft/sec_week7_transfer_results.md`.
- [x] Reframe Week 8 as diagnostic analysis of transfer boundaries rather than clean hypothesis validation.
- [x] Create `docs/week8_diagnostic_revision.md`.
- [x] Generate `output/week8/metrics/E3_lofo_setting_comparison.csv`.
- [x] Generate `output/week8/metrics/E3_hard_negative_family_summary.csv`.
- [x] Generate `output/week8/metrics/E5_structure_vs_lexical_corrected_summary.csv`.
- [x] Generate `output/week8/metrics/week8_diagnostic_rollup.csv`.
- [x] Create `paper/draft/sec_week8_diagnostic_analysis.md`.
- [x] Create `docs/evidence_boundary_table.md`.
- [x] Extend `analysis/final_artifact_audit.py` to check the new interpretation-boundary tables.
- [x] Verify no original Week 7/Week 8/Week 8.5/Week 8 patch result tables were deleted or overwritten.

### Phase 25 Result Notes
- All new interpretation-boundary tables include `is_posthoc_diagnostic=True`.
- Week 7 taxonomy:
  - T1_Nordic: hard transfer.
  - T2_PH: few-shot recoverable hard transfer.
  - T2_ON: unreliable transfer boundary.
  - T3_DiagnoseFrance: shortcut-sensitive easy transfer.
- Week 8 diagnostic rollup:
  - E1: mixed.
  - E2: failed.
  - E3: limited.
  - E4: limited.
  - E5: failed.
- E3 setting comparison now explicitly separates licensed-negative LOFO from hard illegal-negative LOFO.
- E5 corrected summary states that T3 graph-only AUC is lower than lex-only, lex-only is strong, and no-ccTLD causes a performance drop.
- `output/final_artifact_audit.csv` now contains `60` checks, all passing.

## Phase 26: Final Superseded Draft Markers
- [x] Check old Week 7 draft `paper/draft/sec4_3_week7_transfer.md` against `paper/draft/sec_week7_transfer_results.md`.
- [x] Check old Week 8 draft `paper/draft/sec4_4_week8_mechanism.md` against `paper/draft/sec_week8_diagnostic_analysis.md`.
- [x] Add superseded note to the old Week 7 draft.
- [x] Add superseded note to the old Week 8 draft.
- [x] Add final paper-safe Week 7/8 interpretation pointer to `README.md`.
- [x] Add final paper-safe Week 7/8 interpretation pointer to `MANIFEST.md`.
- [x] Run `analysis/final_artifact_audit.py`.
- [x] Run `python -m py_compile` over `analysis/` and `src/`.

### Phase 26 Result Notes
- The older Week 7 and Week 8 draft files are preserved as history but marked superseded for paper-safe interpretation.
- Final Week 7/8 interpretation should follow `docs/evidence_boundary_table.md`, `paper/draft/sec_week7_transfer_results.md`, and `paper/draft/sec_week8_diagnostic_analysis.md`.
- `output/final_artifact_audit.csv` remains `60/60` pass.
- Recursive `py_compile` over `analysis/` and `src/` passed.

## Phase 27: Week 9 Few-Shot Calibration P0
- [x] Restore Week 7 / Week 8 interpretation boundaries before changing Week 9 artifacts.
- [x] Inspect existing `output/week9/` preliminary few-shot outputs and preserve them as historical artifacts.
- [x] Verify T2_PH and T3_DiagnoseFrance are binary target-test settings under the Week 7 split files.
- [x] Tighten `analysis/week9_fewshot_calibration.py` to emit the requested W9_E1/W9_E2/W9_E3/rollup schemas, per-run split files, and paper-safe docs.
- [x] Run Week 9 P0 for T2_PH and T3_DiagnoseFrance with shots `[0, 1, 3, 5, 10]`, seeds `[0, 1, 2, 3, 4]`, and at least `feature_condition=full`.
- [x] Generate required Week 9 plots and draft text.
- [x] Run final artifact audit and recursive Python compile checks.

### Phase 27 Guardrails
- Do not delete or overwrite Week 7, Week 8, Week 8.5, or Week 8 patch result trees.
- Preserve older Week 9 preliminary files such as `fewshot_raw_runs.csv` and add the formal Week 9 P0 files under new W9_* names.
- Because Week 7 did not save source-only classifier checkpoints, the formal P0 implementation may use the documented fallback: saved HeCo encoder checkpoint plus source_train and balanced target support joint fine-tuning.
- Requested 10-shot per class may be capped when the target adaptation pool is too small; record `actual_shot_per_class`.

### Phase 27 Result Notes
- Formal Week 9 P0/P1 generated `100` rows in `output/week9/metrics/W9_E1_fewshot_curve.csv`.
- Formal split files were saved for all `100` rows under `output/week9/splits/`; target few-shot support nodes do not overlap target test nodes.
- Formal W9 run/log/prediction counts are `80` each for shot > 0; the older `fewshot_*` preliminary artifacts remain preserved.
- T2_PH full-feature few-shot is stable: best requested shot is `10`, capped to actual `8` per class, with mean AUC `0.9542` and mean delta `+0.3125`.
- T3_DiagnoseFrance full-feature few-shot provides little additional benefit: best requested shot is `10`, mean AUC `0.9958`, mean delta `+0.0096`, and `success_rate_delta_ge_005 = 0.0`.
- P1 no-ccTLD was also run and summarized in `W9_E3_shortcut_aware_summary.csv`; shortcut dependence remains mixed and should not be overstated.
- `analysis/final_artifact_audit.py` now checks both historical Week 9 preliminary artifacts and formal W9_* outputs; latest audit is `75/75` pass.
- Recursive `compileall` over `analysis/` and `src/` passed.

## Phase 28: Week 9 Result Unification and Feature-Condition Expansion
- [x] Mark legacy Week 9 preliminary files as superseded by the W9_* formal result package.
- [x] Add `configs/week9_fewshot.yaml` with the formal Week 9 targets, shots, feature conditions, and post-hoc status.
- [x] Patch Week 9 runner so reruns can merge new feature-condition rows without dropping existing formal W9 rows.
- [x] Add T2_PH `no-website-lexical` and `graph-only` few-shot conditions across shots `[0, 1, 3, 5, 10]` and seeds `[0, 1, 2, 3, 4]`.
- [x] Generate `output/week9/metrics/fewshot_acceptance_audit_v2.csv` from `week9_fewshot_rollup.csv`.
- [x] Refresh Week 9 docs/plots, final artifact audit, and compile checks.

### Phase 28 Guardrails
- Treat `W9_E1_fewshot_curve.csv`, `W9_E2_fewshot_seed_stability.csv`, `W9_E3_shortcut_aware_summary.csv`, and `week9_fewshot_rollup.csv` as the authoritative Week 9 result口径.
- Do not relabel old preliminary results as current evidence; mark them legacy or superseded instead.
- Interpret T2_PH feature-condition expansion as a mechanism diagnostic for few-shot repair, not as automatic proof of pure structural generalization.

### Phase 28 Result Notes
- `configs/week9_fewshot.yaml` records the base Week 9 targets, shots, seeds, feature conditions, post-hoc status, and authoritative output files.
- The old preliminary files `fewshot_raw_runs.csv`, `fewshot_summary.csv`, and `fewshot_acceptance_audit.csv` now carry `week9_result_status=legacy_preformal`, `superseded_by`, and `paper_use=legacy_only_do_not_use_as_authoritative`.
- Formal Week 9 now has `150` W9_E1 rows and `150` split files:
  - T2_PH: full, no-ccTLD, no-website-lexical, graph-only.
  - T3_DiagnoseFrance: full, no-ccTLD.
- T2_PH requested 10-shot remains stable across all four feature conditions:
  - full mean AUC `0.9542`, delta `+0.3125`.
  - no-ccTLD mean AUC `0.9417`, delta `+0.3375`.
  - no-website-lexical mean AUC `0.9354`, delta `+0.3375`.
  - graph-only mean AUC `0.9354`, delta `+0.3375`.
- `fewshot_acceptance_audit_v2.csv` is generated from `week9_fewshot_rollup.csv`: T2_PH is `posthoc_support`; T3 is `no_clear_gain`.
- Final validation: `analysis/final_artifact_audit.py` reports `80/80` pass, `compileall` passes, and a split leakage check reports `150` rows / `150` splits / `0` support-test overlap.

## Phase 29: Week 10 Final Interpretability, Ablation, and RQ Freeze
- [x] Add `configs/week10_final.yaml` with Week4/6/7/8/8_patch/85/9 inputs and `output/week10/` destinations.
- [x] Replace/upgrade `analysis/week10_final_rq_tables.py` to generate RQ evidence, hypothesis rollup, and RQ-specific tables with safe/forbidden claims.
- [x] Implement `analysis/feature_bucket_transfer.py` for feature-bucket transfer summary and Figure 7.
- [x] Implement `analysis/week10_ablation_all.py` for the complete ablation table.
- [x] Implement `analysis/week10_fewshot_final.py` for final few-shot summary, Figure 8, and RQ4 table.
- [x] Implement `analysis/week10_hard_negative_final.py` for family hard-negative boundary summary, Figure 10, and table.
- [x] Implement `analysis/week10_deviation_case.py` and `src/explain/deviation_case_viz.py` for Figure 9 case visualization.
- [x] Implement `analysis/week10_acceptance_audit.py` and run Week10 acceptance checks.
- [x] Generate paper draft sections for RQ2/RQ3/RQ4 and failure discussion.
- [x] Run all Week10 scripts, final audit, compile checks, and update project logs.

### Phase 29 Guardrails
- Do not rerun or rewrite old experiments; Week10 is a reporting, auditing, explanation, and freeze layer.
- Do not delete Week4-Week9 outputs.
- Do not claim model-wide success, HeCo universal superiority, DANN/StruRW solved transfer, LogME reliability, or illegal-family attribution success.
- Required final frame: structural commonality, regional bias, evidence hierarchy, and few-shot local calibration jointly define the final interpretation.

### Phase 29 Result Notes
- `configs/week10_final.yaml` now centralizes Week4/5/6/7/8/8_patch/85/9 inputs and Week10 output directories.
- Week10 generated final RQ and hypothesis tables under `output/week10/metrics/` and RQ-specific tables under `output/week10/tables/`.
- Feature-bucket diagnostics generated `feature_bucket_transfer_summary.csv` and `fig7_feature_bucket_transfer.pdf`; T3 France is explicitly marked as lexical/ccTLD shortcut-sensitive rather than pure structural transfer.
- The complete ablation table generated `table6_ablation_all.csv` / `table_ablation_all.csv` with representation, transfer method, feature bucket, few-shot, and hard-negative family rows.
- Few-shot final outputs use the formal W9 rollup/stability口径 and preserve post-hoc flags; T2_PH is post-hoc support, T3 is saturated/no-clear-gain.
- Hard-negative final outputs preserve the negative family-boundary interpretation: illegal-vs-licensed is easier than illegal-vs-illegal-family discrimination.
- Deviation cases cover France ccTLD shortcut, Philippines few-shot recovery, and tsars/family hard-negative failure.
- Week10 acceptance audit reports `31/31` pass; final artifact audit reports `104/104` pass; recursive `compileall` over `analysis/` and `src/` passes.

## Phase 30: Week 11 Aggressive Improvement Layer
- [x] Read the Week11 design package from `<user_home>\Desktop\files.zip` and treat it as the source-of-truth for this phase.
- [x] Add `configs/week11.yaml` with prerun-frozen P1-P5 hypotheses, thresholds, inputs, and `output/week11/` destinations.
- [x] Implement and run P1 family-aware metric head using frozen HeCo embeddings, a SupCon projection head, hard-negative LOFO, alpha/shuffle ablations, centroid distances, and t-SNE.
- [x] Implement and run P2 IRM/EERM comparison with real-environment IRM where applicable and virtual-environment EERM for all four transfer tasks.
- [x] Implement and run P3 full transfer few-shot matrix by adding one-class boundary few-shot checks for T1_Nordic and T2_ON while reusing formal Week9 full-feature T2_PH/T3 rows.
- [x] Implement and run P4 paired bootstrap, Wilcoxon, and Holm-Bonferroni tests for W7 method pairs and W9 few-shot deltas.
- [x] Implement and run P5 paper-section expansion plus the honest evidence boundary diagram.
- [x] Generate Week11 acceptance audit, patch summary tables, score estimate, final artifact audit, compile checks, and P3 split leakage check.

### Phase 30 Guardrails
- Week11 is a new prerun-frozen improvement layer after Week10, not a rewrite of old experiments.
- Do not delete or overwrite Week4-Week10 output trees.
- Failed Week11 hypotheses must remain failed and be reported as boundary evidence.
- P1 family metric learning must not be written as illegal-family attribution success.
- P2 IRM/EERM must not be written as solving transfer, even where a local improvement is observed.
- P3 one-class targets must not be forced into standard ROC-AUC claims.

### Phase 30 Result Notes
- Week11 generated `output/week11/metrics/P1_family_metric_lofo.csv` with `35` rows, `P1_alpha_ablation.csv` with `40` rows, and `P1_family_centroid_distance.csv` with `105` rows.
- P1 did not improve the family hard-negative boundary: mean hard-negative AUC is `0.1259`, tsars AUC is `0.0741`, and all four P1 hypotheses fail. This should be written as a negative result for the current family metric design.
- Week11 generated `output/week11/metrics/P2_method_comparison.csv` with `125` rows and `P2_eerm_K_sensitivity.csv` with `4` rows. P2 passes H_P2a/H_P2b/H_P2c, but should be interpreted as method-boundary evidence rather than a universal transfer solution.
- Week11 generated `output/week11/metrics/P3_full_fewshot_matrix.csv` with `100` rows. T1_Nordic 5-shot illegal recall reaches `0.4581`, below the preregistered `0.50` threshold, while T2_ON 5-shot mean predicted illegal probability is `0.2145`, passing the licensed-only boundary criterion.
- P4 passes: W7 has `9` Holm-significant bootstrap method pairs, and W9 has `5` Holm-significant few-shot-vs-0-shot comparisons.
- P5 passes the writing checks, including `section5_chars >= 3500`, RQ-plus-section5 text length `>= 8000`, and explicit `docs/evidence_boundary_table.md` citation.
- Week11 acceptance audit reports `9/15` pass and `6/15` fail. The score estimate remains a bounded improvement, not a model-wide success claim.
- Final artifact audit now reports `128/128` pass; recursive `compileall` over `analysis/` and `src/` passes; P3 split leakage check reports `40` split files and `0` support-test overlaps.
