# Final RQ Evidence Tables

These tables freeze the Week10 interpretation layer. They summarize evidence boundaries rather than claiming comprehensive success.

## RQ Evidence

| rq | evidence_item | metric | value | status | is_posthoc | paper_safe_claim | paper_forbidden_claim |
| --- | --- | --- | --- | --- | --- | --- | --- |
| RQ1 | cross_verified_vs_licensed | roc_auc | 0.9797619047619048 | support_illegal_vs_licensed | False | The model separates illegal websites from licensed controls in the cross-verified evidence layer. | Do not claim the model performs reliable illegal-family attribution. |
| RQ1 | hard_negative_family_boundary | mean_hard_negative_auc | 0.6218496472663139 | family_boundary_failed | True | Illegal-vs-licensed separability is strong, but family-level discrimination among illegal sites remains limited. | Do not claim successful illegal-family attribution or robust unseen-family recognition. |
| RQ1 | head_ablation_best | roc_auc_mean | 0.9735092576177454 | mixed_model_integration | True | Head ablation suggests stability depends on downstream head design. | Do not claim HeCo universally outperforms HeteroGNN or that one head solves all settings. |
| RQ2 | T3_ccTLD_sensitivity | full_minus_no_cctld_auc | 0.0815357374918778 | regional_shortcut_boundary | True | T3 France transfer is substantially assisted by ccTLD/local lexical boundary information. | Do not present T3 France as pure structural transfer. |
| RQ2 | T3_lexical_vs_graph | lex_only_minus_graph_only_auc | 0.2395692007797271 | regional_lexical_boundary | True | Regional heterogeneity includes lexical/ccTLD shortcuts that must be separated from structural evidence. | Do not claim graph-only regional generalization from high full-feature T3 AUC. |
| RQ2 | T3_C1_full | primary_metric_value_mean | 0.9861234567901236 | diagnostic_context | True | Feature-bucket diagnostics expose which regional signals support or weaken transfer. | Do not turn a single feature bucket into a universal mechanism claim. |
| RQ2 | T3_C2_no_cctld | primary_metric_value_mean | 0.9045877192982456 | diagnostic_context | True | Feature-bucket diagnostics expose which regional signals support or weaken transfer. | Do not turn a single feature bucket into a universal mechanism claim. |
| RQ2 | T3_C3_no_website_lexical | primary_metric_value_mean | 0.7134294996751137 | diagnostic_context | True | Feature-bucket diagnostics expose which regional signals support or weaken transfer. | Do not turn a single feature bucket into a universal mechanism claim. |
| RQ2 | T3_C4_graph_only | primary_metric_value_mean | 0.7134294996751137 | diagnostic_context | True | Feature-bucket diagnostics expose which regional signals support or weaken transfer. | Do not turn a single feature bucket into a universal mechanism claim. |
| RQ2 | T3_C5_lex_only | primary_metric_value_mean | 0.9529987004548408 | diagnostic_context | True | Feature-bucket diagnostics expose which regional signals support or weaken transfer. | Do not turn a single feature bucket into a universal mechanism claim. |
| RQ2 | T3_C6_no_edges | primary_metric_value_mean | 0.9529987004548408 | diagnostic_context | True | Feature-bucket diagnostics expose which regional signals support or weaken transfer. | Do not turn a single feature bucket into a universal mechanism claim. |
| RQ2 | T3_C7_no_cert_edge | primary_metric_value_mean | 0.9154392462638076 | diagnostic_context | True | Feature-bucket diagnostics expose which regional signals support or weaken transfer. | Do not turn a single feature bucket into a universal mechanism claim. |
| RQ2 | T3_C8_no_registrar_edge | primary_metric_value_mean | 0.9210526315789472 | diagnostic_context | True | Feature-bucket diagnostics expose which regional signals support or weaken transfer. | Do not turn a single feature bucket into a universal mechanism claim. |
| RQ2 | T3_C9_infra_edges_only | primary_metric_value_mean | 0.9861234567901236 | diagnostic_context | True | Feature-bucket diagnostics expose which regional signals support or weaken transfer. | Do not turn a single feature bucket into a universal mechanism claim. |
| RQ3 | T1_Nordic | illegal_recall_at_youden | 0.2064516129032258 | hard transfer | True | Transfer is target-dependent and must be interpreted with target-specific metric boundaries. | Do not claim DANN/StruRW solve cross-domain transfer or force ROC-AUC onto one-class targets. |
| RQ3 | T2_PH | roc_auc | 0.6416666666666666 | few-shot recoverable hard transfer | True | Transfer is target-dependent and must be interpreted with target-specific metric boundaries. | Do not claim DANN/StruRW solve cross-domain transfer or force ROC-AUC onto one-class targets. |
| RQ3 | T2_ON | mean_pred_illegal | 0.3027596900275405 | unreliable transfer boundary | True | Transfer is target-dependent and must be interpreted with target-specific metric boundaries. | Do not claim DANN/StruRW solve cross-domain transfer or force ROC-AUC onto one-class targets. |
| RQ3 | T3_DiagnoseFrance | roc_auc | 0.9861234567901236 | shortcut-sensitive easy transfer | True | Transfer is target-dependent and must be interpreted with target-specific metric boundaries. | Do not claim DANN/StruRW solve cross-domain transfer or force ROC-AUC onto one-class targets. |
| RQ3 | LogME_transferability | diagnostic_status | failed | transferability_proxy_failed | True | LogME is not reliable as a transferability prescreen in this setting. | Do not claim LogME reliably predicts transfer success. |
| RQ4 | T2_PH | best_mean_delta_auc | 0.3125 | posthoc_support | True | Few-shot calibration substantially improves a hard but recoverable target. | Do not claim few-shot universally improves transfer. |
| RQ4 | T3_DiagnoseFrance | best_mean_delta_auc | 0.0096419753086419 | no_clear_gain | True | Few-shot provides little additional benefit for saturated / shortcut-sensitive target. | Do not write saturated T3 as a few-shot success or structural proof. |

## Hypothesis Rollup

| evidence_layer | experiment_id | hypothesis | status | is_posthoc | paper_safe_claim | paper_forbidden_claim |
| --- | --- | --- | --- | --- | --- | --- |
| week7_transfer |  |  | fail_but_interpretable | True | Diagnostic evidence should be interpreted within its stated boundary. | Do not overgeneralize diagnostic or post-hoc evidence. |
| week7_transfer |  |  | pass | True | Diagnostic evidence should be interpreted within its stated boundary. | Do not overgeneralize diagnostic or post-hoc evidence. |
| week7_transfer |  |  | pass | True | Diagnostic evidence should be interpreted within its stated boundary. | Do not overgeneralize diagnostic or post-hoc evidence. |
| week7_transfer |  |  | fail | True | Diagnostic evidence should be interpreted within its stated boundary. | Do not overgeneralize diagnostic or post-hoc evidence. |
| week8_diagnostic | E1 |  | mixed | True | Edge channels are diagnostically useful but not uniformly beneficial across targets. | Do not claim edge structure generally explains transfer success. |
| week8_diagnostic | E2 |  | failed | True | LogME is not reliable as a transferability proxy in this setting. | Do not use LogME as a validated prescreen. |
| week8_diagnostic | E3 |  | limited | True | Illegal-vs-licensed separability is strong; family-level generalization among illegal sites is limited. | Do not claim strong unseen-family generalization from licensed-negative LOFO. |
| week8_diagnostic | E4 |  | limited | True | Control-group separability is supported, but embedding-distance explanation is insufficient. | Do not claim embedding distance explains the transfer mechanism. |
| week8_diagnostic | E5 |  | failed | True | T3 France transfer is substantially lexical/ccTLD-assisted. | Do not present T3 France as pure structural transfer evidence. |
| week8_patch_e5 |  | W8P_E5_ccTLD_sensitivity | diagnostic_support | True | Diagnostic evidence should be interpreted within its stated boundary. | Do not overgeneralize diagnostic or post-hoc evidence. |
| week8_patch_e5 |  | W8P_E5_website_lexical_removed | diagnostic_support | True | Diagnostic evidence should be interpreted within its stated boundary. | Do not overgeneralize diagnostic or post-hoc evidence. |
| week8_patch_e5 |  | W8P_E5_lex_only_vs_graph_only | diagnostic_support | True | Diagnostic evidence should be interpreted within its stated boundary. | Do not overgeneralize diagnostic or post-hoc evidence. |
| week8_patch_e5 |  | W8P_E5_no_edges_equivalent_to_lex_only | consistent | True | Diagnostic evidence should be interpreted within its stated boundary. | Do not overgeneralize diagnostic or post-hoc evidence. |
| week8_patch_e5 |  | W8P_E5_edge_sensitivity | diagnostic_support | True | Diagnostic evidence should be interpreted within its stated boundary. | Do not overgeneralize diagnostic or post-hoc evidence. |
| week8_patch_e5 |  | W8P_E5_infra_only | diagnostic_support | True | Diagnostic evidence should be interpreted within its stated boundary. | Do not overgeneralize diagnostic or post-hoc evidence. |
| week85_patch |  | E3_balanced_licensed_negative | pass | True | Diagnostic evidence should be interpreted within its stated boundary. | Do not overgeneralize diagnostic or post-hoc evidence. |
| week85_patch |  | E3_hard_illegal_family_detection | fail | True | Diagnostic evidence should be interpreted within its stated boundary. | Do not overgeneralize diagnostic or post-hoc evidence. |
| week85_patch |  | E3_hard_tsars_family_detection | fail | True | Diagnostic evidence should be interpreted within its stated boundary. | Do not overgeneralize diagnostic or post-hoc evidence. |
| week85_patch |  | E2_expanded_LogME_predicts_quasi_metric | fail | True | Diagnostic evidence should be interpreted within its stated boundary. | Do not overgeneralize diagnostic or post-hoc evidence. |
| week85_patch |  | E2_expanded_LogME_beats_distance_baselines | fail | True | Diagnostic evidence should be interpreted within its stated boundary. | Do not overgeneralize diagnostic or post-hoc evidence. |
| week85_patch |  | H_E4b_prime_direction_only | pass | True | Diagnostic evidence should be interpreted within its stated boundary. | Do not overgeneralize diagnostic or post-hoc evidence. |
| week85_patch |  | H_E4c_prime_direction_only | fail | True | Diagnostic evidence should be interpreted within its stated boundary. | Do not overgeneralize diagnostic or post-hoc evidence. |
| week85_patch |  | H_E5a_prime_cctld_key | pass | True | Diagnostic evidence should be interpreted within its stated boundary. | Do not overgeneralize diagnostic or post-hoc evidence. |
| week85_patch |  | H_E5b_prime_lex_only_working | pass | True | Diagnostic evidence should be interpreted within its stated boundary. | Do not overgeneralize diagnostic or post-hoc evidence. |
| week85_patch |  | H_E5c_prime_transfer_specific_dominance | pass | True | Diagnostic evidence should be interpreted within its stated boundary. | Do not overgeneralize diagnostic or post-hoc evidence. |
| week85_patch |  | H_E1b_prime_T2PH_negative_transfer_edge | pass | True | Diagnostic evidence should be interpreted within its stated boundary. | Do not overgeneralize diagnostic or post-hoc evidence. |
| week85_patch |  | H_E1c_prime_T2ON_any_significant_edge | pass | True | Diagnostic evidence should be interpreted within its stated boundary. | Do not overgeneralize diagnostic or post-hoc evidence. |
| week9_fewshot |  |  | posthoc_support | True | Diagnostic evidence should be interpreted within its stated boundary. | Do not overgeneralize diagnostic or post-hoc evidence. |
| week9_fewshot |  |  | no_clear_gain | True | Diagnostic evidence should be interpreted within its stated boundary. | Do not overgeneralize diagnostic or post-hoc evidence. |
| week9_hard_negative |  | hard_negative_family_mean_auc | fail | True | Diagnostic evidence should be interpreted within its stated boundary. | Do not overgeneralize diagnostic or post-hoc evidence. |
| week9_hard_negative |  | hard_negative_tsars_auc | fail | True | Diagnostic evidence should be interpreted within its stated boundary. | Do not overgeneralize diagnostic or post-hoc evidence. |
