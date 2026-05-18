# Week 9 Few-Shot Target Calibration

Week 9 evaluates few-shot target calibration for recoverable transfer. The result should not be read as universal few-shot improvement. Instead, few-shot calibration is target-dependent: it may repair a hard but recoverable target, while saturated or shortcut-sensitive targets may show little additional gain.

## Rollup

| target_id | target_type | source_only_auc | best_shot | best_mean_auc | best_std_auc | best_mean_delta_auc | success_rate_positive | success_rate_delta_ge_005 | success_rate | shortcut_condition_result | final_status | paper_safe_claim | overclaim_to_avoid | is_posthoc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| T2_PH | primary hard-transfer target | 0.6417 | 10 | 0.9542 | 0.0401 | 0.3125 | 1.0000 | 1.0000 | 1.0000 | mixed shortcut sensitivity | recoverable_target_supported | Few-shot calibration substantially improves a hard but recoverable target. | Do not claim few-shot universally improves transfer; this is a hard but recoverable target-specific result. | True |
| T3_DiagnoseFrance | saturated / shortcut-sensitive control target | 0.9861 | 10 | 0.9958 | 0.0043 | 0.0096 | 0.4000 | 0.0000 | 0.4000 | mixed shortcut sensitivity | saturated_no_clear_gain | Few-shot provides little additional benefit for saturated / shortcut-sensitive target. | Do not use high T3 AUC as structural-generalization evidence; Week 8 shows lexical/ccTLD assistance. | True |

## Seed Stability

| target_id | shot | feature_condition | training_mode | mean_auc | std_auc | mean_delta_auc | success_rate_positive | success_rate_delta_ge_005 | stability_label |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| T2_PH | 0 | full | source_only_baseline_from_corrected_E5 | 0.6417 | 0.1369 | 0.0000 | 0.0000 | 0.0000 | no meaningful improvement |
| T2_PH | 0 | no-ccTLD | source_only_baseline_from_corrected_E5 | 0.6042 | 0.1816 | 0.0000 | 0.0000 | 0.0000 | no meaningful improvement |
| T2_PH | 5 | full | heco_encoder_checkpoint_source_plus_balanced_target_joint_finetune | 0.9375 | 0.0390 | 0.2958 | 1.0000 | 1.0000 | stable improvement |
| T2_PH | 5 | no-ccTLD | heco_encoder_checkpoint_source_plus_balanced_target_joint_finetune | 0.9042 | 0.1339 | 0.3000 | 0.8000 | 0.8000 | stable improvement |
| T2_PH | 10 | full | heco_encoder_checkpoint_source_plus_balanced_target_joint_finetune | 0.9542 | 0.0401 | 0.3125 | 1.0000 | 1.0000 | stable improvement |
| T2_PH | 10 | no-ccTLD | heco_encoder_checkpoint_source_plus_balanced_target_joint_finetune | 0.9417 | 0.0401 | 0.3375 | 1.0000 | 1.0000 | stable improvement |
| T3_DiagnoseFrance | 0 | full | source_only_baseline_from_corrected_E5 | 0.9861 | 0.0208 | 0.0000 | 0.0000 | 0.0000 | no meaningful improvement |
| T3_DiagnoseFrance | 0 | no-ccTLD | source_only_baseline_from_corrected_E5 | 0.9046 | 0.0507 | 0.0000 | 0.0000 | 0.0000 | no meaningful improvement |
| T3_DiagnoseFrance | 5 | full | heco_encoder_checkpoint_source_plus_balanced_target_joint_finetune | 0.8863 | 0.2424 | -0.0998 | 0.4000 | 0.0000 | unstable/sample-sensitive |
| T3_DiagnoseFrance | 5 | no-ccTLD | heco_encoder_checkpoint_source_plus_balanced_target_joint_finetune | 0.9652 | 0.0358 | 0.0606 | 0.8000 | 0.6000 | moderate improvement |
| T3_DiagnoseFrance | 10 | full | heco_encoder_checkpoint_source_plus_balanced_target_joint_finetune | 0.9958 | 0.0043 | 0.0096 | 0.4000 | 0.0000 | no meaningful improvement |
| T3_DiagnoseFrance | 10 | no-ccTLD | heco_encoder_checkpoint_source_plus_balanced_target_joint_finetune | 0.9614 | 0.0392 | 0.0568 | 1.0000 | 0.4000 | unstable/sample-sensitive |

## Shortcut-Aware Interpretation

| target_id | shot | feature_condition | training_mode | mean_delta_vs_source_only | success_rate_delta_ge_005 | shortcut_dependency_label | paper_safe_claim |
| --- | --- | --- | --- | --- | --- | --- | --- |
| T2_PH | 0 | full | source_only_baseline_from_corrected_E5 | 0.0000 | 0.0000 | mixed shortcut sensitivity | Few-shot calibration has potential for T2_PH but remains sample-sensitive. |
| T2_PH | 0 | graph-only | source_only_baseline_from_corrected_E5 | 0.0000 | 0.0000 | mixed shortcut sensitivity | Few-shot calibration has potential for T2_PH but remains sample-sensitive. |
| T2_PH | 0 | no-ccTLD | source_only_baseline_from_corrected_E5 | 0.0000 | 0.0000 | mixed shortcut sensitivity | Few-shot calibration has potential for T2_PH but remains sample-sensitive. |
| T2_PH | 0 | no-website-lexical | source_only_baseline_from_corrected_E5 | 0.0000 | 0.0000 | mixed shortcut sensitivity | Few-shot calibration has potential for T2_PH but remains sample-sensitive. |
| T2_PH | 1 | full | heco_encoder_checkpoint_source_plus_balanced_target_joint_finetune | 0.1667 | 0.8000 | mixed shortcut sensitivity | Few-shot calibration substantially improves a hard but recoverable target. |
| T2_PH | 1 | graph-only | heco_encoder_checkpoint_source_plus_balanced_target_joint_finetune | -0.0375 | 0.2000 | partly shortcut-dependent | Few-shot calibration has potential for T2_PH but remains sample-sensitive. |
| T2_PH | 1 | no-ccTLD | heco_encoder_checkpoint_source_plus_balanced_target_joint_finetune | 0.2542 | 1.0000 | mixed shortcut sensitivity | Few-shot calibration substantially improves a hard but recoverable target. |
| T2_PH | 1 | no-website-lexical | heco_encoder_checkpoint_source_plus_balanced_target_joint_finetune | -0.0375 | 0.2000 | partly shortcut-dependent | Few-shot calibration has potential for T2_PH but remains sample-sensitive. |
| T2_PH | 3 | full | heco_encoder_checkpoint_source_plus_balanced_target_joint_finetune | 0.3000 | 1.0000 | partly shortcut-dependent | Few-shot calibration substantially improves a hard but recoverable target. |
| T2_PH | 3 | graph-only | heco_encoder_checkpoint_source_plus_balanced_target_joint_finetune | 0.2708 | 0.8000 | partly shortcut-dependent | Few-shot calibration substantially improves a hard but recoverable target. |
| T2_PH | 3 | no-ccTLD | heco_encoder_checkpoint_source_plus_balanced_target_joint_finetune | 0.2583 | 0.8000 | partly shortcut-dependent | Few-shot calibration substantially improves a hard but recoverable target. |
| T2_PH | 3 | no-website-lexical | heco_encoder_checkpoint_source_plus_balanced_target_joint_finetune | 0.2708 | 0.8000 | partly shortcut-dependent | Few-shot calibration substantially improves a hard but recoverable target. |
| T2_PH | 5 | full | heco_encoder_checkpoint_source_plus_balanced_target_joint_finetune | 0.2958 | 1.0000 | mixed shortcut sensitivity | Few-shot calibration substantially improves a hard but recoverable target. |
| T2_PH | 5 | graph-only | heco_encoder_checkpoint_source_plus_balanced_target_joint_finetune | 0.2583 | 1.0000 | partly shortcut-dependent | Few-shot calibration substantially improves a hard but recoverable target. |
| T2_PH | 5 | no-ccTLD | heco_encoder_checkpoint_source_plus_balanced_target_joint_finetune | 0.3000 | 0.8000 | mixed shortcut sensitivity | Few-shot calibration substantially improves a hard but recoverable target. |
| T2_PH | 5 | no-website-lexical | heco_encoder_checkpoint_source_plus_balanced_target_joint_finetune | 0.2583 | 1.0000 | partly shortcut-dependent | Few-shot calibration substantially improves a hard but recoverable target. |
| T2_PH | 10 | full | heco_encoder_checkpoint_source_plus_balanced_target_joint_finetune | 0.3125 | 1.0000 | mixed shortcut sensitivity | Few-shot calibration substantially improves a hard but recoverable target. |
| T2_PH | 10 | graph-only | heco_encoder_checkpoint_source_plus_balanced_target_joint_finetune | 0.3375 | 1.0000 | mixed shortcut sensitivity | Few-shot calibration substantially improves a hard but recoverable target. |
| T2_PH | 10 | no-ccTLD | heco_encoder_checkpoint_source_plus_balanced_target_joint_finetune | 0.3375 | 1.0000 | mixed shortcut sensitivity | Few-shot calibration substantially improves a hard but recoverable target. |
| T2_PH | 10 | no-website-lexical | heco_encoder_checkpoint_source_plus_balanced_target_joint_finetune | 0.3375 | 1.0000 | mixed shortcut sensitivity | Few-shot calibration substantially improves a hard but recoverable target. |
| T3_DiagnoseFrance | 0 | full | source_only_baseline_from_corrected_E5 | 0.0000 | 0.0000 | partly shortcut-dependent | Few-shot provides little additional benefit for saturated / shortcut-sensitive target. |
| T3_DiagnoseFrance | 0 | no-ccTLD | source_only_baseline_from_corrected_E5 | 0.0000 | 0.0000 | partly shortcut-dependent | Few-shot provides little additional benefit for saturated / shortcut-sensitive target. |
| T3_DiagnoseFrance | 1 | full | heco_encoder_checkpoint_source_plus_balanced_target_joint_finetune | -0.0275 | 0.0000 | partly shortcut-dependent | Few-shot provides little additional benefit for saturated / shortcut-sensitive target. |
| T3_DiagnoseFrance | 1 | no-ccTLD | heco_encoder_checkpoint_source_plus_balanced_target_joint_finetune | -0.0491 | 0.2000 | partly shortcut-dependent | Few-shot provides little additional benefit for saturated / shortcut-sensitive target. |
| T3_DiagnoseFrance | 3 | full | heco_encoder_checkpoint_source_plus_balanced_target_joint_finetune | -0.0045 | 0.0000 | mixed shortcut sensitivity | Few-shot provides little additional benefit for saturated / shortcut-sensitive target. |
| T3_DiagnoseFrance | 3 | no-ccTLD | heco_encoder_checkpoint_source_plus_balanced_target_joint_finetune | 0.0432 | 0.4000 | mixed shortcut sensitivity | Few-shot provides little additional benefit for saturated / shortcut-sensitive target. |
| T3_DiagnoseFrance | 5 | full | heco_encoder_checkpoint_source_plus_balanced_target_joint_finetune | -0.0998 | 0.0000 | mixed shortcut sensitivity | Few-shot provides little additional benefit for saturated / shortcut-sensitive target. |
| T3_DiagnoseFrance | 5 | no-ccTLD | heco_encoder_checkpoint_source_plus_balanced_target_joint_finetune | 0.0606 | 0.6000 | mixed shortcut sensitivity | Few-shot provides little additional benefit for saturated / shortcut-sensitive target. |
| T3_DiagnoseFrance | 10 | full | heco_encoder_checkpoint_source_plus_balanced_target_joint_finetune | 0.0096 | 0.0000 | mixed shortcut sensitivity | Few-shot provides little additional benefit for saturated / shortcut-sensitive target. |
| T3_DiagnoseFrance | 10 | no-ccTLD | heco_encoder_checkpoint_source_plus_balanced_target_joint_finetune | 0.0568 | 0.4000 | mixed shortcut sensitivity | Few-shot provides little additional benefit for saturated / shortcut-sensitive target. |

## Feature-Condition Expansion

For T2_PH, the expanded feature-condition audit shows that requested 10-shot calibration remains strong even after removing Website lexical features or using the graph-only condition: full: mean AUC 0.9542, delta +0.3125, success(delta>=0.05) 1.00; no-ccTLD: mean AUC 0.9417, delta +0.3375, success(delta>=0.05) 1.00; no-website-lexical: mean AUC 0.9354, delta +0.3375, success(delta>=0.05) 1.00; graph-only: mean AUC 0.9354, delta +0.3375, success(delta>=0.05) 1.00. This supports a stronger recoverable-target claim than the full/no-ccTLD matrix alone, but it should still be written as target calibration evidence rather than proof of broad pure structural generalization.

Paper-safe interpretation: T2_PH is the key hard-transfer target. If its few-shot improvement is stable, the safe claim is that few-shot calibration substantially improves a hard but recoverable target. T3 France is already saturated and Week 8 E5 showed lexical/ccTLD assistance, so little additional few-shot gain should be interpreted as target saturation rather than structural proof.
