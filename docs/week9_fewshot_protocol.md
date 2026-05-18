# Week 9 Few-Shot Protocol

The protocol uses existing Week 7 target adaptation pools and target test splits. Few-shot samples are drawn only from `target_adapt_unlabeled`; selected support nodes are removed from that pool and never appear in `target_test`.

Sampling is class-balanced. If the requested shot count exceeds the minority-class adaptation pool, the actual balanced support size is capped and recorded in `actual_shot_per_class`. For T2_PH, requested 10-shot is capped at 8 per class in the current Week 7 splits.

Training mode: saved HeCo encoder checkpoint plus source_train and balanced target support joint fine-tuning. This is the documented fallback because Week 7 did not save source-only classifier checkpoint files. The 0-shot rows are source-only baselines from corrected E5 for the matching feature condition, split seed, and checkpoint seed.

Every result row records target_id, shot, actual_shot_per_class, seed, checkpoint_seed, feature_condition, source_only_auc, fewshot_auc, delta_auc, target metrics, and target train/test counts. Selected target support sample ids are saved in `output/week9/audits/W9_fewshot_support_samples.csv`, and every run writes its formal split under `output/week9/splits/`.
