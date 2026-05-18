# Week 9 Few-Shot Experiment Design

Title: Few-shot Target Calibration for Recoverable Transfer.

Week 9 tests whether a small number of balanced target labels can repair hard transfer targets. It is not framed as universal few-shot improvement. The primary target is T2_PH; T3_DiagnoseFrance is a saturated / shortcut-sensitive control. T1_Nordic and T2_ON are one-class targets and are not included in the standard ROC-AUC main experiment.

Main questions:

- Does few-shot target calibration improve source-only transfer?
- Is improvement concentrated on the hard transfer target?
- Is improvement stable across random seeds?
- Does improvement depend on ccTLD / Website lexical shortcut information?

Implemented P0/P1 matrix:

- targets: T2_PH, T3_DiagnoseFrance
- shots: 0, 1, 3, 5, 10
- seeds: 0, 1, 2, 3, 4, mapped to existing split/checkpoint seeds 42, 43, 44, 45, 46
- base feature conditions in `configs/week9_fewshot.yaml`: full, no-ccTLD
- training fallback: saved HeCo encoder checkpoint plus source_train and balanced target support joint fine-tuning

Formal result feature-condition coverage:

- T2_PH: full, graph-only, no-ccTLD, no-website-lexical
- T3_DiagnoseFrance: full, no-ccTLD

All outputs are post-hoc calibration evidence and should not be treated as preregistered Week 8 evidence.
