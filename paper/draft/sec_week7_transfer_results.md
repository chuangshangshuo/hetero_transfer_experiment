# Week 7 Transfer Results: Performance, Not Structural Proof

Week 7 evaluates transfer performance across four target settings using the
Graph v2 / HeCo stack. The purpose of this section is not to prove cross-domain
structural generalization. Instead, it reports source-only and adaptation
performance and then defers mechanism attribution to the Week 8 diagnostic
analysis.

The model achieved high transfer performance on some targets, but the evidence
is target-dependent. A high AUC is not sufficient to infer that the model used
portable graph structure rather than lexical, geographic, or infrastructure
shortcuts.

## Transfer Taxonomy

The revised taxonomy is saved in
`output/week7/metrics/week7_transfer_taxonomy.csv`.

| Transfer | Main Observation | Revised Class | Paper-Safe Interpretation |
| --- | --- | --- | --- |
| T1_Nordic | One-class Denmark target; source-only illegal recall is low. | hard transfer | Report as hard one-class transfer, not as structural proof. |
| T2_PH | Source-only AUC is 0.6417 and best Week 7 adaptation remains below 0.80. | few-shot recoverable hard transfer | Week 9 few-shot reaches AUC 0.9250, so the target is recoverable with small supervision. |
| T2_ON | Ontario has no illegal target labels. | unreliable transfer boundary | Use only licensed-consistency/anomaly-style metrics. |
| T3_DiagnoseFrance | Source-only AUC is 0.9861. | shortcut-sensitive easy transfer | High performance must be qualified by Week 8 E5. |

## Bridge to Week 8

T3 France requires the strongest interpretive caution. The high Week 7 AUC
should not be presented as pure structural transfer evidence. The later
corrected E5 diagnostic shows that no-ccTLD removal causes a material
performance drop, graph-only AUC is lower than lex-only AUC, and lex-only
features remain strong. Therefore T3 France is substantially lexical/ccTLD
assisted.

T2_PH should also be written carefully. Week 7 unsupervised transfer is weak,
but Week 9 few-shot evaluation shows recovery with small target supervision.
This supports a target-dependent few-shot recovery story, not a claim that
unsupervised transfer is generally solved.

T1_Nordic and T2_ON are one-class target settings, so ROC-AUC is undefined.
They are useful boundary cases, but they should not be mixed into a standard
binary transfer narrative.
