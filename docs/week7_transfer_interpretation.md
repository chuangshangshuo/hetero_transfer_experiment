# Week 7 Transfer Interpretation Revision

This note revises the paper interpretation of Week 7 without changing any
saved Week 7 results. The Week 7 matrix should be treated as transfer
performance analysis, not as direct proof of structural generalization.

## Scope

Week 7 reports source-only and adaptation transfer performance across four
target settings:

- `T1_Nordic`: Sweden + Spain + Italy -> Denmark
- `T2_PH`: European pooled -> Philippines
- `T2_ON`: European pooled -> Ontario
- `T3_DiagnoseFrance`: Spain + Italy -> France

The goal is to measure how well the existing Graph v2 / HeCo stack transfers
under the available labels and target metrics. A high target score is not, by
itself, evidence that the model learned portable structural mechanisms.

Week 8 is responsible for diagnosing the source of high or low Week 7 scores:
edge sensitivity, proxy failure, hard-negative family behavior, and
structure-vs-lexical boundaries.

## Interpretation Rules

- Do not write that Week 7 proves structural generalization.
- Do not force ROC-AUC for one-class targets.
- Treat source-only transfer as an observed performance baseline.
- Treat DANN and StruRW as context-dependent methods, not general winners.
- Use Week 8 and Week 8.5 diagnostics to explain what a high or low Week 7
  score can mean.
- Mark Week 9 few-shot evidence as post-hoc recovery evidence.

## Transfer Classes

The current taxonomy is saved at
`output/week7/metrics/week7_transfer_taxonomy.csv`.

| Transfer | Class | Safe Interpretation |
| --- | --- | --- |
| T1_Nordic | hard transfer | Low Denmark illegal recall under one-class target labels; not a structural-generalization test. |
| T2_PH | few-shot recoverable hard transfer | Unsupervised transfer is weak, but post-hoc few-shot target supervision recovers performance. |
| T2_ON | unreliable transfer boundary | Ontario is a one-class licensed target; use licensed-consistency/anomaly-style metrics only. |
| T3_DiagnoseFrance | shortcut-sensitive easy transfer | High AUC, but Week 8 E5 shows substantial lexical/ccTLD assistance. |

## Week 7 to Week 8 Bridge

The bridge table is saved at
`output/week7/metrics/week7_to_week8_bridge.csv`.

| Week 7 Observation | Initial Risk | Week 8 Diagnostic | Revised Interpretation |
| --- | --- | --- | --- |
| T3 source-only AUC is high. | Mistaking high AUC for structural generalization. | Corrected E5 shows no-ccTLD drop, weak graph-only AUC, and strong lex-only AUC. | T3 is shortcut-sensitive and lexical/ccTLD-assisted. |
| T2_PH unsupervised transfer is below 0.80. | Treating this as total non-transferability. | Week 9 few-shot reaches AUC 0.9250 at 5 shots per class. | T2_PH is hard but few-shot recoverable. |
| T1_Nordic has one-class target labels and low recall. | Forcing invalid ROC-AUC. | One-class metric boundary plus hard-negative family audits. | T1 is hard transfer evidence, not structural proof. |
| T2_ON has no illegal target labels. | Calling anomaly-style scores a standard transfer victory. | One-class target policy and edge/channel diagnostics. | T2_ON is licensed-consistency boundary evidence only. |

## Paper-Safe Summary

The model achieved high transfer performance on some targets, especially T3
France, but the evidence is target-dependent. T3 must be paired with the Week 8
E5 diagnostic showing lexical/ccTLD dependence. T2_PH should be described as a
hard transfer target that becomes recoverable with small post-hoc few-shot
target supervision. T1 and T2_ON remain one-class boundary cases and should not
be summarized with invalid ROC-AUC.
