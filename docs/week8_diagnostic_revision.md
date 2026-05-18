# Week 8 Diagnostic Revision

Week 8 should be positioned as **Diagnostic Analysis of Transfer Boundaries**,
not as a clean hypothesis-validation chapter. Its core role is to diagnose why
Week 7 transfer scores are high, low, unstable, or target-dependent.

The diagnostic axes are:

- shortcut sensitivity
- hard-negative family failure
- transferability proxy failure
- structure-vs-lexical boundaries
- insufficient embedding-geometry explanation

All new synthesis tables in this revision are post-hoc diagnostic summaries.
They do not overwrite the original `output/week8/` files or the Week 8.5 /
Week 8 patch outputs.

## E1 Edge-Channel Ablation

| Field | Revision |
| --- | --- |
| original hypothesis | Certain graph edge channels should reveal stable structural transfer mechanisms. |
| original result | T3 shows a `uses_cert` edge effect, but T2_PH and T2_ON effects are mixed; H_E1a passes while H_E1b/H_E1c fail under the preregistered Week 8 audit. |
| diagnostic correction | Edge effects are target-specific. They can indicate shortcut or channel sensitivity, but do not establish a general structural mechanism. |
| revised conclusion | Mixed. Edge channels are diagnostically useful but not uniformly beneficial. |
| what can be claimed | Some transfer settings are sensitive to specific infrastructure edges. |
| what cannot be claimed | Edge structure generally explains transfer success across targets. |

## E2 LogME Transferability Proxy

| Field | Revision |
| --- | --- |
| original hypothesis | LogME should predict transferability. |
| original result | Week 8 LogME-AUC Spearman rho is 0.2754 and fails the threshold. |
| diagnostic correction | Week 8.5 expands the pairing with one-class-aware target metrics; LogME rho remains weak at 0.0733 and does not beat distance baselines. |
| revised conclusion | Failed proxy. |
| what can be claimed | LogME is not reliable as a transferability prescreen in this project. |
| what cannot be claimed | LogME validates or predicts target transfer performance. |

## E3 Family LOFO

| Field | Revision |
| --- | --- |
| original hypothesis | Family LOFO validates unseen illegal-family generalization. |
| original result | Licensed-negative LOFO is very high, with mean family AUC around 0.9973. |
| diagnostic correction | Licensed-negative LOFO mostly tests illegal-vs-licensed separability. The hard illegal-negative audit tests held-out family illegal sites against other illegal-family sites. |
| revised conclusion | Limited. Illegal-vs-licensed separability is strong, but hard-negative family recognition is not strongly supported. |
| what can be claimed | The model separates illegal-family positives from licensed negatives under held-out-family partitions. |
| what cannot be claimed | Strong unseen illegal-family generalization among illegal sites. |

The setting comparison is saved at
`output/week8/metrics/E3_lofo_setting_comparison.csv`.

## E4 Control Group and Embedding Geometry

| Field | Revision |
| --- | --- |
| original hypothesis | Embedding distances would explain the illegal/licensed/control relationship. |
| original result | Illegal-vs-control classification is strong, but H_E4b/H_E4c embedding-distance hypotheses fail. |
| diagnostic correction | The classifier can separate the control group, but embedding geometry is an insufficient mechanism explanation. |
| revised conclusion | Limited. |
| what can be claimed | Control-group separability exists under the supervised classifier. |
| what cannot be claimed | The proposed embedding-distance geometry explains transfer behavior. |

## E5 Structure vs Lexical

| Field | Revision |
| --- | --- |
| original hypothesis | T3 transfer would be primarily graph-structural rather than lexical. |
| original result | Original E5 hypotheses fail. |
| diagnostic correction | Corrected E5 shows T3 graph-only AUC is lower than lex-only, lex-only is strong, and removing ccTLD/local-TLD features causes a material drop. |
| revised conclusion | Failed as pure structural evidence. T3 France is substantially lexical/ccTLD-assisted. |
| what can be claimed | T3 is a high-performing but shortcut-sensitive transfer target. |
| what cannot be claimed | T3 France proves structural transfer. |

The corrected summary is saved at
`output/week8/metrics/E5_structure_vs_lexical_corrected_summary.csv`.

## Diagnostic Rollup

The final Week 8 diagnostic rollup is saved at
`output/week8/metrics/week8_diagnostic_rollup.csv`.

| Experiment | Final Status | Paper-Safe Claim |
| --- | --- | --- |
| E1 | mixed | Edge channels are diagnostic but not uniformly explanatory. |
| E2 | failed | LogME is not reliable as a transferability proxy. |
| E3 | limited | Illegal-vs-licensed separation is strong; family-level generalization among illegal sites is limited. |
| E4 | limited | Control separability exists, but embedding geometry is insufficient. |
| E5 | failed | T3 France is substantially lexical/ccTLD-assisted. |

Do not write that all hypotheses were validated.
