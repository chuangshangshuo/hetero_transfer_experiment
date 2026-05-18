# Week 8 Diagnostic Analysis of Transfer Boundaries

Week 8 is revised as a diagnostic analysis of transfer boundaries, not a clean
hypothesis-validation section. Its purpose is to identify shortcut sensitivity,
hard-negative failure, proxy failure, and structure-vs-lexical boundaries in
the Week 7 transfer results.

## E3 Family LOFO Boundary

The original licensed-negative LOFO scores are high, but they can only be
interpreted as illegal-vs-licensed separability under held-out positive-family
partitions. They do not by themselves test whether the model distinguishes one
illegal family from other illegal families.

The hard illegal-negative audit is the relevant diagnostic for family-level
generalization among illegal sites. Under that setting, mean hard-negative AUC
is 0.6218 and tsars AUC is 0.3864. These results do not support strong
family-level generalization. The safe claim is that illegal-vs-licensed
separability is strong, while unseen illegal-family discrimination is limited.

## E2 Transferability Proxy Boundary

LogME should be described as limited or failed in this project. The original
Week 8 LogME correlation is below the planned threshold, and the expanded
Week 8.5 one-class-aware analysis does not rescue it. LogME should not be used
as a validated transferability prescreen.

## E4 Embedding Geometry Boundary

E4 supports supervised separability between illegal sites and commercial
controls, but the embedding-distance hypotheses do not hold. Embedding geometry
is therefore an insufficient explanation of transfer behavior.

## E5 Structure-vs-Lexical Boundary

E5 is the main diagnostic for interpreting the high T3 France result. Corrected
E5 shows that T3 France is substantially lexical/ccTLD-assisted: graph-only AUC
is lower than lex-only AUC, lex-only remains strong, and removing ccTLD/local
TLD features causes a material drop. T3 should not be written as pure structural
transfer evidence.

## Rollup

The revised diagnostic rollup is saved in
`output/week8/metrics/week8_diagnostic_rollup.csv`.

| Experiment | Final Status | Safe Claim |
| --- | --- | --- |
| E1 | mixed | Edge channels are diagnostic but not uniformly explanatory. |
| E2 | failed | LogME is not reliable as a transferability proxy. |
| E3 | limited | Illegal-vs-licensed separability is strong; family-level discrimination is limited. |
| E4 | limited | Control-group separability exists, but embedding geometry is insufficient. |
| E5 | failed | T3 France is substantially lexical/ccTLD-assisted. |

The paper should not state that all hypotheses were validated.
