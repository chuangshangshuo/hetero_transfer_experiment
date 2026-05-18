# Evidence Boundary Table

This table records the current paper-safe claim boundaries after the Week 7
and Week 8 interpretation revision.

| Claim | Original Supporting Result | Diagnostic Correction | Revised Conclusion | Paper-Safe Wording | Overclaim to Avoid |
| --- | --- | --- | --- | --- | --- |
| model detects illegal vs licensed websites | Week 6 cross-verified pooled AUC 0.9798; Week 8/8.5 licensed-negative results are high | Hard-negative family audit separates this from family-level discrimination | supported | The model reliably separates illegal from licensed baseline sites under the current label policy. | The model understands all illegal-family structure. |
| model generalizes to unseen illegal families | Original Week 8 E3 licensed-negative LOFO had very high AUC | Week 8.5 hard illegal-negative LOFO mean AUC 0.6218, tsars 0.3864 | limited / not strongly supported | Held-out-family positives are separable from licensed negatives, but family-level discrimination among illegal sites is limited. | Strong unseen illegal-family generalization is validated. |
| T3 France proves structural transfer | Week 7 T3 source-only AUC 0.9861 | Corrected E5: graph-only AUC 0.7134, lex-only AUC 0.9530, no-ccTLD drop 0.0815 | not pure structural evidence | T3 France is high-performing but substantially lexical/ccTLD-assisted. | T3 proves structural transfer. |
| LogME predicts transferability | Week 8 included LogME transferability scoring | Week 8 rho 0.2754 and Week 8.5 expanded rho 0.0733 fail the proxy criterion | not reliable | LogME is not reliable as a transferability prescreen in this project. | LogME is validated as a transfer predictor. |
| embedding distance explains transfer | E4 reintroduced control group and measured embedding distances | H_E4b/H_E4c fail; direction-only revisions are post-hoc | insufficient | Control separability is supported, but embedding-distance geometry is not a sufficient explanation. | Embedding distance explains transfer behavior. |
| few-shot improves transfer | Week 9 few-shot raises T2_PH to AUC 0.9250 at 5 shots per class | T3 shows no clear gain because source-only is already saturated | target-dependent support | Few-shot target labels can recover hard transfer for T2_PH, but the effect is target-dependent and post-hoc. | Few-shot universally improves transfer. |

Rows based on Week 8.5, Week 8 patch, and Week 9 are post-hoc diagnostic
evidence. They must not be reported as preregistered Week 8 confirmations.
