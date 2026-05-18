# Week 8.5 Patch Audit

Week 8.5 is a post-hoc patch audit. It does not overwrite the prerun Week 8 hypothesis table.
Patch outcomes: pass=7, fail=5.

## Acceptance Table

| patch | hypothesis | observed | threshold | status |
| --- | --- | ---: | --- | --- |
| P1 | E3_balanced_licensed_negative | 1.0000 | min family AUC >= 0.85 | pass |
| P1 | E3_hard_illegal_family_detection | 0.6218 | mean hard-negative family AUC >= 0.80 | fail |
| P1 | E3_hard_tsars_family_detection | 0.3864 | tsars hard-negative AUC >= 0.80 | fail |
| P2 | E2_expanded_LogME_predicts_quasi_metric | 0.0733 | Spearman rho >= 0.60 | fail |
| P2 | E2_expanded_LogME_beats_distance_baselines | 0.0733 | |rho_LogME| > |rho_H-div| and |rho_W| | fail |
| P3_posthoc | H_E4b_prime_direction_only | 4.0000 | >=4/5 seeds | pass |
| P3_posthoc | H_E4c_prime_direction_only | 0.0000 | >=4/5 seeds | fail |
| P4_posthoc | H_E5a_prime_cctld_key | 0.0815 | T3 full-minus-no-cctld >= 0.05 | pass |
| P4_posthoc | H_E5b_prime_lex_only_working | 0.9530 | T3 lex-only >= 0.85 | pass |
| P4_posthoc | H_E5c_prime_transfer_specific_dominance | 3.0000 | 3/3 specified directional checks | pass |
| P4_posthoc | H_E1b_prime_T2PH_negative_transfer_edge | -0.0833 | at least one edge has effect<0 and p<0.05 | pass |
| P4_posthoc | H_E1c_prime_T2ON_any_significant_edge | 2.0000 | at least one edge p<0.05 | pass |

## Interpretation

- The original Week 8 E3 LOFO is downgraded because it mainly tested illegal-vs-licensed discrimination.
- The corrected balanced licensed-negative LOFO remains strong, but the hard illegal-negative audit shows weak family-level discrimination for several families.
- P3/P4 are explicitly post-hoc direction changes and should be described as revised interpretation rather than prerun confirmations.