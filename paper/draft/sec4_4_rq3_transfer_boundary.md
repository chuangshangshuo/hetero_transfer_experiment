# 4.4 RQ3: Transfer Boundary and Transferability Proxies

Week7 and Week8 together define RQ3 as target-dependent transfer boundary analysis. Source-only transfer is weak for T2_PH, one-class targets such as T1_Nordic and T2_ON require boundary metrics rather than standard ROC-AUC, and T3 France is high-performing but shortcut-sensitive. Adaptation methods such as DANN and StruRW do not uniformly improve over source-only baselines.

The transferability diagnostics also remain negative. LogME does not provide a reliable prescreen under the current heterogeneous graph and label-availability setting, and embedding-distance explanations are insufficient as a global mechanism.

The paper-safe claim is: transfer is bounded by target label structure, regional feature shortcuts, and evidence composition. RQ3 should be written as a boundary map, not as a solved transfer-learning result.

Forbidden interpretation: do not claim DANN or StruRW solves transfer, do not force ROC-AUC on one-class targets, and do not describe LogME as reliable.
