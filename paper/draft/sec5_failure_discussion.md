# 5 Failure Discussion: Boundaries as Findings

The final interpretation is intentionally conservative. The model does not comprehensively solve illegal website detection across every target, transfer setting, and family boundary. Instead, the evidence supports a four-part framework:

In the final paper framing, this corresponds to four linked ideas: 结构共性, 地区偏置, 证据层级, and few-shot 本地校准.

1. Structural commonality: illegal and licensed controls are separable in the cross-verified evidence layer.
2. Regional bias: some high transfer scores are assisted by regional lexical and ccTLD cues.
3. Evidence hierarchy: illegal-vs-licensed detection is easier than illegal-family hard-negative discrimination.
4. Few-shot local calibration: weak transfer can be repaired for a hard but recoverable target when a small number of balanced target labels are available.

The strongest negative boundary is the family hard-negative audit. High licensed-negative family LOFO performance does not imply illegal-family attribution. When held-out illegal family sites are compared against other illegal sites, performance is weak and family-dependent.

Model and method boundaries should also remain visible. HeCo integration is useful but not universally superior to the HeteroGNN baseline under all heads and splits. DANN and StruRW are diagnostic transfer methods, not solved adaptation. LogME is not a reliable transferability proxy in this project.

The final paper should present these failures as part of the contribution: the system identifies where structural signal appears, where regional shortcuts intervene, where family attribution fails, and where few-shot target calibration can provide a local repair.
