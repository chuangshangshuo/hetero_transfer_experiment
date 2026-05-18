# §4.1 RQ1 共性:从 illegal-vs-licensed 到 family boundary

RQ1 的核心问题是模型是否能够区分非法网站与 licensed controls。Week6 的 cross-verified 评估已经给出强证据:在高质量 officially cross-verified illegal 层上，模型能够稳定地区分 illegal 与 licensed baseline。这说明异质图和网站层特征确实捕捉到了非法网站生态的一部分共性。

但是，Week8.5 和 Week9 的 hard-negative family audit 也暴露了更严格的问题:当负例不再是 licensed controls，而是其他 illegal family 时，原来的 binary head 会把几乎所有 illegal 都推到接近 1 的非法概率，导致 family-vs-other-illegal 的排序能力不足。这不是非法识别能力的完全失败，而是训练目标与评估目标错位。binary classifier 被训练来回答“是否非法”，并没有被训练来回答“属于哪个非法家族”。

Week11 P1 因此新增 family-aware metric learning head。它冻结既有 HeCo encoder，只在表示上增加一个 family projection head，并用 supervised contrastive loss 拉近同 family 样本、推远不同 family 样本，同时保留 binary head 维护 illegal-vs-licensed 能力。该 patch 的证据解释有明确边界:若 hard-negative mean AUC 提升，它只能说明 family-aware calibration 有助于 family boundary；若提升有限，则说明在当前 7 个丹麦 family、小样本规模与弱 family 元数据条件下，家族级归因仍然不稳。

当前表包含 35 行，可用于后续论文表格。

论文中应把 RQ1 写成分层结论:第一层，模型对 officially confirmed illegal 与 licensed controls 的区分有较强支持；第二层，family-level hard-negative 区分在原 binary head 下较弱；第三层，Week11 P1 作为预声明改进检验 family metric learning 是否能够补这个洞。无论 P1 通过还是失败，都不能把结果写成“模型可以稳定完成非法家族归因”。更安全的写法是:模型识别 illegal-vs-licensed 的能力强于识别 family-vs-other-illegal 的能力，family-aware head 是对这个边界的针对性校准。
