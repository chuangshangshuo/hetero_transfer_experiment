# §4.3 RQ3 迁移:真实地区环境、IRM/EERM 与方法边界

RQ3 关注跨地区迁移是否稳定。Week7 已经表明迁移表现高度 target-dependent:T2_PH 是 hard but recoverable target，T3 France 是 source-only 已接近饱和但 shortcut-sensitive 的 target，T1_Nordic 与 T2_ON 则属于 one-class 或边界一致性任务，不能纳入标准 ROC-AUC 主表。

Week11 P2 补上 IRM/EERM 方法对比。这里最重要的发现可能不是某个方法显著获胜，而是 IRM 的适用性边界。真实地区 source pool 往往不是每个 environment 都有 illegal 和 licensed 两类样本:有的地区几乎只有 illegal，有的地区几乎只有 licensed。IRM 依赖按环境计算可比 risk gradient，这个假设在小样本跨地区情报数据中经常先天不满足。EERM 通过 embedding clustering 构造 virtual environment，可以绕过真实地区单类的问题，但它本身又引入 K-means 聚类和 K 值敏感性。

当前表包含 125 行，可用于后续论文表格。

因此，RQ3 的安全结论不是“DANN、StruRW、IRM 或 EERM 解决了迁移”，而是“迁移边界需要按目标域和环境可比性来解释”。如果 IRM/EERM 在某个 transfer 上超过 source-only，论文可以把它作为方法补充；如果没有显著超过，论文仍可把它写成小样本异质图迁移的限制证据。这个结论与 Week8 的 LogME 失败相互呼应:迁移可行性不能只由单一 proxy 或单一高 AUC 决定，而必须结合目标标签结构、feature shortcut、family hard-negative 和 few-shot recovery 一起判断。
