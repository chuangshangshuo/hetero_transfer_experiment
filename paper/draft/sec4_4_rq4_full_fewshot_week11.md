# §4.4 RQ4 弱监督与本地校准:从两个 binary target 扩展到四类 transfer

Week9 的 few-shot 结果已经说明，少量目标域标签可以显著修复 T2_PH 这样的 hard but recoverable target，但对 T3 France 这种 source-only 已经接近饱和且存在 lexical/ccTLD shortcut 风险的 target 几乎没有额外收益。这个结论必须以 delta versus source-only 为核心，而不能只看 few-shot 后的绝对 AUC。

Week11 P3 把 RQ4 扩展到 T1_Nordic 与 T2_ON 两个 one-class/boundary target。对 T1_Nordic，目标域测试集只有 illegal，因此评估指标是 illegal recall at Youden，而不是 ROC-AUC。对 T2_ON，目标域是 licensed-only，核心指标是 mean predicted illegal probability 是否保持低位。这个设计承认了真实迁移任务并不总是标准二分类:有些目标域只能检验模型是否把全非法目标识别出来，有些目标域只能检验模型是否不会把全 licensed 目标误报为非法。

当前表包含 100 行，可用于后续论文表格。

最终 RQ4 应写成 target-dependent local calibration。若 T1_Nordic 的 5-shot illegal recall 明显提升，可以说明 one-class hard target 也有本地校准空间；若 T2_ON 的 mean_pred_illegal 维持低位，则说明 licensed-only target 的边界没有被 few-shot 支持破坏；若任何一项失败，也应作为弱监督边界公开报告。few-shot 的价值不在于普遍提升，而在于揭示哪些 target 能被少量本地标签修复，哪些 target 已饱和或不适合标准二分类指标。
