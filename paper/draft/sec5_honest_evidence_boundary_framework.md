# §5 Honest Evidence Boundary Framework:失败、边界与可审计证据

本文最终采用“诚实证据边界”作为解释框架，而不是把一组实验拼成模型全面成功的叙事。跨地区非法网络赌博团簇研究面对的是高度不均衡、来源异质、地区标签不完整且存在 shortcut 风险的数据。若只报告最高 AUC，就会把 licensed-vs-illegal 的可分性、family-level generalization、跨地区迁移和 few-shot 校准混在一起。这样的写法会高估模型能力，也会掩盖最有研究价值的失败模式。

第一条边界是证据层级边界。Week6 的 cross-verified 结果支持模型能够识别 officially confirmed illegal 与 licensed controls 的差异，但 Week8.5 的 hard-negative family audit 说明同一个 binary head 不能稳定区分 held-out family 与其他 illegal family。这个差异非常关键:前者是非法性识别，后者是家族归因或 family-level discrimination。两者都可以使用 AUC，但它们回答的问题不同。本文因此把 illegal-vs-licensed 作为主任务支持，把 family-vs-other-illegal 作为高难边界审计，不把二者互相替代。

第二条边界是地区偏置与 shortcut 边界。T3 France 的 source-only AUC 很高，但 corrected E5 显示 lex-only 很强、graph-only 明显较弱、no-ccTLD 会造成性能下降。因此，T3 France 不能被写成 pure structural transfer 的证据。更合适的解释是:目标域的 lexical 和 ccTLD 边界与模型表示发生了配合，形成了 shortcut-assisted transfer。这个发现不是削弱论文，而是让论文更可信，因为它说明研究没有把 shortcut 当作结构泛化。

第三条边界是迁移方法适用性边界。DANN、StruRW、IRM、EERM 这类方法都有理论动机，但真实跨地区数据经常出现 environment 单类、目标域无完整标签、source-target label composition 不对称等情况。IRM 特别依赖每个环境内部的可比 risk，而本研究的地区环境往往不满足这一点。EERM 用 virtual environment 缓解这个问题，但又必须报告 K 值敏感性。本文因此不把方法对比写成“某方法解决迁移”，而是写成“不同迁移方法在小样本异质图上的适用条件不同”。

第四条边界是 few-shot 本地校准边界。Week9 显示 T2_PH 可以通过少量目标标签显著修复，而 T3 France 因为 source-only 已饱和，few-shot 的额外收益很小。Week11 把这个检验扩展到 T1_Nordic 和 T2_ON 后，RQ4 的核心就不再是“few-shot 是否普遍有效”，而是“哪些目标域是 recoverable，哪些目标域只是 saturated、one-class 或 consistency check”。这种写法保留了强结果，也给弱结果留下解释空间。

第五条边界是 post-hoc 与 prerun 的认识论边界。Week8 patch、Week8.5、Week9 和 Week10 中相当一部分分析属于 diagnostic 或 correction layer。它们有价值，但不能伪装成原始预注册假设的胜利。Week11 的三份设计文档把 15 个假设在运行前冻结，并要求每条结果带 is_prerun_frozen、is_posthoc、paper_safe_claim 与 paper_forbidden_claim。这种双标记体系让审稿人能够区分:哪些结论是预先设定的检验，哪些是后验诊断，哪些只是附录性的负结果。

第六条边界是失败假设的贡献边界。传统实验写作常把失败结果藏在附录，但本研究把失败看作研究对象本身。LogME 不能可靠预测迁移，说明 transferability proxy 在该数据规模下不稳；embedding distance 不能稳定解释迁移，说明表示空间几何不能直接等同于跨域可迁移性；family hard-negative 弱，说明非法识别和家族归因之间存在真实任务差异。这些失败共同构成了情报学模型应用的风险地图。

基于 docs/evidence_boundary_table.md，本文最终框架由四个部分组成:结构共性、地区偏置、证据层级和 few-shot 本地校准。结构共性解释为什么 illegal-vs-licensed 可以被学到；地区偏置解释为什么高 AUC 可能来自 ccTLD 或词法边界；证据层级解释为什么 family hard-negative 不能被 licensed-negative LOFO 替代；few-shot 本地校准解释为什么某些 hard target 可以被少量标签修复。四者共同避免了“模型全面成功”的过度叙事，也让研究结论更适合后续复现、扩展和审计。

在写作实践上，诚实证据边界要求每张表都同时给出正向结论和禁止结论。正向结论说明该结果支持什么，例如 T2_PH 的 few-shot recovery 支持“弱迁移目标可由少量本地标签校准”；禁止结论说明该结果不能支持什么，例如同一个结果不能推出“few-shot 对所有地区都有效”。这种双栏写法看似保守，却能显著降低审稿阶段最常见的质疑:高指标是否来自数据泄漏、是否来自词法 shortcut、是否把 one-class target 当成二分类、是否把后验诊断包装成预注册成功。

该框架还要求把样本来源的质量层级写入解释。licensed baseline、officially confirmed single、officially cross-verified、gray candidate 与 control commercial 并不是可互换标签。cross-verified illegal 更适合作为高置信主证据，single-tier illegal 更适合作为扩展支持，gray candidate 只能用于发现与敏感性分析，不能自动升级为 confirmed illegal。Week11 的 P3 在 T1_Nordic 上特别强调 few-shot support 与评估集不重叠，就是因为 cross-verified 目标样本质量高，如果直接抽入训练会造成校准看似成功但评估被污染。

诚实证据边界也改变了失败结果的呈现方式。P1 若失败，并不意味着研究失败，而是说明“在当前数据规模和 family 元数据条件下，简单 family metric head 不能把 binary illegal signal 转换成稳定 family boundary”。P3 若 T1_Nordic 未达 0.50，也不意味着 few-shot 无效，而是说明 one-class illegal target 的本地校准比 T2_PH 这类标准 binary target 更难。P2 若某个方法显著改善，也只能写成目标域上的方法补充；若没有改善，则写成小样本异质图迁移方法的适用性边界。每个失败都对应一个可复查的问题定义，而不是被模糊处理。

最后，该框架把论文从“模型报告”提升为“评估协议报告”。模型只是研究对象之一，真正可复用的是如何在风险任务中组织证据:先固定图版本和 split，保留历史输出；再把原始假设、post-hoc correction、diagnostic patch 与 final freeze 分层；随后对每个 RQ 给出 status、is_posthoc、paper_safe_claim 与 paper_forbidden_claim；最后用 acceptance audit 检查表图是否一致。这样的协议可以迁移到其他非法生态识别任务，包括欺诈域名、钓鱼网站、违规交易网络或灰黑产广告网络。

从实践角度看，这种评估协议也能指导后续数据建设。若目标是提升 RQ1 的 family boundary，就不应只增加 licensed controls，而应补充 family-labeled illegal 样本和跨 family hard negatives；若目标是提升 RQ2 的地区差异解释，就需要更完整的 ccTLD、语言、注册商和基础设施来源标注；若目标是提升 RQ3 的迁移方法比较，就需要在每个真实 environment 内尽量保留双类样本，否则 IRM 类方法只能作为有限适用性分析；若目标是提升 RQ4 的 few-shot 校准，就必须预先区分 binary target、illegal-only target 与 licensed-only target。换言之，失败结果反过来定义了下一轮数据采集和实验设计的优先级。

这也解释了为什么本文坚持保留 legacy、patch、corrected 与 final freeze 多个输出层。删除旧结果会让方法演化不可追踪，而直接覆盖旧表会让 post-hoc correction 看起来像原始实验成功。保留历史层并在最终表中显式标记，能让读者看到研究如何从“高 AUC 是否可靠”逐步推进到“高 AUC 在什么边界内可靠”。这种透明性本身就是高风险情报建模的必要组成部分。

因此，本文的最终贡献不是证明某个异质图模型在所有场景下都强，而是提出并实证展示一个更稳健的评估姿态:在小样本、跨地区、非法生态识别任务中，模型能力必须按任务边界、目标域、特征来源和标签可得性分层报告。这样的结论更克制，但也更有操作价值。
