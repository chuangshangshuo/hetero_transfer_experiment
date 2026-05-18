# §3 方法深化:异质图、对比预训练与边界化评估

本研究把非法网站识别问题建模为一个以 Website 为核心的异质图学习任务。图中包含 Website、IP、Certificate、NameServer、Registrar 与 ExternalReference 六类节点，边关系覆盖 hosted_on、uses_cert、uses_ns、registered_via 与 referenced_by 五种语义连接，并保留反向边以支持异质消息传递。这个设计的目的不是把所有线索压成一个黑箱特征向量，而是保留网站基础设施、注册链条、证书复用、名称服务器复用与外部引用之间的证据层级。Graph v2 同时明确移除了 redirects_to 的主动图边，只把跳转关系作为审计材料保留，避免把容易受采集环境影响的短期跳转当成稳定结构。

表示学习部分采用 HeCo 风格的双视图对比预训练。schema view 汇总异质边上的消息，metapath view 则通过 Website-IP-Website、Website-Certificate-Website、Website-NameServer-Website 与 Website-ExternalReference-Website 等共现路径形成面向网站的邻域视图。两个视图分别投影后进行对比学习，目标是让同一网站在不同结构视图中的表示靠近，同时让缺少共现证据的样本保持区分。这个阶段不使用非法/合法标签构造正例，因此它更接近结构自监督预训练，而不是监督标签泄漏。

监督阶段采用分层证据口径。Week6 的 pooled primary 任务评估 licensed baseline 与 officially confirmed illegal 的可分性；Week7 把模型转到不同目标地区，观察 source-only、DANN、StruRW 与组合方法的迁移边界；Week8/8.5/patch 把高分来源拆成 family hard-negative、feature bucket 与 shortcut 诊断；Week9 再考察少量目标标签能否校准 hard but recoverable target。Week11 在这个框架上新增三个改进层:family-aware metric head、IRM/EERM 方法对比、全 transfer few-shot 补齐，并以 P4 的配对显著性检验把之前的 mean/std 结果升级为带不确定性的证据表。

因此，本文的方法贡献不是单一模型结构的胜利，而是一个可审计的证据管线:先用异质图保留多类基础设施信号，再用对比预训练获得共享表示，随后用按地区、按 family、按 feature bucket、按 few-shot 条件的诊断把模型能力边界写清楚。这个管线特别适合情报学小样本场景，因为现实数据往往同时存在标签稀疏、地区单类、family 元数据不完整和词法 shortcut 等问题。

这种方法章节也定义了复现实验的最低标准:任何后续改动都必须说明使用哪个 graph version、哪个 split family、哪个 evidence layer、哪个 output namespace，以及是否属于 post-hoc correction。没有这些元数据，即使模型分数更高，也不能直接并入主结论。

因此，方法部分不仅描述模型怎么训练，也描述证据如何被允许进入论文主线。

这能防止后续写作把探索性修补误写成原始假设验证。

也能让读者按同一套口径复核每一张表、每一个图和每一句结论。

这正是本文区别于单纯模型 benchmark 的地方。

因此它也是最终投稿版本的组织骨架。

结论必须可复查。

也必须可靠。
