# 学术审查委员会 — 数据、引用与可复现性审查报告

- **审查对象**：`manuscript_v9.md`（《图神经网络在跨地区非法网络赌博识别中的应用边界》，信息安全研究）
- **发布包**：`D:\hetero_transfer_experiment_release\hetero_transfer_experiment_release`（只读）
- **公开仓库**：https://github.com/chuangshangshuo/hetero_transfer_experiment @ c29fb11
- **审查范围**：参考文献真实性、正文-文献对应、可复现性命令链与审计文件、数据可用性声明
- **明确排除**：DOI、投稿编号、版面费、长期存档等出版社定调内容
- **审查状态**：**已完成**（29 条文献逐条核验 + 发布包实测 + 公开仓库远程核对）
- **审查结论**：**Major Revision** — 14 项 Blocking、3 项 Major、11 项 Minor
- **已由他人核实、本报告不重复**：613/623 样本口径与 `data/node_stats.csv` 一致；发布包内 checkpoint 文件数为 0（与 §1.3 矛盾，本报告 D1 从"§1.3 vs §6.4 内部矛盾"角度另行记录）

---

## A. 参考文献逐条真实性核验

文献表未加编号，按出现顺序推定为 [1]–[29]（共 29 条）。以下逐条记录。

### [2] Min 等 — Journal of Gambling Studies 2024 ❌ Needs fix

- **稿件著录**：`Min M, Lee J J, Park M. Illegal online gambling site detection using multiple resource-oriented machine learning[J]. Journal of Gambling Studies, 2024, 40(4): 2237-2255`
- **核实结果**（Springer/PubMed）：题名、刊名、年、卷期、页码**全部正确**（J Gambl Stud 40, 2237–2255, 2024）。
- **问题（🔴 Critical）**：**作者著录错误**。真实作者为 **Min M, Lee D A**（仅 2 位），稿件写成 "Min M, **Lee J J, Park M**"——第二作者名缩写错误，且**凭空多出第三作者 Park M**，并因此误加 "et al." 语义。
- **问题（🟡 Warning）**：正文 §1.1、§2.1 称 "Min等(2024)…提出AIOG框架"。原文中 AIOG = *absolute illegal online gambling*，是该文**定义的一个目标类别概念**，不是所提出的框架/模型名称。称"AIOG 框架"属对原文的实质性改写。
- **正确项**：正文所述 "11,172 个站点"、"综合 URL、WHOIS、INDEX、landing page"、"文本+图像集成分类器"与原文摘要一致 ✅；"来源以韩国为主"与作者机构一致（可信）。
- **定级**：**Major / Blocking**（作者著录错误属可直接核验的硬错误）
- **定位**：参考文献第 2 条；正文第 24 行、第 50 行、第 51 行
- **建议**：改为 `Min M, Lee D A. Illegal online gambling site detection using multiple resource-oriented machine learning[J]. Journal of Gambling Studies, 2024, 40(4): 2237-2255`；正文改为"提出面向绝对非法网络赌博(AIOG)的识别模型"或"提出 AIOG 概念与相应识别模型"，删去"框架"提法。

### [3] Prieto 等 — IEEE Access 2021 ✅ Verified

- **稿件著录**：`Prieto J C, Fernandez-Isabel A, Martin de Diego I, et al. Knowledge-based approach to detect potentially risky websites[J]. IEEE Access, 2021, 9: 11633-11643`
- **核实结果**（IEEE Xplore doc 9321392）：作者 J. C. Prieto, A. Fernandez-Isabel, I. Martín de Diego, F. Ortega, J. M. Moguerza；题名、刊名、卷 9、页 11633–11643、年 2021 **完全一致** ✅
- **框架名核查**：原文确实提出 **DOCRIW**（DOmains Classifier based on RIsky Websites），由"已建知识库 + 二元分类器"两部分构成——**与正文 §1.1 表述完全吻合** ✅
- **定级**：无问题
- **备注（🟢 Info）**：正文写"Prieto 等提出 DOCRIW 框架"但未标注年份，而同段落的杨哲、Min 均标了年份，体例不统一（见 B 节）。

### [16] HGCML — SDM 2023 ❌ Needs fix（作者著录整体错位）

- **稿件著录**：`Wang H, Wang X, Han H, et al. Heterogeneous graph contrastive multi-view learning[C]//Proceedings of the SIAM International Conference on Data Mining (SDM). 2023`
- **核实结果**（SIAM epubs 10.1137/1.9781611977653.ch16；arXiv:2210.00248）：真实作者为 **Zehong Wang, Qi Li, Donghua Yu, Xiaolong Han, Xiao-Zhi Gao, Shigen Shen**；SDM 2023 **pp. 136–144**。
- **问题（🔴 Critical）**：稿件著录的 "Wang H, Wang X, Han H" 与真实作者**完全不符**。更严重的是，"Wang X, Han H" 恰是文献 [4] HeCo 的第二、第三作者（Xiao Wang, Hui Han）——**疑似把 HeCo 的作者列表误植到 HGCML 上**。这是典型的引用装配错误，而非笔误。
- **问题（🟡 Warning）**：缺页码（应补 136-144）。
- **定级**：**Major / Blocking**
- **定位**：参考文献第 16 条；正文第 56 行 "Wang等(2023)的HGCML"
- **建议**：改为 `Wang Z, Li Q, Yu D, et al. Heterogeneous graph contrastive multi-view learning[C]//Proceedings of the 2023 SIAM International Conference on Data Mining (SDM). 2023: 136-144`。注意正文 "Wang等(2023)" 仍可成立（第一作者姓 Wang），但需确认指向 Zehong Wang 而非 Xiao Wang——同段落已有 "Wang等(2021)" 指 Xiao Wang（HeCo）、"Wang等(2019)" 指 Xiao Wang（HAN），三个 "Wang等" 并存且分属不同人，读者极易混淆。

### [17] HeCo++ / arXiv:2304.12228 ⚠️ Check suggested（正文年份与著录冲突）

- **稿件著录**：`Liu N, Wang X, Han H, et al. Hierarchical contrastive learning enhanced heterogeneous graph neural network[J]. arXiv preprint arXiv:2304.12228, 2023`
- **核实结果**（arXiv 官方页）：作者 **Nian Liu, Xiao Wang, Hui Han, Chuan Shi** ✅；题名逐字一致 ✅；v1 = 2023-04-24，v2 = 2024-03-05；页面注明 "has been accepted by TKDE as a regular paper"。
- **题名是否含 HeCo++**：**不含**。HeCo++ 是文内提出的模型名（"a modified model called HeCo++ … conducts hierarchical contrastive learning, including cross-view and intra-view contrasts"）。正文用 "HeCo++" 指代该模型**符合原文**，且正文对其"引入层次化对比学习、把 view 内部对比和跨 view 对比结合"的描述**与原文摘要一致** ✅。
- **问题（🟡 Warning）**：正文第 56 行写 **"Liu等(2024)的HeCo++"**，而著录年为 **2023**（arXiv v1）。二者需统一。
- **定级**：**Minor / 非 Blocking**（但必须改）
- **定位**：正文第 56 行；参考文献第 17 条
- **建议**：二选一——(a) 正文改为 "Liu等(2023)"；(b) 若欲用 2024，则应改引 TKDE 正式版（arXiv v2 = 2024-03），并补 TKDE 卷期页。推荐 (b)，因该文已被 TKDE 正式接收，引 arXiv 预印本属次优选择。

### [20] CDAN — NeurIPS 2018 ✅ Verified（但正文存在出处缺失，见 B 节）

- **稿件著录**：`Long M, Cao Z, Wang J, et al. Conditional adversarial domain adaptation[C]//NeurIPS. 2018: 1647-1657`
- **核实结果**：Long M, Cao Z, Wang J, Jordan M I，NIPS'18 Proceedings（第 32 届），pp. **1647–1657** ✅ 作者、题名、年、页码全部一致。
- **定级**：著录本身无问题。

### [5] DANN / Ganin 等 — JMLR 2016 ❌ Needs fix（页码为不可能区间）

- **稿件著录**：`Ganin Y, Ustinova E, Ajakan H, et al. Domain-adversarial training of neural networks[J]. Journal of Machine Learning Research, 2016, 17(1): 2096-2030`
- **核实结果**（dblp / jmlr.org/papers/v17/15-239.html）：JMLR **vol. 17, article no. 59, pp. 59:1–59:35**（即正文 1–35 页）；作者 Ganin Y, Ustinova E, Ajakan H, Germain P, Larochelle H, Laviolette F, Marchand M, Lempitsky V ✅（前三位一致）。
- **问题（🔴 Critical）**：页码 **"2096-2030" 的终页小于起页，是物理上不可能的区间**。该串源自 Google Scholar 长期讹传的一条 BibTeX 记录，被大量论文原样复制；审稿人一眼即可识破，属"未回原文核对、批量导入文献管理器"的直接证据。期号 "(1)" 亦错（JMLR 惯例为卷+文章号，此文为 17(59)）。
- **定级**：**Major / Blocking**
- **定位**：参考文献第 5 条
- **建议**：改为 `Ganin Y, Ustinova E, Ajakan H, et al. Domain-adversarial training of neural networks[J]. Journal of Machine Learning Research, 2016, 17(59): 1-35`
- **附带（🟢 Info）**：正文第 61 行称 "Ganin等(2016)在JMLR发表的DANN" 与著录一致；但同句 "DANN提出后的8年间" 隐含以 2016 为起点，而 DANN 首次提出实为 Ganin & Lempitsky, ICML 2015（arXiv 2014），且其后所列 CDAN(2018)/ADDA(2017)/JAN(2017) 均在 2 年内出现，"8年间"表述与列举实例不匹配，建议改为"其后数年间"。

### [8] 傅建明等 / HADMW — 软件学报 2021 ❌ Needs fix（作者完全错误）

- **稿件著录**：`傅建明, 黎琳, 郑锐, 等. 融合多种特征的恶意URL检测方法[J]. 软件学报, 2021, 32(9): 2916-2934`
- **核实结果**（jos.org.cn 官方页 5983）：题名 `融合多种特征的恶意URL检测方法` ✅；刊名 `软件学报` ✅；2021 ✅；**32(9): 2916-2934** ✅；方法名 **HADMW** ✅；精确率 96.2%、召回率 94.6% ✅。
- **问题（🔴 Critical）**：**真实作者为「吴森焱, 罗熹, 王伟平, 覃岩」**（中南大学），稿件著录的 **傅建明、黎琳、郑锐** 与该文无任何关系。除作者外的所有字段（含罕见的三方面 25 特征、96.2%/94.6% 两个数字）都精确正确，说明是**把正确的文献元数据配了一组错误的作者名**——这是最危险的一类引用错误，因为其余字段的高度吻合会掩盖问题。
- **定级**：**Major / Blocking**（涉及对他人成果的错误署名）
- **定位**：参考文献第 8 条；正文第 49 行 "傅建明等将页面内容…提出HADMW方法"、第 51 行 "傅建明等(2021)的数据集主要是中国国内黑名单"
- **建议**：著录改为 `吴森焱, 罗熹, 王伟平, 等. 融合多种特征的恶意URL检测方法[J]. 软件学报, 2021, 32(9): 2916-2934`；正文两处 "傅建明等" 一并改为 "吴森焱等"。

### [7] 沙泓州等 — 计算机系统应用 2018 ❌ Needs fix（作者完全错误 + 正文数字错误 + 方法误述）

- **稿件著录**：`沙泓州, 刘庆云, 柳厅文, 等. 基于深度学习的恶意URL识别[J]. 计算机系统应用, 2018, 27(6): 27-33`
- **核实结果**（c-s-a.org.cn 官方页 6370）：题名、刊名、2018、**27(6): 27-33** 全部正确 ✅；**真实作者为「陈康, 付华峥, 向勇」**（中国电信股份有限公司广东研究院）。
- **问题 1（🔴 Critical）**：作者著录完全错误。沙泓州、刘庆云、柳厅文（中科院信工所）确有相关工作，但那是《恶意网页识别研究综述》，计算机学报, 2016, 39(3): 529-542，与本条为两篇不同文献。**这是本文第二处"正确元数据 + 错误作者"的组合**（另一处为 [8]），指向系统性的文献装配问题而非偶发笔误。
- **问题 2（🔴 Critical）**：正文第 49 行称 "精度0.973、F1 0.918"。原文实测为 **准确率 0.962、召回率 0.879、F1 0.918**。F1 正确，但**精度 0.973 在原文中不存在**（应为 0.962）。
- **问题 3（🟡 Warning）**：正文称该文"基于**词法特征**训练**2层神经网络**"。原文实为：字符嵌入（用 2 层网络生成字符分布式表示）+ **4 路并行卷积（滤波器高度 2/3/4/5，各 256 个）的 CNN 分类器**。即该文的卖点恰恰是**免手工特征的端到端深度学习**，与稿件"手工设计的丰富特征弥补样本规模不足"的归类论断相反。这一误述削弱了 §2.1 的文献脉络叙事。
- **定级**：**Major / Blocking**
- **定位**：参考文献第 7 条；正文第 49 行、第 51 行（"沙泓州等(2018)…数据集主要是中国国内黑名单"，同样错误署名）
- **建议**：著录改为 `陈康, 付华峥, 向勇. 基于深度学习的恶意URL识别[J]. 计算机系统应用, 2018, 27(6): 27-33`；正文两处作者名改为"陈康等"；"精度0.973"改为"准确率0.962"；方法描述改为"字符嵌入+CNN 端到端识别"，并把它移出"特征工程精细化"这一归类（或改换一篇真正做词法特征工程的文献支撑该论点）。

### [9] Chen 等 / MEDAL — ❌ Needs fix（年份、作者、载体全错；框架名 MEDAL 与 tri-training 在原文中不存在）

- **稿件著录**：`Chen X, Zheng Y, et al. Automatic detection of pornographic and gambling websites based on visual and textual content using a decision mechanism[EB/OL]. 2022`
- **核实结果**（Sensors 官方 doi:10.3390/s20143989；PMC7411926）：**Chen Y, Zheng R, Zhou A, Liao S, Liu L. Sensors, 2020, 20(14): 3989**（四川大学），2020 年 7 月发表。
- **问题 1（🔴 Critical）**：**年份错误** —— 2020 误作 2022（正文第 50、51 行两处均写 "Chen等(2022)"）。
- **问题 2（🔴 Critical）**：**作者名缩写错误** —— 第一作者 Yang Chen 应为 **Chen Y**（稿件作 Chen X）；第二作者 Rongfeng Zheng 应为 **Zheng R**（稿件作 Zheng Y）。
- **问题 3（🔴 Critical）**：**文献类型错误** —— 该文是 MDPI *Sensors* 上的正式同行评议期刊论文，稿件却标为 `[EB/OL]` 且**完全省略刊名、卷、期、文章号**，使读者无法定位。
- **问题 4（🔴 Critical，最严重）**：正文第 50 行称 "**Chen等(2022)提出MEDAL 框架，基于tri-training联合图像、文本、HTML三种模态做半监督学习**"。核对原文：该文题名与方法为**视觉+文本двух模态 + 决策机制(decision mechanism)**，**原文中不存在 "MEDAL" 这一框架名，也不存在 tri-training 与半监督学习设定，更没有 HTML 第三模态**。稿件对该文的方法学描述**与被引文献无对应关系**。
- **定级**：**Major / Blocking**（属"引文与论断不对应"，是审稿中最重的引用问题）
- **定位**：参考文献第 9 条；正文第 50 行、第 51 行
- **建议**：若作者本意是引 *Sensors* 2020 那篇，则著录改为 `Chen Y, Zheng R, Zhou A, et al. Automatic detection of pornographic and gambling websites based on visual and textual content using a decision mechanism[J]. Sensors, 2020, 20(14): 3989`，正文相应改为"Chen等(2020)基于视觉与文本内容并引入决策机制进行赌博/色情站点识别"，**删除 MEDAL、tri-training、三模态、半监督等原文不支持的表述**；若作者本意确是另一篇名为 MEDAL 的工作，则必须补上该文的完整著录，并把当前这条 Sensors 文献单列。二者不可混为一条。

### ⚠️ 关键交叉发现：[9] 是两篇不同文献被合并成的一条"混合引用"

对 MEDAL 的独立追查结果如下：

- **MEDAL 真实出处**：`Li W, Zhang M, Wang C, et al. MEDAL: A multimodality-based effective data augmentation framework for illegal website identification[J]. Electronics, 2024, 13(11): 2199`（DOI 10.3390/electronics13112199；作者 Li Wen, Min Zhang, Chenyang Wang, Bingyang Guo, Huimin Ma, Pengfei Xue, Wanmeng Ding, Jinghua Zheng）。
- 该文**确实**基于 **tri-training**、**确实**联合 **image / text / HTML 三模态**、**确实**是半监督学习并针对"赌博站伪装成游戏站"——**与稿件正文第 50 行的描述逐条吻合** ✅
- 但**第一作者是 Li W（李文），不是 Chen；年份是 2024，不是 2022**。

**结论**：稿件文献 [9] 把 *Sensors* 2020（Chen Y 等，视觉+文本+决策机制）的**题名**，与 *Electronics* 2024（Li W 等，MEDAL/tri-training/三模态）的**内容论断**拼成了一条参考文献。这不是笔误，而是**两篇文献被折叠为一条**。

- **定级**：**Major / Blocking**
- **建议**：拆为两条。若 §2.1 只需支撑 MEDAL 论断，则直接替换为 `Li W, Zhang M, Wang C, et al. MEDAL: A multimodality-based effective data augmentation framework for illegal website identification[J]. Electronics, 2024, 13(11): 2199`，正文改为 "Li等(2024)提出MEDAL框架…"；同时 §2.1 第 51 行 "Chen等(2022)的实验同样聚焦单一辖区" 的论断需要重新核对其实际针对的是哪一篇。

### [10] URLBERT / arXiv:2402.11495 ❌ Needs fix（作者为虚构团体名）

- **稿件著录**：`URLBERT Team. Continuous multi-task pre-training for malicious URL detection and webpage classification[EB/OL]. arXiv preprint arXiv:2402.11495, 2024`
- **核实结果**（arXiv 官方 abs 页）：题名 `Continuous Multi-Task Pre-training for Malicious URL Detection and Webpage Classification` ✅ 与当前版本题名逐字一致；v1 = 2024-02-18，v2 = 2025-05-24；**作者为 Yujie Li, Yiwei Liu, Peiyue Li, Yifan Jia, Yanbin Wang**（v1 题名为 *URLBERT: A Contrastive and Adversarial Pre-trained Model for URL Classification*，作者 Yujie Li, Yanbin Wang, Haitao Xu, Zhenhao Guo, Zheng Cao, Lun Zhang）。
- **问题 1（🔴 Critical）**：**"URLBERT Team" 不是该文的作者**。arXiv 从未以团体名署名该文。以虚构团体名充当作者，属编造著录信息。
- **问题 2（🟡 Warning）**：正文第 50 行称 "用**30亿条**无标签URL训练通用编码器"。arXiv 摘要仅称 "**billions of** unlabeled URLs"，**未给出 30 亿这一具体数字**。该数字在可查来源中无法证实，建议改为"数十亿量级无标签 URL"或补出原文页码依据。
- **问题 3（🟢 Info）**：题名在 v1/v2 之间发生过变更，若沿用 "URLBERT" 这一称呼指代该工作，宜在正文注明"该工作早期版本题为 URLBERT"。
- **定级**：**Major / Blocking**（问题 1）
- **定位**：参考文献第 10 条；正文第 50 行
- **建议**：改为 `Li Y, Liu Y, Li P, et al. Continuous multi-task pre-training for malicious URL detection and webpage classification[EB/OL]. arXiv preprint arXiv:2402.11495, 2024`

### [19] 李晨晨等 / HNN4RP — 电子与信息学报 2025 ⚠️ Check suggested（著录不全）

- **稿件著录**：`李晨晨, 金海, 吴敏睿, 等. HNN4RP: 基于异构图神经网络的以太坊地毯拉动骗局检测[J]. 电子与信息学报, 2025`
- **核实结果**（jeit.ac.cn，DOI 10.11999/JEIT250160）：作者 **李晨晨, 金海, 吴敏睿, 肖江** ✅（前三位与稿件一致，"等"对应肖江，合规）；题名逐字一致 ✅；刊名、2025 ✅；**卷期页为 47(10): 3395-3409**。
- **问题（🟡 Warning）**：**缺卷、期、页码**。该文已正式见刊（2025 年第 10 期），无理由只写年份。
- **定级**：**Minor / 非 Blocking**
- **建议**：补全为 `…[J]. 电子与信息学报, 2025, 47(10): 3395-3409`

### [1] 杨哲, 陈应虎 — 信息安全研究 2023 ⚠️ Check suggested

- **稿件著录**：`杨哲,陈应虎.赌博网站自动识别技术研究[J].信息安全研究,2023,9(5):440-445.`
- **核实结果**（维普 id 7109433911）：题名 `赌博网站自动识别技术研究`、刊名 `信息安全研究`、**2023 年第 5 期、440-445 页** 全部一致 ✅；内容（云平台 AS 信息取 IP 段 → 反向 DNS → 分布式爬虫取截图 → **dHash 清洗正样本** → **CNN 二分类**）与正文第 24 行描述 ✅ 完全吻合。
- **问题 1（🟡 Warning）**：正文第 24 行写 **"杨哲等(2024)"**，著录为 **2023**。**年份冲突**（本文四处正文-著录年份冲突之一）。
- **问题 2（🟢 Info）**：该文仅 2 位作者，中文规范下宜写"杨哲和陈应虎(2023)"，用"等"不妥。
- **定级**：**Minor / 非 Blocking**
- **定位**：正文第 24 行；参考文献第 1 条
- **建议**：正文改为 "杨哲和陈应虎(2023)"。

### [6] 专利 CN104735074A ⚠️ Check suggested（缺责任者）

- **稿件著录**：`一种恶意URL检测方法及其实现系统: CN104735074A[P]. 2015`
- **核实结果**（Google Patents）：专利号、题名一致 ✅；申请日 2015-03-31，公开日 **2015-06-24**；**申请人为江苏通付盾信息科技有限公司，发明人为汪德嘉、叶芸、胡振中、葛彦霆、刘伟**。
- **内容核查**：正文第 48 行称其架构为 "URL知识库+顶级域名匹配+预定义的'短URL服务商'和'三层路径'等启发式规则" —— **与权利要求逐项吻合** ✅（知识库匹配含合法站点顶级域名；未命中时按"短URL服务商"清单与"仅含字母或数字的三层路径"启发式判定）。这是全文对被引文献内容描述最准确的一条。
- **问题（🟡 Warning）**：**完全缺失主要责任者（申请人/发明人）**，不符合 GB/T 7714 专利著录格式；公告日期只写年份。
- **定级**：**Minor / 非 Blocking**
- **建议**：改为 `汪德嘉, 叶芸, 胡振中, 等. 一种恶意URL检测方法及其实现系统: CN104735074A[P]. 2015-06-24`

### [4][11][12][13][14][15][18][21][22][23][24][25][26][27][28][29] 批量核验

经 dblp / 官方 proceedings / arXiv 逐条比对，以下条目**作者、题名、会议/刊名、年、页码均一致**，无需修改：

| # | 条目 | 核验结论 |
|---|------|---------|
| [4] | HeCo，Wang X, Liu N, Han H, Shi C，KDD 2021: 1726-1736 | ✅ Verified（dblp 逐字一致） |
| [11] | DeepWalk，Perozzi B, Al-Rfou R, Skiena S，KDD 2014: 701-710 | ✅ Verified（条目本身；但正文用法有问题，见 B 节） |
| [12] | HAN，Wang X, Ji H, Shi C 等，WWW 2019: 2022-2032 | ✅ Verified（dblp） |
| [13] | MAGNN，Fu X, Zhang J, Meng Z, King I，WWW 2020: 2331-2341 | ✅ Verified（dblp） |
| [14] | R-GCN，Schlichtkrull M, Kipf T N, Bloem P 等，ESWC 2018: 593-607 | ✅ Verified（dblp） |
| [15] | HGT，Hu Z, Dong Y, Wang K, Sun Y，WWW 2020: 2704-2710 | ✅ Verified（dblp） |
| [18] | Liu Z 等，CIKM 2018: 2077-2085 | ✅ Verified |
| [21] | StruRW，Liu S, Li T, Feng Y 等，ICML 2023 (PMLR 202): 21778-21793 | ✅ Verified（PMLR 官方） |
| [22] | IRM，Arjovsky M 等，arXiv:1907.02893, 2019 | ✅ Verified |
| [23] | Rosenfeld E, Ravikumar P, Risteski A，ICLR 2021 | ✅ Verified |
| [24] | Kamath P 等，AISTATS 2021 | ✅ Verified |
| [25] | EERM，Wu Q, Zhang H, Yan J, Wipf D，ICLR 2022 | ✅ Verified |
| [26] | LogME，You K 等，ICML 2021: 12133-12143 | ✅ Verified |
| [27] | Ben-David S 等，Machine Learning, 2010, 79(1): 151-175 | ✅ 页码/卷/年一致；🟢 部分索引记作 79(1-2)，可不改 |
| [28] | Snell J, Swersky K, Zemel R，NIPS 2017: 4077-4087 | ✅ Verified |
| [29] | SupCon，Khosla P 等，NeurIPS 2020: 18661-18673 | ✅ Verified |

### A 节小结（29 条）

| 置信度 | 条数 | 编号 | 主要问题类型 |
|--------|------|------|------------|
| ❌ **Needs fix** | **7** | [2] [5] [7] [8] [9] [10] [16] | 作者错误 5 条、页码不可能区间 1 条、两文献折叠 1 条、虚构团体作者 1 条 |
| ⚠️ Check suggested | **4** | [1] [6] [17] [19] | 正文-著录年份冲突 2 条、著录不全 2 条 |
| ✅ Verified | **18** | [3] [4] [11] [12] [13] [14] [15] [18] [20] [21] [22] [23] [24] [25] [26] [27] [28] [29] | 无需修改 |
| | **29** | 合计 | |

**A 节总体判断（Major / Blocking）**：29 条中有 **7 条存在必须修正的硬错误**，错误率 **24%**。其中 **[7] [8] [16] 三条属"元数据正确但作者列表整体错误"**，[9] 属"两篇文献折叠为一条且框架名与方法学描述均无原文依据"，[10] 属"以虚构团体名署名"。这一错误模式（其余字段高度精确、唯独作者/年份错位）表明文献表**未经逐条回原文核对**。对一篇以"诚实证据边界"与"可审计性"为核心方法学贡献的论文而言，参考文献层的这一质量水平与论文自身主张构成直接张力，建议作者在返修时对全部 29 条重新做一次回源核验。

---

# B. 正文引用与文献表对应

## B1. 编号顺序 ✅ 通过

按正文首次出现顺序追踪全部 29 条：

`[1][2][3]`(§1.1) → `[4][5]`(§1.1) → `[6]`(§2.1) → `[7][8]`(§2.1) → `[9][10]`(§2.1) → `[11]`(§2.2) → `[12][13][14][15]`(§2.2) → `[16][17]`(§2.2) → `[18][19]`(§2.2) → `[20]`(§2.3) → `[21]`(§2.3) → `[22][23][24]`(§2.3) → `[25]`(§2.3) → `[26]`(§2.4) → `[27]`(§2.4) → `[28][29]`(§2.4)

- **首次出现顺序与编号严格一致，无跳号、无逆序** ✅
- **29 条全部在正文被引用，无"僵尸文献"** ✅
- **定级**：无问题

## B2. 出处缺失（Major，Blocking）

审查任务点名的两处均**确认成立**，且属同一类问题：**把多个并列的方法名挂在单一文献编号下**。

### B2-1 「DeepWalk、Node2vec、LINE」共用 [11]

- **定位**：正文第 54 行 —— "早期的同构图嵌入如DeepWalk、Node2vec、LINE基于随机游走学习节点表示[11]"
- **[11] 实为**：Perozzi B 等 *DeepWalk*（KDD 2014）——**只覆盖三者中的第一个**。
- **问题**：Node2vec 与 LINE 是两项独立工作，各有自己的原始文献，被并列点名却无出处。**且 LINE 并非随机游走方法**——LINE 优化的是一阶/二阶邻近度目标函数，不做随机游走采样；"三者均基于随机游走"是**事实性错误**。
- **定级**：**Major / Blocking**（出处缺失 + 事实错误）
- **建议**：补两条独立文献并分别标号，例如
  - `Grover A, Leskovec J. node2vec: Scalable feature learning for networks[C]//KDD. 2016: 855-864`
  - `Tang J, Qu M, Wang M, et al. LINE: Large-scale information network embedding[C]//WWW. 2015: 1067-1077`
  
  正文改为 "…如 DeepWalk[11]、Node2vec[新]基于随机游走、LINE[新]基于一阶与二阶邻近度学习节点表示"。（上述两条为审查者建议的标准出处，作者仍须回原文自行核验后采用。）

### B2-2 「CDAN、ADDA、JAN」共用 [20]

- **定位**：正文第 61 行 —— "DANN提出后的8年间，衍生出CDAN、ADDA、JAN等大量改进[20]"
- **[20] 实为**：Long M 等 *Conditional Adversarial Domain Adaptation*（NeurIPS 2018）——**只覆盖 CDAN**。
- **问题**：ADDA 与 JAN 无出处。此处比 B2-1 更敏感，因为 §2.3 是本文方法栈（DANN 分支）的直接理论铺垫章节。
- **附带（🟢 Info）**：**"8年间"与所列实例不匹配**——以 [5] 的 2016 为起点计至 2024 为 8 年，但 ADDA(2017)、JAN(2017)、CDAN(2018) 全部集中在起点后 1–2 年内，"8年间衍生出"的时间跨度叙述与证据不符。
- **定级**：**Major / Blocking**
- **建议**：补 ADDA 与 JAN 两条独立文献并分别标号（审查者建议的标准出处：Tzeng E 等, CVPR 2017: 7167-7176；Long M 等, ICML 2017: 2208-2217，作者须回原文核验）；"8年间"改为"其后数年间"。

## B3. 引用错配（Major，Blocking）

| # | 位置 | 正文论断 | 被引文献实际内容 | 定级 |
|---|------|---------|----------------|------|
| 1 | 第 50、51 行 | "Chen等(2022)提出**MEDAL 框架**，基于 **tri-training** 联合**图像、文本、HTML 三种模态**做**半监督**学习" | [9] 所引 *Sensors* 2020 论文为 **Chen Y 等，视觉+文本两模态 + 决策机制**；**原文无 MEDAL、无 tri-training、无 HTML 模态、非半监督**。所述内容实属另一篇 Li W 等 *Electronics* 2024, 13(11): 2199 | **Major / Blocking** |
| 2 | 第 49 行 | "沙泓州等基于**词法特征**训练 **2 层神经网络**…精度 **0.973**" | [7] 实为陈康等，**字符嵌入 + 4 路并行 CNN**（非手工词法特征）；**准确率 0.962**（非 0.973） | **Major / Blocking** |
| 3 | 第 24、50、51 行 | "Min等(2024)…提出 **AIOG 框架**" | [2] 中 AIOG = *absolute illegal online gambling*，是该文**定义的目标类别概念**，非所提框架名 | **Minor** |
| 4 | 第 49、51 行 | "**傅建明**等…提出 HADMW" | [8] 真实作者为**吴森焱等**（HADMW 方法本身正确） | **Major / Blocking** |

## B4. 正文年份与文献表年份冲突汇总（Minor，但须全部修正）

| 位置 | 正文写法 | 文献表年份 | 真实年份 | 处置 |
|------|---------|-----------|---------|------|
| 第 24 行 | 杨哲等**(2024)** | 2023 | 2023 | 正文改 2023 |
| 第 56 行 | Liu等**(2024)** 的 HeCo++ | 2023 | arXiv v1 2023 / v2 2024，已录用 TKDE | 建议改引 TKDE 正式版并统一为 2024 |
| 第 50、51 行 | Chen等**(2022)** | 2022 | Sensors **2020** / MEDAL 为 **2024** | 见 B3-1，需拆分重写 |

## B5. 未标注出处的技术名词（Minor，非 Blocking）

以下在正文中首次出现且承担实质论证功能，但未给任何出处，建议补引或明确说明为通用术语：

- §2.3 第 65 行：**Office-Home、DomainNet、ColoredMNIST、WILDS** 四个 benchmark 全部无出处，而该句正是"现有实证依据来自大规模 benchmark"这一核心对比论点的证据基础。
- §3.3.1 第 141 行：**InfoNCE**（噪声对比估计）损失无出处。
- §3.3.3 第 158 行：**CSBM（上下文随机块模型）** 无出处——该模型是 StruRW 重加权公式 (4) 的直接依据。
- §3.3.4 第 170 行：**Youden 阈值** 无出处（通用统计量，可豁免）。
- §5.3 第 360 行：**t-SNE** 无出处。
- §5.5 第 395 行：**PRISMA、GRADE** 无出处——该处以二者作为"诚实证据边界框架"的设计精神来源，属实质性类比，宜补引。

## B6. 文献表格式（Minor，非 Blocking）

- **参考文献表全部 29 条未标序号**（正文用 [1]–[29]，文献表却是无编号纯文本行）。若非 Markdown 转换丢失，则不符合《信息安全研究》著录格式，须补 `[1]`–`[29]`。
- 体例不统一：[1] 用中文全角标点无空格、其余中文条目用半角逗号加空格；[2][3] 等外文条目**句末无终止句点**，[1] 有。
- [9] [10] 标 `[EB/OL]` 却无访问日期与 URL；其中 [9] 实为正式期刊论文，类型标识本身即错（见 A 节）。

---

# C. 可复现性

## C1. 附录 C 命令链与实际脚本的一一对应 ✅ 通过

对附录 C 全部 9 组命令（含 `#0` 环境准备到 `#8.5` 精确检验补充）逐条比对发布包，**41 个被点名的脚本 / 配置文件全部存在，缺失 0 项**：

- `src/data/`：`build_hetero_graph.py`、`convert_npz_to_pyg.py`、`sanity_check.py`、`family_extractor.py` ✅
- `src/train/`：`train_full.py`、`reaggregate_week4_group_split.py`、`train_contrastive.py`、`train_transfer.py`、`train_transfer_dann.py`、`train_family_metric.py`、`train_invariance_methods.py`、`train_week11_fewshot.py` ✅
- `analysis/`：`ablation_week6.py`、`rq1_cross_verified_eval.py`、`rq1_pooled_eval.py`、`week7_transfer_acceptance.py`、`week8_acceptance.py`、`week8_patch_corrected_e5.py`、`week85_patch.py`、`feature_bucket_transfer.py`、`rq3_logme_matrix.py`、`week9_fewshot_calibration.py`、`week9_hard_negative_family_audit.py`、`rq4_fewshot_eval.py`、`week10_*.py`(5)、`rq2_rq3_rq4_final.py`、`week11_significance.py`、`week11_paper_updates.py`、`week11_acceptance.py`、`build_week11_docx_report.py`、`final_artifact_audit.py`、`supp_exact_significance.py` ✅
- `configs/`：`environment_pyg_cpu.yml`、`week4_experiments.yaml`、`week5_heco.yaml`、`week6_heco_finetune.yaml` ✅
- `docs/reproducibility_checklist.md` ✅ 存在

**评价**：命令链的**脚本层映射是完整且诚实的**，这一点应予肯定。§4.6 对照组跨源评估虽未在附录 C 单列，但其实现位于 `analysis/week8_acceptance.py`（E4 子实验），已被 `#5 阶段四机制诊断(E1-E5)` 覆盖 ✅。

## C2. §4.5 点名的 20 份少样本审计文件：**在发布包中完全不存在**（Major，Blocking）

- **稿件声明**（第 284 行）："每个seed×shot设定的实际抽样记录保存在 `audits/P3_fewshot_T1_Nordic__seed*__shots*_split.csv`，**共20份审计文件，用以二次审计无cross_verified泄漏到训练**。"
- **实测**：对发布包（及其 `.git` 之外全部路径）做全盘检索，`P3_fewshot*` 只命中**一个文件**——`results/week11/plots/P3_fewshot_curves.png`（一张曲线图）。**`*_split.csv` 模式在整个发布包中命中 0 个文件**。
- **发布包中不存在任何顶层 `audits/` 目录**；11 个 `results/*/audits/` 子目录中，与少样本相关的 `results/week9/audits/` **为空目录（0 个文件）**。
- **更关键的政策性矛盾**：`DATA_AVAILABILITY.md` 的排除清单**明文写入** `few-shot support sample lists` 与 `split files with selected sample identifiers`；`README.md` 第 72–73 行同样明文排除 `support sample lists, and split sample dumps`。**即这 20 份文件不是遗漏，而是被发布政策有意排除。**
- **后果**：§4.5 用这 20 份文件作为"few-shot 抽样池未污染评估集"这一**方法学风险的唯一缓解证据**。既然文件按政策永不公开，该缓解措施**对任何外部读者均不可验证**，"用以二次审计"的承诺无法兑现。这与全文反复主张的"可被版本控制系统审计的元数据"（§5.5）构成正面冲突。
- **定级**：**Major / Blocking**
- **定位**：正文第 284 行；对照 `DATA_AVAILABILITY.md`、`README.md` L72-73
- **建议**（三选一，不可回避）：
  1. **推荐**：将 20 份 split 文件**脱敏后释出**——只保留 `seed, shot, sample_index(匿名序号), tier(single/cross_verified)` 四列，不含域名或站点标识。这样既不泄漏站点身份，又能让读者验证"cross_verified 计数恒为 0"这一唯一需要审计的性质；
  2. 释出一份**聚合审计表**（20 行：seed×shot × 抽样池规模 × cross_verified 命中数），把"二次审计"的对象从原始清单降级为可核验的汇总数；
  3. 若坚持完全不释出，则必须删除"用以二次审计"的措辞，并在正文明示"该抽样记录因数据发布政策不公开，读者无法独立验证"。

## C3. §3.3.4 点名的 24 行 Holm 完整记录：**不在公开仓库中**（Major，Blocking）

- **稿件声明**（第 173 行）："含假设编号、情境、指标方向、配对单元、效应量、95%置信区间、原始p值、Holm秩次与阈值、校正后p值及判定的24行完整记录**随代码释出**(`reports/supplementary_exact_significance/supp_holm_24row_record.csv`)。"
- **实测**：该文件**存在于本地发布包**（24 数据行，字段与稿件描述**逐项吻合**：`transfer_id, method_a, method_b, metric_direction, n_pairs, mean_delta_a_minus_b, ci95_low, ci95_high, bootstrap_p, holm_rank, holm_threshold, holm_adjusted_p, reject_holm`），但 `git status` 显示其为 **untracked（`?? reports/supplementary_exact_significance/supp_holm_24row_record.csv`）**。
- **远程确认**：抓取公开仓库 `c29fb11` 下该目录，实际只有 **2 个文件**——`supp_exact_fewshot_vs_0shot.csv` 与 `supp_exact_w7_method_pairwise.csv`。**`supp_holm_24row_record.csv` 不在公开仓库中**。
- **后果**：读者按稿件给出的精确路径去公开仓库取证会得到 404。而该文件恰是 §4.4"9 对 Holm 校正后显著"及其三分类解读（5 对方法间互比 / 2 对 source_only 反超 / 2 对退化伪优）的**唯一原始凭据**。
- **定级**：**Major / Blocking**（可一次 `git add` 解决，成本极低，但不解决即为虚假声明）
- **建议**：提交该文件至公开仓库并更新发布 tag；或将稿件路径改指向已公开的 `supp_exact_w7_method_pairwise.csv`（但后者不含 bootstrap 的 CI 与 Holm 秩次，无法完整支撑 §4.4 的表述）。

## C4. 附录 C 声称的"sha256 哈希"在核查清单中并不存在（Major）

- **稿件声明**（第 526 行）："完整可复现性核查清单（**含每步预期输出与sha256哈希**）见仓库 `docs/reproducibility_checklist.md`。"
- **实测**：`docs/reproducibility_checklist.md` 共 132 行，**"sha256" / "sha" / "hash" / "哈希" 的出现次数均为 0**。该文件只有环境说明、结果包路径列表与可重跑脚本清单，**既无"每步预期输出"，也无任何哈希值**。
- **旁证**：哈希确实存在于别处——`data/versions/hetero_graph_v2.sha256`、`data/external/maxmind/GeoLite2-ASN.sha256`。但这与"核查清单含每步 sha256"是两回事。
- **附加问题（Major）**：`docs/reproducibility_checklist.md` **是原始工作区的清单，未针对发布包改写**。它所列的"Saved Result Packages"中，`output/`（全部 10 个 week 目录）、`data/graphs/hetero_graph_v2.pt`、`data/graphs/hetero_graph_v2.npz`、`data/audits/*.csv` **在发布包中一律不存在**。读者按该清单逐项核对会得到大面积 MISS，反而制造"发布包不完整/被删改"的误判。
- **定级**：**Major**（C4 前半段为 Blocking——正文对已发布文件的内容作了不实描述）
- **建议**：(a) 删除"含每步预期输出与sha256哈希"这一限定，或 (b) 在 `reproducibility_checklist.md` 中真正补上每步预期输出文件名与其 sha256；同时把该清单中指向非公开路径的条目**显式标注为"仅限原始工作区，不在公开包内"**。

## C5. StruRW 执行入口的自相矛盾（Minor）

- 附录 C 第 490 行注释："`#StruRW及组合由train_transfer.py/train_transfer_dann.py按strurw配置执行`"。
- 实测发布包中**存在独立脚本 `src/train/train_transfer_strurw.py`**，却未在命令链中出现。
- **定级**：**Minor / 非 Blocking**
- **建议**：核实实际执行入口。若 `train_transfer_strurw.py` 才是真正入口，应在附录 C 中列出并删去该注释；若确为遗留脚本，宜在仓库中注明其已废弃，避免读者误用得到不一致结果。

## C6. 可核验的数字：**全部对上**（正面结论）

以下稿件数字由审查者直接对发布包中的聚合文件重算核对，**全部一致**：

| 稿件位置 | 稿件数字 | 发布包核验 | 结论 |
|---------|---------|-----------|------|
| §3.2 / 摘要 | Website 623、IP 1325、Cert 510、NS 959、Registrar 79、ExtRef 53 | `data/node_stats.csv` 逐行一致 | ✅ |
| §3.2 / 摘要 | 边 1516+1662+519+252+57 = **4006** | `data/edge_stats.csv` 求和 = 4006 | ✅ |
| 表 6（20 格） | C1/C2/C3/C4/C5 × 4 情境全部数值 | `results/week8/metrics/E5_structure_vs_lexical_summary.csv` 逐格一致 | ✅ |
| §5.1 | lex_only 0.953 / graph_only 0.713 / 结构边际增益 ≈3 个百分点 | 0.9530 / 0.7134，0.9861−0.9530=0.033 | ✅ |
| §4.4 | "9 对在 Holm 校正后显著" | `supp_holm_24row_record.csv` 中 `reject_holm=True` 计数 = **9** | ✅ |
| §4.4 / 附录 E | "24 个方法对" | `supp_exact_w7_method_pairwise.csv` 数据行 = **24** | ✅ |
| §4.4 / 附录 E | "6 对停在两侧精确下限 p=0.0625" | 该文件 `perm_p_two_sided==0.0625` 计数 = **6** | ✅ |
| 附录 E | "少样本 16 个档位中有 9 档停在该下限" | `supp_exact_fewshot_vs_0shot.csv` 16 行，命中 = **9** | ✅ |
| 附录 E | "tests/ 包含 32 项单元测试" | 5 个测试文件类内外 `def test_` 合计 = **32**（9+6+4+4+9） | ✅ |
| §4.5 | "共 20 份审计文件" | 算术自洽（T1_Nordic × 5 seed × 4 档位）但**文件不存在**，见 C2 | ⚠️ |

## C7. §5.1 关于两份 E5 文件的主动披露：**核验属实，应予肯定**（正面结论）

- 稿件第 297 行主动声明："表6中C3配置的数值取自E5消融的**原始配置组**(`E5_structure_vs_lexical_summary.csv`)；该配置在**修正配置组**中被重定义为完全移除词法特征，故开源仓库两份E5文件的C3数值不同。"
- **核验**：`results/week8/metrics/` 中 C3 标签为 `C3_no_lex`（T1=0.219, T2_ON=0.372, T2_PH=0.650, T3=0.953）；`results/week8_patch/metrics/` 中同位标签改为 `C3_no_website_lexical`，数值变为与 `C4_graph_only` **完全相同**（T1=0.045, T2_ON=0.426, T2_PH=0.598, T3=0.713）。表 6 采用的是前者，**与声明完全一致** ✅
- **评价**：这是一处主动、准确、可核验的口径披露，是本文"诚实证据边界"主张少数落到实处的地方，建议在返修中保留原文措辞。

---

# D. 数据可用性声明

> 依审查范围，本节**不涉及** DOI、仓库永久标识、存档校验值、发布 tag 等出版社定调内容（稿件第 422 行已以【作者补充】占位，属正常流程）。以下只审查**释出/未释出范围的真实性**与**许可证覆盖范围的一致性**。

## D1. §1.3 与 §6.4 就"检查点"自相矛盾（Major，Blocking）

| 位置 | 表述 |
|------|------|
| §1.3 第 40 行 | "…**5个HeCo编码器(encoder)模型检查点**、完整的可复现性命令链等成果均已整理。**这些产出向后续研究者开放**" |
| §6.4 第 422 行 | "出于安全与隐私考虑，站点级原始数据（站点登记表、逐站预测明细、**训练检查点**）**不公开**" |

- **同一篇稿件的两处对同一对象给出相反声明。** 发布包侧的事实站在 §6.4 一边：`README.md` L72 明文排除 `*.pt/*.pth/*.ckpt/*.safetensors`，`DATA_AVAILABILITY.md` 明文排除 `PyTorch graph/checkpoint binaries`，全盘检索确认发布包内 `.pt/.pth/.ckpt/.npz` 文件数为 **0**。
- **定级**：**Major / Blocking**
- **建议**：删去 §1.3 中"5个HeCo编码器模型检查点"，或改写为"5 个 HeCo 编码器检查点保留于受控环境，可按 §6.4 的受控访问流程申请"。

## D2. §1.3 声称开放的"异构图"实际未释出，且 §6.4 的排除清单**漏列了它**（Major，Blocking）

- §1.3 第 40 行称 "**11国613站点/6节点类型/4,006边的异构图**…均已整理…向后续研究者开放"。
- **实测**：`data/graphs/` 下**只有** `hetero_graph_v2_metadata.json`（元数据）；`hetero_graph_v2.pt` 与 `hetero_graph_v2.npz` **均不在发布包内**（被 README/DATA_AVAILABILITY 的 `PyTorch graph/checkpoint files` 一条排除）。`data/` 全目录只有 11 个文件，均为 schema、聚合计数与版本记录。
- **更严重的是 §6.4 的排除清单本身不完整**：它只列了"站点登记表、逐站预测明细、训练检查点"三项，**未提及冻结图 `hetero_graph_v2.pt` 本身不公开**，也未提及"少样本 support 清单与 split 文件"不公开（见 C2）。读者据 §6.4 会合理推断"图是公开的、只有站点级明细不公开"，而事实相反——**没有图，附录 C 中 `#2` 之后的全部命令均无法执行**。
- **定级**：**Major / Blocking**（声明与事实方向相反，且遗漏了对复现最关键的一项）
- **建议**：§6.4 的不公开清单补齐为"站点登记表与站点级节点/边 CSV、逐站预测明细、**冻结异构图二进制（`hetero_graph_v2.pt/.npz`）**、训练检查点、**少样本 support 样本清单与划分文件**"；§1.3 相应改为"图的 schema、节点/边聚合统计与版本校验值已开放，图二进制本身按受控流程提供"。

## D3. MIT 覆盖范围：稿件的限缩表述在 LICENSE 中无对应条款（Minor）

- §6.4 称："代码与聚合结果采用MIT许可证；**许可证覆盖范围限于本仓库内的源代码与聚合指标文件**，不覆盖第三方依赖库与外部数据源。"
- **实测 LICENSE**：标准 MIT 全文，**无任何范围限缩条款**，其 "the Software" 按字面覆盖仓库全部内容——包括 `figs/`、`results/**/plots/`（38 个图像文件）、`paper/` 下的论文草稿与 `docs/` 下的报告。
- **两处不一致**：
  1. 稿件把 MIT 说得比实际**更窄**（漏了图表与论文草稿）。这在方向上不损害使用者，但属声明不准确；
  2. 稿件称"不覆盖第三方依赖库与外部数据源"——这一点**发布包侧有实据支撑** ✅：`data/external/maxmind/` 只放了 `GeoLite2-ASN.sha256` 与 README，未再分发 MaxMind 数据本体，`REPRODUCIBILITY`/`reproducibility_checklist` 亦注明 "MaxMind GeoLite2-ASN and other local-only data dependencies are not redistributed"。
- **定级**：**Minor / 非 Blocking**
- **建议**：§6.4 改为"本仓库内容（源代码、配置、聚合指标、图表与报告）整体采用 MIT 许可证；第三方依赖库与外部数据源（如 MaxMind GeoLite2-ASN）不在此授权范围内，须遵循其各自许可"。同时注意 §3.2 的 IP 节点特征依赖 Top-16 ASN 编码，该外部依赖对复现是硬约束，宜在 §6.4 一并点明。

## D4. 仓库署名与稿件作者不一致（Minor，但涉及归属）

- `LICENSE`：`Copyright (c) 2026 **Duziteng**`
- `CITATION.cff`：`authors: - name: "**Duziteng**"`（唯一作者，`date-released: 2026-05-10`）
- 稿件署名为**三位作者**：张李斌、杜宇宸、王红杰。
- **问题**：公开仓库的版权持有人与唯一 CITATION 作者为单一名称 "Duziteng"，与稿件三位作者中任何一位的规范拉丁转写均不对应（杜宇宸 = Du Yuchen）。若为个人账号代号，则仓库层的著作权归属与论文层的作者贡献声明之间缺少衔接；若他人据 `CITATION.cff` 引用，将只引到一个与论文作者表不匹配的名字。
- **定级**：**Minor / 非 Blocking**
- **建议**：`CITATION.cff` 补齐三位作者及其机构，`LICENSE` 版权行改为与论文一致的署名（或"作者团队 + 单位"）；`CITATION.cff` 的 `date-released` (2026-05-10) 亦早于 README/MANIFEST 的最后更新（2026-07），宜同步。

## D5. §1.3 "5层证据分级标签"的开放程度未作区分（Minor）

- §1.3 称"5层证据分级标签…向后续研究者开放"。实际发布包中只有 `data/node_stats.csv` 层面的**聚合计数**，**逐站点的证据层标签不公开**（属被排除的"processed node/edge CSVs with website-level rows"）。
- **定级**：**Minor / 非 Blocking**
- **建议**：改为"5 层证据分级的**定义与各层聚合计数**已开放"。

## D6. 正面结论：发布包侧的政策文档本身是自洽的

`README.md`、`DATA_AVAILABILITY.md`、`REPRODUCIBILITY.md`、`SECURITY.md` 四份文件对"排除什么、为什么排除、剩下什么可以审计"的表述**相互一致，且与发布包实际内容吻合**（审查者逐项抽查排除清单的 5 类文件，均确认为 0 命中）。`DATA_AVAILABILITY.md` 的 "Reproduction Note" 更明确写出"Without those files, this repository supports code review, claim audit, table/figure inspection, ... **but not a complete end-to-end rerun from raw data**"——这是准确且负责任的表述。

**本节的全部问题，均出在稿件正文（§1.3、§6.4、§4.5、附录 C）对发布包的描述比发布包自己的说法更乐观**。修改成本低：把正文的四处表述向发布包侧的既有政策文档对齐即可，无需改动仓库。

---

# E. 问题清单（按优先级）

## Blocking（必须修正后方可接受）

| # | 类别 | 问题 | 定位 |
|---|------|------|------|
| 1 | 文献 | [8] 作者应为吴森焱等，非傅建明等 | 文献表 8；正文 L49, L51 |
| 2 | 文献 | [7] 作者应为陈康等，非沙泓州等；且正文"精度0.973"应为 0.962，方法非词法特征而是字符嵌入+CNN | 文献表 7；正文 L49, L51 |
| 3 | 文献 | [16] 作者应为 Wang Z(Zehong) 等，现著录疑似误植 HeCo 作者；缺页码 136-144 | 文献表 16 |
| 4 | 文献 | [2] 作者应为 Min M, Lee D A（2 人），现多出 Park M 且第二作者缩写错 | 文献表 2 |
| 5 | 文献 | [9] 把 Sensors 2020（Chen Y 等）的题名与 Electronics 2024（Li W 等 MEDAL）的内容折叠为一条；MEDAL / tri-training / 三模态 / 半监督在所引文献中均不存在 | 文献表 9；正文 L50, L51 |
| 6 | 文献 | [10] 作者著录为虚构的 "URLBERT Team" | 文献表 10 |
| 7 | 文献 | [5] 页码 "2096-2030" 终页小于起页，应为 17(59): 1-35 | 文献表 5 |
| 8 | 引用 | Node2vec、LINE 无出处（共用 DeepWalk 的 [11]）；且"LINE 基于随机游走"为事实错误 | 正文 L54 |
| 9 | 引用 | ADDA、JAN 无出处（共用 CDAN 的 [20]） | 正文 L61 |
| 10 | 复现 | §4.5 点名的 20 份 `audits/P3_fewshot_*_split.csv` 在发布包中 0 命中，且被发布政策明文永久排除，"二次审计"承诺不可兑现 | 正文 L284 |
| 11 | 复现 | §3.3.4 点名的 `supp_holm_24row_record.csv` 为 untracked，**不在公开仓库 c29fb11 中** | 正文 L173 |
| 12 | 复现 | 附录 C 称核查清单"含每步预期输出与 sha256 哈希"，该文件中 sha256 出现次数为 0 | 正文 L526 |
| 13 | 数据 | §1.3"检查点已开放" 与 §6.4"检查点不公开" 直接矛盾；事实为不公开 | 正文 L40 vs L422 |
| 14 | 数据 | §1.3 称异构图"向后续研究者开放"，实际 `hetero_graph_v2.pt/.npz` 未释出；且 §6.4 的不公开清单**漏列冻结图与少样本 split 文件** | 正文 L40, L422 |

## Major（非 Blocking，但应在返修中处理）

| # | 问题 | 定位 |
|---|------|------|
| 15 | 附录 C 的 `docs/reproducibility_checklist.md` 是工作区版本未改写，所列 `output/`、`data/graphs/*.pt`、`data/audits/` 在发布包中全部不存在 | 仓库 docs/ |
| 16 | 正文称 Min 等提出"AIOG 框架"，AIOG 实为该文定义的目标类别概念 | 正文 L24, L50 |
| 17 | 参考文献表 29 条**全部未标序号** | 文献表 |

## Minor

| # | 问题 | 定位 |
|---|------|------|
| 18 | 正文"杨哲等(2024)" vs 著录 2023；且该文仅 2 位作者不宜用"等" | L24 |
| 19 | 正文"Liu等(2024)的HeCo++" vs 著录 2023；建议改引已录用的 TKDE 正式版 | L56 |
| 20 | [19] HNN4RP 缺卷期页（应补 47(10): 3395-3409） | 文献表 19 |
| 21 | [6] 专利缺申请人/发明人与完整公告日期 | 文献表 6 |
| 22 | "DANN提出后的8年间"与所列 2017–2018 年实例不匹配 | L61 |
| 23 | Office-Home / DomainNet / ColoredMNIST / WILDS、InfoNCE、CSBM、t-SNE、PRISMA、GRADE 均无出处 | L65, L141, L158, L360, L395 |
| 24 | 附录 C 称 StruRW 由 train_transfer*.py 执行，但仓库另有未列出的 `src/train/train_transfer_strurw.py` | L490 |
| 25 | §6.4 的 MIT 覆盖范围表述比 LICENSE 实际授权更窄（漏图表与论文草稿） | L422 |
| 26 | LICENSE / CITATION.cff 署名 "Duziteng" 与稿件三位作者均不对应 | 仓库根目录 |
| 27 | §1.3 称"5层证据分级标签"开放，实际仅开放定义与聚合计数 | L40 |
| 28 | 文献表体例不统一（中英文标点、句末句点、[EB/OL] 缺访问日期与 URL） | 文献表 |

## 应予肯定之处

1. **附录 C 命令链的脚本映射零缺失**——41 个被点名的脚本/配置全部存在，这在同类稿件中并不常见。
2. **可核验数字全部对上**——节点数、边数（4006）、表 6 全部 20 格、9 对 Holm 显著、24 个方法对、6 对停在精确下限、16 个 shot 档位、9 档停在下限、32 项单元测试，逐项复算一致。
3. **§5.1 关于两份 E5 文件 C3 口径差异的主动披露，经核验完全属实**，是"诚实证据边界"落到实处的实例。
4. **发布包侧的四份政策文档相互自洽且与实际内容吻合**，`DATA_AVAILABILITY.md` 明确写出"不支持端到端完整重跑"，表述准确负责。
5. **编号顺序与正文首次出现顺序严格一致，29 条无一条未被引用。**

---

# 总结（审查结论）

参考文献层是本稿最薄弱的环节：29 条中 **7 条含必须修正的硬错误（24%）**，其中 [7][8][16] 三条呈现同一模式——**刊名、年、卷、期、页码乃至论文内报告的具体数值全部精确，唯独作者列表整体错位**；[9] 更把两篇不同论文（Sensors 2020 与 Electronics 2024 的 MEDAL）折叠为一条，致使正文"MEDAL/tri-training/三模态/半监督"的整段论断在所引文献中找不到任何依据；[10] 以虚构的"URLBERT Team"充当作者；[5] 的页码为终页小于起页的不可能区间。审查任务点名的两处出处缺失（DeepWalk/Node2vec/LINE 共用 [11]，CDAN/ADDA/JAN 共用 [20]）均确认成立，其中"LINE 基于随机游走"另属事实错误。编号顺序与引用完整性则无问题。

可复现性方面，附录 C 的命令链**脚本映射零缺失**，且节点/边计数、表 6 全部 20 格、9 对 Holm 显著、24 个方法对、32 项单元测试等可核验数字**逐项复算一致**，§5.1 对两份 E5 文件口径差异的主动披露亦经核验属实。但有三处"承诺了却拿不到"的硬伤：§4.5 点名的 20 份少样本 split 审计文件在发布包中 0 命中且被发布政策明文永久排除；§3.3.4 点名的 24 行 Holm 记录为 untracked，不在公开仓库 c29fb11 中；附录 C 声称核查清单含每步 sha256，实际该文件中 sha256 出现次数为 0。

数据可用性声明的问题**全部出在正文比发布包自己的说法更乐观**：§1.3 称检查点与异构图"向后续研究者开放"，与 §6.4 及发布包事实直接矛盾；§6.4 的不公开清单又漏列了冻结图与少样本 split 文件这两项对复现最关键的排除项。发布包侧四份政策文档本身自洽准确，修改成本极低——把正文四处表述向仓库既有政策对齐即可。

**结论：Major Revision。** 14 项 Blocking 问题须全部处理，重点是参考文献回源重核（建议 29 条逐条重做）与三处"点名却不可获取"的审计文件。鉴于本文以"诚实证据边界"与可审计性为核心方法学贡献，引用层 24% 的硬错误率与承诺的审计文件不可获取，与论文自身主张构成直接张力，此点应在返修意见中明确指出。
