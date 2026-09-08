# 学术审查委员会 · Nature 视角盲审报告

**稿件**：`manuscript_v9.md` —《图神经网络在跨地区非法网络赌博识别中的应用边界——基于11国异构图的域适应失效模式研究》
**审查形式**：盲审（不假设作者身份、不迎合），依据稿件正文 + 开源仓库可核验数据
**开源基线**：`D:\hetero_transfer_experiment_release\hetero_transfer_experiment_release`（只读）
**审查排除项**：DOI、投稿编号、版面费等出版社定调内容
**评估轴**（源自 nature-reviewer 技能）：originality / scientific importance / interdisciplinary readership / technical soundness / readability for nonspecialists

---

## 审稿设置（Review setup）

- **输入范围**：完整中文稿件（536 行，含摘要、正文 6 章、参考文献 29 条、附录 A–E）+ 开源仓库 `results/week4..week11`、`reports/supplementary_exact_significance/`、`src/`、`analysis/`、`configs/`、`tests/`。
- **评估边界**：
  - 稿件所有插图以 `[图N 位置]` 占位符呈现，**未提供图像文件**；仓库 `figs/` 仅有 1 个 PDF（`fig4_pooled_auc.pdf`）。因此"图表与正文一致性"只能就**表格**与**文字数值**核验，图 1–图 8 的一致性**不可评估**。
  - 站点级原始数据、逐站预测、训练检查点、`runs/logs/predictions/splits` 运行记录均未释出（作者自查文件 `reports\rerun_2026-07-18\rerun_final_artifact_audit.csv` 中这些条目状态即为 `fail`）。因此本审查为**聚合指标层面的可核验性审查**，不构成端到端复现。
  - 未做重新训练；所有核验为对已释出 CSV 的直接读取与重算（含对 `analysis/week11_acceptance.py` 关键统计量的逐行复算）。
- **稿件主张摘要**：在 11 辖区 613 站点异构图上，(a) HeCo 对比预训练+微调在"非法 vs 持牌"上达到 0.93 量级合并 AUC，但在未见家族困难负样本上退化至 0.622（tsars 0.386）；(b) DANN/StruRW/IRM/EERM 四类域适应/域泛化方法均未稳健优于 source_only；(c) 少样本目标域校准是唯一稳健路径；(d) 提出"诚实证据边界"预注册与审计框架。
- **可见证据基础**：80 次迁移运行、100 次少样本运行、24 对配对显著性检验（bootstrap-Holm + 精确符号翻转置换）、5 类边消融、5 配置特征/结构消融、7 家族 LOFO、3 场景对照组跨源评估、15 项阶段七预声明假设审计、32 项单元测试。
- **影响置信度的缺失材料**：全部图件；稿件 §4.5 明确点名的 20 份少样本抽样审计文件；1,500 余次运行的记录文件；独立团队复现；inductive 评估。

---

## 第一部分：关键数值抽查核验表

> 方法：逐条读取仓库 CSV 并与稿件数值比对；涉及派生量的做重算。判定分 **相符 / 不符 / 无法核验**。

| # | 稿件数值与定位 | 仓库来源 | 仓库实测 | 判定 |
|---|---|---|---|---|
| 1 | **0.928±0.075**（摘要、§4.1、表2、§6.1） | `results\week4\metrics\pooled_primary_summary.csv` | `hetero_gnn` = 0.927738949 / 0.074954368 | **数值相符；归属不符**（见下方专项核查） |
| 2 | **0.932±0.072** HeCo 微调(分立LR)（§4.1、表2） | `results\week6\metrics\pooled_primary_finetune_summary.csv` | `heco_disc_lr_finetune` = 0.932151489 / 0.071558303 | **数值相符；但配置归属与数据冲突**（见 M2） |
| 3 | **0.934±0.074** HeCo 微调(统一LR)（§4.1、表2） | `results\week6\metrics\pooled_primary_finetune_uniform_lr_summary.csv` | 0.933710784 / 0.074018039 | 相符 |
| 4 | **0.980±0.000**，"n=33，全丹麦"（§4.1、表2、§5.5、§6.1） | `results\week6\metrics\cross_verified_pooled_eval.csv` | pooled_5_seed = 0.979761905；逐seed = 1.0/1.0/1.0/1.0/**0.7209**；pooled 行 `num_rows=234`、`num_cross_verified=24` | **均值相符；"±0.000"与"n=33"不符** |
| 5 | **0.946±0.044**（S1，§4.1、§4.6） | `results\week8\metrics\E4_three_scenario_summary.csv` | 0.946000 / 0.044338 | 相符 |
| 6 | **0.970±0.013**（S2，摘要、§4.6、§5.4） | 同上 | 0.970424 / 0.013125 | 相符 |
| 7 | **0.969±0.022**（S3，§4.6） | 同上 | 0.968492 / 0.022450 | **不符（四舍五入应为 0.968）** |
| 8 | **0.622** 家族困难负样本均值（摘要、§5.3、表8、§6.1） | `results\week9\metrics\hard_negative_family_summary.csv`；`results\week10\metrics\hard_negative_family_final_summary.csv` | 7 家族均值 = 0.6218496 = `overall_mean_auc` | 相符 |
| 9 | **0.386** tsars（摘要、表8、§6.1） | 同上 | 0.3864198 | 相符 |
| 10 | 表 8 其余 6 家族 AUC / recall@FPR0.10 / score_gap（共 21 个数） | 同上 | 0.850/0.7875/0.770/0.7278/0.425/0.40625 及全部 recall、gap | 全部相符 |
| 11 | **0.642 → 0.938**（摘要、§4.5、表5） | `results\week11\metrics\P3_full_fewshot_matrix.csv` | 0-shot 0.641667；5-shot 0.937500 | 相符 |
| 12 | 表 5 全部 20 个 shot 档数值 + 4 个 Δ 对 0 | 同上 | T2_PH 0.642/0.808/0.942/0.938/0.954(Δ+0.3125)；T3 0.986/0.959/0.982/0.886/0.996(Δ+0.0096)；T1 0.206/0.342/0.452/0.458/0.594(Δ+0.3871)；T2_ON 1−mpi 0.697/0.669/0.757/0.786/0.849(Δ+0.1522) | 全部相符 |
| 13 | **0.918** 冻结线性探针（§3.3.1、§4.1、表2） | `results\week5\metrics\heco_temperature_grid_summary.csv` | τ=0.7 frozen = 0.917960；τ=0.5 同值（"并列"表述成立）；τ=0.7 验证集 0.8633 为网格最高（选型逻辑成立） | 相符（但见 m5） |
| 14 | **0.894±0.059 / 0.888±0.057** LR / MLP（§4.1、表2） | `results\week4\metrics\pooled_primary_summary.csv` | 0.893844/0.058959；0.887612/0.057147 | 相符 |
| 15 | §4.2 迁移矩阵 16 个主指标（0.986/0.975/0.975/0.990；0.642/0.537/0.646/0.746；1−mpi 0.697/0.640/1.000/0.976；0.206/0.271/0.142/0.258） | `results\week7\metrics\transfer_method_summary.csv` | 全部逐位吻合（T2_ON 由 mpi 0.302760/0.359625/0.000267/0.024269 换算） | 相符 |
| 16 | 表 4 IRM/EERM 共 8 个 Δ（+0.013/+0.032/−0.079/−0.062/−0.042/+0.019/−0.021/+0.007/−0.001） | `results\week11\metrics\P2_method_comparison.csv`（重算） | 全部相符，且 T2_ON 行方向处理正确 | 相符 |
| 17 | §4.4 "9 对 Holm 校正后显著"及 5/2/2 三类分解 | `reports\supplementary_exact_significance\supp_holm_24row_record.csv` | `reject_holm=True` 恰 9 行；逐行归类与稿件三类划分完全一致 | 相符（本稿最扎实的一处） |
| 18 | §4.4 精确检验"24 对无一 Holm 显著""6 对停在 p=0.0625" | `reports\...\supp_exact_w7_method_pairwise.csv` | `holm_reject_perm_two_sided` 全 False；`perm_p_two_sided==0.0625` 恰 6 行 | 数量相符；**附加断言不符**（见 M6） |
| 19 | §4.5/附录E "16 档中 9 档停在下限（T2_PH 与 T1_DK 的全部档位）" | `reports\...\supp_exact_fewshot_vs_0shot.csv` | 恰 9 行 = T1 全 4 档 + T2_PH 全 4 档 + **T2_ON 10-shot** | 数量相符；**括注枚举不符** |
| 20 | 表 7 边消融 6 行（效应/CI/p） | `results\week8\metrics\E1_edge_ablation_summary.csv` | 5 行相符；**T2_ON×registered_via CI 实为 [+0.015,+0.049]，稿件写 [+0.019,+0.046]** | **不符** |
| 21 | 表 7 表注"霍尔姆校正后 p<0.05" | 同上 | 文件仅有未校正 `p_value`，**无任何 Holm 列**；20 次检验下 T1_DK×registered_via（p=0.028）Holm 阈值 0.05/16=0.0031，**不显著** | **不符** |
| 22 | 表 6 C1/C2/C4/C5 共 16 个数 | `results\week8\metrics\E5_structure_vs_lexical_summary.csv` | 全部相符 | 相符 |
| 23 | 表 6 **"C3 仅留 ccTLD"**（注："保留 ccTLD 与 dot_com 两维"） | `configs\week8.yaml` L177；`src\explain\structure_vs_lexical.py` | 配置名为 `C3_no_website_lexical`，`feature_mode: zero_website_lexical`（**移除全部 Website 词法**）；仓库中**不存在**"仅保留 ccTLD+dot_com"的配置 | **不符（语义无据）** |
| 24 | §5.1 "T2_ON 上图结构明显主导——graph_only(0.426) 显著高于 lex_only(0.323)" | 同上 + `results\week8\metrics\E1_edge_ablation_summary.csv`（`primary_metric_direction=lower_is_better`） | T2_ON 主指标为 mean_pred_illegal，**越低越好**；0.426 实为**更差** | **方向不符（结论反了）** |
| 25 | §4.3 "+0.271 来自 DANN+StruRW 在 T2_PH（0.642→0.746）" | 复算 `analysis\week11_acceptance.py` L120–145 + `P2_method_comparison.csv` + `results\week7\metrics\transfer_summary.csv` | `max_improvement` 仅在 `result_origin=="week11_P2"`（**只含 IRM/EERM**）上计算；最大值 0.27083333 = **IRM 在 T2_PH seed 44**（0.8125 vs 0.541667）。而 0.746−0.642 = **+0.104** | **不符（归属与算术双重错误）** |
| 26 | §4.5 去词法对照：T2_PH 0.856±0.136 / 0-shot 0.598 / 增量 +0.081；T3 去 ccTLD 0.965±0.036 | `results\week9\metrics\W9_E3_shortcut_aware_summary.csv` | 0.856250/0.136295；0.597917；0.9375−0.85625=0.08125；0.965158/0.035783 | 全部相符 |
| 27 | §5.3 SupCon 退化：均值 0.126、tsars 0.074、Δ=−0.496、区间 0.07–0.25 | `results\week11\metrics\P1_family_metric_lofo.csv`（重算） | 0.125892；0.074074；−0.495958；逐家族 0.074–0.250 | 全部相符 |
| 28 | §5.3 消融 A1–A4 共 8 个数（0.620/+0.054、0.078/−0.210、0.082/−0.215、0.308/−0.018） | `results\week11\metrics\P1_alpha_ablation.csv`（重算 ggbet 5 seed） | 0.620/+0.054521；0.078/−0.209663；0.082/−0.214878；0.308/−0.018022 | 全部相符 |
| 29 | 图规模：623/1325/510/959/79/53 节点；4,006 边（1516+519+1662+252+57） | `data\node_stats.csv`、`data\edge_stats.csv` | 逐项吻合，边数合计 4006 | 相符 |
| 30 | 样本口径 613 = 275+223+33+82；531 训练池；+10 灰区 = 623 | `manuscript` §3.1 自洽性 + 节点数 | 算术自洽，且与 Website=623 一致 | 相符 |
| 31 | §3.3.2 "attention_pool head 在 seed46 出现 attention collapse（metapath_attention 降至 0.023）" | `results\week6_head_ablation\audits\attention_collapse_audit.csv` | seed46 `metapath_attention_mean = 0.3069`；**5 个 seed 的 `attention_collapse_flag` 全为 False** | **不符 / 无法核验** |
| 32 | §4.5 "20 份审计文件 `audits/P3_fewshot_T1_Nordic__seed*__shots*_split.csv`" | 全仓库 `find` | **0 份，文件不存在** | **不符** |
| 33 | 附录A/§5.5 阶段七 15 假设、通过 9 失败 6 | `results\week11\metrics\week11_acceptance_audit.csv` | 15 行；pass 9 / fail 6（P1 四项 + P3a + P3c） | 相符 |
| 34 | 附录E "32 项单元测试" | `tests\*.py` | `def test_` 计数 = 32 | 相符 |
| 35 | 附录C 命令链 21 个脚本/文档 | `src/`、`analysis/`、`docs/` | 21/21 全部存在 | 相符 |
| 36 | §3.2 "125 项自动验收" | `src\data\sanity_check.py` docstring | "Run the 125 automated acceptance checks" | 相符 |
| 37 | §5.5/§1.3 "1,500 余次实验运行" | 全仓库聚合 raw_runs 行数合计 ≈ 数百；`runs/logs/predictions/splits` 目录缺失（作者自查 `rerun_final_artifact_audit.csv` 标 fail） | — | **无法核验** |
| 38 | 图 1–图 8 与正文一致性 | `figs\`（仅 `fig4_pooled_auc.pdf`） | — | **无法核验** |

**核验小结**：38 项中 **24 项相符**、**10 项不符**、**2 项无法核验**（另 2 项为混合判定）。相符项集中在"负结果"主干（迁移矩阵、显著性检验、少样本矩阵、家族退化）；不符项**高度集中在第 5 章机制诊断与表 2/表 6/表 7 的口径与标签层**——即稿件自称的"主要差异化贡献"处。

---

## 第二部分：专项核查 —— 0.928±0.075 的模型归属

**结论：稿件内部对同一数字给出了两种互斥归属，其中 §6.1 的归属与开源数据不符。**

1. **仓库事实**：`results\week4\metrics\pooled_primary_summary.csv` 中该行的 `model` 字段为 **`hetero_gnn`**，`roc_auc_mean = 0.927738949`、`roc_auc_std = 0.074954368`。同文件另有 `logistic_regression`(0.8938) 与 `mlp`(0.8876)。该文件中**不存在任何 HeCo 相关模型行**；HeCo 微调结果在 `results\week6\metrics\` 下，为 0.9322(disc-LR) 与 0.9337(uniform-LR)。
2. **稿件 §4.1（第 183 行）**：「HeteroGNN 不预训练直接 finetune，合并 AUC 提升到 0.928±0.075」——**正确**。
3. **稿件表 2**：「HeteroGNN(无预训练) | 0.928 | 0.075 | 异构图消息传递」——**正确**。
4. **稿件 §4.1 口径澄清段（第 205 行）**：「0.928±0.075 对应不经预训练的异构图神经网络」——**正确**，且这段"一一对应协议口径"写得很好。
5. **稿件 §6.1（第 401 行）**：「关于 RQ1……**异构图对比预训练+finetune** 在非法对持牌二元任务上提供合并 AUC **0.928±0.075**」——**错误**。该数字属于无预训练的 HeteroGNN；HeCo 预训练+微调应为 0.932/0.934。
6. **稿件英文摘要**：句序为「The effective boundary of heterogeneous-graph contrastive pre-training with HeCo and supervised fine-tuning is evaluated through 80 ... experiments ... The pooled AUC reaches 0.928±0.075」，中文摘要同构。**在缺少模型限定语的情况下，读者只会把 0.928 读作 HeCo 方法栈的成绩**——与 §6.1 的错误一致，且摘要是被引用最多的部分。

**影响评估**：这不是无害笔误。稿件的 RQ1 论证链是"高 AUC → 但家族级失败 → 故只学到行业级差异"。若把无预训练基线的分数当作方法栈成绩写入结论章与摘要，则 (a) 掩盖了"HeCo 预训练相对 HeteroGNN 仅带来 +0.004～+0.006"这一对 §4.1"约 3–4 个百分点提升"叙事更不利的事实（0.894→0.928 的提升来自图结构，0.928→0.932 才是 HeCo 预训练的贡献，约 0.4 个百分点）；(b) 使摘要中"HeCo 有效边界"的量化锚点错位。

**作者需做**：统一采用 §4.1 第 205 行的口径，改写 §6.1 RQ1 首句为"HeteroGNN(无预训练) 0.928±0.075、HeCo 预训练+微调 0.932±0.072/0.934±0.074"；摘要须显式标注 0.928 所属模型，或直接改用 0.932。并在 §4.1 明确"HeCo 对比预训练相对 HeteroGNN 的净增益约为 +0.004～+0.006（5 seed，std≈0.07），在本样本量下不可与噪声区分"——这一诚实表述反而更契合本文定位。

---

## Reviewer 1 —— 侧重：技术可靠性与统计推断合法性

**总体评价**：本稿的负结果主干（RQ2）在统计上是本文最可靠的部分，作者对 bootstrap 分辨率上限的自省与补做精确检验值得肯定，24 行 Holm 记录可逐行复现。但机制诊断章（第 5 章）的统计声明存在三处不可接受的失真，且主结果配置与开源数据存在硬冲突。

**谁会关心本结果、为什么**：网络治理与情报分析实务方（需要知道"跨辖区通用模型不可行、本地少样本可行"这一工程判断）；图域适应方法学社群（小样本异构图上的失效模式）；关注 n=5 设定下统计推断合法性的方法学读者。

**主要优点**
1. §4.4 的"9 对显著"三类分解（5 对方法间互比 / 2 对 source_only 反超 / 2 对退化伪优）逐行与 `supp_holm_24row_record.csv` 吻合，是本稿最好的一段推断写作。
2. 主动指出并**更正**了早期版本对 T2_ON×DANN 方向的误读（§4.4 第 258 行），并在 §4.2/§4.3/表 4 中一致按 1−mpi 报告——这是罕见的自纠。
3. 补做精确符号翻转置换检验（`analysis/supp_exact_significance.py`，32 种符号分配、双侧下限 2/32=0.0625），并据此把 few-shot 结论的依据从"校正后显著性"降级为"效应量+跨 seed 方向一致性"——统计上正确且诚实。
4. 单元测试覆盖 bootstrap、Holm、精确置换、LogME、家族模式匹配（32 项），可在无图数据条件下运行。

**主要问题**

**[M1] Major / Blocking — 表 7 表注宣称的 Holm 校正在数据中不存在，且关键负迁移结论过不了校正**
定位：表 7 表注、§5.2、§1.3 第二段（贡献声明）
证据：`results\week8\metrics\E1_edge_ablation_summary.csv` 仅含 `p_value`、`significant_05`、`significant_01`，**无任何 Holm 列**。文件共 20 个检验（4 情境×5 边型），稿件所列 6 个"显著"效应恰为**未校正** p<0.05 的 6 行。其中 T1_DK×registered_via 的 p=0.028，Holm 第 5 位阈值为 0.05/16=0.0031，**不显著**。
后果：§1.3 把"registered_via 边在丹麦/菲律宾两个目标域上呈现**统计显著的**负迁移效应"列为本文第一项贡献；丹麦侧这一半没有校正后统计支持。
作者需做：(a) 或补做并释出 20 个检验的 Holm/BH 校正列，据实修改表 7 表注与显著性判定；(b) 或删除"霍尔姆校正后"字样，全表改标"未校正 p 值，仅作描述性证据"，并把 §1.3、§5.2、§6.1 中"统计显著的负迁移"改为"5 seed 方向一致的负向效应（未校正 p=0.028，在 20 次比较的 Holm 校正下不显著）"。同时须与 §5.4 自述的"p 值分辨率上限约 1/2⁵≈0.031"对齐——p=0.028 已低于该自述下限，两处相互矛盾。

**[M2] Major / Blocking — 主结果 0.932 与稿件声明的分类头配置冲突；同 split 下 mlp_64 头结果更高却全文未报**
定位：§3.3.2、附录 B、表 2、§4.1
证据：`results\week6\metrics\pooled_primary_finetune_summary.csv` 中 `heco_disc_lr_finetune = 0.9321514888634772`；`results\week6_head_ablation\metrics\head_ablation_summary.csv` 中 `attention / heco_attention_head = 0.9321514888634772`（**16 位小数完全相同**），而 `mlp_64 / heco_mlp_64_head = 0.9705483583±0.0155`、`fixed_mean = 0.9735±0.0163`、`linear = 0.9475`。两组实验的划分审计文件（`results\week6\audits\pooled_primary_finetune_split_audit.csv` 与 `results\week6_head_ablation\audits\head_ablation_split_audit.csv`）逐 seed 的 train/val/test 行数、组数、标签构成**完全一致**——即同 split 可比。
后果：稿件 §3.3.2 与附录 B 声明"最终的'主结果'配置统一使用 mlp_64"，但表 2 报出的 0.932/0.934 在数值上是 attention 头的结果；而按稿件声明的 mlp_64 头，同一划分下应为 **0.9705±0.0155**（std 仅为报出值的 1/5）。这同时意味着 §4.1 "提升约 3 至 4 个百分点（0.89→0.93）"的叙事、§5.4 第三项限制中"seed 方差极大"的论据，都建立在一个稿件自称已弃用的头上。
作者需做：逐 seed 澄清 `pooled_primary_finetune` 与 `head_ablation` 两套运行的关系，明确表 2 各行实际使用的 head；若主结果确为 mlp_64，须替换为 0.9705±0.0155 并重写 §4.1 提升幅度与 §5.4 方差论述；若确为 attention 头，须修正 §3.3.2 与附录 B 的方法描述，并在正文报出完整 head 消融表（6 行），说明为何弃用表现更好、更稳定的 head。**无论哪种情形，当前版本都存在方法描述与结果不匹配。**

**[M3] Major / Non-blocking — 0.980 的"±0.000"与"n=33"均无数据支持，且掩盖了一个 0.721 的离群 seed**
定位：表 2 第 7 行、§4.1、§5.5、§6.1、摘要相关表述
证据：`results\week6\metrics\cross_verified_pooled_eval.csv` 逐 seed AUC = 1.0 / 1.0 / 1.0 / 1.0 / **0.7209**；`pooled_5_seed` 行 AUC=0.9798、`num_rows=234`、`num_licensed=210`、`num_cross_verified=24`。0.980 是**把 5 个 seed 的预测合并后计算的单一 AUC**，本身不存在标准差；5 个 seed 的均值/标准差实为 0.944/0.112(ddof=0)。稿件的"n=33"是数据集中 cross_verified 的**总数**，而非评估队列规模。仓库权威表 `results\week10\metrics\final_rq_evidence_table.csv` 也只记录 0.9798 而不给 std。
作者需做：表 2 该行改为"pooled(5 seed 合并预测) AUC=0.980；逐 seed 1.00/1.00/1.00/1.00/0.72；评估队列 24 个 cross_verified vs 210 个 licensed（5 seed 测试集合并）"，删除"±0.000"，并在 §4.1 明确披露 seed46 的 0.721。当前写法使一个由 4 个满分 seed + 1 个明显失败 seed 构成的结果看起来像零方差的稳定结论，这与全文"诚实证据边界"定位直接冲突。

**[M4] Major / Blocking — §4.3 的"+0.271 诚实声明"本身是错的，且方向相反**
定位：§4.3 第 251 行
证据：复算 `analysis\week11_acceptance.py`（L120–145）：`max_improvement` 仅在 `P2_method_comparison.csv` 中 `result_origin=="week11_P2"` 的行上计算，而该子集**只包含 eerm_K2(20)、eerm_K4(20)、irm(5)**，不含 dann/strurw/dann_strurw（后者 `result_origin=="existing_week7"`）。逐行重算得最大值 **0.27083333 = IRM 在 T2_PH、seed 44**（0.812500 vs source_only 0.541667）。而稿件称该值"来自 DANN+StruRW 在 T2_PH 上的结果（从 0.642 提升至 0.746）"——该差值为 **+0.104**，与 0.271 不符。
后果：这是稿件用来展示"诚实证据边界"框架有效性的**旗舰案例**（§4.3 明确写"本论文不会用这个 +0.271 数字宣传 IRM/EERM 成功，而是把它如实归功于前期已有方法"）。实际情况恰恰相反：该数字**就是 IRM 的**，只不过是**单 seed 极大值**；IRM 的 5 seed 均值 Δ 为 −0.079。真实的诚实表述应是"H_P2a 这一预声明假设以单 seed 最大改进量为统计量，因而被 IRM 在 seed44 的一次偶然高分判为 pass；该统计量本身设计不当，不能作为方法有效性证据"。
作者需做：按上述事实重写该段；同时在 §5.5 与附录 A 中说明 H_P2a 的判定统计量（单 seed 最大值）存在设计缺陷，并说明这是"预注册框架的一个已识别失效模式"。这样处理反而会**强化**第三项贡献的可信度；维持现状则使"诚实证据边界"框架的示例本身不诚实。

**[m1] Minor / Non-blocking — 标准差口径未声明（ddof=0）**
证据：`results\week7\metrics\transfer_method_summary.csv` 中 T2_PH source_only std=0.122474，而按样本标准差(ddof=1)重算为 0.136931（=0.122474×√(5/4)），后者与 `P3_full_fewshot_matrix.csv` 的 0-shot std 一致。即全文所有 ± 值为**总体标准差**，在 n=5 下比样本标准差小约 12%。
作者需做：在 §3.3.2 或附录 B 声明 std 定义；n=5 时建议改报样本标准差或直接给逐 seed 值。

**[m2] Minor / Non-blocking — §4.4 引用的是未校正 bootstrap p 值，却写在 Holm 结论句里**
证据：`supp_holm_24row_record.csv` 中 T2_PH source_only vs DANN 的 `bootstrap_p=0.0010`、`holm_adjusted_p=0.0170`；T2_ON 对应行 0.0012 / 0.0192。稿件写"p=0.001"。
作者需做：统一报 `holm_adjusted_p`，或标注"原始 p / 校正后 p"两值。

**[m3] Minor / Non-blocking — T1_DK 的主指标是召回率，多处被称作"ΔAUC"**
定位：§4.3（"EERM 自身最大 ΔAUC 是 +0.032(EERM K=4 on T1_DK)"）、§6.1、表 4 表头。
作者需做：改为"Δ主指标"或分情境标注指标名。

**技术失败点（在作者论点成立前必须解决）**：M1、M2、M4 三项；其中 M2 直接动摇表 2 全部主结果的可解释性。

**推荐姿态**：Major Revision（对本领域专业期刊）；在 Nature 系标准下，统计层面尚未达到可评估显著性的门槛。

---

## Reviewer 2 —— 侧重：证据与结论匹配度、机制诊断可信度、图表与正文一致性

**总体评价**：第 5 章被作者定位为"主要差异化贡献"，但恰恰是核验失分最集中的一章。三处问题——表 6 的 C3 语义无据、T2_ON 的方向倒置、去词法归因的过度解读——共同削弱了"边界识别"的可信度。相比之下，第 4 章的负结果证据链是干净的。

**谁会关心本结果、为什么**：跨境监管协作的政策制定者（"共享 5–10 个校准样本而非原始数据库"是可直接落地的建议）；做 shortcut learning / spurious correlation 诊断的机器学习读者（ccTLD 反转是一个漂亮的真实世界案例）。

**主要优点**
1. ccTLD 主导性发现（T3_FR：lex_only 0.953 vs graph_only 0.713，差 24 个百分点）经 `final_rq_evidence_table.csv` 独立记录（`lex_only_minus_graph_only_auc=0.23957`）确认，且与法国监管要求 .fr 域名的制度事实构成合理机制解释。这是本文最有价值的单点发现。
2. T2_ON 的 trivial classifier 退化识别（StruRW/DANN+StruRW 把 58 个 licensed 全判为 lic，mpi≈0.000）——把"表面高分"拆解为退化伪优，是很好的审稿级自查。
3. SupCon 在 LOFO 协议下的结构性信号反转（0.622→0.126，score_gap 由 +0.054 翻转到 −0.210），并用 A4 shuffled-label 对照（0.308/−0.018）证明反转来自 SupCon 而非评估协议——设计正确，全部 8 个数逐位可核验。
4. §4.6 对照组跨源评估直面黑白名单偏差，且**同时报出失败的 H_E4b/H_E4c**（`E4_embedding_distances.csv` 5 seed 全 False），区分"判别可分"与"表征分离"——这是标准很高的处理。

**主要问题**

**[M5] Major / Blocking — 表 6 的 C3 行语义与开源配置不符，且数值取自作者自己判定需修正的运行组**
定位：表 6 第 3 行、表 6 表注、§5.1 第 297 行
证据：`configs\week8.yaml` L177 明确 `C3_no_website_lexical: {feature_mode: zero_website_lexical, edge_mode: full}`，即**移除全部 Website 词法特征**；`src\explain\structure_vs_lexical.py`、`analysis\week10_common.py`、`analysis\week8_patch_corrected_e5.py` 中该配置名一致。仓库中**不存在**任何"仅保留 ccTLD 与 dot_com 两维"的配置。稿件表 6 却把该行标为"C3 仅留 ccTLD"并加注"C3 保留 ccTLD 与 dot_com 两维词法特征"。
更严重的是数值来源：稿件明写"表 6 中 C3 配置的数值取自 E5 消融的**原始配置组**"，而原始组的 C3_no_lex（T3=0.953477）在数值上几乎等于 C5_lex_only（0.952999）、而非 C4_graph_only（0.713429）——这正是该运行被判定需修正的症状；修正组 `E5_structure_vs_lexical_corrected_summary.csv` 与权威表 `final_rq_evidence_table.csv` 中，同一条件（`C3_no_website_lexical`）的 T3 值为 **0.713**。
后果：主表的一整行数据来自已知有缺陷的运行，并被赋予一个开源代码不支持的语义标签。作者虽在正文披露"两份 E5 文件的 C3 数值不同"，但披露的是**文件差异**，未披露**该行语义是重新解释的**。
作者需做：删除表 6 的 C3 行，或用修正组数值（T3=0.713 / T2_PH=0.598 / T2_ON=0.426 / T1=0.045）并恢复其真实语义"移除全部 Website 词法"；若确实想报"仅留 ccTLD"，须新增该配置、重跑、释出 config 与 CSV。当前写法无法通过任何数据可用性核查。

**[M6] Major / Blocking — §5.1 对 T2_ON 的结论方向倒置，且表 6 违反本文自己声明的统一方向约定**
定位：§5.1 第 304 行、表 6 T2_ON 列、表 6 表注
证据：`E1_edge_ablation_summary.csv` 中 T2_ON 的 `metric = mean_pred_illegal_lower_is_better`、`primary_metric_direction = lower_is_better`；表 6 的 T2_ON 列数值（0.303/0.295/0.372/0.426/0.323）与 `E5_structure_vs_lexical_summary.csv` 的原始 mpi 一致，**未按 1−mpi 转换**。而 §3.3.4 第 172 行明确声明"为使全文方向一致，**各表与各图**对该情境统一报告 1−mean_pred_illegal（数值越高越好）"。
后果：§5.1 据此写"T2_ON 上图结构明显主导——graph_only(0.426) 显著高于 lex_only(0.323)"。按正确方向，graph_only 的 1−mpi = 0.574，lex_only = 0.677——**词法配置更好，图结构更差**，结论完全相反。这与 §4.4 作者已自行更正过的 T2_ON×DANN 误读是**同一类错误的复发**，说明方向统一没有做全表扫描。
另需注意同表 T1_DK 列：C5_lex_only=0.355 远高于 C1_full=0.206（召回，越高越好），即在 T1_DK 上**加入图结构反而显著损害目标域表现**。这一对本文核心论点极为相关的观察，稿件完全未提。
作者需做：(a) 表 6 T2_ON 列全部改为 1−mpi 并在表注声明；(b) 重写 §5.1 关于 T2_ON 的段落，改为"T2_ON 上词法配置优于纯图结构"；(c) 补充讨论 T1_DK 上 lex_only > full 的现象；(d) 对全文所有表/图做一次方向一致性审计并在附录附上审计清单——这本应是"诚实证据边界"框架的自动产物。

**[M7] Major / Non-blocking — 表 7 的一处置信区间与源文件不符**
定位：表 7 第 4 行（T2_ON×registered_via）
证据：CSV 中 `effect_ci = [0.014985, 0.049386]`、`delta_raw_ci = [0.015006, 0.049386]`；稿件写 [+0.019, +0.046]。同表相邻行 T2_ON×hosted_on 的 [+0.020,+0.049] 则与 CSV [0.019869,0.048796] 吻合，可排除系统性口径差异。
作者需做：核对并更正；同时逐行复核表 7 其余数值的取数脚本。

**[m4] Minor / Non-blocking — §5.1 把 AUC 数值直接读作"贡献百分比"**
定位：§5.1 第 302 行"词法特征……就贡献了约 95% 的判别力"
问题：0.953 是 AUC，不是贡献份额；随机基线为 0.5，若要谈份额应以 (AUC−0.5)/(AUC_full−0.5) 计（约 (0.453/0.486)=93%，量级接近但概念不同）。同句"图结构只多提供了大约 3 个百分点"则是 AUC 差值，两者混用。
作者需做：改为"lex_only 已达 0.953，full 仅再增 0.033；graph_only 仅 0.713"，避免"贡献 X%"表述。

**[m5] Minor / Non-blocking — 探针 AUC 存在两个口径，稿件只报高的那个**
证据：`heco_temperature_grid_summary.csv` τ=0.7 单 seed frozen probe = 0.9180（稿件采用）；`heco_probe_summary.csv` 中 `heco_frozen_linear` 4 次运行均值 = **0.8997±0.0187**。另 `ablation_table.csv` 第 6 行记录"HeCo pretrain + full finetune 0.9793（单 seed τ=0.7）"，同样未进入正文。
作者需做：表 2 该行标注"单 seed"，并在注中给出多 seed 均值，或直接改报多 seed 结果。

**[m6] Minor / Non-blocking — 目标域样本量在正文中不一致**
证据：`P2_method_comparison.csv` 显示 T2_ON `target_size=58`、`target_test_size=24`。稿件 §4.2 写"目标域 58 个 ON licensed 全部预测为 lic"，§5.4 写"T2_ON n=24 test"。T2_PH 同理（表 3 写"14 非法+19 持牌"=33，而 test=14，且未说明该 14 例的类别构成）。
作者需做：表 3 增列"目标域全量 / 评估测试集规模 / 测试集类别构成"三栏。AUC 建立在 14 例（T2_PH）与 29 例（T3_FR）上，测试集类别构成是读者判断可信度的必需信息。

**推荐姿态**：Major Revision。第 5 章需要一次彻底的口径与标签重做。

---

## Reviewer 3 —— 侧重：新颖性、科学重要性、跨学科读者、可复现性

**总体评价**：选题真实、问题重要、写作坦率，负结果定位在原则上成立。但"新颖性"主要体现在**场景与数据**而非方法或原理；"重要性"受限于 613 个站点的单一冻结图与 transductive 评估。可复现性在**代码与聚合指标层**做得相当好，但在**审计文件与图件层**存在明确的失实陈述。

**谁会关心本结果、为什么**：网络犯罪治理与跨境执法协作研究者；把 GNN 用于安全风控的工程团队（会关心"小样本异构图上 DA 方法集体失效"这一负面经验）；元科学/研究规范社群（对"可执行预注册审计"这一提法感兴趣）。**不会**吸引一般性跨学科读者——问题设定、数据与结论都高度域内化。

**主要优点**
1. 数据集构建扎实：11 辖区、5 层证据分级、互斥角色定义、125 项自动验收后冻结 `hetero_graph_v2.pt`，节点/边统计与 `data\node_stats.csv`、`data\edge_stats.csv` 完全对得上（4,006 边可逐类相加复核）。
2. 明确记录并**保留**了数据构建期的三个质量问题（131 条 redirects_to 自环、4 个 label_* 特征泄漏列、38 个孤立节点）——`docs\Week3_graph_build_summary.md` 可佐证 38 这一数字。
3. §3.2 关于 ExternalReference 节点"不编码标注来源身份"的说明，直面了最容易被审稿人攻击的泄漏路径，写得具体（2 维渠道类型二值编码 + inverse_dst_degree 降权）。
4. 附录 C 的 21 个脚本/文档**逐一存在**；`tests/` 的 32 项单元测试确为 32 个 `def test_`，且可在无原始图数据条件下运行——这是负责任的释出设计。
5. §5.4 三项限制（采样偏差、transductive、n=5）的自陈是同类稿件中少见的清醒；特别是对 inductive 差距"在获得实测值之前不给具体数字"的克制，值得肯定。

**主要问题**

**[M8] Major / Blocking — 稿件点名承诺的关键审计文件在发布包中不存在**
定位：§4.5 第 284 行
证据：稿件写"每个 seed×shot 设定的实际抽样记录保存在 `audits/P3_fewshot_T1_Nordic__seed*__shots*_split.csv`，共 **20 份**审计文件，用以二次审计无 cross_verified 泄漏到训练"。全仓库检索该模式，**匹配 0 份**。
后果：T1_DK 少样本结果（0.206→0.594）的核心防泄漏保证目前**无法被第三方验证**；而 T1_DK 训练池与测试池均含 cross_verified 高质量样本，正是稿件自己识别的污染风险点。
作者需做：补充释出这 20 份文件；若因安全原因不能释出站点级抽样记录，则改为释出脱敏的计数级审计（每 seed×shot 的 cross_verified 命中数=0 的汇总表），并相应修改正文措辞。**当前写法是对不存在材料的具体承诺，属可用性声明失实。**

**[M9] Major / Non-blocking — §3.3.2 的 attention collapse 具体数值与释出审计相反**
定位：§3.3.2 第 148 行
证据：`results\week6_head_ablation\audits\attention_collapse_audit.csv` 中 seed46 的 `metapath_attention_mean = 0.30688`，5 个 seed 的 `attention_collapse_flag` **全部为 False**。稿件称"该 head 在 seed46 上出现 attention collapse（metapath_attention 降至 0.023）"。
后果：这是稿件用来论证"弃用 attention head、改用 mlp_64"的唯一理由；与 M2 合看，该理由既不被数据支持，且被弃用的 head 恰是主结果数值的来源。
作者需做：给出 0.023 的具体来源文件与 seed，或删除该数值并改述；同时正文报出完整 head 消融（6 行，含 `attention_collapse_flag` 列）。

**[M10] Major / Non-blocking — 图件缺失，图—文一致性完全不可核验**
定位：图 1–图 8 全部占位符；§4.1 图 3、§4.2 图 4、§4.5 图 5、§5.1 图 6(a)、§5.2 图 6(b)/图 7、§5.3 图 7/图 8
证据：`figs\` 仅有 `fig4_pooled_auc.pdf`；`results\week10\figures\` 有 4 个 PDF（fig7–fig10），`results\week11\plots\` 有 4 个 PNG，但与稿件图号无对应关系。另发现**图号自身混乱**：第 320 行标 `[图7 位置]` 而紧随的图题是"图 6(b)"；第 369 行标 `[图8 位置]` 而图题是"图 7"。
作者需做：提交全部图件；统一图号与占位符；提供图—源 CSV 的对照表（仓库已有 `results\week10\audits\table_figure_crosswalk.csv` 的机制，扩展到论文图号即可）。

**[m7] Minor / Non-blocking — "1,500 余次实验运行"不可核验**
证据：已释出的 raw_runs 聚合 CSV 行数合计仅数百（E1=120、E5=100、W9 曲线=150、P3=100、P2=90 等）；`runs/`、`logs/`、`predictions/`、`splits/` 目录在作者自查文件 `reports\rerun_2026-07-18\rerun_final_artifact_audit.csv` 中即标为 `fail: missing`。
作者需做：给出该计数的构成表（按阶段×实验×配置×seed 相乘），或改为"已释出聚合指标覆盖 N 次运行"。

**[m8] Minor / Non-blocking — 编码器检查点来源在 seed 间不一致**
证据：`pooled_primary_finetune_raw_runs.csv` 中 seed42 的 `encoder_source = existing_checkpoint`（指向 week5 检查点），seed43–46 为 `week6_pretrain`；`P1_family_metric_lofo.csv` 同样是 seed42 用 week5、其余用 week6。
作者需做：说明两批检查点是否由同一预训练配置产出；若不同，5 seed 的"重复"性质需重新界定。

**[m9] Minor / Non-blocking — §6.4 数据可用性声明留有大段占位**
定位：第 422 行「【作者补充：仓库URL/DOI、发布标签与提交号、存档SHA-256、受控访问申请邮箱与审批责任人】」
说明：按本次审查范围，DOI 等出版社定调内容不评判；但**仓库永久标识、发布标签、存档校验值、受控材料申请流程**属于科学可复现性而非出版流程，应在评审阶段即可提供。
作者需做：在返修稿中补齐永久标识与校验值（可用匿名化的归档快照）。

**新颖性与重要性判断**
- **Originality（中等偏下）**：方法栈（HeCo / DANN / StruRW / IRM / EERM / SupCon / LogME）全部为既有工作的直接应用，作者亦明确声明"不试图提出新 SOTA 方法"。原创性来自 (a) 11 辖区异构图数据集本身，(b) 小样本异构图上 DA 方法集体失效的系统证据，(c) ccTLD 反转这一具体 shortcut 的量化。(c) 是真正新的域内知识。
- **Scientific importance（中等）**：结论"跨辖区通用模型不可行、本地 5–10 样本校准可行"对监管协作有直接操作价值。但外部效度受限：单一冻结图、11 辖区中仅 3 个双类完整、n=613、transductive、5 seed、无独立复现。稿件 §6.3 对此已有清醒陈述，态度可取，但不改变限度本身。
- **Interdisciplinary readership（低）**：结论高度绑定于本数据分布，不构成对图学习或域适应理论的一般性结论；对 Nature 系刊要求的"跨学科广泛兴趣"门槛，差距明显。
- **Readability for nonspecialists（中上）**：中文表达清楚，机制解释有制度背景（法国 .fr 监管要求）辅助，非专家可读性优于同类稿件。术语中英混排（"迁移情境""transferability""finetune"）略显杂乱，建议统一首现中英对照、后文单一用法。

**推荐姿态**：对本领域专业期刊 Major Revision；对 Nature 系刊，significance 与 breadth 均不达门槛。

---

## 交叉综合（Cross-review synthesis）

**共识优点**
1. **负结果主干（RQ2）证据可靠且可复算**：24 对方法比较的 bootstrap-Holm 结果（9 对显著）与精确符号翻转置换结果（0 对显著、6 对停在 p=0.0625）逐行与仓库吻合；作者对 n=5 分辨率上限的自省，以及据此把 few-shot 结论降级为"效应量+方向一致性"的处理，统计上是正确的。三位审稿人一致认为这是本稿最有价值、最经得起检验的部分。
2. **自纠记录真实存在**：§4.4 对 T2_ON×DANN 方向的更正、§4.3 对"退化伪优"的拆解、§4.6 对失败的 H_E4b/H_E4c 的照实报出、附录 D 的失败假设清单（与 `week11_acceptance_audit.csv` 的 9 pass / 6 fail 完全一致），都不是装饰性表述。
3. **代码与聚合指标层的可复现性达标**：附录 C 的 21 个脚本 21/21 存在；32 项单元测试属实；节点/边统计、样本口径算术全部自洽。

**共识技术风险**
1. **口径与方向的统一没有做完整扫描**。作者在 §3.3.4 声明了 1−mpi 的全表统一约定，却在表 6 违反（M6），并因此得出与数据相反的结论；这与 §4.4 已更正过的同类错误同源，说明缺少一次机械化的全表方向审计。
2. **主表存在语义无据与配置冲突**。表 6 的 C3 行（M5）与表 2 的 head 归属（M2）分别在"标签"和"配置"两个层面与开源代码/数据冲突，且都发生在最显眼的主表上。
3. **"诚实证据边界"框架的三个示例中有两个失实**：+0.271 的归属（M4）与 20 份少样本审计文件（M8）。框架被列为本文第三项贡献，其示例的可核验性直接决定该贡献能否成立。
4. **显著性词汇被用在未校正的数据上**（M1），且与作者自述的 p 值分辨率下限自相矛盾。

**审稿人间的分歧与权重差异**
- Reviewer 1 认为 **M2（head 配置冲突）** 是最严重问题，因为它使表 2 全部主结果的可解释性存疑，且可能把一个"更差、更不稳定的配置"当成主结果报出。
- Reviewer 2 认为 **M6（T2_ON 方向倒置）** 最严重，因为它是**结论级**错误（写出了与数据相反的机制判断），而 M2 至少是"数字对、标签错"。
- Reviewer 3 认为 **M4 + M8** 最严重，因为它们攻击的是本文自称的方法学贡献本身：一篇以"可执行诚实审计"为卖点的论文，其审计示例失实与承诺文件缺失，是性质更重的问题。
- 三人**一致**认为：这些问题**都不推翻本文的核心负结论**（DA 方法未稳健优于 source_only；家族级识别失败；少样本校准有效）。这一点很重要——问题集中在归属、标签、方向与可用性声明，而非核心证据链。

**广泛兴趣 / 显著性读数**
- 作为**域内负结果与边界识别研究**：立得住，且有实用价值（5–10 样本校准的政策建议、ccTLD 反转的 shortcut 案例、SupCon 在 LOFO 下的结构性不可用）。
- 作为**Nature 系刊候选**：不达门槛。样本量（613 站点）、单一冻结图、11 辖区中仅 3 个双类完整、transductive-only、5 seed、无外部复现、方法全部为既有工作的应用——这些共同限制了结论的一般性与读者广度。这不是对本文质量的否定，而是定位判断。

**在建立强论证前最需解决的问题（按优先级）**
1. M2 — 澄清并统一主结果的 head 配置（表 2 / §3.3.2 / 附录 B / head 消融全表）。
2. M6 — 表 6 方向修正 + §5.1 的 T2_ON 结论重写 + 全文表图方向一致性审计。
3. M5 — 表 6 C3 行的语义与数据来源二选一（删除 / 用修正组 / 新跑配置）。
4. M4 — 重写 +0.271 段落，改为"预注册统计量设计缺陷"的正面案例。
5. M1 — 表 7 的显著性词汇与校正口径对齐。
6. M8 / M9 / M10 — 补齐少样本抽样审计、attention collapse 证据、全部图件。
7. 专项 — 0.928 的模型归属在 §6.1 与摘要中更正。

---

## 风险 / 不被支持的主张清单（Risk / unsupported claims）

**不被开源数据支持的具体主张**
- §6.1「异构图对比预训练+finetune……合并 AUC 0.928±0.075」——0.928 属 `hetero_gnn`（无预训练）。
- 摘要（中英）未限定模型的「合并 AUC 达到 0.928±0.075」——同上，读作 HeCo 成绩即错误。
- 表 2「HeCo 微调(交叉验证子集) 0.980，标准差 0.000，样本基础 n=33」——std 不存在；评估队列为 24 cross_verified + 210 licensed（5 seed 合并）；逐 seed 含一个 0.721。
- §3.3.2「attention_pool head 在 seed46 出现 attention collapse（metapath_attention 降至 0.023）」——释出审计为 0.3069，collapse_flag 全 False。
- §4.3「+0.271……来自 DANN+StruRW 在 T2_PH 上的结果（从 0.642 提升至 0.746）」——实为 IRM 在 T2_PH seed44 的单 seed 改进；0.746−0.642=+0.104。
- 表 6「C3 仅留 ccTLD（保留 ccTLD 与 dot_com 两维）」——仓库无此配置；`configs/week8.yaml` 定义为"移除全部 Website 词法"。
- §5.1「T2_ON 上图结构明显主导——graph_only(0.426) 显著高于 lex_only(0.323)」——方向倒置，实际结论相反。
- 表 7 表注「霍尔姆校正后 p<0.05」——E1 文件无 Holm 列；T1_DK×registered_via 的 p=0.028 在 20 次比较的 Holm 下不显著。
- §1.3「registered_via 边在丹麦/菲律宾两个目标域上呈现**统计显著的**负迁移效应」——丹麦侧无校正后统计支持。
- 表 7「T2_ON×registered_via 95%CI [+0.019,+0.046]」——源文件为 [+0.015,+0.049]。
- §4.4 与附录 E「6 对……其中不含任何迁移方法优于 source_only 的组合」——6 对中有 2 对（T2_ON 上 StruRW、DANN+StruRW 相对 source_only）确实是迁移方法胜出，尽管作者在别处正确地将其归为退化伪优；§6.1 用的"非退化"限定语应回填到此处。
- §4.5/附录 E「9 档……（T2_PH 与 T1_DK 的全部档位）」——第 9 档为 T2_ON 10-shot。
- §4.5「共 20 份审计文件」——发布包中 0 份。
- §4.6「S3 为 0.969」——实测 0.968。

**不可评估的项目（材料缺失，非错误）**
- 图 1–图 8 与正文/表格的一致性（无图件）。
- 「1,500 余次实验运行」的总数（运行记录目录未释出）。
- T1_DK 少样本抽样池"44 条 single-tier"与"测试集 31 条"相对于丹麦 77 个 illegal 的完整划分（划分审计未释出）。
- 端到端复现（原始图与检查点未释出，属作者声明的安全豁免范围，本身合理）。
- 38 个孤立节点的"意大利 26 + 法国 12"分辨（文档仅记录总数 38）。

**表述层面的过度断言（非数据错误，但需收紧）**
- 「词法特征就贡献了约 95% 的判别力」——把 AUC 读作贡献份额。
- 「5–10 个目标域已确认样本即足以让模型从'几乎无用'恢复到'基本可用'」——该结论在 T2_PH（test n=14）与 T1_DK 上成立，在 T3_FR 上 5-shot 反而失稳（0.886±0.242），§6.2 的政策建议应带上"取决于目标域类别完整性与是否已饱和"的限定，正文 §4.5 有此限定但 §6.2 第一条丢失了。
- 「这套元方法在情报学小样本研究中具备可推广性」——在其自身示例（+0.271、20 份审计文件）失实的前提下，可推广性主张应暂缓。

---

## 录用建议

**本委员会建议：Major Revision（重大修改后再审）**

**理由**
1. **不建议 Reject**：核心科学内容成立。负结果（四类 DA 方法均未稳健优于 source_only）在 bootstrap-Holm 与精确置换两种口径下一致，逐行可复算；家族级识别失败（0.622 / tsars 0.386）、SupCon 在 LOFO 下的结构性反转（0.622→0.126，含 shuffled-label 对照）、少样本校准的三种恢复模式、对照组跨源评估（0.970±0.013）——这些主干证据全部经核验相符。数据集与代码释出质量在同类工作中属上乘。
2. **不建议 Minor**：问题不是文字打磨。共 10 项核验不符，其中 4 项落在主表（表 2 head 归属、表 6 C3 语义、表 6 方向、表 7 校正口径），2 项落在结论章与摘要（0.928 归属），2 项攻击本文自称的第三项贡献（+0.271 归属、20 份审计文件缺失），1 项产生了与数据相反的机制结论（T2_ON）。修正这些需要重跑或重取部分数值、重写 §5.1、重做全文方向审计、补充图件与审计文件——工作量与影响面均超出 Minor 范畴。
3. **不建议 Accept**：当前版本的主表存在方法描述与释出数据的硬冲突（M2、M5），在这一点澄清前，表 2 与表 6 不具备可引用性。

**返修必做清单（Blocking 项）**：M1、M2、M4、M5、M6、M8 + 0.928 归属专项。
**返修应做清单（Non-blocking 但影响评价）**：M3、M7、M9、M10 + m1–m9。
**再审时的核查方式**：建议要求作者随返修稿提交一份"论文数值—源文件—行号"的逐条对照表（仓库已有 `table_figure_crosswalk.csv` 与 `metric_consistency_audit.csv` 的机制，扩展到论文中出现的每一个数字即可）。这既是对本次问题的直接补救，也正是本文"诚实证据边界"框架应当自动产出的东西——把它做出来，第三项贡献才真正成立。

---

## 独立判断：「负结果 + 应用边界识别」定位是否成立、是否足以支撑发表

**定位成立性：成立，且是本稿最恰当的定位。**

理由有三：
1. **负结果本身达到了可发表的证据强度**。稿件不是"试了一个方法没work"，而是覆盖 4 类方法 × 4 类迁移情境 × 5 seed 的完整组合（80 次运行），并同时用 bootstrap-Holm 与 n=5 精确检验两种口径交叉验证，得到方向一致的否定结论。更关键的是，作者对"表面成功"做了拆解：T2_ON 上 StruRW/DANN+StruRW 接近满分的表现被识别为 trivial classifier 退化；T3_FR 的 0.986 被 ccTLD 消融拆解为词法捷径。**能把假阳性的"成功"拆掉，是负结果论文可信度的核心指标，本稿做到了。**
2. **边界识别提供了正向的知识增量，不只是"失败清单"**。至少三项发现具有独立价值：(a) IRM 的"环境内类别完整"假设在真实司法辖区数据上**结构性不可满足**（11 辖区中仅 3 个双类完整），这不是实现问题而是场景约束，对后续研究有直接指导意义；(b) SupCon 在 leave-one-family-out 协议下产生结构性信号反转（留出家族被推成 OOD，得分被压低），并给出了"all-family pretraining + retrieval-only evaluation"的改进方向；(c) ccTLD 反转作为真实监管场景中的 shortcut 案例。
3. **实务结论可操作且与证据匹配**。"共享 5–10 个本地校准样本而非原始数据库"这一政策建议，直接来自 100 次校准实验，并被作者自己用"取决于目标域类别完整性"加以限定（双类 3–5 shot 饱和、单类需 ≥10、已饱和目标域不建议做）。这是负结果研究少见的、能落到治理实践的产出。

**但定位成立 ≠ 当前稿件足以支撑发表。**

关键区分在于：**负结果的可信度完全依赖于执行的严谨度**。一篇宣称"我们没找到效果"的论文，读者唯一能依赖的就是"作者的测量与报告是可信的"。本次核验发现的 10 项不符——尤其是与数据相反的机制结论（M6）、主表语义无据（M5）、以及"诚实审计框架"自身示例的失实（M4、M8）——恰恰打击的是这个唯一的信任基础。**一篇以"诚实证据边界"为第三项贡献的论文，其审计示例经不起审计，这是定位与执行之间最尖锐的矛盾。**

因此本委员会的判断是：
- **定位应当保留**，不建议作者转向"我们提出了新方法"的包装；
- **第 4 章（负结果主干）已达可发表水准**；
- **第 5 章（边界识别/机制诊断）需要一次彻底返工**——这一章目前是全稿核验失分最集中处，而它恰是作者自称的"主要差异化贡献"；
- **第三项贡献（诚实证据边界框架）在修正 M4、M8 并补出"论文数值—源文件"逐条对照表之前，暂不成立**；修正之后，它反而可能成为本文最有传播价值的部分，因为"预注册统计量设计不当导致假 pass"（H_P2a 用单 seed 最大值作判定量）是一个极好的、真实的元科学案例。

**适配层级的判断**：以 Nature 系刊标准衡量，本稿在 originality（方法均为既有工作应用）与 interdisciplinary breadth（结论高度域内化、绑定单一冻结图）两轴上不达门槛，significance 亦受 n=613、transductive-only、无外部复现所限——这一判断与稿件质量无关，是定位与刊物的匹配问题。作为专业领域期刊的负结果/边界识别论文，在完成上述 Blocking 项修正后，本稿具备发表价值。

---

*本报告仅基于稿件文本与开源仓库中可核验的聚合指标、配置与代码。未进行模型重训练；站点级原始数据、检查点与运行日志未释出，相关结论的端到端复现未做。报告不代表编辑部决定。*
