# Week 8 预声明假设与验收标准(Pre-registered Hypotheses)

> **状态**:跑前冻结(prerun-frozen)
> **冻结时间**:_TBD,在第一个 W8 run 启动之前填写_
> **不可修改条款**:实验启动后,所有阈值(右列)不得调整。失败方向预先声明,允许部分通过部分失败。

---

## 通用记号

- **ΔAUC** = AUC(ablated config) − AUC(baseline config),同 seed 配对
- **95% CI** = paired bootstrap on 5 seed,B=1000 次重采样
- **σ** = embedding 空间各类标准差的均值
- 所有 transfer 配置沿用 W7 v2 splits

---

## 实验 W8-E1:边通道消融

| 假设 ID | 假设 | 通过条件 | 应对失败方向 |
|---|---|---|---|
| H_E1a | T3 (SE+IT→FR) 的高 AUC 主要依赖 hosted_on + uses_cert | -hosted_on 或 -uses_cert 任一配置下 ΔAUC ≥ 0.05 且 95% CI 不跨 0 | 失败 → "成本最小化理论在 T3 上不显著",论文写为局限 |
| H_E1b | T2_PH 的低 AUC 不来自任一通道(信号本身缺失) | 所有 5 种 -edge 配置下,\|ΔAUC\| < 0.03 | 失败 → 某通道意外重要,需在 RQ2 章节单独分析 PH 该通道结构 |
| H_E1c | T2_ON licensed_consistency 主要依赖 uses_ns + registered_via | -uses_ns 或 -registered_via 任一配置下,Δmean_pred_illegal ≥ 0.10 | 失败 → 监管合规集中度假设不成立,改用网络效应理论解释 |

**预声明输出**:`output/week8/metrics/E1_edge_ablation_summary.csv`(columns: transfer_id, edge, delta_mean, ci_lo, ci_hi, p_value, significant_05, significant_01)

---

## 实验 W8-E2:LogME 转移性预筛

| 假设 ID | 假设 | 通过条件 | 应对失败方向 |
|---|---|---|---|
| H_E2a | LogME 与 W7 实测 target_AUC 的 Spearman ρ ≥ 0.6 | ρ_LogME ≥ 0.6,且 p < 0.05 | 失败 → "LogME 在异构图小样本设定下不可靠",写为 RQ3 负向贡献 |
| H_E2b | LogME 优于 H-divergence 与 Wasserstein | \|ρ_LogME\| > \|ρ_H-div\| 且 \|ρ_LogME\| > \|ρ_W\| | 失败 → 经典度量更优,写"经典域适应度量在本场景仍占优" |
| H_E2c | LogME 不仅相关,还能正确排序 transfer 难度 | Kendall τ ≥ 0.5 | 失败 → 仅有相关无排序力,需分箱讨论 |

**预声明输出**:
- `output/week8/metrics/E2_pair_scores.csv`(每 (R_s, R_t, seed) 一行)
- `output/week8/metrics/E2_transferability_metrics.csv`(三种度量与 AUC 的相关系数 + 95% CI)

---

## 实验 W8-E3:家族级泛化(leave-one-family-out)

| 假设 ID | 假设 | 通过条件 | 应对失败方向 |
|---|---|---|---|
| H_E3a | 用 root_domain + Louvain 重建丹麦家族后,LOFO 平均 AUC ≥ 0.85 | LOFO 全 fold 平均 AUC ≥ 0.85,5 seed pooled | 失败 → 家族重建质量影响,需在论文中加注 |
| H_E3b | 移除 tsars 家族训练,在 tsars 上的 AUC ≥ 0.85 | tsars LOFO AUC ≥ 0.85,5 seed pooled | **重大失败 → 回退 W6 主结果声明:cross_verified=0.980 表述为"包含家族 memorization 的混合估计"** |
| H_E3c | 各家族留出后,family 内部 std < 0.10 | 各 size≥2 家族 5 seed std < 0.10 | 失败 → 家族级稳健性弱,改写为"家族级方差较高,需更多家族样本" |

**置换检验**:真实家族 LOFO AUC 应**显著低于** random LOFO AUC(单尾 p < 0.05),否则家族标签未提供新信息。

**预声明输出**:
- `output/week8/metrics/E3_lofo_family_sensitivity.csv`(per-family fold AUC)
- `output/week8/audits/E3_dk_family_assignments.csv`(family extractor 输出)
- `output/week8/audits/E3_permutation_aucs.csv`(100 次置换的 AUC 分布)

---

## 实验 W8-E4:对照组 B/C 引入 + 因果可识别性

| 假设 ID | 假设 | 通过条件 | 应对失败方向 |
|---|---|---|---|
| H_E4a | illegal vs control_legal_commercial 二分类 5 seed AUC ≥ 0.85 | S2 AUC mean ≥ 0.85 | **重大失败 → 模型仅靠"博彩业内细节"工作,RQ2 必须新增"模型边界"章节** |
| H_E4b | 嵌入空间中 d(illegal, control) > d(illegal, licensed) + 0.5σ | 5 个 encoder 中至少 4 个满足此条件 | 失败 → 持牌/普通商业在嵌入空间未严格序,改写"监管逃避痕迹未在嵌入层显现" |
| H_E4c | d(licensed, control) > d(licensed, illegal) | 5 个 encoder 中至少 4 个满足此条件 | 失败 → 三层对照体系内部不严格序,改写为"二元区分"叙事 |

**预声明输出**:
- `output/week8/metrics/E4_three_scenario_summary.csv`(S1/S2/S3 各 5 seed AUC + std)
- `output/week8/metrics/E4_embedding_distances.csv`(5 encoder × 3 距离对)

---

## 实验 W8-E5:结构 vs 语义切片

| 假设 ID | 假设 | 通过条件 | 应对失败方向 |
|---|---|---|---|
| H_E5a | 移除 is_local_tld 后,T3 AUC 下降 < 0.05 | T3 (-ccTLD) AUC ≥ 0.95(对比 baseline 0.986) | **重大失败 → T3 高 AUC 主要来自 ccTLD,RQ2 必须独立章节讨论 Phase8 ccTLD 反转** |
| H_E5b | T3 graph-only 配置下 AUC 仍 ≥ 0.85 | C4_graph_only 5 seed AUC mean ≥ 0.85 | 失败 → 图结构独立工作能力弱,改写为"图结构 + 词法特征联合贡献" |
| H_E5c | T3 graph-only AUC − lex-only AUC ≥ 0.10 | 差距 ≥ 0.10,paired bootstrap 95% CI 不跨 0 | 失败 → 图结构与词法特征贡献相当,需分别量化 |

**附加观察**:E5 也对 T1/T2_PH/T2_ON 跑全配置,结果纳入论文附录,但不计入主验收。

**预声明输出**:`output/week8/metrics/E5_structure_vs_lexical.csv`(5 config × 4 transfer × 5 seed = 100 行)

---

## 总验收摘要

- 总假设数:**15**
- ✅ 必须通过(影响主叙事):H_E3b, H_E4a — 任一失败需重写论文章节
- 🟡 期望通过(影响细节):H_E1a, H_E2a, H_E5a, H_E5b — 失败仅影响某章节论证强度
- ⏳ 可观察通过(锦上添花):H_E1b, H_E1c, H_E2b, H_E2c, H_E3a, H_E3c, H_E4b, H_E4c, H_E5c

**论文诚实性承诺**:无论通过率多少,所有假设结果(包括失败)在论文中公开报告,作为研究的诚实贡献。

---

## 跑前签名

| 角色 | 姓名 | 时间 | 签名 |
|---|---|---|---|
| 提议者(Master) | _________ | _________ | _________ |
| 技术顾问(Raphael) | Claude / Raphael | 2026-04-28 | ✓ |

> 一旦签署完成,本文档锁定。Week 8 实验启动后,任何阈值修改请求必须在论文中以"事后调整(post-hoc adjustment)"显式标注。
