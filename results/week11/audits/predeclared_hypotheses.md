# Week 11 预声明假设与验收标准(Pre-registered Hypotheses)

> **状态**:跑前冻结(prerun-frozen)
> **冻结时间**:_TBD,在 W11 第一个 run 启动之前填写_
> **不可修改条款**:实验启动后,所有阈值不得调整。失败方向预先声明。
> **激进路线 B**:5 个 patch / 15 个假设,期望 7.5 → 8.5-9.0
> **最坏情况保底**:P4/P5 零失败风险,保证至少 +0.2 提升
> **最好情况上限**:全 15 通过 → 9.0/10

---

## 通用记号

- **ΔAUC** = AUC(改进 config) − AUC(baseline config),同 seed 配对
- **95% CI** = paired bootstrap on 5 seed,B=10000 次重采样
- **Holm-Bonferroni** = FWER ≤ 0.05 的多重比较校正
- 所有 transfer 配置沿用 W7 v2 splits

---

## Patch P1:Family-aware Metric Learning Head 🔴 最大杠杆

> **杠杆性**:🔴 极高 | **风险**:中 | **期望提升**:+0.5 ~ +0.6

### 假设清单

| ID | 假设 | 阈值 | 关键性 | 应对失败方向 |
|---|---|---|:---:|---|
| **H_P1a** | 7 family hard-negative mean AUC | ≥ 0.75(vs W9 baseline 0.622)| 🔴 | 失败 → 论文写"family metric learning 改善有限",仍为 RQ1 边界证据 |
| **H_P1b** | tsars hard-negative AUC(关键 case)| ≥ 0.70(vs W9 0.386)| 🔴 | 失败 → 单独写 tsars 作为最难 case,但保留其他 family 提升 |
| **H_P1c** | mean AUC vs W9 baseline 提升 | ≥ +0.10 | 🔴 | 失败 → patch 整体不写入论文,作为 negative result 在 appendix |
| **H_P1d** | Ablation A4(family 打乱)gap vs A2(主提案)| A2 - A4 ≥ +0.05 | 🟡 | 失败 → 信号是 spurious correlation,不能宣称 family-aware |

### 预声明输出

- `output/week11/metrics/P1_family_metric_lofo.csv`(7 family × 5 seed × test_form)
- `output/week11/metrics/P1_alpha_ablation.csv`(A1/A2/A3/A4 × 2 family × 5 seed = 30 run)
- `output/week11/metrics/P1_family_centroid_distance.csv`(5 encoder × 7 family centroid pairwise)
- `output/week11/plots/P1_family_tsne.png`(5 seed 选 1 个 representative)

### 关键性判定

H_P1a + H_P1b + H_P1c 同时通过 → P1 大胜利,论文 RQ1 章节质变,**全方位补 RQ1 死穴**

---

## Patch P2:IRM + EERM 实战对比 🟢 高杠杆,无失败风险

> **杠杆性**:🟢 高 | **风险**:低 | **期望提升**:+0.2 ~ +0.3

### 假设清单

| ID | 假设 | 阈值 | 关键性 | 应对失败方向 |
|---|---|---|:---:|---|
| **H_P2a** | IRM/EERM 在任一 transfer 上 vs source_only ΔAUC | ≥ +0.05 | 🔵 | 通过 → 真胜利,论文加为 method 章节 |
| **H_P2b** | IRM/EERM 在所有 transfer 都不显著(零失败假设)| 任何结果都对论文有利 | 🔵 | 通过 → 论文写"小样本异构图的方法学局限",**本身即贡献** |
| **H_P2c** | EERM K=2 vs K=4 ΔAUC range | ≤ 0.10 | 🟡 | 失败 → 写为 K-超参敏感性分析 |

### 关键属性

**P2 是 zero-loss patch**:无论 H_P2a 通过(SOTA 方法对比)还是 H_P2b 通过(方法学边界发现),**论文都受益**。

### 实测约束声明

| Transfer | IRM 适用性 | EERM 适用性 |
|---|---|---|
| T1_Nordic | ✗(IT/ES/SE 全单类)| ✓ |
| T2_PH | 限定(仅 BE+FR)| ✓ |
| T2_ON | 限定(仅 BE+FR)| ✓ |
| T3_DiagnoseFrance | ✗(IT/ES 全单类)| ✓ |

**IRM 仅在 T2_PH 上跑,EERM 全 4 transfer 上跑**。这本身是 RQ3 的方法学发现:**真实地区天然单类,IRM 不适用,EERM 是更通用的 substitute**。

### 预声明输出

- `output/week11/metrics/P2_method_comparison.csv`(5 method × 4 transfer × 5 seed)
- `output/week11/metrics/P2_eerm_K_sensitivity.csv`(EERM K=2,3,4 × 4 transfer)
- `output/week11/plots/P2_irm_convergence.png`

---

## Patch P3:全 transfer few-shot 矩阵 🟡 中杠杆

> **杠杆性**:🟡 中 | **风险**:极低 | **期望提升**:+0.15 ~ +0.2

### 假设清单

| ID | 假设 | 阈值 | 关键性 | 应对失败方向 |
|---|---|---|:---:|---|
| **H_P3a** | T1_DK 5-shot illegal_recall(vs 0.21 baseline)| ≥ 0.50 | 🔴 | 失败 → 写"one-class target few-shot 有限" |
| **H_P3b** | T2_ON 5-shot mean_pred_illegal(保持验收)| < 0.30 | 🟡 | 失败 → 写"licensed-only target few-shot 干扰原边界" |
| **H_P3c** | 4 transfer 中 ≥ 3 个 recoverable | ≥ 3/4 | 🟡 | 失败 → 论文 RQ4 仍写为 "target-dependent" |

### "作弊"防护(关键设计约束)

T1_DK 测试集中含 cross_verified 33 条 高质量样本,**强制约束**:
- few-shot 抽样池 = T1_DK 训练集中的 single-tier illegal(44 条),**不抽 cross_verified**
- 评估集 = T1_DK 测试集中的 cross_verified + single 混合 31 条,**不被 few-shot 触碰**
- 用 group-aware split(operator-level)防止泄漏

### 预声明输出

- `output/week11/metrics/P3_full_fewshot_matrix.csv`(2 transfer × 5 shots × 5 seed = 50 + 已有 T2_PH/T3 的 50 = 100 行)
- `output/week11/plots/P3_fewshot_curves.png`(4-transfer 2×2 panel)

---

## Patch P4:Metric-level 统计检验补全 🟡 必做项,零失败风险

> **杠杆性**:🟡 中 | **风险**:零 | **期望提升**:+0.10 ~ +0.15

### 假设清单

| ID | 假设 | 阈值 | 关键性 | 应对失败方向 |
|---|---|---|:---:|---|
| **H_P4a** | W7 method 24 配对中显著 (Holm 校正 p<0.05)| ≥ 1/24 | 🔵 | 任何结果都补齐统计严谨性 |
| **H_P4b** | W9 few-shot 8 配对中显著 | ≥ 4/8 | 🔵 | 失败 → 仍贡献统计稳健性 |

### 检验设计

- **Paired bootstrap**(B=10000)
- **Wilcoxon signed-rank test**(非参,5 seed 适用)
- **Holm-Bonferroni 多重校正**(FWER ≤ 0.05)
- **零重训成本**:纯事后分析

### 预声明输出

- `output/week11/metrics/P4_w7_method_pairwise.csv`(24 配对)
- `output/week11/metrics/P4_w9_fewshot_significance.csv`(8 配对)
- `output/week11/metrics/P4_w11_overall_significance.csv`(包含 P1-P3 新结果)

---

## Patch P5:论文章节深化 🟢 高杠杆,零失败风险

> **杠杆性**:🟢 高 | **风险**:零 | **期望提升**:+0.15 ~ +0.2

### 假设清单

| ID | 假设 | 阈值 | 关键性 | 应对失败方向 |
|---|---|---|:---:|---|
| **H_P5a** | §5 Honest Evidence Boundary Framework 章节正文字数 | ≥ 3500 字 | 🔵 | — |
| **H_P5b** | 全 RQ1-RQ4 + §5 章节合计字数 | ≥ 8000 字 | 🔵 | — |
| **H_P5c** | evidence_boundary_table.md 在 §5 显式 cite | 显式引用 | 🔵 | — |

### 章节扩展目标

| 章节 | 当前字数 | 目标字数 | 主要内容 |
|---|---:|---:|---|
| §3 方法 | ~150 字(stub)| 1500 字 | + 异构图 schema 图 + HeCo 损失推导 |
| §4.1 RQ1 共性 | ~100 字 | 1200 字 | + W11-P1 结果(如果通过)|
| §4.2 RQ2 差异 | ~80 字 | 1500 字 | E5 + E1 双段叙事 |
| §4.3 RQ3 迁移 | ~100 字 | 1500 字 | + W11-P2 IRM/EERM 全方法对比 |
| §4.4 RQ4 弱监督 | ~150 字 | 1200 字 | + W11-P3 4-transfer 矩阵 |
| **§5 失败讨论** | ~200 字 | **3500 字** | **独立卖点章节** |
| §6 局限与未来 | 待写 | 800 字 | + IRB / 数据更新机制 |
| **合计** | ~780 字 | **~11200 字** | 符合情报学 C 刊典型篇幅 |

---

## 总验收摘要

- **总假设数**:**15**
- 🔴 关键假设(影响主分数提升):**H_P1a, H_P1b, H_P1c, H_P3a**(4 个)
- 🟡 期望假设:H_P1d, H_P2c, H_P3b, H_P3c(4 个)
- 🔵 零失败风险假设:H_P2a/b, H_P4a/b, H_P5a/b/c(7 个)

### 通过率与分数对应表

| 通过组合 | 期望分数 | 概率估计 |
|---|---:|---:|
| 全 15 通过 | **9.0/10** | ~10% |
| 关键 + P2/P3 全通过(13/15)| **8.7/10** | ~25% |
| 关键 P1 通过 + 其他(11/15)| **8.5/10** | ~30% |
| 仅 P2/P3/P4/P5 通过(P1 失败)| **8.0/10** | ~25% |
| 仅 P4/P5 通过(P1/P2/P3 失败)| **7.7/10** | ~10% |

**期望分数(加权平均)**:**8.4/10**

---

## 论文诚实性承诺

无论通过率多少,所有假设结果(包括失败)在论文中公开报告,作为研究的诚实贡献。

**特别承诺**:
1. P1 失败 → §5 公开"family metric learning 在小样本下未成功",作为方法学边界
2. P2 H_P2a 失败 → §4.3 公开"IRM/EERM 在小样本异构图设定下未显著优于 source_only",作为方法学发现
3. P3 H_P3a 失败 → §4.4 公开 T1_DK 的"one-class target few-shot 边界"

---

## 跑前签名

| 角色 | 姓名 | 时间 | 签名 |
|---|---|---|---|
| 提议者(Master) | _________ | _________ | _________ |
| 技术顾问(Raphael) | Claude / Raphael | 2026-05-08 | ✓ |

> 一旦签署完成,本文档锁定。Week 11 实验启动后,任何阈值修改请求必须在论文中以"事后调整(post-hoc adjustment)"显式标注。
