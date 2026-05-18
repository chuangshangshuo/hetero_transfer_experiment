# Week 11 实验设计文档:激进路线 B —— 从 7.5/10 → 8.5–9.0/10

> **课题**:跨地区非法网络赌博团簇共性与差异性研究
> **本周角色**:Week 11 是项目最后的"分数提升周",通过 5 个高杠杆 patch 把整体评分从 7.5/10 推到 8.5–9.0/10
> **设计哲学**:**进取路线**——P1 是真正的 game-changer(可能失败,但代价小,收益大);P2 不论结果如何都对论文有利;P3/P4/P5 是稳态收益
> **预声明承诺**:5 个 patch 的所有验收阈值在跑前冻结;所有失败假设依然作为论文诚实贡献
> **文档版本**:v1.0,2026-05-08

---

## 0. 文档导航

- §1 当前 7.5/10 的扣分项盘点
- §2 5 个 patch 的总体策略
- §3 Patch P1:Family-aware Metric Learning Head(最大杠杆)
- §4 Patch P2:IRM + EERM 实战对比(填补方法对比缺口)
- §5 Patch P3:全 transfer few-shot 矩阵(扩展 RQ4)
- §6 Patch P4:Metric-level 统计检验补全
- §7 Patch P5:论文章节深化与诚实证据框架成章
- §8 基础设施补全清单
- §9 总 run 数 / 时间预算 / 文件结构
- §10 验收标准总表(预声明,prerun-frozen)
- §11 RQ 升级映射与论文产出对照
- §12 执行顺序与风险控制
- §13 关键实测约束与设计调整说明

---

## 1. 当前 7.5/10 的扣分项盘点

| 维度 | 当前分 | 上限 | 扣分项 | 杠杆性 | 对应 Patch |
|---|---:|---:|---|---|---|
| 数据质量 | 8 | 10 | 38 isolated 全剔、10 gray 全剔、family 元数据来源弱 | 🟡 中 | P1 部分激活 |
| 实验广度 | 9 | 10 | IRM/EERM 代码骨架空着,W10 Table6 标 not_run | 🟢 高 | **P2** |
| 方法学严谨性 | 9 | 10 | hard-negative 用 binary head 任务设定错 | 🔴 极高 | **P1** |
| RQ1 共性 | 7.5 | 9 | hard-negative AUC=0.62 是死穴 | 🔴 极高 | **P1** |
| RQ2 差异性 | 8 | 9 | E5 graph/lex 主导无 metric-level 检验 | 🟡 中 | P4 |
| RQ3 迁移 | 6 | 8 | DANN/StruRW 无增益,IRM 缺位,无对比基线 | 🟢 高 | **P2** |
| RQ4 弱监督 | 7 | 9 | 只测了 T2_PH 一个胜利 target,T1/T2_ON 没做 | 🟡 中 | **P3** |
| 论文产出 | 7 | 9 | 12 个 sec 是 stub,需扩展叙事 | 🟢 高 | **P5** |
| 工程可复现 | 9 | 10 | OK,边际收益小 | ⚪ 低 | — |

**核心诊断**:**RQ1 的 hard-negative 死穴**(P1)+ **RQ3 的 IRM/EERM 缺位**(P2)是两个最大杠杆点。其余三个 patch 是稳态收益。

---

## 2. 5 个 patch 的总体策略

### 2.1 五项总览

| Patch | 核心改进 | run 数 | 时间 | 风险 | 期望分数提升 |
|---|---|---:|---|---|---:|
| **P1 Family Metric Learning** | supervised contrastive head + N-way K-shot LOFO | 70 | 2 天 | 中 | **+0.5 → 0.6** |
| **P2 IRM + EERM 对比** | IRM(T2_PH only)+ EERM(全 4 transfer) | 50 | 1 天 | 低 | **+0.2 → 0.3** |
| **P3 全 transfer few-shot** | T1_DK + T2_ON few-shot 补齐 | 50 | 半天 | 极低 | **+0.15 → 0.2** |
| **P4 统计显著性补全** | paired bootstrap + Wilcoxon + Bonferroni | 0 | 半天 | 极低 | **+0.10 → 0.15** |
| **P5 论文章节深化** | §5 写成"诚实证据框架"独立卖点 | 0 | 1 周 | 极低 | **+0.15 → 0.2** |
| **合计** | | **170** | **~10 天** | | **+1.10 → +1.45** |

### 2.2 三条硬约束(继承 W4-W10)

1. **图与 encoder 不再动**
   - `data/graphs/hetero_graph_v2.pt` 不修改
   - 复用 W6 的 5 个 HeCo encoder checkpoint
   - 任何"重新预训练"需求一律延后,不在 Week 11 范围
2. **预声明假设 + 失败应对** —— 所有阈值在跑前冻结,失败本身写入论文
3. **每个 patch 独立可中止** —— P1 失败不阻塞 P2,P2 失败不阻塞 P3,以此类推

### 2.3 工程规范(差异点)

- **新增**:每个 patch 启动前生成独立的 `predeclared_*.md`,git commit hash 绑定
- **新增**:每个 patch 完成后立即写 `acceptance_audit.csv`,带 prerun_frozen=True 字段
- **沿用**:5 seed × {42,43,44,45,46}、group-aware split、paired bootstrap

---

## 3. Patch P1:Family-aware Metric Learning Head 🔴 最大杠杆

### 3.1 问题根因

当前 hard-negative family AUC=0.622 不是模型不行,而是**任务-评估错位**。看 W9 raw:

| family | positive_score_mean | negative_score_mean | gap |
|---|---:|---:|---:|
| icecasino | 0.99999 | 0.99996 | **0.00003** |
| verdecasino | 0.9991 | 0.9982 | 0.001 |
| ggbet | 0.9985 | 0.9351 | 0.063 |
| tsars | 0.9945 | 0.9517 | 0.043 |

**所有 illegal 都被 binary classifier 推到 ~1.0**,因为 W6 finetune 头的训练目标是 "illegal vs licensed",从未告诉它"区分 tsars vs ggbet"。这就是 family AUC=0.62 的根本原因——**不是模型缺乏家族信号,是 head 没用上**。

### 3.2 改进方案:Supervised Contrastive Family Head

在 W6 frozen encoder 之上增加一个独立的 metric-learning head,**与 binary head 并列**(双 head 训练),目标是:**同 family 嵌入靠近,跨 family 嵌入远离**。

#### 3.2.1 模型结构

```python
# src/models/family_metric_head.py
class FamilyMetricHead(nn.Module):
    """监督对比学习 head,产生 family-aware projection。"""
    def __init__(self, embed_dim=64, projection_dim=32, hidden_dim=64):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, projection_dim),
        )

    def forward(self, x):
        z = self.proj(x)
        return F.normalize(z, dim=-1)              # L2 归一化,后续点积即 cosine
```

#### 3.2.2 损失函数(SupCon, Khosla et al. 2020)

```python
def supcon_loss(z, family_labels, temperature=0.1):
    """
    z: (B, D) 已归一化
    family_labels: (B,) 整数家族 ID
    """
    sim = z @ z.T / temperature                   # (B, B)
    # 数值稳定:减去每行最大值
    sim = sim - sim.max(dim=1, keepdim=True).values.detach()

    pos_mask = (family_labels[:, None] == family_labels[None, :]).float()
    pos_mask.fill_diagonal_(0)                    # 排除自身

    # 对比基:除自身外的所有样本(同/异家族)
    logits_mask = torch.ones_like(pos_mask)
    logits_mask.fill_diagonal_(0)

    exp_sim = torch.exp(sim) * logits_mask
    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-12)

    # 仅在有正例的样本上算 loss
    pos_count = pos_mask.sum(dim=1)
    valid = pos_count > 0
    mean_log_prob_pos = (pos_mask * log_prob).sum(dim=1)[valid] / pos_count[valid]
    return -mean_log_prob_pos.mean()
```

#### 3.2.3 双 head 联合训练

```python
total_loss = (
    1.0 * binary_ce_loss(binary_head(emb), illegal_labels)      # 保持原有"illegal vs licensed"能力
    + alpha * supcon_loss(family_head(emb), family_labels)       # 新增 family 能力
)
# 推荐 alpha = 0.5
```

### 3.3 实验设计

#### 3.3.1 训练池

| 部分 | 来源 | 数量 | 角色 |
|---|---|---:|---|
| Family-labeled illegal | DK 7 family(size≥2)| 35 | family-aware 正例 / 负例 |
| Cross-region illegal | 非 DK illegal | 141 | 充当 "其他家族" 类(用 jurisdiction 当 family ID) |
| Licensed | 全 275 licensed | 275 | binary head 的负例 |
| **Total 训练池** | | **451** | |

#### 3.3.2 N-way K-shot Family Sampler

每 batch 构造:
- 抽 **N=4** 个 family(可能 DK 内的家族,也可能 non-DK 的 jurisdiction-as-family)
- 每 family 抽 **K=2** 个 sample
- 加入 **8 个 licensed**(让 binary head 也有信号)
- 总 batch size = 4×2 + 8 = **16**

确保:
- DK family 至少有 1 个出现在 batch 中(否则 SupCon 学不到 DK 家族信号)
- 每 epoch 滚遍所有 DK family ≥ 1 次

#### 3.3.3 Leave-One-Family-Out (LOFO) 评估协议

对每个 DK family F_i(7 个):

```
# 训练
train_pool = (DK\F_i 的 illegal) + (non-DK illegal) + (全 licensed)
train_with_dual_head(train_pool, alpha=0.5, epochs=200)

# 评估:F_i 内 sample 与剩余 illegal 的 family-level 区分
# 关键 metric:在 family head projection 空间做 k-NN 分类
test_form_1: hard_illegal_negative
  positive: F_i 全部 sample
  negative: 等量随机抽样的非 F_i illegal(不含 licensed!)
  # 这是 W9 设定的复刻,保证可比性

test_form_2: family_attribution
  query: F_i 全部 sample
  database: 其他 family 的所有 sample
  metric: top-k retrieval accuracy(query 是否检索到同家族?)
  注意:F_i 是被留出的,所以不应该有同家族 → top-k 必须指向"距离最近的非自己家族"
  改为:测试 query 与"其它每个 family centroid"的距离排序
```

#### 3.3.4 关键消融(Ablation)

为了证明改进确实来自 family-aware loss,而不是简单"额外参数":

| Ablation | 描述 | 期望表现 |
|---|---|---|
| A1: alpha=0(only binary)| 等价于 W9 baseline | hard-neg AUC ≈ 0.62 |
| A2: alpha=0.5 双 head | 主提案 | hard-neg AUC ↑ |
| A3: alpha=1.0(only supcon)| 不做 binary | binary AUC 退化,family AUC 最高 |
| A4: alpha=0.5 但 family ID 打乱 | 验证 family 信号真实性 | 应回到 ≈0.62 |

5 seed × 7 family × 4 alpha = **140 run**(精简后 70 run,见下)

#### 3.3.5 Run 数精简

| 维度 | 取值 | 数量 |
|---|---|---|
| Family fold | DK 7 family(size≥2)| 7 |
| Method | A2(主提案,alpha=0.5)| 1 |
| Seed | {42,43,44,45,46} | 5 |
| **核心 run** | | **35** |
| 消融(只跑 tsars + ggbet) | A1/A3/A4 × 2 family × 5 seed | **30** |
| 置换检验 | A4 在 4 个家族 + 5 seed | **(已含在消融)** |
| **合计** | | **65** |

时间:**~2 天**

### 3.4 主产出

| 产出 | 路径 | 论文位置 |
|---|---|---|
| 7 family hard-neg AUC 表 | `output/week11/metrics/P1_family_metric_lofo.csv` | Table(RQ1 章节) |
| 4 alpha 消融表 | `output/week11/metrics/P1_alpha_ablation.csv` | Table(RQ1 章节) |
| family head 的 t-SNE | `output/week11/plots/P1_family_tsne.png` | Figure(关键图) |
| family centroid distance matrix | `output/week11/metrics/P1_family_centroid_distance.csv` | 附录 |

### 3.5 可证伪假设(预声明)

| ID | 假设 | 通过条件 | 失败应对 |
|---|---|---|---|
| **H_P1a** | 7 family hard-negative mean AUC ≥ 0.75(vs W9 baseline 0.622)| ≥ 0.75 | 失败 → 论文写"family metric learning 改善有限",仍为 RQ1 边界证据 |
| **H_P1b** | tsars hard-negative AUC ≥ 0.70(vs W9 0.386)| ≥ 0.70 | 失败 → 单独写 tsars 作为最难 case,但保留其他 family 的提升 |
| **H_P1c** | mean AUC vs W9 baseline 提升 ≥ +0.10 | ≥ +0.10 | 失败 → patch 整体不写入论文,作为 negative result 在附录 |
| **H_P1d** | Ablation A4(family 打乱)AUC 应显著低于 A2 (≥ +0.05 gap)| A2 - A4 ≥ +0.05 | 失败 → 信号是 spurious correlation,不能宣称 family-aware |

### 3.6 期望分数提升路径

| H_P1a 状态 | H_P1b 状态 | 期望分数变化 |
|---|---|---:|
| 通过 | 通过 | +0.6(论文 RQ1 章节质变) |
| 通过 | 失败 | +0.4(部分胜利) |
| 失败但 ≥ 0.65 | — | +0.2(改善有限) |
| 失败 < 0.65 | — | 0(写入 appendix) |

---

## 4. Patch P2:IRM + EERM 实战对比 🟢 高杠杆,无失败风险

### 4.1 问题根因

W10 Table 6 第 27 行:

| ablation_item | status |
|---|---|
| IRM | **not_implemented_not_citable** |

`src/transfer/irm_penalty.py` 仅 130 字节(空骨架),`src/transfer/eerm_virtual_env.py` 仅 144 字节。审稿人会问:"你为什么不和 IRM 对比?" —— 这是 RQ3 的硬伤之一。

### 4.2 关键实测约束(必须先说)

通过 W7 splits 实测确认:

| Transfer | Source 池 environment 真实情况 | IRM 适用性 | EERM 适用性 |
|---|---|---|---|
| T1_Nordic | IT(纯 ill) + ES(纯 lic) + SE(纯 lic) | **不适用** —— 每 env 单类,无 risk gradient | 适用(virtual env)|
| T2_PH | 8 国,但只有 BE+FR 两个 env 既有 ill 又有 lic | **限定适用** —— IRM 仅用 BE+FR | 适用 |
| T2_ON | 8 国(同 T2_PH source 池)| **限定适用** —— IRM 仅用 BE+FR | 适用 |
| T3_DiagnoseFrance | IT(纯 ill) + ES(纯 lic) | **不适用** —— 与 T1 同问题 | 适用 |

**这个约束本身就是论文价值**:在情报学小样本设定下,**真实地区往往是单类的**(某些国家只有 licensed,某些国家只有 illegal),IRM 的"环境内有可比 risk"假设**先天难以满足**。EERM(用 K-means 聚类生成 virtual environment)绕开这个约束,**反而更适合本场景**。

### 4.3 实现方案

#### 4.3.1 IRM(Arjovsky et al. 2019, IRMv1)

```python
# src/transfer/irm_penalty.py
def irm_penalty(logits, labels, scale_w=1.0):
    """IRMv1 penalty:把 classifier 缩放为 trainable scalar w,对 w 求梯度。"""
    w = torch.tensor(scale_w, requires_grad=True, device=logits.device)
    loss = F.binary_cross_entropy_with_logits(logits * w, labels.float())
    grad = torch.autograd.grad(loss, w, create_graph=True)[0]
    return grad ** 2

def irm_loss_per_env(env_logits_dict, env_labels_dict, lambda_irm=1.0):
    """
    env_*: dict[env_id] -> tensor
    """
    erm_loss = 0.0
    irm_pen = 0.0
    for env in env_logits_dict:
        z, y = env_logits_dict[env], env_labels_dict[env]
        erm_loss += F.binary_cross_entropy_with_logits(z, y.float())
        irm_pen += irm_penalty(z, y)
    n_env = len(env_logits_dict)
    return erm_loss / n_env + lambda_irm * (irm_pen / n_env)
```

#### 4.3.2 EERM(Wu et al. 2022, environment extrapolation)

```python
# src/transfer/eerm_virtual_env.py
def make_virtual_environments(embeddings, K=4, seed=42):
    """
    在 source 嵌入上跑 K-means,生成 K 个 virtual environment ID
    返回 (N,) 的 env_id 数组
    """
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=K, random_state=seed, n_init=10)
    return km.fit_predict(embeddings)

def eerm_loss(embeddings, logits, labels, lambda_irm=1.0, K=4, seed=42):
    """先聚类,然后在 virtual env 上做 IRM。"""
    env_ids = make_virtual_environments(embeddings.detach().cpu().numpy(), K=K, seed=seed)
    env_logits = {k: logits[env_ids == k] for k in range(K) if (env_ids == k).sum() > 4}
    env_labels = {k: labels[env_ids == k] for k in range(K) if (env_ids == k).sum() > 4}
    return irm_loss_per_env(env_logits, env_labels, lambda_irm)
```

EERM 的关键设计:**virtual environment 的多样性来自 K-means 在嵌入空间的硬聚类**,而 K-means 对初始化敏感,所以 K∈{2,3,4} 多 K 实验,取最稳健的。

### 4.4 实验设计

#### 4.4.1 Run 矩阵

| Method | Transfer 适用 | run 数 |
|---|---|---:|
| IRM | T2_PH only(因 IRM 在 BE+FR 两 env 上)| 1 × 5 seed = **5** |
| EERM (K=2) | 全 4 transfer | 4 × 5 = **20** |
| EERM (K=4) | 全 4 transfer | 4 × 5 = **20** |
| **合计** | | **45** |

#### 4.4.2 训练 protocol

- 复用 W7 v2 的 source_only 训练框架,只替换 loss 函数
- λ_IRM:cosine warmup 从 0 → 1.0 over 50 epoch(参考 IRM 原论文)
- 早停:沿用 W7 v2 修复后的 `>=` + min_epochs=50

### 4.5 主产出

| 产出 | 路径 | 论文位置 |
|---|---|---|
| 5 method × 4 transfer 全方法对比表 | `output/week11/metrics/P2_method_comparison.csv` | Table 7(RQ3 章节)|
| K=2 vs K=4 EERM 稳健性 | `output/week11/metrics/P2_eerm_K_sensitivity.csv` | 附录 |
| IRM 收敛曲线 | `output/week11/plots/P2_irm_convergence.png` | 附录 |

### 4.6 可证伪假设(预声明)

| ID | 假设 | 通过条件 | 应对 |
|---|---|---|---|
| **H_P2a** | IRM/EERM 在任一 transfer 上 vs source_only ΔAUC ≥ +0.05 | 通过 → 真胜利,论文加为 method 章节 | — |
| **H_P2b** | IRM/EERM 在所有 transfer 都不显著 | 通过 → 论文写"小样本异构图的方法学局限" | **本身即贡献** |
| **H_P2c** | EERM 在 K=2 vs K=4 之间 ΔAUC 范围 ≤ 0.10 | 通过 → 方法稳健 | 失败 → 写为 K-超参敏感性 |

**P2 关键属性:无论 H_P2a 通过还是 H_P2b 通过,论文都受益**。这是 zero-loss patch。

### 4.7 期望分数提升

无论结果如何:**+0.2 ~ +0.3**(因为补齐了 RQ3 的方法对比缺口)。

---

## 5. Patch P3:全 transfer few-shot 矩阵 🟡 中杠杆

### 5.1 问题根因

W9 only 测了 T2_PH 和 T3 的 few-shot,**T1_DK(illegal_recall=0.21,项目最大硬伤之一)**完全没做。如果 few-shot 能救 T1_DK,论文 RQ4 的叙事从 "target-dependent" 升级到 "comprehensive few-shot framework"。

### 5.2 改进方案

#### 5.2.1 T1_DK few-shot

T1_DK 的目标域全是 illegal,所以传统 few-shot 无法直接抽 licensed。设计:

```python
def t1_dk_few_shot_split(target_test_idx, source_train_idx, shots_per_class):
    # 从 source 池(SE+ES+IT)中抽 K 个 licensed 加入目标域 finetune
    licensed_extra = sample_licensed_from_source(source_train_idx, k=shots_per_class)
    # 从 DK 目标域抽 K 个 illegal(从 cross_verified 主层抽,因为 W6 已证它们最稳定)
    illegal_extra = sample_dk_illegal(target_test_idx - eval_idx, k=shots_per_class)

    fewshot_train = source_train + licensed_extra + illegal_extra
    eval_set = target_test - illegal_extra              # 留出来真正测试
    return fewshot_train, eval_set
```

关键:**licensed 部分用 source 抽样**(因为 DK 没有 licensed);**illegal 部分用 cross_verified 优先抽样**(质量最高)。

#### 5.2.2 T2_ON few-shot

T2_ON 目标域全是 licensed:

```python
def t2_on_few_shot_split(target_test_idx, source_train_idx, shots_per_class):
    # 从 source 池(8 国)中抽 K 个 illegal 加入目标域 finetune
    illegal_extra = sample_illegal_from_source(source_train_idx, k=shots_per_class)
    # 从 ON 目标域抽 K 个 licensed
    licensed_extra = sample_on_licensed(target_test_idx - eval_idx, k=shots_per_class)

    fewshot_train = source_train + illegal_extra + licensed_extra
    eval_set = target_test - licensed_extra
    return fewshot_train, eval_set
```

### 5.3 实验矩阵

| Transfer | shots_per_class | seed | run |
|---|---|---|---:|
| T1_DK | {0, 1, 3, 5, 10} | 5 | 25 |
| T2_ON | {0, 1, 3, 5, 10} | 5 | 25 |
| **合计** | | | **50** |

加上 W9 已有的 T2_PH + T3 = **完整 4-transfer few-shot 矩阵**。

### 5.4 主产出

| 产出 | 路径 | 论文位置 |
|---|---|---|
| 4-transfer few-shot 矩阵 | `output/week11/metrics/P3_full_fewshot_matrix.csv` | Table(RQ4 章节)|
| Few-shot recovery curve(2×2 panel)| `output/week11/plots/P3_fewshot_curves.png` | Figure(RQ4 关键图)|

### 5.5 可证伪假设(预声明)

| ID | 假设 | 通过条件 | 应对 |
|---|---|---|---|
| **H_P3a** | T1_DK 5-shot illegal_recall ≥ 0.50(vs 0.21)| ≥ 0.50 | 失败 → 写"one-class target few-shot 有限" |
| **H_P3b** | T2_ON 5-shot mean_pred_illegal 仍 < 0.30(保持验收)| < 0.30 | 失败 → 写"licensed-only target few-shot 干扰原边界" |
| **H_P3c** | 所有 4 transfer 中至少 3 个表现"recoverable" | ≥ 3/4 | 失败 → 论文 RQ4 仍写为 "target-dependent" |

### 5.6 期望分数提升

H_P3a 通过的概率较高(因为 T1 目标域有 cross_verified 高质量 illegal,加入训练应有显著矫正)。预期 **+0.15 ~ +0.2**。

---

## 6. Patch P4:Metric-level 统计检验补全 🟡 必做项

### 6.1 问题根因

W7 method 对比、W9 few-shot delta 都**只有 mean ± std**,没有正式显著性检验。审稿人:"DANN vs source_only 显著吗?"——拿不出 p-value。

### 6.2 改进方案(零重训,纯事后分析)

#### 6.2.1 三类检验补全

| 检验类型 | 适用 | 公式 / 算法 |
|---|---|---|
| Paired bootstrap(95% CI + p)| 任何 method 对比 | B=10000 次重采样 |
| Wilcoxon signed-rank test | 5 seed paired,非参 | scipy.stats.wilcoxon |
| Holm-Bonferroni 多重校正 | 16+ 配对(method × transfer)| FWER ≤ 0.05 |

#### 6.2.2 实现

```python
# src/utils/significance_testing.py
def paired_bootstrap_pvalue(a, b, B=10000, seed=42):
    """两个配对 paired vector 的差异显著性。"""
    rng = np.random.default_rng(seed)
    n = len(a)
    diffs = a - b
    obs_mean = diffs.mean()
    boot_means = []
    for _ in range(B):
        idx = rng.choice(n, n, replace=True)
        boot_means.append(diffs[idx].mean())
    boot_means = np.array(boot_means)
    # 双侧 p
    p = 2 * min(
        (boot_means <= 0).mean() if obs_mean > 0 else (boot_means >= 0).mean(),
        0.5
    )
    ci = np.percentile(boot_means, [2.5, 97.5])
    return obs_mean, ci[0], ci[1], p


def holm_bonferroni(p_values, alpha=0.05):
    """逐步 Holm-Bonferroni,返回 reject (boolean) array。"""
    m = len(p_values)
    sorted_idx = np.argsort(p_values)
    reject = np.zeros(m, dtype=bool)
    for rank, idx in enumerate(sorted_idx):
        threshold = alpha / (m - rank)
        if p_values[idx] < threshold:
            reject[idx] = True
        else:
            break
    return reject
```

#### 6.2.3 应用范围

- **W7 method 对比**:4 method × 4 transfer × C(4,2) = **24 配对** + Holm 校正
- **W9 few-shot delta**:每 shot vs 0-shot,2 transfer × 4 shot levels = **8 配对**
- **W11 P1/P2/P3 新结果**:全部加入

### 6.3 主产出

| 产出 | 路径 | 论文位置 |
|---|---|---|
| W7 method 配对显著性 | `output/week11/metrics/P4_w7_method_pairwise.csv` | Table(RQ3 鲁棒性附录)|
| W9 few-shot 显著性 | `output/week11/metrics/P4_w9_fewshot_significance.csv` | Table |
| W11 全周显著性总表 | `output/week11/metrics/P4_w11_overall_significance.csv` | 附录 |

### 6.4 期望分数提升

**+0.10 ~ +0.15**(纯方法学严谨性)。零失败风险。

---

## 7. Patch P5:论文章节深化 + 诚实证据框架成章

### 7.1 问题根因

`paper/draft/` 12 个 sec 加起来才 402 行,大部分是 table stub。情报学 C 刊更看重叙事深度。

### 7.2 改进方案

#### 7.2.1 §5 重写为 "Honest Evidence Boundary Framework" 独立章节

把这件事从"附录补丁"提升为**论文的最大差异化点**。结构:

- **§5.1 理论框架**:为什么跨地区情报学研究需要"诚实证据边界"
- **§5.2 双标记体系**:prerun vs posthoc 在方法学上的认识论意义
- **§5.3 失败假设作为研究贡献**:从 H_E5a 到 H_P1b/c 的全列表
- **§5.4 边界识别框架**:RQ1 family-level 边界、RQ2 ccTLD shortcut 边界、RQ3 IRM/EERM 失败边界、RQ4 saturated target 边界

#### 7.2.2 各 RQ 章节的扩展段落

| 章节 | 当前 stub | 扩展目标 |
|---|---:|---:|
| §3 方法 | 短 | 1500 字 + 异构图 schema 图 + HeCo 损失推导 |
| §4.1 RQ1 共性 | 11 行 | 1200 字 + W11-P1 结果(如果通过)|
| §4.2 RQ2 差异 | 7 行 | 1500 字 + E5 + E1 双段叙事 |
| §4.3 RQ3 迁移 | 9 行 | 1500 字 + W11-P2 IRM/EERM 全方法对比 |
| §4.4 RQ4 弱监督 | 11 行 | 1200 字 + W11-P3 4-transfer 矩阵 |
| §5 失败讨论 | 16 行 | **3500 字**(独立卖点章节)|
| §6 局限与未来 | 待写 | 800 字 + IRB / 数据更新机制 |

合计 **~11200 字** 论文正文,符合情报学 C 刊典型篇幅(8000-15000 字)。

### 7.3 工程量

写作 **~5-7 天**,集中在 W11 后半周。

### 7.4 期望分数提升

**+0.15 ~ +0.2**(论文"成色"层面)。

---

## 8. 基础设施补全清单

| 模块 | 路径 | 工程量 |
|---|---|---:|
| Family metric head | `src/models/family_metric_head.py` | ~80 行 |
| SupCon loss + N-way K-shot sampler | `src/train/supcon_sampler.py` | ~100 行 |
| Family-aware LOFO driver | `src/train/train_family_metric.py` | ~150 行 |
| IRM 实现 | `src/transfer/irm_penalty.py`(填充)| ~80 行 |
| EERM 实现 | `src/transfer/eerm_virtual_env.py`(填充)| ~100 行 |
| Method comparison driver | `src/train/train_invariance_methods.py` | ~120 行 |
| T1/T2_ON few-shot driver | `src/transfer/few_shot_finetune.py`(填充)| ~150 行 |
| 显著性检验工具 | `src/utils/significance_testing.py` | ~80 行 |
| W11 验收 driver | `analysis/week11_acceptance.py` | ~120 行 |
| **总代码量** | | **~980 行** |

加配置文件 `configs/week11.yaml`(~100 行)。

---

## 9. 总 run 数 / 时间预算 / 文件结构

### 9.1 Run 总数

| Patch | core run | 消融 / 辅助 | 合计 |
|---|---:|---:|---:|
| P1 | 35 | 30 | 65 |
| P2 | 45 | — | 45 |
| P3 | 50 | — | 50 |
| P4 | 0(纯分析)| — | 0 |
| **合计** | **130** | **30** | **160** |

### 9.2 时间预算(单机 CPU,W11 单 run 约 1-3 分钟)

| Patch | wall-time |
|---|---|
| P1 Family metric learning | ~2 天(实现 + 跑 65 run + 分析)|
| P2 IRM + EERM | ~1 天(实现 + 跑 45 run)|
| P3 全 transfer few-shot | ~半天(扩展 W9 driver + 跑 50 run)|
| P4 统计检验 | ~半天(纯分析)|
| P5 论文写作 | **~1 周** |
| **合计** | **~10-11 天** |

### 9.3 输出目录结构

```
output/week11/
├── runs/                            # 160 个 run 的 manifest JSON
│   ├── P1_familyloo_tsars__seed42__alpha0.5.json
│   ├── P2_irm_T2_PH__seed42.json
│   ├── P2_eerm_T3__seed42__K2.json
│   ├── P3_fewshot_T1_DK__seed42__shots5.json
│   └── ...
├── metrics/
│   ├── P1_family_metric_lofo.csv
│   ├── P1_alpha_ablation.csv
│   ├── P1_family_centroid_distance.csv
│   ├── P2_method_comparison.csv
│   ├── P2_eerm_K_sensitivity.csv
│   ├── P3_full_fewshot_matrix.csv
│   ├── P4_w7_method_pairwise.csv
│   ├── P4_w9_fewshot_significance.csv
│   ├── P4_w11_overall_significance.csv
│   └── week11_acceptance_audit.csv
├── audits/
│   └── predeclared_hypotheses.md    # 跑前冻结(15 假设)
├── plots/
│   ├── P1_family_tsne.png
│   ├── P2_irm_convergence.png
│   ├── P3_fewshot_curves.png        # 4-transfer 2×2 panel
│   └── P5_evidence_boundary_diagram.png
└── logs/
    └── *_history.csv
```

### 9.4 配置文件结构

`configs/week11.yaml`:

```yaml
inherit: configs/week7_transfer.yaml

P1_family_metric:
  encoder_seeds: [42, 43, 44, 45, 46]
  alpha: 0.5
  temperature: 0.1
  projection_dim: 32
  N_way: 4
  K_shot: 2
  licensed_in_batch: 8
  epochs: 200
  patience: 50
  ablations:
    - name: A1_only_binary
      alpha: 0.0
    - name: A2_main
      alpha: 0.5
    - name: A3_only_supcon
      alpha: 1.0
      binary_weight: 0.0
    - name: A4_shuffled_family
      alpha: 0.5
      shuffle_family_labels: true

P2_invariance_methods:
  irm:
    enabled_for: [T2_PH]
    lambda_irm_warmup: cosine
    lambda_irm_max: 1.0
    warmup_epochs: 50
  eerm:
    enabled_for: [T1_Nordic, T2_PH, T2_ON, T3_DiagnoseFrance]
    K_values: [2, 4]
    embedding_for_clustering: heco_finetune

P3_full_fewshot:
  transfers:
    T1_Nordic:
      shots_per_class: [0, 1, 3, 5, 10]
      illegal_source_priority: cross_verified
    T2_ON:
      shots_per_class: [0, 1, 3, 5, 10]
      illegal_source: pooled_8_european

P4_significance:
  bootstrap_B: 10000
  multiple_comparison: holm_bonferroni
  alpha: 0.05
```

---

## 10. 验收标准总表(预声明,prerun-frozen)

> 实验启动前冻结。15 个假设,5 个关键假设标红,3 个零失败风险标蓝。

| Patch | ID | 假设 | 阈值 | 关键性 |
|---|---|---|---|:---:|
| **P1** | H_P1a | 7 family hard-neg mean AUC | ≥ 0.75(vs 0.622)| 🔴 |
| **P1** | H_P1b | tsars hard-neg AUC | ≥ 0.70(vs 0.386)| 🔴 |
| **P1** | H_P1c | mean AUC 提升 vs W9 baseline | ≥ +0.10 | 🔴 |
| **P1** | H_P1d | A2 - A4(shuffled)gap | ≥ +0.05 | 🟡 |
| **P2** | H_P2a | IRM/EERM 任一 transfer 上 ΔAUC | ≥ +0.05 vs source_only | 🔵 |
| **P2** | H_P2b | IRM/EERM 全部不显著(零失败假设)| 任何结果都对论文有利 | 🔵 |
| **P2** | H_P2c | EERM K=2 vs K=4 ΔAUC range | ≤ 0.10 | 🟡 |
| **P3** | H_P3a | T1_DK 5-shot illegal_recall | ≥ 0.50(vs 0.21)| 🔴 |
| **P3** | H_P3b | T2_ON 5-shot mean_pred_illegal | < 0.30 | 🟡 |
| **P3** | H_P3c | 4 transfer ≥ 3 个 recoverable | ≥ 3/4 | 🟡 |
| **P4** | H_P4a | W7 method 配对显著性结果 | 至少 1/24 配对 p<0.05 | 🔵 |
| **P4** | H_P4b | W9 few-shot delta 显著性 | 至少 4/8 配对 p<0.05 | 🔵 |
| **P5** | H_P5a | §5 章节正文 ≥ 3500 字 | ≥ 3500 | 🔵 |
| **P5** | H_P5b | 全 RQ 章节合计 ≥ 8000 字 | ≥ 8000 | 🔵 |
| **P5** | H_P5c | evidence_boundary_table.md 在 §5 引用 | 显式 cite | 🔵 |

🔴 = 关键假设(影响主分数提升)
🟡 = 期望假设
🔵 = 零失败风险(任何结果都受益)

### 10.1 五个 patch 通过率与分数对应表

| 通过组合 | 期望分数 |
|---|---:|
| 全 15 通过 | **9.0/10**(理论上限)|
| H_P1a + H_P1b + H_P3a + 全部 P2/P4/P5 | **8.7/10** |
| H_P1a + H_P3a + 全部 P2/P4/P5 | **8.5/10** |
| 仅 P2/P3/P4/P5 通过(P1 失败)| **8.0/10**(策略 A 保底)|
| 仅 P4/P5 通过(P1/P2/P3 都失败)| **7.7/10**(最坏)|

**最坏情况下,W11 仍能保证 +0.2 提升**(因为 P4/P5 是零失败风险)。最好情况下 **+1.5 提升**。

---

## 11. RQ 升级映射与论文产出对照

### 11.1 各 patch → RQ 贡献矩阵

| Patch | RQ1 共性 | RQ2 差异 | RQ3 迁移 | RQ4 弱监督 | §5 失败讨论 |
|---|:---:|:---:|:---:|:---:|:---:|
| P1 Family Metric | ✓✓ | ✓ | — | ✓ | ✓ |
| P2 IRM/EERM | — | ✓ | ✓✓ | — | ✓ |
| P3 全 transfer few-shot | — | ✓ | ✓ | ✓✓ | ✓ |
| P4 显著性检验 | ✓ | ✓ | ✓ | ✓ | — |
| P5 论文写作 | ✓ | ✓ | ✓ | ✓ | ✓✓ |

### 11.2 Week 11 完成后论文核心 figure / table 清单

| 编号 | 内容 | 来源 |
|---|---|---|
| Fig 1 | Heterogeneous graph schema | W3 |
| Fig 2 | Pipeline overview(HeCo + finetune + transfer)| W5/W6/W7 |
| Fig 3 | Pooled AUC vs sample tier | W6 |
| Fig 4 | Cross-region t-SNE(licensed/illegal/control) | W8-E4 |
| Fig 5 | Edge ablation heatmap | W8-E1 |
| Fig 6 | Feature bucket heatmap(graph vs lex per transfer)| W8-E5 / W8 patch |
| **Fig 7** | **Family metric learning t-SNE**(P1 关键)| **W11-P1** |
| **Fig 8** | **IRM/EERM convergence**(P2)| **W11-P2** |
| **Fig 9** | **4-transfer few-shot recovery curves**(P3)| **W11-P3** |
| **Fig 10**| **Evidence boundary diagram**(P5)| **W11-P5** |

| 编号 | 内容 | 来源 |
|---|---|---|
| Tab 1 | Sample tier × jurisdiction 分布 | W2 |
| Tab 2 | W6 pooled AUC + cross_verified | W6 |
| Tab 3 | Method comparison(LR / MLP / HeteroGNN / HeCo finetune)| W4-W6 |
| Tab 4 | Head ablation | W6 head |
| Tab 5 | Transfer matrix(4 method × 4 transfer)+ paired bootstrap p | W7 + **P4** |
| **Tab 6**| **Full method comparison(+IRM/EERM)** | **W11-P2** |
| **Tab 7**| **Family LOFO(W9 baseline vs P1)** | **W11-P1** |
| **Tab 8**| **4-transfer few-shot matrix** | **W11-P3** |
| Tab 9 | Edge channel + feature bucket effect sizes | W8 + **P4** |
| Tab 10| Honest evidence boundary table | W10 + **P5** |

W11 直接贡献 **4 张 figure + 4 张 table**,论文产出量级再增 30%。

---

## 12. 执行顺序与风险控制

### 12.1 推荐执行顺序

| Day | 内容 | 中止条件 |
|---|---|---|
| **Day 1** | 写 P1 代码(family_metric_head + SupCon + sampler)| 代码 review 通过 |
| **Day 2** | 跑 P1 65 run + 分析 | 如 H_P1c 失败(< +0.10)→ 中止 P1 ablation 部分,继续 P2 |
| **Day 3** | 写 P2 代码(IRM + EERM)+ smoke test | smoke 通过 |
| **Day 4** | 跑 P2 45 run + 分析 | 无中止条件(P2 零失败风险)|
| **Day 5** | P3:T1_DK + T2_ON few-shot 50 run | 无中止条件 |
| **Day 6** | P4:全显著性检验事后分析 | 无中止条件 |
| **Day 7-10** | P5:论文 §3-§6 全章节扩写 | 完成 8000+ 字 |

### 12.2 风险点与应对

| 风险 | 概率 | 影响 | 应对 |
|---|---|---|---|
| P1 SupCon 在 7 family 上学不稳(每 family ≤ 10 个 sample)| 中 | 高 | 引入 jurisdiction-as-family 的非 DK illegal 充当对比类(141 sample),保证 batch 多样性 |
| P1 alpha=0.5 不是最佳 | 低 | 中 | 加 alpha={0.3, 0.5, 1.0} 三点 sweep,5 seed × 3 alpha × 1 family = 15 run 额外 |
| P2 IRM 在 BE+FR 仅 2 env 上不收敛 | 中 | 低 | 预声明 IRM 仅作 T2_PH 的方法学注释,EERM 作为主对比(EERM 不依赖真 env)|
| P2 EERM 的 K-means 初始化敏感 | 中 | 低 | 5 seed × 不同 K-means seed 跑,取均值;加 K∈{2,3,4} sweep |
| P3 T1_DK few-shot 在 cross_verified 上"作弊"(因为评估也用 cross_verified)| 中 | 高 | **关键:few-shot 抽样必须留出测试集**,确保 train/test 不重叠;split 用 group-aware (operator level) |
| P5 写作时间不够 | 高 | 中 | Day 7-10 集中写,Day 1-6 期间利用 idle time(等 run 跑)同步写 |

### 12.3 中止规则(stop conditions)

如果 **P1 三个核心假设(H_P1a/b/c)同时失败**:
- 暂停 P1 ablation 部分的 30 个 run
- P2/P3/P4 继续执行
- 论文中 P1 写为"探索性 negative result",放在附录

如果 **P3 H_P3a 失败**(T1_DK few-shot 不能救):
- 不影响其他 patch
- 论文 RQ4 章节维持 "target-dependent" 叙事,但补 T1/T2_ON 的诚实 boundary

---

## 13. 关键实测约束与设计调整说明

### 13.1 IRM 的真实 environment 限制

W7 splits 实测:

| Transfer | Source 真 environment | IRM 适用 | EERM 适用 |
|---|---|---|---|
| T1 | IT/ES/SE 全单类 | ✗ | ✓ |
| T2_PH | 仅 BE/FR 双类(其余 6 国单类)| 限定 | ✓ |
| T2_ON | 同 T2_PH | 限定 | ✓ |
| T3 | IT/ES 全单类 | ✗ | ✓ |

**这本身是论文价值**:在情报学小样本设定下,**真实地区天然单类**(某些国家全 licensed,某些全 illegal),IRM 的"环境内有可比 risk"假设难满足。**EERM 的 virtual environment 反而更适合本场景**,这是 RQ3 的方法学贡献。

### 13.2 P1 训练池的 family 多样性

DK 7 family(共 35 sample)+ 用 jurisdiction-as-family 的 non-DK illegal(141 sample,5 种 jurisdiction):**每 epoch 至少有 5+7=12 种 family**,batch 多样性足够 SupCon 学习。

### 13.3 P3 T1_DK few-shot 的"作弊"防护

T1_DK 测试集中的 cross_verified 33 条是高质量样本,如果直接抽进训练,会污染评估。**强制约束**:
- few-shot 抽样池 = T1_DK 训练集中的 single-tier illegal(44 条)
- 评估集 = T1_DK 测试集中的 cross_verified + single 混合 31 条,**不被 few-shot 触碰**

### 13.4 W11 与 W12+ 的关系

W11 完成后:
- W12-W14:论文写作 + 投稿前检查 + 反向审计
- W15-W17(预留):rebuttal 准备 / 备用实验

---

## 14. 文档版本与签名

- 版本:v1.0
- 创建日期:2026-05-08
- 课题:跨地区非法网络赌博团簇共性与差异性研究
- 阶段:Week 11(共 17 周,激进路线 B)
- 前置依赖:W4-W10 全部产出 + W10 final RQ rollup + evidence_boundary_table.md
- 后续依赖:W12 论文最终稿 + 投稿

---

> **预声明承诺**:本文档第 10 节验收标准在 Week 11 第一个 run 之前冻结,实验后不再调整阈值。所有失败假设直接写入论文,作为研究的诚实贡献。
>
> **激进路线判定**:5 个 patch 全做,期望 7.5 → 8.5–9.0;最坏情况(P1/P2/P3 都失败)仍保 7.7;最好情况(全通过)冲到 9.0。
