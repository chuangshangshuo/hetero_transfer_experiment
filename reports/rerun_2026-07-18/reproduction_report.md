# 复现测试报告（2026-07-18）

对照论文《图神经网络在跨地区非法网络赌博识别中的应用边界——基于11国异构图的域适应失效模式研究》（修改版 v1，期刊版式 DOCX），在本发布仓库上重新执行可行范围内的实验测试，并将论文全部关键数字与发布数据逐项核对。

**环境**：Windows 11 / CPython 3.12.12 venv（pandas·numpy·scipy·pyyaml）。发布包按 `DATA_AVAILABILITY.md` 不含原始图数据（`hetero_graph_v2.pt`）与 checkpoint，故 HeCo 预训练/finetune/迁移训练等**训练阶段无法重跑**；以下为可重跑部分与数据核对结果。

## 一、重跑结果

| 测试 | 结果 | 说明 |
|---|---|---|
| `python -m compileall src analysis` | ✅ 通过 | 全部代码无语法错误 |
| `analysis/final_artifact_audit.py`（126 项检查） | ✅ 101 通过 / 27 失败 | 失败项**全部**是发布策略剔除的文件（runs/logs/predictions/样本级 CSV/40 个 split 审计），与 README"What Is Intentionally Excluded"一致；**所有可运行的指标数值检查 100% 通过**。发布版审计（原始工作区）为 0 失败。副本见 `rerun_final_artifact_audit.csv` |
| `analysis/week11_significance.py`（§4.4 配对 bootstrap B=10,000 + Holm） | ✅ **比特级复现** | 从 80 行 seed 级数据重算，三张 P4 表与发布版数值差 = 0、Holm 判定完全一致：24 对中 9 对显著；唯一"迁移方法真胜 source_only"= T2_ON×DANN（Δ=+0.057，p=0.0012） |
| `analysis/week11_acceptance.py`（15 项预声明假设） | ✅ **逐列完全一致** | 9 通过 / 6 失败，与论文及发布版一致（SupCon 后均值 0.1259、tsars 0.0741、Δ=−0.496、T1 5-shot 0.458 未过 0.50 线等） |

## 二、论文数字 vs 发布数据核对（111 项）

**107 项吻合**（容差 0.0015，即三位小数舍入）：

- 图统计：Website 623（613 标注 + 10 灰区）、IP 1325、Cert 510、NS 959、Registrar 79、ExtRef 53；边 1516+1662+519+252+57 = **4006** ✓
- 表2 全部（LR 0.894±0.059 → HeCo finetune 0.934±0.074；cross_verified 0.980 = pooled 行 0.9798）✓
- §4.2 迁移矩阵 16 个均值全对（含 StruRW 在 T2_ON 退化为 mpi≈0.0003 的 trivial classifier）✓
- 表4 IRM T2_PH −0.079、EERM 6/8 项 ✓（T2_ON 两项见下）
- 表5 少样本 4×5 全部 20 格 ✓；表6 消融 20 格 ✓；表7 六个显著边效应 ✓；表8 七家族×3 指标 + 均值 0.622 ✓
- §5.3 A1–A4 的 4 个 AUC 全对；测试集规模 n=31/14/24/29 ✓

**4 项发现问题，建议投稿前处理**：

1. **摘要+§4.4 分类口径写反**：论文称 9 对显著中"**5 对**是退化方法被 source_only 反超、**3 对**是方法间互比"。重算数据为：**3 对** source_only 反超（T2_ON×StruRW +0.302、T2_ON×DANN+StruRW +0.278、T2_PH×DANN +0.104）、**5 对**方法互比（T1 两对、T2_ON 两对、T2_PH 一对）、1 对真胜。两个数字对调了（中英文摘要与 §4.4 正文三处）。
2. **表4 T2_ON 行 EERM K=2/K=4 符号口径**：数据（mpi 原始尺度）K2 = −0.019（mpi 下降=**改善**）、K4 = +0.021（恶化）；表4 行标注为 1−mpi 尺度，按该尺度应为 K2 = **+0.019**、K4 = **−0.021**。现表数值等于 mpi 尺度增量，与行标尺度矛盾（对"EERM 自身最大 ΔAUC +0.032"的结论无影响）。
3. **§5.3 A2/A3 score_gap 与数据小幅不符**：论文 A2 −0.213、A3 −0.224；`P1_alpha_ablation.csv`（ggbet，5 seed 均值）为 A2 **−0.210**、A3 **−0.215**。AUC（0.078/0.082）完全一致，仅 gap 偏差 0.003–0.009，疑似取自早期运行，建议与释出 CSV 同步。
4. **表6 C3 行数据源说明**：C3"−lex(仅ccTLD)"数值来自 week8 **原版** `E5_structure_vs_lexical_summary.csv` 的 `C3_no_lex`（完全吻合）；而修正版（week8_patch）已将 C3 重定义为 `no_website_lexical`（数值与 C4 相同）。建议表注说明 C3 引自原版配置，避免读者用修正版文件对不上。

另两处小提示：τ=0.7 的 0.918 是 frozen probe **测试集** AUC（验证集为 0.863，仍为四温度中最高，选温结论不变，但 §3.3.1"验证集AUC最高(0.918)"措辞可再准确些；且 τ=0.5 测试集同为 0.918）；T2_ON×DANN 的"ΔAUC=+0.057"实际是 1−mpi 尺度增量（单类目标域无 AUC），正文口径已在阅读提示中标注。

## 三、结论

在发布包可验证的范围内，论文的证据链**高度可靠**：统计检验与假设审计可比特级复现，111 项数字核对 107 项精确吻合，4 项差异均属文字口径/来源标注问题而非结果错误，且不动摇任何主结论（RQ1–RQ4 的边界判断全部得到数据支持）。

**产物清单**：`rerun_final_artifact_audit.csv`（本包审计重跑）、`rerun_P4_w7_method_pairwise.csv`（显著性重算，与发布版 0 差异）、`rerun_week11_acceptance_audit.csv`（15 假设重审，与发布版 0 差异）。论文精读工件见 `paper/reader/`（paper.md·source_map.json·translation_notes.md·assets/）。
