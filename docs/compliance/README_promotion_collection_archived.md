# PromotionAccount Collection Archive

归档日期：2026-04-23

## 归档原因

本目录保存被剔除的 PromotionAccount (PA) 采集尝试，作为论文补充材料和合规性证据链。根据实验执行手册 v2，PA 维度不再进入主异构图 schema，也不参与 Week 3-10 的主实验。

## 归档内容

- `collect_promotion_osint.py`：X API / Telegram 公开页面采集脚本。
- `promotion_collection_audit.csv`：采集审计表。
- `promotion_accounts.csv`：PA 账号输出表。
- `promotion_edges.csv`：PA -> Website 候选 promotes 边输出表。

## 采集结论

- 采集日期：2026-04-22
- 目标站点数：120
- 初始品牌词弱匹配命中：1/120
- 初始命中率：0.83%
- 高置信域名/URL 直接命中：0
- 高置信主图可用 promotes 边：0

## 放弃决定

正式放弃 PA 维度日期：2026-04-23。

放弃理由：

- X API recent search 只有短时间窗口，难以覆盖长期推广行为。
- 品牌词匹配噪声高，出现同名节目、普通词和体育词误匹配。
- 单次采集命中率仅 0.83%，且最终高置信域名/URL 直接命中为 0。
- Telegram 公开频道需要人工维护频道种子，当前无法形成均衡、可复现、跨地区覆盖。
- 将低置信 PA 候选写入主图会污染 `promotes` 边，并放大跨地区迁移实验中的噪声。

## v2 替代方案

从 2026-04-23 起，主实验迁移到 ExternalReference (ExtRef) 方案：

- 使用 `search_discovery_registry` 和 `history_osint_evidence` 构建 ExternalReference 节点。
- 用 `referenced_by: Website -> ExternalReference` 替代原计划中的 `promotes: PromotionAccount -> Website`。
- 在论文 §3.1 主动说明 PA 的合规性和可获得性限制。
- 在论文 §5.5 讨论 PA 维度缺失的影响与未来工作。
- 在 Table 7 增加 ExtRef / M4 消融，检验 PA 替代信号是否有贡献。

## 论文引用口径

建议在补充材料中引用本 README：

> We attempted to collect PromotionAccount evidence through public X API and Telegram public previews. Due to the short X recent-search window, high noise in brand-token matches, and a 0.83% initial hit rate with no high-confidence URL/domain matches, PA was excluded from the main graph. The archived script and audit tables are retained as compliance and reproducibility evidence.

## 安全边界

- 未登录非法站点。
- 未注册、未充值、未加入私密群组。
- 未绕过验证码、地理限制或反爬机制。
- X 使用官方 API。
- Telegram 方案仅限公开 `t.me/s/<channel>` 页面，但当前未启用频道采集。
