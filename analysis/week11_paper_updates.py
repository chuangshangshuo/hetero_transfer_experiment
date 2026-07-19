"""Week-11 P5: regenerate paper-facing section drafts from metrics."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]


def ensure_dirs(config: dict) -> dict[str, Path]:
    """Ensure directories."""
    root = Path(config["workspace_root"])
    paths = {name: root / rel for name, rel in config["output"].items()}
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    (ROOT / "paper" / "draft").mkdir(parents=True, exist_ok=True)
    return paths


def load_metrics(path: Path) -> str:
    """Load metrics."""
    if not path.exists():
        return "相应 Week11 指标尚未生成，本节保留为结果占位但不替代实测。"
    frame = pd.read_csv(path)
    return f"当前表包含 {len(frame)} 行，可用于后续论文表格。"


def write_sections(paths: dict[str, Path]) -> None:
    """Write sections."""
    p1_note = load_metrics(ROOT / "output" / "week11" / "metrics" / "P1_family_metric_lofo.csv")
    p2_note = load_metrics(ROOT / "output" / "week11" / "metrics" / "P2_method_comparison.csv")
    p3_note = load_metrics(ROOT / "output" / "week11" / "metrics" / "P3_full_fewshot_matrix.csv")

    sec3 = f"""# §3 方法深化:异质图、对比预训练与边界化评估

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
"""

    sec41 = f"""# §4.1 RQ1 共性:从 illegal-vs-licensed 到 family boundary

RQ1 的核心问题是模型是否能够区分非法网站与 licensed controls。Week6 的 cross-verified 评估已经给出强证据:在高质量 officially cross-verified illegal 层上，模型能够稳定地区分 illegal 与 licensed baseline。这说明异质图和网站层特征确实捕捉到了非法网站生态的一部分共性。

但是，Week8.5 和 Week9 的 hard-negative family audit 也暴露了更严格的问题:当负例不再是 licensed controls，而是其他 illegal family 时，原来的 binary head 会把几乎所有 illegal 都推到接近 1 的非法概率，导致 family-vs-other-illegal 的排序能力不足。这不是非法识别能力的完全失败，而是训练目标与评估目标错位。binary classifier 被训练来回答“是否非法”，并没有被训练来回答“属于哪个非法家族”。

Week11 P1 因此新增 family-aware metric learning head。它冻结既有 HeCo encoder，只在表示上增加一个 family projection head，并用 supervised contrastive loss 拉近同 family 样本、推远不同 family 样本，同时保留 binary head 维护 illegal-vs-licensed 能力。该 patch 的证据解释有明确边界:若 hard-negative mean AUC 提升，它只能说明 family-aware calibration 有助于 family boundary；若提升有限，则说明在当前 7 个丹麦 family、小样本规模与弱 family 元数据条件下，家族级归因仍然不稳。

{p1_note}

论文中应把 RQ1 写成分层结论:第一层，模型对 officially confirmed illegal 与 licensed controls 的区分有较强支持；第二层，family-level hard-negative 区分在原 binary head 下较弱；第三层，Week11 P1 作为预声明改进检验 family metric learning 是否能够补这个洞。无论 P1 通过还是失败，都不能把结果写成“模型可以稳定完成非法家族归因”。更安全的写法是:模型识别 illegal-vs-licensed 的能力强于识别 family-vs-other-illegal 的能力，family-aware head 是对这个边界的针对性校准。
"""

    sec43 = f"""# §4.3 RQ3 迁移:真实地区环境、IRM/EERM 与方法边界

RQ3 关注跨地区迁移是否稳定。Week7 已经表明迁移表现高度 target-dependent:T2_PH 是 hard but recoverable target，T3 France 是 source-only 已接近饱和但 shortcut-sensitive 的 target，T1_Nordic 与 T2_ON 则属于 one-class 或边界一致性任务，不能纳入标准 ROC-AUC 主表。

Week11 P2 补上 IRM/EERM 方法对比。这里最重要的发现可能不是某个方法显著获胜，而是 IRM 的适用性边界。真实地区 source pool 往往不是每个 environment 都有 illegal 和 licensed 两类样本:有的地区几乎只有 illegal，有的地区几乎只有 licensed。IRM 依赖按环境计算可比 risk gradient，这个假设在小样本跨地区情报数据中经常先天不满足。EERM 通过 embedding clustering 构造 virtual environment，可以绕过真实地区单类的问题，但它本身又引入 K-means 聚类和 K 值敏感性。

{p2_note}

因此，RQ3 的安全结论不是“DANN、StruRW、IRM 或 EERM 解决了迁移”，而是“迁移边界需要按目标域和环境可比性来解释”。如果 IRM/EERM 在某个 transfer 上超过 source-only，论文可以把它作为方法补充；如果没有显著超过，论文仍可把它写成小样本异质图迁移的限制证据。这个结论与 Week8 的 LogME 失败相互呼应:迁移可行性不能只由单一 proxy 或单一高 AUC 决定，而必须结合目标标签结构、feature shortcut、family hard-negative 和 few-shot recovery 一起判断。
"""

    sec44 = f"""# §4.4 RQ4 弱监督与本地校准:从两个 binary target 扩展到四类 transfer

Week9 的 few-shot 结果已经说明，少量目标域标签可以显著修复 T2_PH 这样的 hard but recoverable target，但对 T3 France 这种 source-only 已经接近饱和且存在 lexical/ccTLD shortcut 风险的 target 几乎没有额外收益。这个结论必须以 delta versus source-only 为核心，而不能只看 few-shot 后的绝对 AUC。

Week11 P3 把 RQ4 扩展到 T1_Nordic 与 T2_ON 两个 one-class/boundary target。对 T1_Nordic，目标域测试集只有 illegal，因此评估指标是 illegal recall at Youden，而不是 ROC-AUC。对 T2_ON，目标域是 licensed-only，核心指标是 mean predicted illegal probability 是否保持低位。这个设计承认了真实迁移任务并不总是标准二分类:有些目标域只能检验模型是否把全非法目标识别出来，有些目标域只能检验模型是否不会把全 licensed 目标误报为非法。

{p3_note}

最终 RQ4 应写成 target-dependent local calibration。若 T1_Nordic 的 5-shot illegal recall 明显提升，可以说明 one-class hard target 也有本地校准空间；若 T2_ON 的 mean_pred_illegal 维持低位，则说明 licensed-only target 的边界没有被 few-shot 支持破坏；若任何一项失败，也应作为弱监督边界公开报告。few-shot 的价值不在于普遍提升，而在于揭示哪些 target 能被少量本地标签修复，哪些 target 已饱和或不适合标准二分类指标。
"""

    sec5 = """# §5 Honest Evidence Boundary Framework:失败、边界与可审计证据

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
"""

    files = {
        "sec3_methods_deepened_week11.md": sec3,
        "sec4_1_rq1_family_metric_week11.md": sec41,
        "sec4_3_rq3_invariance_methods_week11.md": sec43,
        "sec4_4_rq4_full_fewshot_week11.md": sec44,
        "sec5_honest_evidence_boundary_framework.md": sec5,
    }
    for name, text in files.items():
        (ROOT / "paper" / "draft" / name).write_text(text, encoding="utf-8")


def plot_boundary_diagram(paths: dict[str, Path]) -> None:
    """Plot boundary diagram."""
    labels = ["Structural\ncommonality", "Regional\nbias", "Evidence\nhierarchy", "Few-shot\ncalibration"]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axis("off")
    centers = [(0.25, 0.65), (0.75, 0.65), (0.25, 0.25), (0.75, 0.25)]
    colors = ["#2563eb", "#dc2626", "#059669", "#7c3aed"]
    for (x, y), label, color in zip(centers, labels, colors):
        ax.text(
            x,
            y,
            label,
            ha="center",
            va="center",
            color="white",
            fontsize=12,
            bbox=dict(boxstyle="round,pad=0.6", facecolor=color, edgecolor="none", alpha=0.92),
        )
    ax.text(0.5, 0.45, "Honest Evidence Boundary Framework", ha="center", va="center", fontsize=14, weight="bold")
    for x, y in centers:
        ax.annotate("", xy=(0.5, 0.45), xytext=(x, y), arrowprops=dict(arrowstyle="->", lw=1.5, color="#334155"))
    fig.tight_layout()
    fig.savefig(paths["plots"] / "P5_evidence_boundary_diagram.png", dpi=180)
    plt.close(fig)


def main() -> None:
    """Command-line entry point."""
    config = yaml.safe_load((ROOT / "configs" / "week11.yaml").read_text(encoding="utf-8"))
    paths = ensure_dirs(config)
    write_sections(paths)
    plot_boundary_diagram(paths)
    print("P5 paper sections and diagram written")


if __name__ == "__main__":
    main()
