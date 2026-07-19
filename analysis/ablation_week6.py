"""Week-6 head/backbone ablation aggregation."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.train.utils import ensure_directory, save_dataframe


def load_week4(model_name: str) -> dict[str, float | str]:
    """Load week4."""
    group_aware_path = ROOT / "output" / "week4" / "metrics" / "pooled_primary_group_aware_reaggregated_summary.csv"
    if group_aware_path.exists():
        summary = pd.read_csv(group_aware_path)
        task_name = "pooled_primary_group_aware_reaggregated"
        source = "Week 4 group-aware pooled baseline"
    else:
        summary = pd.read_csv(ROOT / "output" / "week4" / "metrics" / "pooled_primary_summary.csv")
        task_name = "pooled_primary"
        source = "Week 4 pooled baseline"
    row = summary[(summary["task"] == task_name) & (summary["model"] == model_name)].iloc[0]
    return {
        "auc_mean": float(row["roc_auc_mean"]),
        "auc_std": float(row["roc_auc_std"]),
        "source": source,
    }


def load_week5(model_name: str, temperature: float) -> dict[str, float | str]:
    """Load week5."""
    raw = pd.read_csv(ROOT / "output" / "week5" / "metrics" / "heco_probe_raw_runs.csv")
    row = raw[(raw["model"] == model_name) & (raw["temperature"] == temperature)].iloc[0]
    return {
        "auc_mean": float(row["roc_auc"]),
        "auc_std": float("nan"),
        "source": f"Week 5 single seed tau={temperature}",
    }


def load_week6_disc_lr() -> dict[str, float | str]:
    """Load week6 disc lr."""
    path = ROOT / "output" / "week6" / "metrics" / "pooled_primary_finetune_summary.csv"
    summary = pd.read_csv(path)
    row = summary[
        (summary["task"] == "pooled_primary_finetune")
        & (summary["model"] == "heco_disc_lr_finetune")
    ].iloc[0]
    return {
        "auc_mean": float(row["roc_auc_mean"]),
        "auc_std": float(row["roc_auc_std"]),
        "source": "Week 6 five-seed HeCo discriminative-LR finetune",
    }


def load_week6_uniform_lr() -> dict[str, float | str]:
    """Load week6 uniform lr."""
    path = ROOT / "output" / "week6" / "metrics" / "pooled_primary_finetune_uniform_lr_summary.csv"
    if not path.exists():
        return {
            "auc_mean": float("nan"),
            "auc_std": float("nan"),
            "source": "Week 6 uniform-LR fallback not run",
        }
    summary = pd.read_csv(path)
    row = summary[
        (summary["task"] == "pooled_primary_finetune_uniform_lr")
        & (summary["model"] == "heco_uniform_lr_finetune")
    ].iloc[0]
    return {
        "auc_mean": float(row["roc_auc_mean"]),
        "auc_std": float(row["roc_auc_std"]),
        "source": "Week 6 five-seed HeCo uniform-LR fallback audit",
    }


def load_cross_verified() -> dict[str, float | str]:
    """Load cross verified."""
    path = ROOT / "output" / "week6" / "metrics" / "cross_verified_pooled_eval.csv"
    if not path.exists():
        return {"auc_mean": float("nan"), "auc_std": float("nan"), "source": "Week 6 RQ1 eval pending"}
    frame = pd.read_csv(path)
    row = frame[frame["scope"] == "pooled_5_seed"].iloc[0]
    return {
        "auc_mean": float(row["roc_auc"]),
        "auc_std": float("nan"),
        "source": "Week 6 cross_verified vs licensed pooled test evaluation",
    }


def build_ablation_table() -> pd.DataFrame:
    """Build ablation table."""
    rows = [
        {
            "row": 1,
            "model": "RF + GroupKFold (legacy phase2)",
            "auc_mean": 0.921,
            "auc_std": float("nan"),
            "source": "phase2 historical",
            "notes": "Historical reference, not rerun in Week 6.",
        },
        {"row": 2, "model": "Logistic Regression", **load_week4("logistic_regression"), "notes": "Website feature baseline."},
        {"row": 3, "model": "MLP (features)", **load_week4("mlp"), "notes": "Website feature MLP baseline."},
        {"row": 4, "model": "HeteroGNN-only", **load_week4("hetero_gnn"), "notes": "Week 4 HeteroConv graph baseline."},
        {
            "row": 5,
            "model": "HeCo frozen encoder + linear probe",
            **load_week5("heco_frozen_linear", 0.7),
            "notes": "Single seed tau=0.7 selected by frozen validation AUC.",
        },
        {
            "row": 6,
            "model": "HeCo pretrain + full finetune",
            **load_week5("heco_full_finetune", 0.7),
            "notes": "Single seed tau=0.7, uniform LR Week 5 reference.",
        },
        {
            "row": 7,
            "model": "HeCo + Disc-LR finetune (5 seed)",
            **load_week6_disc_lr(),
            "notes": "Week 6 planned main result; threshold audit required because seed variance is high.",
        },
        {
            "row": 8,
            "model": "HeCo + uniform-LR finetune fallback (5 seed)",
            **load_week6_uniform_lr(),
            "notes": "Fallback audit from Week 6 risk plan; not better than Disc-LR under current splits.",
        },
        {
            "row": 9,
            "model": "Cross-verified primary layer evaluation",
            **load_cross_verified(),
            "notes": "RQ1 evidence-layer evaluation, not a model ablation row.",
        },
    ]
    return pd.DataFrame(rows)


def plot_fig4(table: pd.DataFrame) -> None:
    """Plot fig4."""
    fig_rows = table[table["row"] <= 8].copy()
    output_path = ROOT / "figs" / "fig4_pooled_auc.pdf"
    ensure_directory(output_path.parent)
    colors = ["#6b7280", "#3b82f6", "#60a5fa", "#2563eb", "#f59e0b", "#f97316", "#dc2626", "#991b1b"]
    fig, axis = plt.subplots(figsize=(9, 5))
    axis.bar(
        fig_rows["model"],
        fig_rows["auc_mean"],
        yerr=fig_rows["auc_std"].fillna(0.0),
        color=colors,
        capsize=5,
    )
    axis.set_ylim(0.75, 1.0)
    axis.set_ylabel("ROC-AUC")
    axis.set_title("Figure 4. RQ1 pooled licensed-vs-illegal AUC")
    axis.grid(axis="y", alpha=0.2)
    axis.tick_params(axis="x", rotation=35)
    for tick in axis.get_xticklabels():
        tick.set_horizontalalignment("right")
    for _, row in fig_rows[fig_rows["auc_std"].isna()].iterrows():
        axis.text(
            int(row["row"]) - 1,
            float(row["auc_mean"]) + 0.006,
            "single seed",
            ha="center",
            va="bottom",
            fontsize=7,
        )
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def write_paper_draft(table: pd.DataFrame) -> None:
    """Write paper draft."""
    week6 = table[table["row"] == 7].iloc[0]
    uniform = table[table["row"] == 8].iloc[0]
    cv = table[table["row"] == 9].iloc[0]
    hetero = table[table["model"] == "HeteroGNN-only"].iloc[0]
    delta = float(week6["auc_mean"]) - float(hetero["auc_mean"])
    uniform_delta = float(uniform["auc_mean"]) - float(hetero["auc_mean"])
    draft = f"""# Section 4.2 RQ1 Draft

Week 6 freezes the RQ1 main result on Graph v2. The final supervised label space is `licensed_baseline` versus `illegal_confirmed_official_single + illegal_confirmed_official_cross_verified`; gray candidates and legal-commercial controls are excluded from the main training label space. All reported Week 6 splits are group-aware and reuse the Week 4 grouping policy to reduce brand/operator leakage.

The strongest Week 4 graph-only baseline is HeteroGNN-only, with mean ROC-AUC `{hetero['auc_mean']:.4f}` over five seeds. After HeCo pretraining and discriminative learning-rate supervised finetuning, the Week 6 planned main model reaches mean ROC-AUC `{week6['auc_mean']:.4f}` with standard deviation `{week6['auc_std']:.4f}`. This is a delta of `{delta:+.4f}` over the graph-only baseline. The uniform-LR fallback reaches mean ROC-AUC `{uniform['auc_mean']:.4f}` with standard deviation `{uniform['auc_std']:.4f}`, a delta of `{uniform_delta:+.4f}` over the same baseline.

The Week 6 result should therefore be reported as a mixed integration result rather than a clean improvement over HeteroGNN-only. Seeds 42-45 are strong, but seed 46 is an outlier under the fixed group-aware split family; both the discriminative-LR and uniform-LR variants miss the predeclared pooled mean and variance targets. This does not invalidate the audit trail, but it means the final paper should avoid claiming that the integrated HeCo finetune is the locked best model until the variance issue is addressed.

The cross-verified evidence layer is evaluated separately because it is the core RQ1 primary-evidence tier rather than a separate model. Pooling the five test predictions and comparing `illegal_confirmed_official_cross_verified` against licensed-baseline test samples gives ROC-AUC `{cv['auc_mean']:.4f}`. This number should be presented as the primary-layer robustness check, not as an additional training run.

Table 2 should use `output/week6/metrics/ablation_table.csv` as its source, and Figure 4 should use `figs/fig4_pooled_auc.pdf`.
"""
    output_path = ROOT / "paper" / "draft" / "sec4_2_rq1.md"
    ensure_directory(output_path.parent)
    output_path.write_text(draft, encoding="utf-8")


def write_acceptance_audit(table: pd.DataFrame) -> None:
    """Write acceptance audit."""
    thresholds = {
        "pooled_auc_mean_min": 0.95,
        "pooled_auc_std_max": 0.04,
        "heterognn_delta_auc_min": 0.02,
        "cross_verified_auc_min": 0.95,
    }
    hetero = table[table["model"] == "HeteroGNN-only"].iloc[0]
    disc = table[table["row"] == 7].iloc[0]
    uniform = table[table["row"] == 8].iloc[0]
    cv = table[table["row"] == 9].iloc[0]
    audit_rows = []
    for label, row in [("disc_lr", disc), ("uniform_lr_fallback", uniform)]:
        delta = float(row["auc_mean"]) - float(hetero["auc_mean"])
        audit_rows.extend(
            [
                {
                    "scope": label,
                    "criterion": "pooled_auc_mean_min",
                    "observed": float(row["auc_mean"]),
                    "threshold": thresholds["pooled_auc_mean_min"],
                    "passed": float(row["auc_mean"]) >= thresholds["pooled_auc_mean_min"],
                },
                {
                    "scope": label,
                    "criterion": "pooled_auc_std_max",
                    "observed": float(row["auc_std"]),
                    "threshold": thresholds["pooled_auc_std_max"],
                    "passed": float(row["auc_std"]) <= thresholds["pooled_auc_std_max"],
                },
                {
                    "scope": label,
                    "criterion": "heterognn_delta_auc_min",
                    "observed": delta,
                    "threshold": thresholds["heterognn_delta_auc_min"],
                    "passed": delta >= thresholds["heterognn_delta_auc_min"],
                },
            ]
        )
    audit_rows.append(
        {
            "scope": "cross_verified",
            "criterion": "cross_verified_auc_min",
            "observed": float(cv["auc_mean"]),
            "threshold": thresholds["cross_verified_auc_min"],
            "passed": float(cv["auc_mean"]) >= thresholds["cross_verified_auc_min"],
        }
    )
    save_dataframe(pd.DataFrame(audit_rows), ROOT / "output" / "week6" / "metrics" / "week6_acceptance_audit.csv")


def main() -> None:
    """Command-line entry point."""
    output_dir = ROOT / "output" / "week6" / "metrics"
    ensure_directory(output_dir)
    table = build_ablation_table()
    save_dataframe(table, output_dir / "ablation_table.csv")
    write_acceptance_audit(table)
    plot_fig4(table)
    write_paper_draft(table)


if __name__ == "__main__":
    main()
