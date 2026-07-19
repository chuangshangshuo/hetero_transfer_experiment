"""Aggregate feature-bucket transfer contributions into summary tables."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.week10_common import (
    DEFAULT_CONFIG,
    canonical_feature_condition,
    ensure_output_dirs,
    input_path,
    load_config,
    output_path,
    read_input,
    save_csv,
    with_claim_columns,
)


FEATURE_ORDER = [
    "full",
    "no_cctld",
    "no_website_lexical",
    "graph_only",
    "lexical_only",
    "no_edges",
    "infra_edges_only",
]


def build_feature_bucket_summary(config: dict) -> pd.DataFrame:
    """Build feature bucket summary."""
    summary = read_input(config, "week8_patch", "corrected_e5_summary")
    if summary.empty:
        return pd.DataFrame()
    rows = []
    for _, row in summary.iterrows():
        condition = canonical_feature_condition(row["config_id"])
        if condition not in FEATURE_ORDER:
            continue
        rows.append(
            {
                "transfer_id": row["transfer_id"],
                "feature_condition": condition,
                "config_id": row["config_id"],
                "num_runs": int(row["num_runs"]),
                "metric_mean": float(row["primary_metric_value_mean"]),
                "metric_std": float(row["primary_metric_value_std"]),
                "source_file": str(input_path(config, "week8_patch", "corrected_e5_summary").relative_to(ROOT)),
                "is_posthoc": True,
            }
        )
    frame = pd.DataFrame(rows)
    full = frame[frame["feature_condition"].eq("full")][["transfer_id", "metric_mean"]].rename(columns={"metric_mean": "full_metric_mean"})
    frame = frame.merge(full, on="transfer_id", how="left")
    frame["delta_vs_full"] = frame["metric_mean"] - frame["full_metric_mean"]
    frame["paper_safe_claim"] = frame.apply(safe_claim, axis=1)
    frame["paper_forbidden_claim"] = frame.apply(forbidden_claim, axis=1)
    frame["status"] = frame.apply(status_label, axis=1)
    return frame.sort_values(["transfer_id", "feature_condition"]).reset_index(drop=True)


def status_label(row: pd.Series) -> str:
    """Status label."""
    if row["transfer_id"] == "T3_DiagnoseFrance" and row["feature_condition"] in {"no_cctld", "no_website_lexical", "graph_only", "lexical_only"}:
        return "shortcut_boundary_diagnostic"
    if abs(float(row["delta_vs_full"])) >= 0.05:
        return "material_feature_bucket_effect"
    return "limited_feature_bucket_effect"


def safe_claim(row: pd.Series) -> str:
    """Safe claim."""
    if row["transfer_id"] == "T3_DiagnoseFrance":
        return "T3 France is substantially assisted by lexical/ccTLD boundary features; graph-only is weaker than lexical-only."
    return "Feature-bucket ablation is diagnostic and target-dependent."


def forbidden_claim(row: pd.Series) -> str:
    """Forbidden claim."""
    if row["transfer_id"] == "T3_DiagnoseFrance":
        return "Do not claim T3 France proves pure structural transfer."
    return "Do not infer a universal feature mechanism from one target."


def plot_feature_buckets(frame: pd.DataFrame, output: Path) -> None:
    """Plot feature buckets."""
    focus = frame[frame["feature_condition"].isin(FEATURE_ORDER)].copy()
    if focus.empty:
        return
    targets = list(focus["transfer_id"].drop_duplicates())
    fig, axes = plt.subplots(len(targets), 1, figsize=(8.5, max(3.2, 2.4 * len(targets))), sharex=True)
    if len(targets) == 1:
        axes = [axes]
    for axis, target_id in zip(axes, targets):
        part = focus[focus["transfer_id"].eq(target_id)].copy()
        part["feature_condition"] = pd.Categorical(part["feature_condition"], categories=FEATURE_ORDER, ordered=True)
        part = part.sort_values("feature_condition")
        colors = ["#4c78a8" if cond == "full" else "#f58518" if target_id == "T3_DiagnoseFrance" and cond in {"no_cctld", "no_website_lexical", "graph_only", "lexical_only"} else "#72b7b2" for cond in part["feature_condition"]]
        axis.bar(part["feature_condition"].astype(str), part["metric_mean"], yerr=part["metric_std"], color=colors, alpha=0.9, capsize=3)
        axis.axhline(float(part.loc[part["feature_condition"].astype(str).eq("full"), "metric_mean"].iloc[0]), color="black", linewidth=1, linestyle="--")
        axis.set_title(target_id)
        axis.set_ylabel("metric")
        axis.grid(axis="y", alpha=0.25)
    axes[-1].set_xlabel("feature bucket")
    axes[-1].tick_params(axis="x", rotation=35)
    fig.suptitle("Figure 7. Feature Bucket Transfer Diagnostics", y=0.995)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)


def main() -> None:
    """Command-line entry point."""
    config = load_config(DEFAULT_CONFIG)
    ensure_output_dirs(config)
    frame = build_feature_bucket_summary(config)
    save_csv(frame, output_path(config, "metrics", "feature_bucket_transfer_summary.csv"))
    plot_feature_buckets(frame, output_path(config, "figures", "fig7_feature_bucket_transfer.pdf"))
    print(frame[["transfer_id", "feature_condition", "metric_mean", "delta_vs_full", "status"]].to_string(index=False))


if __name__ == "__main__":
    main()
