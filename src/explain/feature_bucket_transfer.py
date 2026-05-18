from __future__ import annotations

import argparse
import copy
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.train.train_transfer import (
    build_transfer_split,
    load_encoder_state,
    summarize_run_metrics,
    train_heco_transfer_model,
)
from src.train.utils import (
    GraphBundle,
    compute_sample_counts,
    load_graph_bundle,
    make_common_manifest,
    record_run_manifest,
    save_dataframe,
)
from src.utils.paired_bootstrap import paired_bootstrap


REVERSE_RELATIONS = {
    "hosted_on": "rev_hosted_on",
    "uses_cert": "rev_uses_cert",
    "uses_ns": "rev_uses_ns",
    "registered_via": "rev_registered_via",
    "referenced_by": "rev_referenced_by",
}


def redirect_output_root(bundle: GraphBundle, root_name: str) -> None:
    workspace = Path(bundle.config["workspace_root"])
    for key in list(bundle.output_paths.keys()):
        subdir = key if key != "root" else ""
        path = workspace / "output" / root_name / subdir
        path.mkdir(parents=True, exist_ok=True)
        bundle.output_paths[key] = path
        bundle.config["output"][key] = str(Path("output") / root_name / subdir) if subdir else str(Path("output") / root_name)


def edge_removed_slug(edge_removed: str) -> str:
    return "full" if edge_removed == "none" else f"minus_{edge_removed}"


def apply_edge_ablation(graph_data: Any, edge_removed: str) -> Any:
    data = copy.deepcopy(graph_data.cpu())
    if edge_removed == "none":
        return data
    if edge_removed == "ALL":
        rels = {edge_type[1] for edge_type in data.edge_types}
    else:
        if edge_removed not in REVERSE_RELATIONS:
            raise ValueError(f"Unsupported edge ablation: {edge_removed}")
        rels = {edge_removed, REVERSE_RELATIONS[edge_removed]}
    for edge_type in list(data.edge_types):
        if edge_type[1] not in rels:
            continue
        store = data[edge_type]
        store.edge_index = torch.empty((2, 0), dtype=store.edge_index.dtype)
        store.edge_weight = torch.empty((0,), dtype=store.edge_weight.dtype)
    return data


def load_week7_split_or_build(bundle: GraphBundle, transfer_id: str, seed: int) -> tuple[pd.DataFrame, Path]:
    workspace = Path(bundle.config["workspace_root"])
    week7_path = workspace / "output" / "week7" / "splits" / f"{transfer_id}__seed{seed}.csv"
    if week7_path.exists():
        return pd.read_csv(week7_path), week7_path
    split_frame, _ = build_transfer_split(bundle, transfer_id, seed)
    return split_frame, week7_path


def run_one_edge_ablation(
    bundle: GraphBundle,
    transfer_id: str,
    edge_removed: str,
    seed: int,
    smoke_test: bool,
) -> tuple[dict[str, Any], pd.DataFrame]:
    split_frame, split_path = load_week7_split_or_build(bundle, transfer_id, seed)
    encoder_state, encoder_checkpoint = load_encoder_state(bundle, seed)
    graph_data = apply_edge_ablation(bundle.graph_data, edge_removed)
    train_info, history, probabilities = train_heco_transfer_model(
        bundle=bundle,
        graph_data=graph_data,
        split_frame=split_frame,
        encoder_state=encoder_state,
        seed=seed,
        method="source_only",
        smoke_test=smoke_test,
        purpose="E1_edge_ablation",
    )
    metrics, predictions = summarize_run_metrics(
        bundle=bundle,
        split_frame=split_frame,
        probabilities_all=probabilities,
        method="source_only",
        seed=seed,
        transfer_id=transfer_id,
        train_info=train_info,
        domain_acc_final=float("nan"),
        pseudo_label_quality=float("nan"),
    )
    metrics.update(
        {
            "experiment": "E1_edge_ablation",
            "edge_removed": edge_removed,
            "edge_config": edge_removed_slug(edge_removed),
            "encoder_checkpoint": encoder_checkpoint,
        }
    )
    suffix = f"E1_{transfer_id}__{edge_removed_slug(edge_removed)}__seed{seed}"
    save_dataframe(history, bundle.output_paths["logs"] / f"{suffix}_history.csv")
    save_dataframe(predictions, bundle.output_paths["predictions"] / f"{suffix}_predictions.csv")
    manifest = make_common_manifest(
        bundle=bundle,
        task=f"E1_{transfer_id}",
        model="source_only",
        split_name=split_path.name,
        sample_counts=compute_sample_counts(split_frame),
        seed=seed,
    )
    manifest["encoder_checkpoint"] = encoder_checkpoint
    manifest["edge_removed"] = edge_removed
    manifest["metrics"] = metrics
    record_run_manifest(bundle.output_paths["runs"] / f"{suffix}.json", manifest)
    return metrics, history


def summarize_e1(bundle: GraphBundle, raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame()
    B = int(bundle.config["E1_edge_ablation"].get("paired_bootstrap_B", 1000))
    rows: list[dict[str, Any]] = []
    full = raw[raw["edge_removed"] == "none"].copy()
    for transfer_id, transfer_full in full.groupby("transfer_id"):
        direction = str(transfer_full["primary_metric_direction"].iloc[0])
        full_by_seed = transfer_full.set_index("seed")["primary_metric_value"]
        for edge_removed, ablated in raw[(raw["transfer_id"] == transfer_id) & (raw["edge_removed"] != "none")].groupby("edge_removed"):
            ablated_by_seed = ablated.set_index("seed")["primary_metric_value"]
            aligned = pd.concat([full_by_seed.rename("full"), ablated_by_seed.rename("ablated")], axis=1).dropna()
            raw_delta = aligned["ablated"] - aligned["full"]
            if direction == "lower_is_better":
                effect = raw_delta
            else:
                effect = aligned["full"] - aligned["ablated"]
            boot = paired_bootstrap(effect.to_numpy(dtype=float), B=B, seed=42)
            raw_boot = paired_bootstrap(raw_delta.to_numpy(dtype=float), B=B, seed=43)
            rows.append(
                {
                    "experiment": "E1_edge_ablation",
                    "transfer_id": transfer_id,
                    "edge_removed": edge_removed,
                    "metric": str(transfer_full["effective_primary_metric"].iloc[0]),
                    "primary_metric_direction": direction,
                    "n_pairs": int(boot.n),
                    "full_mean": float(aligned["full"].mean()) if len(aligned) else float("nan"),
                    "ablated_mean": float(aligned["ablated"].mean()) if len(aligned) else float("nan"),
                    "delta_raw_mean": float(raw_delta.mean()) if len(raw_delta) else float("nan"),
                    "delta_raw_ci_lo": raw_boot.ci_lo,
                    "delta_raw_ci_hi": raw_boot.ci_hi,
                    "effect_mean": boot.mean,
                    "effect_ci_lo": boot.ci_lo,
                    "effect_ci_hi": boot.ci_hi,
                    "p_value": boot.p_value_two_sided,
                    "significant_05": bool(math.isfinite(boot.ci_lo) and math.isfinite(boot.ci_hi) and not (boot.ci_lo <= 0 <= boot.ci_hi)),
                    "significant_01": bool(boot.p_value_two_sided < 0.01) if math.isfinite(boot.p_value_two_sided) else False,
                }
            )
    return pd.DataFrame(rows)


def plot_e1_heatmap(summary: pd.DataFrame, output_path: Path) -> None:
    if summary.empty:
        return
    pivot = summary.pivot(index="transfer_id", columns="edge_removed", values="effect_mean")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(10, 4.8))
    values = pivot.to_numpy(dtype=float)
    vmax = np.nanmax(np.abs(values)) if np.isfinite(values).any() else 1.0
    vmax = max(vmax, 0.01)
    image = axis.imshow(values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    axis.set_xticks(np.arange(len(pivot.columns)), labels=pivot.columns, rotation=25, ha="right")
    axis.set_yticks(np.arange(len(pivot.index)), labels=pivot.index)
    axis.set_title("Week 8 E1 Edge Ablation Effect vs Full")
    for y in range(values.shape[0]):
        for x in range(values.shape[1]):
            value = values[y, x]
            if np.isfinite(value):
                axis.text(x, y, f"{value:.3f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=axis, label="Positive = ablation harms primary metric")
    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def run_e1(
    bundle: GraphBundle,
    smoke_test: bool,
    transfer_filter: str | None,
    edge_filter: str | None,
    seed_filter: int | None,
) -> None:
    if smoke_test:
        redirect_output_root(bundle, "week8_smoke")
    transfers = list(bundle.config["E1_edge_ablation"]["transfers"])
    edges = list(bundle.config["E1_edge_ablation"]["edges_to_remove"])
    seeds = [int(seed) for seed in bundle.config["seeds"]]
    if transfer_filter:
        transfers = [transfer_filter]
    if edge_filter:
        edges = ["none", edge_filter] if edge_filter != "none" else ["none"]
    if seed_filter is not None:
        seeds = [int(seed_filter)]
    if smoke_test and not transfer_filter:
        transfers = ["T3_DiagnoseFrance"]
    if smoke_test and not edge_filter:
        edges = ["none", "hosted_on"]
    if smoke_test and seed_filter is None:
        seeds = seeds[:1]

    rows: list[dict[str, Any]] = []
    for transfer_id in transfers:
        for seed in seeds:
            for edge_removed in edges:
                metrics, _history = run_one_edge_ablation(
                    bundle=bundle,
                    transfer_id=transfer_id,
                    edge_removed=edge_removed,
                    seed=seed,
                    smoke_test=smoke_test,
                )
                rows.append(metrics)
                raw = pd.DataFrame(rows)
                save_dataframe(raw, bundle.output_paths["metrics"] / "E1_edge_ablation_raw_runs.csv")

    raw = pd.DataFrame(rows)
    save_dataframe(raw, bundle.output_paths["metrics"] / "E1_edge_ablation_raw_runs.csv")
    summary = summarize_e1(bundle, raw)
    save_dataframe(summary, bundle.output_paths["metrics"] / "E1_edge_ablation_summary.csv")
    plot_e1_heatmap(summary, bundle.output_paths["plots"] / "E1_edge_ablation_heatmap.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Week 8 E1 edge-channel ablation.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "week8.yaml"))
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--transfer", default=None)
    parser.add_argument("--edge", default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle = load_graph_bundle(args.config)
    run_e1(
        bundle=bundle,
        smoke_test=args.smoke_test,
        transfer_filter=args.transfer,
        edge_filter=args.edge,
        seed_filter=args.seed,
    )


if __name__ == "__main__":
    main()
