"""E5 ablation runner: structure-only / lexical-only / mixed configurations."""
from __future__ import annotations

import argparse
import copy
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

from src.explain.feature_bucket_transfer import load_week7_split_or_build, redirect_output_root
from src.train.train_transfer import (
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
    summarise_runs,
)


EDGE_REVERSE = {
    "hosted_on": "rev_hosted_on",
    "uses_cert": "rev_uses_cert",
    "uses_ns": "rev_uses_ns",
    "registered_via": "rev_registered_via",
    "referenced_by": "rev_referenced_by",
}

WEBSITE_LEXICAL_COLUMNS = {
    "feat_domain_len_z",
    "feat_root_len_z",
    "feat_domain_segment_count_z",
    "feat_digit_count_z",
    "feat_hyphen_count_z",
    "feat_is_dot_com",
    "feat_is_local_tld",
    "is_dot_com",
    "is_local_tld",
}

INFRA_RELATIONS = {
    "hosted_on",
    "uses_cert",
    "uses_ns",
    "registered_via",
    "rev_hosted_on",
    "rev_uses_cert",
    "rev_uses_ns",
    "rev_registered_via",
}


def _empty_edge_store(store: Any) -> None:
    """Helper: empty edge store."""
    store.edge_index = torch.empty((2, 0), dtype=store.edge_index.dtype)
    store.edge_weight = torch.empty((0,), dtype=store.edge_weight.dtype)


def apply_feature_mode(data: Any, schema: dict[str, Any], mode: str) -> Any:
    """Apply feature mode."""
    data = copy.deepcopy(data.cpu())
    website_x = data["Website"].x.clone()
    columns = list(schema["Website"]["feature_columns"])
    lexical_indices = [
        idx for idx, column in enumerate(columns)
        if column in WEBSITE_LEXICAL_COLUMNS or column.startswith("feat_")
    ]
    cctld_indices = [
        idx for idx, column in enumerate(columns)
        if column in {"feat_is_local_tld", "is_local_tld"}
    ]
    if mode == "full":
        pass
    elif mode == "no_cctld":
        for idx in cctld_indices:
            website_x[:, idx] = 0.0
    elif mode == "zero_website_lexical":
        for idx in lexical_indices:
            website_x[:, idx] = 0.0
    elif mode == "zero_website_all":
        website_x = torch.zeros_like(website_x)
    elif mode == "website_lexical_only":
        keep = set(lexical_indices)
        for idx, column in enumerate(columns):
            if idx not in keep:
                website_x[:, idx] = 0.0
    else:
        raise ValueError(f"Unsupported feature_mode: {mode}")
    data["Website"].x = website_x
    return data


def apply_edge_mode(data: Any, mode: str) -> Any:
    """Apply edge mode."""
    data = copy.deepcopy(data.cpu())
    if mode == "full":
        return data
    if mode == "empty_all":
        remove_relations = {edge_type[1] for edge_type in data.edge_types}
    elif mode == "remove_uses_cert":
        remove_relations = {"uses_cert", "rev_uses_cert"}
    elif mode == "remove_registered_via":
        remove_relations = {"registered_via", "rev_registered_via"}
    elif mode == "infra_only":
        remove_relations = {edge_type[1] for edge_type in data.edge_types if edge_type[1] not in INFRA_RELATIONS}
    else:
        raise ValueError(f"Unsupported edge_mode: {mode}")
    for edge_type in list(data.edge_types):
        if edge_type[1] in remove_relations:
            _empty_edge_store(data[edge_type])
    return data


def apply_e5_config(bundle: GraphBundle, config_id: str) -> Any:
    """Apply E5 config."""
    config = bundle.config["E5_structure_vs_lexical"]["configs"][config_id]
    data = apply_feature_mode(bundle.graph_data, bundle.feature_schema, str(config["feature_mode"]))
    data = apply_edge_mode(data, str(config["edge_mode"]))
    return data


def make_transform_audit(bundle: GraphBundle, config_id: str, graph_data: Any) -> pd.DataFrame:
    """Construct transform audit."""
    config = bundle.config["E5_structure_vs_lexical"]["configs"][config_id]
    original = bundle.graph_data.cpu()
    rows: list[dict[str, Any]] = []
    for edge_type in original.edge_types:
        rows.append(
            {
                "config_id": config_id,
                "feature_mode": str(config["feature_mode"]),
                "edge_mode": str(config["edge_mode"]),
                "node_or_edge": "edge",
                "name": "|".join(edge_type),
                "original_count": int(original[edge_type].edge_index.shape[1]),
                "transformed_count": int(graph_data[edge_type].edge_index.shape[1]),
            }
        )
    original_x = original["Website"].x
    transformed_x = graph_data["Website"].x
    rows.append(
        {
            "config_id": config_id,
            "feature_mode": str(config["feature_mode"]),
            "edge_mode": str(config["edge_mode"]),
            "node_or_edge": "node_feature",
            "name": "Website.x_nonzero_columns",
            "original_count": int((original_x.abs().sum(dim=0) > 0).sum().item()),
            "transformed_count": int((transformed_x.abs().sum(dim=0) > 0).sum().item()),
        }
    )
    return pd.DataFrame(rows)


def make_split_group_overlap_audit(split_frame: pd.DataFrame, config_id: str, transfer_id: str, seed: int) -> pd.DataFrame:
    """Construct split group overlap audit."""
    rows: list[dict[str, Any]] = []
    for role in sorted(split_frame["domain_role"].dropna().unique()):
        part = split_frame[split_frame["domain_role"] == role]
        overlaps = (
            part.groupby("split_group")["split"]
            .nunique()
            .reset_index(name="num_splits")
            .query("num_splits > 1")
        )
        rows.append(
            {
                "config_id": config_id,
                "transfer_id": transfer_id,
                "seed": int(seed),
                "audit_type": "group_overlap",
                "domain_role": role,
                "num_overlapping_groups": int(len(overlaps)),
                "num_rows": int(len(part)),
                "num_groups": int(part["split_group"].nunique()),
                "is_posthoc": True,
            }
        )
    return pd.DataFrame(rows)


def run_one_e5(
    bundle: GraphBundle,
    transfer_id: str,
    config_id: str,
    seed: int,
    smoke_test: bool,
) -> dict[str, Any]:
    """Run one E5."""
    split_frame, split_path = load_week7_split_or_build(bundle, transfer_id, seed)
    encoder_state, encoder_checkpoint = load_encoder_state(bundle, seed)
    graph_data = apply_e5_config(bundle, config_id)
    transform_audit = make_transform_audit(bundle, config_id, graph_data)
    split_audit = make_split_group_overlap_audit(split_frame, config_id, transfer_id, seed)
    train_info, history, probabilities = train_heco_transfer_model(
        bundle=bundle,
        graph_data=graph_data,
        split_frame=split_frame,
        encoder_state=encoder_state,
        seed=seed,
        method="source_only",
        smoke_test=smoke_test,
        purpose="E5_structure_vs_lexical",
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
            "experiment": "E5_structure_vs_lexical",
            "config_id": config_id,
            "feature_mode": str(bundle.config["E5_structure_vs_lexical"]["configs"][config_id]["feature_mode"]),
            "edge_mode": str(bundle.config["E5_structure_vs_lexical"]["configs"][config_id]["edge_mode"]),
            "encoder_checkpoint": encoder_checkpoint,
            "is_posthoc": True,
        }
    )
    suffix = f"E5_{transfer_id}__{config_id}__seed{seed}"
    save_dataframe(history, bundle.output_paths["logs"] / f"{suffix}_history.csv")
    save_dataframe(predictions, bundle.output_paths["predictions"] / f"{suffix}_predictions.csv")
    audit = transform_audit.copy()
    audit["transfer_id"] = transfer_id
    audit["seed"] = seed
    audit["is_posthoc"] = True
    audit = pd.concat([audit, split_audit], ignore_index=True, sort=False)
    save_dataframe(audit, bundle.output_paths["audits"] / f"{suffix}_audit.csv")
    manifest = make_common_manifest(
        bundle=bundle,
        task=f"E5_{transfer_id}",
        model="source_only",
        split_name=split_path.name,
        sample_counts=compute_sample_counts(split_frame),
        seed=seed,
    )
    manifest["encoder_checkpoint"] = encoder_checkpoint
    manifest["e5_config"] = bundle.config["E5_structure_vs_lexical"]["configs"][config_id]
    manifest["is_posthoc"] = True
    manifest["audit_file"] = f"{suffix}_audit.csv"
    manifest["metrics"] = metrics
    record_run_manifest(bundle.output_paths["runs"] / f"{suffix}.json", manifest)
    return metrics


def plot_e5(summary: pd.DataFrame, output_path: Path) -> None:
    """Plot E5."""
    if summary.empty:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    transfers = list(dict.fromkeys(summary["transfer_id"].tolist()))
    config_order = [
        "C1_full",
        "C2_no_cctld",
        "C3_no_website_lexical",
        "C4_graph_only",
        "C5_lex_only",
        "C6_no_edges",
        "C7_no_cert_edge",
        "C8_no_registrar_edge",
        "C9_infra_edges_only",
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
    axes_flat = axes.flatten()
    for axis, transfer_id in zip(axes_flat, transfers):
        part = summary[summary["transfer_id"] == transfer_id].copy()
        part["config_id"] = pd.Categorical(part["config_id"], categories=config_order, ordered=True)
        part = part.sort_values("config_id")
        axis.bar(part["config_id"].astype(str), part["primary_metric_value_mean"], color="#2563eb", alpha=0.75)
        axis.set_title(transfer_id)
        axis.set_ylim(0.0, 1.02)
        axis.tick_params(axis="x", rotation=25)
        axis.grid(axis="y", alpha=0.2)
    for axis in axes_flat[len(transfers) :]:
        axis.axis("off")
    fig.suptitle("Week 8 E5 Structure vs Lexical Primary Metrics")
    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def run_e5(
    bundle: GraphBundle,
    smoke_test: bool,
    transfer_filter: str | None,
    config_filter: str | None,
    seed_filter: int | None,
) -> None:
    """Run E5."""
    if smoke_test:
        redirect_output_root(bundle, "week8_smoke")
    transfers = list(bundle.config["E5_structure_vs_lexical"]["transfers"])
    configs = list(bundle.config["E5_structure_vs_lexical"]["configs"].keys())
    seeds = [int(seed) for seed in bundle.config["seeds"]]
    if transfer_filter:
        transfers = [transfer_filter]
    if config_filter:
        configs = ["C1_full", config_filter] if config_filter != "C1_full" else ["C1_full"]
    if seed_filter is not None:
        seeds = [int(seed_filter)]
    if smoke_test and not transfer_filter:
        transfers = ["T3_DiagnoseFrance"]
    if smoke_test and not config_filter:
        configs = ["C1_full", "C2_no_cctld"]
    if smoke_test and seed_filter is None:
        seeds = seeds[:1]
    rows: list[dict[str, Any]] = []
    for transfer_id in transfers:
        for seed in seeds:
            for config_id in configs:
                rows.append(run_one_e5(bundle, transfer_id, config_id, seed, smoke_test))
                save_dataframe(pd.DataFrame(rows), bundle.output_paths["metrics"] / "E5_structure_vs_lexical.csv")
    raw = pd.DataFrame(rows)
    save_dataframe(raw, bundle.output_paths["metrics"] / "E5_structure_vs_lexical.csv")
    summary = summarise_runs(raw, ["experiment", "transfer_id", "config_id"], ["primary_metric_value"]) if not raw.empty else pd.DataFrame()
    save_dataframe(summary, bundle.output_paths["metrics"] / "E5_structure_vs_lexical_summary.csv")
    plot_e5(summary, bundle.output_paths["plots"] / "E5_config_bar.png")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run Week 8 E5 structure-vs-lexical slicing.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "week8.yaml"))
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--transfer", default=None)
    parser.add_argument("--e5-config", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--output-root",
        default="week8_patch",
        help="Write corrected post-hoc E5 outputs under output/<output-root>.",
    )
    return parser.parse_args()


def main() -> None:
    """Command-line entry point."""
    args = parse_args()
    bundle = load_graph_bundle(args.config)
    if not args.smoke_test:
        redirect_output_root(bundle, args.output_root)
    run_e5(bundle, args.smoke_test, args.transfer, args.e5_config, args.seed)


if __name__ == "__main__":
    main()
