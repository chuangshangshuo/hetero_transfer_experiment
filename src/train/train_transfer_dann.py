"""Week-4 pilot DANN transfer runner (superseded; kept for provenance only).

This is NOT the source of the reported Week-7 transfer results. It defaults to
``configs/week4_experiments.yaml``, reads ``model_defaults.domain_loss_lambda_max`` /
``domain_loss_warmup_fraction`` (keys that exist only in the Week-4 config), and
selects the best epoch on *target_val* AUC. The 80 reported transfer runs come from
``src/train/train_transfer.py`` with ``configs/week7_transfer.yaml``, which warms the
GRL lambda over ``dann.lambda_warmup_epochs`` and selects on the *source* validation
split -- see ``selection_policy=best_source_val_after_dann_warmup`` in
``results/week7/metrics/transfer_summary.csv``.
"""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.hetero_full import WeightedHeteroGNNClassifier
from src.train.utils import (
    GraphBundle,
    build_prediction_frame,
    build_transfer_source_target_frames,
    choose_threshold_by_youden,
    compute_binary_metrics,
    compute_sample_counts,
    load_graph_bundle,
    make_common_manifest,
    make_transfer_france_split,
    metric_columns,
    plot_transfer_comparison,
    record_run_manifest,
    resolve_device,
    save_dataframe,
    set_random_seed,
    summarise_runs,
    utc_now_iso,
)
from src.transfer.dann_head import DomainDiscriminator


def build_full_label_vector(bundle: GraphBundle, frame: pd.DataFrame) -> torch.Tensor:
    """Build full label vector."""
    labels = torch.full((len(bundle.website_frame),), -1, dtype=torch.long)
    for _, row in frame.iterrows():
        labels[int(row["graph_node_index"])] = int(row["label"])
    return labels


def evaluate_transfer_predictions(
    split_frame: pd.DataFrame,
    probability_map: dict[int, float],
    threshold: float,
    model_name: str,
    seed: int,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Evaluate transfer predictions."""
    prediction_frame = build_prediction_frame(
        split_frame=split_frame,
        probabilities=probability_map,
        threshold=threshold,
        model_name=model_name,
        task_name="transfer_pooled_to_france",
        seed=seed,
    )
    prediction_frame["used_for_label_supervision"] = prediction_frame["split"].eq("source_train")
    prediction_frame["used_for_domain_supervision"] = prediction_frame["split"].isin(
        ["source_train", "target_adapt_unlabeled"]
    )
    val_frame = prediction_frame[prediction_frame["split"] == "target_val"].copy()
    test_frame = prediction_frame[prediction_frame["split"] == "target_test"].copy()
    metrics = compute_binary_metrics(
        y_true=test_frame["label"].to_numpy(dtype=int),
        y_score=test_frame["pred_prob_illegal"].to_numpy(dtype=float),
        threshold=threshold,
    )
    metrics["val_roc_auc"] = compute_binary_metrics(
        y_true=val_frame["label"].to_numpy(dtype=int),
        y_score=val_frame["pred_prob_illegal"].to_numpy(dtype=float),
        threshold=threshold,
    )["roc_auc"]
    return metrics, prediction_frame


def warmup_lambda(epoch: int, max_epochs: int, warmup_fraction: float, max_lambda: float) -> float:
    """Warmup lambda."""
    warmup_epochs = max(1, int(max_epochs * warmup_fraction))
    if epoch >= warmup_epochs:
        return max_lambda
    return max_lambda * (epoch / warmup_epochs)


def run_transfer_model(
    bundle: GraphBundle,
    split_frame: pd.DataFrame,
    model_name: str,
    seed: int,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Run transfer model."""
    set_random_seed(seed)
    config = bundle.config["model_defaults"]
    device = resolve_device(bundle.config)
    graph_data = bundle.graph_data.to(device)
    labels = build_full_label_vector(bundle, split_frame).to(device)

    source_indices = torch.tensor(
        split_frame.loc[split_frame["split"] == "source_train", "graph_node_index"].to_numpy(),
        dtype=torch.long,
        device=device,
    )
    adapt_indices = torch.tensor(
        split_frame.loc[split_frame["split"] == "target_adapt_unlabeled", "graph_node_index"].to_numpy(),
        dtype=torch.long,
        device=device,
    )
    val_indices = torch.tensor(
        split_frame.loc[split_frame["split"] == "target_val", "graph_node_index"].to_numpy(),
        dtype=torch.long,
        device=device,
    )
    node_feature_dims = {
        node_type: int(graph_data[node_type].x.shape[1]) for node_type in graph_data.node_types
    }
    model = WeightedHeteroGNNClassifier(
        node_feature_dims=node_feature_dims,
        edge_types=list(graph_data.edge_types),
        hidden_dim=int(config["hidden_dim"]),
        dropout=float(config["dropout"]),
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["lr"]),
        weight_decay=float(config["weight_decay"]),
    )
    domain_discriminator: DomainDiscriminator | None = None
    if model_name == "dann":
        domain_discriminator = DomainDiscriminator(
            input_dim=int(config["hidden_dim"]),
            hidden_dim=int(config["hidden_dim"]),
            dropout=float(config["dropout"]),
        ).to(device)
        optimizer = torch.optim.Adam(
            list(model.parameters()) + list(domain_discriminator.parameters()),
            lr=float(config["lr"]),
            weight_decay=float(config["weight_decay"]),
        )

    best_state: dict[str, Any] | None = None
    best_domain_state: dict[str, Any] | None = None
    best_val_auc = float("-inf")
    best_epoch = 0
    history_rows: list[dict[str, Any]] = []

    max_epochs = int(config["max_epochs"])
    patience = int(config["patience"])
    max_lambda = float(config["domain_loss_lambda_max"])
    warmup_fraction = float(config["domain_loss_warmup_fraction"])

    for epoch in range(1, max_epochs + 1):
        model.train()
        if domain_discriminator is not None:
            domain_discriminator.train()
        optimizer.zero_grad()
        logits, embedding_dict = model(graph_data)
        website_embeddings = embedding_dict["Website"]

        classification_loss = F.cross_entropy(logits[source_indices], labels[source_indices])
        domain_loss = torch.tensor(0.0, device=device)
        lambda_value = 0.0
        if model_name == "dann" and domain_discriminator is not None:
            lambda_value = warmup_lambda(epoch, max_epochs, warmup_fraction, max_lambda)
            domain_indices = torch.cat([source_indices, adapt_indices], dim=0)
            domain_labels = torch.cat(
                [
                    torch.zeros(source_indices.shape[0], dtype=torch.long, device=device),
                    torch.ones(adapt_indices.shape[0], dtype=torch.long, device=device),
                ],
                dim=0,
            )
            domain_logits = domain_discriminator(website_embeddings[domain_indices], reverse_scale=1.0)
            domain_loss = F.cross_entropy(domain_logits, domain_labels)
            total_loss = classification_loss + (lambda_value * domain_loss)
        else:
            total_loss = classification_loss

        total_loss.backward()
        optimizer.step()

        model.eval()
        if domain_discriminator is not None:
            domain_discriminator.eval()
        with torch.no_grad():
            val_logits, _ = model(graph_data)
            val_scores = torch.softmax(val_logits[val_indices], dim=1)[:, 1].detach().cpu().numpy()
        val_auc = compute_binary_metrics(
            split_frame.loc[split_frame["split"] == "target_val", "label"].to_numpy(dtype=int),
            val_scores,
            threshold=0.5,
        )["roc_auc"]
        history_rows.append(
            {
                "epoch": epoch,
                "classification_loss": float(classification_loss.item()),
                "domain_loss": float(domain_loss.item()),
                "lambda": float(lambda_value),
                "val_roc_auc": float(val_auc),
                "target_adapt_uses_label_supervision": False,
            }
        )
        if val_auc > best_val_auc:
            best_val_auc = float(val_auc)
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            if domain_discriminator is not None:
                best_domain_state = copy.deepcopy(domain_discriminator.state_dict())
        elif epoch - best_epoch >= patience:
            break

    if best_state is None:
        raise RuntimeError("Transfer training failed to produce a best encoder state")

    model.load_state_dict(best_state)
    if domain_discriminator is not None and best_domain_state is not None:
        domain_discriminator.load_state_dict(best_domain_state)
    model.eval()
    with torch.no_grad():
        logits, _ = model(graph_data)
        all_probabilities = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
        val_scores = all_probabilities[val_indices.detach().cpu().numpy()]

    threshold = choose_threshold_by_youden(
        split_frame.loc[split_frame["split"] == "target_val", "label"].to_numpy(dtype=int),
        val_scores,
    )
    relevant_indices = split_frame["graph_node_index"].to_numpy(dtype=int)
    probability_map = {int(idx): float(all_probabilities[int(idx)]) for idx in relevant_indices}
    metrics, prediction_frame = evaluate_transfer_predictions(
        split_frame=split_frame,
        probability_map=probability_map,
        threshold=threshold,
        model_name=model_name,
        seed=seed,
    )
    history = pd.DataFrame(history_rows)
    metrics_row: dict[str, Any] = {
        "task": "transfer_pooled_to_france",
        "model": model_name,
        "seed": seed,
        "created_at_utc": utc_now_iso(),
        **metrics,
    }
    return metrics_row, prediction_frame, history


def save_run_outputs(
    bundle: GraphBundle,
    model_name: str,
    seed: int,
    metrics_row: dict[str, Any],
    prediction_frame: pd.DataFrame,
    history: pd.DataFrame,
    split_frame: pd.DataFrame,
    split_filename: str,
) -> None:
    """Save run outputs."""
    suffix = f"transfer_pooled_to_france__{model_name}__seed{seed}"
    save_dataframe(prediction_frame, bundle.output_paths["predictions"] / f"{suffix}.csv")
    save_dataframe(history, bundle.output_paths["logs"] / f"{suffix}_history.csv")

    manifest = make_common_manifest(
        bundle=bundle,
        task="transfer_pooled_to_france",
        model=model_name,
        split_name=split_filename,
        sample_counts=compute_sample_counts(split_frame),
        seed=seed,
    )
    manifest["metrics"] = metrics_row
    manifest["target_adapt_uses_label_supervision"] = False
    record_run_manifest(bundle.output_paths["runs"] / f"{suffix}.json", manifest)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run Week 4 transfer experiments for pooled -> France.")
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "week4_experiments.yaml"),
        help="Path to the Week 4 experiment config.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["source_only", "dann"],
        choices=["source_only", "dann"],
        help="Transfer models to run.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run only the first seed for a quick validation pass.",
    )
    return parser.parse_args()


def main() -> None:
    """Command-line entry point."""
    args = parse_args()
    bundle = load_graph_bundle(args.config)
    source_frame, target_frame = build_transfer_source_target_frames(bundle)
    seeds = list(bundle.config["seeds"])
    if args.smoke_test:
        seeds = seeds[:1]

    raw_metrics: list[dict[str, Any]] = []
    for seed in seeds:
        split_frame = make_transfer_france_split(source_frame, target_frame, seed, bundle.config)
        split_path = bundle.output_paths["splits"] / f"transfer_pooled_to_france__seed{seed}.csv"
        save_dataframe(split_frame, split_path)

        for model_name in args.models:
            metrics_row, prediction_frame, history = run_transfer_model(bundle, split_frame, model_name, seed)
            raw_metrics.append(metrics_row)
            save_run_outputs(
                bundle=bundle,
                model_name=model_name,
                seed=seed,
                metrics_row=metrics_row,
                prediction_frame=prediction_frame,
                history=history,
                split_frame=split_frame,
                split_filename=split_path.name,
            )

    raw_metrics_frame = pd.DataFrame(raw_metrics).sort_values(["model", "seed"]).reset_index(drop=True)
    summary_frame = summarise_runs(raw_metrics_frame, ["task", "model"], metric_columns())
    save_dataframe(raw_metrics_frame, bundle.output_paths["metrics"] / "transfer_france_raw_runs.csv")
    save_dataframe(summary_frame, bundle.output_paths["metrics"] / "transfer_france_summary.csv")
    plot_transfer_comparison(
        summary_frame,
        bundle.output_paths["plots"] / "transfer_france_source_only_vs_dann.png",
    )


if __name__ == "__main__":
    main()
