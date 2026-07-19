"""Train a single configuration; shared by the sweep entry points."""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.hetero_full import WeightedHeteroGNNClassifier
from src.train.utils import (
    GraphBundle,
    build_prediction_frame,
    build_primary_task_frame,
    build_same_region_task_frame,
    choose_threshold_by_youden,
    compute_binary_metrics,
    compute_sample_counts,
    load_graph_bundle,
    make_common_manifest,
    make_pooled_primary_split,
    make_same_region_folds,
    make_split_audit,
    metric_columns,
    plot_mean_roc_pr,
    record_run_manifest,
    resolve_device,
    save_dataframe,
    set_random_seed,
    summarise_runs,
    utc_now_iso,
)


class WebsiteMLP(torch.nn.Module):
    """Website M L P (PyTorch module)."""
    def __init__(self, input_dim: int, hidden_dims: list[int], dropout: float) -> None:
        """Initialise the instance."""
        super().__init__()
        layers: list[torch.nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    torch.nn.Linear(prev_dim, hidden_dim),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(dropout),
                ]
            )
            prev_dim = hidden_dim
        layers.append(torch.nn.Linear(prev_dim, 2))
        self.network = torch.nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Run the forward pass."""
        return self.network(features)


def feature_columns(bundle: GraphBundle) -> list[str]:
    """Feature columns."""
    return list(bundle.feature_schema["Website"]["feature_columns"])


def extract_tabular_arrays(
    split_frame: pd.DataFrame,
    feature_names: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Extract tabular arrays."""
    features = split_frame[feature_names].to_numpy(dtype=np.float32)
    labels = split_frame["label"].to_numpy(dtype=np.int64)
    return features, labels


def evaluate_split_predictions(
    split_frame: pd.DataFrame,
    probabilities: dict[int, float],
    threshold: float,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Evaluate split predictions."""
    prediction_frame = build_prediction_frame(
        split_frame=split_frame,
        probabilities=probabilities,
        threshold=threshold,
        model_name="",
        task_name=str(split_frame["task"].iloc[0]),
        seed=int(split_frame["seed"].iloc[0]) if "seed" in split_frame.columns else None,
        fold=int(split_frame["fold"].iloc[0]) if "fold" in split_frame.columns else None,
    )
    test_frame = prediction_frame[prediction_frame["split"] == "test"].copy()
    metrics = compute_binary_metrics(
        y_true=test_frame["label"].to_numpy(dtype=int),
        y_score=test_frame["pred_prob_illegal"].to_numpy(dtype=float),
        threshold=threshold,
    )
    val_frame = prediction_frame[prediction_frame["split"] == "val"].copy()
    metrics["val_roc_auc"] = compute_binary_metrics(
        y_true=val_frame["label"].to_numpy(dtype=int),
        y_score=val_frame["pred_prob_illegal"].to_numpy(dtype=float),
        threshold=threshold,
    )["roc_auc"]
    return metrics, prediction_frame


def run_logistic_regression(
    bundle: GraphBundle,
    split_frame: pd.DataFrame,
    model_name: str,
    seed: int | None = None,
    fold: int | None = None,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Run logistic regression."""
    feature_names = feature_columns(bundle)
    train_frame = split_frame[split_frame["split"] == "train"].copy()
    val_frame = split_frame[split_frame["split"] == "val"].copy()
    all_frame = split_frame.copy()
    train_x, train_y = extract_tabular_arrays(train_frame, feature_names)
    val_x, val_y = extract_tabular_arrays(val_frame, feature_names)
    all_x, _ = extract_tabular_arrays(all_frame, feature_names)

    classifier = LogisticRegression(max_iter=2000, solver="liblinear", random_state=seed or 42)
    classifier.fit(train_x, train_y)
    val_scores = classifier.predict_proba(val_x)[:, 1]
    threshold = choose_threshold_by_youden(val_y, val_scores)
    all_scores = classifier.predict_proba(all_x)[:, 1]
    probability_map = dict(zip(all_frame["graph_node_index"], all_scores))
    metrics, prediction_frame = evaluate_split_predictions(all_frame, probability_map, threshold)
    prediction_frame["model"] = model_name
    history = pd.DataFrame(
        [
            {
                "epoch": 0,
                "train_loss": float("nan"),
                "val_roc_auc": metrics["val_roc_auc"],
                "stopped_early": False,
            }
        ]
    )
    return metrics, prediction_frame, history


def run_mlp(
    bundle: GraphBundle,
    split_frame: pd.DataFrame,
    model_name: str,
    seed: int,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Run mlp."""
    set_random_seed(seed)
    config = bundle.config["model_defaults"]
    device = resolve_device(bundle.config)
    feature_names = feature_columns(bundle)
    train_frame = split_frame[split_frame["split"] == "train"].copy()
    val_frame = split_frame[split_frame["split"] == "val"].copy()
    all_frame = split_frame.copy()

    train_x, train_y = extract_tabular_arrays(train_frame, feature_names)
    val_x, val_y = extract_tabular_arrays(val_frame, feature_names)
    all_x, _ = extract_tabular_arrays(all_frame, feature_names)

    train_tensor = torch.tensor(train_x, device=device)
    train_labels = torch.tensor(train_y, dtype=torch.long, device=device)
    val_tensor = torch.tensor(val_x, device=device)
    all_tensor = torch.tensor(all_x, device=device)

    model = WebsiteMLP(
        input_dim=train_tensor.shape[1],
        hidden_dims=list(config["mlp_hidden_dims"]),
        dropout=float(config["dropout"]),
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["lr"]),
        weight_decay=float(config["weight_decay"]),
    )

    best_state: dict[str, Any] | None = None
    best_val_auc = float("-inf")
    best_epoch = 0
    patience = int(config["patience"])
    history_rows: list[dict[str, Any]] = []

    for epoch in range(1, int(config["max_epochs"]) + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(train_tensor)
        loss = F.cross_entropy(logits, train_labels)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(val_tensor)
            val_scores = torch.softmax(val_logits, dim=1)[:, 1].detach().cpu().numpy()
        val_auc = compute_binary_metrics(val_y, val_scores, threshold=0.5)["roc_auc"]
        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": float(loss.item()),
                "val_roc_auc": float(val_auc),
            }
        )
        if val_auc > best_val_auc:
            best_val_auc = float(val_auc)
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
        elif epoch - best_epoch >= patience:
            break

    if best_state is None:
        raise RuntimeError("MLP training failed to produce a best state")

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_logits = model(val_tensor)
        val_scores = torch.softmax(val_logits, dim=1)[:, 1].detach().cpu().numpy()
        all_logits = model(all_tensor)
        all_scores = torch.softmax(all_logits, dim=1)[:, 1].detach().cpu().numpy()
    threshold = choose_threshold_by_youden(val_y, val_scores)
    probability_map = dict(zip(all_frame["graph_node_index"], all_scores))
    metrics, prediction_frame = evaluate_split_predictions(all_frame, probability_map, threshold)
    prediction_frame["model"] = model_name
    history = pd.DataFrame(history_rows)
    return metrics, prediction_frame, history


def build_full_label_vector(bundle: GraphBundle, task_frame: pd.DataFrame) -> torch.Tensor:
    """Build full label vector."""
    labels = torch.full((len(bundle.website_frame),), -1, dtype=torch.long)
    for _, row in task_frame.iterrows():
        labels[int(row["graph_node_index"])] = int(row["label"])
    return labels


def run_hetero_gnn(
    bundle: GraphBundle,
    task_frame: pd.DataFrame,
    split_frame: pd.DataFrame,
    model_name: str,
    seed: int,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Run hetero gnn."""
    set_random_seed(seed)
    config = bundle.config["model_defaults"]
    device = resolve_device(bundle.config)
    graph_data = bundle.graph_data.to(device)
    labels = build_full_label_vector(bundle, task_frame).to(device)

    train_indices = torch.tensor(
        split_frame.loc[split_frame["split"] == "train", "graph_node_index"].to_numpy(),
        dtype=torch.long,
        device=device,
    )
    val_indices = torch.tensor(
        split_frame.loc[split_frame["split"] == "val", "graph_node_index"].to_numpy(),
        dtype=torch.long,
        device=device,
    )
    all_indices = split_frame["graph_node_index"].to_numpy(dtype=int)

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

    best_state: dict[str, Any] | None = None
    best_val_auc = float("-inf")
    best_epoch = 0
    history_rows: list[dict[str, Any]] = []

    for epoch in range(1, int(config["max_epochs"]) + 1):
        model.train()
        optimizer.zero_grad()
        logits, _ = model(graph_data)
        loss = F.cross_entropy(logits[train_indices], labels[train_indices])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits, _ = model(graph_data)
            val_scores = torch.softmax(val_logits[val_indices], dim=1)[:, 1].detach().cpu().numpy()
        val_auc = compute_binary_metrics(
            split_frame.loc[split_frame["split"] == "val", "label"].to_numpy(dtype=int),
            val_scores,
            threshold=0.5,
        )["roc_auc"]
        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": float(loss.item()),
                "val_roc_auc": float(val_auc),
            }
        )
        if val_auc > best_val_auc:
            best_val_auc = float(val_auc)
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
        elif epoch - best_epoch >= int(config["patience"]):
            break

    if best_state is None:
        raise RuntimeError("Hetero GNN training failed to produce a best state")

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits, _ = model(graph_data)
        probabilities = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
        val_scores = probabilities[val_indices.detach().cpu().numpy()]
    threshold = choose_threshold_by_youden(
        split_frame.loc[split_frame["split"] == "val", "label"].to_numpy(dtype=int),
        val_scores,
    )
    probability_map = dict(zip(bundle.website_frame["graph_node_index"], probabilities))
    probability_map = {int(node_idx): float(probability_map[int(node_idx)]) for node_idx in all_indices}
    metrics, prediction_frame = evaluate_split_predictions(split_frame, probability_map, threshold)
    prediction_frame["model"] = model_name
    history = pd.DataFrame(history_rows)
    return metrics, prediction_frame, history


def run_single_model(
    bundle: GraphBundle,
    task_frame: pd.DataFrame,
    split_frame: pd.DataFrame,
    task_name: str,
    model_name: str,
    seed: int | None = None,
    fold: int | None = None,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Run single model."""
    if model_name == "logistic_regression":
        metrics, predictions, history = run_logistic_regression(bundle, split_frame, model_name, seed, fold)
    elif model_name == "mlp":
        if seed is None:
            raise ValueError("MLP requires a seed")
        metrics, predictions, history = run_mlp(bundle, split_frame, model_name, seed)
    elif model_name == "hetero_gnn":
        if seed is None:
            raise ValueError("Hetero GNN requires a seed")
        metrics, predictions, history = run_hetero_gnn(bundle, task_frame, split_frame, model_name, seed)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    metrics_row: dict[str, Any] = {
        "task": task_name,
        "model": model_name,
        "created_at_utc": utc_now_iso(),
        **metrics,
    }
    if seed is not None:
        metrics_row["seed"] = seed
    if fold is not None:
        metrics_row["fold"] = fold
    return metrics_row, predictions, history


def save_run_outputs(
    bundle: GraphBundle,
    task_name: str,
    model_name: str,
    split_name: str,
    metrics_row: dict[str, Any],
    prediction_frame: pd.DataFrame,
    history: pd.DataFrame,
    split_frame: pd.DataFrame,
    seed: int | None = None,
    fold: int | None = None,
) -> None:
    """Save run outputs."""
    suffix_parts = [task_name, model_name]
    if seed is not None:
        suffix_parts.append(f"seed{seed}")
    if fold is not None:
        suffix_parts.append(f"fold{fold}")
    suffix = "__".join(suffix_parts)

    save_dataframe(
        prediction_frame,
        bundle.output_paths["predictions"] / f"{suffix}.csv",
    )
    save_dataframe(
        history,
        bundle.output_paths["logs"] / f"{suffix}_history.csv",
    )

    manifest = make_common_manifest(
        bundle=bundle,
        task=task_name,
        model=model_name,
        split_name=split_name,
        sample_counts=compute_sample_counts(split_frame),
        seed=seed,
        fold=fold,
    )
    manifest["metrics"] = metrics_row
    record_run_manifest(bundle.output_paths["runs"] / f"{suffix}.json", manifest)


def run_pooled_primary(bundle: GraphBundle, smoke_test: bool, models: list[str]) -> None:
    """Run pooled primary."""
    task_frame = build_primary_task_frame(bundle)
    seeds = list(bundle.config["seeds"])
    if smoke_test:
        seeds = seeds[:1]

    raw_metrics: list[dict[str, Any]] = []
    pooled_test_predictions: list[pd.DataFrame] = []
    split_audits: list[pd.DataFrame] = []
    split_dir = bundle.output_paths["splits"]
    for seed in seeds:
        split_frame = make_pooled_primary_split(task_frame, seed, bundle.config)
        split_path = split_dir / f"pooled_primary__seed{seed}.csv"
        save_dataframe(split_frame, split_path)
        split_audits.append(make_split_audit(split_frame, task="pooled_primary", seed=seed))

        for model_name in models:
            metrics_row, prediction_frame, history = run_single_model(
                bundle=bundle,
                task_frame=task_frame,
                split_frame=split_frame,
                task_name="pooled_primary",
                model_name=model_name,
                seed=seed,
            )
            raw_metrics.append(metrics_row)
            save_run_outputs(
                bundle=bundle,
                task_name="pooled_primary",
                model_name=model_name,
                split_name=split_path.name,
                metrics_row=metrics_row,
                prediction_frame=prediction_frame,
                history=history,
                split_frame=split_frame,
                seed=seed,
            )
            pooled_test_predictions.append(
                prediction_frame[prediction_frame["split"] == "test"].copy()
            )

    raw_metrics_frame = pd.DataFrame(raw_metrics).sort_values(["model", "seed"]).reset_index(drop=True)
    summary_frame = summarise_runs(raw_metrics_frame, ["task", "model"], metric_columns())
    save_dataframe(raw_metrics_frame, bundle.output_paths["metrics"] / "pooled_primary_raw_runs.csv")
    save_dataframe(summary_frame, bundle.output_paths["metrics"] / "pooled_primary_summary.csv")
    save_dataframe(
        pd.concat(split_audits, ignore_index=True),
        bundle.output_paths["audits"] / "pooled_primary_group_aware_split_audit.csv",
    )
    plot_mean_roc_pr(
        pd.concat(pooled_test_predictions, ignore_index=True),
        bundle.output_paths["plots"] / "pooled_primary_roc_pr.png",
    )


def run_same_region_sensitivity(bundle: GraphBundle, smoke_test: bool, models: list[str]) -> None:
    """Run same region sensitivity."""
    jurisdictions = list(bundle.config["tasks"]["same_region_jurisdictions"])
    if smoke_test:
        jurisdictions = jurisdictions[:1]
    raw_metrics: list[dict[str, Any]] = []
    split_audits: list[pd.DataFrame] = []

    for jurisdiction in jurisdictions:
        task_frame = build_same_region_task_frame(bundle, jurisdiction)
        split_frame = make_same_region_folds(task_frame, jurisdiction, bundle.config)
        split_path = bundle.output_paths["splits"] / f"same_region_sensitivity__{jurisdiction}.csv"
        save_dataframe(split_frame, split_path)
        fold_ids = sorted(split_frame["fold"].unique())
        if smoke_test:
            fold_ids = fold_ids[:1]
        for fold_id in fold_ids:
            fold_frame = split_frame[split_frame["fold"] == fold_id].copy().reset_index(drop=True)
            split_audits.append(
                make_split_audit(
                    fold_frame,
                    task="same_region_sensitivity",
                    jurisdiction=jurisdiction,
                    fold=int(fold_id),
                )
            )
            for model_name in models:
                metrics_row, prediction_frame, history = run_single_model(
                    bundle=bundle,
                    task_frame=task_frame,
                    split_frame=fold_frame,
                    task_name="same_region_sensitivity",
                    model_name=model_name,
                    seed=42 + int(fold_id),
                    fold=int(fold_id),
                )
                metrics_row["jurisdiction_eval"] = jurisdiction
                raw_metrics.append(metrics_row)
                save_run_outputs(
                    bundle=bundle,
                    task_name=f"same_region_sensitivity__{jurisdiction}",
                    model_name=model_name,
                    split_name=split_path.name,
                    metrics_row=metrics_row,
                    prediction_frame=prediction_frame,
                    history=history,
                    split_frame=fold_frame,
                    seed=42 + int(fold_id),
                    fold=int(fold_id),
                )

    raw_metrics_frame = (
        pd.DataFrame(raw_metrics)
        .sort_values(["jurisdiction_eval", "model", "fold"])
        .reset_index(drop=True)
    )
    summary_frame = summarise_runs(
        raw_metrics_frame,
        ["task", "jurisdiction_eval", "model"],
        metric_columns(),
    )
    save_dataframe(raw_metrics_frame, bundle.output_paths["metrics"] / "same_region_sensitivity_raw_runs.csv")
    save_dataframe(summary_frame, bundle.output_paths["metrics"] / "same_region_sensitivity_summary.csv")
    save_dataframe(
        pd.concat(split_audits, ignore_index=True),
        bundle.output_paths["audits"] / "same_region_group_aware_split_audit.csv",
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run Week 4 pooled and same-region baseline experiments.")
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "week4_experiments.yaml"),
        help="Path to the Week 4 experiment config.",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["pooled_primary", "same_region_sensitivity"],
        choices=["pooled_primary", "same_region_sensitivity"],
        help="Which baseline tasks to run.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["logistic_regression", "mlp", "hetero_gnn"],
        choices=["logistic_regression", "mlp", "hetero_gnn"],
        help="Which models to run.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run only the first seed and first same-region fold for quick validation.",
    )
    return parser.parse_args()


def main() -> None:
    """Command-line entry point."""
    args = parse_args()
    bundle = load_graph_bundle(args.config)

    if "pooled_primary" in args.tasks:
        run_pooled_primary(bundle, smoke_test=args.smoke_test, models=args.models)
    if "same_region_sensitivity" in args.tasks:
        run_same_region_sensitivity(bundle, smoke_test=args.smoke_test, models=args.models)


if __name__ == "__main__":
    main()
