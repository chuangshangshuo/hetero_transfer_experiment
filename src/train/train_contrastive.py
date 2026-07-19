"""Week-5 HeCo contrastive pre-training entry point (temperature grid, checkpoints)."""
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
import torch.nn.functional as F
from sklearn.manifold import TSNE

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.heco_head import HeCoContrastiveLoss, HeCoModel
from src.models.view_generators import build_metapath_artifacts
from src.train.utils import (
    GraphBundle,
    build_prediction_frame,
    build_primary_task_frame,
    choose_threshold_by_youden,
    compute_binary_metrics,
    compute_sample_counts,
    load_graph_bundle,
    make_common_manifest,
    make_pooled_primary_split,
    metric_columns,
    record_run_manifest,
    resolve_device,
    save_dataframe,
    set_random_seed,
    summarise_runs,
    utc_now_iso,
)


class LinearProbe(torch.nn.Module):
    """Linear Probe (PyTorch module)."""
    def __init__(self, input_dim: int) -> None:
        """Initialise the instance."""
        super().__init__()
        self.classifier = torch.nn.Linear(input_dim, 2)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Run the forward pass."""
        return self.classifier(features)


class FineTuneClassifier(torch.nn.Module):
    """Fine Tune Classifier (PyTorch module)."""
    def __init__(self, encoder: HeCoModel, embedding_dim: int) -> None:
        """Initialise the instance."""
        super().__init__()
        self.encoder = encoder
        self.classifier = torch.nn.Linear(embedding_dim, 2)

    def forward(self, data, metapath_adjacency: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the forward pass."""
        output = self.encoder(data, metapath_adjacency)
        return self.classifier(output.combined_embedding), output.combined_embedding


def node_feature_dims(bundle: GraphBundle) -> dict[str, int]:
    """Node feature dims."""
    return {node_type: int(bundle.graph_data[node_type].x.shape[1]) for node_type in bundle.graph_data.node_types}


def metapath_names(config: dict[str, Any]) -> list[str]:
    """Metapath names."""
    return list(config["heco"]["metapaths"].keys())


def make_heco_model(bundle: GraphBundle) -> HeCoModel:
    """Construct HeCo model."""
    config = bundle.config["heco"]
    return HeCoModel(
        node_feature_dims=node_feature_dims(bundle),
        edge_types=list(bundle.graph_data.edge_types),
        metapath_names=metapath_names(bundle.config),
        hidden_dim=int(config["hidden_dim"]),
        projection_dim=int(config["projection_dim"]),
        dropout=float(config["dropout"]),
    )


def build_labels(bundle: GraphBundle, task_frame: pd.DataFrame) -> torch.Tensor:
    """Build labels."""
    labels = torch.full((len(bundle.website_frame),), -1, dtype=torch.long)
    for _, row in task_frame.iterrows():
        labels[int(row["graph_node_index"])] = int(row["label"])
    return labels


def pretrain_heco(
    bundle: GraphBundle,
    temperature: float,
    seed: int,
    smoke_test: bool,
) -> tuple[HeCoModel, pd.DataFrame, dict[str, Any]]:
    """Pretrain HeCo."""
    set_random_seed(seed)
    device = resolve_device(bundle.config)
    graph_data = bundle.graph_data.to(device)
    heco_config = bundle.config["heco"]
    artifacts = build_metapath_artifacts(
        data=bundle.graph_data.cpu(),
        metapath_relations=dict(heco_config["metapaths"]),
        top_k=int(heco_config["top_k_positive"]),
        min_metapath_support=int(heco_config["min_metapath_support"]),
    )
    metapath_adjacency = {name: adjacency.to(device) for name, adjacency in artifacts.adjacency.items()}
    positive_mask = artifacts.positive_mask.to(device)

    model = make_heco_model(bundle).to(device)
    loss_fn = HeCoContrastiveLoss(
        temperature=float(temperature),
        hard_negative_ratio=float(heco_config["hard_negative_ratio"]),
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(heco_config["lr"]),
        weight_decay=float(heco_config["weight_decay"]),
    )
    max_epochs = 5 if smoke_test else int(heco_config["pretrain_epochs"])
    patience = max_epochs if smoke_test else int(heco_config["patience"])
    best_state: dict[str, Any] | None = None
    best_loss = float("inf")
    best_epoch = 0
    history_rows: list[dict[str, Any]] = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        optimizer.zero_grad()
        output = model(graph_data, metapath_adjacency)
        loss, stats = loss_fn(output.schema_projection, output.metapath_projection, positive_mask)
        loss.backward()
        optimizer.step()

        current_loss = float(loss.detach().cpu().item())
        history_rows.append(
            {
                "epoch": epoch,
                "temperature": temperature,
                "contrastive_loss": current_loss,
                **stats,
            }
        )
        if current_loss < best_loss:
            best_loss = current_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
        elif epoch - best_epoch >= patience:
            break

    if best_state is None:
        raise RuntimeError("HeCo pretraining did not produce a best state")
    model.load_state_dict(best_state)
    history = pd.DataFrame(history_rows)
    artifact_summary = {
        "temperature": temperature,
        "seed": seed,
        "best_epoch": best_epoch,
        "best_contrastive_loss": best_loss,
        "num_positive_pairs": int(artifacts.positive_mask.sum().item()),
        "avg_positive_targets": float(artifacts.positive_audit["num_positive_targets"].mean()),
    }
    positive_audit = artifacts.positive_audit.merge(
        bundle.website_frame[["graph_node_index", "node_id", "root_domain", "jurisdiction", "sample_tier"]],
        on="graph_node_index",
        how="left",
    )
    save_dataframe(
        positive_audit,
        bundle.output_paths["audits"] / f"heco_positive_pair_audit__tau{temperature}.csv",
    )
    return model, history, artifact_summary


def get_embeddings(
    bundle: GraphBundle,
    model: HeCoModel,
) -> np.ndarray:
    """Return embeddings."""
    device = resolve_device(bundle.config)
    graph_data = bundle.graph_data.to(device)
    artifacts = build_metapath_artifacts(
        data=bundle.graph_data.cpu(),
        metapath_relations=dict(bundle.config["heco"]["metapaths"]),
        top_k=int(bundle.config["heco"]["top_k_positive"]),
        min_metapath_support=int(bundle.config["heco"]["min_metapath_support"]),
    )
    metapath_adjacency = {name: adjacency.to(device) for name, adjacency in artifacts.adjacency.items()}
    model.eval()
    with torch.no_grad():
        output = model(graph_data, metapath_adjacency)
    return output.combined_embedding.detach().cpu().numpy()


def evaluate_probe_predictions(
    split_frame: pd.DataFrame,
    probabilities: np.ndarray,
    threshold: float,
    model_name: str,
    seed: int,
    temperature: float,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Evaluate probe predictions."""
    probability_map = dict(zip(split_frame["graph_node_index"], probabilities))
    prediction_frame = build_prediction_frame(
        split_frame=split_frame,
        probabilities=probability_map,
        threshold=threshold,
        model_name=model_name,
        task_name="heco_pooled_primary",
        seed=seed,
    )
    prediction_frame["temperature"] = temperature
    val_frame = prediction_frame[prediction_frame["split"] == "val"].copy()
    test_frame = prediction_frame[prediction_frame["split"] == "test"].copy()
    metrics = compute_binary_metrics(
        test_frame["label"].to_numpy(dtype=int),
        test_frame["pred_prob_illegal"].to_numpy(dtype=float),
        threshold,
    )
    metrics["val_roc_auc"] = compute_binary_metrics(
        val_frame["label"].to_numpy(dtype=int),
        val_frame["pred_prob_illegal"].to_numpy(dtype=float),
        threshold,
    )["roc_auc"]
    metrics_row = {
        "task": "heco_pooled_primary",
        "model": model_name,
        "seed": seed,
        "temperature": temperature,
        "created_at_utc": utc_now_iso(),
        **metrics,
    }
    return metrics_row, prediction_frame


def run_frozen_linear_probe(
    bundle: GraphBundle,
    embeddings: np.ndarray,
    split_frame: pd.DataFrame,
    temperature: float,
    seed: int,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Run frozen linear probe."""
    set_random_seed(seed)
    config = bundle.config["probe"]["frozen_linear"]
    device = resolve_device(bundle.config)
    embedding_tensor = torch.tensor(embeddings, dtype=torch.float32, device=device)
    train_idx = torch.tensor(split_frame.loc[split_frame["split"] == "train", "graph_node_index"].to_numpy(), device=device)
    val_idx = torch.tensor(split_frame.loc[split_frame["split"] == "val", "graph_node_index"].to_numpy(), device=device)
    labels = build_labels(bundle, split_frame).to(device)

    probe = LinearProbe(input_dim=embedding_tensor.shape[1]).to(device)
    optimizer = torch.optim.Adam(
        probe.parameters(),
        lr=float(config["lr"]),
        weight_decay=float(config["weight_decay"]),
    )
    best_state: dict[str, Any] | None = None
    best_val_auc = float("-inf")
    best_epoch = 0
    history_rows: list[dict[str, Any]] = []
    for epoch in range(1, int(config["max_epochs"]) + 1):
        probe.train()
        optimizer.zero_grad()
        logits = probe(embedding_tensor[train_idx])
        loss = F.cross_entropy(logits, labels[train_idx])
        loss.backward()
        optimizer.step()

        probe.eval()
        with torch.no_grad():
            val_logits = probe(embedding_tensor[val_idx])
            val_scores = torch.softmax(val_logits, dim=1)[:, 1].detach().cpu().numpy()
        val_auc = compute_binary_metrics(
            split_frame.loc[split_frame["split"] == "val", "label"].to_numpy(dtype=int),
            val_scores,
            threshold=0.5,
        )["roc_auc"]
        history_rows.append({"epoch": epoch, "loss": float(loss.item()), "val_roc_auc": float(val_auc)})
        if val_auc > best_val_auc:
            best_val_auc = float(val_auc)
            best_epoch = epoch
            best_state = copy.deepcopy(probe.state_dict())
        elif epoch - best_epoch >= int(config["patience"]):
            break

    if best_state is None:
        raise RuntimeError("Frozen linear probe failed to produce a best state")
    probe.load_state_dict(best_state)
    probe.eval()
    with torch.no_grad():
        logits = probe(embedding_tensor)
        probabilities_all = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
        val_scores = probabilities_all[val_idx.detach().cpu().numpy()]
    threshold = choose_threshold_by_youden(
        split_frame.loc[split_frame["split"] == "val", "label"].to_numpy(dtype=int),
        val_scores,
    )
    split_probabilities = probabilities_all[split_frame["graph_node_index"].to_numpy(dtype=int)]
    metrics, predictions = evaluate_probe_predictions(
        split_frame,
        split_probabilities,
        threshold,
        "heco_frozen_linear",
        seed,
        temperature,
    )
    return metrics, predictions, pd.DataFrame(history_rows)


def run_full_finetune_probe(
    bundle: GraphBundle,
    pretrained_model: HeCoModel,
    split_frame: pd.DataFrame,
    temperature: float,
    seed: int,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Run full finetune probe."""
    set_random_seed(seed)
    config = bundle.config["probe"]["full_finetune"]
    device = resolve_device(bundle.config)
    graph_data = bundle.graph_data.to(device)
    artifacts = build_metapath_artifacts(
        data=bundle.graph_data.cpu(),
        metapath_relations=dict(bundle.config["heco"]["metapaths"]),
        top_k=int(bundle.config["heco"]["top_k_positive"]),
        min_metapath_support=int(bundle.config["heco"]["min_metapath_support"]),
    )
    metapath_adjacency = {name: adjacency.to(device) for name, adjacency in artifacts.adjacency.items()}
    encoder = make_heco_model(bundle).to(device)
    encoder.load_state_dict(copy.deepcopy(pretrained_model.state_dict()))
    model = FineTuneClassifier(encoder, embedding_dim=int(bundle.config["heco"]["hidden_dim"]) * 2).to(device)
    labels = build_labels(bundle, split_frame).to(device)
    train_idx = torch.tensor(split_frame.loc[split_frame["split"] == "train", "graph_node_index"].to_numpy(), device=device)
    val_idx = torch.tensor(split_frame.loc[split_frame["split"] == "val", "graph_node_index"].to_numpy(), device=device)

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
        logits, _ = model(graph_data, metapath_adjacency)
        loss = F.cross_entropy(logits[train_idx], labels[train_idx])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits, _ = model(graph_data, metapath_adjacency)
            val_scores = torch.softmax(val_logits[val_idx], dim=1)[:, 1].detach().cpu().numpy()
        val_auc = compute_binary_metrics(
            split_frame.loc[split_frame["split"] == "val", "label"].to_numpy(dtype=int),
            val_scores,
            threshold=0.5,
        )["roc_auc"]
        history_rows.append({"epoch": epoch, "loss": float(loss.item()), "val_roc_auc": float(val_auc)})
        if val_auc > best_val_auc:
            best_val_auc = float(val_auc)
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
        elif epoch - best_epoch >= int(config["patience"]):
            break

    if best_state is None:
        raise RuntimeError("Full fine-tune probe failed to produce a best state")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits, _ = model(graph_data, metapath_adjacency)
        probabilities_all = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
        val_scores = probabilities_all[val_idx.detach().cpu().numpy()]
    threshold = choose_threshold_by_youden(
        split_frame.loc[split_frame["split"] == "val", "label"].to_numpy(dtype=int),
        val_scores,
    )
    split_probabilities = probabilities_all[split_frame["graph_node_index"].to_numpy(dtype=int)]
    metrics, predictions = evaluate_probe_predictions(
        split_frame,
        split_probabilities,
        threshold,
        "heco_full_finetune",
        seed,
        temperature,
    )
    return metrics, predictions, pd.DataFrame(history_rows)


def save_probe_outputs(
    bundle: GraphBundle,
    metrics: dict[str, Any],
    predictions: pd.DataFrame,
    history: pd.DataFrame,
    split_frame: pd.DataFrame,
) -> None:
    """Save probe outputs."""
    suffix = f"{metrics['model']}__tau{metrics['temperature']}__seed{metrics['seed']}"
    save_dataframe(predictions, bundle.output_paths["predictions"] / f"{suffix}.csv")
    save_dataframe(history, bundle.output_paths["logs"] / f"{suffix}_history.csv")
    manifest = make_common_manifest(
        bundle=bundle,
        task="heco_pooled_primary",
        model=str(metrics["model"]),
        split_name=f"heco_pooled_primary__seed{metrics['seed']}.csv",
        sample_counts=compute_sample_counts(split_frame),
        seed=int(metrics["seed"]),
    )
    manifest["temperature"] = float(metrics["temperature"])
    manifest["metrics"] = metrics
    record_run_manifest(bundle.output_paths["runs"] / f"{suffix}.json", manifest)


def save_embeddings(bundle: GraphBundle, embeddings: np.ndarray, temperature: float, seed: int) -> pd.DataFrame:
    """Save embeddings."""
    metadata_frame = bundle.website_frame[
        ["graph_node_index", "node_id", "root_domain", "jurisdiction", "sample_tier", "brand", "operator_or_case"]
    ].copy()
    embedding_columns = [f"emb_{dim_idx:03d}" for dim_idx in range(embeddings.shape[1])]
    embedding_values = pd.DataFrame(embeddings, columns=embedding_columns)
    embedding_frame = pd.concat([metadata_frame.reset_index(drop=True), embedding_values], axis=1)
    embedding_frame["temperature"] = temperature
    embedding_frame["seed"] = seed
    save_dataframe(embedding_frame, bundle.output_paths["embeddings"] / f"heco_embeddings__tau{temperature}__seed{seed}.csv")
    return embedding_frame


def plot_tsne(bundle: GraphBundle, embedding_frame: pd.DataFrame, temperature: float, seed: int) -> None:
    """Plot tsne."""
    embedding_columns = [column for column in embedding_frame.columns if column.startswith("emb_")]
    embeddings = embedding_frame[embedding_columns].to_numpy(dtype=np.float32)
    tsne_config = bundle.config["visualization"]
    perplexity = min(int(tsne_config["tsne_perplexity"]), max(5, len(embedding_frame) // 4))
    coords = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=int(tsne_config["tsne_random_state"]),
        init="pca",
        learning_rate="auto",
    ).fit_transform(embeddings)
    tsne_frame = embedding_frame[
        ["graph_node_index", "node_id", "root_domain", "jurisdiction", "sample_tier", "brand", "operator_or_case"]
    ].copy()
    tsne_frame["tsne_x"] = coords[:, 0]
    tsne_frame["tsne_y"] = coords[:, 1]
    text_blob = (
        tsne_frame["root_domain"].fillna("")
        + " "
        + tsne_frame["brand"].fillna("")
        + " "
        + tsne_frame["operator_or_case"].fillna("")
    ).str.lower()
    normalized_text_blob = text_blob.str.replace("-", "", regex=False).str.replace("cazino", "casino", regex=False)
    family_keywords = list(tsne_config["denmark_family_keywords"])
    tsne_frame["denmark_family_keyword"] = ""
    for keyword in family_keywords:
        normalized_keyword = keyword.lower().replace("-", "").replace("cazino", "casino")
        mask = (tsne_frame["jurisdiction"] == "Denmark") & normalized_text_blob.str.contains(
            normalized_keyword,
            regex=False,
        )
        tsne_frame.loc[mask, "denmark_family_keyword"] = keyword
    save_dataframe(tsne_frame, bundle.output_paths["embeddings"] / f"heco_tsne__tau{temperature}__seed{seed}.csv")

    tier_colors = {
        "licensed_baseline": "#2563eb",
        "illegal_confirmed_official_single": "#dc2626",
        "illegal_confirmed_official_cross_verified": "#7c2d12",
        "gray_candidate_strong": "#6b7280",
        "gray_candidate_minimum": "#9ca3af",
        "control_legal_commercial": "#059669",
    }
    fig, axis = plt.subplots(figsize=(8, 6))
    for tier, tier_frame in tsne_frame.groupby("sample_tier"):
        axis.scatter(
            tier_frame["tsne_x"],
            tier_frame["tsne_y"],
            s=18,
            alpha=0.75,
            label=tier,
            color=tier_colors.get(tier, "#111827"),
        )
    axis.set_title("Week 5 HeCo t-SNE by Sample Tier")
    axis.set_xticks([])
    axis.set_yticks([])
    axis.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(bundle.output_paths["plots"] / "heco_tsne_sample_tier.png", dpi=170)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8, 6))
    axis.scatter(tsne_frame["tsne_x"], tsne_frame["tsne_y"], s=12, alpha=0.20, color="#9ca3af", label="other")
    for keyword in family_keywords:
        family_frame = tsne_frame[tsne_frame["denmark_family_keyword"] == keyword]
        if family_frame.empty:
            continue
        axis.scatter(family_frame["tsne_x"], family_frame["tsne_y"], s=42, alpha=0.95, label=keyword)
    axis.set_title("Week 5 Denmark Family t-SNE Check")
    axis.set_xticks([])
    axis.set_yticks([])
    axis.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(bundle.output_paths["plots"] / "heco_tsne_denmark_families.png", dpi=170)
    plt.close(fig)


def run_week5(bundle: GraphBundle, smoke_test: bool) -> None:
    """Run week5."""
    seed = int(bundle.config["seeds"][0])
    task_frame = build_primary_task_frame(bundle)
    split_frame = make_pooled_primary_split(task_frame, seed, bundle.config)
    save_dataframe(split_frame, bundle.output_paths["splits"] / f"heco_pooled_primary__seed{seed}.csv")

    temperatures = list(bundle.config["heco"]["temperatures"])
    if smoke_test:
        temperatures = [0.5]

    grid_rows: list[dict[str, Any]] = []
    probe_rows: list[dict[str, Any]] = []
    pretrained: dict[float, HeCoModel] = {}
    for temperature in temperatures:
        model, pretrain_history, artifact_summary = pretrain_heco(bundle, float(temperature), seed, smoke_test)
        pretrained[float(temperature)] = copy.deepcopy(model).cpu()
        save_dataframe(
            pretrain_history,
            bundle.output_paths["logs"] / f"heco_pretrain__tau{temperature}__seed{seed}_history.csv",
        )
        torch.save(
            model.cpu().state_dict(),
            bundle.output_paths["checkpoints"] / f"heco_encoder__tau{temperature}__seed{seed}.pt",
        )
        embeddings = get_embeddings(bundle, model)
        frozen_metrics, frozen_predictions, frozen_history = run_frozen_linear_probe(
            bundle,
            embeddings,
            split_frame,
            float(temperature),
            seed,
        )
        full_metrics, full_predictions, full_history = run_full_finetune_probe(
            bundle,
            model,
            split_frame,
            float(temperature),
            seed,
        )
        save_probe_outputs(bundle, frozen_metrics, frozen_predictions, frozen_history, split_frame)
        save_probe_outputs(bundle, full_metrics, full_predictions, full_history, split_frame)
        probe_rows.extend([frozen_metrics, full_metrics])
        grid_rows.append(
            {
                **artifact_summary,
                "frozen_linear_roc_auc": frozen_metrics["roc_auc"],
                "full_finetune_roc_auc": full_metrics["roc_auc"],
                "frozen_linear_val_roc_auc": frozen_metrics["val_roc_auc"],
                "full_finetune_val_roc_auc": full_metrics["val_roc_auc"],
            }
        )

    probe_frame = pd.DataFrame(probe_rows).sort_values(["model", "temperature"]).reset_index(drop=True)
    grid_frame = pd.DataFrame(grid_rows).sort_values("temperature").reset_index(drop=True)
    grid_frame["selected_by"] = ""
    best_idx = int(grid_frame["frozen_linear_val_roc_auc"].idxmax())
    best_temperature = float(grid_frame.loc[best_idx, "temperature"])
    grid_frame.loc[best_idx, "selected_by"] = "max_frozen_linear_val_roc_auc"
    summary_frame = summarise_runs(probe_frame, ["task", "model"], metric_columns())
    save_dataframe(grid_frame, bundle.output_paths["metrics"] / "heco_temperature_grid_summary.csv")
    save_dataframe(probe_frame, bundle.output_paths["metrics"] / "heco_probe_raw_runs.csv")
    save_dataframe(summary_frame, bundle.output_paths["metrics"] / "heco_probe_summary.csv")

    best_model = pretrained[best_temperature]
    embeddings = get_embeddings(bundle, best_model)
    embedding_frame = save_embeddings(bundle, embeddings, best_temperature, seed)
    plot_tsne(bundle, embedding_frame, best_temperature, seed)

    best_frozen = probe_frame[
        (probe_frame["model"] == "heco_frozen_linear") & (probe_frame["temperature"] == best_temperature)
    ].iloc[0]
    acceptance = {
        "created_at_utc": utc_now_iso(),
        "best_temperature": best_temperature,
        "frozen_linear_roc_auc": float(best_frozen["roc_auc"]),
        "threshold": float(bundle.config["acceptance_thresholds"]["frozen_linear_probe_auc_min"]),
        "pass": bool(best_frozen["roc_auc"] >= bundle.config["acceptance_thresholds"]["frozen_linear_probe_auc_min"]),
        "tsne_sample_tier_plot": str(bundle.output_paths["plots"] / "heco_tsne_sample_tier.png"),
        "tsne_denmark_family_plot": str(bundle.output_paths["plots"] / "heco_tsne_denmark_families.png"),
    }
    record_run_manifest(bundle.output_paths["runs"] / "week5_heco_acceptance.json", acceptance)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run Week 5 HeCo contrastive pretraining and probing.")
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "week5_heco.yaml"),
        help="Path to the Week 5 HeCo config.",
    )
    parser.add_argument("--smoke-test", action="store_true", help="Run 5 pretrain epochs at tau=0.5 only.")
    return parser.parse_args()


def main() -> None:
    """Command-line entry point."""
    args = parse_args()
    bundle = load_graph_bundle(args.config)
    run_week5(bundle, smoke_test=args.smoke_test)


if __name__ == "__main__":
    main()
