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

from src.models.hetero_full_with_heco import HeCoFineTuneClassifier
from src.models.heco_head import HeCoModel
from src.models.view_generators import build_metapath_artifacts
from src.train.train_contrastive import make_heco_model, pretrain_heco
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
    make_split_audit,
    metric_columns,
    record_run_manifest,
    resolve_device,
    resolve_workspace_path,
    save_dataframe,
    set_random_seed,
    summarise_runs,
    utc_now_iso,
)


HEAD_ABLATION_TYPES = ["attention", "mlp_64", "linear", "fixed_mean", "schema_only", "metapath_only"]


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in str(value)).strip("_")


def redirect_output_root(bundle: GraphBundle, root_name: str) -> None:
    workspace = Path(bundle.config["workspace_root"])
    for key in list(bundle.output_paths.keys()):
        subdir = key if key != "root" else ""
        path = workspace / "output" / root_name / subdir
        path.mkdir(parents=True, exist_ok=True)
        bundle.output_paths[key] = path
        bundle.config["output"][key] = str(Path("output") / root_name / subdir) if subdir else str(Path("output") / root_name)


def build_labels(bundle: GraphBundle, task_frame: pd.DataFrame) -> torch.Tensor:
    labels = torch.full((len(bundle.website_frame),), -1, dtype=torch.long)
    for _, row in task_frame.iterrows():
        labels[int(row["graph_node_index"])] = int(row["label"])
    return labels


def build_metapath_adjacency(bundle: GraphBundle, device: torch.device) -> dict[str, torch.Tensor]:
    artifacts = build_metapath_artifacts(
        data=bundle.graph_data.cpu(),
        metapath_relations=dict(bundle.config["heco"]["metapaths"]),
        top_k=int(bundle.config["heco"]["top_k_positive"]),
        min_metapath_support=int(bundle.config["heco"]["min_metapath_support"]),
    )
    return {name: adjacency.to(device) for name, adjacency in artifacts.adjacency.items()}


def checkpoint_candidates(bundle: GraphBundle, seed: int) -> list[Path]:
    heco_config = bundle.config["heco"]
    candidates = [
        resolve_workspace_path(bundle.config, heco_config["generated_checkpoint_template"].format(seed=seed)),
        resolve_workspace_path(bundle.config, heco_config["week5_checkpoint_template"].format(seed=seed)),
    ]
    return candidates


def load_or_pretrain_encoder(
    bundle: GraphBundle,
    seed: int,
    smoke_test: bool,
) -> tuple[HeCoModel, dict[str, Any], pd.DataFrame | None]:
    temperature = float(bundle.config["heco"]["selected_temperature"])
    for checkpoint_path in checkpoint_candidates(bundle, seed):
        if checkpoint_path.exists():
            encoder = make_heco_model(bundle)
            state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            encoder.load_state_dict(state_dict)
            return (
                encoder,
                {
                    "seed": seed,
                    "temperature": temperature,
                    "encoder_checkpoint": str(checkpoint_path),
                    "encoder_source": "existing_checkpoint",
                },
                None,
            )

    encoder, history, pretrain_summary = pretrain_heco(
        bundle=bundle,
        temperature=temperature,
        seed=seed,
        smoke_test=smoke_test,
    )
    generated_path = resolve_workspace_path(
        bundle.config,
        bundle.config["heco"]["generated_checkpoint_template"].format(seed=seed),
    )
    generated_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(encoder.cpu().state_dict(), generated_path)
    return (
        encoder,
        {
            **pretrain_summary,
            "encoder_checkpoint": str(generated_path),
            "encoder_source": "week6_pretrain",
        },
        history,
    )


def evaluate_predictions(
    split_frame: pd.DataFrame,
    probabilities: np.ndarray,
    threshold: float,
    seed: int,
    model_name: str,
    task_name: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    probability_map = dict(zip(split_frame["graph_node_index"], probabilities))
    prediction_frame = build_prediction_frame(
        split_frame=split_frame,
        probabilities=probability_map,
        threshold=threshold,
        model_name=model_name,
        task_name=task_name,
        seed=seed,
    )
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
        "task": task_name,
        "model": model_name,
        "seed": seed,
        "temperature": float(split_frame["temperature"].iloc[0]),
        "created_at_utc": utc_now_iso(),
        **metrics,
    }
    return metrics_row, prediction_frame


def run_single_seed_finetune(
    bundle: GraphBundle,
    task_frame: pd.DataFrame,
    split_frame: pd.DataFrame,
    pretrained_encoder: HeCoModel,
    seed: int,
    smoke_test: bool,
    optimizer_strategy: str,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, dict[str, torch.Tensor]]:
    set_random_seed(seed)
    config = bundle.config["finetune"]
    device = resolve_device(bundle.config)
    graph_data = bundle.graph_data.to(device)
    metapath_adjacency = build_metapath_adjacency(bundle, device)
    encoder = make_heco_model(bundle).to(device)
    encoder.load_state_dict(copy.deepcopy(pretrained_encoder.state_dict()))
    active_head_type = str(config["head_type"])
    model = HeCoFineTuneClassifier(
        heco_encoder=encoder,
        head_type=active_head_type,
        hidden_dim=int(bundle.config["heco"]["hidden_dim"]),
        dropout=float(config["dropout"]),
    ).to(device)
    labels = build_labels(bundle, task_frame).to(device)
    train_idx = torch.tensor(split_frame.loc[split_frame["split"] == "train", "graph_node_index"].to_numpy(), device=device)
    val_idx = torch.tensor(split_frame.loc[split_frame["split"] == "val", "graph_node_index"].to_numpy(), device=device)

    if optimizer_strategy == "discriminative":
        optimizer = torch.optim.Adam(
            [
                {"params": model.encoder_parameters(), "lr": float(config["encoder_lr"])},
                {"params": model.head_parameters(), "lr": float(config["classifier_lr"])},
            ],
            weight_decay=float(config["weight_decay"]),
        )
        model_name = "heco_disc_lr_finetune"
        task_name = "pooled_primary_finetune"
    elif optimizer_strategy == "uniform":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=float(config["uniform_lr"]),
            weight_decay=float(config["weight_decay"]),
        )
        model_name = "heco_uniform_lr_finetune"
        task_name = "pooled_primary_finetune_uniform_lr"
    else:
        raise ValueError(f"Unsupported optimizer_strategy: {optimizer_strategy}")
    if bool(bundle.config.get("active_head_ablation", False)):
        model_name = f"heco_{safe_name(active_head_type)}_head"
        task_name = "head_ablation"
    max_epochs = 5 if smoke_test else int(config["max_epochs"])
    patience = max_epochs if smoke_test else int(config["patience"])
    min_epochs = 0 if smoke_test else int(config["min_epochs_before_early_stop"])
    best_state: dict[str, Any] | None = None
    best_val_auc = float("-inf")
    best_epoch = 0
    history_rows: list[dict[str, Any]] = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits, _, aux = model(graph_data, metapath_adjacency)
        loss = F.cross_entropy(logits[train_idx], labels[train_idx])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits, _, val_aux = model(graph_data, metapath_adjacency)
            val_scores = torch.softmax(val_logits[val_idx], dim=1)[:, 1].detach().cpu().numpy()
        val_auc = compute_binary_metrics(
            split_frame.loc[split_frame["split"] == "val", "label"].to_numpy(dtype=int),
            val_scores,
            threshold=0.5,
        )["roc_auc"]
        attention = val_aux.get("attention_weights")
        if attention is not None:
            attention_mean = attention[val_idx].detach().cpu().mean(dim=0).numpy()
            schema_attention = float(attention_mean[0])
            metapath_attention = float(attention_mean[1])
        else:
            schema_attention = float("nan")
            metapath_attention = float("nan")
        history_rows.append(
            {
                "epoch": epoch,
                "loss": float(loss.item()),
                "val_roc_auc": float(val_auc),
                "schema_attention_mean": schema_attention,
                "metapath_attention_mean": metapath_attention,
            }
        )
        if val_auc > best_val_auc:
            best_val_auc = float(val_auc)
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
        elif epoch >= min_epochs and epoch - best_epoch >= patience:
            break

    if best_state is None:
        raise RuntimeError("Week 6 finetune failed to produce a best state")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits, embeddings, aux = model(graph_data, metapath_adjacency)
        probabilities_all = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
        val_scores = probabilities_all[val_idx.detach().cpu().numpy()]
        embedding_payload = {
            "fused_embedding": embeddings.detach().cpu(),
            "schema_embedding": aux["schema_embedding"].detach().cpu(),
            "metapath_embedding": aux["metapath_embedding"].detach().cpu(),
        }
        if "attention_weights" in aux:
            embedding_payload["attention_weights"] = aux["attention_weights"].detach().cpu()

    threshold = choose_threshold_by_youden(
        split_frame.loc[split_frame["split"] == "val", "label"].to_numpy(dtype=int),
        val_scores,
    )
    split_probabilities = probabilities_all[split_frame["graph_node_index"].to_numpy(dtype=int)]
    metrics, predictions = evaluate_predictions(
        split_frame,
        split_probabilities,
        threshold,
        seed,
        model_name,
        task_name,
    )
    metrics["best_epoch"] = int(best_epoch)
    metrics["best_val_roc_auc"] = float(best_val_auc)
    metrics["head_type"] = active_head_type
    checkpoint_path = bundle.output_paths["checkpoints"] / f"{model_name}__seed{seed}.pt"
    torch.save(model.cpu().state_dict(), checkpoint_path)
    metrics["finetune_checkpoint"] = str(checkpoint_path)
    return metrics, predictions, pd.DataFrame(history_rows), embedding_payload


def save_finetune_outputs(
    bundle: GraphBundle,
    metrics: dict[str, Any],
    predictions: pd.DataFrame,
    history: pd.DataFrame,
    split_frame: pd.DataFrame,
    pretrain_summary: dict[str, Any],
    optimizer_strategy: str,
) -> None:
    seed = int(metrics["seed"])
    suffix = f"heco_finetune__seed{seed}"
    if optimizer_strategy == "uniform":
        suffix = f"heco_finetune_uniform_lr__seed{seed}"
    save_dataframe(predictions, bundle.output_paths["predictions"] / f"{suffix}.csv")
    save_dataframe(history, bundle.output_paths["logs"] / f"{suffix}_history.csv")
    manifest = make_common_manifest(
        bundle=bundle,
        task=str(metrics["task"]),
        model=str(metrics["model"]),
        split_name=f"pooled_primary_finetune__seed{seed}.csv",
        sample_counts=compute_sample_counts(split_frame),
        seed=seed,
    )
    manifest["temperature"] = float(metrics["temperature"])
    manifest["pretrain"] = pretrain_summary
    manifest["metrics"] = metrics
    record_run_manifest(bundle.output_paths["runs"] / f"{suffix}.json", manifest)


def save_after_finetune_tsne(
    bundle: GraphBundle,
    embedding_payload: dict[str, torch.Tensor],
    seed: int,
) -> None:
    embedding_tensor = embedding_payload["fused_embedding"].numpy()
    metadata = bundle.website_frame[
        ["graph_node_index", "node_id", "root_domain", "jurisdiction", "sample_tier", "brand", "operator_or_case"]
    ].copy()
    embedding_columns = [f"finetune_emb_{idx:03d}" for idx in range(embedding_tensor.shape[1])]
    embedding_frame = pd.concat(
        [metadata.reset_index(drop=True), pd.DataFrame(embedding_tensor, columns=embedding_columns)],
        axis=1,
    )
    if "attention_weights" in embedding_payload:
        attention = embedding_payload["attention_weights"].numpy()
        embedding_frame["schema_attention"] = attention[:, 0]
        embedding_frame["metapath_attention"] = attention[:, 1]
    embedding_frame["seed"] = seed
    save_dataframe(embedding_frame, bundle.output_paths["embeddings"] / f"heco_finetune_embeddings__seed{seed}.csv")

    tsne_config = bundle.config["visualization"]
    perplexity = min(int(tsne_config["tsne_perplexity"]), max(5, len(embedding_frame) // 4))
    coords = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=int(tsne_config["tsne_random_state"]),
        init="pca",
        learning_rate="auto",
    ).fit_transform(embedding_tensor)
    plot_frame = embedding_frame[
        ["graph_node_index", "node_id", "root_domain", "jurisdiction", "sample_tier", "seed"]
    ].copy()
    plot_frame["tsne_x"] = coords[:, 0]
    plot_frame["tsne_y"] = coords[:, 1]
    save_dataframe(plot_frame, bundle.output_paths["embeddings"] / f"heco_finetune_tsne__seed{seed}.csv")

    tier_colors = {
        "licensed_baseline": "#2563eb",
        "illegal_confirmed_official_single": "#dc2626",
        "illegal_confirmed_official_cross_verified": "#7c2d12",
        "gray_candidate_strong": "#6b7280",
        "gray_candidate_minimum": "#9ca3af",
        "control_legal_commercial": "#059669",
    }
    fig, axis = plt.subplots(figsize=(8, 6))
    for tier, tier_frame in plot_frame.groupby("sample_tier"):
        axis.scatter(
            tier_frame["tsne_x"],
            tier_frame["tsne_y"],
            s=18,
            alpha=0.75,
            label=tier,
            color=tier_colors.get(tier, "#111827"),
        )
    axis.set_title("Week 6 HeCo after finetune t-SNE")
    axis.set_xticks([])
    axis.set_yticks([])
    axis.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(bundle.output_paths["plots"] / "heco_finetune_tsne_sample_tier.png", dpi=170)
    plt.close(fig)


def make_attention_audit_row(
    metrics: dict[str, Any],
    split_frame: pd.DataFrame,
    embedding_payload: dict[str, torch.Tensor],
) -> dict[str, Any]:
    row = {
        "task": metrics["task"],
        "model": metrics["model"],
        "head_type": metrics.get("head_type"),
        "seed": int(metrics["seed"]),
        "schema_attention_mean": float("nan"),
        "metapath_attention_mean": float("nan"),
        "schema_attention_std": float("nan"),
        "metapath_attention_std": float("nan"),
        "schema_attention_min": float("nan"),
        "schema_attention_max": float("nan"),
        "attention_collapse_flag": False,
    }
    attention = embedding_payload.get("attention_weights")
    if attention is None:
        return row
    indices = split_frame["graph_node_index"].to_numpy(dtype=int)
    weights = attention[indices].detach().cpu().numpy()
    schema = weights[:, 0]
    metapath = weights[:, 1]
    row.update(
        {
            "schema_attention_mean": float(schema.mean()),
            "metapath_attention_mean": float(metapath.mean()),
            "schema_attention_std": float(schema.std()),
            "metapath_attention_std": float(metapath.std()),
            "schema_attention_min": float(schema.min()),
            "schema_attention_max": float(schema.max()),
            "attention_collapse_flag": bool(schema.mean() <= 0.05 or schema.mean() >= 0.95),
        }
    )
    return row


def save_head_ablation_outputs(
    bundle: GraphBundle,
    metrics: dict[str, Any],
    predictions: pd.DataFrame,
    history: pd.DataFrame,
    split_frame: pd.DataFrame,
    pretrain_summary: dict[str, Any],
) -> None:
    seed = int(metrics["seed"])
    head = safe_name(str(metrics["head_type"]))
    suffix = f"head_ablation__{head}__seed{seed}"
    save_dataframe(predictions, bundle.output_paths["predictions"] / f"{suffix}.csv")
    save_dataframe(history, bundle.output_paths["logs"] / f"{suffix}_history.csv")
    save_dataframe(split_frame, bundle.output_paths["splits"] / f"{suffix}_split.csv")
    manifest = make_common_manifest(
        bundle=bundle,
        task="head_ablation",
        model=str(metrics["model"]),
        split_name=f"{suffix}_split.csv",
        sample_counts=compute_sample_counts(split_frame),
        seed=seed,
    )
    manifest["temperature"] = float(metrics["temperature"])
    manifest["pretrain"] = pretrain_summary
    manifest["head_type"] = str(metrics["head_type"])
    manifest["is_posthoc"] = True
    manifest["metrics"] = metrics
    record_run_manifest(bundle.output_paths["runs"] / f"{suffix}.json", manifest)


def run_head_ablation(bundle: GraphBundle, smoke_test: bool, head_type_filter: str | None = None) -> None:
    redirect_output_root(bundle, "week6_head_ablation")
    task_frame = build_primary_task_frame(bundle)
    heads = [head_type_filter] if head_type_filter else HEAD_ABLATION_TYPES
    if smoke_test and head_type_filter is None:
        heads = ["attention", "fixed_mean"]
    seeds = [int(seed) for seed in bundle.config["seeds"]]
    if smoke_test:
        seeds = seeds[:1]
    bundle.config["active_head_ablation"] = True
    raw_rows: list[dict[str, Any]] = []
    split_audits: list[pd.DataFrame] = []
    attention_rows: list[dict[str, Any]] = []
    pretrain_rows: list[dict[str, Any]] = []
    original_head = str(bundle.config["finetune"]["head_type"])

    for head_type in heads:
        bundle.config["finetune"]["head_type"] = str(head_type)
        for seed in seeds:
            pretrained_encoder, pretrain_summary, pretrain_history = load_or_pretrain_encoder(bundle, seed, smoke_test)
            pretrain_rows.append({**pretrain_summary, "head_type": head_type})
            if pretrain_history is not None:
                save_dataframe(
                    pretrain_history,
                    bundle.output_paths["logs"] / f"head_ablation_pretrain__{safe_name(head_type)}__seed{seed}_history.csv",
                )
            split_frame = make_pooled_primary_split(task_frame, seed, bundle.config)
            split_frame["temperature"] = float(bundle.config["heco"]["selected_temperature"])
            metrics, predictions, history, embedding_payload = run_single_seed_finetune(
                bundle=bundle,
                task_frame=task_frame,
                split_frame=split_frame,
                pretrained_encoder=pretrained_encoder,
                seed=seed,
                smoke_test=smoke_test,
                optimizer_strategy="discriminative",
            )
            metrics["encoder_source"] = pretrain_summary["encoder_source"]
            metrics["encoder_checkpoint"] = pretrain_summary["encoder_checkpoint"]
            metrics["is_posthoc"] = True
            raw_rows.append(metrics)
            split_audits.append(make_split_audit(split_frame, "head_ablation", seed=seed))
            attention_rows.append(make_attention_audit_row(metrics, split_frame, embedding_payload))
            save_head_ablation_outputs(bundle, metrics, predictions, history, split_frame, pretrain_summary)

    bundle.config["finetune"]["head_type"] = original_head
    raw_frame = pd.DataFrame(raw_rows)
    summary_frame = summarise_runs(
        raw_frame,
        ["task", "head_type", "model"],
        ["roc_auc", "pr_auc", "balanced_accuracy", "macro_f1", "val_roc_auc"],
    ) if not raw_frame.empty else pd.DataFrame()
    save_dataframe(raw_frame, bundle.output_paths["metrics"] / "head_ablation_raw_runs.csv")
    save_dataframe(summary_frame, bundle.output_paths["metrics"] / "head_ablation_summary.csv")
    save_dataframe(pd.concat(split_audits, ignore_index=True), bundle.output_paths["audits"] / "head_ablation_split_audit.csv")
    save_dataframe(pd.DataFrame(attention_rows), bundle.output_paths["audits"] / "attention_collapse_audit.csv")
    save_dataframe(pd.DataFrame(pretrain_rows), bundle.output_paths["audits"] / "head_ablation_checkpoint_audit.csv")


def run_week6(bundle: GraphBundle, smoke_test: bool) -> None:
    task_frame = build_primary_task_frame(bundle)
    seeds = list(bundle.config["seeds"])
    if smoke_test:
        seeds = seeds[:1]
    raw_rows: list[dict[str, Any]] = []
    split_audits: list[pd.DataFrame] = []
    pretrain_rows: list[dict[str, Any]] = []
    optimizer_strategy = str(bundle.config.get("active_optimizer_strategy", "discriminative"))

    for seed in seeds:
        pretrained_encoder, pretrain_summary, pretrain_history = load_or_pretrain_encoder(bundle, int(seed), smoke_test)
        pretrain_rows.append(pretrain_summary)
        if pretrain_history is not None:
            save_dataframe(
                pretrain_history,
                bundle.output_paths["logs"] / f"heco_pretrain__tau{bundle.config['heco']['selected_temperature']}__seed{seed}_history.csv",
            )
        split_frame = make_pooled_primary_split(task_frame, int(seed), bundle.config)
        split_frame["temperature"] = float(bundle.config["heco"]["selected_temperature"])
        save_dataframe(split_frame, bundle.output_paths["splits"] / f"pooled_primary_finetune__seed{seed}.csv")
        split_audits.append(make_split_audit(split_frame, "pooled_primary_finetune", seed=int(seed)))

        metrics, predictions, history, embedding_payload = run_single_seed_finetune(
            bundle=bundle,
            task_frame=task_frame,
            split_frame=split_frame,
            pretrained_encoder=pretrained_encoder,
            seed=int(seed),
            smoke_test=smoke_test,
            optimizer_strategy=optimizer_strategy,
        )
        metrics["encoder_source"] = pretrain_summary["encoder_source"]
        metrics["encoder_checkpoint"] = pretrain_summary["encoder_checkpoint"]
        raw_rows.append(metrics)
        save_finetune_outputs(bundle, metrics, predictions, history, split_frame, pretrain_summary, optimizer_strategy)
        if int(seed) == 42 and optimizer_strategy == "discriminative":
            save_after_finetune_tsne(bundle, embedding_payload, int(seed))

    raw_frame = pd.DataFrame(raw_rows).sort_values("seed").reset_index(drop=True)
    summary_frame = summarise_runs(raw_frame, ["task", "model"], metric_columns())
    stem = "pooled_primary_finetune"
    if optimizer_strategy == "uniform":
        stem = "pooled_primary_finetune_uniform_lr"
    save_dataframe(raw_frame, bundle.output_paths["metrics"] / f"{stem}_raw_runs.csv")
    save_dataframe(summary_frame, bundle.output_paths["metrics"] / f"{stem}_summary.csv")
    save_dataframe(pd.concat(split_audits, ignore_index=True), bundle.output_paths["audits"] / f"{stem}_split_audit.csv")
    save_dataframe(pd.DataFrame(pretrain_rows), bundle.output_paths["audits"] / "heco_pretrain_checkpoint_audit.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Week 6 HeCo discriminative-LR fine-tuning.")
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "week6_heco_finetune.yaml"),
        help="Path to the Week 6 config.",
    )
    parser.add_argument("--smoke-test", action="store_true", help="Run only seed42 with short finetuning.")
    parser.add_argument(
        "--optimizer-strategy",
        choices=["discriminative", "uniform"],
        default="discriminative",
        help="Use discriminative encoder/head LR or the uniform-LR fallback.",
    )
    parser.add_argument(
        "--head-type",
        choices=HEAD_ABLATION_TYPES,
        default=None,
        help="Override the Week 6 finetune head type for a single run or head-ablation filter.",
    )
    parser.add_argument(
        "--head-ablation",
        action="store_true",
        help="Run all configured head types into output/week6_head_ablation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle = load_graph_bundle(args.config)
    bundle.config["active_optimizer_strategy"] = args.optimizer_strategy
    if args.head_ablation:
        run_head_ablation(bundle, smoke_test=args.smoke_test, head_type_filter=args.head_type)
    else:
        if args.head_type is not None:
            bundle.config["finetune"]["head_type"] = args.head_type
        run_week6(bundle, smoke_test=args.smoke_test)


if __name__ == "__main__":
    main()
