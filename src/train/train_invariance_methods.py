"""Week-11 P2: IRM and EERM comparison runs on the transfer scenarios."""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.train.train_transfer import (
    build_label_vector,
    build_metapath_adjacency,
    choose_threshold_by_youden,
    evaluate_binary,
    load_encoder_state,
    make_finetune_model,
    summarize_run_metrics,
)
from src.train.utils import load_graph_bundle, set_random_seed, utc_now_iso
from src.transfer.eerm_virtual_env import eerm_loss, make_env_index_dict, make_virtual_environments
from src.transfer.irm_penalty import cosine_warmup_lambda, irm_loss_per_env


def load_config(path: Path) -> dict[str, Any]:
    """Load config."""
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def ensure_output_dirs(config: dict[str, Any]) -> dict[str, Path]:
    """Ensure output directories."""
    root = Path(config["workspace_root"])
    paths = {name: root / rel for name, rel in config["output"].items()}
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def index_tensor(split_frame: pd.DataFrame, split_name: str, device: torch.device) -> torch.Tensor:
    """Index tensor."""
    return torch.tensor(
        split_frame.loc[split_frame["split"] == split_name, "graph_node_index"].to_numpy(dtype=np.int64),
        dtype=torch.long,
        device=device,
    )


def build_real_env_indices(
    split_frame: pd.DataFrame,
    device: torch.device,
    min_env_samples: int,
    require_two_classes: bool,
) -> dict[str, torch.Tensor]:
    """Build real environment indices."""
    envs: dict[str, torch.Tensor] = {}
    source_train = split_frame[split_frame["split"] == "source_train"].copy()
    for env, part in source_train.groupby("jurisdiction"):
        if len(part) < min_env_samples:
            continue
        if require_two_classes and part["label"].nunique() < 2:
            continue
        envs[str(env)] = torch.tensor(part["graph_node_index"].to_numpy(dtype=np.int64), dtype=torch.long, device=device)
    if not envs:
        raise ValueError("No valid real environments remain for IRM")
    return envs


def compute_initial_embeddings(model, data, metapath_adjacency) -> np.ndarray:
    """Compute initial embeddings."""
    model.eval()
    with torch.no_grad():
        _, fused, _ = model(data, metapath_adjacency)
    return fused.detach().cpu().numpy()


def run_invariance_method(
    bundle,
    graph_data: Any,
    split_frame: pd.DataFrame,
    transfer_id: str,
    seed: int,
    method: str,
    week11_config: dict[str, Any],
    eerm_k: int | None = None,
    smoke_test: bool = False,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Run invariance method."""
    started = time.time()
    set_random_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = copy.deepcopy(graph_data).to(device)
    metapath_adjacency = build_metapath_adjacency(bundle, graph_data, device)
    encoder_state, encoder_checkpoint = load_encoder_state(bundle, seed)
    model = make_finetune_model(bundle, encoder_state, device)
    labels = build_label_vector(bundle, split_frame).to(device)
    source_train_idx = index_tensor(split_frame, "source_train", device)
    source_val_idx = index_tensor(split_frame, "source_val", device)
    target_test_idx = index_tensor(split_frame, "target_test", device)
    p2 = week11_config["P2_invariance_methods"]
    optimizer = torch.optim.Adam(
        [
            {"params": model.encoder_parameters(), "lr": float(p2["learning_rate_encoder"])},
            {"params": model.head_parameters(), "lr": float(p2["learning_rate_head"])},
        ],
        weight_decay=float(p2["weight_decay"]),
    )
    if method == "irm":
        irm_cfg = p2["irm"]
        env_indices = build_real_env_indices(
            split_frame,
            device=device,
            min_env_samples=int(irm_cfg["min_env_samples"]),
            require_two_classes=bool(irm_cfg["require_two_classes_per_env"]),
        )
        env_source = "real_jurisdiction"
    elif method == "eerm":
        if eerm_k is None:
            raise ValueError("EERM requires eerm_k")
        with torch.no_grad():
            initial = compute_initial_embeddings(model, data, metapath_adjacency)
        source_np = initial[source_train_idx.detach().cpu().numpy()]
        env_ids = make_virtual_environments(source_np, K=int(eerm_k), seed=seed)
        env_indices = make_env_index_dict(
            source_train_idx,
            env_ids,
            min_env_samples=int(p2["eerm"]["min_env_samples"]),
        )
        env_source = f"kmeans_K{eerm_k}"
    else:
        raise ValueError(f"Unsupported method: {method}")

    max_epochs = 5 if smoke_test else int(p2["max_epochs"])
    min_epochs = 0 if smoke_test else int(p2["min_epochs_before_early_stop"])
    patience = max_epochs if smoke_test else int(p2["patience"])
    warmup_epochs = int(p2["irm"]["warmup_epochs"])
    lambda_max = float(p2["irm"]["lambda_irm_max"])
    best_state: dict[str, Any] | None = None
    best_score = float("-inf")
    best_epoch = 0
    last_epoch = 0
    history_rows: list[dict[str, Any]] = []
    for epoch in range(1, max_epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits, _, _ = model(data, metapath_adjacency)
        lambda_value = cosine_warmup_lambda(epoch, warmup_epochs=warmup_epochs, lambda_max=lambda_max)
        if method == "irm":
            env_logits = {env: logits[idx] for env, idx in env_indices.items()}
            env_labels = {env: labels[idx] for env, idx in env_indices.items()}
            loss, stats = irm_loss_per_env(env_logits, env_labels, lambda_irm=lambda_value)
        else:
            loss, stats = eerm_loss(logits, labels, env_indices, lambda_irm=lambda_value)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            eval_logits, _, _ = model(data, metapath_adjacency)
            probs = torch.softmax(eval_logits, dim=1)[:, 1].detach().cpu().numpy()
        source_val_scores = probs[source_val_idx.detach().cpu().numpy()]
        source_val_labels = split_frame.loc[split_frame["split"] == "source_val", "label"].to_numpy(dtype=int)
        source_val_auc = float("nan")
        if np.unique(source_val_labels).size == 2:
            source_val_auc = evaluate_binary(source_val_labels, source_val_scores, 0.5)["roc_auc"]
        target_scores = probs[target_test_idx.detach().cpu().numpy()]
        target_labels = split_frame.loc[split_frame["split"] == "target_test", "label"].to_numpy(dtype=int)
        target_auc = float("nan")
        if np.unique(target_labels).size == 2:
            target_auc = evaluate_binary(target_labels, target_scores, 0.5)["roc_auc"]
        score = source_val_auc if math.isfinite(source_val_auc) else -float(loss.detach().cpu().item())
        history_rows.append(
            {
                "epoch": epoch,
                "method": method if eerm_k is None else f"eerm_K{eerm_k}",
                "transfer_id": transfer_id,
                "seed": seed,
                "loss": float(loss.detach().cpu().item()),
                "source_val_auc": float(source_val_auc),
                "target_test_auc": float(target_auc),
                "runtime_s": float(time.time() - started),
                **stats,
            }
        )
        last_epoch = epoch
        if math.isfinite(score) and score >= best_score:
            best_score = float(score)
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
        elif epoch >= min_epochs and best_state is not None and epoch - best_epoch >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        final_logits, _, _ = model(data, metapath_adjacency)
        probs_all = torch.softmax(final_logits, dim=1)[:, 1].detach().cpu().numpy()

    source_val = split_frame[split_frame["split"] == "source_val"].copy()
    threshold = choose_threshold_by_youden(
        source_val["label"].to_numpy(dtype=int),
        probs_all[source_val["graph_node_index"].to_numpy(dtype=int)],
    )
    train_info = {
        "best_epoch": int(best_epoch),
        "best_source_val_score": float(best_score),
        "selection_policy": "best_source_val",
        "best_state_min_epoch": 1,
        "trained_epochs": int(last_epoch),
        "force_full_epochs": False,
        "purpose": "invariance_method",
        "runtime_s": float(time.time() - started),
    }
    method_name = method if eerm_k is None else f"eerm_K{eerm_k}"
    metrics, prediction_frame = summarize_run_metrics(
        bundle=bundle,
        split_frame=split_frame,
        probabilities_all=probs_all,
        method=method_name,
        seed=seed,
        transfer_id=transfer_id,
        train_info=train_info,
        domain_acc_final=float("nan"),
        pseudo_label_quality=float("nan"),
    )
    metrics.update(
        {
            "method": method_name,
            "base_method_family": method,
            "eerm_k": int(eerm_k) if eerm_k is not None else float("nan"),
            "env_source": env_source,
            "num_envs": int(len(env_indices)),
            "encoder_checkpoint": encoder_checkpoint,
            "is_prerun_frozen": True,
            "is_posthoc": False,
            "threshold": float(threshold),
        }
    )
    prediction_frame["method"] = method_name
    return metrics, prediction_frame, pd.DataFrame(history_rows)


def plot_irm_convergence(output_paths: dict[str, Path]) -> None:
    """Plot IRM convergence."""
    histories = sorted(output_paths["logs"].glob("P2_irm_*_history.csv"))
    if not histories:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    for path in histories:
        hist = pd.read_csv(path)
        label = path.stem.replace("P2_irm_", "").replace("_history", "")
        ax.plot(hist["epoch"], hist["loss"], alpha=0.75, label=label)
    ax.set_title("Week 11 P2 IRM Convergence")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("IRM objective")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output_paths["plots"] / "P2_irm_convergence.png", dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Week 11 P2 IRM/EERM method comparison.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "week11.yaml"))
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Command-line entry point."""
    args = parse_args()
    config = load_config(Path(args.config))
    output_paths = ensure_output_dirs(config)
    bundle = load_graph_bundle(ROOT / config["inputs"]["week7_config"])
    graph_data = bundle.graph_data
    p2 = config["P2_invariance_methods"]
    transfer_ids = list(p2["transfer_ids"])
    seeds = [int(seed) for seed in config["seeds"]]
    if args.smoke_test:
        transfer_ids = ["T2_PH"]
        seeds = seeds[:1]

    rows: list[dict[str, Any]] = []
    for transfer_id in transfer_ids:
        for seed in seeds:
            split_path = ROOT / "output" / "week7" / "splits" / f"{transfer_id}__seed{seed}.csv"
            split_frame = pd.read_csv(split_path)
            if transfer_id in set(p2["irm"]["enabled_for"]):
                try:
                    metrics, preds, history = run_invariance_method(
                        bundle,
                        graph_data,
                        split_frame,
                        transfer_id,
                        seed,
                        "irm",
                        config,
                        smoke_test=args.smoke_test,
                    )
                    rows.append(metrics)
                    suffix = f"P2_irm_{transfer_id}__seed{seed}"
                    preds.to_csv(output_paths["runs"] / f"{suffix}_predictions.csv", index=False, encoding="utf-8")
                    history.to_csv(output_paths["logs"] / f"{suffix}_history.csv", index=False, encoding="utf-8")
                    (output_paths["runs"] / f"{suffix}.json").write_text(
                        json.dumps({"created_at_utc": utc_now_iso(), "metrics": metrics}, indent=2),
                        encoding="utf-8",
                    )
                except ValueError as exc:
                    rows.append(
                        {
                            "created_at_utc": utc_now_iso(),
                            "transfer_id": transfer_id,
                            "method": "irm",
                            "seed": seed,
                            "primary_metric_value": float("nan"),
                            "status": "not_applicable",
                            "not_applicable_reason": str(exc),
                            "is_prerun_frozen": True,
                            "is_posthoc": False,
                        }
                    )
            if transfer_id in set(p2["eerm"]["enabled_for"]):
                for k in p2["eerm"]["K_values"]:
                    metrics, preds, history = run_invariance_method(
                        bundle,
                        graph_data,
                        split_frame,
                        transfer_id,
                        seed,
                        "eerm",
                        config,
                        eerm_k=int(k),
                        smoke_test=args.smoke_test,
                    )
                    rows.append(metrics)
                    suffix = f"P2_eerm_{transfer_id}__seed{seed}__K{k}"
                    preds.to_csv(output_paths["runs"] / f"{suffix}_predictions.csv", index=False, encoding="utf-8")
                    history.to_csv(output_paths["logs"] / f"{suffix}_history.csv", index=False, encoding="utf-8")
                    (output_paths["runs"] / f"{suffix}.json").write_text(
                        json.dumps({"created_at_utc": utc_now_iso(), "metrics": metrics}, indent=2),
                        encoding="utf-8",
                    )

    week7_raw = pd.read_csv(ROOT / config["inputs"]["week7_transfer_summary"])
    week7_raw = week7_raw.copy()
    week7_raw["base_method_family"] = week7_raw["method"]
    week7_raw["eerm_k"] = np.nan
    week7_raw["env_source"] = "week7_existing"
    week7_raw["num_envs"] = np.nan
    week7_raw["is_prerun_frozen"] = False
    week7_raw["is_posthoc"] = True
    week7_raw["result_origin"] = "existing_week7"
    new_frame = pd.DataFrame(rows)
    new_frame["result_origin"] = "week11_P2"
    combined = pd.concat([week7_raw, new_frame], ignore_index=True, sort=False)
    combined.to_csv(output_paths["metrics"] / "P2_method_comparison.csv", index=False, encoding="utf-8")
    eerm = new_frame[new_frame["method"].astype(str).str.startswith("eerm_K")].copy()
    sens_rows = []
    if not eerm.empty:
        for transfer_id, part in eerm.groupby("transfer_id"):
            means = part.groupby("method")["primary_metric_value"].mean()
            sens_rows.append(
                {
                    "transfer_id": transfer_id,
                    "eerm_K2_mean": float(means.get("eerm_K2", np.nan)),
                    "eerm_K4_mean": float(means.get("eerm_K4", np.nan)),
                    "absolute_range": float(means.max() - means.min()) if len(means.dropna()) else float("nan"),
                    "is_prerun_frozen": True,
                    "is_posthoc": False,
                }
            )
    pd.DataFrame(sens_rows).to_csv(output_paths["metrics"] / "P2_eerm_K_sensitivity.csv", index=False, encoding="utf-8")
    plot_irm_convergence(output_paths)
    print(f"P2 rows={len(new_frame)} combined_rows={len(combined)}")


if __name__ == "__main__":
    main()
