"""Week-11 P1: dual-head family metric learning under leave-one-family-out."""
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
from sklearn.manifold import TSNE
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.family_metric_head import FamilyDualHead, supervised_contrastive_loss
from src.train.train_contrastive import get_embeddings, make_heco_model
from src.train.train_transfer import build_metapath_adjacency, load_encoder_state
from src.train.utils import build_primary_task_frame, load_graph_bundle, set_random_seed, utc_now_iso


def load_week11_config(path: Path) -> dict[str, Any]:
    """Load week11 config."""
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def ensure_week11_dirs(config: dict[str, Any]) -> dict[str, Path]:
    """Ensure week11 directories."""
    root = Path(config["workspace_root"])
    paths = {name: root / rel for name, rel in config["output"].items()}
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def load_frozen_embeddings(bundle, seed: int) -> tuple[np.ndarray, str]:
    """Load frozen embeddings."""
    encoder_state, checkpoint = load_encoder_state(bundle, seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = make_heco_model(bundle).to(device)
    model.load_state_dict(copy.deepcopy(encoder_state))
    emb = get_embeddings(bundle, model)
    return emb, checkpoint


def build_family_frame(bundle, config: dict[str, Any]) -> pd.DataFrame:
    """Build family frame."""
    root = Path(config["workspace_root"])
    family_path = root / config["inputs"]["week85_family_assignments"]
    family = pd.read_csv(family_path)
    family = family[
        [
            "node_id",
            "week8_family_id",
            "week8_family_size",
            "week8_lofo_eligible",
        ]
    ].copy()
    primary = build_primary_task_frame(bundle)
    frame = primary.merge(family, on="node_id", how="left")
    frame["p1_family_id"] = pd.NA
    eligible = frame["week8_lofo_eligible"].astype(str).str.lower().eq("true")
    frame.loc[eligible, "p1_family_id"] = frame.loc[eligible, "week8_family_id"].astype(str)
    non_dk_illegal = (frame["label"] == 1) & frame["p1_family_id"].isna() & frame["jurisdiction"].ne("Denmark")
    frame.loc[non_dk_illegal, "p1_family_id"] = "non_dk::" + frame.loc[non_dk_illegal, "jurisdiction"].astype(str)
    return frame.reset_index(drop=True)


def eligible_dk_families(frame: pd.DataFrame, min_size: int) -> list[str]:
    """Eligible dk families."""
    dk = frame[
        frame["jurisdiction"].eq("Denmark")
        & frame["p1_family_id"].notna()
        & (frame["label"] == 1)
    ].copy()
    counts = dk["p1_family_id"].value_counts()
    return sorted(counts[counts >= int(min_size)].index.astype(str).tolist())


def prepare_tensors(
    frame: pd.DataFrame,
    embeddings: np.ndarray,
    heldout_family: str,
    shuffle_family_labels: bool,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, pd.DataFrame, dict[str, int]]:
    """Prepare tensors."""
    train = frame[
        frame["sample_tier"].isin(
            [
                "licensed_baseline",
                "illegal_confirmed_official_single",
                "illegal_confirmed_official_cross_verified",
            ]
        )
        & ~(frame["p1_family_id"].astype(str).eq(heldout_family))
    ].copy()
    train = train[train["label"].isin([0, 1])].copy()
    family_counts = train.loc[(train["label"] == 1) & train["p1_family_id"].notna(), "p1_family_id"].value_counts()
    valid_families = sorted(family_counts[family_counts >= 2].index.astype(str).tolist())
    family_to_int = {family: i for i, family in enumerate(valid_families)}
    family_labels = np.full(len(train), -1, dtype=np.int64)
    for i, family in enumerate(train["p1_family_id"].astype(str).tolist()):
        if family in family_to_int:
            family_labels[i] = family_to_int[family]
    if shuffle_family_labels:
        valid = np.where(family_labels >= 0)[0]
        rng = np.random.default_rng(seed * 104729 + 17)
        family_labels[valid] = rng.permutation(family_labels[valid])
    x = torch.tensor(embeddings[train["graph_node_index"].to_numpy(dtype=int)], dtype=torch.float32)
    y = torch.tensor(train["label"].to_numpy(dtype=np.int64), dtype=torch.long)
    f = torch.tensor(family_labels, dtype=torch.long)
    return x, y, f, train, family_to_int


def train_dual_head(
    x: torch.Tensor,
    y: torch.Tensor,
    family_labels: torch.Tensor,
    method_cfg: dict[str, Any],
    base_cfg: dict[str, Any],
    seed: int,
) -> tuple[FamilyDualHead, pd.DataFrame]:
    """Train dual head."""
    set_random_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = x.to(device)
    y = y.to(device)
    family_labels = family_labels.to(device)
    model = FamilyDualHead(
        embed_dim=x.shape[1],
        projection_dim=int(base_cfg["projection_dim"]),
        hidden_dim=int(base_cfg["hidden_dim"]),
    ).to(device)
    opt = torch.optim.Adam(
        model.parameters(),
        lr=float(base_cfg["learning_rate"]),
        weight_decay=float(base_cfg["weight_decay"]),
    )
    best_state: dict[str, torch.Tensor] | None = None
    best_loss = float("inf")
    best_epoch = 0
    rows: list[dict[str, float]] = []
    epochs = int(base_cfg["epochs"])
    patience = int(base_cfg["patience"])
    alpha = float(method_cfg.get("alpha", base_cfg["alpha"]))
    binary_weight = float(method_cfg.get("binary_weight", base_cfg.get("binary_weight", 1.0)))
    for epoch in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        logits, z = model(x)
        ce = F.cross_entropy(logits, y)
        sup = supervised_contrastive_loss(z, family_labels, temperature=float(base_cfg["temperature"]))
        loss = binary_weight * ce + alpha * sup
        loss.backward()
        opt.step()
        current = float(loss.detach().cpu().item())
        rows.append(
            {
                "epoch": epoch,
                "loss": current,
                "binary_ce": float(ce.detach().cpu().item()),
                "supcon_loss": float(sup.detach().cpu().item()),
                "alpha": alpha,
                "binary_weight": binary_weight,
            }
        )
        if current < best_loss:
            best_loss = current
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
        elif epoch - best_epoch >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, pd.DataFrame(rows)


def projection_scores(
    model: FamilyDualHead,
    embeddings: np.ndarray,
    frame: pd.DataFrame,
    train_frame: pd.DataFrame,
    candidate_frame: pd.DataFrame,
    score_mode: str,
) -> np.ndarray:
    """Projection scores."""
    device = next(model.parameters()).device
    x_all = torch.tensor(embeddings, dtype=torch.float32, device=device)
    model.eval()
    with torch.no_grad():
        logits, z_all = model(x_all)
        if score_mode == "binary":
            return torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()[
                candidate_frame["graph_node_index"].to_numpy(dtype=int)
            ]
        z_np = z_all.detach().cpu().numpy()
    train_dk = train_frame[
        train_frame["jurisdiction"].eq("Denmark")
        & (train_frame["label"] == 1)
        & train_frame["p1_family_id"].notna()
    ]
    centroids = []
    for _, part in train_dk.groupby("p1_family_id"):
        centroids.append(z_np[part["graph_node_index"].to_numpy(dtype=int)].mean(axis=0))
    if not centroids:
        train_illegal = train_frame[train_frame["label"] == 1]
        centroids.append(z_np[train_illegal["graph_node_index"].to_numpy(dtype=int)].mean(axis=0))
    c = np.vstack(centroids)
    c = c / np.maximum(np.linalg.norm(c, axis=1, keepdims=True), 1.0e-12)
    q = z_np[candidate_frame["graph_node_index"].to_numpy(dtype=int)]
    q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1.0e-12)
    return (q @ c.T).max(axis=1)


def evaluate_fold(
    frame: pd.DataFrame,
    embeddings: np.ndarray,
    model: FamilyDualHead,
    train_frame: pd.DataFrame,
    heldout_family: str,
    seed: int,
    score_mode: str,
) -> dict[str, Any]:
    """Evaluate fold."""
    positives = frame[frame["p1_family_id"].astype(str).eq(heldout_family)].copy()
    negative_pool = frame[
        frame["jurisdiction"].eq("Denmark")
        & (frame["label"] == 1)
        & frame["p1_family_id"].notna()
        & ~frame["p1_family_id"].astype(str).eq(heldout_family)
    ].copy()
    rng = np.random.default_rng(seed * 1009 + len(heldout_family))
    take = min(len(positives), len(negative_pool))
    if take == 0:
        raise ValueError(f"Cannot evaluate heldout family {heldout_family}: no positives or negatives")
    neg_idx = rng.choice(negative_pool.index.to_numpy(), size=take, replace=False)
    negatives = negative_pool.loc[neg_idx].copy()
    candidates = pd.concat([positives, negatives], ignore_index=True)
    labels = np.r_[np.ones(len(positives), dtype=int), np.zeros(len(negatives), dtype=int)]
    scores = projection_scores(model, embeddings, frame, train_frame, candidates, score_mode=score_mode)
    auc = float(roc_auc_score(labels, scores)) if np.unique(labels).size == 2 else float("nan")
    pr_auc = float(average_precision_score(labels, scores)) if np.unique(labels).size == 2 else float("nan")
    return {
        "hard_negative_auc": auc,
        "hard_negative_pr_auc": pr_auc,
        "positive_score_mean": float(np.mean(scores[: len(positives)])),
        "negative_score_mean": float(np.mean(scores[len(positives) :])),
        "score_gap_pos_minus_neg": float(np.mean(scores[: len(positives)]) - np.mean(scores[len(positives) :])),
        "heldout_positive_count": int(len(positives)),
        "negative_count": int(len(negatives)),
        "score_mode": score_mode,
    }


def run_fold(
    frame: pd.DataFrame,
    embeddings: np.ndarray,
    family: str,
    seed: int,
    method_cfg: dict[str, Any],
    base_cfg: dict[str, Any],
    output_paths: dict[str, Path],
    run_prefix: str,
) -> dict[str, Any]:
    """Run fold."""
    started = time.time()
    x, y, fam, train_frame, family_to_int = prepare_tensors(
        frame,
        embeddings,
        heldout_family=family,
        shuffle_family_labels=bool(method_cfg.get("shuffle_family_labels", False)),
        seed=seed,
    )
    model, history = train_dual_head(x, y, fam, method_cfg, base_cfg, seed=seed)
    score_mode = "binary" if float(method_cfg.get("alpha", 0.0)) == 0.0 else "metric"
    metrics = evaluate_fold(frame, embeddings, model, train_frame, family, seed, score_mode=score_mode)
    suffix = f"{run_prefix}_{family}__seed{seed}__{method_cfg['name']}"
    history.to_csv(output_paths["logs"] / f"{suffix}_history.csv", index=False, encoding="utf-8")
    manifest = {
        "created_at_utc": utc_now_iso(),
        "task": "W11_P1_family_metric_lofo",
        "family_id": family,
        "seed": seed,
        "method": method_cfg["name"],
        "alpha": float(method_cfg.get("alpha", base_cfg["alpha"])),
        "binary_weight": float(method_cfg.get("binary_weight", base_cfg.get("binary_weight", 1.0))),
        "shuffle_family_labels": bool(method_cfg.get("shuffle_family_labels", False)),
        "num_train_rows": int(len(train_frame)),
        "num_family_labels": int(sum(1 for _ in family_to_int)),
        "metrics": metrics,
    }
    (output_paths["runs"] / f"{suffix}.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {
        "created_at_utc": utc_now_iso(),
        "family_id": family,
        "seed": seed,
        "method": method_cfg["name"],
        "alpha": float(method_cfg.get("alpha", base_cfg["alpha"])),
        "binary_weight": float(method_cfg.get("binary_weight", base_cfg.get("binary_weight", 1.0))),
        "shuffle_family_labels": bool(method_cfg.get("shuffle_family_labels", False)),
        "test_form": "hard_illegal_negative",
        "train_scope": "frozen_heco_embedding_plus_family_dual_head",
        "is_prerun_frozen": True,
        "is_posthoc": False,
        "runtime_s": float(time.time() - started),
        **metrics,
    }


def build_centroid_distances(
    frame: pd.DataFrame,
    embeddings_by_seed: dict[int, np.ndarray],
    families: list[str],
) -> pd.DataFrame:
    """Build centroid distances."""
    rows = []
    for seed, embeddings in embeddings_by_seed.items():
        for i, fam_a in enumerate(families):
            idx_a = frame.loc[frame["p1_family_id"].astype(str).eq(fam_a), "graph_node_index"].to_numpy(dtype=int)
            ca = embeddings[idx_a].mean(axis=0)
            ca = ca / max(float(np.linalg.norm(ca)), 1.0e-12)
            for fam_b in families[i + 1 :]:
                idx_b = frame.loc[frame["p1_family_id"].astype(str).eq(fam_b), "graph_node_index"].to_numpy(dtype=int)
                cb = embeddings[idx_b].mean(axis=0)
                cb = cb / max(float(np.linalg.norm(cb)), 1.0e-12)
                cosine = float(np.dot(ca, cb))
                rows.append(
                    {
                        "seed": seed,
                        "family_a": fam_a,
                        "family_b": fam_b,
                        "cosine_similarity": cosine,
                        "cosine_distance": float(1.0 - cosine),
                        "is_prerun_frozen": True,
                    }
                )
    return pd.DataFrame(rows)


def plot_family_tsne(frame: pd.DataFrame, embeddings: np.ndarray, families: list[str], output_path: Path) -> None:
    """Plot family tsne."""
    family_frame = frame[frame["p1_family_id"].astype(str).isin(families)].copy()
    x = embeddings[family_frame["graph_node_index"].to_numpy(dtype=int)]
    perplexity = max(2, min(10, len(family_frame) // 3))
    coords = TSNE(n_components=2, perplexity=perplexity, init="pca", learning_rate="auto", random_state=42).fit_transform(x)
    family_frame["tsne_x"] = coords[:, 0]
    family_frame["tsne_y"] = coords[:, 1]
    fig, ax = plt.subplots(figsize=(8, 6))
    for family, part in family_frame.groupby("p1_family_id"):
        ax.scatter(part["tsne_x"], part["tsne_y"], s=34, label=str(family), alpha=0.85)
    ax.set_title("Week 11 P1 Family Embedding t-SNE")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Week 11 P1 family metric head.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "week11.yaml"))
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Command-line entry point."""
    args = parse_args()
    config = load_week11_config(Path(args.config))
    output_paths = ensure_week11_dirs(config)
    bundle = load_graph_bundle(ROOT / config["inputs"]["week7_config"])
    p1 = config["P1_family_metric"]
    frame = build_family_frame(bundle, config)
    families = eligible_dk_families(frame, min_size=int(p1["eligible_family_min_size"]))
    seeds = [int(seed) for seed in p1["encoder_seeds"]]
    if args.smoke_test:
        families = families[:1]
        seeds = seeds[:1]
    embeddings_by_seed: dict[int, np.ndarray] = {}
    rows: list[dict[str, Any]] = []
    ablation_rows: list[dict[str, Any]] = []
    main_cfg = next(item for item in p1["ablations"] if item["name"] == p1["core_method"])
    for seed in seeds:
        embeddings, checkpoint = load_frozen_embeddings(bundle, seed)
        embeddings_by_seed[seed] = embeddings
        for family in families:
            row = run_fold(frame, embeddings, family, seed, main_cfg, p1, output_paths, "P1_familyloo")
            row["encoder_checkpoint"] = checkpoint
            rows.append(row)
        for family in list(p1["ablation_families"]):
            if family not in families:
                continue
            for method_cfg in p1["ablations"]:
                row = run_fold(frame, embeddings, family, seed, method_cfg, p1, output_paths, "P1_ablation")
                row["encoder_checkpoint"] = checkpoint
                ablation_rows.append(row)
    lofo = pd.DataFrame(rows).sort_values(["family_id", "seed"]).reset_index(drop=True)
    if ablation_rows:
        ablation = pd.DataFrame(ablation_rows).sort_values(["family_id", "method", "seed"]).reset_index(drop=True)
    else:
        ablation = pd.DataFrame(
            columns=[
                "created_at_utc",
                "family_id",
                "seed",
                "method",
                "alpha",
                "binary_weight",
                "shuffle_family_labels",
                "hard_negative_auc",
            ]
        )
    centroid = build_centroid_distances(frame, embeddings_by_seed, families)
    lofo.to_csv(output_paths["metrics"] / "P1_family_metric_lofo.csv", index=False, encoding="utf-8")
    ablation.to_csv(output_paths["metrics"] / "P1_alpha_ablation.csv", index=False, encoding="utf-8")
    centroid.to_csv(output_paths["metrics"] / "P1_family_centroid_distance.csv", index=False, encoding="utf-8")
    if embeddings_by_seed:
        plot_family_tsne(frame, embeddings_by_seed[seeds[0]], families, output_paths["plots"] / "P1_family_tsne.png")
    print(f"P1 lofo rows={len(lofo)} ablation rows={len(ablation)} families={len(families)}")


if __name__ == "__main__":
    main()
