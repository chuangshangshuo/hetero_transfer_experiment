from __future__ import annotations

import numpy as np
import torch
from sklearn.cluster import KMeans

from src.transfer.irm_penalty import irm_loss_per_env


def make_virtual_environments(embeddings: np.ndarray, K: int = 4, seed: int = 42) -> np.ndarray:
    """Cluster source embeddings into fixed virtual environments."""

    embeddings = np.asarray(embeddings, dtype=float)
    n_clusters = max(1, min(int(K), embeddings.shape[0]))
    if n_clusters == 1:
        return np.zeros(embeddings.shape[0], dtype=int)
    km = KMeans(n_clusters=n_clusters, random_state=int(seed), n_init=10)
    return km.fit_predict(embeddings)


def make_env_index_dict(
    base_indices: torch.Tensor,
    env_ids: np.ndarray,
    min_env_samples: int = 4,
) -> dict[str, torch.Tensor]:
    env_ids = np.asarray(env_ids, dtype=int)
    if env_ids.shape[0] != int(base_indices.shape[0]):
        raise ValueError("env_ids length must match base_indices length")
    envs: dict[str, torch.Tensor] = {}
    for env_id in sorted(np.unique(env_ids).tolist()):
        local = np.where(env_ids == env_id)[0]
        if local.size < int(min_env_samples):
            continue
        local_tensor = torch.tensor(local, dtype=torch.long, device=base_indices.device)
        envs[f"virtual_{env_id}"] = base_indices[local_tensor]
    if not envs:
        envs["virtual_all"] = base_indices
    return envs


def eerm_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    env_indices: dict[str, torch.Tensor],
    lambda_irm: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    env_logits = {env: logits[idx] for env, idx in env_indices.items()}
    env_labels = {env: labels[idx] for env, idx in env_indices.items()}
    return irm_loss_per_env(env_logits, env_labels, lambda_irm=lambda_irm)
