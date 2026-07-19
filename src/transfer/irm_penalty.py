"""IRMv1 penalty terms and the cosine warm-up schedule for lambda."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def irm_penalty(logits: torch.Tensor, labels: torch.Tensor, scale_w: float = 1.0) -> torch.Tensor:
    """IRMv1 penalty using a trainable scalar classifier multiplier."""

    if logits.ndim == 2 and logits.shape[1] == 2:
        binary_logits = logits[:, 1] - logits[:, 0]
    else:
        binary_logits = logits.reshape(-1)
    labels = labels.float().reshape(-1)
    w = torch.tensor(float(scale_w), requires_grad=True, device=logits.device)
    loss = F.binary_cross_entropy_with_logits(binary_logits * w, labels)
    grad = torch.autograd.grad(loss, w, create_graph=True)[0]
    return grad.pow(2)


def irm_loss_per_env(
    env_logits: dict[str, torch.Tensor],
    env_labels: dict[str, torch.Tensor],
    lambda_irm: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Average ERM plus IRMv1 penalty across non-empty environments."""

    losses: list[torch.Tensor] = []
    penalties: list[torch.Tensor] = []
    for env_id, logits in env_logits.items():
        labels = env_labels[env_id]
        if logits.numel() == 0:
            continue
        losses.append(F.cross_entropy(logits, labels.long()))
        penalties.append(irm_penalty(logits, labels))
    if not losses:
        raise ValueError("IRM requires at least one non-empty environment")
    erm = torch.stack(losses).mean()
    penalty = torch.stack(penalties).mean()
    total = erm + float(lambda_irm) * penalty
    return total, {
        "erm_loss": float(erm.detach().cpu().item()),
        "irm_penalty": float(penalty.detach().cpu().item()),
        "lambda_irm": float(lambda_irm),
        "num_envs": float(len(losses)),
    }


def cosine_warmup_lambda(epoch: int, warmup_epochs: int, lambda_max: float) -> float:
    """Cosine warm-up: ramp lambda from 0 to lambda_max over warmup_epochs."""
    if warmup_epochs <= 0 or epoch >= warmup_epochs:
        return float(lambda_max)
    progress = max(0.0, min(1.0, float(epoch) / float(warmup_epochs)))
    return float(lambda_max) * 0.5 * (1.0 - math.cos(math.pi * progress))
