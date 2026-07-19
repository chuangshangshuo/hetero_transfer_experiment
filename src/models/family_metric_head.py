"""Dual-head module adding a family-level SupCon projection next to the binary head."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class FamilyMetricHead(nn.Module):
    """Projection head for family-aware supervised contrastive calibration."""

    def __init__(self, embed_dim: int = 128, projection_dim: int = 32, hidden_dim: int = 64) -> None:
        """Initialise the instance."""
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, projection_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the forward pass."""
        return F.normalize(self.proj(x), dim=-1)


class FamilyDualHead(nn.Module):
    """Binary illegal/licensed head plus an independent family metric head."""

    def __init__(self, embed_dim: int = 128, projection_dim: int = 32, hidden_dim: int = 64) -> None:
        """Initialise the instance."""
        super().__init__()
        self.family_head = FamilyMetricHead(embed_dim, projection_dim, hidden_dim)
        self.binary_head = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the forward pass."""
        return self.binary_head(x), self.family_head(x)


def supervised_contrastive_loss(
    z: torch.Tensor,
    family_labels: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """SupCon loss over rows with non-negative family labels.

    Rows whose label is negative are ignored. Rows without a same-family
    positive in the current batch are also ignored, which makes full-batch
    training robust to tiny family sizes.
    """

    valid = family_labels >= 0
    if int(valid.sum().item()) < 2:
        return z.sum() * 0.0
    z_valid = F.normalize(z[valid], dim=-1)
    labels_valid = family_labels[valid]
    sim = (z_valid @ z_valid.T) / float(temperature)
    sim = sim - sim.max(dim=1, keepdim=True).values.detach()
    same = labels_valid[:, None].eq(labels_valid[None, :])
    eye = torch.eye(same.shape[0], dtype=torch.bool, device=same.device)
    pos_mask = same & ~eye
    logits_mask = ~eye
    pos_count = pos_mask.sum(dim=1)
    usable = pos_count > 0
    if int(usable.sum().item()) == 0:
        return z.sum() * 0.0
    exp_sim = torch.exp(sim) * logits_mask.float()
    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1.0e-12)
    mean_log_prob_pos = (pos_mask.float() * log_prob).sum(dim=1)[usable] / pos_count[usable].float()
    return -mean_log_prob_pos.mean()
