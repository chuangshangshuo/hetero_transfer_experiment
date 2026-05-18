from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class TwoLayerProjectionHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.30) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, inputs: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        projected = self.network(inputs)
        if normalize:
            projected = F.normalize(projected, p=2, dim=1)
        return projected
