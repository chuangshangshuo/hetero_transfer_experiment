"""Domain-adversarial training helpers (gradient reversal, lambda schedule)."""
from __future__ import annotations

import math

import torch
from torch import nn

from src.models.heco_head import HeCoModel
from src.transfer.dann_head import DomainDiscriminator


def lambda_schedule(
    epoch: int,
    warmup_epochs: int = 50,
    lambda_max: float = 1.0,
    schedule: str = "cosine",
) -> float:
    """Lambda schedule."""
    if warmup_epochs <= 0 or epoch >= warmup_epochs:
        return float(lambda_max)
    progress = max(0.0, min(1.0, float(epoch) / float(warmup_epochs)))
    if schedule == "linear":
        return float(lambda_max) * progress
    if schedule == "cosine":
        return float(lambda_max) * 0.5 * (1.0 - math.cos(math.pi * progress))
    raise ValueError(f"Unsupported DANN lambda schedule: {schedule}")


class HeCoDANNClassifier(nn.Module):
    """He Co D A N N Classifier (PyTorch module)."""
    def __init__(
        self,
        heco_encoder: HeCoModel,
        hidden_dim: int = 64,
        dropout: float = 0.30,
        head_type: str = "mlp_64",
    ) -> None:
        """Initialise the instance."""
        super().__init__()
        if head_type != "mlp_64":
            raise ValueError("Week 7 DANN currently uses the mlp_64 HeCo head.")
        self.encoder = heco_encoder
        embedding_dim = hidden_dim * 2
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )
        self.domain_discriminator = DomainDiscriminator(
            input_dim=embedding_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

    def encoder_parameters(self):
        """Encoder parameters."""
        return self.encoder.parameters()

    def head_parameters(self):
        """Head parameters."""
        return self.classifier.parameters()

    def domain_parameters(self):
        """Domain parameters."""
        return self.domain_discriminator.parameters()

    def forward(
        self,
        data,
        metapath_adjacency: dict[str, torch.Tensor],
        reverse_scale: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Run the forward pass."""
        output = self.encoder(data, metapath_adjacency)
        class_logits = self.classifier(output.combined_embedding)
        domain_logits = self.domain_discriminator(output.combined_embedding, reverse_scale=reverse_scale)
        return class_logits, domain_logits, {
            "schema_embedding": output.schema_embedding,
            "metapath_embedding": output.metapath_embedding,
            "combined_embedding": output.combined_embedding,
        }
