"""HeCo co-contrastive model: network-schema view, metapath view, and InfoNCE loss."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from src.models.projection_heads import TwoLayerProjectionHead
from src.models.view_generators import MetapathViewEncoder, SchemaViewEncoder


@dataclass
class HeCoForwardOutput:
    """He Co Forward Output."""
    schema_embedding: torch.Tensor
    metapath_embedding: torch.Tensor
    schema_projection: torch.Tensor
    metapath_projection: torch.Tensor
    combined_embedding: torch.Tensor


class HeCoModel(nn.Module):
    """He Co Model (PyTorch module)."""
    def __init__(
        self,
        node_feature_dims: dict[str, int],
        edge_types: list[tuple[str, str, str]],
        metapath_names: list[str],
        hidden_dim: int = 64,
        projection_dim: int = 64,
        dropout: float = 0.30,
    ) -> None:
        """Initialise the instance."""
        super().__init__()
        self.schema_encoder = SchemaViewEncoder(
            node_feature_dims=node_feature_dims,
            edge_types=edge_types,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        self.metapath_encoder = MetapathViewEncoder(
            website_feature_dim=node_feature_dims["Website"],
            metapath_names=metapath_names,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        self.schema_projection = TwoLayerProjectionHead(hidden_dim, hidden_dim, projection_dim, dropout)
        self.metapath_projection = TwoLayerProjectionHead(hidden_dim, hidden_dim, projection_dim, dropout)

    def forward(self, data, metapath_adjacency: dict[str, torch.Tensor]) -> HeCoForwardOutput:
        """Run the forward pass."""
        schema_embedding = self.schema_encoder(data)
        metapath_embedding = self.metapath_encoder(data["Website"].x, metapath_adjacency)
        schema_projection = self.schema_projection(schema_embedding)
        metapath_projection = self.metapath_projection(metapath_embedding)
        combined_embedding = torch.cat([schema_embedding, metapath_embedding], dim=1)
        return HeCoForwardOutput(
            schema_embedding=schema_embedding,
            metapath_embedding=metapath_embedding,
            schema_projection=schema_projection,
            metapath_projection=metapath_projection,
            combined_embedding=combined_embedding,
        )


class HeCoContrastiveLoss(nn.Module):
    """He Co Contrastive Loss (PyTorch module)."""
    def __init__(self, temperature: float = 0.5, hard_negative_ratio: float = 0.30) -> None:
        """Initialise the instance."""
        super().__init__()
        self.temperature = temperature
        self.hard_negative_ratio = hard_negative_ratio

    def _denominator_mask(self, similarity: torch.Tensor, positive_mask: torch.Tensor) -> torch.Tensor:
        """Helper: denominator mask."""
        if self.hard_negative_ratio <= 0:
            return torch.ones_like(positive_mask, dtype=torch.bool)
        negative_mask = ~positive_mask
        denominator_mask = positive_mask.clone()
        num_nodes = similarity.shape[0]
        for row_idx in range(num_nodes):
            negatives = negative_mask[row_idx].nonzero(as_tuple=False).flatten()
            if negatives.numel() == 0:
                continue
            k = max(1, int(round(float(negatives.numel()) * self.hard_negative_ratio)))
            k = min(k, int(negatives.numel()))
            selected = negatives[torch.topk(similarity[row_idx, negatives], k=k).indices]
            denominator_mask[row_idx, selected] = True
        return denominator_mask

    def _directional_loss(self, anchor: torch.Tensor, target: torch.Tensor, positive_mask: torch.Tensor) -> torch.Tensor:
        """Helper: directional loss."""
        anchor = F.normalize(anchor, p=2, dim=1)
        target = F.normalize(target, p=2, dim=1)
        similarity = (anchor @ target.t()) / self.temperature
        denominator_mask = self._denominator_mask(similarity.detach(), positive_mask)
        positive_logits = similarity.masked_fill(~positive_mask, float("-inf"))
        denominator_logits = similarity.masked_fill(~denominator_mask, float("-inf"))
        numerator = torch.logsumexp(positive_logits, dim=1)
        denominator = torch.logsumexp(denominator_logits, dim=1)
        return -(numerator - denominator).mean()

    def forward(
        self,
        schema_projection: torch.Tensor,
        metapath_projection: torch.Tensor,
        positive_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Run the forward pass."""
        positive_mask = positive_mask.to(schema_projection.device)
        schema_to_metapath = self._directional_loss(schema_projection, metapath_projection, positive_mask)
        metapath_to_schema = self._directional_loss(
            metapath_projection,
            schema_projection,
            positive_mask.t(),
        )
        loss = 0.5 * (schema_to_metapath + metapath_to_schema)
        stats = {
            "loss_schema_to_metapath": float(schema_to_metapath.detach().cpu().item()),
            "loss_metapath_to_schema": float(metapath_to_schema.detach().cpu().item()),
            "positive_pairs": float(positive_mask.sum().detach().cpu().item()),
        }
        return loss, stats
