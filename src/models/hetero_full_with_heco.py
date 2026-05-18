from __future__ import annotations

import torch
from torch import nn

from src.models.heco_head import HeCoModel


class AttentionFusionHead(nn.Module):
    def __init__(self, hidden_dim: int = 64, dropout: float = 0.30) -> None:
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, schema_embedding: torch.Tensor, metapath_embedding: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        stacked = torch.stack([schema_embedding, metapath_embedding], dim=1)
        weights = torch.softmax(self.attention(stacked), dim=1)
        fused = (weights * stacked).sum(dim=1)
        logits = self.classifier(fused)
        return logits, fused, weights.squeeze(-1)


class FixedFusionHead(nn.Module):
    def __init__(
        self,
        mode: str,
        hidden_dim: int = 64,
        dropout: float = 0.30,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, schema_embedding: torch.Tensor, metapath_embedding: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.mode == "fixed_mean":
            fused = 0.5 * (schema_embedding + metapath_embedding)
            weights = torch.full((schema_embedding.shape[0], 2), 0.5, device=schema_embedding.device)
        elif self.mode == "schema_only":
            fused = schema_embedding
            weights = torch.stack(
                [
                    torch.ones(schema_embedding.shape[0], device=schema_embedding.device),
                    torch.zeros(schema_embedding.shape[0], device=schema_embedding.device),
                ],
                dim=1,
            )
        elif self.mode == "metapath_only":
            fused = metapath_embedding
            weights = torch.stack(
                [
                    torch.zeros(schema_embedding.shape[0], device=schema_embedding.device),
                    torch.ones(schema_embedding.shape[0], device=schema_embedding.device),
                ],
                dim=1,
            )
        else:
            raise ValueError(f"Unsupported fixed fusion mode: {self.mode}")
        return self.classifier(fused), fused, weights


class HeCoFineTuneClassifier(nn.Module):
    def __init__(
        self,
        heco_encoder: HeCoModel,
        head_type: str = "attention_pool_mlp_64",
        hidden_dim: int = 64,
        dropout: float = 0.30,
    ) -> None:
        super().__init__()
        self.encoder = heco_encoder
        self.head_type = head_type
        canonical_head = "attention_pool_mlp_64" if head_type == "attention" else head_type
        if canonical_head == "linear":
            self.classifier = nn.Linear(hidden_dim * 2, 2)
            self.attention_head = None
            self.fixed_head = None
        elif canonical_head == "mlp_64":
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 2),
            )
            self.attention_head = None
            self.fixed_head = None
        elif canonical_head == "attention_pool_mlp_64":
            self.classifier = None
            self.attention_head = AttentionFusionHead(hidden_dim=hidden_dim, dropout=dropout)
            self.fixed_head = None
        elif canonical_head in {"fixed_mean", "schema_only", "metapath_only"}:
            self.classifier = None
            self.attention_head = None
            self.fixed_head = FixedFusionHead(canonical_head, hidden_dim=hidden_dim, dropout=dropout)
        else:
            raise ValueError(f"Unsupported HeCo finetune head_type: {head_type}")

    def encoder_parameters(self):
        return self.encoder.parameters()

    def head_parameters(self):
        if self.attention_head is not None:
            return self.attention_head.parameters()
        if self.fixed_head is not None:
            return self.fixed_head.parameters()
        if self.classifier is None:
            raise RuntimeError("Classifier head is not initialized")
        return self.classifier.parameters()

    def forward(self, data, metapath_adjacency: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        output = self.encoder(data, metapath_adjacency)
        if self.attention_head is not None:
            logits, fused_embedding, attention_weights = self.attention_head(
                output.schema_embedding,
                output.metapath_embedding,
            )
            return logits, fused_embedding, {
                "schema_embedding": output.schema_embedding,
                "metapath_embedding": output.metapath_embedding,
                "combined_embedding": output.combined_embedding,
                "attention_weights": attention_weights,
            }
        if self.fixed_head is not None:
            logits, fused_embedding, fixed_weights = self.fixed_head(
                output.schema_embedding,
                output.metapath_embedding,
            )
            return logits, fused_embedding, {
                "schema_embedding": output.schema_embedding,
                "metapath_embedding": output.metapath_embedding,
                "combined_embedding": output.combined_embedding,
                "attention_weights": fixed_weights,
            }
        if self.classifier is None:
            raise RuntimeError("Classifier head is not initialized")
        logits = self.classifier(output.combined_embedding)
        return logits, output.combined_embedding, {
            "schema_embedding": output.schema_embedding,
            "metapath_embedding": output.metapath_embedding,
            "combined_embedding": output.combined_embedding,
        }
