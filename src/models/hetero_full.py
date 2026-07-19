"""Supervised HeteroConv baseline model over the full heterogeneous graph."""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GraphConv, HeteroConv, Linear


class WeightedHeteroGNNClassifier(nn.Module):
    """Weighted Hetero G N N Classifier (PyTorch module)."""
    def __init__(
        self,
        node_feature_dims: dict[str, int],
        edge_types: list[tuple[str, str, str]],
        hidden_dim: int = 64,
        dropout: float = 0.30,
    ) -> None:
        """Initialise the instance."""
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.input_linears = nn.ModuleDict(
            {node_type: Linear(feature_dim, hidden_dim) for node_type, feature_dim in node_feature_dims.items()}
        )
        self.convs = nn.ModuleList(
            [
                HeteroConv(
                    {
                        edge_type: GraphConv((hidden_dim, hidden_dim), hidden_dim, aggr="mean")
                        for edge_type in edge_types
                    },
                    aggr="sum",
                )
                for _ in range(2)
            ]
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def _project_inputs(self, data: Any) -> dict[str, torch.Tensor]:
        """Helper: project inputs."""
        x_dict: dict[str, torch.Tensor] = {}
        for node_type, linear in self.input_linears.items():
            features = data[node_type].x.float()
            x_dict[node_type] = F.dropout(F.relu(linear(features)), p=self.dropout, training=self.training)
        return x_dict

    def encode(self, data: Any) -> dict[str, torch.Tensor]:
        """Encode."""
        x_dict = self._project_inputs(data)
        edge_weight_dict = {
            edge_type: data[edge_type].edge_weight.float() for edge_type in data.edge_types
        }
        for conv in self.convs:
            x_dict = conv(x_dict, data.edge_index_dict, edge_weight_dict=edge_weight_dict)
            x_dict = {
                node_type: F.dropout(F.relu(features), p=self.dropout, training=self.training)
                for node_type, features in x_dict.items()
            }
        return x_dict

    def forward(self, data: Any) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Run the forward pass."""
        embedding_dict = self.encode(data)
        logits = self.classifier(embedding_dict["Website"])
        return logits, embedding_dict
