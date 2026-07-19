"""Generate network-schema and metapath views used by HeCo pre-training."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GraphConv, HeteroConv, Linear


FORWARD_RELATION_TRIPLES: dict[str, tuple[str, str, str]] = {
    "hosted_on": ("Website", "hosted_on", "IP"),
    "uses_cert": ("Website", "uses_cert", "Certificate"),
    "uses_ns": ("Website", "uses_ns", "NameServer"),
    "registered_via": ("Website", "registered_via", "Registrar"),
    "referenced_by": ("Website", "referenced_by", "ExternalReference"),
}


@dataclass
class MetapathArtifacts:
    """Metapath Artifacts."""
    adjacency: dict[str, torch.Tensor]
    score_matrices: dict[str, torch.Tensor]
    positive_mask: torch.Tensor
    positive_audit: pd.DataFrame


class SchemaViewEncoder(nn.Module):
    """Schema View Encoder (PyTorch module)."""
    def __init__(
        self,
        node_feature_dims: dict[str, int],
        edge_types: list[tuple[str, str, str]],
        hidden_dim: int = 64,
        dropout: float = 0.30,
    ) -> None:
        """Initialise the instance."""
        super().__init__()
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

    def forward(self, data: Any) -> torch.Tensor:
        """Run the forward pass."""
        x_dict = {
            node_type: F.dropout(F.relu(linear(data[node_type].x.float())), p=self.dropout, training=self.training)
            for node_type, linear in self.input_linears.items()
        }
        edge_weight_dict = {
            edge_type: data[edge_type].edge_weight.float() for edge_type in data.edge_types
        }
        for conv in self.convs:
            x_dict = conv(x_dict, data.edge_index_dict, edge_weight_dict=edge_weight_dict)
            x_dict = {
                node_type: F.dropout(F.relu(features), p=self.dropout, training=self.training)
                for node_type, features in x_dict.items()
            }
        return x_dict["Website"]


class MetapathViewEncoder(nn.Module):
    """Metapath View Encoder (PyTorch module)."""
    def __init__(
        self,
        website_feature_dim: int,
        metapath_names: list[str],
        hidden_dim: int = 64,
        dropout: float = 0.30,
    ) -> None:
        """Initialise the instance."""
        super().__init__()
        self.dropout = dropout
        self.website_linear = nn.Linear(website_feature_dim, hidden_dim)
        self.path_linears = nn.ModuleDict(
            {metapath_name: nn.Linear(hidden_dim, hidden_dim) for metapath_name in metapath_names}
        )
        self.output_linear = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, website_features: torch.Tensor, metapath_adjacency: dict[str, torch.Tensor]) -> torch.Tensor:
        """Run the forward pass."""
        base = F.dropout(F.relu(self.website_linear(website_features.float())), p=self.dropout, training=self.training)
        path_outputs: list[torch.Tensor] = []
        for metapath_name, adjacency in metapath_adjacency.items():
            propagated = torch.sparse.mm(adjacency.to(base.device), base)
            path_outputs.append(F.relu(self.path_linears[metapath_name](propagated)))
        if not path_outputs:
            raise ValueError("MetapathViewEncoder requires at least one metapath adjacency")
        stacked = torch.stack(path_outputs, dim=0).mean(dim=0)
        return F.dropout(F.relu(self.output_linear(stacked)), p=self.dropout, training=self.training)


def _build_weighted_incidence(
    data: Any,
    edge_type: tuple[str, str, str],
    num_websites: int,
) -> torch.Tensor:
    """Build weighted incidence."""
    edge_index = data[edge_type].edge_index.detach().cpu()
    edge_weight = data[edge_type].edge_weight.detach().cpu().float()
    target_count = int(data[edge_type[2]].x.shape[0])
    incidence = torch.zeros((num_websites, target_count), dtype=torch.float32)
    incidence[edge_index[0].long(), edge_index[1].long()] = edge_weight
    return incidence


def _row_normalize_dense(matrix: torch.Tensor) -> torch.Tensor:
    """Helper: row normalize dense."""
    matrix = matrix.clone()
    matrix.fill_diagonal_(0.0)
    row_sum = matrix.sum(dim=1, keepdim=True).clamp_min(1e-12)
    return matrix / row_sum


def _dense_to_sparse(matrix: torch.Tensor) -> torch.Tensor:
    """Helper: dense to sparse."""
    indices = matrix.nonzero(as_tuple=False).t().contiguous()
    if indices.numel() == 0:
        indices = torch.empty((2, 0), dtype=torch.long)
        values = torch.empty((0,), dtype=torch.float32)
    else:
        values = matrix[indices[0], indices[1]].float()
    return torch.sparse_coo_tensor(indices, values, matrix.shape).coalesce()


def build_metapath_artifacts(
    data: Any,
    metapath_relations: dict[str, str],
    top_k: int,
    min_metapath_support: int,
) -> MetapathArtifacts:
    """Build metapath artifacts."""
    num_websites = int(data["Website"].x.shape[0])
    dense_scores: dict[str, torch.Tensor] = {}
    normalized_adjacency: dict[str, torch.Tensor] = {}
    support_count = torch.zeros((num_websites, num_websites), dtype=torch.int16)
    total_score = torch.zeros((num_websites, num_websites), dtype=torch.float32)

    for metapath_name, relation in metapath_relations.items():
        edge_type = FORWARD_RELATION_TRIPLES[relation]
        incidence = _build_weighted_incidence(data, edge_type, num_websites)
        score = incidence @ incidence.t()
        score.fill_diagonal_(0.0)
        dense_scores[metapath_name] = score
        normalized_adjacency[metapath_name] = _dense_to_sparse(_row_normalize_dense(score))
        support_count += (score > 0).to(torch.int16)
        total_score += score

    positive_mask = torch.eye(num_websites, dtype=torch.bool)
    required_support = max(1, int(min_metapath_support))
    eligible = support_count >= required_support
    if eligible.sum().item() == 0 and required_support > 1:
        eligible = support_count >= 1

    audit_rows: list[dict[str, int | float | str]] = []
    for node_idx in range(num_websites):
        candidate_scores = total_score[node_idx].clone()
        candidate_scores[~eligible[node_idx]] = 0.0
        candidate_scores[node_idx] = 0.0
        nonzero = (candidate_scores > 0).nonzero(as_tuple=False).flatten()
        if nonzero.numel() > 0:
            k = min(top_k, int(nonzero.numel()))
            selected = torch.topk(candidate_scores, k=k).indices
            positive_mask[node_idx, selected] = True
        selected_count = int(positive_mask[node_idx].sum().item())
        audit_rows.append(
            {
                "graph_node_index": node_idx,
                "num_positive_targets": selected_count,
                "num_nonself_positive_targets": max(0, selected_count - 1),
                "metapath_support_nonzero": int((support_count[node_idx] > 0).sum().item()),
                "top_total_score": float(candidate_scores.max().item()) if nonzero.numel() else 0.0,
            }
        )

    audit = pd.DataFrame(audit_rows)
    return MetapathArtifacts(
        adjacency=normalized_adjacency,
        score_matrices=dense_scores,
        positive_mask=positive_mask,
        positive_audit=audit,
    )
