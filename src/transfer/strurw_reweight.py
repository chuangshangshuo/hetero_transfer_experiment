from __future__ import annotations

import copy
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch


FORWARD_RELATION_TRIPLES: dict[str, tuple[str, str, str]] = {
    "hosted_on": ("Website", "hosted_on", "IP"),
    "uses_cert": ("Website", "uses_cert", "Certificate"),
    "uses_ns": ("Website", "uses_ns", "NameServer"),
    "registered_via": ("Website", "registered_via", "Registrar"),
    "referenced_by": ("Website", "referenced_by", "ExternalReference"),
}

REVERSE_RELATION_TRIPLES: dict[str, tuple[str, str, str]] = {
    "hosted_on": ("IP", "rev_hosted_on", "Website"),
    "uses_cert": ("Certificate", "rev_uses_cert", "Website"),
    "uses_ns": ("NameServer", "rev_uses_ns", "Website"),
    "registered_via": ("Registrar", "rev_registered_via", "Website"),
    "referenced_by": ("ExternalReference", "rev_referenced_by", "Website"),
}


@dataclass
class StruRWResult:
    graph_data: Any
    edge_audit: pd.DataFrame
    csbm_audit: pd.DataFrame


class StruRWReweighter:
    def __init__(
        self,
        relations: list[str],
        edge_weight_clip: tuple[float, float] = (0.1, 10.0),
        laplace_alpha: float = 1.0,
    ) -> None:
        unknown = sorted(set(relations) - set(FORWARD_RELATION_TRIPLES))
        if unknown:
            raise ValueError(f"Unsupported StruRW relations: {unknown}")
        self.relations = relations
        self.clip_min = float(edge_weight_clip[0])
        self.clip_max = float(edge_weight_clip[1])
        self.laplace_alpha = float(laplace_alpha)

    @staticmethod
    def _eligible_attr_members(
        data: Any,
        edge_type: tuple[str, str, str],
        labels: torch.Tensor,
        eligible_mask: torch.Tensor,
    ) -> dict[int, list[int]]:
        edge_index = data[edge_type].edge_index.detach().cpu()
        labels = labels.detach().cpu()
        eligible_mask = eligible_mask.detach().cpu().bool()
        members: dict[int, list[int]] = defaultdict(list)
        for website_idx, attr_idx in edge_index.t().tolist():
            if bool(eligible_mask[int(website_idx)]) and int(labels[int(website_idx)]) in (0, 1):
                members[int(attr_idx)].append(int(website_idx))
        return members

    def _estimate_transition(
        self,
        data: Any,
        edge_type: tuple[str, str, str],
        labels: torch.Tensor,
        eligible_mask: torch.Tensor,
    ) -> tuple[np.ndarray, np.ndarray]:
        labels_np = labels.detach().cpu().numpy().astype(int)
        counts = np.full((2, 2), self.laplace_alpha, dtype=np.float64)
        members = self._eligible_attr_members(data, edge_type, labels, eligible_mask)
        for websites in members.values():
            unique_websites = sorted(set(websites))
            if len(unique_websites) < 2:
                continue
            for source_idx in unique_websites:
                source_label = labels_np[source_idx]
                if source_label not in (0, 1):
                    continue
                for neighbor_idx in unique_websites:
                    if neighbor_idx == source_idx:
                        continue
                    neighbor_label = labels_np[neighbor_idx]
                    if neighbor_label in (0, 1):
                        counts[source_label, neighbor_label] += 1.0
        probs = counts / counts.sum(axis=1, keepdims=True).clip(min=1e-12)
        return counts, probs

    def reweight(
        self,
        data: Any,
        source_labels: torch.Tensor,
        source_mask: torch.Tensor,
        target_pseudo_labels: torch.Tensor,
        target_confident_mask: torch.Tensor,
        iteration: int,
    ) -> StruRWResult:
        new_data = copy.deepcopy(data)
        source_labels = source_labels.detach().cpu().long()
        source_mask = source_mask.detach().cpu().bool()
        target_pseudo_labels = target_pseudo_labels.detach().cpu().long()
        target_confident_mask = target_confident_mask.detach().cpu().bool()

        edge_rows: list[dict[str, float | int | str]] = []
        csbm_rows: list[dict[str, float | int | str]] = []
        for relation in self.relations:
            edge_type = FORWARD_RELATION_TRIPLES[relation]
            reverse_edge_type = REVERSE_RELATION_TRIPLES[relation]
            source_counts, source_probs = self._estimate_transition(
                data=data,
                edge_type=edge_type,
                labels=source_labels,
                eligible_mask=source_mask,
            )
            target_counts, target_probs = self._estimate_transition(
                data=data,
                edge_type=edge_type,
                labels=target_pseudo_labels,
                eligible_mask=target_confident_mask,
            )
            target_observed_pairs = float(target_counts.sum() - (4.0 * self.laplace_alpha))
            if target_observed_pairs <= 0:
                ratio = np.ones_like(source_probs)
            else:
                ratio = target_probs / np.clip(source_probs, 1e-12, None)
            source_members = self._eligible_attr_members(data, edge_type, source_labels, source_mask)
            edge_index = data[edge_type].edge_index.detach().cpu()
            old_weight = data[edge_type].edge_weight.detach().cpu().float()
            new_weight = old_weight.clone()
            labels_np = source_labels.numpy().astype(int)
            multipliers: list[float] = []
            pair_weight_map: dict[tuple[int, int], float] = {}

            for edge_pos, (website_idx, attr_idx) in enumerate(edge_index.t().tolist()):
                website_idx = int(website_idx)
                attr_idx = int(attr_idx)
                if not bool(source_mask[website_idx]) or labels_np[website_idx] not in (0, 1):
                    pair_weight_map[(website_idx, attr_idx)] = float(old_weight[edge_pos].item())
                    continue
                own_label = int(labels_np[website_idx])
                neighbor_labels = [
                    int(labels_np[neighbor_idx])
                    for neighbor_idx in source_members.get(attr_idx, [])
                    if int(neighbor_idx) != website_idx and int(labels_np[neighbor_idx]) in (0, 1)
                ]
                if neighbor_labels:
                    multiplier = float(np.mean([ratio[own_label, neighbor_label] for neighbor_label in neighbor_labels]))
                else:
                    multiplier = float(np.mean(ratio[own_label, :]))
                multiplier = float(np.clip(multiplier, self.clip_min, self.clip_max))
                raw_weight = float(np.expm1(float(old_weight[edge_pos].item())))
                reweighted = float(np.log1p(max(0.0, raw_weight * multiplier)))
                new_weight[edge_pos] = reweighted
                multipliers.append(multiplier)
                pair_weight_map[(website_idx, attr_idx)] = reweighted

            new_data[edge_type].edge_weight = new_weight
            reverse_edge_index = data[reverse_edge_type].edge_index.detach().cpu()
            reverse_old_weight = data[reverse_edge_type].edge_weight.detach().cpu().float()
            reverse_new_weight = reverse_old_weight.clone()
            for edge_pos, (attr_idx, website_idx) in enumerate(reverse_edge_index.t().tolist()):
                mapped = pair_weight_map.get((int(website_idx), int(attr_idx)))
                if mapped is not None:
                    reverse_new_weight[edge_pos] = float(mapped)
            new_data[reverse_edge_type].edge_weight = reverse_new_weight

            if multipliers:
                multiplier_array = np.array(multipliers, dtype=np.float64)
                min_multiplier = float(multiplier_array.min())
                max_multiplier = float(multiplier_array.max())
                mean_multiplier = float(multiplier_array.mean())
            else:
                min_multiplier = max_multiplier = mean_multiplier = float("nan")
            edge_rows.append(
                {
                    "iteration": iteration,
                    "relation": relation,
                    "source_edges_reweighted": int(len(multipliers)),
                    "multiplier_min": min_multiplier,
                    "multiplier_mean": mean_multiplier,
                    "multiplier_max": max_multiplier,
                    "edge_weight_min": float(new_weight.min().item()) if new_weight.numel() else float("nan"),
                    "edge_weight_mean": float(new_weight.mean().item()) if new_weight.numel() else float("nan"),
                    "edge_weight_max": float(new_weight.max().item()) if new_weight.numel() else float("nan"),
                    "clip_min": self.clip_min,
                    "clip_max": self.clip_max,
                    "target_observed_pairs": target_observed_pairs,
                }
            )
            for self_label in (0, 1):
                for neighbor_label in (0, 1):
                    csbm_rows.append(
                        {
                            "iteration": iteration,
                            "relation": relation,
                            "self_label": self_label,
                            "neighbor_label": neighbor_label,
                            "source_count": float(source_counts[self_label, neighbor_label]),
                            "target_pseudo_count": float(target_counts[self_label, neighbor_label]),
                            "source_probability": float(source_probs[self_label, neighbor_label]),
                            "target_pseudo_probability": float(target_probs[self_label, neighbor_label]),
                            "ratio": float(ratio[self_label, neighbor_label]),
                        }
                    )

        return StruRWResult(
            graph_data=new_data,
            edge_audit=pd.DataFrame(edge_rows),
            csbm_audit=pd.DataFrame(csbm_rows),
        )
