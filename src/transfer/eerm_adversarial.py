"""A faithful EERM implementation: adversarial graph editors + variance-regularised risk.

The frozen pipeline labelled its "EERM" arm as such but actually ran a fixed K-means partition of
the frozen HeCo embedding plus an IRMv1 penalty. Wu et al. (ICLR 2022) instead *learn* K graph
editors that adversarially modify the graph structure so as to maximise the variance of risks
across the generated environments, while the GNN minimises mean risk plus beta times that
variance.

This module implements that mechanism:

* ``EdgeEditor`` holds, for each environment k and each forward relation, a logit per existing
  edge (a keep/drop policy) and a logit per candidate non-edge drawn once per run (an add policy).
* Each environment is realised by sampling those Bernoulli policies, giving a genuinely different
  message-passing graph per environment.
* The editors are updated by REINFORCE with the per-environment squared deviation from the mean
  risk as reward, i.e. they are pushed to make the environments disagree.
* The GNN is then updated on the sampled environments with ``mean(risk) + beta * var(risk)``.

Differences from the paper that are stated in the manuscript: the editors act on the
Website-centric forward relations (and their materialised reverses) rather than on a homogeneous
adjacency; the metapath view used by the HeCo encoder is held fixed while the schema view is
edited; and the candidate-addition pool is sampled once per run rather than re-proposed each step.
"""
from __future__ import annotations

import copy
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

FORWARD_RELATIONS = ("hosted_on", "uses_cert", "uses_ns", "registered_via", "referenced_by")


def _forward_edge_types(data) -> list[tuple[str, str, str]]:
    """Return the Website-centric forward edge types present in the graph."""
    return [et for et in data.edge_types if et[1] in FORWARD_RELATIONS]


def _reverse_of(edge_type: tuple[str, str, str]) -> tuple[str, str, str]:
    """Return the materialised reverse edge type for a forward edge type."""
    src, rel, dst = edge_type
    return (dst, f"rev_{rel}", src)


class EdgeEditor(nn.Module):
    """K per-environment edge keep/add policies over the Website-centric relations."""

    def __init__(self, data, n_env: int, add_pool: dict[tuple, torch.Tensor], init_keep_logit: float = 2.0) -> None:
        """Create keep logits for existing edges and add logits for the candidate pool."""
        super().__init__()
        self.n_env = int(n_env)
        self.edge_types = _forward_edge_types(data)
        self.keep = nn.ParameterDict()
        self.add = nn.ParameterDict()
        self.add_pool = {}
        for et in self.edge_types:
            key = "__".join(et)
            n_edge = int(data[et].edge_index.shape[1])
            self.keep[key] = nn.Parameter(torch.full((self.n_env, n_edge), float(init_keep_logit)))
            pool = add_pool.get(et)
            if pool is not None and pool.numel():
                self.add_pool[key] = pool
                self.add[key] = nn.Parameter(torch.full((self.n_env, pool.shape[1]), -2.0))

    def sample(self, env: int) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor]:
        """Sample one environment; returns keep masks, add masks and the summed log-probability."""
        keep_masks, add_masks, log_prob = {}, {}, 0.0
        for et in self.edge_types:
            key = "__".join(et)
            p_keep = torch.sigmoid(self.keep[key][env])
            m = torch.bernoulli(p_keep)
            log_prob = log_prob + (m * torch.log(p_keep + 1e-9) + (1 - m) * torch.log(1 - p_keep + 1e-9)).sum()
            keep_masks[key] = m.bool()
            if key in self.add:
                p_add = torch.sigmoid(self.add[key][env])
                a = torch.bernoulli(p_add)
                log_prob = log_prob + (a * torch.log(p_add + 1e-9) + (1 - a) * torch.log(1 - p_add + 1e-9)).sum()
                add_masks[key] = a.bool()
        return keep_masks, add_masks, log_prob


def build_add_pool(data, n_candidates: int, generator: torch.Generator) -> dict[tuple, torch.Tensor]:
    """Draw a fixed pool of candidate non-edges per forward relation."""
    pool = {}
    for et in _forward_edge_types(data):
        src, _, dst = et
        n_src = int(data[src].x.shape[0])
        n_dst = int(data[dst].x.shape[0])
        k = min(int(n_candidates), n_src * n_dst)
        if k <= 0:
            continue
        s = torch.randint(0, n_src, (k,), generator=generator)
        d = torch.randint(0, n_dst, (k,), generator=generator)
        pool[et] = torch.stack([s, d], dim=0)
    return pool


def apply_edits(data, editor: EdgeEditor, keep_masks: dict, add_masks: dict):
    """Return a copy of `data` whose Website-centric edges follow the sampled policies."""
    edited = copy.copy(data)
    for et in editor.edge_types:
        key = "__".join(et)
        base = data[et].edge_index
        kept = base[:, keep_masks[key]]
        if key in add_masks and add_masks[key].any():
            extra = editor.add_pool[key][:, add_masks[key]].to(base.device)
            kept = torch.cat([kept, extra], dim=1)
        rev = _reverse_of(et)
        edited[et].edge_index = kept
        edited[et].edge_weight = torch.ones(kept.shape[1], device=base.device, dtype=data[et].edge_weight.dtype)
        if rev in data.edge_types:
            edited[rev].edge_index = kept.flip(0)
            edited[rev].edge_weight = edited[et].edge_weight
    return edited


def eerm_step(
    model,
    data,
    metapath_adjacency,
    labels: torch.Tensor,
    train_idx: torch.Tensor,
    editor: EdgeEditor,
    editor_opt: torch.optim.Optimizer,
    model_opt: torch.optim.Optimizer,
    beta: float,
) -> dict[str, float]:
    """One EERM iteration: adversarial editor update, then variance-regularised model update."""
    # --- editor step: push environments apart (REINFORCE on squared deviation from mean risk)
    log_probs, risks = [], []
    for k in range(editor.n_env):
        keep_masks, add_masks, logp = editor.sample(k)
        edited = apply_edits(data, editor, keep_masks, add_masks)
        with torch.no_grad():
            logits, _, _ = model(edited, metapath_adjacency)
            risk = F.cross_entropy(logits[train_idx], labels[train_idx])
        log_probs.append(logp)
        risks.append(risk)
    risk_vec = torch.stack(risks)
    # REINFORCE on the variance objective. For Var = (1/K) * sum_j (r_j - rbar)^2 the partial
    # derivative w.r.t. environment k's risk is (2/K) * (r_k - rbar), so the per-environment
    # reward is the SIGNED deviation from the mean risk, not its square. Squaring makes the
    # reward symmetric: with K=2 both environments then receive an identical reward, the mean
    # baseline cancels it exactly, and the editors get no gradient at all.
    reward = (risk_vec - risk_vec.mean()).detach()
    # the signed deviations already sum to zero, so the mean baseline is zero by construction
    editor_loss = -torch.stack([lp * r for lp, r in zip(log_probs, reward)]).mean()
    editor_opt.zero_grad()
    editor_loss.backward()
    editor_opt.step()

    # --- model step: minimise mean risk + beta * variance across the sampled environments
    model_risks = []
    with torch.no_grad():
        sampled = [editor.sample(k) for k in range(editor.n_env)]
    for keep_masks, add_masks, _ in sampled:
        edited = apply_edits(data, editor, keep_masks, add_masks)
        logits, _, _ = model(edited, metapath_adjacency)
        model_risks.append(F.cross_entropy(logits[train_idx], labels[train_idx]))
    mr = torch.stack(model_risks)
    loss = mr.mean() + float(beta) * mr.var(unbiased=False)
    model_opt.zero_grad()
    loss.backward()
    model_opt.step()
    return {
        "loss": float(loss.detach()),
        "risk_mean": float(mr.mean().detach()),
        "risk_var": float(mr.var(unbiased=False).detach()),
        "editor_reward": float(reward.mean()),
    }
