"""Tests for the adversarial EERM editor objective.

The first implementation rewarded each environment with the *squared* deviation of its risk
from the mean risk. That reward is symmetric, so with K=2 both environments received an
identical reward, the mean baseline cancelled it exactly, and the editors got no gradient.
These tests pin the corrected objective: the reward is the signed deviation, which is the
partial derivative of the risk variance with respect to each environment's risk.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


def editor_reward(risks: "torch.Tensor") -> "torch.Tensor":
    """Per-environment REINFORCE reward used by the editor step.

    Mirrors ``src/transfer/eerm_adversarial.eerm_step``: the signed deviation from the mean.
    """
    return risks - risks.mean()


def squared_reward_with_baseline(risks: "torch.Tensor") -> "torch.Tensor":
    """The defective original: squared deviation minus its own mean, kept as a regression guard."""
    reward = (risks - risks.mean()).pow(2)
    return reward - reward.mean()


@pytest.mark.parametrize("risks", [[0.25, 0.75], [0.61, 0.42], [0.5, 0.5000001]])
def test_two_environment_advantage_is_non_degenerate(risks):
    """With K=2 the corrected reward separates the two environments; the old one cannot."""
    r = torch.tensor(risks, dtype=torch.double)
    old = squared_reward_with_baseline(r)
    assert torch.allclose(old, torch.zeros_like(old), atol=1e-15), (
        "the squared reward must be shown to vanish for K=2, otherwise this guard is pointless"
    )
    new = editor_reward(r)
    assert not torch.allclose(new, torch.zeros_like(new), atol=1e-12)
    # the higher-risk environment is pushed up, the lower-risk one down
    assert new[int(torch.argmax(r))] > 0 > new[int(torch.argmin(r))]


def test_reward_matches_variance_gradient():
    """reward_k must be proportional to d(Var)/d(risk_k) for any K."""
    r = torch.tensor([0.3, 0.5, 0.7, 0.9], dtype=torch.double, requires_grad=True)
    variance = (r - r.mean()).pow(2).mean()
    (grad,) = torch.autograd.grad(variance, r)
    reward = editor_reward(r.detach())
    scale = grad / reward
    assert torch.allclose(scale, scale[0].expand_as(scale)), "reward is not parallel to the gradient"
    assert float(scale[0]) > 0, "reward must point along increasing variance"


def test_reward_is_zero_mean_so_no_baseline_is_needed():
    """The signed deviations sum to zero, so subtracting their mean would be a no-op."""
    for risks in ([0.1, 0.9], [0.2, 0.4, 0.6], [0.5] * 4):
        r = torch.tensor(risks, dtype=torch.double)
        assert abs(float(editor_reward(r).mean())) < 1e-15


def test_equal_risks_give_no_signal():
    """Identical risks carry no variance information, so every advantage must be zero."""
    r = torch.full((3,), 0.42, dtype=torch.double)
    assert torch.allclose(editor_reward(r), torch.zeros_like(r))
