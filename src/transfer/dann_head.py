from __future__ import annotations

import torch
from torch import nn
from torch.autograd import Function


class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, scale: float) -> torch.Tensor:
        ctx.scale = scale
        return inputs.view_as(inputs)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        return grad_output.neg() * ctx.scale, None


class GradientReversalLayer(nn.Module):
    def __init__(self, scale: float = 1.0) -> None:
        super().__init__()
        self.scale = scale

    def forward(self, inputs: torch.Tensor, scale: float | None = None) -> torch.Tensor:
        scale = self.scale if scale is None else scale
        return GradientReversalFunction.apply(inputs, scale)


class DomainDiscriminator(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.30) -> None:
        super().__init__()
        self.grl = GradientReversalLayer(scale=1.0)
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, inputs: torch.Tensor, reverse_scale: float = 1.0) -> torch.Tensor:
        reversed_inputs = self.grl(inputs, scale=reverse_scale)
        return self.network(reversed_inputs)
