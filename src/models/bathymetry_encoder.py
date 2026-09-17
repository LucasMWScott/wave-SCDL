"""Compact bathymetry encoder blocks for coastal transformer models."""

from __future__ import annotations

import math

import torch
from torch import nn


class BathymetryBranchDropout(nn.Module):
    """Drop the entire bathymetry branch per sample during training."""

    def __init__(self, p: float = 0.0) -> None:
        super().__init__()
        self.p = float(max(0.0, min(1.0, p)))

    def forward(
        self,
        tokens: torch.Tensor,
        summary: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if (not self.training) or self.p <= 0.0:
            return tokens, summary
        keep_prob = 1.0 - self.p
        if keep_prob <= 0.0:
            return torch.zeros_like(tokens), torch.zeros_like(summary)
        mask = (
            torch.bernoulli(
                torch.full(
                    (tokens.size(0), 1, 1), keep_prob, device=tokens.device, dtype=tokens.dtype
                )
            )
            / keep_prob
        )
        tokens = tokens * mask
        summary = summary * mask.squeeze(1)
        return tokens, summary


class BathymetryCNNEncoder(nn.Module):
    """Compact CNN that emits coarse spatial bathymetry tokens plus a summary."""

    def __init__(
        self,
        in_channels: int = 2,
        conv_channels: tuple[int, ...] = (32, 64, 96),
        model_dim: int = 128,
        num_tokens: int = 4,
        dropout2d: float = 0.05,
        token_dropout: float = 0.10,
    ) -> None:
        super().__init__()
        if int(in_channels) < 1:
            raise ValueError(f"in_channels must be >= 1, got {in_channels}")
        if not conv_channels:
            raise ValueError("conv_channels must contain at least one stage")
        if int(num_tokens) < 1:
            raise ValueError(f"num_tokens must be >= 1, got {num_tokens}")

        grid_side = int(round(math.sqrt(int(num_tokens))))
        if grid_side * grid_side != int(num_tokens):
            raise ValueError("num_tokens must be a perfect square so spatial pooling stays 2D")

        def _group_count(channels: int) -> int:
            for groups in (8, 4, 2, 1):
                if channels % groups == 0:
                    return groups
            return 1

        def block(cin: int, cout: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(cin, cout, kernel_size=3, padding=1),
                nn.GroupNorm(_group_count(cout), cout),
                nn.GELU(),
                nn.Conv2d(cout, cout, kernel_size=3, padding=1),
                nn.GroupNorm(_group_count(cout), cout),
                nn.GELU(),
                nn.AvgPool2d(kernel_size=2),
                nn.Dropout2d(dropout2d),
            )

        layers = []
        prev_channels = int(in_channels)
        for channels in conv_channels:
            layers.append(block(prev_channels, int(channels)))
            prev_channels = int(channels)

        self.in_channels = int(in_channels)
        self.num_tokens = int(num_tokens)
        self.grid_side = grid_side
        self.net = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((self.grid_side, self.grid_side))
        self.proj = nn.Linear(prev_channels, int(model_dim))
        self.norm = nn.LayerNorm(int(model_dim))
        self.token_dropout = nn.Dropout(float(token_dropout))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 4:
            raise ValueError(
                f"Bathymetry encoder expects x shape [B, C, H, W], got {tuple(x.shape)}"
            )
        if x.size(1) != self.in_channels:
            raise ValueError(
                f"Bathymetry encoder channel mismatch: expected {self.in_channels}, got {x.size(1)}"
            )

        h = self.pool(self.net(x))
        tokens = h.flatten(start_dim=2).transpose(1, 2)
        tokens = self.norm(self.proj(tokens))
        tokens = self.token_dropout(tokens)
        summary = tokens.mean(dim=1)
        return tokens, summary
