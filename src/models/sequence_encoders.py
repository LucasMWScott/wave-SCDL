"""Temporal sequence encoders for coastal models."""

from __future__ import annotations

import torch
from torch import nn


class TransformerSequenceEncoder(nn.Module):
    """Wrapper that applies a stack of transformer blocks."""

    def __init__(self, blocks: nn.ModuleList) -> None:
        super().__init__()
        self.blocks = blocks

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


class LSTMSequenceEncoder(nn.Module):
    """LSTM temporal encoder returning `[B, T, model_dim]` tokens."""

    def __init__(
        self,
        input_dim: int,
        model_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = False,
        pooling: str = "last",
        layer_norm: bool = True,
    ) -> None:
        super().__init__()
        if input_dim < 1:
            raise ValueError(f"input_dim must be >= 1, got {input_dim}")
        if model_dim < 1:
            raise ValueError(f"model_dim must be >= 1, got {model_dim}")
        if hidden_dim < 1:
            raise ValueError(f"hidden_dim must be >= 1, got {hidden_dim}")
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")

        self.input_dim = int(input_dim)
        self.model_dim = int(model_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.bidirectional = bool(bidirectional)
        self.pooling = str(pooling or "last").strip().lower()
        self.use_layer_norm = bool(layer_norm)

        if self.pooling not in {"last", "mean", "attention"}:
            raise ValueError("LSTM pooling must be one of: last, mean, attention")

        self.input_proj = (
            nn.Identity()
            if self.input_dim == self.model_dim
            else nn.Linear(self.input_dim, self.model_dim)
        )
        lstm_dropout = float(dropout) if self.num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=self.model_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=lstm_dropout,
            bidirectional=self.bidirectional,
            batch_first=True,
        )
        lstm_out_dim = self.hidden_dim * (2 if self.bidirectional else 1)
        self.output_proj = (
            nn.Identity()
            if lstm_out_dim == self.model_dim
            else nn.Linear(lstm_out_dim, self.model_dim)
        )
        self.output_norm = nn.LayerNorm(self.model_dim) if self.use_layer_norm else nn.Identity()
        self.output_dropout = nn.Dropout(float(dropout))
        self.attn_pool = nn.Linear(self.model_dim, 1) if self.pooling == "attention" else None

    def pool_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        if self.pooling == "mean":
            return tokens.mean(dim=1)
        if self.pooling == "attention":
            assert self.attn_pool is not None
            scores = self.attn_pool(tokens).squeeze(-1)
            weights = torch.softmax(scores, dim=-1)
            return torch.sum(tokens * weights.unsqueeze(-1), dim=1)
        return tokens[:, -1, :]

    def forward(
        self,
        x: torch.Tensor,
        return_pooled: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        x = self.input_proj(x)
        tokens, _ = self.lstm(x)
        tokens = self.output_proj(tokens)
        tokens = self.output_norm(tokens)
        tokens = self.output_dropout(tokens)
        if return_pooled:
            return tokens, self.pool_tokens(tokens)
        return tokens
