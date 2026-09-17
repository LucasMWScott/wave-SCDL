"""Coastal-conditioned transformer architecture for hybrid wave downscaling.

This module introduces an additive architecture path that preserves the
existing model interface:
    forward(x_dynamic, x_static=None, ...) -> dict[str, torch.Tensor]

The static branch can be disabled for no-static baselines without changing
the batch interface.

Design highlights:
- RoPE-enabled self-attention over dynamic sequence tokens.
- GeGLU feed-forward blocks.
- SDPA fast path with deterministic manual-attention fallback.
- Task-decoupled cross-attention heads for hs regression and tp/dir/dp classification.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from .bathymetry_encoder import BathymetryBranchDropout, BathymetryCNNEncoder
from .sequence_encoders import LSTMSequenceEncoder, TransformerSequenceEncoder


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate even/odd channels used by rotary position embedding."""
    x_even = x[..., ::2]
    x_odd = x[..., 1::2]
    rotated = torch.stack((-x_odd, x_even), dim=-1)
    return rotated.flatten(start_dim=-2)


def _apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embedding to query/key tensors."""
    q_rot = (q * cos) + (_rotate_half(q) * sin)
    k_rot = (k * cos) + (_rotate_half(k) * sin)
    return q_rot, k_rot


def _build_bin_centers(
    lower: float, upper: float, num_bins: int, circular: bool = False
) -> torch.Tensor:
    if num_bins < 1:
        raise ValueError(f"num_bins must be >= 1, got {num_bins}")
    if num_bins == 1:
        return torch.tensor([(float(lower) + float(upper)) * 0.5], dtype=torch.float32)
    if circular:
        return torch.linspace(float(lower), float(upper), int(num_bins) + 1, dtype=torch.float32)[
            :-1
        ]
    return torch.linspace(float(lower), float(upper), int(num_bins), dtype=torch.float32)


def _infer_alias_set(raw_aliases: object, default: Sequence[str]) -> set[str]:
    if raw_aliases is None:
        return {str(item).strip().lower() for item in default}
    if isinstance(raw_aliases, dict):
        items = []
        for value in raw_aliases.values():
            if isinstance(value, (list, tuple, set)):
                items.extend(value)
            else:
                items.append(value)
        return {str(item).strip().lower() for item in items if str(item).strip()}
    if isinstance(raw_aliases, (list, tuple, set)):
        return {str(item).strip().lower() for item in raw_aliases if str(item).strip()}
    return {str(raw_aliases).strip().lower()}


def _feature_matches_alias(feature_name: str, aliases: set[str]) -> bool:
    lower = str(feature_name).strip().lower()
    if lower in aliases:
        return True
    for alias in aliases:
        if lower.endswith(f"_{alias}"):
            return True
        if f"_{alias}_" in lower:
            return True
    return False


class RotaryEmbedding(nn.Module):
    """Rotary embedding cache for a given attention head dimension."""

    def __init__(self, head_dim: int, base: float = 10000.0) -> None:
        super().__init__()
        if head_dim < 2:
            raise ValueError(f"RoPE head_dim must be >= 2, got {head_dim}")
        if head_dim % 2 != 0:
            raise ValueError(f"RoPE head_dim must be even, got {head_dim}")

        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def cos_sin(
        self, seq_len: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return cos/sin tensors with shape [1, 1, T, head_dim]."""
        positions = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.einsum("t,f->tf", positions, self.inv_freq)
        cos = freqs.cos().repeat_interleave(2, dim=-1)
        sin = freqs.sin().repeat_interleave(2, dim=-1)
        cos = cos.to(dtype=dtype).unsqueeze(0).unsqueeze(0)
        sin = sin.to(dtype=dtype).unsqueeze(0).unsqueeze(0)
        return cos, sin


class GeGLU(nn.Module):
    """Feed-forward block using GeGLU gating."""

    def __init__(self, model_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.proj_in = nn.Linear(model_dim, hidden_dim * 2)
        self.proj_out = nn.Linear(hidden_dim, model_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value, gate = self.proj_in(x).chunk(2, dim=-1)
        x = value * F.gelu(gate)
        x = self.dropout(x)
        x = self.proj_out(x)
        return self.dropout(x)


class MultiheadAttentionWithFallback(nn.Module):
    """MHA wrapper with SDPA fast path and manual fallback."""

    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        attn_dropout: float,
        proj_dropout: float,
        use_sdpa: bool,
        rope: Optional[RotaryEmbedding] = None,
    ) -> None:
        super().__init__()
        if model_dim % num_heads != 0:
            raise ValueError(
                f"model_dim ({model_dim}) must be divisible by num_heads ({num_heads})"
            )

        self.model_dim = int(model_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.model_dim // self.num_heads
        self.scale = self.head_dim**-0.5
        self.attn_dropout = float(attn_dropout)
        self.use_sdpa = bool(use_sdpa)
        self.rope = rope

        self.q_proj = nn.Linear(self.model_dim, self.model_dim)
        self.k_proj = nn.Linear(self.model_dim, self.model_dim)
        self.v_proj = nn.Linear(self.model_dim, self.model_dim)
        self.out_proj = nn.Linear(self.model_dim, self.model_dim)
        self.out_dropout = nn.Dropout(proj_dropout)

    def _shape(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        x = x.view(bsz, seq_len, self.num_heads, self.head_dim)
        return x.transpose(1, 2)

    def _manual_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = torch.softmax(scores, dim=-1, dtype=torch.float32).to(dtype=q.dtype)
        attn = F.dropout(attn, p=self.attn_dropout, training=self.training)
        out = torch.matmul(attn, v)
        if return_attention:
            return out, attn
        return out, None

    def forward(
        self,
        query: torch.Tensor,
        key_value: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ):
        kv = query if key_value is None else key_value
        q = self._shape(self.q_proj(query))
        k = self._shape(self.k_proj(kv))
        v = self._shape(self.v_proj(kv))

        if self.rope is not None and key_value is None:
            cos, sin = self.rope.cos_sin(seq_len=q.size(-2), device=q.device, dtype=q.dtype)
            q, k = _apply_rope(q, k, cos, sin)

        attn_weights = None
        if self.use_sdpa and hasattr(F, "scaled_dot_product_attention") and not return_attention:
            attn_out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.attn_dropout if self.training else 0.0,
                is_causal=False,
            )
        else:
            attn_out, attn_weights = self._manual_attention(
                q,
                k,
                v,
                return_attention=return_attention,
            )

        attn_out = (
            attn_out.transpose(1, 2).contiguous().view(query.size(0), query.size(1), self.model_dim)
        )
        out = self.out_dropout(self.out_proj(attn_out))
        if return_attention:
            return out, attn_weights
        return out


class TransformerBlock(nn.Module):
    """Pre-norm transformer block with GeGLU feed-forward."""

    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        ff_hidden_dim: int,
        attn_dropout: float,
        ff_dropout: float,
        use_sdpa: bool,
        rope: Optional[RotaryEmbedding],
    ) -> None:
        super().__init__()
        self.norm_attn = nn.LayerNorm(model_dim)
        self.self_attn = MultiheadAttentionWithFallback(
            model_dim=model_dim,
            num_heads=num_heads,
            attn_dropout=attn_dropout,
            proj_dropout=ff_dropout,
            use_sdpa=use_sdpa,
            rope=rope,
        )
        self.norm_ff = nn.LayerNorm(model_dim)
        self.ff = GeGLU(model_dim=model_dim, hidden_dim=ff_hidden_dim, dropout=ff_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.norm_attn(x))
        x = x + self.ff(self.norm_ff(x))
        return x


class TaskCrossAttentionBlock(nn.Module):
    """Task-query block using cross-attention over encoded context tokens."""

    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        ff_hidden_dim: int,
        attn_dropout: float,
        ff_dropout: float,
        use_sdpa: bool,
    ) -> None:
        super().__init__()
        self.norm_q = nn.LayerNorm(model_dim)
        self.norm_ctx = nn.LayerNorm(model_dim)
        self.cross_attn = MultiheadAttentionWithFallback(
            model_dim=model_dim,
            num_heads=num_heads,
            attn_dropout=attn_dropout,
            proj_dropout=ff_dropout,
            use_sdpa=use_sdpa,
            rope=None,
        )
        self.norm_ff = nn.LayerNorm(model_dim)
        self.ff = GeGLU(model_dim=model_dim, hidden_dim=ff_hidden_dim, dropout=ff_dropout)

    def forward(
        self,
        task_tokens: torch.Tensor,
        context_tokens: torch.Tensor,
        return_attention: bool = False,
    ):
        if return_attention:
            cross_out, attn_weights = self.cross_attn(
                self.norm_q(task_tokens),
                key_value=self.norm_ctx(context_tokens),
                return_attention=True,
            )
        else:
            cross_out = self.cross_attn(
                self.norm_q(task_tokens), key_value=self.norm_ctx(context_tokens)
            )
            attn_weights = None

        task_tokens = task_tokens + cross_out
        task_tokens = task_tokens + self.ff(self.norm_ff(task_tokens))
        if return_attention:
            return task_tokens, attn_weights
        return task_tokens


class DenseTaskDecoderBlock(nn.Module):
    """Task decoder that replaces cross-attention with pooled-context MLP updates."""

    def __init__(
        self,
        model_dim: int,
        ff_hidden_dim: int,
        ff_dropout: float,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(model_dim)
        self.mlp = nn.Sequential(
            nn.Linear(model_dim, ff_hidden_dim),
            nn.GELU(),
            nn.Dropout(ff_dropout),
            nn.Linear(ff_hidden_dim, model_dim),
            nn.Dropout(ff_dropout),
        )

    def forward(
        self,
        task_tokens: torch.Tensor,
        context_tokens: torch.Tensor,
        return_attention: bool = False,
    ):
        pooled_context = context_tokens.mean(dim=1, keepdim=True)
        task_seed = task_tokens + pooled_context
        task_tokens = task_seed + self.mlp(self.norm(task_seed))
        if return_attention:
            return task_tokens, None
        return task_tokens


class MultiSourceDynamicEncoder(nn.Module):
    """Project `[B, T, K, D]` source features and aggregate across `K`."""

    def __init__(
        self,
        source_input_dim: int,
        model_dim: int,
        aggregation: str = "attention",
        source_geometry_dim: int | None = None,
        use_geometry_features: bool = True,
    ) -> None:
        super().__init__()
        if source_input_dim < 1:
            raise ValueError(f"source_input_dim must be >= 1, got {source_input_dim}")

        self.source_input_dim = int(source_input_dim)
        self.model_dim = int(model_dim)
        self.aggregation = str(aggregation or "attention").strip().lower()
        self.use_geometry_features = bool(use_geometry_features)
        self.source_geometry_dim = None if source_geometry_dim is None else int(source_geometry_dim)
        if self.aggregation not in {"attention", "weighted_pool"}:
            raise ValueError("multi_source aggregation must be one of: attention, weighted_pool")
        if self.use_geometry_features:
            if self.source_geometry_dim is None or self.source_geometry_dim < 1:
                raise ValueError("source_geometry_dim must be >= 1 when use_geometry_features=True")
        elif self.aggregation == "weighted_pool":
            raise ValueError("weighted_pool aggregation requires use_geometry_features=True")

        self.source_proj = nn.Linear(self.source_input_dim, self.model_dim)
        self.source_norm = nn.LayerNorm(self.model_dim)
        self.geometry_proj = (
            nn.Linear(self.source_geometry_dim, self.model_dim)
            if self.use_geometry_features and self.source_geometry_dim is not None
            else None
        )
        self.geometry_norm = (
            nn.LayerNorm(self.model_dim) if self.geometry_proj is not None else None
        )
        self.score_proj = nn.Linear(self.model_dim, 1)

    def forward(
        self,
        x_dynamic_sources: torch.Tensor,
        source_geometry: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if x_dynamic_sources.ndim != 4:
            raise ValueError(
                "x_dynamic_sources must be 4D [B, T, K, D], "
                f"got shape {tuple(x_dynamic_sources.shape)}"
            )

        source_tokens = self.source_norm(self.source_proj(x_dynamic_sources))
        fused = source_tokens
        if self.use_geometry_features:
            if source_geometry is None:
                raise ValueError("source_geometry is required when use_geometry_features=True")
            if source_geometry.ndim != 3:
                raise ValueError(
                    "source_geometry must be 3D [B, K, Dg], "
                    f"got shape {tuple(source_geometry.shape)}"
                )
            if x_dynamic_sources.size(0) != source_geometry.size(0) or x_dynamic_sources.size(
                2
            ) != source_geometry.size(1):
                raise ValueError(
                    "Source dynamic/source geometry shape mismatch: "
                    f"x_dynamic_sources={tuple(x_dynamic_sources.shape)} source_geometry={tuple(source_geometry.shape)}"
                )
            if (
                self.source_geometry_dim is not None
                and source_geometry.size(-1) != self.source_geometry_dim
            ):
                raise ValueError(
                    "Source geometry feature dimension mismatch: "
                    f"expected {self.source_geometry_dim}, got {source_geometry.size(-1)}"
                )
            assert self.geometry_proj is not None
            assert self.geometry_norm is not None
            geometry_tokens = self.geometry_norm(self.geometry_proj(source_geometry)).unsqueeze(1)
            fused = fused + geometry_tokens

        scores = self.score_proj(torch.tanh(fused)).squeeze(-1)
        if self.aggregation == "weighted_pool":
            assert source_geometry is not None
            if source_geometry.size(-1) < 4:
                raise ValueError(
                    "weighted_pool aggregation expects inverse-distance weights in source_geometry"
                )
            weights = torch.softmax(source_geometry[..., 3], dim=-1).unsqueeze(1)
            pooled = torch.sum(fused * weights.unsqueeze(-1), dim=2)
            attention = weights.expand(-1, x_dynamic_sources.size(1), -1).contiguous()
            return pooled, attention

        attention = torch.softmax(scores, dim=-1)
        pooled = torch.sum(fused * attention.unsqueeze(-1), dim=2)
        return pooled, attention


class CoastalConditionedTransformer(nn.Module):
    """Coastal-conditioned transformer with task-decoupled prediction heads."""

    task_order = ("hs", "tp", "dir", "dp")

    def __init__(
        self,
        dynamic_input_dim: int,
        static_input_dim: int,
        output_dim: int,
        dynamic_feature_names: Sequence[str] | None = None,
        source_dynamic_input_dim: int | None = None,
        source_geometry_input_dim: int | None = None,
        model_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        ff_multiplier: float = 4.0,
        attn_dropout: float = 0.1,
        ff_dropout: float = 0.1,
        static_hidden_dims: Sequence[int] | None = None,
        static_dropout: float = 0.1,
        task_dropout: float = 0.1,
        rope_base: float = 10000.0,
        use_sdpa: bool = True,
        decoder_type: str = "cross_attention",
        sequence_encoder_type: str = "transformer",
        lstm_hidden_dim: int = 128,
        lstm_num_layers: int = 2,
        lstm_bidirectional: bool = False,
        lstm_dropout: float = 0.2,
        lstm_pooling: str = "last",
        lstm_layer_norm: bool = True,
        num_tp_bins: int = 32,
        num_dp_bins: int = 36,
        target_mode: str = "physical",
        target_max_tp_delta: float = 15.0,
        transfer_representation: str = "legacy",
        residual_config: dict | None = None,
        hybrid_input_aliases: dict | None = None,
        use_multi_source: bool = False,
        multi_source_aggregation: str = "attention",
        use_source_geometry_features: bool = True,
        source_feature_names: Sequence[str] | None = None,
        use_expert_heads: bool = False,
        expert_combine_hs_energy_space: bool = True,
        expert_combine_direction_circular: bool = True,
        expert_combine_tp: str = "energy_weighted",
        use_static_features: bool = True,
        use_bathymetry: bool = False,
        bathy_in_channels: int = 2,
        bathy_conv_channels: Sequence[int] | None = None,
        bathy_num_tokens: int = 4,
        bathy_dropout2d: float = 0.05,
        bathy_token_dropout: float = 0.10,
        bathy_branch_dropout: float = 0.10,
    ) -> None:
        super().__init__()

        self.dynamic_input_dim = int(dynamic_input_dim)
        self.static_input_dim = int(static_input_dim)
        self.output_dim = int(output_dim)
        self.dynamic_feature_names = [str(name) for name in (dynamic_feature_names or [])]
        self.source_dynamic_input_dim = (
            None if source_dynamic_input_dim is None else int(source_dynamic_input_dim)
        )
        self.source_geometry_input_dim = (
            None if source_geometry_input_dim is None else int(source_geometry_input_dim)
        )
        self.source_feature_names = [str(name) for name in (source_feature_names or [])]
        self.model_dim = int(model_dim)
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.decoder_type = str(decoder_type or "cross_attention").strip().lower()
        self.sequence_encoder_type = str(sequence_encoder_type or "transformer").strip().lower()
        self.num_tp_bins = int(num_tp_bins)
        self.num_dp_bins = int(num_dp_bins)
        self.target_mode = str(target_mode or "physical").strip().lower()
        self.target_max_tp_delta = float(target_max_tp_delta)
        self.transfer_representation = str(transfer_representation or "legacy").strip().lower()
        self.residual_config = dict(residual_config or {})
        self.hybrid_input_aliases = hybrid_input_aliases or {}
        self.use_multi_source = bool(use_multi_source)
        self.multi_source_aggregation = str(multi_source_aggregation or "attention").strip().lower()
        self.use_source_geometry_features = bool(use_source_geometry_features)
        self.use_expert_heads = bool(use_expert_heads)
        self.expert_combine_hs_energy_space = bool(expert_combine_hs_energy_space)
        self.expert_combine_direction_circular = bool(expert_combine_direction_circular)
        self.expert_combine_tp = str(expert_combine_tp or "energy_weighted").strip().lower()
        self.use_static_features = bool(use_static_features)
        self.use_bathymetry = bool(use_bathymetry)
        self.static_branch_hidden_dims = tuple(
            int(v) for v in (static_hidden_dims or (self.model_dim,))
        )
        self.lstm_hidden_dim = int(lstm_hidden_dim)
        self.lstm_num_layers = int(lstm_num_layers)
        self.lstm_bidirectional = bool(lstm_bidirectional)
        self.lstm_dropout = float(lstm_dropout)
        self.lstm_pooling = str(lstm_pooling or "last").strip().lower()
        self.lstm_layer_norm = bool(lstm_layer_norm)

        if self.output_dim not in {4, 6}:
            raise ValueError(
                "CoastalConditionedTransformer expects output_dim=4 for hybrid heads or "
                "output_dim=6 for legacy sin/cos outputs."
            )
        if self.target_mode not in {"physical", "transfer", "physical_and_transfer"}:
            raise ValueError(
                "target_mode must be one of: physical, transfer, physical_and_transfer; "
                f"got '{self.target_mode}'"
            )
        if self.transfer_representation not in {"legacy", "residual_correction"}:
            raise ValueError(
                "transfer_representation must be one of: legacy, residual_correction; "
                f"got '{self.transfer_representation}'"
            )
        if self.target_max_tp_delta <= 0.0:
            raise ValueError(f"target_max_tp_delta must be > 0, got {self.target_max_tp_delta}")
        if self.target_mode != "physical" and self.use_expert_heads:
            raise ValueError("Expert heads are currently supported only for target_mode='physical'")
        if self.sequence_encoder_type not in {"transformer", "lstm"}:
            raise ValueError(
                "sequence_encoder_type must be one of: transformer, lstm; "
                f"got '{self.sequence_encoder_type}'"
            )
        if self.decoder_type not in {"cross_attention", "dense"}:
            raise ValueError(
                f"decoder_type must be one of: cross_attention, dense; got '{self.decoder_type}'"
            )
        if self.model_dim % self.num_heads != 0:
            raise ValueError(
                f"model_dim ({self.model_dim}) must be divisible by num_heads ({self.num_heads})"
            )
        if self.sequence_encoder_type == "transformer" and self.num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {self.num_layers}")
        if self.sequence_encoder_type == "lstm" and self.lstm_num_layers < 1:
            raise ValueError(f"lstm_num_layers must be >= 1, got {self.lstm_num_layers}")
        if self.use_multi_source:
            if self.source_dynamic_input_dim is None or self.source_dynamic_input_dim < 1:
                raise ValueError("source_dynamic_input_dim must be set when use_multi_source=True")
            if self.use_source_geometry_features and (
                self.source_geometry_input_dim is None or self.source_geometry_input_dim < 1
            ):
                raise ValueError("source_geometry_input_dim must be set when use_multi_source=True")
        if any(hidden_dim < 1 for hidden_dim in self.static_branch_hidden_dims):
            raise ValueError(
                "static_hidden_dims must contain only positive integers; "
                f"got {list(self.static_branch_hidden_dims)}"
            )

        ff_hidden_dim = int(max(self.model_dim, round(self.model_dim * float(ff_multiplier))))
        self.dynamic_proj = (
            None if self.use_multi_source else nn.Linear(self.dynamic_input_dim, self.model_dim)
        )
        self.dynamic_norm = None if self.use_multi_source else nn.LayerNorm(self.model_dim)

        alias_cfg = {str(k).strip().lower(): v for k, v in (hybrid_input_aliases or {}).items()}
        self.tp_aliases = _infer_alias_set(
            alias_cfg.get("tp"), default=("tp", "tp_swell", "tm1", "tm2", "tmp")
        )
        self.dp_aliases = _infer_alias_set(
            alias_cfg.get("dp"),
            default=("dir", "dp", "pdir", "thq", "thq_swell", "wind_direction_10m"),
        )

        self.dynamic_embedded_indices: dict[int, str] = {}
        self.dynamic_embeddings = nn.ModuleDict()
        self._continuous_dynamic_indices: list[int] = []
        tp_centers = _build_bin_centers(0.0, 25.0, self.num_tp_bins, circular=False)
        dp_centers = _build_bin_centers(0.0, 360.0, self.num_dp_bins, circular=True)
        self.register_buffer("tp_bin_centers", tp_centers, persistent=False)
        self.register_buffer("dp_bin_centers", dp_centers, persistent=False)
        if not self.use_multi_source:
            self._register_dynamic_embeddings()

        if self.use_multi_source:
            self.multi_source_encoder = MultiSourceDynamicEncoder(
                source_input_dim=int(self.source_dynamic_input_dim),
                model_dim=self.model_dim,
                aggregation=self.multi_source_aggregation,
                source_geometry_dim=self.source_geometry_input_dim,
                use_geometry_features=self.use_source_geometry_features,
            )
        else:
            self.multi_source_encoder = None

        if self.use_static_features:
            if self.static_input_dim < 1:
                raise ValueError("static_input_dim must be >= 1 when use_static_features=True")
            static_layers: list[nn.Module] = [nn.LayerNorm(self.static_input_dim)]
            static_in_dim = self.static_input_dim
            for hidden_dim in self.static_branch_hidden_dims:
                static_layers.extend(
                    [
                        nn.Linear(static_in_dim, int(hidden_dim)),
                        nn.SiLU(),
                        nn.Dropout(static_dropout),
                    ]
                )
                static_in_dim = int(hidden_dim)
            static_layers.extend(
                [
                    nn.Linear(static_in_dim, self.model_dim),
                    nn.LayerNorm(self.model_dim),
                ]
            )
            self.static_encoder = nn.Sequential(*static_layers)
            self.static_branch_config = {
                "hidden_dims": list(self.static_branch_hidden_dims),
                "dropout": float(static_dropout),
            }
        else:
            self.static_encoder = None
            self.static_branch_config = {
                "hidden_dims": [],
                "dropout": float(static_dropout),
            }

        if self.use_bathymetry:
            conv_channels = tuple(int(v) for v in (bathy_conv_channels or (32, 64, 96)))
            self.bathy_encoder = BathymetryCNNEncoder(
                in_channels=int(bathy_in_channels),
                conv_channels=conv_channels,
                model_dim=self.model_dim,
                num_tokens=int(bathy_num_tokens),
                dropout2d=float(bathy_dropout2d),
                token_dropout=float(bathy_token_dropout),
            )
            self.bathy_branch_dropout = BathymetryBranchDropout(float(bathy_branch_dropout))
        else:
            self.bathy_encoder = None
            self.bathy_branch_dropout = None

        self.encoder_blocks = nn.ModuleList()
        if self.sequence_encoder_type == "transformer":
            rope = RotaryEmbedding(head_dim=self.model_dim // self.num_heads, base=float(rope_base))
            self.encoder_blocks = nn.ModuleList(
                [
                    TransformerBlock(
                        model_dim=self.model_dim,
                        num_heads=self.num_heads,
                        ff_hidden_dim=ff_hidden_dim,
                        attn_dropout=attn_dropout,
                        ff_dropout=ff_dropout,
                        use_sdpa=use_sdpa,
                        rope=rope,
                    )
                    for _ in range(self.num_layers)
                ]
            )
            self.sequence_encoder = TransformerSequenceEncoder(self.encoder_blocks)
            self.sequence_encoder_config = {
                "type": "transformer",
                "model_dim": self.model_dim,
                "num_layers": self.num_layers,
                "num_heads": self.num_heads,
                "ff_multiplier": float(ff_multiplier),
                "attn_dropout": float(attn_dropout),
                "ff_dropout": float(ff_dropout),
                "rope_base": float(rope_base),
                "use_sdpa": bool(use_sdpa),
            }
        else:
            self.sequence_encoder = LSTMSequenceEncoder(
                input_dim=self.model_dim,
                model_dim=self.model_dim,
                hidden_dim=self.lstm_hidden_dim,
                num_layers=self.lstm_num_layers,
                dropout=self.lstm_dropout,
                bidirectional=self.lstm_bidirectional,
                pooling=self.lstm_pooling,
                layer_norm=self.lstm_layer_norm,
            )
            self.sequence_encoder_config = {
                "type": "lstm",
                "hidden_dim": self.lstm_hidden_dim,
                "num_layers": self.lstm_num_layers,
                "bidirectional": self.lstm_bidirectional,
                "dropout": self.lstm_dropout,
                "pooling": self.lstm_pooling,
                "layer_norm": self.lstm_layer_norm,
            }

        self.context_norm = nn.LayerNorm(self.model_dim)
        self.task_queries = nn.Parameter(torch.randn(len(self.task_order), self.model_dim) * 0.02)
        if self.decoder_type == "cross_attention":
            self.task_decoder = TaskCrossAttentionBlock(
                model_dim=self.model_dim,
                num_heads=self.num_heads,
                ff_hidden_dim=ff_hidden_dim,
                attn_dropout=attn_dropout,
                ff_dropout=ff_dropout,
                use_sdpa=use_sdpa,
            )
        else:
            self.task_decoder = DenseTaskDecoderBlock(
                model_dim=self.model_dim,
                ff_hidden_dim=ff_hidden_dim,
                ff_dropout=ff_dropout,
            )
        self.decoder_config = {
            "type": self.decoder_type,
            "ff_hidden_dim": ff_hidden_dim,
            "ff_dropout": float(ff_dropout),
            "pooling": "mean" if self.decoder_type == "dense" else None,
            "num_heads": self.num_heads if self.decoder_type == "cross_attention" else None,
            "attn_dropout": float(attn_dropout) if self.decoder_type == "cross_attention" else None,
            "use_sdpa": bool(use_sdpa) if self.decoder_type == "cross_attention" else None,
        }
        self.task_dropout = nn.Dropout(task_dropout)

        self.hs_head = nn.Linear(self.model_dim, 1)
        self.tp_head = nn.Sequential(
            nn.Linear(self.model_dim, self.num_tp_bins), nn.LogSoftmax(dim=-1)
        )
        self.dir_head = nn.Sequential(
            nn.Linear(self.model_dim, self.num_dp_bins), nn.LogSoftmax(dim=-1)
        )
        self.dp_head = nn.Sequential(
            nn.Linear(self.model_dim, self.num_dp_bins), nn.LogSoftmax(dim=-1)
        )
        self.transfer_tp_head = (
            nn.Linear(self.model_dim, 1) if self.target_mode != "physical" else None
        )
        self.transfer_dir_head = (
            nn.Linear(self.model_dim, 2) if self.target_mode != "physical" else None
        )
        self.transfer_dp_head = (
            nn.Linear(self.model_dim, 2) if self.target_mode != "physical" else None
        )
        self.transfer_hs_residual_head = (
            nn.Linear(self.model_dim, 1) if self.target_mode != "physical" else None
        )
        self.transfer_tp_residual_head = (
            nn.Linear(self.model_dim, 1) if self.target_mode != "physical" else None
        )
        self.transfer_dir_residual_head = (
            nn.Linear(self.model_dim, 1) if self.target_mode != "physical" else None
        )
        self.transfer_dp_residual_head = (
            nn.Linear(self.model_dim, 1) if self.target_mode != "physical" else None
        )
        if self.use_expert_heads:
            self.component_names = ("offshore_swell", "offshore_windsea", "local_windsea")
            self.expert_energy_head = nn.Linear(self.model_dim, len(self.component_names))
            self.expert_tp_head = nn.Linear(self.model_dim, len(self.component_names))
            self.expert_dir_head = nn.Linear(self.model_dim, len(self.component_names) * 2)
            self.expert_dp_head = nn.Linear(self.model_dim, len(self.component_names) * 2)
        else:
            self.component_names = ()
            self.expert_energy_head = None
            self.expert_tp_head = None
            self.expert_dir_head = None
            self.expert_dp_head = None

        if (
            self.target_mode != "physical"
            and self.transfer_representation == "residual_correction"
            and bool(self.residual_config.get("zero_init_output_head", True))
        ):
            self._zero_init_residual_heads()

    @staticmethod
    def _signed_angle_from_vector(components: torch.Tensor) -> torch.Tensor:
        return torch.rad2deg(torch.atan2(components[..., 0], components[..., 1]))

    def _zero_init_residual_heads(self) -> None:
        heads = (
            self.transfer_hs_residual_head,
            self.transfer_tp_residual_head,
            self.transfer_dir_residual_head,
            self.transfer_dp_residual_head,
        )
        for head in heads:
            if head is None:
                continue
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def _apply_residual_bound(self, values: torch.Tensor, key: str) -> torch.Tensor:
        method = str(self.residual_config.get("bound_method", "none")).strip().lower()
        if method == "none":
            return values

        bound_lookup = {
            "log_hs_ratio": float(self.residual_config.get("max_abs_log_hs", 1.25)),
            "tp_delta": float(self.residual_config.get("max_abs_tp", self.target_max_tp_delta)),
            "dir_delta_deg": float(self.residual_config.get("max_abs_dir_deg", 120.0)),
            "dp_delta_deg": float(self.residual_config.get("max_abs_dp_deg", 120.0)),
        }
        max_abs = bound_lookup[str(key)]
        if method == "tanh":
            return max_abs * torch.tanh(values)
        if method == "clamp":
            return torch.clamp(values, min=-max_abs, max=max_abs)
        raise ValueError(f"Unsupported residual bound method '{method}'")

    def _register_dynamic_embeddings(self) -> None:
        if (
            not self.dynamic_feature_names
            or len(self.dynamic_feature_names) != self.dynamic_input_dim
        ):
            self._continuous_dynamic_indices = list(range(self.dynamic_input_dim))
            return

        for idx, feature_name in enumerate(self.dynamic_feature_names):
            lower = str(feature_name).strip().lower()
            if _feature_matches_alias(lower, self.tp_aliases):
                self.dynamic_embedded_indices[idx] = "tp"
                self.dynamic_embeddings[str(idx)] = nn.Embedding(self.num_tp_bins, self.model_dim)
            elif _feature_matches_alias(lower, self.dp_aliases):
                self.dynamic_embedded_indices[idx] = "dp"
                self.dynamic_embeddings[str(idx)] = nn.Embedding(self.num_dp_bins, self.model_dim)
            else:
                self._continuous_dynamic_indices.append(idx)

        if not self._continuous_dynamic_indices:
            self._continuous_dynamic_indices = []

    def _quantize_feature(
        self, values: torch.Tensor, centers: torch.Tensor, circular: bool
    ) -> torch.Tensor:
        centers = centers.to(device=values.device, dtype=values.dtype)
        if circular:
            diff = torch.abs(values.unsqueeze(-1) - centers.view(1, 1, -1))
            diff = torch.minimum(diff, 360.0 - diff)
        else:
            diff = torch.abs(values.unsqueeze(-1) - centers.view(1, 1, -1))
        return torch.argmin(diff, dim=-1)

    def _encode_dynamic_tokens(self, x_dynamic: torch.Tensor) -> torch.Tensor:
        if not self.dynamic_embedded_indices:
            return self.dynamic_norm(self.dynamic_proj(x_dynamic))

        masked_x = x_dynamic.clone()
        embedded_cols = list(self.dynamic_embedded_indices.keys())
        if embedded_cols:
            masked_x[..., embedded_cols] = 0.0
        dynamic_tokens = self.dynamic_proj(masked_x)

        for idx, mode in self.dynamic_embedded_indices.items():
            feature_values = x_dynamic[..., idx]
            if mode == "tp":
                token_ids = self._quantize_feature(
                    feature_values, self.tp_bin_centers, circular=False
                )
            else:
                token_ids = self._quantize_feature(
                    feature_values, self.dp_bin_centers, circular=True
                )
            embedded = self.dynamic_embeddings[str(idx)](token_ids)
            dynamic_tokens = dynamic_tokens + embedded

        return self.dynamic_norm(dynamic_tokens)

    def _recover_distribution(
        self, log_probs: torch.Tensor, centers: torch.Tensor, circular: bool = False
    ) -> torch.Tensor:
        probs = torch.exp(log_probs)
        centers = centers.to(device=probs.device, dtype=probs.dtype)
        if circular:
            angles = torch.deg2rad(centers)
            sin_mean = torch.sum(probs * torch.sin(angles), dim=-1)
            cos_mean = torch.sum(probs * torch.cos(angles), dim=-1)
            angle = torch.rad2deg(torch.atan2(sin_mean, cos_mean)) % 360.0
            return angle
        return torch.sum(probs * centers, dim=-1)

    def _distribution_from_value(
        self,
        values: torch.Tensor,
        centers: torch.Tensor,
        circular: bool = False,
        sigma: float = 1.0,
    ) -> torch.Tensor:
        centers = centers.to(device=values.device, dtype=values.dtype)
        diff = torch.abs(values.unsqueeze(-1) - centers.view(1, -1))
        if circular:
            diff = torch.minimum(diff, 360.0 - diff)
        logits = -0.5 * torch.square(diff / max(float(sigma), 1e-6))
        return torch.log_softmax(logits, dim=-1)

    def _combine_component_angles(
        self, components: torch.Tensor, weights: torch.Tensor
    ) -> torch.Tensor:
        weights = weights / torch.clamp(weights.sum(dim=-1, keepdim=True), min=1e-6)
        sin_mean = torch.sum(weights * torch.sin(torch.deg2rad(components)), dim=-1)
        cos_mean = torch.sum(weights * torch.cos(torch.deg2rad(components)), dim=-1)
        return torch.rad2deg(torch.atan2(sin_mean, cos_mean)) % 360.0

    def _build_expert_outputs(
        self, task_tokens: torch.Tensor
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        if not self.use_expert_heads:
            raise RuntimeError("Expert outputs requested while expert heads are disabled")
        assert self.expert_energy_head is not None
        assert self.expert_tp_head is not None
        assert self.expert_dir_head is not None
        assert self.expert_dp_head is not None

        energies = F.softplus(self.expert_energy_head(task_tokens[:, 0, :])) + 1e-6
        tp_components = F.softplus(self.expert_tp_head(task_tokens[:, 1, :]))
        dir_components = self.expert_dir_head(task_tokens[:, 2, :]).view(
            task_tokens.size(0), len(self.component_names), 2
        )
        dp_components = self.expert_dp_head(task_tokens[:, 3, :]).view(
            task_tokens.size(0), len(self.component_names), 2
        )

        dir_angles = (
            torch.rad2deg(torch.atan2(dir_components[..., 0], dir_components[..., 1])) % 360.0
        )
        dp_angles = torch.rad2deg(torch.atan2(dp_components[..., 0], dp_components[..., 1])) % 360.0

        total_energy = energies.sum(dim=-1)
        hs_pred = (
            torch.sqrt(torch.clamp(total_energy, min=1e-8))
            if self.expert_combine_hs_energy_space
            else total_energy
        )
        energy_weights = energies / torch.clamp(total_energy.unsqueeze(-1), min=1e-6)

        if self.expert_combine_tp == "dominance":
            dominant_idx = torch.argmax(energies, dim=-1)
            tp_pred = tp_components.gather(1, dominant_idx.unsqueeze(-1)).squeeze(-1)
        else:
            tp_pred = torch.sum(energy_weights * tp_components, dim=-1)

        if self.expert_combine_direction_circular:
            dir_pred = self._combine_component_angles(dir_angles, energies)
            dp_pred = self._combine_component_angles(dp_angles, energies)
        else:
            dir_pred = torch.sum(energy_weights * dir_angles, dim=-1) % 360.0
            dp_pred = torch.sum(energy_weights * dp_angles, dim=-1) % 360.0

        tp_log_probs = self._distribution_from_value(
            tp_pred, self.tp_bin_centers, circular=False, sigma=0.8
        )
        dir_log_probs = self._distribution_from_value(
            dir_pred, self.dp_bin_centers, circular=True, sigma=10.0
        )
        dp_log_probs = self._distribution_from_value(
            dp_pred, self.dp_bin_centers, circular=True, sigma=10.0
        )

        diagnostics = {
            "component_energies": energies,
            "component_energy_fractions": energy_weights,
            "component_tp": tp_components,
            "component_dir_deg": dir_angles,
            "component_dp_deg": dp_angles,
        }
        return {
            "hs": hs_pred,
            "tp_log_probs": tp_log_probs,
            "dir_log_probs": dir_log_probs,
            "dp_log_probs": dp_log_probs,
            "tp_pred": tp_pred,
            "dir_pred": dir_pred,
            "dp_pred": dp_pred,
            "expert_diagnostics": diagnostics,
        }

    def forward(
        self,
        x_dynamic: torch.Tensor | None,
        x_static: torch.Tensor | None = None,
        x_bathy: torch.Tensor | None = None,
        x_dynamic_sources: torch.Tensor | None = None,
        source_geometry: torch.Tensor | None = None,
        return_attention: bool = False,
        return_diagnostics: bool = False,
    ):
        if x_static is not None and x_static.ndim != 2:
            raise ValueError(f"x_static must be 2D [B, S], got shape {tuple(x_static.shape)}")
        if self.use_static_features and x_static is None:
            raise ValueError("Static branch enabled but x_static was not provided")
        if (
            self.use_static_features
            and x_static is not None
            and x_static.size(-1) != self.static_input_dim
        ):
            raise ValueError(
                "Static feature dimension mismatch: "
                f"expected {self.static_input_dim}, got {x_static.size(-1)}"
            )
        source_attention_weights = None
        sequence_input: torch.Tensor
        if self.use_multi_source:
            if x_dynamic_sources is None:
                raise ValueError("Multi-source mode requires x_dynamic_sources [B, T, K, D]")
            if self.use_source_geometry_features and source_geometry is None:
                raise ValueError(
                    "Multi-source mode with geometry enabled requires source_geometry [B, K, Dg]"
                )
            if x_dynamic_sources.ndim != 4:
                raise ValueError(
                    "x_dynamic_sources must be 4D [B, T, K, D], "
                    f"got shape {tuple(x_dynamic_sources.shape)}"
                )
            if (
                self.use_source_geometry_features
                and source_geometry is not None
                and source_geometry.ndim != 3
            ):
                raise ValueError(
                    "source_geometry must be 3D [B, K, Dg], "
                    f"got shape {tuple(source_geometry.shape)}"
                )
            if x_dynamic_sources.size(-1) != int(self.source_dynamic_input_dim):
                raise ValueError(
                    "Source dynamic feature dimension mismatch: "
                    f"expected {self.source_dynamic_input_dim}, got {x_dynamic_sources.size(-1)}"
                )
            if (
                self.use_source_geometry_features
                and source_geometry is not None
                and source_geometry.size(-1) != int(self.source_geometry_input_dim)
            ):
                raise ValueError(
                    "Source geometry feature dimension mismatch: "
                    f"expected {self.source_geometry_input_dim}, got {source_geometry.size(-1)}"
                )
            assert self.multi_source_encoder is not None
            sequence_input, source_attention_weights = self.multi_source_encoder(
                x_dynamic_sources, source_geometry
            )
            batch_size = int(x_dynamic_sources.size(0))
        else:
            if x_dynamic is None:
                raise ValueError("Single-source mode requires x_dynamic [B, T, D]")
            if x_dynamic.ndim != 3:
                raise ValueError(
                    f"x_dynamic must be 3D [B, T, D], got shape {tuple(x_dynamic.shape)}"
                )
            if x_dynamic.size(-1) != self.dynamic_input_dim:
                raise ValueError(
                    "Dynamic feature dimension mismatch: "
                    f"expected {self.dynamic_input_dim}, got {x_dynamic.size(-1)}"
                )
            sequence_input = self._encode_dynamic_tokens(x_dynamic)
            batch_size = int(x_dynamic.size(0))

        if x_static is not None and x_static.size(0) != batch_size:
            raise ValueError(
                "Batch mismatch between dynamic sequence input and static branches: "
                f"{batch_size} vs {x_static.size(0)}"
            )

        if self.use_bathymetry:
            if x_bathy is None:
                raise ValueError("Bathymetry branch enabled but x_bathy was not provided")
            if x_bathy.ndim != 4:
                raise ValueError(
                    f"x_bathy must be 4D [B, C, H, W], got shape {tuple(x_bathy.shape)}"
                )
            if x_bathy.size(0) != batch_size:
                raise ValueError(
                    "Batch mismatch between dynamic and bathymetry branches: "
                    f"{batch_size} vs {x_bathy.size(0)}"
                )

        dynamic_tokens = self.sequence_encoder(sequence_input)

        context_parts = [dynamic_tokens]
        task_bias = torch.zeros(
            (batch_size, len(self.task_order), self.model_dim),
            dtype=dynamic_tokens.dtype,
            device=dynamic_tokens.device,
        )
        static_token = None
        bathy_tokens = None
        bathy_summary = None
        context_token_types: list[str] = ["dynamic"] * int(dynamic_tokens.size(1))

        if self.use_static_features:
            assert x_static is not None
            assert self.static_encoder is not None
            static_token = self.static_encoder(x_static).unsqueeze(1)
            context_parts.append(static_token)
            task_bias = task_bias + static_token.expand(-1, len(self.task_order), -1)
            context_token_types.append("static")

        if self.use_bathymetry:
            assert self.bathy_encoder is not None
            bathy_tokens, bathy_summary = self.bathy_encoder(x_bathy)
            if self.bathy_branch_dropout is not None:
                bathy_tokens, bathy_summary = self.bathy_branch_dropout(bathy_tokens, bathy_summary)
            context_parts.append(bathy_tokens)
            task_bias = task_bias + bathy_summary.unsqueeze(1)
            context_token_types.extend(["bathy"] * int(bathy_tokens.size(1)))

        context_tokens = self.context_norm(torch.cat(context_parts, dim=1))
        task_tokens = self.task_queries.unsqueeze(0).expand(batch_size, -1, -1) + task_bias

        if return_attention:
            task_tokens, cross_attention_weights = self.task_decoder(
                task_tokens,
                context_tokens,
                return_attention=True,
            )
        else:
            task_tokens = self.task_decoder(task_tokens, context_tokens)
            cross_attention_weights = None

        task_tokens = self.task_dropout(task_tokens)

        if self.target_mode != "physical":
            assert self.transfer_tp_head is not None
            assert self.transfer_dir_head is not None
            assert self.transfer_dp_head is not None
            if self.transfer_representation == "residual_correction":
                assert self.transfer_hs_residual_head is not None
                assert self.transfer_tp_residual_head is not None
                assert self.transfer_dir_residual_head is not None
                assert self.transfer_dp_residual_head is not None

                raw_log_hs_ratio = self.transfer_hs_residual_head(task_tokens[:, 0, :]).squeeze(-1)
                raw_tp_delta = self.transfer_tp_residual_head(task_tokens[:, 1, :]).squeeze(-1)
                raw_dir_delta = self.transfer_dir_residual_head(task_tokens[:, 2, :]).squeeze(-1)
                raw_dp_delta = self.transfer_dp_residual_head(task_tokens[:, 3, :]).squeeze(-1)
                out = {
                    "log_hs_ratio": self._apply_residual_bound(raw_log_hs_ratio, "log_hs_ratio"),
                    "tp_delta": self._apply_residual_bound(raw_tp_delta, "tp_delta"),
                    "dir_delta_deg": self._apply_residual_bound(raw_dir_delta, "dir_delta_deg"),
                    "dp_delta_deg": self._apply_residual_bound(raw_dp_delta, "dp_delta_deg"),
                    "raw_log_hs_ratio": raw_log_hs_ratio,
                    "raw_tp_delta": raw_tp_delta,
                    "raw_dir_delta_deg": raw_dir_delta,
                    "raw_dp_delta_deg": raw_dp_delta,
                }
            else:
                log_hs_ratio = self.hs_head(task_tokens[:, 0, :]).squeeze(-1)
                raw_tp_delta = self.transfer_tp_head(task_tokens[:, 1, :]).squeeze(-1)
                tp_delta = self.target_max_tp_delta * torch.tanh(raw_tp_delta)
                dir_delta_components = self.transfer_dir_head(task_tokens[:, 2, :])
                dp_delta_components = self.transfer_dp_head(task_tokens[:, 3, :])
                out = {
                    "log_hs_ratio": log_hs_ratio,
                    "raw_tp_delta": raw_tp_delta,
                    "tp_delta": tp_delta,
                    "dir_delta_deg": self._signed_angle_from_vector(dir_delta_components),
                    "dp_delta_deg": self._signed_angle_from_vector(dp_delta_components),
                }
        elif self.use_expert_heads:
            out = self._build_expert_outputs(task_tokens)
        else:
            hs = self.hs_head(task_tokens[:, 0, :]).squeeze(-1)
            tp = self.tp_head(task_tokens[:, 1, :])
            direction = self.dir_head(task_tokens[:, 2, :])
            dp = self.dp_head(task_tokens[:, 3, :])
            out = {
                "hs": hs,
                "tp_log_probs": tp,
                "dir_log_probs": direction,
                "dp_log_probs": dp,
                "tp_pred": self._recover_distribution(tp, self.tp_bin_centers, circular=False),
                "dir_pred": self._recover_distribution(
                    direction, self.dp_bin_centers, circular=True
                ),
                "dp_pred": self._recover_distribution(dp, self.dp_bin_centers, circular=True),
            }
        if return_attention:
            out["cross_attention_weights"] = cross_attention_weights
            if source_attention_weights is not None:
                out["source_attention_weights"] = source_attention_weights
        if return_diagnostics:
            out["diagnostics"] = {
                "dynamic_tokens": dynamic_tokens,
                "context_tokens": context_tokens,
                "context_token_types": tuple(context_token_types),
                "task_tokens": task_tokens,
                "static_token": static_token,
                "bathy_tokens": bathy_tokens,
                "bathy_summary": bathy_summary,
            }
        if return_attention:
            return out
        if source_attention_weights is not None:
            out["source_attention_weights"] = source_attention_weights
        return out
