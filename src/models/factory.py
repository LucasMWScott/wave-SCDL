"""Create the configured coastal-conditioned neural network.

``build_model_from_config`` is the single construction point used by training,
evaluation, inference, and tests.  It translates the declarative model/data
configuration and observed feature dimensions into a fully wired PyTorch
module, including optional static, multi-source, and bathymetry branches.
"""

from __future__ import annotations

import warnings

from torch import nn

from .coastal_transformer import CoastalConditionedTransformer

try:
    from config_resolution import resolve_config
    from preprocessing.transfer_targets import resolve_targets_config
except Exception:
    from src.config_resolution import resolve_config
    from src.preprocessing.transfer_targets import resolve_targets_config


_COASTAL_ARCH_ALIASES = {
    "coastal_transformer",
    "coastal-conditioned-transformer",
    "coastal_conditioned_transformer",
    "coastal_transformer_v1",
}


def _resolve_use_source_geometry_features(config: dict) -> bool:
    data_cfg = config.get("data", {}) or {}
    model_cfg = config.get("model", {}) or {}
    coastal_cfg = model_cfg.get("coastal_transformer", {}) or {}
    multi_source_cfg = coastal_cfg.get("multi_source", {}) or {}

    if "use_geometry_features" in multi_source_cfg:
        return bool(multi_source_cfg.get("use_geometry_features", False))
    if "use_geometry" in data_cfg:
        return bool(data_cfg.get("use_geometry", False))
    return False


def build_model_from_config(
    config: dict,
    dynamic_input_dim: int,
    static_input_dim: int,
    output_dim: int,
    dynamic_feature_names: list[str] | None = None,
    source_dynamic_input_dim: int | None = None,
    source_geometry_input_dim: int | None = None,
    source_feature_names: list[str] | None = None,
) -> nn.Module:
    """Instantiate a coastal transformer model from config.

    Legacy encoder families have been removed and are rejected
    explicitly to avoid silent fallback behavior.
    """
    config = resolve_config(config)
    data_cfg = config.get("data", {}) or {}
    model_cfg = config.get("model", {}) or {}
    architecture = str(model_cfg.get("architecture", "coastal_transformer")).strip().lower()

    if architecture not in _COASTAL_ARCH_ALIASES:
        raise ValueError(
            "Unsupported architecture: "
            f"{architecture}. Supported: coastal_transformer "
            "(aliases: coastal-conditioned-transformer, "
            "coastal_conditioned_transformer, coastal_transformer_v1). "
            "Legacy architectures were removed. "
            "Update model.architecture in configs/training.yaml."
        )

    coastal_cfg = model_cfg.get("coastal_transformer", {}) or {}
    bathy_cfg = coastal_cfg.get("bathy", {}) or {}
    multi_source_cfg = coastal_cfg.get("multi_source", {}) or {}
    experts_cfg = coastal_cfg.get("experts", {}) or {}
    static_branch_cfg = coastal_cfg.get("static_branch", {}) or {}
    decoder_cfg = coastal_cfg.get("decoder", {}) or {}
    sequence_cfg = coastal_cfg.get("sequence_encoder", {}) or {}
    sequence_transformer_cfg = sequence_cfg.get("transformer", {}) or {}
    sequence_lstm_cfg = sequence_cfg.get("lstm", {}) or {}
    targets_cfg = resolve_targets_config(data_cfg)
    use_static_features = (
        bool(coastal_cfg.get("use_static"))
        if "use_static" in coastal_cfg
        else bool(data_cfg.get("use_static_features", True))
    )
    shared_dropout = float(coastal_cfg.get("dropout", model_cfg.get("dropout", 0.1)))
    sequence_encoder_type = str(sequence_cfg.get("type", "transformer")).strip().lower()
    model_dim = int(
        sequence_cfg.get(
            "model_dim",
            sequence_transformer_cfg.get(
                "model_dim",
                coastal_cfg.get("model_dim", model_cfg.get("hidden_dim", 256)),
            ),
        )
    )
    transformer_num_layers = int(
        sequence_transformer_cfg.get(
            "num_layers", coastal_cfg.get("num_layers", model_cfg.get("num_layers", 4))
        )
    )
    transformer_num_heads = int(
        sequence_transformer_cfg.get("num_heads", coastal_cfg.get("num_heads", 8))
    )
    transformer_ff_multiplier = float(
        sequence_transformer_cfg.get("ff_multiplier", coastal_cfg.get("ff_multiplier", 4.0))
    )
    transformer_attn_dropout = float(
        sequence_transformer_cfg.get(
            "attn_dropout", coastal_cfg.get("attn_dropout", shared_dropout)
        )
    )
    transformer_ff_dropout = float(
        sequence_transformer_cfg.get("ff_dropout", coastal_cfg.get("ff_dropout", shared_dropout))
    )
    transformer_rope_base = float(
        sequence_transformer_cfg.get("rope_base", coastal_cfg.get("rope_base", 10000.0))
    )
    transformer_use_sdpa = bool(
        sequence_transformer_cfg.get("use_sdpa", coastal_cfg.get("use_sdpa", True))
    )
    static_hidden_dims_raw = static_branch_cfg.get("hidden_dims", None)
    if static_hidden_dims_raw is None:
        static_hidden_dims = [model_dim]
    elif isinstance(static_hidden_dims_raw, (list, tuple)):
        static_hidden_dims = [int(v) for v in static_hidden_dims_raw]
    else:
        static_hidden_dims = [int(static_hidden_dims_raw)]
    static_dropout = float(
        static_branch_cfg.get("dropout", coastal_cfg.get("static_dropout", shared_dropout))
    )
    use_bathymetry = bool(data_cfg.get("use_bathymetry", False)) and bool(
        bathy_cfg.get("enabled", True)
    )
    data_multi_source_enabled = bool((data_cfg.get("multi_source", {}) or {}).get("enabled", False))
    model_multi_source_enabled = bool(multi_source_cfg.get("enabled", True))
    use_source_geometry_features = _resolve_use_source_geometry_features(config)
    source_dims_available = (
        source_dynamic_input_dim is not None
        and int(source_dynamic_input_dim) > 0
        and (
            not use_source_geometry_features
            or (source_geometry_input_dim is not None and int(source_geometry_input_dim) > 0)
        )
    )
    use_multi_source = bool(
        data_multi_source_enabled and (model_multi_source_enabled or source_dims_available)
    )

    if data_multi_source_enabled and not model_multi_source_enabled and source_dims_available:
        warnings.warn(
            "Auto-enabling the model multi-source path because data.multi_source.enabled=true "
            "and multi-source tensors are present, even though "
            "model.coastal_transformer.multi_source.enabled=false. "
            "Set both flags to true in the config to keep the configuration explicit.",
            stacklevel=2,
        )

    return CoastalConditionedTransformer(
        dynamic_input_dim=int(dynamic_input_dim),
        static_input_dim=int(static_input_dim),
        output_dim=int(output_dim),
        dynamic_feature_names=dynamic_feature_names,
        source_dynamic_input_dim=source_dynamic_input_dim,
        source_geometry_input_dim=source_geometry_input_dim,
        model_dim=model_dim,
        num_layers=transformer_num_layers,
        num_heads=transformer_num_heads,
        ff_multiplier=transformer_ff_multiplier,
        attn_dropout=transformer_attn_dropout,
        ff_dropout=transformer_ff_dropout,
        static_hidden_dims=static_hidden_dims,
        static_dropout=static_dropout,
        task_dropout=float(coastal_cfg.get("task_dropout", shared_dropout)),
        rope_base=transformer_rope_base,
        use_sdpa=transformer_use_sdpa,
        decoder_type=str(decoder_cfg.get("type", "cross_attention")),
        sequence_encoder_type=sequence_encoder_type,
        lstm_hidden_dim=int(sequence_lstm_cfg.get("hidden_dim", 128)),
        lstm_num_layers=int(sequence_lstm_cfg.get("num_layers", 2)),
        lstm_bidirectional=bool(sequence_lstm_cfg.get("bidirectional", False)),
        lstm_dropout=float(sequence_lstm_cfg.get("dropout", 0.2)),
        lstm_pooling=str(sequence_lstm_cfg.get("pooling", "last")),
        lstm_layer_norm=bool(sequence_lstm_cfg.get("layer_norm", True)),
        num_tp_bins=int(coastal_cfg.get("num_tp_bins", 32)),
        num_dp_bins=int(coastal_cfg.get("num_dp_bins", 36)),
        target_mode=str(targets_cfg.get("mode", "physical")),
        target_max_tp_delta=float(targets_cfg.get("max_tp_delta", 15.0)),
        transfer_representation=str(targets_cfg.get("transfer_representation", "legacy")),
        residual_config=targets_cfg.get("residual_correction", {}) or {},
        hybrid_input_aliases=coastal_cfg.get("hybrid_input_aliases", {}),
        use_multi_source=use_multi_source,
        multi_source_aggregation=str(multi_source_cfg.get("aggregation", "attention")),
        use_source_geometry_features=use_source_geometry_features,
        source_feature_names=source_feature_names,
        use_expert_heads=bool(experts_cfg.get("enabled", False)),
        expert_combine_hs_energy_space=bool(experts_cfg.get("combine_hs_energy_space", True)),
        expert_combine_direction_circular=bool(experts_cfg.get("combine_direction_circular", True)),
        expert_combine_tp=str(experts_cfg.get("combine_tp", "energy_weighted")),
        use_static_features=use_static_features,
        use_bathymetry=use_bathymetry,
        bathy_in_channels=int(bathy_cfg.get("in_channels", 2)),
        bathy_conv_channels=list(bathy_cfg.get("conv_channels", [32, 64, 96])),
        bathy_num_tokens=int(bathy_cfg.get("num_tokens", 4)),
        bathy_dropout2d=float(bathy_cfg.get("dropout2d", 0.05)),
        bathy_token_dropout=float(bathy_cfg.get("token_dropout", 0.10)),
        bathy_branch_dropout=float(bathy_cfg.get("branch_dropout", 0.10)),
    )


__all__ = ["build_model_from_config"]
