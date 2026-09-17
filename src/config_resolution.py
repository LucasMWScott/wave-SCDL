"""Configuration resolution helpers for canonical training/runtime settings."""

from __future__ import annotations

import copy
from typing import Any, Mapping


def _deepcopy_dict(config: Mapping[str, Any] | None) -> dict[str, Any]:
    return copy.deepcopy(dict(config or {}))


def _ensure_dict(parent: dict[str, Any], key: str) -> dict[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict):
        value = {}
        parent[key] = value
    return value


def get_nested(mapping: Mapping[str, Any], path: str, default: Any = None) -> Any:
    cursor: Any = mapping
    for part in [piece for piece in str(path).split(".") if piece]:
        if not isinstance(cursor, Mapping) or part not in cursor:
            return default
        cursor = cursor[part]
    return cursor


def set_nested(mapping: dict[str, Any], path: str, value: Any) -> None:
    parts = [piece for piece in str(path).split(".") if piece]
    if not parts:
        raise ValueError("Config path must not be empty")

    cursor = mapping
    for part in parts[:-1]:
        cursor = _ensure_dict(cursor, part)
    cursor[parts[-1]] = copy.deepcopy(value)


def resolve_config(config: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return a deep-copied, fully populated runtime configuration."""
    resolved = _deepcopy_dict(config)
    resolution_meta: dict[str, Any] = {}

    data_cfg = _ensure_dict(resolved, "data")
    train_cfg = _ensure_dict(resolved, "training")
    model_cfg = _ensure_dict(resolved, "model")
    coastal_cfg = _ensure_dict(model_cfg, "coastal_transformer")
    sequence_cfg = _ensure_dict(coastal_cfg, "sequence_encoder")
    transformer_cfg = _ensure_dict(sequence_cfg, "transformer")
    split_cfg = _ensure_dict(resolved, "split")
    scheduler_cfg = _ensure_dict(train_cfg, "scheduler")
    optimizer_cfg = _ensure_dict(train_cfg, "optimizer")

    cosine_min_lr = get_nested(resolved, "training.scheduler.cosine.min_lr", default=None)
    plateau_min_lr = get_nested(resolved, "training.scheduler.plateau.min_lr", default=None)
    if plateau_min_lr is None and cosine_min_lr is not None:
        set_nested(resolved, "training.scheduler.plateau.min_lr", cosine_min_lr)

    # Default target representation keys.
    targets_cfg = _ensure_dict(data_cfg, "targets")
    targets_cfg.setdefault("transfer_representation", "legacy")
    residual_cfg = _ensure_dict(targets_cfg, "residual_correction")
    residual_cfg.setdefault("enabled", False)
    residual_cfg.setdefault("hs_form", "log_ratio")
    residual_cfg.setdefault("tp_form", "additive")
    residual_cfg.setdefault("direction_form", "circular_additive")
    residual_cfg.setdefault("dp_form", "circular_additive")
    residual_cfg.setdefault("eps_hs", float(targets_cfg.get("eps", 1e-3)))
    residual_cfg.setdefault("bound_method", "tanh")
    residual_cfg.setdefault("max_abs_log_hs", 1.25)
    residual_cfg.setdefault("max_abs_tp", float(targets_cfg.get("max_tp_delta", 15.0)))
    residual_cfg.setdefault("max_abs_dir_deg", 120.0)
    residual_cfg.setdefault("max_abs_dp_deg", 120.0)
    residual_cfg.setdefault("zero_init_output_head", True)

    static_reg_cfg = _ensure_dict(data_cfg, "static_regularization")
    static_reg_cfg.setdefault("enabled", False)
    noise_cfg = _ensure_dict(static_reg_cfg, "noise")
    noise_cfg.setdefault("enabled", False)
    noise_cfg.setdefault("std", 0.0)
    noise_cfg.setdefault("apply_after_standardization", True)
    noise_cfg.setdefault("train_only", True)
    group_dropout_cfg = _ensure_dict(static_reg_cfg, "group_dropout")
    group_dropout_cfg.setdefault("enabled", False)
    group_dropout_cfg.setdefault("p", 0.0)
    group_dropout_cfg.setdefault("train_only", True)
    group_dropout_cfg.setdefault("mode", "per_sample")
    group_dropout_cfg.setdefault("replacement_value", 0.0)
    group_dropout_cfg.setdefault("groups", {})

    train_sample_subsampling_cfg = _ensure_dict(data_cfg, "train_sample_subsampling")
    train_sample_subsampling_cfg.setdefault("enabled", False)
    train_sample_subsampling_cfg.setdefault("fraction", 1.0)
    train_sample_subsampling_cfg.setdefault("seed", 42)
    train_sample_subsampling_cfg.setdefault("mode", "per_site")
    train_sample_subsampling_cfg.setdefault("validate_sequence_continuity", False)
    train_sample_subsampling_cfg.setdefault("validation_samples", 1000)

    train_site_subsampling_cfg = _ensure_dict(data_cfg, "train_site_subsampling")
    train_site_subsampling_cfg.setdefault("enabled", False)
    train_site_subsampling_cfg.setdefault("fraction", 1.0)
    train_site_subsampling_cfg.setdefault("seed", 42)
    split_cfg.setdefault("site_holdout_temporal_mode", "legacy")

    sampler_cfg = _ensure_dict(train_cfg, "sampler")
    sampler_cfg.setdefault("enabled", False)
    sampler_cfg.setdefault("strategy", "hs_squared")
    train_cfg.setdefault("validation_every_n_epochs", 1)
    train_cfg.setdefault("target_modeling", "joint")
    initialization_cfg = _ensure_dict(train_cfg, "initialization")
    initialization_cfg.setdefault("checkpoint_path", None)
    initialization_cfg.setdefault("strict", False)
    initialization_cfg.setdefault("ignore_optimizer", True)
    site_balanced_cfg = _ensure_dict(sampler_cfg, "site_balanced")
    site_balanced_cfg.setdefault("replacement", True)
    site_balanced_cfg.setdefault("train_only", True)
    site_balanced_cfg.setdefault("min_samples_per_site", 1)
    site_balanced_cfg.setdefault("combine_with_hs_weight", False)
    site_balanced_cfg.setdefault("hs_power", 2.0)
    site_balanced_cfg.setdefault("normalize_within_site", True)

    resolution_meta["resolved"] = {
        "batch_size": int(train_cfg.get("batch_size", 64)),
        "lr": float(optimizer_cfg.get("lr", 1e-3)),
        "weight_decay": float(optimizer_cfg.get("weight_decay", 0.0)),
        "scheduler_type": str(scheduler_cfg.get("type", "step")).strip().lower(),
        "scheduler": copy.deepcopy(scheduler_cfg),
        "model_dim": int(transformer_cfg.get("model_dim", 256)),
        "num_layers": int(transformer_cfg.get("num_layers", 4)),
        "num_heads": int(transformer_cfg.get("num_heads", 8)),
        "ff_multiplier": float(transformer_cfg.get("ff_multiplier", 4.0)),
        "target_mode": str(targets_cfg.get("mode", "physical")).strip().lower(),
        "transfer_reference": str(targets_cfg.get("transfer_reference", "nearest_bulk"))
        .strip()
        .lower(),
        "transfer_representation": str(targets_cfg.get("transfer_representation", "legacy"))
        .strip()
        .lower(),
        "static_regularization": copy.deepcopy(static_reg_cfg),
        "train_sample_subsampling": copy.deepcopy(train_sample_subsampling_cfg),
        "train_site_subsampling": copy.deepcopy(train_site_subsampling_cfg),
        "sampler": copy.deepcopy(sampler_cfg),
        "target_modeling": str(train_cfg.get("target_modeling", "joint")).strip().lower(),
        "initialization": copy.deepcopy(initialization_cfg),
    }

    resolved["_config_resolution"] = resolution_meta
    return resolved


__all__ = ["get_nested", "resolve_config", "set_nested"]
