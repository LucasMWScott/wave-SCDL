"""Helpers for opt-in independent per-target training and composite inference."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Callable, Mapping

import torch
from torch import nn

try:
    from config_resolution import resolve_config
except Exception:
    from src.config_resolution import resolve_config


TARGET_NAMES: tuple[str, str, str, str] = ("hs", "tp", "dir", "dp")
VALID_TARGET_MODELING = {"joint", "independent_per_target", "joint_then_independent_finetune"}

_TARGET_TO_TRANSFER_KEYS = {
    "hs": ("log_hs_ratio", "raw_log_hs_ratio"),
    "tp": ("tp_delta", "raw_tp_delta"),
    "dir": ("dir_delta_deg", "raw_dir_delta_deg"),
    "dp": ("dp_delta_deg", "raw_dp_delta_deg"),
}
_TARGET_TO_HYBRID_KEYS = {
    "hs": ("hs",),
    "tp": ("tp_log_probs", "tp_pred"),
    "dir": ("dir_log_probs", "dir_pred"),
    "dp": ("dp_log_probs", "dp_pred"),
}
_TARGET_TO_TENSOR_CHANNELS = {
    "hs": (0,),
    "tp": (1,),
    "dir": (2, 3),
    "dp": (4, 5),
}


def resolve_target_modeling(config: Mapping[str, Any] | None) -> str:
    resolved = resolve_config(config or {})
    training_cfg = (resolved.get("training", {}) or {}) if isinstance(resolved, Mapping) else {}
    mode = str(training_cfg.get("target_modeling", "joint")).strip().lower()
    if mode not in VALID_TARGET_MODELING:
        raise ValueError(
            "training.target_modeling must be one of: "
            "joint, independent_per_target, joint_then_independent_finetune; "
            f"got '{mode}'"
        )
    return mode


def _target_weight_map(active_target: str) -> dict[str, float]:
    if active_target not in TARGET_NAMES:
        raise ValueError(f"Unsupported target '{active_target}'. Expected one of {TARGET_NAMES}.")
    return {name: 1.0 if name == active_target else 0.0 for name in TARGET_NAMES}


def build_independent_target_child_config(
    parent_config: Mapping[str, Any],
    *,
    target_name: str,
    child_output_dir: str | Path,
    child_checkpoint_name: str,
    init_checkpoint_path: str | Path | None = None,
    child_stage: str = "from_scratch",
) -> dict[str, Any]:
    """Return a child training config that specializes supervision to one target."""
    if target_name not in TARGET_NAMES:
        raise ValueError(f"Unsupported target '{target_name}'. Expected one of {TARGET_NAMES}.")

    cfg = resolve_config(copy.deepcopy(dict(parent_config or {})))
    training_cfg = cfg.setdefault("training", {})
    loss_cfg = training_cfg.setdefault("loss", {})
    selection_cfg = training_cfg.setdefault("selection", {})
    pcgrad_cfg = training_cfg.setdefault("pcgrad", {})
    logging_cfg = cfg.setdefault("logging", {})

    weights = _target_weight_map(target_name)
    training_cfg["target_modeling"] = "joint"
    training_cfg["independent_target_child"] = {
        "enabled": True,
        "target_name": str(target_name),
        "parent_target_modeling": "independent_per_target",
        "stage": str(child_stage),
    }

    init_cfg = training_cfg.setdefault("initialization", {})
    init_cfg["checkpoint_path"] = (
        None if init_checkpoint_path in {None, ""} else str(Path(init_checkpoint_path).expanduser())
    )
    init_cfg["strict"] = False
    init_cfg["ignore_optimizer"] = True

    training_cfg["loss_weights"] = copy.deepcopy(weights)
    pcgrad_cfg["enabled"] = False

    weighted_cfg = loss_cfg.setdefault("weighted_multitask", {})
    weighted_cfg["weights"] = copy.deepcopy(weights)

    coastal_cfg = loss_cfg.setdefault("coastal_multitask", {})
    coastal_cfg["task_weights"] = copy.deepcopy(weights)

    blueprint_cfg = loss_cfg.setdefault("blueprint_multitask", {})
    blueprint_cfg["weights"] = copy.deepcopy(weights)

    hybrid_cfg = loss_cfg.setdefault("blueprint_hybrid", {})
    hybrid_cfg["weights"] = copy.deepcopy(weights)

    existing_selection_weights = copy.deepcopy(selection_cfg.get("weights", {}) or {})
    normalized_selection_weights = {}
    for key, value in existing_selection_weights.items():
        metric_name = str(key).strip().lower()
        if metric_name.startswith("hs_"):
            normalized_selection_weights[key] = float(value) if target_name == "hs" else 0.0
        elif metric_name.startswith("tp_"):
            normalized_selection_weights[key] = float(value) if target_name == "tp" else 0.0
        elif metric_name.startswith("dp_"):
            normalized_selection_weights[key] = float(value) if target_name == "dp" else 0.0
        elif metric_name.startswith("dir_") or metric_name.startswith("direction_"):
            normalized_selection_weights[key] = float(value) if target_name == "dir" else 0.0
        else:
            normalized_selection_weights[key] = float(value)
    if normalized_selection_weights:
        selection_cfg["weights"] = normalized_selection_weights

    logging_cfg["output_dir"] = str(Path(child_output_dir).expanduser())
    logging_cfg["checkpoint_name"] = str(child_checkpoint_name)
    return cfg


def is_independent_target_composite_metadata(training_metadata: Mapping[str, Any] | None) -> bool:
    runtime = (
        (dict(training_metadata or {}).get("runtime", {}) or {})
        if isinstance(training_metadata, Mapping)
        else {}
    )
    composite = runtime.get("independent_target_composite", {}) or {}
    return bool(composite.get("enabled", False))


def get_independent_target_child_entries(
    training_metadata: Mapping[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    runtime = (
        (dict(training_metadata or {}).get("runtime", {}) or {})
        if isinstance(training_metadata, Mapping)
        else {}
    )
    composite = runtime.get("independent_target_composite", {}) or {}
    if not bool(composite.get("enabled", False)):
        raise ValueError(
            "Training metadata does not describe an independent per-target composite run."
        )

    raw_targets = composite.get("targets", {}) or {}
    entries: dict[str, dict[str, Any]] = {}
    missing = [name for name in TARGET_NAMES if name not in raw_targets]
    if missing:
        raise ValueError(f"Composite run metadata is missing child target entries for: {missing}")

    for name in TARGET_NAMES:
        raw_entry = raw_targets.get(name, {}) or {}
        checkpoint_path = str(raw_entry.get("checkpoint_path", "")).strip()
        run_dir = str(raw_entry.get("run_dir", "")).strip()
        if not checkpoint_path:
            raise ValueError(
                f"Composite run metadata for target '{name}' is missing checkpoint_path."
            )
        if not run_dir:
            raise ValueError(f"Composite run metadata for target '{name}' is missing run_dir.")
        entries[name] = {
            "target_name": name,
            "checkpoint_path": checkpoint_path,
            "run_dir": run_dir,
            "config_path": str(raw_entry.get("config_path", "")).strip(),
            "metadata_path": str(raw_entry.get("metadata_path", "")).strip(),
        }
    return entries


def aggregate_child_histories(
    child_histories: Mapping[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Aggregate per-target history rows into one parent-facing train_history payload."""
    epoch_rows: dict[int, dict[str, Any]] = {}
    for target_name, rows in child_histories.items():
        for row in rows or []:
            epoch = int(row.get("epoch", 0))
            if epoch < 1:
                continue
            merged = epoch_rows.setdefault(
                epoch, {"epoch": epoch, "target_modeling": "independent_per_target"}
            )
            merged[f"{target_name}_child_validation_ran"] = bool(row.get("validation_ran", False))
            for key, value in row.items():
                if key == "epoch":
                    continue
                merged[f"{target_name}_{key}"] = value

    aggregated: list[dict[str, Any]] = []
    for epoch in sorted(epoch_rows):
        row = epoch_rows[epoch]
        train_losses = [
            float(row[f"{name}_train_loss"]) for name in TARGET_NAMES if f"{name}_train_loss" in row
        ]
        val_losses = [
            float(row[f"{name}_val_loss"])
            for name in TARGET_NAMES
            if row.get(f"{name}_validation_ran") and row.get(f"{name}_val_loss") is not None
        ]
        selection_scores = [
            float(row[f"{name}_selection_score"])
            for name in TARGET_NAMES
            if row.get(f"{name}_validation_ran") and row.get(f"{name}_selection_score") is not None
        ]
        if train_losses:
            row["train_loss"] = float(sum(train_losses) / len(train_losses))
        row["validation_ran"] = bool(val_losses)
        row["val_loss"] = float(sum(val_losses) / len(val_losses)) if val_losses else None
        row["selection_score"] = (
            float(sum(selection_scores) / len(selection_scores)) if selection_scores else None
        )
        row["selection_mode"] = "val_loss"
        aggregated.append(row)
    return aggregated


def _extract_target_tensor_slice(tensor: torch.Tensor, target_name: str) -> torch.Tensor:
    index = TARGET_NAMES.index(target_name)
    if tensor.ndim >= 3 and tensor.shape[-2] == len(TARGET_NAMES):
        parts = [slice(None)] * tensor.ndim
        parts[-2] = slice(index, index + 1)
        return tensor[tuple(parts)]
    if tensor.ndim >= 2 and tensor.shape[1] == len(TARGET_NAMES):
        return tensor[:, index : index + 1, ...]
    if tensor.ndim >= 1 and tensor.shape[0] == len(TARGET_NAMES):
        return tensor[index : index + 1, ...]
    raise ValueError(
        "Could not infer task axis while merging composite diagnostics. "
        f"Observed tensor shape={tuple(tensor.shape)}"
    )


def _merge_transfer_outputs(child_outputs: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for target_name, keys in _TARGET_TO_TRANSFER_KEYS.items():
        child = child_outputs[target_name]
        for key in keys:
            if key in child:
                merged[key] = child[key]
    return merged


def _merge_hybrid_outputs(child_outputs: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for target_name, keys in _TARGET_TO_HYBRID_KEYS.items():
        child = child_outputs[target_name]
        for key in keys:
            if key in child:
                merged[key] = child[key]
    return merged


def _merge_tensor_outputs(child_outputs: Mapping[str, torch.Tensor]) -> torch.Tensor:
    first = child_outputs[TARGET_NAMES[0]]
    if first.ndim != 2:
        raise ValueError(
            "Composite tensor outputs currently expect rank-2 tensors shaped [batch, channels]. "
            f"Observed shape={tuple(first.shape)}"
        )
    merged = torch.zeros_like(first)
    for target_name, channel_indices in _TARGET_TO_TENSOR_CHANNELS.items():
        child_tensor = child_outputs[target_name]
        for index in channel_indices:
            merged[:, index] = child_tensor[:, index]
    return merged


class IndependentTargetCompositeModel(nn.Module):
    """Composite wrapper that exposes four child models as one multitask model."""

    task_order = TARGET_NAMES

    def __init__(self, child_models: Mapping[str, nn.Module]) -> None:
        super().__init__()
        missing = [name for name in TARGET_NAMES if name not in child_models]
        if missing:
            raise ValueError(f"Composite model is missing child models for targets: {missing}")
        self.child_models = nn.ModuleDict({name: child_models[name] for name in TARGET_NAMES})
        exemplar = self.child_models[TARGET_NAMES[0]]
        for attr in (
            "use_static_features",
            "use_source_geometry_features",
            "use_bathymetry",
            "use_multi_source",
            "sequence_encoder_type",
            "decoder_type",
            "target_mode",
            "transfer_representation",
            "bathy_encoder",
        ):
            if hasattr(exemplar, attr):
                setattr(self, attr, getattr(exemplar, attr))

    def forward(self, *args, **kwargs):
        child_outputs = {name: self.child_models[name](*args, **kwargs) for name in TARGET_NAMES}
        first = child_outputs[TARGET_NAMES[0]]

        if torch.is_tensor(first):
            return _merge_tensor_outputs(
                {name: value for name, value in child_outputs.items() if torch.is_tensor(value)}
            )

        if not isinstance(first, Mapping):
            raise TypeError(
                "IndependentTargetCompositeModel expects child outputs to be either tensors or mappings. "
                f"Observed child output type={type(first).__name__}"
            )

        if "log_hs_ratio" in first:
            merged = _merge_transfer_outputs(child_outputs)
        else:
            merged = _merge_hybrid_outputs(child_outputs)

        if any("cross_attention_weights" in output for output in child_outputs.values()):
            slices = []
            for name in TARGET_NAMES:
                attn = child_outputs[name].get("cross_attention_weights")
                if torch.is_tensor(attn):
                    slices.append(_extract_target_tensor_slice(attn, name))
            if slices:
                merged["cross_attention_weights"] = torch.cat(slices, dim=-2)

        source_attention = next(
            (
                output.get("source_attention_weights")
                for output in child_outputs.values()
                if torch.is_tensor(output.get("source_attention_weights"))
            ),
            None,
        )
        if torch.is_tensor(source_attention):
            merged["source_attention_weights"] = source_attention

        diagnostics = {}
        for key in (
            "context_tokens",
            "dynamic_tokens",
            "static_token",
            "bathy_tokens",
            "bathy_summary",
            "context_token_types",
        ):
            value = next(
                (
                    (output.get("diagnostics") or {}).get(key)
                    for output in child_outputs.values()
                    if isinstance(output.get("diagnostics"), Mapping)
                    and (output.get("diagnostics") or {}).get(key) is not None
                ),
                None,
            )
            if value is not None:
                diagnostics[key] = value

        task_token_slices = []
        for name in TARGET_NAMES:
            diag = child_outputs[name].get("diagnostics") or {}
            task_tokens = diag.get("task_tokens")
            if torch.is_tensor(task_tokens):
                task_token_slices.append(_extract_target_tensor_slice(task_tokens, name))
        if task_token_slices:
            diagnostics["task_tokens"] = torch.cat(task_token_slices, dim=1)

        if diagnostics:
            merged["diagnostics"] = diagnostics
        return merged


def load_model_from_training_metadata(
    *,
    training_metadata: Mapping[str, Any],
    build_model: Callable[[], nn.Module],
    checkpoint_loader: Callable[[Path], object],
    state_dict_getter: Callable[[object], dict],
    state_dict_loader: Callable[[nn.Module, dict], Any],
) -> nn.Module:
    """Load a saved single-checkpoint or composite results payload into a model."""
    if is_independent_target_composite_metadata(training_metadata):
        entries = get_independent_target_child_entries(training_metadata)
        child_models: dict[str, nn.Module] = {}
        for target_name in TARGET_NAMES:
            model = build_model()
            checkpoint = checkpoint_loader(
                Path(entries[target_name]["checkpoint_path"]).expanduser()
            )
            state_dict = state_dict_getter(checkpoint)
            state_dict_loader(model, state_dict)
            model.eval()
            child_models[target_name] = model
        composite = IndependentTargetCompositeModel(child_models)
        composite.eval()
        return composite

    runtime = (
        (dict(training_metadata or {}).get("runtime", {}) or {})
        if isinstance(training_metadata, Mapping)
        else {}
    )
    checkpoint_path = str(runtime.get("checkpoint_path", "")).strip()
    if not checkpoint_path:
        raise ValueError("Training metadata does not include runtime.checkpoint_path.")

    model = build_model()
    checkpoint = checkpoint_loader(Path(checkpoint_path).expanduser())
    state_dict = state_dict_getter(checkpoint)
    state_dict_loader(model, state_dict)
    model.eval()
    return model


__all__ = [
    "IndependentTargetCompositeModel",
    "TARGET_NAMES",
    "VALID_TARGET_MODELING",
    "aggregate_child_histories",
    "build_independent_target_child_config",
    "get_independent_target_child_entries",
    "is_independent_target_composite_metadata",
    "load_model_from_training_metadata",
    "resolve_target_modeling",
]
