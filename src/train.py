"""Train a coastal-conditioned wave model from prepared point-centric data.

The command-line entry point reads one training configuration, loads its saved
preprocessing artifacts, builds the configured model branches and loss, then
writes checkpoints, histories, predictions, and run metadata.  Configuration
defines the full experiment; this module does not discover or normalize raw
time series.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import pickle
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn
from torch.optim import Adam, AdamW, RMSprop, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR

try:
    from config_loader import read_yaml_config
except Exception:
    from src.config_loader import read_yaml_config

try:
    from evaluate import (
        _extract_state_dict,
        _load_checkpoint_compat,
        _load_state_dict_best_effort,
        infer,
        prepare_evaluation_outputs,
    )
except Exception:
    from src.evaluate import (
        _extract_state_dict,
        _load_checkpoint_compat,
        _load_state_dict_best_effort,
        infer,
        prepare_evaluation_outputs,
    )

try:
    from preprocessing.transfer_targets import resolve_targets_config
except Exception:
    from src.preprocessing.transfer_targets import resolve_targets_config

try:
    from config_resolution import resolve_config
except Exception:
    from src.config_resolution import resolve_config

try:
    from runtime_paths import normalize_runtime_config_paths
    from training_batch import load_batch_training_plan
except Exception:
    from src.runtime_paths import normalize_runtime_config_paths
    from src.training_batch import load_batch_training_plan

try:
    from independent_target_mode import (
        TARGET_NAMES,
        aggregate_child_histories,
        build_independent_target_child_config,
        load_model_from_training_metadata,
        resolve_target_modeling,
    )
except Exception:
    from src.independent_target_mode import (
        TARGET_NAMES,
        aggregate_child_histories,
        build_independent_target_child_config,
        load_model_from_training_metadata,
        resolve_target_modeling,
    )

try:
    from data_pipeline import (
        apply_runtime_static_ablation,
        build_split_dataloader,
        load_point_centric_arrays,
        resolve_site_split_config,
        resolve_static_ablation_config_path,
    )
    from losses import build_loss_from_config
    from metrics import (
        compute_site_target_metric_rows,
        compute_split_target_metric_rows,
        compute_wave_metrics,
    )
    from models import build_model_from_config
    from transfer_runtime import (
        load_transfer_scaler_stats,
        physical_matrix_to_metric_channels,
        recover_transfer_metric_arrays,
    )
    from training.pcgrad_bridge import create_pcgrad_optimizer
except Exception:
    from src.data_pipeline import (
        apply_runtime_static_ablation,
        build_split_dataloader,
        load_point_centric_arrays,
        resolve_site_split_config,
        resolve_static_ablation_config_path,
    )
    from src.losses import build_loss_from_config
    from src.metrics import (
        compute_site_target_metric_rows,
        compute_split_target_metric_rows,
        compute_wave_metrics,
    )
    from src.models import build_model_from_config
    from src.transfer_runtime import (
        load_transfer_scaler_stats,
        physical_matrix_to_metric_channels,
        recover_transfer_metric_arrays,
    )
    from src.training.pcgrad_bridge import create_pcgrad_optimizer

try:
    from tqdm.auto import tqdm as _tqdm
except Exception:
    _tqdm = None


DEFAULT_PHYSICAL_SCORE_WEIGHTS: Dict[str, float] = {
    "hs_rmse": 5.0,
    "tp_rmse": 1.0,
    "direction_rmse_deg": 0.05,
    "dp_rmse_deg": 0.05,
}
SUPPORTED_PHYSICAL_SCORE_METRICS = {
    "hs_rmse",
    "hs_r2",
    "tp_rmse",
    "tp_r2",
    "dir_rmse_deg",
    "dir_r2",
    "dp_rmse_deg",
    "dp_r2",
    "direction_rmse_deg",
    "direction_r2",
}
COMPONENT_NAMES = ("hs", "tp", "dir", "dp")
DEFAULT_BREAKING_PHYSICS_CONFIG: Dict[str, object] = {
    "enabled": False,
    "mode": "local_depth",
    "cap_key": "local_breaking_hs_cap",
    "valid_key": "local_breaking_cap_valid",
    "penalty_weight": 0.02,
}


def read_yaml(path: str) -> dict:
    return read_yaml_config(path)


def _read_json(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as fh:
        return json.load(fh) or {}


def run_batch_training(batch_config_path: str, device: str) -> int:
    batch_payload, case_paths = load_batch_training_plan(batch_config_path)
    stop_on_error = bool(batch_payload.get("stop_on_error", True))
    batch_name = str(batch_payload.get("name", Path(batch_config_path).stem))
    script_path = Path(__file__).resolve()

    print(f"=== Batch training: {batch_name} ===", flush=True)
    print(f"Batch config: {Path(batch_config_path).resolve()}", flush=True)
    print(f"Device argument: {device}", flush=True)
    for idx, case_path in enumerate(case_paths, start=1):
        print(f"  {idx}. {case_path}", flush=True)

    results: list[tuple[Path, int]] = []
    for idx, case_path in enumerate(case_paths, start=1):
        print("", flush=True)
        print(f"=== [{idx}/{len(case_paths)}] Starting {case_path.name} ===", flush=True)
        command = [
            sys.executable,
            str(script_path),
            "--config",
            str(case_path),
            "--device",
            device,
        ]
        completed = subprocess.run(command, check=False)
        results.append((case_path, int(completed.returncode)))
        if completed.returncode != 0:
            print(
                f"=== {case_path.name} failed with exit code {completed.returncode} ===", flush=True
            )
            if stop_on_error:
                break
        else:
            print(f"=== {case_path.name} completed successfully ===", flush=True)

    print("", flush=True)
    print(f"=== Batch summary: {batch_name} ===", flush=True)
    for case_path, returncode in results:
        status = "ok" if returncode == 0 else f"failed ({returncode})"
        print(f"  - {case_path.name}: {status}", flush=True)

    failed = [case_path for case_path, returncode in results if returncode != 0]
    remaining = case_paths[len(results) :]
    for case_path in remaining:
        print(f"  - {case_path.name}: not run", flush=True)

    if failed:
        return 1
    return 0


def _load_target_scaler_stats(point_centric_dir: str) -> tuple[float, float] | None:
    metadata_path = Path(point_centric_dir) / "point_centric_metadata.json"
    if not metadata_path.exists():
        return None

    try:
        with metadata_path.open("r") as fh:
            payload = json.load(fh) or {}
    except Exception:
        return None

    norm = payload.get("normalization", {}) or {}
    target_scaler = norm.get("target_scaler", {}) or {}
    if not isinstance(target_scaler, dict):
        return None

    feature_names = list(target_scaler.get("feature_names", []) or [])
    means = list(target_scaler.get("mean", []) or [])
    scales = list(target_scaler.get("scale", []) or [])
    if (
        "hs" not in feature_names
        or len(feature_names) != len(means)
        or len(feature_names) != len(scales)
    ):
        return None

    idx = feature_names.index("hs")
    scale = float(scales[idx]) or 1.0
    return float(means[idx]), scale


def _resolve_runtime_bathy_request(config: dict) -> tuple[list[str] | None, int | None]:
    model_cfg = (config.get("model", {}) or {}).get("coastal_transformer", {}) or {}
    bathy_cfg = model_cfg.get("bathy", {}) or {}
    configured_channels = [str(name) for name in (bathy_cfg.get("channels", []) or [])]
    requested_in_channels_raw = bathy_cfg.get("in_channels", None)
    requested_in_channels = (
        int(requested_in_channels_raw) if requested_in_channels_raw is not None else None
    )
    requested_channels = configured_channels if configured_channels else None
    return requested_channels, requested_in_channels


def _recover_hybrid_metrics_arrays(
    pred,
    target,
    target_scaler_stats: tuple[float, float] | None,
) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(pred, dict) or not isinstance(target, dict):
        return np.asarray(pred), np.asarray(target)

    hs_mean, hs_scale = target_scaler_stats if target_scaler_stats is not None else (0.0, 1.0)

    hs_true = target["hs"].detach().cpu().numpy().reshape(-1)
    hs_pred = pred["hs"].detach().cpu().numpy().reshape(-1)
    hs_true = hs_true * hs_scale + hs_mean
    hs_pred = hs_pred * hs_scale + hs_mean

    tp_true = (
        target.get("tp_value", target["tp_soft"].argmax(dim=-1)).detach().cpu().numpy().reshape(-1)
    )
    tp_pred = pred["tp_pred"].detach().cpu().numpy().reshape(-1)

    dir_true = (
        target.get("dir_value", target["dir_soft"].argmax(dim=-1))
        .detach()
        .cpu()
        .numpy()
        .reshape(-1)
    )
    dir_pred = pred["dir_pred"].detach().cpu().numpy().reshape(-1)

    dp_true = (
        target.get("dp_value", target["dp_soft"].argmax(dim=-1)).detach().cpu().numpy().reshape(-1)
    )
    dp_pred = pred["dp_pred"].detach().cpu().numpy().reshape(-1)

    def _angle_channels(angle_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        radians = np.deg2rad(angle_deg)
        return np.sin(radians), np.cos(radians)

    dir_true_sin, dir_true_cos = _angle_channels(dir_true)
    dir_pred_sin, dir_pred_cos = _angle_channels(dir_pred)
    dp_true_sin, dp_true_cos = _angle_channels(dp_true)
    dp_pred_sin, dp_pred_cos = _angle_channels(dp_pred)

    y_true = np.column_stack(
        [hs_true, tp_true, dir_true_sin, dir_true_cos, dp_true_sin, dp_true_cos]
    )
    y_pred = np.column_stack(
        [hs_pred, tp_pred, dir_pred_sin, dir_pred_cos, dp_pred_sin, dp_pred_cos]
    )
    return y_pred, y_true


def _compute_transfer_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    diff = y_pred - y_true
    dir_delta = ((y_pred[:, 2] - y_true[:, 2] + 180.0) % 360.0) - 180.0
    dp_delta = ((y_pred[:, 3] - y_true[:, 3] + 180.0) % 360.0) - 180.0
    return {
        "log_hs_ratio_rmse": float(np.sqrt(np.mean(np.square(diff[:, 0])))),
        "tp_delta_rmse": float(np.sqrt(np.mean(np.square(diff[:, 1])))),
        "dir_delta_rmse_deg": float(np.sqrt(np.mean(np.square(dir_delta)))),
        "dp_delta_rmse_deg": float(np.sqrt(np.mean(np.square(dp_delta)))),
    }


def _all_finite(output) -> bool:
    if torch.is_tensor(output):
        return bool(torch.isfinite(output).all())
    if isinstance(output, dict):
        tensors = [value for value in output.values() if torch.is_tensor(value)]
        return all(bool(torch.isfinite(tensor).all()) for tensor in tensors)
    return True


def _to_device_batch(target, device: torch.device):
    if isinstance(target, dict):
        return {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in target.items()
        }
    return target.to(device)


def _count_non_finite(output) -> int:
    if torch.is_tensor(output):
        return int((~torch.isfinite(output)).sum().item())
    if isinstance(output, dict):
        total = 0
        for value in output.values():
            if torch.is_tensor(value):
                total += int((~torch.isfinite(value)).sum().item())
        return total
    return 0


def _extract_model_inputs(
    batch: dict,
    device: torch.device,
    *,
    use_static_features: bool = True,
    use_source_geometry_features: bool = True,
) -> dict:
    x_dynamic = batch["x_dynamic"].to(device) if "x_dynamic" in batch else None
    x_dynamic_sources = (
        batch["x_dynamic_sources"].to(device) if "x_dynamic_sources" in batch else None
    )
    source_geometry = (
        batch["source_geometry"].to(device)
        if use_source_geometry_features and "source_geometry" in batch
        else None
    )
    x_static = batch["x_static"].to(device) if use_static_features and "x_static" in batch else None
    x_bathy = batch["x_bathy"].to(device) if "x_bathy" in batch else None
    return {
        "x_dynamic": x_dynamic,
        "x_dynamic_sources": x_dynamic_sources,
        "source_geometry": source_geometry,
        "x_static": x_static,
        "x_bathy": x_bathy,
    }


def _as_float_pair(values: Iterable[float], default: Tuple[float, float]) -> Tuple[float, float]:
    vals = list(values)
    if len(vals) != 2:
        return default
    return float(vals[0]), float(vals[1])


def resolve_breaking_physics_config(config: dict) -> Dict[str, object]:
    physics_cfg = (config.get("physics", {}) or {}).get("breaking", {}) or {}
    merged = {**DEFAULT_BREAKING_PHYSICS_CONFIG, **physics_cfg}
    merged["enabled"] = bool(merged.get("enabled", False))
    merged["mode"] = str(merged.get("mode", "local_depth")).strip().lower()
    merged["cap_key"] = (
        str(merged.get("cap_key", "local_breaking_hs_cap")).strip() or "local_breaking_hs_cap"
    )
    merged["valid_key"] = (
        str(merged.get("valid_key", "local_breaking_cap_valid")).strip()
        or "local_breaking_cap_valid"
    )
    merged["penalty_weight"] = float(merged.get("penalty_weight", 0.02))
    if merged["mode"] != "local_depth":
        raise ValueError("physics.breaking.mode currently supports only 'local_depth'")
    if merged["penalty_weight"] < 0.0:
        raise ValueError("physics.breaking.penalty_weight must be >= 0")
    return merged


def _recover_pred_hs_physical(
    pred,
    target,
    target_scaler_stats: tuple[float, float] | None,
    transfer_scaler_stats: Dict[int, Tuple[float, float]] | None,
    transfer_representation: str = "legacy",
) -> torch.Tensor | None:
    hs_mean, hs_scale = target_scaler_stats if target_scaler_stats is not None else (0.0, 1.0)
    transfer_stats = transfer_scaler_stats or {}

    if torch.is_tensor(pred):
        if pred.ndim < 2 or pred.shape[-1] < 1:
            return None
        hs_scaled = pred[..., 0].reshape(-1)
        return (hs_scaled * float(hs_scale)) + float(hs_mean)

    if isinstance(pred, dict):
        if "log_hs_ratio" in pred and isinstance(target, dict) and "ref_hs" in target:
            log_ratio = pred["log_hs_ratio"].reshape(-1)
            if (
                str(transfer_representation).strip().lower() != "residual_correction"
                and 0 in transfer_stats
            ):
                mean, scale = transfer_stats[0]
                log_ratio = (log_ratio * float(scale)) + float(mean)
            ref_hs = target["ref_hs"].reshape(-1)
            return ref_hs * torch.exp(log_ratio)

        if "hs" in pred:
            hs_scaled = pred["hs"].reshape(-1)
            return (hs_scaled * float(hs_scale)) + float(hs_mean)
    return None


def _compute_breaking_penalty(
    pred,
    target,
    batch: dict,
    *,
    cap_key: str,
    valid_key: str,
    target_scaler_stats: tuple[float, float] | None,
    transfer_scaler_stats: Dict[int, Tuple[float, float]] | None,
    transfer_representation: str = "legacy",
) -> torch.Tensor | None:
    if cap_key not in batch or valid_key not in batch:
        return None
    cap = batch[cap_key]
    valid = batch[valid_key]
    if not torch.is_tensor(cap) or not torch.is_tensor(valid):
        return None

    pred_hs_physical = _recover_pred_hs_physical(
        pred=pred,
        target=target,
        target_scaler_stats=target_scaler_stats,
        transfer_scaler_stats=transfer_scaler_stats,
        transfer_representation=transfer_representation,
    )
    if pred_hs_physical is None:
        return None

    cap_vec = cap.reshape(-1).to(device=pred_hs_physical.device, dtype=pred_hs_physical.dtype)
    valid_mask = (
        valid.reshape(-1).to(device=pred_hs_physical.device, dtype=pred_hs_physical.dtype) > 0.5
    )
    finite_mask = torch.isfinite(cap_vec) & torch.isfinite(pred_hs_physical)
    mask = valid_mask & finite_mask
    if not bool(torch.any(mask)):
        return None

    excess = torch.relu(pred_hs_physical[mask] - cap_vec[mask])
    return torch.mean(excess.pow(2))


def set_random_seed(seed: int, deterministic: bool = False) -> None:
    """Set numpy/torch seeds for reproducible runs when desired."""
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _resolve_progress_bar_config(config: dict) -> Dict[str, object]:
    progress_cfg = ((config.get("training", {}) or {}).get("progress_bar", {})) or {}
    return {
        "enabled": bool(progress_cfg.get("enabled", False)),
        "train": bool(progress_cfg.get("train", True)),
        "val": bool(progress_cfg.get("val", False)),
        "leave": bool(progress_cfg.get("leave", False)),
        "update_interval": max(1, int(progress_cfg.get("update_interval", 1))),
    }


def _should_enable_progress_bar(progress_cfg: Dict[str, object], phase: str) -> bool:
    if not bool(progress_cfg.get("enabled", False)):
        return False
    if _tqdm is None:
        return False
    if not bool(progress_cfg.get(phase, phase == "train")):
        return False
    return bool(getattr(sys.stderr, "isatty", lambda: False)())


def maybe_tqdm(iterable, enabled: bool, desc: str, leave: bool, total: int | None = None):
    if not enabled:
        return iterable
    return _tqdm(iterable, desc=desc, leave=leave, total=total, dynamic_ncols=True)


def _format_site_preview(sites: Iterable[str], limit: int = 12) -> str:
    items = [str(site) for site in sites]
    if len(items) <= limit:
        return str(items)
    preview = items[:limit]
    return f"{preview} ... (+{len(items) - limit} more)"


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, torch.dtype):
        return str(value)
    return value


def _write_serialized_artifact(
    out_dir: Path, stem: str, payload: Dict[str, Any]
) -> Tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{stem}.json"
    pickle_path = out_dir / f"{stem}.pkl"
    safe_payload = _json_safe(payload)
    with json_path.open("w") as fh:
        json.dump(safe_payload, fh, indent=2)
    with pickle_path.open("wb") as fh:
        pickle.dump(safe_payload, fh)
    return json_path, pickle_path


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _update_training_timing_metadata(
    run_metadata: Dict[str, Any],
    *,
    training_started_at_utc: str,
    training_finished_at_utc: str | None = None,
    training_duration_seconds: float | None = None,
) -> None:
    runtime = run_metadata.setdefault("runtime", {})
    runtime["training_started_at_utc"] = str(training_started_at_utc)
    runtime["training_finished_at_utc"] = (
        None if training_finished_at_utc is None else str(training_finished_at_utc)
    )
    runtime["training_duration_seconds"] = (
        None if training_duration_seconds is None else float(training_duration_seconds)
    )


def _tensor_summary(
    tensor: torch.Tensor | None,
    *,
    feature_names: Sequence[str] | None = None,
    feature_axis: int | None = None,
) -> Dict[str, Any]:
    if tensor is None:
        return {"present": False}

    cpu = tensor.detach().cpu()
    summary: Dict[str, Any] = {
        "present": True,
        "shape": list(cpu.shape),
        "dtype": str(cpu.dtype),
        "numel": int(cpu.numel()),
    }

    if feature_names is not None:
        names = [str(name) for name in feature_names]
        summary["feature_names"] = names
        summary["feature_name_count"] = len(names)
        if feature_axis is not None and cpu.ndim > 0:
            resolved_axis = int(feature_axis)
            if resolved_axis < 0:
                resolved_axis += cpu.ndim
            if 0 <= resolved_axis < cpu.ndim:
                summary["feature_axis"] = resolved_axis
                summary["feature_count_from_tensor"] = int(cpu.shape[resolved_axis])
                summary["feature_count_matches_names"] = int(cpu.shape[resolved_axis]) == len(names)

    if cpu.numel() == 0:
        return summary

    if cpu.is_floating_point() or cpu.is_complex():
        finite_mask = torch.isfinite(cpu)
        non_finite_count = int((~finite_mask).sum().item())
        summary["non_finite_count"] = non_finite_count
        summary["all_finite"] = non_finite_count == 0
        finite_values = cpu[finite_mask]
        if finite_values.numel() > 0:
            summary["min"] = float(finite_values.min().item())
            summary["max"] = float(finite_values.max().item())
            summary["mean"] = float(finite_values.mean().item())
    else:
        summary["non_finite_count"] = 0
        summary["all_finite"] = True

    flat = cpu.reshape(-1)
    preview_len = min(8, int(flat.numel()))
    summary["preview_values"] = _json_safe(flat[:preview_len])
    return summary


def _resolve_effective_dynamic_feature_names(arrays) -> list[str]:
    return [
        *[str(name) for name in (getattr(arrays, "dynamic_feature_names", []) or [])],
        *[str(name) for name in (getattr(arrays, "site_dynamic_feature_names", []) or [])],
        "time_sin",
        "time_cos",
    ]


def _resolve_effective_source_feature_names(arrays) -> list[str]:
    return [
        *[str(name) for name in (getattr(arrays, "source_feature_names", []) or [])],
        "time_sin",
        "time_cos",
    ]


def _resolve_target_tensor_feature_names(
    name: str, tensor: torch.Tensor, arrays
) -> list[str] | None:
    if name in {"transfer", "transfer_scaled"}:
        return [str(item) for item in (getattr(arrays, "transfer_target_names", []) or [])]
    if name == "reference":
        return [str(item) for item in (getattr(arrays, "reference_target_names", []) or [])]
    if name == "physical":
        return [str(item) for item in (getattr(arrays, "physical_target_names", []) or [])]
    if name in {"tp_soft", "dir_soft", "dp_soft"} and tensor.ndim > 0:
        return [f"{name}_bin_{idx}" for idx in range(int(tensor.shape[-1]))]
    if name == "hs":
        return ["hs"]
    if name.endswith("_index"):
        return [name]
    if name.endswith("_value"):
        return [name]
    if (
        name.startswith("ref_")
        or name.startswith("physical_")
        or name.endswith("_deg")
        or name.endswith("_delta")
        or name == "log_hs_ratio"
    ):
        return [name]
    return None


def _build_model_io_manifest(
    *,
    cfg: dict,
    arrays,
    batch: dict,
    model_inputs: dict,
    target,
    model: nn.Module,
    epoch_index: int,
    batch_idx: int,
    split_name: str,
) -> Dict[str, Any]:
    resolved_cfg = resolve_config(cfg)
    resolved_targets = resolve_targets_config((resolved_cfg.get("data", {}) or {}))
    resolution_meta = resolved_cfg.get("_config_resolution", {}) or {}
    dynamic_feature_names = _resolve_effective_dynamic_feature_names(arrays)
    static_feature_names = [
        str(name) for name in (getattr(arrays, "static_feature_names", []) or [])
    ]
    source_feature_names = _resolve_effective_source_feature_names(arrays)
    source_geometry_feature_names = [
        str(name) for name in (getattr(arrays, "source_geometry_feature_names", []) or [])
    ]
    bathy_channel_names = [str(name) for name in (getattr(arrays, "bathy_channel_names", []) or [])]
    model_decoder_cfg = (
        (resolved_cfg.get("model", {}) or {}).get("coastal_transformer", {}) or {}
    ).get("decoder", {}) or {}

    manifest: Dict[str, Any] = {
        "captured_from": {
            "split": str(split_name),
            "epoch": int(epoch_index),
            "batch_index": int(batch_idx),
            "batch_size": int(len(batch.get("site", [])))
            if isinstance(batch.get("site"), list)
            else None,
            "sites_preview": [str(site) for site in (batch.get("site", []) or [])[:8]],
            "timestamps_preview": [str(ts) for ts in (batch.get("timestamp", []) or [])[:8]],
        },
        "config_flags": {
            "data_use_static_features": bool(
                (resolved_cfg.get("data", {}) or {}).get("use_static_features", True)
            ),
            "data_use_bathymetry": bool(
                (resolved_cfg.get("data", {}) or {}).get("use_bathymetry", False)
            ),
            "data_multi_source_enabled": bool(
                ((resolved_cfg.get("data", {}) or {}).get("multi_source", {}) or {}).get(
                    "enabled", False
                )
            ),
            "model_decoder_type": str(model_decoder_cfg.get("type", "cross_attention")),
            "model_use_source_geometry_features": bool(
                (
                    (
                        (resolved_cfg.get("model", {}) or {}).get("coastal_transformer", {}) or {}
                    ).get("multi_source", {})
                    or {}
                ).get("use_geometry_features", False)
            ),
            "target_mode": str(resolved_targets.get("mode", "physical")),
            "transfer_reference": str(resolved_targets.get("transfer_reference", "nearest_bulk")),
            "transfer_representation": str(
                resolved_targets.get("transfer_representation", "legacy")
            ),
        },
        "model_flags": {
            "architecture": str(
                (resolved_cfg.get("model", {}) or {}).get("architecture", "coastal_transformer")
            ),
            "model_class": type(model).__name__,
            "sequence_encoder_type": str(getattr(model, "sequence_encoder_type", "unknown")),
            "decoder_type": str(getattr(model, "decoder_type", "cross_attention")),
            "use_static_features": bool(getattr(model, "use_static_features", False)),
            "use_bathymetry": bool(getattr(model, "use_bathymetry", False)),
            "use_multi_source": bool(getattr(model, "use_multi_source", False)),
            "use_source_geometry_features": bool(
                getattr(model, "use_source_geometry_features", False)
            ),
        },
        "config_resolution": {
            "warnings": list(resolution_meta.get("warnings", []) or []),
            "resolved": resolution_meta.get("resolved", {}),
        },
        "targets_config": resolved_targets,
        "inputs_passed_to_model": {
            "x_dynamic": _tensor_summary(
                model_inputs.get("x_dynamic"), feature_names=dynamic_feature_names, feature_axis=-1
            ),
            "x_static": _tensor_summary(
                model_inputs.get("x_static"), feature_names=static_feature_names, feature_axis=-1
            ),
            "x_dynamic_sources": _tensor_summary(
                model_inputs.get("x_dynamic_sources"),
                feature_names=source_feature_names,
                feature_axis=-1,
            ),
            "source_geometry": _tensor_summary(
                model_inputs.get("source_geometry"),
                feature_names=source_geometry_feature_names,
                feature_axis=-1,
            ),
            "x_bathy": _tensor_summary(
                model_inputs.get("x_bathy"), feature_names=bathy_channel_names, feature_axis=1
            ),
        },
        "available_batch_tensors_not_passed_to_model": {},
        "targets": {},
    }

    if "x_dynamic_static_concat" in batch and torch.is_tensor(batch["x_dynamic_static_concat"]):
        manifest["available_batch_tensors_not_passed_to_model"]["x_dynamic_static_concat"] = (
            _tensor_summary(
                batch["x_dynamic_static_concat"],
                feature_names=[*dynamic_feature_names, *static_feature_names],
                feature_axis=-1,
            )
        )
    if (
        model_inputs.get("x_static") is None
        and "x_static" in batch
        and torch.is_tensor(batch["x_static"])
    ):
        manifest["available_batch_tensors_not_passed_to_model"]["x_static"] = _tensor_summary(
            batch["x_static"],
            feature_names=static_feature_names,
            feature_axis=-1,
        )
    if (
        model_inputs.get("source_geometry") is None
        and "source_geometry" in batch
        and torch.is_tensor(batch["source_geometry"])
    ):
        manifest["available_batch_tensors_not_passed_to_model"]["source_geometry"] = (
            _tensor_summary(
                batch["source_geometry"],
                feature_names=source_geometry_feature_names,
                feature_axis=-1,
            )
        )

    if isinstance(target, dict):
        manifest["targets"]["structure"] = "mapping"
        manifest["targets"]["tensors"] = {}
        for name, value in target.items():
            if not torch.is_tensor(value):
                continue
            feature_names = _resolve_target_tensor_feature_names(str(name), value, arrays)
            feature_axis = (
                -1 if feature_names and value.ndim > 0 and len(feature_names) > 1 else None
            )
            manifest["targets"]["tensors"][str(name)] = _tensor_summary(
                value,
                feature_names=feature_names,
                feature_axis=feature_axis,
            )
    else:
        output_columns = [
            str(col) for col in ((cfg.get("data", {}) or {}).get("output_columns", []) or [])
        ]
        if (
            not output_columns
            and torch.is_tensor(target)
            and target.ndim > 0
            and target.shape[-1] == len(getattr(arrays, "target_feature_names", []) or [])
        ):
            output_columns = [
                str(col) for col in (getattr(arrays, "target_feature_names", []) or [])
            ]
        manifest["targets"] = {
            "structure": "tensor",
            "tensor": _tensor_summary(
                target,
                feature_names=output_columns or None,
                feature_axis=-1 if output_columns else None,
            ),
        }

    return manifest


def build_optimizer(
    model: nn.Module,
    config: dict,
    extra_parameters: Iterable[nn.Parameter] | None = None,
) -> Tuple[torch.optim.Optimizer, str]:
    """Create optimizer from config with backward-compatible defaults."""
    config = resolve_config(config)
    train_cfg = config.get("training", {}) or {}
    opt_cfg = train_cfg.get("optimizer", {}) or {}

    kind = str(opt_cfg.get("type", train_cfg.get("optimizer_type", "adamw"))).lower()
    lr = float(opt_cfg.get("lr", 1e-3))
    weight_decay = float(opt_cfg.get("weight_decay", 1e-5))

    model_params: list[nn.Parameter] = [p for p in model.parameters() if p.requires_grad]
    loss_params: list[nn.Parameter] = []
    if extra_parameters is not None:
        for p in extra_parameters:
            if p is None or not p.requires_grad:
                continue
            if any(p is q for q in model_params):
                continue
            if any(p is q for q in loss_params):
                continue
            loss_params.append(p)

    params: list[nn.Parameter] = [*model_params, *loss_params]
    if not params:
        raise ValueError("No trainable parameters found for optimizer")

    if kind == "adamw":
        betas = _as_float_pair(opt_cfg.get("betas", [0.9, 0.999]), default=(0.9, 0.999))
        eps = float(opt_cfg.get("eps", 1e-8))
        amsgrad = bool(opt_cfg.get("amsgrad", False))

        param_groups = []
        if model_params:
            param_groups.append(
                {
                    "params": model_params,
                    "lr": lr,
                    "weight_decay": weight_decay,
                }
            )
        if loss_params:
            param_groups.append(
                {
                    "params": loss_params,
                    "lr": lr * 0.1,
                    "weight_decay": 0.0,
                }
            )

        optimizer = AdamW(
            param_groups,
            betas=betas,
            eps=eps,
            amsgrad=amsgrad,
        )
        return optimizer, kind

    if kind == "adam":
        betas = _as_float_pair(opt_cfg.get("betas", [0.9, 0.999]), default=(0.9, 0.999))
        eps = float(opt_cfg.get("eps", 1e-8))
        amsgrad = bool(opt_cfg.get("amsgrad", False))
        optimizer = Adam(
            params,
            lr=lr,
            weight_decay=weight_decay,
            betas=betas,
            eps=eps,
            amsgrad=amsgrad,
        )
        return optimizer, kind

    if kind == "sgd":
        momentum = float(opt_cfg.get("momentum", 0.9))
        dampening = float(opt_cfg.get("dampening", 0.0))
        nesterov = bool(opt_cfg.get("nesterov", False))
        optimizer = SGD(
            params,
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            dampening=dampening,
            nesterov=nesterov,
        )
        return optimizer, kind

    if kind == "rmsprop":
        momentum = float(opt_cfg.get("momentum", 0.0))
        alpha = float(opt_cfg.get("alpha", 0.99))
        eps = float(opt_cfg.get("eps", 1e-8))
        centered = bool(opt_cfg.get("centered", False))
        optimizer = RMSprop(
            params,
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            alpha=alpha,
            eps=eps,
            centered=centered,
        )
        return optimizer, kind

    raise ValueError("optimizer.type must be one of: adamw, adam, sgd, rmsprop")


def build_scheduler(optimizer: torch.optim.Optimizer, config: dict):
    """Create LR scheduler from config.

    Returns:
        scheduler, scheduler_kind, plateau_monitor
    """
    config = resolve_config(config)
    train_cfg = config.get("training", {}) or {}
    sched_cfg = train_cfg.get("scheduler", {}) or {}
    kind = str(sched_cfg.get("type", "step")).lower()

    if kind in {"none", "off", "disabled"}:
        return None, "none", None

    if kind == "step":
        step_cfg = sched_cfg.get("step", {}) or {}
        scheduler = StepLR(
            optimizer,
            step_size=int(step_cfg.get("step_size", sched_cfg.get("step_size", 10))),
            gamma=float(step_cfg.get("gamma", sched_cfg.get("gamma", 0.5))),
            last_epoch=int(step_cfg.get("last_epoch", -1)),
        )
        return scheduler, kind, None

    if kind == "cosine":
        cosine_cfg = sched_cfg.get("cosine", {}) or {}
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=int(cosine_cfg.get("t_max", sched_cfg.get("t_max", train_cfg.get("epochs", 50)))),
            eta_min=float(cosine_cfg.get("min_lr", sched_cfg.get("min_lr", 1e-6))),
            last_epoch=int(cosine_cfg.get("last_epoch", -1)),
        )
        return scheduler, kind, None

    if kind == "plateau":
        plateau_cfg = sched_cfg.get("plateau", {}) or {}
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode=str(plateau_cfg.get("mode", "min")),
            factor=float(plateau_cfg.get("factor", sched_cfg.get("plateau_factor", 0.5))),
            patience=int(plateau_cfg.get("patience", sched_cfg.get("plateau_patience", 5))),
            threshold=float(plateau_cfg.get("threshold", 1e-4)),
            threshold_mode=str(plateau_cfg.get("threshold_mode", "rel")),
            cooldown=int(plateau_cfg.get("cooldown", 0)),
            min_lr=float(plateau_cfg.get("min_lr", sched_cfg.get("min_lr", 1e-6))),
            eps=float(plateau_cfg.get("eps", 1e-8)),
        )
        monitor = str(plateau_cfg.get("monitor", "val_loss"))
        if monitor not in {"val_loss", "train_loss"}:
            raise ValueError(
                "training.scheduler.plateau.monitor must be 'val_loss' or 'train_loss'"
            )
        return scheduler, kind, monitor

    raise ValueError("scheduler.type must be one of: none, step, cosine, plateau")


def run_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    grad_clip_norm: float,
    pcgrad_optimizer=None,
    max_batches: int | None = None,
    point_centric_dir: str | None = None,
    transfer_representation: str = "legacy",
    residual_cfg: dict | None = None,
    transfer_tp_min: float = 0.5,
    transfer_tp_max: float = 30.0,
    epoch_index: int | None = None,
    total_epochs: int | None = None,
    progress_cfg: Dict[str, object] | None = None,
    progress_phase: str = "train",
    breaking_enabled: bool = False,
    breaking_cap_key: str = "local_breaking_hs_cap",
    breaking_valid_key: str = "local_breaking_cap_valid",
    breaking_penalty_weight: float = 0.02,
    batch_observer=None,
) -> Tuple[float, Dict[str, float], Dict[str, float]]:
    """Run one epoch. If optimizer is None, runs in eval mode."""
    train_mode = optimizer is not None
    if train_mode:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    n_samples = 0
    component_totals: Dict[str, float] = {}
    all_pred = []
    all_true = []
    all_transfer_pred = []
    all_transfer_true = []
    target_scaler_stats = (
        _load_target_scaler_stats(point_centric_dir) if point_centric_dir else None
    )
    transfer_scaler_stats = (
        load_transfer_scaler_stats(point_centric_dir) if point_centric_dir else {}
    )
    progress_settings = progress_cfg or {}
    progress_enabled = _should_enable_progress_bar(progress_settings, progress_phase)
    progress_desc = (
        f"{progress_phase.title()} Epoch {int(epoch_index or 0):03d}/{int(total_epochs or 0):03d}"
        if epoch_index is not None and total_epochs is not None
        else progress_phase.title()
    )
    progress_total = None
    if hasattr(loader, "__len__"):
        progress_total = len(loader)
        if max_batches is not None:
            progress_total = min(progress_total, max_batches)

    iterator = maybe_tqdm(
        loader,
        enabled=progress_enabled,
        desc=progress_desc,
        leave=bool(progress_settings.get("leave", False)),
        total=progress_total,
    )

    try:
        for batch_idx, batch in enumerate(iterator):
            if max_batches is not None and batch_idx >= max_batches:
                break

            model_inputs = _extract_model_inputs(
                batch,
                device,
                use_static_features=bool(getattr(model, "use_static_features", True)),
                use_source_geometry_features=bool(
                    getattr(model, "use_source_geometry_features", True)
                ),
            )
            x_dynamic = model_inputs["x_dynamic"]
            x_dynamic_sources = model_inputs["x_dynamic_sources"]
            source_geometry = model_inputs["source_geometry"]
            x_static = model_inputs["x_static"]
            x_bathy = model_inputs["x_bathy"]
            y = _to_device_batch(batch["y"], device)

            if batch_observer is not None:
                batch_observer(
                    batch=batch,
                    model_inputs=model_inputs,
                    target=y,
                    batch_idx=batch_idx,
                    split_name=progress_phase,
                )

            if train_mode:
                optimizer.zero_grad(set_to_none=True)

            with torch.set_grad_enabled(train_mode):
                pred = model(
                    x_dynamic,
                    x_static,
                    x_bathy=x_bathy,
                    x_dynamic_sources=x_dynamic_sources,
                    source_geometry=source_geometry,
                )
                if not _all_finite(pred):
                    raise RuntimeError("Non-finite model outputs detected in run_epoch")

                breaking_loss = None
                breaking_penalty = None
                if bool(breaking_enabled):
                    breaking_loss = _compute_breaking_penalty(
                        pred=pred,
                        target=y,
                        batch=batch,
                        cap_key=str(breaking_cap_key),
                        valid_key=str(breaking_valid_key),
                        target_scaler_stats=target_scaler_stats,
                        transfer_scaler_stats=transfer_scaler_stats,
                        transfer_representation=transfer_representation,
                    )
                    if breaking_loss is not None and float(breaking_penalty_weight) > 0.0:
                        breaking_penalty = float(breaking_penalty_weight) * breaking_loss

                if train_mode:
                    if pcgrad_optimizer is not None:
                        if not hasattr(loss_fn, "task_losses"):
                            raise ValueError(
                                "PCGrad requires a loss module exposing task_losses(pred, target)."
                            )

                        task_per_sample_losses = None
                        if hasattr(loss_fn, "task_loss_details"):
                            task_losses, task_per_sample_losses = loss_fn.task_loss_details(pred, y)
                        else:
                            task_losses = loss_fn.task_losses(pred, y)
                        if not isinstance(task_losses, dict) or not task_losses:
                            raise ValueError(
                                "loss_fn.task_losses must return a non-empty dict of scalar losses"
                            )

                        weighted_task_losses = _weight_pcgrad_task_losses(loss_fn, task_losses)
                        objectives = []
                        for name, task_loss in weighted_task_losses.items():
                            if not torch.isfinite(task_loss):
                                raise RuntimeError(f"Non-finite task loss detected for '{name}'")
                            objectives.append(task_loss)

                        if breaking_penalty is not None:
                            if not torch.isfinite(breaking_penalty):
                                raise RuntimeError("Non-finite breaking penalty detected")
                            objectives.append(breaking_penalty)

                        pcgrad_optimizer.pc_backward(objectives)
                        if grad_clip_norm > 0:
                            torch.nn.utils.clip_grad_norm_(
                                model.parameters(), max_norm=grad_clip_norm
                            )
                        pcgrad_optimizer.step()
                        loss = _combine_pcgrad_task_losses(
                            loss_fn,
                            task_losses,
                            breaking_penalty=breaking_penalty,
                        )
                        if hasattr(loss_fn, "record_task_loss_components"):
                            loss_fn.record_task_loss_components(
                                task_losses,
                                per_sample_losses=task_per_sample_losses,
                                total_override=loss,
                            )
                    else:
                        loss = loss_fn(pred, y)
                        if breaking_penalty is not None:
                            loss = loss + breaking_penalty
                        if not torch.isfinite(loss):
                            bad_x = (
                                int((~torch.isfinite(x_dynamic)).sum().item())
                                if x_dynamic is not None
                                else 0
                            )
                            bad_xs = (
                                int((~torch.isfinite(x_dynamic_sources)).sum().item())
                                if x_dynamic_sources is not None
                                else 0
                            )
                            bad_g = (
                                int((~torch.isfinite(source_geometry)).sum().item())
                                if source_geometry is not None
                                else 0
                            )
                            bad_s = (
                                int((~torch.isfinite(x_static)).sum().item())
                                if x_static is not None
                                else 0
                            )
                            bad_b = (
                                int((~torch.isfinite(x_bathy)).sum().item())
                                if x_bathy is not None
                                else 0
                            )
                            bad_y = _count_non_finite(y)
                            bad_p = _count_non_finite(pred)
                            raise RuntimeError(
                                "Non-finite loss detected. "
                                f"bad(x_dynamic)={bad_x}, bad(x_dynamic_sources)={bad_xs}, "
                                f"bad(source_geometry)={bad_g}, bad(x_static)={bad_s}, bad(x_bathy)={bad_b}, "
                                f"bad(y)={bad_y}, bad(pred)={bad_p}"
                            )

                        loss.backward()
                        if grad_clip_norm > 0:
                            torch.nn.utils.clip_grad_norm_(
                                model.parameters(), max_norm=grad_clip_norm
                            )
                        optimizer.step()
                else:
                    loss = loss_fn(pred, y)
                    if breaking_penalty is not None:
                        loss = loss + breaking_penalty
                    if not torch.isfinite(loss):
                        bad_x = (
                            int((~torch.isfinite(x_dynamic)).sum().item())
                            if x_dynamic is not None
                            else 0
                        )
                        bad_xs = (
                            int((~torch.isfinite(x_dynamic_sources)).sum().item())
                            if x_dynamic_sources is not None
                            else 0
                        )
                        bad_g = (
                            int((~torch.isfinite(source_geometry)).sum().item())
                            if source_geometry is not None
                            else 0
                        )
                        bad_s = (
                            int((~torch.isfinite(x_static)).sum().item())
                            if x_static is not None
                            else 0
                        )
                        bad_b = (
                            int((~torch.isfinite(x_bathy)).sum().item())
                            if x_bathy is not None
                            else 0
                        )
                        bad_y = _count_non_finite(y)
                        bad_p = _count_non_finite(pred)
                        raise RuntimeError(
                            "Non-finite validation/eval loss detected. "
                            f"bad(x_dynamic)={bad_x}, bad(x_dynamic_sources)={bad_xs}, "
                            f"bad(source_geometry)={bad_g}, bad(x_static)={bad_s}, bad(x_bathy)={bad_b}, "
                            f"bad(y)={bad_y}, bad(pred)={bad_p}"
                        )

            if isinstance(y, dict):
                batch_size = int(next(iter(y.values())).shape[0])
            else:
                batch_size = int(y.size(0))
            total_loss += float(loss.detach().cpu()) * batch_size
            n_samples += batch_size
            batch_components = _extract_loss_components(
                loss_fn, total_override=float(loss.detach().cpu())
            )
            if breaking_loss is not None:
                batch_components["breaking_loss"] = float(breaking_loss.detach().cpu())
            if breaking_penalty is not None:
                batch_components["breaking_penalty"] = float(breaking_penalty.detach().cpu())
            _accumulate_component_totals(component_totals, batch_components, batch_size)

            if progress_enabled and hasattr(iterator, "set_postfix"):
                update_interval = int(progress_settings.get("update_interval", 1))
                if (batch_idx + 1) % update_interval == 0 or batch_idx == 0:
                    postfix = {
                        "loss": f"{float(loss.detach().cpu()):.4f}",
                        "batch": f"{batch_idx + 1}",
                    }
                    if train_mode and optimizer is not None and optimizer.param_groups:
                        postfix["lr"] = f"{optimizer.param_groups[0]['lr']:.2e}"
                    for key in ("hs_loss", "tp_loss", "dir_loss", "dp_loss"):
                        if key in batch_components:
                            postfix[key.replace("_loss", "")] = f"{batch_components[key]:.4f}"
                    if "breaking_penalty" in batch_components:
                        postfix["break"] = f"{batch_components['breaking_penalty']:.4f}"
                    iterator.set_postfix(postfix, refresh=False)

            if isinstance(pred, dict) and isinstance(y, dict):
                if (
                    "transfer" in y
                    and "reference" in y
                    and "physical" in y
                    and "log_hs_ratio" in pred
                ):
                    physical_pred, physical_true, transfer_pred, transfer_true = (
                        recover_transfer_metric_arrays(
                            pred,
                            y,
                            transfer_scaler_stats,
                            transfer_representation=transfer_representation,
                            residual_cfg=residual_cfg,
                            tp_min=transfer_tp_min,
                            tp_max=transfer_tp_max,
                        )
                    )
                    all_pred.append(physical_matrix_to_metric_channels(physical_pred))
                    all_true.append(physical_matrix_to_metric_channels(physical_true))
                    all_transfer_pred.append(transfer_pred)
                    all_transfer_true.append(transfer_true)
                else:
                    pred_arr, true_arr = _recover_hybrid_metrics_arrays(
                        pred, y, target_scaler_stats
                    )
                    all_pred.append(pred_arr)
                    all_true.append(true_arr)
            else:
                all_pred.append(pred.detach().cpu().numpy())
                all_true.append(y.detach().cpu().numpy())
    finally:
        if progress_enabled and hasattr(iterator, "close"):
            iterator.close()

    if n_samples == 0:
        raise RuntimeError("DataLoader produced zero samples for this epoch")

    epoch_loss = total_loss / n_samples
    y_pred = np.concatenate(all_pred, axis=0)
    y_true = np.concatenate(all_true, axis=0)
    metrics = compute_wave_metrics(y_true=y_true, y_pred=y_pred)
    if all_transfer_pred and all_transfer_true:
        transfer_pred = np.concatenate(all_transfer_pred, axis=0)
        transfer_true = np.concatenate(all_transfer_true, axis=0)
        metrics.update(_compute_transfer_metrics(transfer_true, transfer_pred))
    components = _finalize_component_means(
        component_totals, n_samples=n_samples, epoch_loss=epoch_loss
    )

    if not np.isfinite(epoch_loss):
        raise RuntimeError("Non-finite epoch loss detected after aggregation")

    return epoch_loss, metrics, components


def format_metric_block(prefix: str, metrics: Dict[str, float]) -> str:
    return (
        f"{prefix} "
        f"Hs(R2={metrics['hs_r2']:.4f}, RMSE={metrics['hs_rmse']:.4f}) | "
        f"Tp(R2={metrics['tp_r2']:.4f}, RMSE={metrics['tp_rmse']:.4f}) | "
        f"Dir(R2={metrics['dir_r2']:.4f}, RMSEdeg={metrics['dir_rmse_deg']:.4f}) | "
        f"Dp(R2={metrics['dp_r2']:.4f}, RMSEdeg={metrics['dp_rmse_deg']:.4f})"
    )


def _to_float_dict(values: Dict[str, float] | None) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for key, value in (values or {}).items():
        out[str(key)] = float(value)
    return out


def _extract_loss_components(
    loss_fn: nn.Module,
    total_override: float | torch.Tensor | None = None,
) -> Dict[str, float]:
    raw = getattr(loss_fn, "last_components", None)
    components = _to_float_dict(raw if isinstance(raw, dict) else {})
    if total_override is not None:
        components["total_loss"] = float(total_override)
    return components


def _combine_pcgrad_task_losses(
    loss_fn: nn.Module,
    task_losses: Dict[str, torch.Tensor],
    *,
    breaking_penalty: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reconstruct the scalar objective for logging when PCGrad is active.

    PCGrad needs the per-task objectives separately for gradient projection, but
    epoch loss reporting should stay on the same scale as the loss module's
    normal `forward()` path so train/val losses remain directly comparable.
    """
    if not task_losses:
        raise ValueError("PCGrad loss reconstruction requires at least one task loss")

    task_weights = getattr(loss_fn, "task_weights", None)
    total = None
    for name, task_loss in task_losses.items():
        weight = float(task_weights.get(name, 1.0)) if isinstance(task_weights, dict) else 1.0
        weighted = weight * task_loss
        total = weighted if total is None else total + weighted

    assert total is not None
    if breaking_penalty is not None:
        total = total + breaking_penalty
    return total


def _weight_pcgrad_task_losses(
    loss_fn: nn.Module,
    task_losses: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Apply configured task weights before PCGrad backward.

    Without this, PCGrad optimizes a different objective from the validation
    loss whenever the active loss module uses non-unit task weights.
    """
    if not task_losses:
        raise ValueError("PCGrad task weighting requires at least one task loss")

    task_weights = getattr(loss_fn, "task_weights", None)
    if not isinstance(task_weights, dict):
        return dict(task_losses)

    weighted: Dict[str, torch.Tensor] = {}
    for name, task_loss in task_losses.items():
        weight = float(task_weights.get(name, 1.0))
        weighted[name] = weight * task_loss
    return weighted


def _accumulate_component_totals(
    totals: Dict[str, float],
    components: Dict[str, float],
    batch_size: int,
) -> None:
    for key, value in components.items():
        totals[key] = totals.get(key, 0.0) + (float(value) * batch_size)


def _finalize_component_means(
    totals: Dict[str, float],
    n_samples: int,
    epoch_loss: float,
) -> Dict[str, float]:
    if n_samples <= 0:
        return {"total_loss": float(epoch_loss)}

    out = {key: float(value) / float(n_samples) for key, value in totals.items()}
    out["total_loss"] = float(epoch_loss)
    return out


def _format_component_line(label: str, components: Dict[str, float]) -> str:
    parts = []
    for name in COMPONENT_NAMES:
        key = f"{name}_loss"
        if key in components:
            parts.append(f"{name}={components[key]:.4f}")

    if not parts and "total_loss" in components:
        parts.append(f"total={components['total_loss']:.4f}")
    if "breaking_penalty" in components:
        parts.append(f"breaking={components['breaking_penalty']:.4f}")

    return f"  {label:<5}{' '.join(parts)}".rstrip()


def compute_physical_score(metrics: Dict[str, float], weights: Dict[str, float]) -> float:
    missing = [key for key in weights if key not in metrics]
    if missing:
        raise ValueError(
            "training.selection.weights contains metric(s) not present in validation metrics: "
            f"{sorted(missing)}. Available keys: {sorted(metrics)}"
        )

    score = 0.0
    for key, weight in weights.items():
        score += float(weight) * float(metrics[key])
    return float(score)


def resolve_selection_config(config: dict) -> Tuple[str, str, Dict[str, float]]:
    train_cfg = config.get("training", {}) or {}
    selection_cfg = train_cfg.get("selection", {}) or {}

    monitor = str(selection_cfg.get("monitor", "val_loss")).strip().lower()
    if monitor not in {"val_loss", "physical_score"}:
        raise ValueError("training.selection.monitor must be 'val_loss' or 'physical_score'")

    mode = str(selection_cfg.get("mode", "min")).strip().lower()
    if mode not in {"min", "max"}:
        raise ValueError("training.selection.mode must be 'min' or 'max'")

    raw_weights = selection_cfg.get("weights", None)
    if raw_weights is None or raw_weights == {}:
        raw_weights = DEFAULT_PHYSICAL_SCORE_WEIGHTS
    if not isinstance(raw_weights, dict):
        raise ValueError("training.selection.weights must be a mapping of metric_name -> weight")

    weights = {str(key): float(value) for key, value in raw_weights.items()}
    unsupported = sorted(set(weights) - SUPPORTED_PHYSICAL_SCORE_METRICS)
    if unsupported:
        raise ValueError(
            "training.selection.weights contains unsupported metric(s): "
            f"{unsupported}. Supported keys: {sorted(SUPPORTED_PHYSICAL_SCORE_METRICS)}"
        )

    return monitor, mode, weights


def _is_improved(candidate: float, best: float, mode: str, min_delta: float = 0.0) -> bool:
    if mode == "max":
        return (candidate - best) > min_delta
    return (best - candidate) > min_delta


def resolve_validation_every_n_epochs(config: dict) -> int:
    train_cfg = config.get("training", {}) or {}
    value = int(train_cfg.get("validation_every_n_epochs", 1))
    if value < 1:
        raise ValueError("training.validation_every_n_epochs must be >= 1")
    return value


def should_run_validation_epoch(
    epoch: int, total_epochs: int, validation_every_n_epochs: int
) -> bool:
    every_n = int(validation_every_n_epochs)
    if every_n < 1:
        raise ValueError("validation_every_n_epochs must be >= 1")
    if int(epoch) >= int(total_epochs):
        return True
    return int(epoch) % every_n == 0


def initialize_model_from_config_checkpoint(
    *,
    model: nn.Module,
    config: dict,
    device: torch.device,
) -> dict[str, Any] | None:
    train_cfg = config.get("training", {}) or {}
    init_cfg = train_cfg.get("initialization", {}) or {}
    checkpoint_path_raw = init_cfg.get("checkpoint_path", None)
    if checkpoint_path_raw in {None, ""}:
        return None

    checkpoint_path = Path(str(checkpoint_path_raw)).expanduser().resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Initialization checkpoint not found: {checkpoint_path}")

    checkpoint = _load_checkpoint_compat(checkpoint_path, device)
    state_dict = _extract_state_dict(checkpoint)
    load_mode, load_details = _load_state_dict_best_effort(model=model, state_dict=state_dict)
    return {
        "checkpoint_path": str(checkpoint_path),
        "load_mode": str(load_mode),
        "load_details": dict(load_details),
        "strict_requested": bool(init_cfg.get("strict", False)),
        "ignore_optimizer": bool(init_cfg.get("ignore_optimizer", True)),
    }


def export_target_metrics_summary_csv(
    *,
    config: dict,
    model: nn.Module,
    device: torch.device,
    val_loader,
    val_ds,
    test_loader,
    test_ds,
    out_dir: Path,
) -> Path | None:
    """Export per-target validation/test site metrics after training completes."""
    data_cfg = config.get("data", {}) or {}
    targets_cfg = resolve_targets_config(data_cfg)

    rows: list[dict[str, object]] = []
    split_payloads = [
        ("val", val_loader, val_ds, "ALL_VALIDATION_SITES", False),
        ("test", test_loader, test_ds, "ALL_TEST_SITES", True),
    ]

    for (
        split_name,
        loader,
        dataset,
        aggregate_site_label,
        include_individual_sites,
    ) in split_payloads:
        if dataset is None or len(dataset) == 0:
            print(
                f"Skipping target-metrics CSV export for split '{split_name}' because it has no samples."
            )
            continue

        raw_outputs = infer(
            model=model,
            loader=loader,
            device=device,
            point_centric_dir=data_cfg.get("point_centric_dir", ""),
            transfer_representation=str(targets_cfg.get("transfer_representation", "legacy")),
            residual_cfg=targets_cfg.get("residual_correction", {}) or {},
            transfer_tp_min=float(targets_cfg.get("tp_min", 0.5)),
            transfer_tp_max=float(targets_cfg.get("tp_max", 30.0)),
        )
        prepared = prepare_evaluation_outputs(
            config=config,
            outputs=raw_outputs,
            split_for_logging=f"{split_name} (best checkpoint)",
            log_site_diagnostics=bool(split_name == "test"),
        )
        site_rows = compute_site_target_metric_rows(
            prepared["y_true"],
            prepared["y_pred"],
            raw_outputs["site"],
            split=split_name,
        )
        rows.extend(
            compute_split_target_metric_rows(
                prepared["y_true"],
                prepared["y_pred"],
                raw_outputs["site"],
                split=split_name,
                aggregation="split_aggregate",
                site=aggregate_site_label,
            )
        )
        if include_individual_sites:
            rows.extend(site_rows)

    if not rows:
        return None

    csv_path = out_dir / "target_metrics_summary.csv"
    fieldnames = [
        "split",
        "aggregation",
        "site",
        "target",
        "mse",
        "rmse",
        "bias",
        "pearson_r",
        "r2",
        "mse_unit",
        "rmse_unit",
        "bias_unit",
        "sample_count",
        "site_count",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return csv_path


def _infer_model_build_kwargs_from_datasets(
    *,
    arrays,
    datasets: Sequence[object],
) -> dict[str, Any]:
    for dataset in datasets:
        if dataset is None or len(dataset) == 0:
            continue
        sample = dataset[0]
        output_dim = 4 if isinstance(sample["y"], dict) else int(sample["y"].shape[-1])
        return {
            "dynamic_input_dim": int(sample["x_dynamic"].shape[-1]) if "x_dynamic" in sample else 0,
            "static_input_dim": int(sample["x_static"].shape[-1]) if "x_static" in sample else 0,
            "output_dim": output_dim,
            "dynamic_feature_names": getattr(arrays, "dynamic_feature_names", None),
            "source_dynamic_input_dim": int(sample["x_dynamic_sources"].shape[-1])
            if "x_dynamic_sources" in sample
            else None,
            "source_geometry_input_dim": int(sample["source_geometry"].shape[-1])
            if "source_geometry" in sample
            else None,
            "source_feature_names": getattr(arrays, "source_feature_names", None),
        }
    raise RuntimeError(
        "Could not infer model input/output dimensions because all datasets were empty."
    )


def _write_prediction_artifacts(
    *,
    config: dict,
    out_dir: Path,
    split_name: str,
    outputs: Dict[str, Any],
    prepared: dict[str, object],
    checkpoint_label: str,
) -> tuple[Path, Path | None]:
    output_columns = [str(col) for col in (prepared.get("output_columns") or [])]
    y_true = np.asarray(prepared["y_true"], dtype=np.float64)
    y_pred = np.asarray(prepared["y_pred"], dtype=np.float64)
    transfer_true = prepared.get("transfer_true")
    transfer_pred = prepared.get("transfer_pred")

    records = {
        "split": [str(split_name)] * len(outputs["site"]),
        "site": [str(site) for site in outputs["site"]],
        "time_index": np.asarray(outputs["time_index"], dtype=int),
        "timestamp": [str(ts) for ts in outputs["timestamp"]],
    }
    for idx, col in enumerate(output_columns):
        records[f"target_{col}"] = y_true[:, idx]
        records[f"pred_{col}"] = y_pred[:, idx]

    if (
        isinstance(transfer_true, np.ndarray)
        and transfer_true.size > 0
        and isinstance(transfer_pred, np.ndarray)
        and transfer_pred.size > 0
    ):
        transfer_names = ["log_hs_ratio", "tp_delta", "dir_delta_deg", "dp_delta_deg"]
        for idx, col in enumerate(transfer_names):
            records[f"target_transfer_{col}"] = transfer_true[:, idx]
            records[f"pred_transfer_{col}"] = transfer_pred[:, idx]

    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records)
    csv_path = out_dir / f"predictions_{split_name}.csv"
    df.to_csv(csv_path, index=False)

    nc_path: Path | None = None
    if bool((config.get("save", {}) or {}).get("save_predictions_nc", True)):
        try:
            import xarray as xr

            data_vars = {
                **{
                    f"target_{col}": ("sample", y_true[:, idx])
                    for idx, col in enumerate(output_columns)
                },
                **{
                    f"pred_{col}": ("sample", y_pred[:, idx])
                    for idx, col in enumerate(output_columns)
                },
            }
            ds_nc = xr.Dataset(
                data_vars=data_vars,
                coords={
                    "sample": np.arange(len(df), dtype=int),
                    "site": ("sample", np.asarray(outputs["site"], dtype=str)),
                    "timestamp": ("sample", np.asarray(outputs["timestamp"], dtype=str)),
                    "time_index": ("sample", np.asarray(outputs["time_index"], dtype=int)),
                },
                attrs={"split": str(split_name), "checkpoint": str(checkpoint_label)},
            )
            nc_path = out_dir / f"predictions_{split_name}.nc"
            ds_nc.to_netcdf(nc_path)
        except Exception as exc:
            print(f"Skipping NetCDF export for split '{split_name}': {exc}")

    return csv_path, nc_path


def run_independent_per_target_training(
    *,
    config_path: str,
    cfg: dict,
    device: torch.device,
    device_arg: str,
    joint_initialization_checkpoint_path: str | Path | None = None,
    target_modeling_label: str = "independent_per_target",
    joint_stage_entry: dict[str, Any] | None = None,
) -> int:
    out_dir = (
        Path((cfg.get("logging", {}) or {}).get("output_dir", "results/independent_target_run"))
        .expanduser()
        .resolve()
    )
    child_root = out_dir / "child_models"
    child_config_dir = child_root / "configs"
    child_config_dir.mkdir(parents=True, exist_ok=True)

    config_stem = Path(config_path).stem
    child_entries: dict[str, dict[str, str]] = {}
    child_histories: dict[str, list[dict[str, Any]]] = {}
    training_started_at_utc = _utc_now_iso()
    parent_start = time.perf_counter()

    child_stage = "fine_tune_from_joint" if joint_initialization_checkpoint_path else "from_scratch"
    print("=== Independent per-target training mode ===")
    print(f"Parent config: {Path(config_path).resolve()}")
    print(f"Parent results dir: {out_dir}")
    if joint_initialization_checkpoint_path:
        print(
            f"Joint initialization checkpoint: {Path(joint_initialization_checkpoint_path).resolve()}"
        )
    for target_name in TARGET_NAMES:
        child_run_dir = (child_root / target_name).resolve()
        child_checkpoint_name = f"cp_{config_stem}_{target_name}.pt"
        child_cfg = build_independent_target_child_config(
            cfg,
            target_name=target_name,
            child_output_dir=child_run_dir,
            child_checkpoint_name=child_checkpoint_name,
            init_checkpoint_path=joint_initialization_checkpoint_path,
            child_stage=child_stage,
        )
        child_cfg_path = (child_config_dir / f"{config_stem}_{target_name}.yaml").resolve()
        with child_cfg_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(_json_safe(child_cfg), handle, sort_keys=False)

        print(f"[independent-target] training child target='{target_name}' from {child_cfg_path}")
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--config",
            str(child_cfg_path),
            "--device",
            str(device_arg),
        ]
        completed = subprocess.run(command, check=False)
        if completed.returncode != 0:
            raise RuntimeError(
                f"Independent per-target child training failed for target '{target_name}' "
                f"with exit code {completed.returncode}."
            )

        child_metadata_path = child_run_dir / "training_run_metadata.json"
        if not child_metadata_path.exists():
            raise FileNotFoundError(
                f"Missing child metadata after training target '{target_name}': {child_metadata_path}"
            )
        child_metadata = _read_json(child_metadata_path)
        child_checkpoint_path = str(
            ((child_metadata.get("runtime", {}) or {}).get("checkpoint_path", "")) or ""
        ).strip()
        if not child_checkpoint_path:
            raise ValueError(
                f"Child metadata for target '{target_name}' is missing runtime.checkpoint_path."
            )
        history_path = child_run_dir / "train_history.json"
        if history_path.exists():
            payload = _read_json(history_path)
            child_histories[target_name] = payload if isinstance(payload, list) else []
        else:
            child_histories[target_name] = []
        child_entries[target_name] = {
            "run_dir": str(child_run_dir),
            "config_path": str(child_cfg_path),
            "metadata_path": str(child_metadata_path),
            "checkpoint_path": str(Path(child_checkpoint_path).resolve()),
        }

    data_cfg = cfg.get("data", {}) or {}
    targets_cfg = resolve_targets_config(data_cfg)
    requested_bathy_channels, requested_bathy_in_channels = _resolve_runtime_bathy_request(cfg)
    arrays = load_point_centric_arrays(
        data_cfg.get("point_centric_dir", "data/processed/point_centric_demo"),
        bathy_channels=requested_bathy_channels,
        bathy_in_channels=requested_bathy_in_channels,
    )
    arrays = apply_runtime_static_ablation(arrays, resolve_static_ablation_config_path(cfg))
    val_loader, val_ds = build_split_dataloader(arrays, cfg, split_name="val", shuffle=False)
    test_loader, test_ds = build_split_dataloader(arrays, cfg, split_name="test", shuffle=False)
    build_kwargs = _infer_model_build_kwargs_from_datasets(
        arrays=arrays, datasets=[val_ds, test_ds]
    )

    def _build_model():
        return build_model_from_config(config=cfg, **build_kwargs).to(device)

    composite_training_metadata = {
        "runtime": {
            "checkpoint_path": child_entries["hs"]["checkpoint_path"],
            "independent_target_composite": {
                "enabled": True,
                "targets": child_entries,
            },
        }
    }
    composite_model = load_model_from_training_metadata(
        training_metadata=composite_training_metadata,
        build_model=_build_model,
        checkpoint_loader=lambda path: torch.load(path, map_location=device),
        state_dict_getter=lambda checkpoint: (
            checkpoint["model_state_dict"]
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
            else checkpoint
        ),
        state_dict_loader=lambda model, state_dict: model.load_state_dict(state_dict),
    ).to(device)
    composite_model.eval()

    prediction_artifacts: list[Path] = []
    for split_name, loader, dataset in (
        ("val", val_loader, val_ds),
        ("test", test_loader, test_ds),
    ):
        if dataset is None or len(dataset) == 0:
            continue
        raw_outputs = infer(
            model=composite_model,
            loader=loader,
            device=device,
            point_centric_dir=data_cfg.get("point_centric_dir", ""),
            transfer_representation=str(targets_cfg.get("transfer_representation", "legacy")),
            residual_cfg=targets_cfg.get("residual_correction", {}) or {},
            transfer_tp_min=float(targets_cfg.get("tp_min", 0.5)),
            transfer_tp_max=float(targets_cfg.get("tp_max", 30.0)),
        )
        prepared = prepare_evaluation_outputs(
            config=cfg,
            outputs=raw_outputs,
            split_for_logging=f"{split_name} (independent_target_composite)",
            log_site_diagnostics=bool(split_name == "test"),
        )
        csv_path, _nc_path = _write_prediction_artifacts(
            config=cfg,
            out_dir=out_dir,
            split_name=split_name,
            outputs=raw_outputs,
            prepared=prepared,
            checkpoint_label="independent_target_composite",
        )
        prediction_artifacts.append(csv_path)

    metrics_summary_csv_path = export_target_metrics_summary_csv(
        config=cfg,
        model=composite_model,
        device=device,
        val_loader=val_loader,
        val_ds=val_ds,
        test_loader=test_loader,
        test_ds=test_ds,
        out_dir=out_dir,
    )

    template_target = TARGET_NAMES[0]
    template_metadata = _read_json(Path(child_entries[template_target]["metadata_path"]))
    template_observed = _read_json(
        Path(child_entries[template_target]["run_dir"]) / "observed_model_io.json"
    )

    runtime_meta = copy.deepcopy(template_metadata.get("runtime", {}) or {})
    runtime_meta["config_path"] = str(Path(config_path).resolve())
    runtime_meta["checkpoint_path"] = child_entries[template_target]["checkpoint_path"]
    runtime_meta["target_modeling"] = str(target_modeling_label)
    runtime_meta["training_started_at_utc"] = training_started_at_utc
    runtime_meta["training_finished_at_utc"] = _utc_now_iso()
    runtime_meta["training_duration_seconds"] = float(time.perf_counter() - parent_start)
    runtime_meta["independent_target_composite"] = {
        "enabled": True,
        "targets": copy.deepcopy(child_entries),
    }
    if joint_stage_entry is not None:
        runtime_meta["joint_stage"] = copy.deepcopy(joint_stage_entry)

    parent_metadata = copy.deepcopy(template_metadata)
    parent_metadata["config"] = copy.deepcopy(cfg)
    parent_metadata["runtime"] = runtime_meta
    parent_metadata.setdefault("model", {})
    parent_metadata["model"]["target_modeling"] = str(target_modeling_label)
    parent_metadata["model"]["child_targets"] = list(TARGET_NAMES)
    if joint_stage_entry is not None:
        parent_metadata["model"]["joint_stage"] = copy.deepcopy(joint_stage_entry)

    observed_manifest = copy.deepcopy(template_observed)
    observed_manifest.setdefault("model_flags", {})
    observed_manifest["model_flags"]["model_class"] = "IndependentTargetCompositeModel"
    observed_manifest["model_flags"]["target_modeling"] = str(target_modeling_label)
    observed_manifest["model_flags"]["child_targets"] = list(TARGET_NAMES)
    if joint_stage_entry is not None:
        observed_manifest["model_flags"]["joint_stage_checkpoint"] = str(
            joint_stage_entry.get("checkpoint_path", "")
        )

    run_metadata_json_path, run_metadata_pickle_path = _write_serialized_artifact(
        out_dir,
        "training_run_metadata",
        parent_metadata,
    )
    observed_io_json_path, observed_io_pickle_path = _write_serialized_artifact(
        out_dir,
        "observed_model_io",
        observed_manifest,
    )

    aggregated_history = aggregate_child_histories(child_histories)
    hist_path = out_dir / "train_history.json"
    with hist_path.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(aggregated_history), handle, indent=2)

    print(f"Independent per-target training finished. Parent metadata: {run_metadata_json_path}")
    print(f"Parent metadata pickle saved to: {run_metadata_pickle_path}")
    print(f"Parent observed model I/O: {observed_io_json_path}")
    print(f"Parent observed model I/O pickle saved to: {observed_io_pickle_path}")
    print(f"Aggregated history saved to: {hist_path}")
    for artifact in prediction_artifacts:
        print(f"Prediction artifact saved to: {artifact}")
    if metrics_summary_csv_path is not None:
        print(f"Target metrics summary CSV saved to: {metrics_summary_csv_path}")
    return 0


def run_joint_then_independent_finetune_training(
    *,
    config_path: str,
    cfg: dict,
    device: torch.device,
    device_arg: str,
) -> int:
    out_dir = (
        Path(
            (cfg.get("logging", {}) or {}).get(
                "output_dir", "results/joint_then_independent_finetune"
            )
        )
        .expanduser()
        .resolve()
    )
    joint_dir = out_dir / "joint_model"
    joint_cfg = copy.deepcopy(cfg)
    joint_cfg.setdefault("training", {})
    joint_cfg["training"]["target_modeling"] = "joint"
    joint_cfg["training"].setdefault("initialization", {})
    joint_cfg["training"]["initialization"]["checkpoint_path"] = None
    joint_cfg["logging"] = copy.deepcopy(cfg.get("logging", {}) or {})
    joint_cfg["logging"]["output_dir"] = str(joint_dir)
    parent_ckpt_name = str(
        (cfg.get("logging", {}) or {}).get("checkpoint_name", "coastal_transformer_best.pt")
    )
    joint_cfg["logging"]["checkpoint_name"] = f"joint_{parent_ckpt_name}"

    joint_cfg_path = joint_dir / f"{Path(config_path).stem}_joint_stage.yaml"
    joint_dir.mkdir(parents=True, exist_ok=True)
    with joint_cfg_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(_json_safe(joint_cfg), handle, sort_keys=False)

    print("=== Joint then independent fine-tune mode ===")
    print(f"Parent config: {Path(config_path).resolve()}")
    print(f"Parent results dir: {out_dir}")
    print(f"Joint stage config: {joint_cfg_path}")
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--config",
        str(joint_cfg_path),
        "--device",
        str(device_arg),
    ]
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"Joint warm-start stage failed with exit code {completed.returncode}.")

    joint_metadata_path = joint_dir / "training_run_metadata.json"
    if not joint_metadata_path.exists():
        raise FileNotFoundError(
            f"Missing joint-stage metadata after training: {joint_metadata_path}"
        )
    joint_metadata = _read_json(joint_metadata_path)
    joint_checkpoint_path = str(
        ((joint_metadata.get("runtime", {}) or {}).get("checkpoint_path", "")) or ""
    ).strip()
    if not joint_checkpoint_path:
        raise ValueError("Joint-stage metadata is missing runtime.checkpoint_path.")

    joint_stage_entry = {
        "run_dir": str(joint_dir.resolve()),
        "config_path": str(joint_cfg_path.resolve()),
        "metadata_path": str(joint_metadata_path.resolve()),
        "checkpoint_path": str(Path(joint_checkpoint_path).resolve()),
    }
    return run_independent_per_target_training(
        config_path=config_path,
        cfg=cfg,
        device=device,
        device_arg=device_arg,
        joint_initialization_checkpoint_path=joint_stage_entry["checkpoint_path"],
        target_modeling_label="joint_then_independent_finetune",
        joint_stage_entry=joint_stage_entry,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train coastal-transformer point-centric model")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--config", default=None, help="Path to YAML config")
    group.add_argument(
        "--batch-config",
        default=None,
        help="Path to a batch YAML listing training case configs to run sequentially",
    )
    parser.add_argument("--device", default="auto", help="cuda | cpu | auto")
    args = parser.parse_args()

    if args.batch_config:
        raise SystemExit(run_batch_training(args.batch_config, args.device))

    config_path = args.config or "configs/training.yaml"
    cfg = resolve_config(read_yaml(config_path))
    cfg = normalize_runtime_config_paths(cfg, config_path=config_path)

    resolved_device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if args.device == "auto" and not torch.cuda.is_available():
        resolved_device = "cpu"
    device = torch.device(resolved_device)

    target_modeling = resolve_target_modeling(cfg)
    if target_modeling == "independent_per_target":
        raise SystemExit(
            run_independent_per_target_training(
                config_path=config_path,
                cfg=cfg,
                device=device,
                device_arg=args.device,
            )
        )
    if target_modeling == "joint_then_independent_finetune":
        raise SystemExit(
            run_joint_then_independent_finetune_training(
                config_path=config_path,
                cfg=cfg,
                device=device,
                device_arg=args.device,
            )
        )

    data_cfg = cfg.get("data", {})
    train_cfg = cfg.get("training", {})
    log_cfg = cfg.get("logging", {})
    targets_cfg = resolve_targets_config(data_cfg)
    resolution_meta = cfg.get("_config_resolution", {}) or {}
    breaking_cfg = resolve_breaking_physics_config(cfg)

    for warning in resolution_meta.get("warnings", []) or []:
        print(f"WARNING: {warning}")

    seed = train_cfg.get("seed", None)
    deterministic = bool(train_cfg.get("deterministic", False))
    if seed is not None:
        set_random_seed(int(seed), deterministic=deterministic)

    requested_bathy_channels, requested_bathy_in_channels = _resolve_runtime_bathy_request(cfg)
    arrays = load_point_centric_arrays(
        data_cfg.get("point_centric_dir", "data/processed/point_centric_demo"),
        bathy_channels=requested_bathy_channels,
        bathy_in_channels=requested_bathy_in_channels,
    )
    arrays = apply_runtime_static_ablation(arrays, resolve_static_ablation_config_path(cfg))
    ablation_summary = arrays.ablation_summary or {
        "config_path": "configs/ablation.yaml",
        "enabled": False,
        "matched_raw_features": [],
        "matched_transformed_features": [],
        "warnings": [],
        "static_feature_count_before": len(arrays.static_feature_names),
        "static_feature_count_after": len(arrays.static_feature_names),
    }
    date_range_meta = (
        (arrays.metadata or {}).get("date_range", {}) if isinstance(arrays.metadata, dict) else {}
    )
    sample_counts_meta = (
        (arrays.metadata or {}).get("sample_counts", {})
        if isinstance(arrays.metadata, dict)
        else {}
    )
    for warning in ablation_summary.get("warnings", []):
        print(f"WARNING: {warning}")

    # Build dataloaders for each split so we can inspect which sites are included
    train_loader, train_ds = build_split_dataloader(
        arrays,
        cfg,
        split_name="train",
        shuffle=True,
        apply_train_sample_subsampling=True,
    )
    val_loader, val_ds = build_split_dataloader(arrays, cfg, split_name="val", shuffle=False)
    test_loader, test_ds = build_split_dataloader(arrays, cfg, split_name="test", shuffle=False)
    split_info = resolve_site_split_config(arrays.target_sites, cfg)

    train_sites = set(train_ds.sites)
    val_sites = set(val_ds.sites)
    test_sites = set(test_ds.sites)
    train_val_overlap = sorted(train_sites.intersection(val_sites))
    train_test_overlap = sorted(train_sites.intersection(test_sites))
    val_test_overlap = sorted(val_sites.intersection(test_sites))
    if train_val_overlap or train_test_overlap or val_test_overlap:
        raise RuntimeError(
            "Split leakage detected: train/val/test site sets must be mutually exclusive. "
            f"train_val_overlap={train_val_overlap} "
            f"train_test_overlap={train_test_overlap} "
            f"val_test_overlap={val_test_overlap}."
        )

    if len(train_ds) == 0:
        raise RuntimeError("Training dataset is empty. Check seq_len and split/site configuration.")

    sample = train_ds[0]
    use_multi_source = "x_dynamic_sources" in sample
    dynamic_mode = "multi-source" if use_multi_source else "single-source"
    dynamic_input_dim = int(sample["x_dynamic"].shape[-1]) if "x_dynamic" in sample else 0
    source_dynamic_input_dim = (
        int(sample["x_dynamic_sources"].shape[-1]) if "x_dynamic_sources" in sample else None
    )
    source_geometry_input_dim = (
        int(sample["source_geometry"].shape[-1]) if "source_geometry" in sample else None
    )
    x_dynamic_sources_shape = (
        tuple(sample["x_dynamic_sources"].shape) if "x_dynamic_sources" in sample else None
    )
    source_geometry_shape = (
        tuple(sample["source_geometry"].shape) if "source_geometry" in sample else None
    )
    static_input_dim = int(sample["x_static"].shape[-1]) if "x_static" in sample else 0
    bathy_shape = tuple(sample["x_bathy"].shape) if "x_bathy" in sample else None
    bathy_channel_names = list(getattr(arrays, "bathy_channel_names", []) or [])
    output_dim = 4 if isinstance(sample["y"], dict) else int(sample["y"].shape[-1])

    if output_dim not in {4, 6}:
        raise ValueError(
            "Coastal transformer training expects either 4 hybrid targets [hs, tp, dir, dp] "
            "or 6 legacy sin/cos channels [hs, tp, dir_sin, dir_cos, dp_sin, dp_cos]. "
            f"Found output_dim={output_dim}. Check data.output_columns in configs/training.yaml."
        )

    model = build_model_from_config(
        config=cfg,
        dynamic_input_dim=dynamic_input_dim,
        static_input_dim=static_input_dim,
        output_dim=output_dim,
        dynamic_feature_names=getattr(arrays, "dynamic_feature_names", None),
        source_dynamic_input_dim=source_dynamic_input_dim,
        source_geometry_input_dim=source_geometry_input_dim,
        source_feature_names=getattr(arrays, "source_feature_names", None),
    ).to(device)
    dynamic_mode = (
        "multi-source" if bool(getattr(model, "use_multi_source", False)) else "single-source"
    )
    initialization_summary = initialize_model_from_config_checkpoint(
        model=model,
        config=cfg,
        device=device,
    )

    # Build loss/optimizer/scheduler from config
    loss_fn = build_loss_from_config(cfg)

    # Guardrail: directional channels should usually use a circular-aware loss.
    if output_dim == 6:
        has_direction_component = (
            hasattr(loss_fn, "angular_loss")
            or hasattr(loss_fn, "task_losses")
            or type(loss_fn).__name__ == "AngularLoss"
        )
        if not has_direction_component:
            print(
                "WARNING: Resolved loss has no angular component for direction channels. "
                "This can degrade dir/dp predictions on holdout sites. "
                "Consider loss_type=coastal_multitask, weighted_multitask, or angular."
            )

    loss_trainable_params = [p for p in loss_fn.parameters() if p.requires_grad]
    optimizer, optimizer_kind = build_optimizer(
        model,
        cfg,
        extra_parameters=loss_trainable_params,
    )
    scheduler, scheduler_kind, plateau_monitor = build_scheduler(optimizer, cfg)
    selection_monitor, selection_mode, selection_weights = resolve_selection_config(cfg)

    pcgrad_cfg = train_cfg.get("pcgrad", {}) or {}
    pcgrad_enabled = bool(pcgrad_cfg.get("enabled", False))
    pcgrad_optimizer = None
    pcgrad_backend = "disabled"
    if pcgrad_enabled:
        try:
            pass  # Hard dependency gate.
        except Exception as exc:
            raise ImportError(
                "training.pcgrad.enabled=true requires torch-optimizer. "
                "Install dependency and retry."
            ) from exc

        if not hasattr(loss_fn, "task_losses"):
            raise ValueError(
                "training.pcgrad.enabled=true requires a loss module with task_losses(pred, target)."
            )
        pcgrad_optimizer = create_pcgrad_optimizer(
            optimizer=optimizer,
            reduction=str(pcgrad_cfg.get("reduction", "mean")),
        )
        pcgrad_backend = str(
            getattr(pcgrad_optimizer, "_pcgrad_backend", type(pcgrad_optimizer).__name__)
        )

    # Verbose startup summary for transparency
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    sched_cfg = train_cfg.get("scheduler", {})
    loss_cfg = train_cfg.get("loss", {}) or {}
    early_cfg = train_cfg.get("early_stopping", {})
    progress_cfg = _resolve_progress_bar_config(cfg)

    print("--- Training startup summary ---")
    print(f"Config path: {Path(config_path).resolve()}")
    print(f"Device: {device}")
    print(f"Seed: {seed} | Deterministic: {deterministic}")
    print(f"Model architecture: {cfg.get('model', {}).get('architecture', 'unknown')}")
    print(f"Model params: total={total_params:,} trainable={trainable_params:,}")
    temporal_encoder = str(getattr(model, "sequence_encoder_type", "transformer")).strip().lower()
    temporal_encoder_label = "LSTM" if temporal_encoder == "lstm" else "Transformer"
    print(f"Temporal encoder: {temporal_encoder_label}")
    if temporal_encoder == "lstm":
        lstm_cfg = getattr(model, "sequence_encoder_config", {}) or {}
        print(
            "LSTM: "
            f"hidden_dim={int(lstm_cfg.get('hidden_dim', 0))} "
            f"num_layers={int(lstm_cfg.get('num_layers', 0))} "
            f"bidirectional={bool(lstm_cfg.get('bidirectional', False))} "
            f"dropout={float(lstm_cfg.get('dropout', 0.0))} "
            f"pooling={str(lstm_cfg.get('pooling', 'last'))} "
            f"layer_norm={bool(lstm_cfg.get('layer_norm', True))}"
        )
    print(f"Dynamic mode: {dynamic_mode}")
    print(f"Model multi-source path: {bool(getattr(model, 'use_multi_source', False))}")
    print(f"Source geometry enabled: {bool(getattr(model, 'use_source_geometry_features', False))}")
    print(
        "Target mode: "
        f"{targets_cfg.get('mode', 'physical')} "
        f"(transfer_reference={targets_cfg.get('transfer_reference', 'nearest_bulk')}, "
        f"transfer_representation={targets_cfg.get('transfer_representation', 'legacy')})"
    )
    if str(targets_cfg.get("transfer_representation", "legacy")) == "residual_correction":
        residual_cfg = targets_cfg.get("residual_correction", {}) or {}
        print(
            "Residual correction: "
            f"bound_method={residual_cfg.get('bound_method', 'none')} "
            f"max_abs_log_hs={float(residual_cfg.get('max_abs_log_hs', 1.25)):.3f} "
            f"max_abs_tp={float(residual_cfg.get('max_abs_tp', 15.0)):.3f} "
            f"max_abs_dir_deg={float(residual_cfg.get('max_abs_dir_deg', 120.0)):.3f} "
            f"max_abs_dp_deg={float(residual_cfg.get('max_abs_dp_deg', 120.0)):.3f} "
            f"zero_init_output_head={bool(residual_cfg.get('zero_init_output_head', True))}"
        )
    print(
        f"Input dims: dynamic={dynamic_input_dim} dynamic_sources={x_dynamic_sources_shape if x_dynamic_sources_shape is not None else 'disabled'} "
        f"source_geometry={source_geometry_shape if source_geometry_shape is not None else 'disabled'} "
        f"static={static_input_dim} bathy={bathy_shape if bathy_shape is not None else 'disabled'} output={output_dim}"
    )
    if bathy_shape is not None:
        print(
            f"Bathy channels: count={len(bathy_channel_names)} "
            f"names={bathy_channel_names if bathy_channel_names else 'unknown'}"
        )
    print(f"Static enabled: {bool(getattr(model, 'use_static_features', False))}")
    print(f"Bathy enabled: {bool(getattr(model, 'use_bathymetry', False))}")
    print(f"Expert mode: {bool(getattr(model, 'use_expert_heads', False))}")
    if initialization_summary is not None:
        print(
            "Initialization checkpoint: "
            f"path={initialization_summary['checkpoint_path']} "
            f"load_mode={initialization_summary['load_mode']} "
            f"details={initialization_summary['load_details']}"
        )
    dropped_preview = list(
        ablation_summary.get("matched_transformed_features")
        or ablation_summary.get("matched_raw_features")
        or []
    )
    dropped_preview_text = str(dropped_preview[:20])
    if len(dropped_preview) > 20:
        dropped_preview_text = f"{dropped_preview[:20]} ... and {len(dropped_preview) - 20} more"
    print(
        f"Ablation: enabled={bool(ablation_summary.get('enabled', False))} "
        f"config={ablation_summary.get('config_path', 'configs/ablation.yaml')}"
    )
    print(
        "Static features: "
        f"before={int(ablation_summary.get('static_feature_count_before', static_input_dim))} "
        f"dropped={len(dropped_preview)} "
        f"after={int(ablation_summary.get('static_feature_count_after', static_input_dim))}"
    )
    print(f"Ablation dropped preview: {dropped_preview_text}")
    print(
        "Date range: "
        f"enabled={bool(date_range_meta.get('enabled', False))} "
        f"requested=[{date_range_meta.get('requested_start')}, {date_range_meta.get('requested_end')}] "
        f"resolved=[{date_range_meta.get('resolved_start')}, {date_range_meta.get('resolved_end')}]"
    )
    print(
        "Preprocess sample counts: "
        f"timesteps={sample_counts_meta.get('timesteps', len(arrays.timestamps))} "
        f"target_sites={sample_counts_meta.get('target_sites', len(arrays.target_sites))} "
        f"site_time_samples={sample_counts_meta.get('site_time_samples', len(arrays.timestamps) * len(arrays.target_sites))}"
    )
    print(f"Data samples: train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")
    train_filter_summary = getattr(train_ds, "sample_filter_summary", {}) or {}
    val_filter_summary = getattr(val_ds, "sample_filter_summary", {}) or {}
    test_filter_summary = getattr(test_ds, "sample_filter_summary", {}) or {}
    train_subsampling_summary = getattr(train_ds, "train_sample_subsampling_summary", {}) or {}
    if any(
        bool(summary.get("enabled", False))
        for summary in (train_filter_summary, val_filter_summary, test_filter_summary)
    ):
        print(
            "Runtime sample filter: "
            f"train_excluded={int(train_filter_summary.get('excluded_count', 0))} "
            f"val_excluded={int(val_filter_summary.get('excluded_count', 0))} "
            f"test_excluded={int(test_filter_summary.get('excluded_count', 0))}"
        )
        if bool(train_filter_summary.get("applied", False)):
            print(
                "Runtime sample filter details: "
                f"statistic={train_filter_summary.get('statistic', 'unknown')} "
                f"match={train_filter_summary.get('match', 'unknown')} "
                f"hs_min={train_filter_summary.get('hs_min')} "
                f"tp_min={train_filter_summary.get('tp_min')} "
                f"before={int(train_filter_summary.get('before_count', len(train_ds)))} "
                f"kept={int(train_filter_summary.get('kept_count', len(train_ds)))}"
            )
    if bool(train_subsampling_summary.get("enabled", False)):
        print(
            "Train sample subsampling: "
            f"enabled={bool(train_subsampling_summary.get('enabled', False))} "
            f"mode={train_subsampling_summary.get('mode', 'unknown')} "
            f"fraction={float(train_subsampling_summary.get('fraction', 1.0))} "
            f"seed={int(train_subsampling_summary.get('seed', 42))} "
            f"before={int(train_subsampling_summary.get('before_count', len(train_ds)))} "
            f"after={int(train_subsampling_summary.get('after_count', len(train_ds)))} "
            f"removed={int(train_subsampling_summary.get('removed_count', 0))}"
        )
        if str(train_subsampling_summary.get("mode", "")) == "per_site":
            print(
                "Train sample subsampling per-site: "
                f"train_sites={int(train_subsampling_summary.get('train_sites', 0))} "
                f"min_before={int(train_subsampling_summary.get('min_before', 0))} "
                f"max_before={int(train_subsampling_summary.get('max_before', 0))} "
                f"min_after={int(train_subsampling_summary.get('min_after', 0))} "
                f"max_after={int(train_subsampling_summary.get('max_after', 0))}"
            )
    train_site_subsampling_summary = split_info.get("train_site_subsampling", {}) or {}
    if bool(train_site_subsampling_summary.get("enabled", False)):
        print(
            "Train site subsampling: "
            f"enabled={bool(train_site_subsampling_summary.get('enabled', False))} "
            f"applied={bool(train_site_subsampling_summary.get('applied', False))} "
            f"fraction={float(train_site_subsampling_summary.get('fraction', 1.0))} "
            f"seed={int(train_site_subsampling_summary.get('seed', 42))} "
            f"before={int(train_site_subsampling_summary.get('before_count', len(train_ds.sites)))} "
            f"after={int(train_site_subsampling_summary.get('after_count', len(train_ds.sites)))} "
            f"removed={int(train_site_subsampling_summary.get('removed_count', 0))}"
        )
    print(f"Sites: train={len(train_ds.sites)} val={len(val_ds.sites)} test={len(test_ds.sites)}")
    print(f"Validation mode: {split_info.get('validation_mode', 'legacy-temporal')}")
    print(f"Validation site-heldout: {bool(split_info.get('validation_site_heldout', False))}")
    print(f"Site-holdout temporal mode: {split_info.get('site_holdout_temporal_mode', 'legacy')}")
    print(
        f"Site-holdout temporal active: {bool(split_info.get('site_holdout_temporal_active', False))}"
    )
    print(
        f"Site-holdout recent fraction: {split_info.get('site_holdout_temporal_recent_fraction', None)}"
    )
    print(f"Train sites: {_format_site_preview(train_ds.sites)}")
    print(f"Validation sites: {_format_site_preview(val_ds.sites)}")
    test_preview_sites = split_info.get("test_sites", []) or []
    print(f"Test sites: {_format_site_preview(test_preview_sites)}")
    print(
        f"Hyperparams: batch_size={int(train_cfg.get('batch_size', 128))}, "
        f"lr={float((train_cfg.get('optimizer', {}) or {}).get('lr', 1e-3))}, "
        f"epochs={int(train_cfg.get('epochs', 50))}, "
        f"weight_decay={float((train_cfg.get('optimizer', {}) or {}).get('weight_decay', 0.0))}"
    )
    print(f"Loss: {str(loss_cfg.get('type', train_cfg.get('loss_type', 'mse'))).lower()}")
    print(f"Resolved loss module: {type(loss_fn).__name__}")
    print(
        "Breaking physics penalty: "
        f"enabled={bool(breaking_cfg.get('enabled', False))} "
        f"mode={str(breaking_cfg.get('mode', 'local_depth'))} "
        f"weight={float(breaking_cfg.get('penalty_weight', 0.0)):.5f}"
    )
    if bool(breaking_cfg.get("enabled", False)) and (
        arrays.local_breaking_hs_cap is None or arrays.local_breaking_cap_valid is None
    ):
        print(
            "WARNING: Breaking physics is enabled in config, but the dataset has no physics payload. "
            "The penalty will be inactive for this run."
        )
    print(f"PCGrad enabled: {pcgrad_enabled} | backend: {pcgrad_backend}")
    if loss_trainable_params:
        loss_trainable_total = sum(p.numel() for p in loss_trainable_params)
        print(f"Trainable loss params: {loss_trainable_total:,}")
    if (
        hasattr(loss_fn, "hs_weight")
        and hasattr(loss_fn, "tp_weight")
        and hasattr(loss_fn, "direction_weight")
    ):
        print(
            "Weighted loss weights: "
            f"hs={float(getattr(loss_fn, 'hs_weight'))}, "
            f"tp={float(getattr(loss_fn, 'tp_weight'))}, "
            f"direction={float(getattr(loss_fn, 'direction_weight'))}"
        )
    if hasattr(loss_fn, "angular_loss") and hasattr(loss_fn.angular_loss, "pair_indices"):
        print(f"Direction pair indices: {list(loss_fn.angular_loss.pair_indices)}")
    print(f"Optimizer: {optimizer_kind}")
    if scheduler_kind == "plateau":
        print(f"Scheduler: plateau (monitor={plateau_monitor})")
    else:
        print(f"Scheduler: {sched_cfg.get('type', 'step')}")
    resolved_scheduler = (resolution_meta.get("resolved", {}) or {}).get("scheduler", {})
    if resolved_scheduler:
        print(f"Resolved scheduler params: {resolved_scheduler}")
    resolved_model = resolution_meta.get("resolved", {}) or {}
    print(
        "Resolved transformer: "
        f"model_dim={resolved_model.get('model_dim')} "
        f"num_layers={resolved_model.get('num_layers')} "
        f"num_heads={resolved_model.get('num_heads')} "
        f"ff_multiplier={resolved_model.get('ff_multiplier')}"
    )
    print(
        f"Sampler strategy: {str(((train_cfg.get('sampler', {}) or {}).get('strategy', 'hs_squared')))}"
    )
    print(f"Static regularization: {(data_cfg.get('static_regularization', {}) or {})}")
    print(f"Checkpoint selection: monitor={selection_monitor} mode={selection_mode}")
    if selection_monitor == "physical_score":
        print(f"Physical score weights: {selection_weights}")
    validation_every_n_epochs = resolve_validation_every_n_epochs(cfg)
    print(
        f"Validation cadence: every {validation_every_n_epochs} epoch(s) (final epoch always validates)"
    )
    print(
        f"EarlyStopping: enabled={bool(early_cfg.get('enabled', False))}, patience={int(early_cfg.get('patience', 0))}"
    )
    print(f"Progress bars: {progress_cfg}")
    print("-------------------------------")

    epochs = int(train_cfg.get("epochs", 50))
    grad_clip_norm = float(train_cfg.get("gradient_clip_norm", 1.0))
    max_train_batches = train_cfg.get("max_train_batches", None)
    max_val_batches = train_cfg.get("max_val_batches", None)
    max_train_batches = None if max_train_batches is None else int(max_train_batches)
    max_val_batches = None if max_val_batches is None else int(max_val_batches)

    early_cfg = train_cfg.get("early_stopping", {})
    early_enabled = bool(early_cfg.get("enabled", False))
    early_patience = int(early_cfg.get("patience", 10))
    early_min_delta = float(early_cfg.get("min_delta", 0.0))
    if early_enabled and early_patience < 1:
        raise ValueError(
            "training.early_stopping.patience must be >= 1 when early stopping is enabled"
        )

    out_dir = Path(log_cfg.get("output_dir", "results/coastal_transformer"))
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_name = str(log_cfg.get("checkpoint_name", "coastal_transformer_best.pt"))
    ckpt_path = out_dir / ckpt_name

    best_selection_score = float("inf") if selection_mode == "min" else float("-inf")
    best_checkpoint_val_loss = float("nan")
    best_epoch = 0
    early_best_score = float("inf") if selection_mode == "min" else float("-inf")
    early_bad_epochs = 0
    history = []
    architecture_name = str(cfg.get("model", {}).get("architecture", "coastal_transformer")).lower()
    model_api_version = "coastal_transformer_v2"
    effective_dynamic_feature_names = _resolve_effective_dynamic_feature_names(arrays)
    effective_source_feature_names = _resolve_effective_source_feature_names(arrays)
    loss_trainable_total = sum(p.numel() for p in loss_trainable_params)
    training_started_at_utc = _utc_now_iso()
    training_perf_counter_start = time.perf_counter()
    run_metadata = {
        "runtime": {
            "config_path": str(Path(config_path).resolve()),
            "device": str(device),
            "torch_version": str(torch.__version__),
            "output_dir": str(out_dir),
            "checkpoint_path": str(ckpt_path),
        },
        "config": cfg,
        "model": {
            "architecture": architecture_name,
            "model_api_version": model_api_version,
            "model_class": type(model).__name__,
            "sequence_encoder_type": str(getattr(model, "sequence_encoder_type", "unknown")),
            "sequence_encoder_config": getattr(model, "sequence_encoder_config", {}),
            "decoder_type": str(getattr(model, "decoder_type", "cross_attention")),
            "decoder_config": getattr(model, "decoder_config", {}),
            "use_static_features": bool(getattr(model, "use_static_features", False)),
            "use_bathymetry": bool(getattr(model, "use_bathymetry", False)),
            "use_multi_source": bool(getattr(model, "use_multi_source", False)),
            "use_source_geometry_features": bool(
                getattr(model, "use_source_geometry_features", False)
            ),
            "use_expert_heads": bool(getattr(model, "use_expert_heads", False)),
            "dynamic_mode": dynamic_mode,
            "parameter_counts": {
                "total": int(total_params),
                "trainable": int(trainable_params),
                "loss_trainable": int(loss_trainable_total),
            },
            "tensor_dims": {
                "dynamic_input_dim": int(dynamic_input_dim),
                "source_dynamic_input_dim": None
                if source_dynamic_input_dim is None
                else int(source_dynamic_input_dim),
                "source_geometry_input_dim": None
                if source_geometry_input_dim is None
                else int(source_geometry_input_dim),
                "static_input_dim": int(static_input_dim),
                "bathy_shape": list(bathy_shape) if bathy_shape is not None else None,
                "output_dim": int(output_dim),
            },
        },
        "training": {
            "optimizer_kind": str(optimizer_kind),
            "scheduler_kind": str(scheduler_kind),
            "plateau_monitor": plateau_monitor,
            "selection_monitor": str(selection_monitor),
            "selection_mode": str(selection_mode),
            "selection_weights": selection_weights,
            "resolved_loss_module": type(loss_fn).__name__,
            "config_resolution": resolution_meta,
            "pcgrad_enabled": bool(pcgrad_enabled),
            "pcgrad_backend": str(pcgrad_backend),
            "validation_every_n_epochs": int(validation_every_n_epochs),
            "progress": progress_cfg,
            "initialization": copy.deepcopy(initialization_summary),
        },
        "data": {
            "point_centric_dir": str(data_cfg.get("point_centric_dir", "")),
            "date_range": date_range_meta,
            "sample_counts": sample_counts_meta,
            "ablation_summary": ablation_summary,
            "feature_layout": {
                "x_dynamic": effective_dynamic_feature_names,
                "x_static": [
                    str(name) for name in (getattr(arrays, "static_feature_names", []) or [])
                ],
                "x_dynamic_sources": effective_source_feature_names,
                "source_geometry": [
                    str(name)
                    for name in (getattr(arrays, "source_geometry_feature_names", []) or [])
                ],
                "x_bathy_channels": bathy_channel_names,
                "targets_encoded": [
                    str(name) for name in (getattr(arrays, "target_feature_names", []) or [])
                ],
                "targets_physical": [
                    str(name) for name in (getattr(arrays, "physical_target_names", []) or [])
                ],
                "targets_transfer": [
                    str(name) for name in (getattr(arrays, "transfer_target_names", []) or [])
                ],
                "targets_reference": [
                    str(name) for name in (getattr(arrays, "reference_target_names", []) or [])
                ],
            },
            "targets_config": targets_cfg,
            "static_regularization": copy.deepcopy(data_cfg.get("static_regularization", {}) or {}),
            "train_site_subsampling": copy.deepcopy(
                data_cfg.get("train_site_subsampling", {}) or {}
            ),
            "train_sample_subsampling": copy.deepcopy(
                data_cfg.get("train_sample_subsampling", {}) or {}
            ),
            "sampler": copy.deepcopy(train_cfg.get("sampler", {}) or {}),
        },
        "splits": {
            "validation_mode": split_info.get("validation_mode", "legacy-temporal"),
            "validation_site_heldout": bool(split_info.get("validation_site_heldout", False)),
            "site_holdout_temporal_mode": split_info.get("site_holdout_temporal_mode", "legacy"),
            "site_holdout_temporal_active": bool(
                split_info.get("site_holdout_temporal_active", False)
            ),
            "site_holdout_temporal_recent_fraction": split_info.get(
                "site_holdout_temporal_recent_fraction", None
            ),
            "train_sites": list(train_ds.sites),
            "val_sites": list(val_ds.sites),
            "test_sites": list(test_ds.sites),
            "train_site_subsampling": copy.deepcopy(train_site_subsampling_summary),
            "train_sample_subsampling_summary": copy.deepcopy(train_subsampling_summary),
            "sample_counts": {
                "train": int(len(train_ds)),
                "val": int(len(val_ds)),
                "test": int(len(test_ds)),
            },
        },
    }
    _update_training_timing_metadata(
        run_metadata,
        training_started_at_utc=training_started_at_utc,
    )
    run_metadata_json_path, run_metadata_pickle_path = _write_serialized_artifact(
        out_dir,
        "training_run_metadata",
        run_metadata,
    )
    observed_io_state = {"saved": False}
    observed_io_json_path = out_dir / "observed_model_io.json"
    observed_io_pickle_path = out_dir / "observed_model_io.pkl"

    def _make_batch_observer(epoch_value: int):
        def _observer(
            *, batch: dict, model_inputs: dict, target, batch_idx: int, split_name: str
        ) -> None:
            if observed_io_state["saved"] or str(split_name) != "train":
                return
            manifest = _build_model_io_manifest(
                cfg=cfg,
                arrays=arrays,
                batch=batch,
                model_inputs=model_inputs,
                target=target,
                model=model,
                epoch_index=int(epoch_value),
                batch_idx=int(batch_idx),
                split_name=str(split_name),
            )
            _write_serialized_artifact(out_dir, "observed_model_io", manifest)
            observed_io_state["saved"] = True
            print(f"Observed model I/O manifest saved to: {observed_io_json_path}")

        return _observer

    for epoch in range(1, epochs + 1):
        train_loss, train_metrics, train_components = run_epoch(
            model=model,
            loader=train_loader,
            loss_fn=loss_fn,
            device=device,
            optimizer=optimizer,
            grad_clip_norm=grad_clip_norm,
            pcgrad_optimizer=pcgrad_optimizer,
            max_batches=max_train_batches,
            point_centric_dir=data_cfg.get("point_centric_dir", ""),
            transfer_representation=str(targets_cfg.get("transfer_representation", "legacy")),
            residual_cfg=targets_cfg.get("residual_correction", {}) or {},
            transfer_tp_min=float(targets_cfg.get("tp_min", 0.5)),
            transfer_tp_max=float(targets_cfg.get("tp_max", 30.0)),
            epoch_index=epoch,
            total_epochs=epochs,
            progress_cfg=progress_cfg,
            progress_phase="train",
            breaking_enabled=bool(breaking_cfg.get("enabled", False)),
            breaking_cap_key=str(breaking_cfg.get("cap_key", "local_breaking_hs_cap")),
            breaking_valid_key=str(breaking_cfg.get("valid_key", "local_breaking_cap_valid")),
            breaking_penalty_weight=float(breaking_cfg.get("penalty_weight", 0.02)),
            batch_observer=None if observed_io_state["saved"] else _make_batch_observer(epoch),
        )

        validation_ran = should_run_validation_epoch(epoch, epochs, validation_every_n_epochs)
        val_loss = None
        val_metrics: Dict[str, float] = {}
        val_components: Dict[str, float] = {}
        selection_score = None
        if validation_ran:
            val_loss, val_metrics, val_components = run_epoch(
                model=model,
                loader=val_loader,
                loss_fn=loss_fn,
                device=device,
                optimizer=None,
                grad_clip_norm=0.0,
                pcgrad_optimizer=None,
                max_batches=max_val_batches,
                point_centric_dir=data_cfg.get("point_centric_dir", ""),
                transfer_representation=str(targets_cfg.get("transfer_representation", "legacy")),
                residual_cfg=targets_cfg.get("residual_correction", {}) or {},
                transfer_tp_min=float(targets_cfg.get("tp_min", 0.5)),
                transfer_tp_max=float(targets_cfg.get("tp_max", 30.0)),
                epoch_index=epoch,
                total_epochs=epochs,
                progress_cfg=progress_cfg,
                progress_phase="val",
                breaking_enabled=bool(breaking_cfg.get("enabled", False)),
                breaking_cap_key=str(breaking_cfg.get("cap_key", "local_breaking_hs_cap")),
                breaking_valid_key=str(breaking_cfg.get("valid_key", "local_breaking_cap_valid")),
                breaking_penalty_weight=float(breaking_cfg.get("penalty_weight", 0.02)),
                batch_observer=None,
            )

            selection_score = (
                float(val_loss)
                if selection_monitor == "val_loss"
                else compute_physical_score(val_metrics, selection_weights)
            )

        if scheduler is not None:
            if scheduler_kind == "plateau":
                if plateau_monitor == "val_loss":
                    if validation_ran and val_loss is not None:
                        scheduler.step(val_loss)
                else:
                    scheduler.step(train_loss)
            else:
                scheduler.step()

        lr = optimizer.param_groups[0]["lr"]

        if validation_ran and val_loss is not None and selection_score is not None:
            print(
                f"Epoch {epoch:03d}/{epochs:03d} | "
                f"LR={lr:.6e} | TrainLoss={train_loss:.6f} | ValLoss={val_loss:.6f} | "
                f"{selection_monitor}={selection_score:.6f}"
            )
        else:
            print(
                f"Epoch {epoch:03d}/{epochs:03d} | "
                f"LR={lr:.6e} | TrainLoss={train_loss:.6f} | "
                f"Val=SKIPPED (validation_every_n_epochs={validation_every_n_epochs})"
            )
        print("Loss components:")
        print(_format_component_line("Train", train_components))
        if validation_ran:
            print(_format_component_line("Val", val_components))
            print(format_metric_block("  Val  :", val_metrics))
        else:
            print("  Val  skipped")

        history.append(
            {
                "epoch": epoch,
                "lr": float(lr),
                "train_loss": float(train_loss),
                "validation_ran": bool(validation_ran),
                "val_loss": None if val_loss is None else float(val_loss),
                "train_components": _to_float_dict(train_components),
                "val_components": _to_float_dict(val_components) if validation_ran else {},
                "val_metrics": _to_float_dict(val_metrics) if validation_ran else {},
                "selection_score": None if selection_score is None else float(selection_score),
                "selection_mode": selection_monitor,
                "ablation_enabled": bool(ablation_summary.get("enabled", False)),
                "static_feature_count_before_ablation": int(
                    ablation_summary.get("static_feature_count_before", static_input_dim)
                ),
                "static_feature_count_after_ablation": int(
                    ablation_summary.get("static_feature_count_after", static_input_dim)
                ),
                "static_feature_drop_count": len(dropped_preview),
                **{f"train_{k}": float(v) for k, v in train_metrics.items()},
                **(
                    {f"val_{k}": float(v) for k, v in val_metrics.items()} if validation_ran else {}
                ),
            }
        )

        if (
            validation_ran
            and selection_score is not None
            and _is_improved(selection_score, best_selection_score, selection_mode)
        ):
            best_selection_score = float(selection_score)
            best_checkpoint_val_loss = float(val_loss) if val_loss is not None else float("nan")
            best_epoch = epoch
            torch.save(
                {
                    "checkpoint_version": 2,
                    "architecture": architecture_name,
                    "model_api_version": model_api_version,
                    "torch_version": str(torch.__version__),
                    "model_state_dict": model.state_dict(),
                    "config": cfg,
                    "ablation_summary": ablation_summary,
                    "dynamic_input_dim": dynamic_input_dim,
                    "source_dynamic_input_dim": source_dynamic_input_dim,
                    "source_geometry_input_dim": source_geometry_input_dim,
                    "dynamic_mode": dynamic_mode,
                    "static_input_dim": static_input_dim,
                    "bathy_shape": list(bathy_shape) if bathy_shape is not None else None,
                    "output_dim": output_dim,
                },
                ckpt_path,
            )

        if early_enabled and validation_ran and selection_score is not None:
            if _is_improved(
                selection_score, early_best_score, selection_mode, min_delta=early_min_delta
            ):
                early_best_score = float(selection_score)
                early_bad_epochs = 0
            else:
                early_bad_epochs += 1
                if early_bad_epochs >= early_patience:
                    print(
                        "Early stopping triggered: "
                        f"no {selection_monitor} improvement > {early_min_delta} "
                        f"for {early_patience} epoch(s)."
                    )
                    break

    _update_training_timing_metadata(
        run_metadata,
        training_started_at_utc=training_started_at_utc,
        training_finished_at_utc=_utc_now_iso(),
        training_duration_seconds=time.perf_counter() - training_perf_counter_start,
    )
    run_metadata_json_path, run_metadata_pickle_path = _write_serialized_artifact(
        out_dir,
        "training_run_metadata",
        run_metadata,
    )

    if best_epoch == 0:
        raise RuntimeError(
            "Validation never ran successfully for checkpoint selection. "
            "Check training.validation_every_n_epochs and epoch count."
        )

    hist_path = out_dir / "train_history.json"
    with hist_path.open("w") as fh:
        json.dump(history, fh, indent=2)

    checkpoint_payload = torch.load(ckpt_path, map_location=device)
    if not isinstance(checkpoint_payload, dict) or "model_state_dict" not in checkpoint_payload:
        raise RuntimeError(f"Best checkpoint is missing model_state_dict: {ckpt_path}")
    model.load_state_dict(checkpoint_payload["model_state_dict"])

    metrics_summary_csv_path = export_target_metrics_summary_csv(
        config=cfg,
        model=model,
        device=device,
        val_loader=val_loader,
        val_ds=val_ds,
        test_loader=test_loader,
        test_ds=test_ds,
        out_dir=out_dir,
    )

    print(f"Best checkpoint: {ckpt_path}")
    print(f"Best checkpoint selected by: {selection_monitor}")
    print(f"Best selection score: {best_selection_score:.6f}")
    print(f"Best val_loss: {best_checkpoint_val_loss:.6f}")
    print(f"Best epoch: {best_epoch}")
    print(f"History saved to: {hist_path}")
    if metrics_summary_csv_path is not None:
        print(f"Target metrics summary CSV saved to: {metrics_summary_csv_path}")
    print(f"Run metadata saved to: {run_metadata_json_path}")
    print(f"Run metadata pickle saved to: {run_metadata_pickle_path}")
    if observed_io_state["saved"]:
        print(f"Observed model I/O saved to: {observed_io_json_path}")
        print(f"Observed model I/O pickle saved to: {observed_io_pickle_path}")


if __name__ == "__main__":
    main()
