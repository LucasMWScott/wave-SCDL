"""Evaluate a saved coastal-wave checkpoint on prepared data.

Evaluation rebuilds the model from the training configuration and metadata,
restores its preprocessing contract, predicts a requested split, converts
outputs back to physical units, and writes metrics plus prediction tables.
Use the same prepared-data directory that was used for training.
"""

from __future__ import annotations

import argparse
import json
import pickle
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

try:
    from config_loader import read_yaml_config
except Exception:
    from src.config_loader import read_yaml_config

try:
    from data_pipeline import (
        apply_runtime_static_ablation,
        build_split_dataloader,
        load_point_centric_arrays,
        resolve_static_ablation_config_path,
    )
    from metrics import compute_wave_metrics
    from models import build_model_from_config
    from transfer_runtime import (
        load_transfer_scaler_stats,
        physical_matrix_to_metric_channels,
        recover_transfer_metric_arrays,
    )
except Exception:
    from src.data_pipeline import (
        apply_runtime_static_ablation,
        build_split_dataloader,
        load_point_centric_arrays,
        resolve_static_ablation_config_path,
    )
    from src.metrics import compute_wave_metrics
    from src.models import build_model_from_config
    from src.transfer_runtime import (
        load_transfer_scaler_stats,
        physical_matrix_to_metric_channels,
        recover_transfer_metric_arrays,
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
except Exception:
    from src.runtime_paths import normalize_runtime_config_paths

try:
    from independent_target_mode import (
        is_independent_target_composite_metadata,
        load_model_from_training_metadata,
    )
except Exception:
    from src.independent_target_mode import (
        is_independent_target_composite_metadata,
        load_model_from_training_metadata,
    )


def _looks_like_state_dict(payload: object) -> bool:
    if not isinstance(payload, dict) or len(payload) == 0:
        return False
    return all(torch.is_tensor(v) for v in payload.values())


def _extract_state_dict(checkpoint: object) -> dict:
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state = checkpoint["model_state_dict"]
        if not _looks_like_state_dict(state):
            raise RuntimeError(
                "Checkpoint contains 'model_state_dict' but it is not a valid tensor mapping."
            )
        return state

    if _looks_like_state_dict(checkpoint):
        return checkpoint  # legacy direct state-dict checkpoints

    raise RuntimeError(
        "Unsupported checkpoint format. Expected either {'model_state_dict': ...} or a direct state_dict."
    )


def _load_state_dict_best_effort(model: torch.nn.Module, state_dict: dict) -> tuple[str, dict]:
    """Try strict checkpoint load first, then shape-compatible partial fallback."""
    try:
        model.load_state_dict(state_dict)
        return "strict", {
            "loaded": len(state_dict),
            "missing": 0,
            "unexpected": 0,
            "shape_mismatch": 0,
        }
    except RuntimeError as strict_error:
        current = model.state_dict()
        compatible = {}
        shape_mismatch = []
        unexpected = []

        for key, value in state_dict.items():
            if key not in current:
                unexpected.append(key)
                continue
            if current[key].shape != value.shape:
                shape_mismatch.append(key)
                continue
            compatible[key] = value

        if not compatible:
            raise RuntimeError(
                "Checkpoint is incompatible with the current architecture and no shape-compatible "
                "parameters were found for fallback loading."
            ) from strict_error

        incompatible = model.load_state_dict(compatible, strict=False)
        details = {
            "loaded": len(compatible),
            "missing": len(incompatible.missing_keys),
            "unexpected": len(unexpected),
            "shape_mismatch": len(shape_mismatch),
        }
        return "partial", details


def _resolve_runtime_decoder_type(config: dict) -> str:
    model_cfg = (config.get("model", {}) or {}).get("coastal_transformer", {}) or {}
    decoder_cfg = model_cfg.get("decoder", {}) or {}
    return str(decoder_cfg.get("type", "cross_attention")).strip().lower()


def _validate_checkpoint_runtime_compatibility(checkpoint_meta: dict, runtime_cfg: dict) -> None:
    checkpoint_use_static_features = None
    checkpoint_use_bathymetry = None
    checkpoint_decoder_type = None

    checkpoint_cfg = checkpoint_meta.get("config")
    if isinstance(checkpoint_cfg, dict):
        checkpoint_data_cfg = checkpoint_cfg.get("data", {}) or {}
        checkpoint_use_static_features = bool(checkpoint_data_cfg.get("use_static_features", True))
        checkpoint_use_bathymetry = bool(checkpoint_data_cfg.get("use_bathymetry", False))
        checkpoint_decoder_type = _resolve_runtime_decoder_type(checkpoint_cfg)

    runtime_data_cfg = runtime_cfg.get("data", {}) or {}
    runtime_use_static_features = bool(runtime_data_cfg.get("use_static_features", True))
    runtime_use_bathymetry = bool(runtime_data_cfg.get("use_bathymetry", False))
    runtime_decoder_type = _resolve_runtime_decoder_type(runtime_cfg)

    if (
        checkpoint_use_static_features is not None
        and checkpoint_use_static_features != runtime_use_static_features
    ):
        raise ValueError(
            "Checkpoint/config static-feature mismatch: "
            f"checkpoint use_static_features={checkpoint_use_static_features}, "
            f"runtime use_static_features={runtime_use_static_features}. "
            "Use the matching training config for evaluation."
        )
    if (
        checkpoint_use_bathymetry is not None
        and checkpoint_use_bathymetry != runtime_use_bathymetry
    ):
        raise ValueError(
            "Checkpoint/config bathymetry mismatch: "
            f"checkpoint use_bathymetry={checkpoint_use_bathymetry}, "
            f"runtime use_bathymetry={runtime_use_bathymetry}. "
            "Use the matching training config for evaluation."
        )
    if checkpoint_decoder_type is not None and checkpoint_decoder_type != runtime_decoder_type:
        raise ValueError(
            "Checkpoint/config decoder mismatch: "
            f"checkpoint decoder.type={checkpoint_decoder_type}, "
            f"runtime decoder.type={runtime_decoder_type}. "
            "Use the matching training config for evaluation."
        )


def _load_checkpoint_compat(ckpt_path: Path, device: torch.device) -> object:
    """Load checkpoints robustly across PyTorch versions.

    PyTorch 2.6 changed torch.load default `weights_only` from False to True,
    which can fail on older checkpoints containing trusted metadata objects.
    """
    load_kwargs = {"map_location": device}

    torch_version_cls = getattr(getattr(torch, "torch_version", None), "TorchVersion", None)
    safe_globals = getattr(getattr(torch, "serialization", None), "safe_globals", None)
    if callable(safe_globals) and torch_version_cls is not None:
        safe_ctx = safe_globals([torch_version_cls])
    else:
        safe_ctx = nullcontext()

    try:
        with safe_ctx:
            return torch.load(ckpt_path, weights_only=True, **load_kwargs)
    except TypeError:
        # Older PyTorch versions may not support `weights_only`.
        return torch.load(ckpt_path, **load_kwargs)
    except pickle.UnpicklingError as exc:
        # Fall back to full checkpoint load for trusted local artifacts.
        print(
            "WARNING: weights-only checkpoint load failed; falling back to weights_only=False. "
            "Only use trusted checkpoint files."
        )
        try:
            return torch.load(ckpt_path, weights_only=False, **load_kwargs)
        except TypeError:
            return torch.load(ckpt_path, **load_kwargs)
        except Exception as fallback_exc:
            raise RuntimeError(
                "Checkpoint load failed in both safe (weights_only=True) and fallback "
                "(weights_only=False) modes."
            ) from fallback_exc


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


def read_yaml(path: str) -> dict:
    return read_yaml_config(path)


def _read_json(path: Path):
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


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


def _angle_channels(angle_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    radians = np.deg2rad(angle_deg)
    return np.sin(radians), np.cos(radians)


def _recover_hybrid_metrics_arrays(
    pred,
    target,
    target_scaler_stats: tuple[float, float] | None,
) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(pred, dict) or not isinstance(target, dict):
        return np.asarray(pred), np.asarray(target)

    hs_mean, hs_scale = target_scaler_stats if target_scaler_stats is not None else (0.0, 1.0)
    hs_true = target["hs"].detach().cpu().numpy().reshape(-1) * hs_scale + hs_mean
    hs_pred = pred["hs"].detach().cpu().numpy().reshape(-1) * hs_scale + hs_mean

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


def _extract_model_inputs(
    batch: dict,
    device: torch.device,
    *,
    use_static_features: bool = True,
    use_source_geometry_features: bool = True,
) -> dict:
    return {
        "x_dynamic": batch["x_dynamic"].to(device) if "x_dynamic" in batch else None,
        "x_dynamic_sources": batch["x_dynamic_sources"].to(device)
        if "x_dynamic_sources" in batch
        else None,
        "source_geometry": (
            batch["source_geometry"].to(device)
            if use_source_geometry_features and "source_geometry" in batch
            else None
        ),
        "x_static": batch["x_static"].to(device)
        if use_static_features and "x_static" in batch
        else None,
        "x_bathy": batch["x_bathy"].to(device) if "x_bathy" in batch else None,
    }


def infer(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    point_centric_dir: str | None = None,
    transfer_representation: str = "legacy",
    residual_cfg: dict | None = None,
    transfer_tp_min: float = 0.5,
    transfer_tp_max: float = 30.0,
) -> Dict[str, np.ndarray | List[str]]:
    """Run model inference for one split."""
    model.eval()

    preds = []
    trues = []
    transfer_preds = []
    transfer_trues = []
    sites: List[str] = []
    timestamps: List[str] = []
    time_index: List[int] = []
    target_scaler_stats = (
        _load_target_scaler_stats(point_centric_dir) if point_centric_dir else None
    )
    transfer_scaler_stats = (
        load_transfer_scaler_stats(point_centric_dir) if point_centric_dir else {}
    )
    is_hybrid = False

    with torch.no_grad():
        for batch in loader:
            model_inputs = _extract_model_inputs(
                batch,
                device,
                use_static_features=bool(getattr(model, "use_static_features", True)),
                use_source_geometry_features=bool(
                    getattr(model, "use_source_geometry_features", True)
                ),
            )
            y = _to_device_batch(batch["y"], device)

            pred = model(
                model_inputs["x_dynamic"],
                model_inputs["x_static"],
                x_bathy=model_inputs["x_bathy"],
                x_dynamic_sources=model_inputs["x_dynamic_sources"],
                source_geometry=model_inputs["source_geometry"],
            )
            if isinstance(pred, dict) and isinstance(y, dict):
                is_hybrid = True
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
                    preds.append(physical_matrix_to_metric_channels(physical_pred))
                    trues.append(physical_matrix_to_metric_channels(physical_true))
                    transfer_preds.append(transfer_pred)
                    transfer_trues.append(transfer_true)
                else:
                    pred_arr, true_arr = _recover_hybrid_metrics_arrays(
                        pred, y, target_scaler_stats
                    )
                    preds.append(pred_arr)
                    trues.append(true_arr)
            else:
                preds.append(pred.detach().cpu().numpy())
                trues.append(y.detach().cpu().numpy())

            sites.extend(list(batch["site"]))
            timestamps.extend(list(batch["timestamp"]))
            idx_arr = batch["time_index"]
            if torch.is_tensor(idx_arr):
                time_index.extend([int(v) for v in idx_arr.cpu().numpy().tolist()])
            else:
                time_index.extend([int(v) for v in idx_arr])

    y_pred = np.concatenate(preds, axis=0)
    y_true = np.concatenate(trues, axis=0)

    return {
        "pred": y_pred,
        "true": y_true,
        "is_hybrid": is_hybrid,
        "transfer_pred": np.concatenate(transfer_preds, axis=0)
        if transfer_preds
        else np.array([], dtype=np.float32),
        "transfer_true": np.concatenate(transfer_trues, axis=0)
        if transfer_trues
        else np.array([], dtype=np.float32),
        "site": sites,
        "timestamp": timestamps,
        "time_index": np.asarray(time_index, dtype=int),
    }


def resolve_output_columns(config: dict) -> List[str]:
    data_cfg = (config.get("data", {}) or {}) if isinstance(config, dict) else {}
    return list(
        data_cfg.get(
            "output_columns",
            ["hs", "tp", "dir_sin", "dir_cos", "dp_sin", "dp_cos"],
        )
    )


def _load_json(path: Path):
    try:
        with path.open("r") as fh:
            return json.load(fh)
    except Exception:
        return None


def _get_target_standard_scaler_from_metadata(pc_dir: Path):
    meta = _load_json(pc_dir / "point_centric_metadata.json")
    if not isinstance(meta, dict):
        return {}, None

    norm = meta.get("normalization") or {}
    target_scaler = norm.get("target_scaler") or {}
    if not isinstance(target_scaler, dict):
        return {}, None

    feature_names = target_scaler.get("feature_names")
    if target_scaler.get("method") != "standard" or not isinstance(feature_names, list):
        return {}, None

    required = ["mean", "scale", "fill_values"]
    if not all(k in target_scaler for k in required):
        return {}, None

    return target_scaler, (pc_dir / "point_centric_metadata.json")


def _build_standard_scaler_for_columns(scaler_meta: dict, columns: List[str]):
    """Reconstruct a fitted sklearn StandardScaler aligned to `columns`."""
    feature_names = list(scaler_meta.get("feature_names", []) or [])
    mean = np.asarray(scaler_meta.get("mean", []), dtype=float)
    scale = np.asarray(scaler_meta.get("scale", []), dtype=float)

    if not feature_names or len(feature_names) != len(mean) or len(mean) != len(scale):
        return None

    pos = {name: i for i, name in enumerate(feature_names)}
    missing = [c for c in columns if c not in pos]
    if missing:
        raise KeyError(f"Target scaler metadata missing configured output columns: {missing}")

    order = [pos[c] for c in columns]
    mean_aligned = mean[order]
    scale_aligned = scale[order]
    safe_scale = scale_aligned.copy()
    safe_scale[safe_scale == 0.0] = 1.0

    try:
        from sklearn.preprocessing import StandardScaler
    except Exception:
        return None

    scaler = StandardScaler()
    scaler.mean_ = mean_aligned
    scaler.scale_ = safe_scale
    scaler.var_ = np.square(safe_scale)
    scaler.n_features_in_ = int(len(columns))
    scaler.n_samples_seen_ = np.array([1], dtype=np.int64)
    return scaler


def log_physical_prediction_bounds(
    arr_true: np.ndarray, arr_pred: np.ndarray, cols: List[str]
) -> None:
    """Log physical-unit ranges for quick sanity checks."""
    units = {"hs": "m", "tp": "s"}
    for var in ["hs", "tp"]:
        if var not in cols:
            continue
        i = cols.index(var)
        t = arr_true[:, i]
        p = arr_pred[:, i]
        t = t[np.isfinite(t)]
        p = p[np.isfinite(p)]
        if t.size == 0 or p.size == 0:
            print(f"WARNING: {var} contains no finite values after inverse transform")
            continue

        print(
            f"{var.upper()} physical range | "
            f"target=[{float(np.nanmin(t)):.3f}, {float(np.nanmax(t)):.3f}] {units[var]} | "
            f"pred=[{float(np.nanmin(p)):.3f}, {float(np.nanmax(p)):.3f}] {units[var]}"
        )

    if "hs" in cols:
        hs_i = cols.index("hs")
        pred_hs = arr_pred[:, hs_i]
        pred_hs = pred_hs[np.isfinite(pred_hs)]
        if pred_hs.size > 0:
            hs_min = float(np.nanmin(pred_hs))
            hs_max = float(np.nanmax(pred_hs))
            if hs_min < 0.0 or hs_max > 15.0:
                print(
                    "WARNING: Predicted Hs falls outside typical physical bounds [0.0, 15.0] m: "
                    f"[{hs_min:.3f}, {hs_max:.3f}]"
                )
            else:
                print("Predicted Hs is within typical physical bounds [0.0, 15.0] m.")


def log_per_site_test_diagnostics(
    arr_true: np.ndarray,
    arr_pred: np.ndarray,
    site_names: List[str],
    cols: List[str],
) -> None:
    """Log per-site variability and direction quality to detect collapse."""
    if len(site_names) != arr_true.shape[0]:
        return

    site_arr = np.asarray(site_names, dtype=str)
    unique_sites = sorted(set(site_arr.tolist()))
    if not unique_sites:
        return

    hs_idx = cols.index("hs") if "hs" in cols else None
    tp_idx = cols.index("tp") if "tp" in cols else None

    dir_pair = None
    if "dir_sin" in cols and "dir_cos" in cols:
        dir_pair = (cols.index("dir_sin"), cols.index("dir_cos"))

    dp_pair = None
    if "dp_sin" in cols and "dp_cos" in cols:
        dp_pair = (cols.index("dp_sin"), cols.index("dp_cos"))

    print("Per-site test diagnostics (variance ratios and directional RMSE):")
    for site in unique_sites:
        mask = site_arr == site
        n = int(np.count_nonzero(mask))
        if n == 0:
            continue

        parts = [f"site={site}", f"n={n}"]

        if hs_idx is not None:
            hs_t = arr_true[mask, hs_idx]
            hs_p = arr_pred[mask, hs_idx]
            hs_t = hs_t[np.isfinite(hs_t)]
            hs_p = hs_p[np.isfinite(hs_p)]
            if hs_t.size > 1 and hs_p.size > 1:
                hs_std_t = float(np.nanstd(hs_t))
                hs_std_p = float(np.nanstd(hs_p))
                hs_ratio = hs_std_p / max(hs_std_t, 1e-8)
                parts.append(f"hs_std_ratio={hs_ratio:.3f}")
                if hs_ratio < 0.10:
                    parts.append("hs_flatline=YES")

        if tp_idx is not None:
            tp_t = arr_true[mask, tp_idx]
            tp_p = arr_pred[mask, tp_idx]
            tp_t = tp_t[np.isfinite(tp_t)]
            tp_p = tp_p[np.isfinite(tp_p)]
            if tp_t.size > 1 and tp_p.size > 1:
                tp_std_t = float(np.nanstd(tp_t))
                tp_std_p = float(np.nanstd(tp_p))
                tp_ratio = tp_std_p / max(tp_std_t, 1e-8)
                parts.append(f"tp_std_ratio={tp_ratio:.3f}")
                if tp_ratio < 0.10:
                    parts.append("tp_flatline=YES")

        if dir_pair is not None:
            i_sin, i_cos = dir_pair
            t_ang = np.arctan2(arr_true[mask, i_sin], arr_true[mask, i_cos])
            p_ang = np.arctan2(arr_pred[mask, i_sin], arr_pred[mask, i_cos])
            good = np.isfinite(t_ang) & np.isfinite(p_ang)
            if np.any(good):
                d = np.arctan2(np.sin(p_ang[good] - t_ang[good]), np.cos(p_ang[good] - t_ang[good]))
                parts.append(f"dir_rmse_deg={float(np.sqrt(np.mean(np.degrees(d) ** 2))):.2f}")

        if dp_pair is not None:
            i_sin, i_cos = dp_pair
            t_ang = np.arctan2(arr_true[mask, i_sin], arr_true[mask, i_cos])
            p_ang = np.arctan2(arr_pred[mask, i_sin], arr_pred[mask, i_cos])
            good = np.isfinite(t_ang) & np.isfinite(p_ang)
            if np.any(good):
                d = np.arctan2(np.sin(p_ang[good] - t_ang[good]), np.cos(p_ang[good] - t_ang[good]))
                parts.append(f"dp_rmse_deg={float(np.sqrt(np.mean(np.degrees(d) ** 2))):.2f}")

        print("  " + " | ".join(parts))


def _build_col_mapping(stats: dict, output_cols: List[str]):
    # Only Hs/Tp are ever inverse-transformed during evaluation export.
    # Direction outputs must remain sin/cos and are normalized separately.
    mapping = {}
    feat_range = stats.get("feature_range", [0.0, 1.0])
    cols = stats.get("columns")
    invertible_cols = ["hs", "tp"]

    if cols and isinstance(cols, list):
        for col in invertible_cols:
            if col not in output_cols or col not in cols:
                continue
            idx = cols.index(col)
            if "min" in stats and "max" in stats:
                mapping[col] = {
                    "method": stats.get("method", "minmax"),
                    "min": stats["min"][idx],
                    "max": stats["max"][idx],
                    "feature_range": feat_range,
                }
            elif "mean" in stats and "std" in stats:
                mapping[col] = {
                    "method": "zscore",
                    "mean": stats["mean"][idx],
                    "std": stats["std"][idx],
                }
        return mapping

    # Fallback when no explicit columns are present: assume first two are hs/tp.
    if "min" in stats and "max" in stats:
        mins = stats["min"]
        maxs = stats["max"]
        for i, colname in enumerate(["hs", "tp"]):
            if colname in output_cols and i < len(mins):
                mapping[colname] = {
                    "method": stats.get("method", "minmax"),
                    "min": mins[i],
                    "max": maxs[i],
                    "feature_range": feat_range,
                }
    return mapping


def _get_mapping_from_point_centric_metadata(pc_dir: Path):
    meta = _load_json(pc_dir / "point_centric_metadata.json")
    if not isinstance(meta, dict):
        return {}
    norm = meta.get("normalization") or {}

    vars_map = norm.get("vars")
    if vars_map is None:
        vars_map = (norm.get("target_scaler") or {}).get("vars")
    if not isinstance(vars_map, dict):
        return {}

    out = {}
    for key in ["hs", "tp"]:
        info = vars_map.get(key)
        if isinstance(info, dict):
            out[key] = info
    return out


def _find_stats_mapping(
    pc_dir: Path, output_columns: List[str], explicit_stats_file: str | None = None
):
    candidates: List[Path] = []

    if explicit_stats_file:
        candidates.append(Path(explicit_stats_file))
    else:
        candidates.extend(
            [
                pc_dir / "normalization_stats.json",
                pc_dir.parent / "normalization_stats.json",
            ]
        )

    for path in candidates:
        if not path.exists():
            continue
        stats = _load_json(path)
        if not isinstance(stats, dict):
            continue
        mapping = _build_col_mapping(stats, output_columns)
        if "hs" in mapping and "tp" in mapping:
            return mapping, path
    return {}, None


def _looks_normalized_hs_tp(arr_true: np.ndarray, arr_pred: np.ndarray, cols: List[str]) -> bool:
    if "hs" not in cols or "tp" not in cols:
        return False

    hs_i = cols.index("hs")
    tp_i = cols.index("tp")
    hs_vals = np.concatenate([arr_true[:, hs_i], arr_pred[:, hs_i]], axis=0)
    tp_vals = np.concatenate([arr_true[:, tp_i], arr_pred[:, tp_i]], axis=0)
    hs_vals = hs_vals[np.isfinite(hs_vals)]
    tp_vals = tp_vals[np.isfinite(tp_vals)]

    if hs_vals.size == 0 or tp_vals.size == 0:
        return False

    hs_ok = float(np.nanmin(hs_vals)) >= -0.05 and float(np.nanmax(hs_vals)) <= 1.05
    tp_ok = float(np.nanmin(tp_vals)) >= -0.05 and float(np.nanmax(tp_vals)) <= 1.05
    return hs_ok and tp_ok


def _inverse_transform_array(arr: np.ndarray, cols: List[str], mapping: dict):
    out = arr.copy()
    for i, col in enumerate(cols):
        if col not in mapping:
            continue
        info = mapping[col]
        try:
            if info.get("method") == "minmax":
                a, b = info.get("feature_range", [0.0, 1.0])
                mn = float(info["min"])
                mx = float(info["max"])
                if b == a or np.isnan(mn) or np.isnan(mx):
                    continue
                out[:, i] = ((out[:, i] - a) / (b - a)) * (mx - mn) + mn
            elif info.get("method") == "zscore":
                mean = float(info["mean"])
                std = float(info["std"])
                out[:, i] = out[:, i] * std + mean
        except Exception:
            continue
    return out


def normalize_direction_pairs(arr: np.ndarray, cols: List[str]):
    """Normalize predicted direction sin/cos pairs to unit vectors."""
    out = arr.copy()
    normalized_pairs: List[str] = []
    for sin_col, cos_col in [("dir_sin", "dir_cos"), ("dp_sin", "dp_cos")]:
        if sin_col not in cols or cos_col not in cols:
            continue

        i_sin = cols.index(sin_col)
        i_cos = cols.index(cos_col)

        s = out[:, i_sin]
        c = out[:, i_cos]
        norms = np.hypot(s, c)
        finite = np.isfinite(s) & np.isfinite(c) & np.isfinite(norms)
        if not np.any(finite):
            continue

        safe = norms.copy()
        safe[(~np.isfinite(safe)) | (safe == 0.0)] = 1.0
        out[finite, i_sin] = s[finite] / safe[finite]
        out[finite, i_cos] = c[finite] / safe[finite]
        out[:, i_sin] = np.clip(out[:, i_sin], -1.0, 1.0)
        out[:, i_cos] = np.clip(out[:, i_cos], -1.0, 1.0)
        normalized_pairs.append(f"({sin_col},{cos_col})")

    return out, normalized_pairs


def prepare_evaluation_outputs(
    config: dict,
    outputs: Dict[str, np.ndarray | List[str]],
    *,
    stats_file: str | None = None,
    split_for_logging: str | None = None,
    log_site_diagnostics: bool = False,
) -> dict[str, object]:
    """Convert raw inference outputs into physical-unit evaluation arrays."""
    cfg = resolve_config(config)
    data_cfg = cfg.get("data", {}) or {}
    targets_cfg = resolve_targets_config(data_cfg)
    output_columns = resolve_output_columns(cfg)

    y_pred = np.asarray(outputs["pred"])
    y_true = np.asarray(outputs["true"])
    transfer_pred = outputs.get("transfer_pred")
    transfer_true = outputs.get("transfer_true")
    site_names = [str(site) for site in (outputs.get("site", []) or [])]

    if len(output_columns) != y_pred.shape[1]:
        raise ValueError(
            "Configured output_columns length does not match model output dimension: "
            f"{len(output_columns)} vs {y_pred.shape[1]}"
        )

    pc_dir = Path(data_cfg.get("point_centric_dir", "data/processed/point_centric_demo"))
    if outputs.get("is_hybrid", False):
        target_scaler, target_scaler_source = None, None
    else:
        target_scaler, target_scaler_source = _get_target_standard_scaler_from_metadata(pc_dir)

    if str(targets_cfg.get("mode", "physical")).strip().lower() != "physical":
        print(
            "Transfer evaluation reconstructs physical nearshore predictions directly from "
            "predicted transfer targets and stored Y_reference values. "
            "No additional target-scaler inversion is applied."
        )
    elif target_scaler:
        scaler = _build_standard_scaler_for_columns(target_scaler, output_columns)
        if scaler is None:
            raise RuntimeError(
                "Target scaler metadata found, but sklearn StandardScaler could not be reconstructed. "
                "Install scikit-learn in the active environment."
            )
        y_true = scaler.inverse_transform(y_true)
        y_pred = scaler.inverse_transform(y_pred)
        print(
            "Applied inverse normalization via StandardScaler.inverse_transform() "
            f"using metadata from {target_scaler_source}"
        )
    else:
        col_mapping = _get_mapping_from_point_centric_metadata(pc_dir)
        mapping_source: Path | None = (
            (pc_dir / "point_centric_metadata.json") if col_mapping else None
        )

        if not col_mapping:
            col_mapping, mapping_source = _find_stats_mapping(pc_dir, output_columns, stats_file)

        needs_inverse = _looks_normalized_hs_tp(y_true, y_pred, output_columns)

        if col_mapping and needs_inverse:
            y_true = _inverse_transform_array(y_true, output_columns, col_mapping)
            y_pred = _inverse_transform_array(y_pred, output_columns, col_mapping)
            print(f"Applied inverse normalization for columns ['hs', 'tp'] using {mapping_source}")
        elif col_mapping:
            print("Hs/Tp already appear un-normalized; skipping inverse transform.")
        elif needs_inverse:
            raise RuntimeError(
                "Hs/Tp values appear normalized, but no dataset-local scaler mapping was found. "
                "Rebuild point-centric metadata with normalization vars or pass --stats-file "
                "to a matching normalization_stats.json for this dataset."
            )
        else:
            print(
                "Physical-mode evaluation found unnormalized dataset outputs and no dataset-local "
                "Hs/Tp scaler mapping, so predictions are exported exactly as stored."
            )

    log_physical_prediction_bounds(y_true, y_pred, output_columns)
    if log_site_diagnostics:
        log_per_site_test_diagnostics(y_true, y_pred, site_names, output_columns)

    y_pred, normalized_pairs = normalize_direction_pairs(y_pred, output_columns)
    if normalized_pairs:
        print("Normalized predicted direction pairs to unit vectors:", ", ".join(normalized_pairs))

    metrics = compute_wave_metrics(y_true=y_true, y_pred=y_pred)
    if (
        isinstance(transfer_pred, np.ndarray)
        and transfer_pred.size > 0
        and isinstance(transfer_true, np.ndarray)
        and transfer_true.size > 0
    ):
        metrics.update(_compute_transfer_metrics(transfer_true, transfer_pred))

    if split_for_logging:
        print(
            f"{split_for_logging} metrics | "
            f"Hs(R2={metrics['hs_r2']:.4f}, RMSE={metrics['hs_rmse']:.4f}) | "
            f"Tp(R2={metrics['tp_r2']:.4f}, RMSE={metrics['tp_rmse']:.4f}) | "
            f"Dir(R2={metrics['dir_r2']:.4f}, RMSEdeg={metrics['dir_rmse_deg']:.4f}) | "
            f"Dp(R2={metrics['dp_r2']:.4f}, RMSEdeg={metrics['dp_rmse_deg']:.4f})"
        )
        if (
            isinstance(transfer_pred, np.ndarray)
            and transfer_pred.size > 0
            and isinstance(transfer_true, np.ndarray)
            and transfer_true.size > 0
        ):
            print(
                "Transfer metrics | "
                f"log_hs_ratio_RMSE={metrics['log_hs_ratio_rmse']:.4f} | "
                f"tp_delta_RMSE={metrics['tp_delta_rmse']:.4f} | "
                f"dir_delta_RMSEdeg={metrics['dir_delta_rmse_deg']:.4f} | "
                f"dp_delta_RMSEdeg={metrics['dp_delta_rmse_deg']:.4f}"
            )

    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "metrics": metrics,
        "output_columns": output_columns,
        "transfer_pred": transfer_pred,
        "transfer_true": transfer_true,
    }


def build_breaking_diagnostics(
    *,
    site_names: List[str],
    target_hs: np.ndarray,
    pred_hs: np.ndarray,
    target_sites: List[str],
    local_depth_m: np.ndarray | None,
    local_breaking_hs_cap: np.ndarray,
    local_breaking_cap_valid: np.ndarray,
) -> tuple[pd.DataFrame, float | None]:
    """Build per-site local-breaking diagnostics from physical Hs predictions."""
    site_series = np.asarray(site_names, dtype=str)
    target_hs = np.asarray(target_hs, dtype=np.float64)
    pred_hs = np.asarray(pred_hs, dtype=np.float64)
    local_cap_vec = np.asarray(local_breaking_hs_cap, dtype=np.float64)
    local_valid_vec = np.asarray(local_breaking_cap_valid, dtype=np.float64)
    local_depth_vec = (
        np.asarray(local_depth_m, dtype=np.float64)
        if local_depth_m is not None
        else np.full(len(target_sites), np.nan, dtype=np.float64)
    )
    site_to_idx = {str(site): idx for idx, site in enumerate(target_sites)}

    rows = []
    global_target_exceed_mask: List[bool] = []
    for site in target_sites:
        site_key = str(site)
        site_idx = site_to_idx.get(site_key)
        if site_idx is None:
            continue

        cap = float(local_cap_vec[site_idx]) if site_idx < local_cap_vec.shape[0] else float("nan")
        depth = (
            float(local_depth_vec[site_idx])
            if site_idx < local_depth_vec.shape[0]
            else float("nan")
        )
        valid = int(
            site_idx < local_valid_vec.shape[0]
            and np.isfinite(local_valid_vec[site_idx])
            and float(local_valid_vec[site_idx]) > 0.0
        )

        mask = site_series == site_key
        site_target = target_hs[mask]
        site_pred = pred_hs[mask]
        finite_mask = np.isfinite(site_target) & np.isfinite(site_pred)
        site_target = site_target[finite_mask]
        site_pred = site_pred[finite_mask]

        pct_target_exceeds_cap = np.nan
        pct_pred_exceeds_cap = np.nan
        mean_pred_excess = np.nan
        max_pred_excess = np.nan
        if valid == 1 and np.isfinite(cap) and site_target.size > 0:
            target_exceed = site_target > cap
            pred_exceed = site_pred > cap
            pred_excess = np.maximum(site_pred - cap, 0.0)
            pct_target_exceeds_cap = float(np.mean(target_exceed) * 100.0)
            pct_pred_exceeds_cap = float(np.mean(pred_exceed) * 100.0)
            mean_pred_excess = float(np.mean(pred_excess))
            max_pred_excess = float(np.max(pred_excess))
            global_target_exceed_mask.extend(target_exceed.tolist())

        rows.append(
            {
                "site": site_key,
                "local_depth_m": depth,
                "local_breaking_hs_cap": cap,
                "local_breaking_cap_valid": valid,
                "pct_target_exceeds_cap": pct_target_exceeds_cap,
                "pct_pred_exceeds_cap": pct_pred_exceeds_cap,
                "mean_pred_excess": mean_pred_excess,
                "max_pred_excess": max_pred_excess,
            }
        )

    diagnostics_df = pd.DataFrame(rows)
    if not global_target_exceed_mask:
        return diagnostics_df, None
    exceed_pct = float(np.mean(np.asarray(global_target_exceed_mask, dtype=np.float64)) * 100.0)
    return diagnostics_df, exceed_pct


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate coastal-transformer checkpoint on a split"
    )
    parser.add_argument("--config", default="configs/training.yaml", help="Path to YAML config")
    parser.add_argument("--checkpoint", default=None, help="Path to .pt checkpoint")
    parser.add_argument(
        "-n", "--name", default=None, help="Results folder name (or path) to load checkpoint from"
    )
    parser.add_argument(
        "--split", default="test", choices=["train", "val", "test"], help="Dataset split"
    )
    parser.add_argument("--device", default="auto", help="cuda | cpu | auto")
    parser.add_argument(
        "--out-dir",
        default=None,
        help=(
            "Optional output directory for predictions. "
            "If omitted, files are saved next to the selected checkpoint (run folder)."
        ),
    )
    parser.add_argument(
        "--stats-file",
        default=None,
        help=(
            "Optional path to normalization stats JSON used to inverse-transform Hs/Tp. "
            "Use this only when point-centric metadata does not contain normalization vars."
        ),
    )
    args = parser.parse_args()

    cfg = resolve_config(read_yaml(args.config))
    cfg = normalize_runtime_config_paths(cfg, config_path=args.config)
    targets_cfg = resolve_targets_config(cfg.get("data", {}) or {})

    resolved_device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if args.device == "auto" and not torch.cuda.is_available():
        resolved_device = "cpu"
    device = torch.device(resolved_device)

    log_cfg = cfg.get("logging", {})
    # Resolve checkpoint default from logging.output_dir and checkpoint_name
    configured_out = Path(log_cfg.get("output_dir", "results/coastal_transformer"))
    ckpt_name = str(log_cfg.get("checkpoint_name", "coastal_transformer_best.pt"))

    # If user supplied -n/--name, interpret it as either a full path or a
    # folder name under the same parent as configured_out (commonly `results/<name>`).
    if args.name:
        name_path = Path(args.name)
        if name_path.is_absolute() or name_path.parent != Path("."):
            ckpt_dir = name_path
        else:
            ckpt_dir = configured_out.parent / args.name
        ckpt_default = ckpt_dir / ckpt_name
    else:
        ckpt_default = configured_out / ckpt_name

    metadata_payload = None
    metadata_path = (ckpt_default.parent / "training_run_metadata.json").resolve()
    if metadata_path.exists():
        maybe_payload = _read_json(metadata_path)
        if isinstance(maybe_payload, dict):
            metadata_payload = maybe_payload

    ckpt_path = Path(args.checkpoint) if args.checkpoint else ckpt_default
    if not ckpt_path.exists():
        # Try a fallback: if the target directory exists, pick the newest .pt file inside.
        candidate_dir = ckpt_path.parent
        if candidate_dir.exists() and candidate_dir.is_dir():
            pts = sorted(candidate_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
            if pts:
                ckpt_path = pts[0]
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        print(f"Using checkpoint (fallback): {ckpt_path}")

    data_cfg = cfg.get("data", {})
    requested_bathy_channels, requested_bathy_in_channels = _resolve_runtime_bathy_request(cfg)
    arrays = load_point_centric_arrays(
        data_cfg.get("point_centric_dir", "data/processed/point_centric_demo"),
        bathy_channels=requested_bathy_channels,
        bathy_in_channels=requested_bathy_in_channels,
    )
    arrays = apply_runtime_static_ablation(arrays, resolve_static_ablation_config_path(cfg))

    loader, ds = build_split_dataloader(arrays, cfg, split_name=args.split, shuffle=False)
    if len(ds) == 0:
        raise RuntimeError(
            f"No samples in split '{args.split}'. "
            "Check seq_len, split indices and site lists (including unseen NORAC test sites)."
        )

    sample = ds[0]
    print(
        "Evaluation target mode: "
        f"{targets_cfg.get('mode', 'physical')} "
        f"(transfer_reference={targets_cfg.get('transfer_reference', 'nearest_bulk')}, "
        f"transfer_representation={targets_cfg.get('transfer_representation', 'legacy')})"
    )
    if str(targets_cfg.get("transfer_representation", "legacy")) == "residual_correction":
        print(f"Residual correction config: {targets_cfg.get('residual_correction', {}) or {}}")
    dynamic_input_dim = int(sample["x_dynamic"].shape[-1]) if "x_dynamic" in sample else 0
    source_dynamic_input_dim = (
        int(sample["x_dynamic_sources"].shape[-1]) if "x_dynamic_sources" in sample else None
    )
    source_geometry_input_dim = (
        int(sample["source_geometry"].shape[-1]) if "source_geometry" in sample else None
    )
    model = build_model_from_config(
        config=cfg,
        dynamic_input_dim=dynamic_input_dim,
        static_input_dim=int(sample["x_static"].shape[-1]) if "x_static" in sample else 0,
        output_dim=4 if isinstance(sample["y"], dict) else int(sample["y"].shape[-1]),
        dynamic_feature_names=getattr(arrays, "dynamic_feature_names", None),
        source_dynamic_input_dim=source_dynamic_input_dim,
        source_geometry_input_dim=source_geometry_input_dim,
        source_feature_names=getattr(arrays, "source_feature_names", None),
    ).to(device)

    if not isinstance(sample["y"], dict) and int(sample["y"].shape[-1]) != 6:
        raise ValueError(
            "Evaluation expects 6 target features in this order: "
            "[hs, tp, dir_sin, dir_cos, dp_sin, dp_cos]."
        )

    if isinstance(metadata_payload, dict) and is_independent_target_composite_metadata(
        metadata_payload
    ):

        def _build_model() -> torch.nn.Module:
            return build_model_from_config(
                config=cfg,
                dynamic_input_dim=dynamic_input_dim,
                static_input_dim=int(sample["x_static"].shape[-1]) if "x_static" in sample else 0,
                output_dim=4 if isinstance(sample["y"], dict) else int(sample["y"].shape[-1]),
                dynamic_feature_names=getattr(arrays, "dynamic_feature_names", None),
                source_dynamic_input_dim=source_dynamic_input_dim,
                source_geometry_input_dim=source_geometry_input_dim,
                source_feature_names=getattr(arrays, "source_feature_names", None),
            ).to(device)

        model = load_model_from_training_metadata(
            training_metadata=metadata_payload,
            build_model=_build_model,
            checkpoint_loader=lambda path: _load_checkpoint_compat(path, device),
            state_dict_getter=_extract_state_dict,
            state_dict_loader=lambda target_model, state_dict: _load_state_dict_best_effort(
                model=target_model, state_dict=state_dict
            ),
        )
        print(f"Loaded independent-target composite run from: {metadata_path}")
    else:
        checkpoint = _load_checkpoint_compat(ckpt_path, device)
        if _looks_like_state_dict(checkpoint):
            raise ValueError(
                "Legacy bare state_dict checkpoints are no longer supported. "
                "Please load a checkpoint with model metadata generated by current coastal training "
                "(checkpoint_version >= 2, model_api_version=coastal_transformer_v1 or coastal_transformer_v2)."
            )

        checkpoint_meta = checkpoint if isinstance(checkpoint, dict) else {}
        checkpoint_api = str(checkpoint_meta.get("model_api_version", "")).strip().lower()
        checkpoint_arch = str(checkpoint_meta.get("architecture", "")).strip().lower()
        if "dual_branch" in checkpoint_api or checkpoint_arch in {
            "lstm",
            "gru",
            "tcn",
            "transformer",
        }:
            raise ValueError(
                "Legacy checkpoints from removed architecture families are not supported in coastal-only mode. "
                f"Found model_api_version='{checkpoint_api or 'missing'}', architecture='{checkpoint_arch or 'missing'}'. "
                "Retrain with model.architecture=coastal_transformer and evaluate the new checkpoint."
            )

        _validate_checkpoint_runtime_compatibility(checkpoint_meta, cfg)

        if isinstance(checkpoint, dict):
            if checkpoint_api and checkpoint_api not in {
                "coastal_transformer_v1",
                "coastal_transformer_v2",
            }:
                raise ValueError(
                    "Unsupported checkpoint model_api_version for coastal-only runtime: "
                    f"'{checkpoint_api}'. Expected 'coastal_transformer_v1' or 'coastal_transformer_v2'."
                )
            if checkpoint_arch and checkpoint_arch not in {
                "coastal_transformer",
                "coastal-conditioned-transformer",
                "coastal_conditioned_transformer",
                "coastal_transformer_v1",
            }:
                raise ValueError(
                    "Unsupported checkpoint architecture for coastal-only runtime: "
                    f"'{checkpoint_arch}'."
                )

        state_dict = _extract_state_dict(checkpoint)
        load_mode, load_details = _load_state_dict_best_effort(model=model, state_dict=state_dict)
        if load_mode == "partial":
            print(
                "WARNING: Loaded checkpoint with partial fallback "
                f"(loaded={load_details['loaded']}, missing={load_details['missing']}, "
                f"unexpected={load_details['unexpected']}, shape_mismatch={load_details['shape_mismatch']})."
            )
        else:
            print(
                f"Loaded checkpoint with strict compatibility ({load_details['loaded']} tensors)."
            )

    outputs = infer(
        model=model,
        loader=loader,
        device=device,
        point_centric_dir=data_cfg.get("point_centric_dir", ""),
        transfer_representation=str(targets_cfg.get("transfer_representation", "legacy")),
        residual_cfg=targets_cfg.get("residual_correction", {}) or {},
        transfer_tp_min=float(targets_cfg.get("tp_min", 0.5)),
        transfer_tp_max=float(targets_cfg.get("tp_max", 30.0)),
    )
    y_pred = outputs["pred"]
    y_true = outputs["true"]
    transfer_pred = outputs.get("transfer_pred")
    transfer_true = outputs.get("transfer_true")

    # Determine output columns once so scaling and serialization stay aligned.
    output_columns = data_cfg.get(
        "output_columns",
        ["hs", "tp", "dir_sin", "dir_cos", "dp_sin", "dp_cos"],
    )
    if len(output_columns) != y_pred.shape[1]:
        raise ValueError(
            "Configured output_columns length does not match model output dimension: "
            f"{len(output_columns)} vs {y_pred.shape[1]}"
        )

    # Locate deterministic scaler metadata and invert predictions to physical units.
    def _load_json(path: Path):
        try:
            import json

            with path.open("r") as fh:
                return json.load(fh)
        except Exception:
            return None

    def _get_target_standard_scaler_from_metadata(pc_dir: Path):
        meta = _load_json(pc_dir / "point_centric_metadata.json")
        if not isinstance(meta, dict):
            return {}, None

        norm = meta.get("normalization") or {}
        target_scaler = norm.get("target_scaler") or {}
        if not isinstance(target_scaler, dict):
            return {}, None

        feature_names = target_scaler.get("feature_names")
        if target_scaler.get("method") != "standard" or not isinstance(feature_names, list):
            return {}, None

        required = ["mean", "scale", "fill_values"]
        if not all(k in target_scaler for k in required):
            return {}, None

        return target_scaler, (pc_dir / "point_centric_metadata.json")

    def _build_standard_scaler_for_columns(scaler_meta: dict, columns: List[str]):
        """Reconstruct a fitted sklearn StandardScaler aligned to `columns`."""
        feature_names = list(scaler_meta.get("feature_names", []) or [])
        mean = np.asarray(scaler_meta.get("mean", []), dtype=float)
        scale = np.asarray(scaler_meta.get("scale", []), dtype=float)

        if not feature_names or len(feature_names) != len(mean) or len(mean) != len(scale):
            return None

        pos = {name: i for i, name in enumerate(feature_names)}
        missing = [c for c in columns if c not in pos]
        if missing:
            raise KeyError(f"Target scaler metadata missing configured output columns: {missing}")

        order = [pos[c] for c in columns]
        mean_aligned = mean[order]
        scale_aligned = scale[order]
        safe_scale = scale_aligned.copy()
        safe_scale[safe_scale == 0.0] = 1.0

        try:
            from sklearn.preprocessing import StandardScaler
        except Exception:
            return None

        scaler = StandardScaler()
        scaler.mean_ = mean_aligned
        scaler.scale_ = safe_scale
        scaler.var_ = np.square(safe_scale)
        scaler.n_features_in_ = int(len(columns))
        scaler.n_samples_seen_ = np.array([1], dtype=np.int64)
        return scaler

    def _log_physical_prediction_bounds(
        arr_true: np.ndarray, arr_pred: np.ndarray, cols: List[str]
    ) -> None:
        """Log physical-unit ranges for quick sanity checks."""
        units = {"hs": "m", "tp": "s"}
        for var in ["hs", "tp"]:
            if var not in cols:
                continue
            i = cols.index(var)
            t = arr_true[:, i]
            p = arr_pred[:, i]
            t = t[np.isfinite(t)]
            p = p[np.isfinite(p)]
            if t.size == 0 or p.size == 0:
                print(f"WARNING: {var} contains no finite values after inverse transform")
                continue

            print(
                f"{var.upper()} physical range | "
                f"target=[{float(np.nanmin(t)):.3f}, {float(np.nanmax(t)):.3f}] {units[var]} | "
                f"pred=[{float(np.nanmin(p)):.3f}, {float(np.nanmax(p)):.3f}] {units[var]}"
            )

        if "hs" in cols:
            hs_i = cols.index("hs")
            pred_hs = arr_pred[:, hs_i]
            pred_hs = pred_hs[np.isfinite(pred_hs)]
            if pred_hs.size > 0:
                hs_min = float(np.nanmin(pred_hs))
                hs_max = float(np.nanmax(pred_hs))
                if hs_min < 0.0 or hs_max > 15.0:
                    print(
                        "WARNING: Predicted Hs falls outside typical physical bounds [0.0, 15.0] m: "
                        f"[{hs_min:.3f}, {hs_max:.3f}]"
                    )
                else:
                    print("Predicted Hs is within typical physical bounds [0.0, 15.0] m.")

    def _log_per_site_test_diagnostics(
        arr_true: np.ndarray,
        arr_pred: np.ndarray,
        site_names: List[str],
        cols: List[str],
    ) -> None:
        """Log per-site variability and direction quality to detect collapse."""
        if len(site_names) != arr_true.shape[0]:
            return

        site_arr = np.asarray(site_names, dtype=str)
        unique_sites = sorted(set(site_arr.tolist()))
        if not unique_sites:
            return

        hs_idx = cols.index("hs") if "hs" in cols else None
        tp_idx = cols.index("tp") if "tp" in cols else None

        dir_pair = None
        if "dir_sin" in cols and "dir_cos" in cols:
            dir_pair = (cols.index("dir_sin"), cols.index("dir_cos"))

        dp_pair = None
        if "dp_sin" in cols and "dp_cos" in cols:
            dp_pair = (cols.index("dp_sin"), cols.index("dp_cos"))

        print("Per-site test diagnostics (variance ratios and directional RMSE):")
        for site in unique_sites:
            mask = site_arr == site
            n = int(np.count_nonzero(mask))
            if n == 0:
                continue

            parts = [f"site={site}", f"n={n}"]

            if hs_idx is not None:
                hs_t = arr_true[mask, hs_idx]
                hs_p = arr_pred[mask, hs_idx]
                hs_t = hs_t[np.isfinite(hs_t)]
                hs_p = hs_p[np.isfinite(hs_p)]
                if hs_t.size > 1 and hs_p.size > 1:
                    hs_std_t = float(np.nanstd(hs_t))
                    hs_std_p = float(np.nanstd(hs_p))
                    hs_ratio = hs_std_p / max(hs_std_t, 1e-8)
                    parts.append(f"hs_std_ratio={hs_ratio:.3f}")
                    if hs_ratio < 0.10:
                        parts.append("hs_flatline=YES")

            if tp_idx is not None:
                tp_t = arr_true[mask, tp_idx]
                tp_p = arr_pred[mask, tp_idx]
                tp_t = tp_t[np.isfinite(tp_t)]
                tp_p = tp_p[np.isfinite(tp_p)]
                if tp_t.size > 1 and tp_p.size > 1:
                    tp_std_t = float(np.nanstd(tp_t))
                    tp_std_p = float(np.nanstd(tp_p))
                    tp_ratio = tp_std_p / max(tp_std_t, 1e-8)
                    parts.append(f"tp_std_ratio={tp_ratio:.3f}")
                    if tp_ratio < 0.10:
                        parts.append("tp_flatline=YES")

            if dir_pair is not None:
                i_sin, i_cos = dir_pair
                t_ang = np.arctan2(arr_true[mask, i_sin], arr_true[mask, i_cos])
                p_ang = np.arctan2(arr_pred[mask, i_sin], arr_pred[mask, i_cos])
                good = np.isfinite(t_ang) & np.isfinite(p_ang)
                if np.any(good):
                    d = np.arctan2(
                        np.sin(p_ang[good] - t_ang[good]), np.cos(p_ang[good] - t_ang[good])
                    )
                    parts.append(f"dir_rmse_deg={float(np.sqrt(np.mean(np.degrees(d) ** 2))):.2f}")

            if dp_pair is not None:
                i_sin, i_cos = dp_pair
                t_ang = np.arctan2(arr_true[mask, i_sin], arr_true[mask, i_cos])
                p_ang = np.arctan2(arr_pred[mask, i_sin], arr_pred[mask, i_cos])
                good = np.isfinite(t_ang) & np.isfinite(p_ang)
                if np.any(good):
                    d = np.arctan2(
                        np.sin(p_ang[good] - t_ang[good]), np.cos(p_ang[good] - t_ang[good])
                    )
                    parts.append(f"dp_rmse_deg={float(np.sqrt(np.mean(np.degrees(d) ** 2))):.2f}")

            print("  " + " | ".join(parts))

    def _build_col_mapping(stats: dict, output_cols: List[str]):
        # Only Hs/Tp are ever inverse-transformed during evaluation export.
        # Direction outputs must remain sin/cos and are normalized separately.
        mapping = {}
        feat_range = stats.get("feature_range", [0.0, 1.0])
        cols = stats.get("columns")
        invertible_cols = ["hs", "tp"]

        if cols and isinstance(cols, list):
            for col in invertible_cols:
                if col not in output_cols or col not in cols:
                    continue
                idx = cols.index(col)
                if "min" in stats and "max" in stats:
                    mapping[col] = {
                        "method": stats.get("method", "minmax"),
                        "min": stats["min"][idx],
                        "max": stats["max"][idx],
                        "feature_range": feat_range,
                    }
                elif "mean" in stats and "std" in stats:
                    mapping[col] = {
                        "method": "zscore",
                        "mean": stats["mean"][idx],
                        "std": stats["std"][idx],
                    }
            return mapping

        # Fallback when no explicit columns are present: assume first two are hs/tp.
        if "min" in stats and "max" in stats:
            mins = stats["min"]
            maxs = stats["max"]
            for i, colname in enumerate(["hs", "tp"]):
                if colname in output_cols and i < len(mins):
                    mapping[colname] = {
                        "method": stats.get("method", "minmax"),
                        "min": mins[i],
                        "max": maxs[i],
                        "feature_range": feat_range,
                    }
        return mapping

    def _get_mapping_from_point_centric_metadata(pc_dir: Path):
        meta = _load_json(pc_dir / "point_centric_metadata.json")
        if not isinstance(meta, dict):
            return {}
        norm = meta.get("normalization") or {}

        # Preferred field for new metadata builds.
        vars_map = norm.get("vars")
        # Backward compatibility if vars are nested under target_scaler.
        if vars_map is None:
            vars_map = (norm.get("target_scaler") or {}).get("vars")
        if not isinstance(vars_map, dict):
            return {}

        out = {}
        for key in ["hs", "tp"]:
            info = vars_map.get(key)
            if isinstance(info, dict):
                out[key] = info
        return out

    def _find_stats_mapping(pc_dir: Path, explicit_stats_file: str | None = None):
        candidates: List[Path] = []

        if explicit_stats_file:
            candidates.append(Path(explicit_stats_file))
        else:
            # Default to dataset-local files only; avoid unrelated global stats files.
            candidates.extend(
                [
                    pc_dir / "normalization_stats.json",
                    pc_dir.parent / "normalization_stats.json",
                ]
            )

        for p in candidates:
            if not p.exists():
                continue
            stats = _load_json(p)
            if not isinstance(stats, dict):
                continue
            mapping = _build_col_mapping(stats, output_columns)
            if "hs" in mapping and "tp" in mapping:
                return mapping, p
        return {}, None

    def _looks_normalized_hs_tp(
        arr_true: np.ndarray, arr_pred: np.ndarray, cols: List[str]
    ) -> bool:
        if "hs" not in cols or "tp" not in cols:
            return False

        hs_i = cols.index("hs")
        tp_i = cols.index("tp")
        hs_vals = np.concatenate([arr_true[:, hs_i], arr_pred[:, hs_i]], axis=0)
        tp_vals = np.concatenate([arr_true[:, tp_i], arr_pred[:, tp_i]], axis=0)
        hs_vals = hs_vals[np.isfinite(hs_vals)]
        tp_vals = tp_vals[np.isfinite(tp_vals)]

        if hs_vals.size == 0 or tp_vals.size == 0:
            return False

        hs_ok = float(np.nanmin(hs_vals)) >= -0.05 and float(np.nanmax(hs_vals)) <= 1.05
        tp_ok = float(np.nanmin(tp_vals)) >= -0.05 and float(np.nanmax(tp_vals)) <= 1.05
        return hs_ok and tp_ok

    def _inverse_transform_array(arr: np.ndarray, cols: List[str], mapping: dict):
        arr = arr.copy()
        for i, col in enumerate(cols):
            if col not in mapping:
                continue
            info = mapping[col]
            try:
                if info.get("method") == "minmax":
                    a, b = info.get("feature_range", [0.0, 1.0])
                    mn = float(info["min"])
                    mx = float(info["max"])
                    if b == a or np.isnan(mn) or np.isnan(mx):
                        continue
                    arr[:, i] = ((arr[:, i] - a) / (b - a)) * (mx - mn) + mn
                elif info.get("method") == "zscore":
                    mean = float(info["mean"])
                    std = float(info["std"])
                    arr[:, i] = arr[:, i] * std + mean
            except Exception:
                continue
        return arr

    def _normalize_direction_pairs(arr: np.ndarray, cols: List[str]):
        """Normalize predicted direction sin/cos pairs to unit vectors."""
        out = arr.copy()
        normalized_pairs: List[str] = []
        for sin_col, cos_col in [("dir_sin", "dir_cos"), ("dp_sin", "dp_cos")]:
            if sin_col not in cols or cos_col not in cols:
                continue

            i_sin = cols.index(sin_col)
            i_cos = cols.index(cos_col)

            s = out[:, i_sin]
            c = out[:, i_cos]
            norms = np.hypot(s, c)
            finite = np.isfinite(s) & np.isfinite(c) & np.isfinite(norms)
            if not np.any(finite):
                continue

            safe = norms.copy()
            safe[(~np.isfinite(safe)) | (safe == 0.0)] = 1.0
            out[finite, i_sin] = s[finite] / safe[finite]
            out[finite, i_cos] = c[finite] / safe[finite]

            # Keep strict sin/cos bounds for exported files.
            out[:, i_sin] = np.clip(out[:, i_sin], -1.0, 1.0)
            out[:, i_cos] = np.clip(out[:, i_cos], -1.0, 1.0)
            normalized_pairs.append(f"({sin_col},{cos_col})")

        return out, normalized_pairs

    pc_dir = Path(data_cfg.get("point_centric_dir", "data/processed/point_centric_demo"))
    if outputs.get("is_hybrid", False):
        target_scaler, target_scaler_source = None, None
    else:
        target_scaler, target_scaler_source = _get_target_standard_scaler_from_metadata(pc_dir)

    if str(targets_cfg.get("mode", "physical")).strip().lower() != "physical":
        print(
            "Transfer evaluation reconstructs physical nearshore predictions directly from "
            "predicted transfer targets and stored Y_reference values. "
            "No additional target-scaler inversion is applied."
        )
    elif target_scaler:
        scaler = _build_standard_scaler_for_columns(target_scaler, output_columns)
        if scaler is None:
            raise RuntimeError(
                "Target scaler metadata found, but sklearn StandardScaler could not be reconstructed. "
                "Install scikit-learn in the active environment."
            )

        # Explicitly invert z-scored targets back to physical units before
        # computing RMSE/MAE/R2 or exporting predictions.
        y_true = scaler.inverse_transform(y_true)
        y_pred = scaler.inverse_transform(y_pred)
        print(
            "Applied inverse normalization via StandardScaler.inverse_transform() "
            f"using metadata from {target_scaler_source}"
        )
    else:
        # Backward compatibility for older datasets that only saved Hs/Tp minmax stats.
        col_mapping = _get_mapping_from_point_centric_metadata(pc_dir)
        mapping_source: Path | None = (
            (pc_dir / "point_centric_metadata.json") if col_mapping else None
        )

        if not col_mapping:
            col_mapping, mapping_source = _find_stats_mapping(pc_dir, args.stats_file)

        needs_inverse = _looks_normalized_hs_tp(y_true, y_pred, output_columns)

        if col_mapping and needs_inverse:
            y_true = _inverse_transform_array(y_true, output_columns, col_mapping)
            y_pred = _inverse_transform_array(y_pred, output_columns, col_mapping)
            print(f"Applied inverse normalization for columns ['hs', 'tp'] using {mapping_source}")
        elif col_mapping:
            print("Hs/Tp already appear un-normalized; skipping inverse transform.")
        elif needs_inverse:
            raise RuntimeError(
                "Hs/Tp values appear normalized, but no dataset-local scaler mapping was found. "
                "Rebuild point-centric metadata with normalization vars or pass --stats-file "
                "to a matching normalization_stats.json for this dataset."
            )
        else:
            print(
                "Physical-mode evaluation found unnormalized dataset outputs and no dataset-local "
                "Hs/Tp scaler mapping, so predictions are exported exactly as stored."
            )

    _log_physical_prediction_bounds(y_true, y_pred, output_columns)
    if args.split == "test":
        _log_per_site_test_diagnostics(y_true, y_pred, outputs["site"], output_columns)

    # Ensure exported direction channels are valid sin/cos components.
    y_pred, normalized_pairs = _normalize_direction_pairs(y_pred, output_columns)
    if normalized_pairs:
        print("Normalized predicted direction pairs to unit vectors:", ", ".join(normalized_pairs))

    metrics = compute_wave_metrics(y_true=y_true, y_pred=y_pred)
    if (
        isinstance(transfer_pred, np.ndarray)
        and transfer_pred.size > 0
        and isinstance(transfer_true, np.ndarray)
        and transfer_true.size > 0
    ):
        metrics.update(_compute_transfer_metrics(transfer_true, transfer_pred))
    print(
        f"{args.split} metrics | "
        f"Hs(R2={metrics['hs_r2']:.4f}, RMSE={metrics['hs_rmse']:.4f}) | "
        f"Tp(R2={metrics['tp_r2']:.4f}, RMSE={metrics['tp_rmse']:.4f}) | "
        f"Dir(R2={metrics['dir_r2']:.4f}, RMSEdeg={metrics['dir_rmse_deg']:.4f}) | "
        f"Dp(R2={metrics['dp_r2']:.4f}, RMSEdeg={metrics['dp_rmse_deg']:.4f})"
    )
    if (
        isinstance(transfer_pred, np.ndarray)
        and transfer_pred.size > 0
        and isinstance(transfer_true, np.ndarray)
        and transfer_true.size > 0
    ):
        print(
            "Transfer metrics | "
            f"log_hs_ratio_RMSE={metrics['log_hs_ratio_rmse']:.4f} | "
            f"tp_delta_RMSE={metrics['tp_delta_rmse']:.4f} | "
            f"dir_delta_RMSEdeg={metrics['dir_delta_rmse_deg']:.4f} | "
            f"dp_delta_RMSEdeg={metrics['dp_delta_rmse_deg']:.4f}"
        )

    # By default write inference artifacts to the run/checkpoint folder.
    # This keeps train + eval outputs together under one results run directory.
    out_dir = Path(args.out_dir) if args.out_dir else ckpt_path.parent

    out_dir.mkdir(parents=True, exist_ok=True)

    if (
        hasattr(arrays, "local_breaking_hs_cap")
        and hasattr(arrays, "local_breaking_cap_valid")
        and arrays.local_breaking_hs_cap is not None
        and arrays.local_breaking_cap_valid is not None
        and "hs" in output_columns
    ):
        hs_idx = output_columns.index("hs")
        diagnostics_df, exceed_pct = build_breaking_diagnostics(
            site_names=list(outputs["site"]),
            target_hs=np.asarray(y_true[:, hs_idx], dtype=np.float64),
            pred_hs=np.asarray(y_pred[:, hs_idx], dtype=np.float64),
            target_sites=list(arrays.target_sites),
            local_depth_m=getattr(arrays, "local_depth_m", None),
            local_breaking_hs_cap=np.asarray(arrays.local_breaking_hs_cap, dtype=np.float64),
            local_breaking_cap_valid=np.asarray(arrays.local_breaking_cap_valid, dtype=np.float64),
        )
        diagnostics_path = out_dir / "breaking_diagnostics.csv"
        diagnostics_df.to_csv(diagnostics_path, index=False)
        print(f"Saved breaking diagnostics: {diagnostics_path}")

        if exceed_pct is not None and exceed_pct >= 25.0:
            print(
                "WARNING: Many observed target Hs samples exceed local breaking cap "
                f"({exceed_pct:.2f}% of valid-cap samples). "
                "This may indicate incorrect local depth values, sign convention issues, "
                "or that local-depth-only breaking is too strict."
            )

    records = {
        "split": [args.split] * len(outputs["site"]),
        "site": outputs["site"],
        "time_index": outputs["time_index"],
        "timestamp": outputs["timestamp"],
    }
    for i, col in enumerate(output_columns):
        records[f"target_{col}"] = y_true[:, i]
        records[f"pred_{col}"] = y_pred[:, i]
    if (
        isinstance(transfer_pred, np.ndarray)
        and transfer_pred.size > 0
        and isinstance(transfer_true, np.ndarray)
        and transfer_true.size > 0
    ):
        transfer_names = ["log_hs_ratio", "tp_delta", "dir_delta_deg", "dp_delta_deg"]
        for i, col in enumerate(transfer_names):
            records[f"target_transfer_{col}"] = transfer_true[:, i]
            records[f"pred_transfer_{col}"] = transfer_pred[:, i]

    df = pd.DataFrame(records)
    csv_path = out_dir / f"predictions_{args.split}.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved CSV: {csv_path}")

    save_nc = bool(cfg.get("save", {}).get("save_predictions_nc", True))
    if save_nc:
        try:
            import xarray as xr

            ds_nc = xr.Dataset(
                data_vars={
                    **{
                        f"target_{col}": ("sample", y_true[:, i])
                        for i, col in enumerate(output_columns)
                    },
                    **{
                        f"pred_{col}": ("sample", y_pred[:, i])
                        for i, col in enumerate(output_columns)
                    },
                },
                coords={
                    "sample": np.arange(len(df), dtype=int),
                    "site": ("sample", np.asarray(outputs["site"], dtype=str)),
                    "timestamp": ("sample", np.asarray(outputs["timestamp"], dtype=str)),
                    "time_index": ("sample", outputs["time_index"]),
                },
                attrs={"split": args.split, "checkpoint": str(ckpt_path)},
            )
            nc_path = out_dir / f"predictions_{args.split}.nc"
            ds_nc.to_netcdf(nc_path)
            print(f"Saved NetCDF: {nc_path}")
        except Exception as exc:
            print(f"Skipping NetCDF export (xarray/netcdf unavailable or failed): {exc}")


if __name__ == "__main__":
    main()
