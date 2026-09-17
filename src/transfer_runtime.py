"""Runtime helpers for transfer-target decoding and physical reconstruction."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

try:
    from preprocessing.transfer_targets import (
        reconstruct_from_residuals,
        reconstruct_physical_from_transfer,
    )
except Exception:
    from src.preprocessing.transfer_targets import (
        reconstruct_from_residuals,
        reconstruct_physical_from_transfer,
    )


def _angle_channels(angle_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    radians = np.deg2rad(angle_deg)
    return np.sin(radians), np.cos(radians)


def load_transfer_scaler_stats(point_centric_dir: str | None) -> Dict[int, Tuple[float, float]]:
    if not point_centric_dir:
        return {}

    metadata_path = Path(point_centric_dir) / "point_centric_metadata.json"
    if not metadata_path.exists():
        return {}

    try:
        with metadata_path.open("r") as fh:
            payload = json.load(fh) or {}
    except Exception:
        return {}

    scaler = (payload.get("normalization", {}) or {}).get("transfer_target_scaler", {}) or {}
    feature_names = list(scaler.get("feature_names", []) or [])
    means = list(scaler.get("mean", []) or [])
    scales = list(scaler.get("scale", []) or [])
    if len(feature_names) != len(means) or len(means) != len(scales):
        return {}

    out: Dict[int, Tuple[float, float]] = {}
    lut = {str(name): i for i, name in enumerate(feature_names)}
    for out_idx, name in enumerate(["log_hs_ratio", "tp_delta"]):
        if name not in lut:
            continue
        idx = lut[name]
        scale = float(scales[idx]) or 1.0
        out[int(out_idx)] = (float(means[idx]), scale)
    return out


def decode_transfer_predictions_numpy(
    pred_scaled: np.ndarray,
    transfer_scaler_stats: Dict[int, Tuple[float, float]] | None,
) -> np.ndarray:
    pred = np.asarray(pred_scaled, dtype=np.float64).copy()
    stats = transfer_scaler_stats or {}
    for idx in (0, 1):
        if idx not in stats:
            continue
        mean, scale = stats[idx]
        pred[:, idx] = (pred[:, idx] * scale) + mean
    return pred


def physical_matrix_to_metric_channels(raw_matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(raw_matrix, dtype=np.float64)
    dir_sin, dir_cos = _angle_channels(values[:, 2])
    dp_sin, dp_cos = _angle_channels(values[:, 3])
    return np.column_stack([values[:, 0], values[:, 1], dir_sin, dir_cos, dp_sin, dp_cos])


def recover_transfer_metric_arrays(
    pred: dict,
    target: dict,
    transfer_scaler_stats: Dict[int, Tuple[float, float]] | None,
    *,
    transfer_representation: str = "legacy",
    residual_cfg: dict | None = None,
    tp_min: float = 0.5,
    tp_max: float = 30.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pred_transfer_scaled = np.column_stack(
        [
            pred["log_hs_ratio"].detach().cpu().numpy().reshape(-1),
            pred["tp_delta"].detach().cpu().numpy().reshape(-1),
            pred["dir_delta_deg"].detach().cpu().numpy().reshape(-1),
            pred["dp_delta_deg"].detach().cpu().numpy().reshape(-1),
        ]
    )
    pred_transfer = decode_transfer_predictions_numpy(pred_transfer_scaled, transfer_scaler_stats)
    true_transfer = target["transfer"].detach().cpu().numpy().reshape(-1, 4)
    reference = target["reference"].detach().cpu().numpy().reshape(-1, 4)
    physical_true = target["physical"].detach().cpu().numpy().reshape(-1, 4)
    if str(transfer_representation).strip().lower() == "residual_correction":
        pred_transfer = pred_transfer_scaled
        physical_pred = reconstruct_from_residuals(
            pred_transfer,
            reference,
            residual_cfg=residual_cfg,
        )
    else:
        physical_pred = reconstruct_physical_from_transfer(
            pred_transfer,
            reference,
            tp_min=float(tp_min),
            tp_max=float(tp_max),
        )
    return physical_pred, physical_true, pred_transfer, true_transfer
