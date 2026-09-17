"""Helpers for offshore-to-nearshore transfer target construction.

The stored transfer artifact remains a four-column residual target:

- `log_hs_ratio`
- `tp_delta`
- `dir_delta_deg`
- `dp_delta_deg`

The runtime can interpret those residuals in two representations:

- `legacy`: keep the existing training/runtime behavior where Hs/Tp residuals
  are standardized and only Tp is model-bounded.
- `residual_correction`: treat all four outputs as explicit residual-correction
  predictions in raw units, optionally bounded before reconstruction.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np


VALID_TARGET_MODES = {"physical", "transfer", "physical_and_transfer"}
VALID_TRANSFER_REFERENCES = {
    "weighted_bulk",
    "nearest_bulk",
    "weighted_swell",
    "nearest_swell",
    "weighted_partitioned",
}

PHYSICAL_TARGET_NAMES = ("hs", "tp", "dir", "dp")
TRANSFER_TARGET_NAMES = ("log_hs_ratio", "tp_delta", "dir_delta_deg", "dp_delta_deg")
REFERENCE_TARGET_NAMES = ("ref_hs", "ref_tp", "ref_dir", "ref_dp")
TRANSFER_SCALER_COLUMNS = ("log_hs_ratio", "tp_delta")
VALID_TRANSFER_REPRESENTATIONS = {"legacy", "residual_correction"}
VALID_BOUND_METHODS = {"tanh", "clamp", "none"}


def resolve_targets_config(data_cfg: Mapping[str, object] | None) -> dict:
    """Return validated target-mode config with backward-compatible defaults."""
    cfg = dict((data_cfg or {}).get("targets", {}) or {})
    mode = str(cfg.get("mode", "physical")).strip().lower()
    transfer_reference = str(cfg.get("transfer_reference", "nearest_bulk")).strip().lower()
    eps = float(cfg.get("eps", 1e-3))
    max_tp_delta = float(cfg.get("max_tp_delta", 15.0))
    tp_min = float(cfg.get("tp_min", 0.5))
    tp_max = float(cfg.get("tp_max", 30.0))
    physical_loss_weight = float(cfg.get("physical_loss_weight", 1.0))
    transfer_loss_weight = float(cfg.get("transfer_loss_weight", 0.5))
    transfer_representation = str(cfg.get("transfer_representation", "legacy")).strip().lower()
    residual_cfg = dict(cfg.get("residual_correction", {}) or {})
    residual_enabled = bool(
        residual_cfg.get("enabled", transfer_representation == "residual_correction")
    )
    residual_hs_form = str(residual_cfg.get("hs_form", "log_ratio")).strip().lower()
    residual_tp_form = str(residual_cfg.get("tp_form", "additive")).strip().lower()
    residual_direction_form = (
        str(residual_cfg.get("direction_form", "circular_additive")).strip().lower()
    )
    residual_dp_form = str(residual_cfg.get("dp_form", "circular_additive")).strip().lower()
    residual_eps_hs = float(residual_cfg.get("eps_hs", eps))
    residual_bound_method = str(residual_cfg.get("bound_method", "tanh")).strip().lower()
    residual_max_abs_log_hs = float(residual_cfg.get("max_abs_log_hs", 1.25))
    residual_max_abs_tp = float(residual_cfg.get("max_abs_tp", max_tp_delta))
    residual_max_abs_dir_deg = float(residual_cfg.get("max_abs_dir_deg", 120.0))
    residual_max_abs_dp_deg = float(residual_cfg.get("max_abs_dp_deg", 120.0))
    zero_init_output_head = bool(residual_cfg.get("zero_init_output_head", True))

    if mode not in VALID_TARGET_MODES:
        raise ValueError(
            f"targets.mode must be one of: physical, transfer, physical_and_transfer; got '{mode}'"
        )
    if transfer_reference not in VALID_TRANSFER_REFERENCES:
        raise ValueError(
            "targets.transfer_reference must be one of: "
            "weighted_bulk, nearest_bulk, weighted_swell, nearest_swell, weighted_partitioned; "
            f"got '{transfer_reference}'"
        )
    if eps <= 0.0:
        raise ValueError(f"targets.eps must be > 0, got {eps}")
    if transfer_representation not in VALID_TRANSFER_REPRESENTATIONS:
        raise ValueError(
            "targets.transfer_representation must be one of: legacy, residual_correction; "
            f"got '{transfer_representation}'"
        )
    if max_tp_delta <= 0.0:
        raise ValueError("targets.max_tp_delta must be > 0")
    if tp_min <= 0.0 or tp_max <= tp_min:
        raise ValueError(
            f"targets.tp_min/tp_max must satisfy 0 < tp_min < tp_max, got tp_min={tp_min}, tp_max={tp_max}"
        )
    if physical_loss_weight < 0.0:
        raise ValueError("targets.physical_loss_weight must be >= 0")
    if transfer_loss_weight < 0.0:
        raise ValueError("targets.transfer_loss_weight must be >= 0")
    if residual_hs_form != "log_ratio":
        raise ValueError("targets.residual_correction.hs_form currently supports only 'log_ratio'")
    if residual_tp_form != "additive":
        raise ValueError("targets.residual_correction.tp_form currently supports only 'additive'")
    if residual_direction_form != "circular_additive":
        raise ValueError(
            "targets.residual_correction.direction_form currently supports only 'circular_additive'"
        )
    if residual_dp_form != "circular_additive":
        raise ValueError(
            "targets.residual_correction.dp_form currently supports only 'circular_additive'"
        )
    if residual_eps_hs <= 0.0:
        raise ValueError("targets.residual_correction.eps_hs must be > 0")
    if residual_bound_method not in VALID_BOUND_METHODS:
        raise ValueError(
            "targets.residual_correction.bound_method must be one of: tanh, clamp, none; "
            f"got '{residual_bound_method}'"
        )
    if residual_max_abs_log_hs <= 0.0:
        raise ValueError("targets.residual_correction.max_abs_log_hs must be > 0")
    if residual_max_abs_tp <= 0.0:
        raise ValueError("targets.residual_correction.max_abs_tp must be > 0")
    if residual_max_abs_dir_deg <= 0.0:
        raise ValueError("targets.residual_correction.max_abs_dir_deg must be > 0")
    if residual_max_abs_dp_deg <= 0.0:
        raise ValueError("targets.residual_correction.max_abs_dp_deg must be > 0")

    if residual_enabled and transfer_representation == "legacy":
        transfer_representation = "residual_correction"

    return {
        "mode": mode,
        "transfer_reference": transfer_reference,
        "transfer_representation": transfer_representation,
        "eps": eps,
        "max_tp_delta": max_tp_delta,
        "tp_min": tp_min,
        "tp_max": tp_max,
        "physical_loss_weight": physical_loss_weight,
        "transfer_loss_weight": transfer_loss_weight,
        "residual_correction": {
            "enabled": bool(residual_enabled),
            "hs_form": residual_hs_form,
            "tp_form": residual_tp_form,
            "direction_form": residual_direction_form,
            "dp_form": residual_dp_form,
            "eps_hs": residual_eps_hs,
            "bound_method": residual_bound_method,
            "max_abs_log_hs": residual_max_abs_log_hs,
            "max_abs_tp": residual_max_abs_tp,
            "max_abs_dir_deg": residual_max_abs_dir_deg,
            "max_abs_dp_deg": residual_max_abs_dp_deg,
            "zero_init_output_head": zero_init_output_head,
        },
    }


def circular_difference_deg(target_deg, reference_deg):
    """Return wrapped signed delta in degrees in [-180, 180)."""
    target = np.asarray(target_deg, dtype=np.float64)
    reference = np.asarray(reference_deg, dtype=np.float64)
    return ((target - reference + 180.0) % 360.0) - 180.0


def wrap_360(values_deg):
    """Wrap angular values into [0, 360)."""
    values = np.asarray(values_deg, dtype=np.float64)
    return values % 360.0


def circular_add_deg(reference_deg, delta_deg):
    """Add wrapped angular deltas and return [0, 360) degrees."""
    reference = np.asarray(reference_deg, dtype=np.float64)
    delta = np.asarray(delta_deg, dtype=np.float64)
    return wrap_360(reference + delta)


def circular_delta_deg(target_deg, reference_deg):
    """Backward-compatible alias for circular_difference_deg."""
    return circular_difference_deg(target_deg, reference_deg)


def circular_weighted_mean_deg(
    directions_deg,
    weights,
    axis: int,
):
    """Weighted circular mean in degrees using sin/cos composition."""
    dirs = np.asarray(directions_deg, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if dirs.shape != w.shape:
        raise ValueError(
            "directions_deg and weights must have identical shapes for circular means; "
            f"got {dirs.shape} vs {w.shape}"
        )

    radians = np.deg2rad(dirs)
    sin_mean = np.sum(w * np.sin(radians), axis=axis)
    cos_mean = np.sum(w * np.cos(radians), axis=axis)
    return np.mod(np.rad2deg(np.arctan2(sin_mean, cos_mean)), 360.0)


def compute_weighted_reference(
    source_values,
    source_weights,
    variable_type: str,
    axis: int = 1,
):
    """Aggregate source references with arithmetic or circular weighting."""
    values = np.asarray(source_values, dtype=np.float64)
    weights = np.asarray(source_weights, dtype=np.float64)
    if values.shape != weights.shape:
        raise ValueError(
            "source_values and source_weights must have identical shapes; "
            f"got {values.shape} vs {weights.shape}"
        )

    kind = str(variable_type).strip().lower()
    weight_sum = np.sum(weights, axis=axis, keepdims=True)
    safe_sum = np.where(np.abs(weight_sum) < 1e-12, 1.0, weight_sum)
    normalized = weights / safe_sum

    if kind in {"direction", "circular", "angle"}:
        return circular_weighted_mean_deg(values, normalized, axis=axis)
    if kind in {"scalar", "linear"}:
        return np.sum(values * normalized, axis=axis)

    raise ValueError("variable_type must be one of: scalar, linear, direction, circular, angle")


def build_residual_targets(
    y_physical,
    offshore_reference,
    eps_hs: float = 1e-3,
):
    """Build residual correction targets from physical targets and references."""
    y = np.asarray(y_physical, dtype=np.float64)
    ref = np.asarray(offshore_reference, dtype=np.float64)
    if y.shape != ref.shape:
        raise ValueError(
            f"y_physical and offshore_reference must match, got {y.shape} vs {ref.shape}"
        )
    if y.ndim != 2 or y.shape[1] != 4:
        raise ValueError(f"Expected y_physical shape [N, 4], got {y.shape}")
    if eps_hs <= 0.0:
        raise ValueError("eps_hs must be > 0")

    out = np.empty_like(y, dtype=np.float64)
    out[:, 0] = np.log((y[:, 0] + eps_hs) / (ref[:, 0] + eps_hs))
    out[:, 1] = y[:, 1] - ref[:, 1]
    out[:, 2] = circular_difference_deg(y[:, 2], ref[:, 2])
    out[:, 3] = circular_difference_deg(y[:, 3], ref[:, 3])
    return out


def build_transfer_targets(
    y_physical,
    offshore_reference,
    eps: float = 1e-3,
):
    """Backward-compatible alias for residual target construction."""
    return build_residual_targets(y_physical, offshore_reference, eps_hs=eps)


def apply_residual_bounds(
    residuals,
    residual_cfg: Mapping[str, object] | None,
):
    """Apply optional tanh/clamp bounds to raw residual predictions."""
    values = np.asarray(residuals, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 4:
        raise ValueError(f"Expected residuals shape [N, 4], got {values.shape}")

    cfg = dict(residual_cfg or {})
    method = str(cfg.get("bound_method", "none")).strip().lower()
    max_abs = np.asarray(
        [
            float(cfg.get("max_abs_log_hs", 1.25)),
            float(cfg.get("max_abs_tp", 15.0)),
            float(cfg.get("max_abs_dir_deg", 120.0)),
            float(cfg.get("max_abs_dp_deg", 120.0)),
        ],
        dtype=np.float64,
    ).reshape(1, 4)

    if method == "none":
        return values
    if method == "tanh":
        return max_abs * np.tanh(values)
    if method == "clamp":
        return np.clip(values, -max_abs, max_abs)
    raise ValueError(f"Unsupported residual bound method '{method}'")


def reconstruct_from_residuals(
    pred_residuals,
    reference,
    *,
    residual_cfg: Mapping[str, object] | None = None,
):
    """Reconstruct physical targets from residual corrections and references."""
    transfer = apply_residual_bounds(pred_residuals, residual_cfg)
    ref = np.asarray(reference, dtype=np.float64)
    if transfer.shape != ref.shape:
        raise ValueError(
            f"pred_transfer and reference must match, got {transfer.shape} vs {ref.shape}"
        )
    if transfer.ndim != 2 or transfer.shape[1] != 4:
        raise ValueError(f"Expected pred_transfer shape [N, 4], got {transfer.shape}")

    out = np.empty_like(transfer, dtype=np.float64)
    out[:, 0] = ref[:, 0] * np.exp(transfer[:, 0])
    out[:, 1] = ref[:, 1] + transfer[:, 1]
    out[:, 2] = circular_add_deg(ref[:, 2], transfer[:, 2])
    out[:, 3] = circular_add_deg(ref[:, 3], transfer[:, 3])
    return out


def reconstruct_physical_from_transfer(
    pred_transfer,
    reference,
    *,
    tp_min: float = 0.5,
    tp_max: float = 30.0,
):
    """Backward-compatible legacy reconstruction for transfer predictions."""
    out = reconstruct_from_residuals(
        pred_transfer, reference, residual_cfg={"bound_method": "none"}
    )
    out[:, 1] = np.clip(out[:, 1], a_min=float(tp_min), a_max=float(tp_max))
    return out


def validate_target_name_block(names: Sequence[str], label: str) -> list[str]:
    out = [str(name) for name in names]
    if not out:
        raise ValueError(
            f"Target feature names are empty. Check targets.mode and target column config. ({label})"
        )
    return out
