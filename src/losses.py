"""Custom loss functions for coastal-transformer wave downscaling."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Iterable, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

try:
    from preprocessing.transfer_targets import resolve_targets_config
except Exception:
    from src.preprocessing.transfer_targets import resolve_targets_config


_VALID_REDUCTIONS = {"mean", "sum", "none"}


def _validate_reduction(reduction: str) -> str:
    key = str(reduction).lower()
    if key not in _VALID_REDUCTIONS:
        raise ValueError(f"reduction must be one of {sorted(_VALID_REDUCTIONS)}; got '{reduction}'")
    return key


def _infer_pair_indices_from_output_columns(
    output_columns: Sequence[str] | None,
) -> Sequence[Tuple[int, int]]:
    """Infer (sin, cos) index pairs from output column names.

    Expected naming: <name>_sin and <name>_cos.
    """
    if not output_columns:
        return ((2, 3), (4, 5))

    pair_map: Dict[str, Dict[str, int]] = {}
    order: list[str] = []

    for idx, col in enumerate(output_columns):
        key = str(col).strip().lower()
        if key.endswith("_sin"):
            base = key[:-4]
            pair_map.setdefault(base, {})["sin"] = int(idx)
            if base not in order:
                order.append(base)
        elif key.endswith("_cos"):
            base = key[:-4]
            pair_map.setdefault(base, {})["cos"] = int(idx)
            if base not in order:
                order.append(base)

    parsed: list[Tuple[int, int]] = []
    for base in order:
        p = pair_map.get(base, {})
        if "sin" in p and "cos" in p:
            parsed.append((p["sin"], p["cos"]))

    if not parsed:
        return ((2, 3), (4, 5))
    return tuple(parsed)


def _parse_pair_indices(
    raw_pairs: Iterable[Sequence[int]] | Sequence[str] | None,
    output_columns: Sequence[str] | None = None,
) -> Sequence[Tuple[int, int]]:
    """Parse direction pair configuration.

    Supported forms:
    - None -> inferred from output columns (or default fallback)
    - [[2, 3], [4, 5]]
    - ["dir", "dp"] -> mapped to dir_sin/dir_cos and dp_sin/dp_cos
    """
    if raw_pairs is None:
        return _infer_pair_indices_from_output_columns(output_columns)

    items = list(raw_pairs)
    if not items:
        return _infer_pair_indices_from_output_columns(output_columns)

    output_columns_lut: Dict[str, int] = {}
    if output_columns:
        output_columns_lut = {
            str(col).strip().lower(): int(i) for i, col in enumerate(output_columns)
        }

    if all(isinstance(it, str) for it in items):
        parsed_named: list[Tuple[int, int]] = []
        for name_raw in items:
            name = str(name_raw).strip().lower()
            if name in {"direction", "dir"}:
                base = "dir"
            else:
                base = name

            sin_name = f"{base}_sin"
            cos_name = f"{base}_cos"
            if sin_name not in output_columns_lut or cos_name not in output_columns_lut:
                raise ValueError(
                    "Named direction pair could not be resolved from output columns: "
                    f"'{name_raw}'. Expected '{sin_name}' and '{cos_name}' in {list(output_columns or [])}."
                )
            parsed_named.append((output_columns_lut[sin_name], output_columns_lut[cos_name]))

        if not parsed_named:
            return _infer_pair_indices_from_output_columns(output_columns)
        return tuple(parsed_named)

    parsed = []
    for pair in items:
        if len(pair) != 2:
            raise ValueError(f"Each angular pair must contain exactly 2 indices, got: {pair}")
        parsed.append((int(pair[0]), int(pair[1])))

    if not parsed:
        return _infer_pair_indices_from_output_columns(output_columns)
    return tuple(parsed)


def _get_weight(
    weights_cfg: Dict[str, float],
    aliases: Sequence[str],
    default: float,
) -> float:
    normalized = {str(k).strip().lower(): v for k, v in (weights_cfg or {}).items()}
    for alias in aliases:
        key = str(alias).strip().lower()
        if key in normalized:
            return float(normalized[key])
    return float(default)


def _load_target_scaler_from_metadata(point_centric_dir: str | None) -> dict | None:
    if not point_centric_dir:
        return None

    metadata_path = Path(point_centric_dir) / "point_centric_metadata.json"
    if not metadata_path.exists():
        return None

    try:
        with metadata_path.open("r") as fh:
            payload = json.load(fh) or {}
    except Exception:
        return None

    norm = payload.get("normalization", {}) or {}
    scaler = norm.get("target_scaler", {}) or {}
    if not scaler:
        return None
    return scaler


def _load_transfer_target_scaler_from_metadata(point_centric_dir: str | None) -> dict | None:
    if not point_centric_dir:
        return None

    metadata_path = Path(point_centric_dir) / "point_centric_metadata.json"
    if not metadata_path.exists():
        return None

    try:
        with metadata_path.open("r") as fh:
            payload = json.load(fh) or {}
    except Exception:
        return None

    norm = payload.get("normalization", {}) or {}
    scaler = norm.get("transfer_target_scaler", {}) or {}
    if not scaler:
        return None
    return scaler


def _extract_output_scaler_stats(
    scaler_meta: dict | None,
    output_columns: Sequence[str],
) -> Dict[int, Tuple[float, float]]:
    """Map output indices to (mean, scale) statistics for inverse transform."""
    if not scaler_meta:
        return {}

    feature_names = list(scaler_meta.get("feature_names", []) or [])
    means = list(scaler_meta.get("mean", []) or [])
    scales = list(scaler_meta.get("scale", []) or [])
    if not feature_names:
        return {}
    if len(feature_names) != len(means) or len(feature_names) != len(scales):
        return {}

    lut = {str(name): i for i, name in enumerate(feature_names)}
    out: Dict[int, Tuple[float, float]] = {}
    for idx, col in enumerate(output_columns):
        key = str(col)
        if key not in lut:
            continue
        i = lut[key]
        scale = float(scales[i])
        if scale == 0.0:
            scale = 1.0
        out[int(idx)] = (float(means[i]), scale)
    return out


def _torch_circular_delta_deg(pred_deg: torch.Tensor, target_deg: torch.Tensor) -> torch.Tensor:
    return torch.remainder(pred_deg - target_deg + 180.0, 360.0) - 180.0


def _torch_circular_add_deg(reference_deg: torch.Tensor, delta_deg: torch.Tensor) -> torch.Tensor:
    return torch.remainder(reference_deg + delta_deg, 360.0)


def _torch_circular_loss_deg(pred_deg: torch.Tensor, target_deg: torch.Tensor) -> torch.Tensor:
    delta = _torch_circular_delta_deg(pred_deg, target_deg)
    radians = torch.deg2rad(delta)
    return 1.0 - torch.cos(radians)


class MSELoss(nn.Module):
    """Standard mean squared error loss."""

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self._impl = nn.MSELoss(reduction=_validate_reduction(reduction))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self._impl(pred, target)


class MAELoss(nn.Module):
    """Mean absolute error loss."""

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self._impl = nn.L1Loss(reduction=_validate_reduction(reduction))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self._impl(pred, target)


class HuberLoss(nn.Module):
    """Huber loss robust to outliers."""

    def __init__(self, delta: float = 1.0, reduction: str = "mean") -> None:
        super().__init__()
        self._impl = nn.HuberLoss(delta=float(delta), reduction=_validate_reduction(reduction))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self._impl(pred, target)


class SmoothL1Loss(nn.Module):
    """Smooth L1 loss with configurable beta."""

    def __init__(self, beta: float = 1.0, reduction: str = "mean") -> None:
        super().__init__()
        self._impl = nn.SmoothL1Loss(beta=float(beta), reduction=_validate_reduction(reduction))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self._impl(pred, target)


def _build_scalar_loss(
    name: str,
    reduction: str,
    huber_delta: float,
    smooth_l1_beta: float,
) -> nn.Module:
    key = str(name).lower()
    if key == "mse":
        return MSELoss(reduction=reduction)
    if key in {"mae", "l1"}:
        return MAELoss(reduction=reduction)
    if key == "huber":
        return HuberLoss(delta=huber_delta, reduction=reduction)
    if key in {"smooth_l1", "smoothl1"}:
        return SmoothL1Loss(beta=smooth_l1_beta, reduction=reduction)

    raise ValueError("Unsupported scalar loss. Expected one of: mse, mae, huber, smooth_l1")


class AngularLoss(nn.Module):
    """Angular loss from one or more (sin, cos) index pairs.

    Default pairs correspond to:
    - (2, 3): dir_sin, dir_cos
    - (4, 5): dp_sin, dp_cos
    """

    def __init__(
        self,
        pair_indices: Sequence[Tuple[int, int]] = ((2, 3), (4, 5)),
        reduction: str = "mean",
        squared: bool = True,
    ) -> None:
        super().__init__()
        self.pair_indices = [(int(i), int(j)) for i, j in pair_indices]
        self.reduction = _validate_reduction(reduction)
        self.squared = bool(squared)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pair_losses = []
        n_features = int(pred.shape[-1])

        for sin_idx, cos_idx in self.pair_indices:
            if sin_idx >= n_features or cos_idx >= n_features:
                continue

            pred_angle = torch.atan2(pred[..., sin_idx], pred[..., cos_idx])
            true_angle = torch.atan2(target[..., sin_idx], target[..., cos_idx])

            # Circular angular error in [-pi, pi].
            delta = torch.atan2(
                torch.sin(pred_angle - true_angle), torch.cos(pred_angle - true_angle)
            )
            pair_losses.append(delta.pow(2) if self.squared else delta.abs())

        if not pair_losses:
            raise ValueError(
                "AngularLoss could not find any valid (sin, cos) pairs in model output. "
                f"Configured pairs: {self.pair_indices}, output_dim={n_features}"
            )

        # Per-sample/per-step angular loss averaged over configured angle pairs.
        loss = torch.stack(pair_losses, dim=0).mean(dim=0)
        if self.reduction == "sum":
            return loss.sum()
        if self.reduction == "none":
            return loss
        return loss.mean()


class CoastalKendallVonMisesLoss(nn.Module):
    """Coastal multitask loss with Kendall weighting and cosine-distance angular terms.

    Task components:
    - hs: scalar regression loss on index 0
    - tp: scalar regression loss on index 1
    - dir: cosine-distance angular loss on first direction pair
    - dp: cosine-distance angular loss on second direction pair

    Total per-sample objective:
        sum_i w_i * (exp(-s_i) * L_i + s_i)
    where s_i are task log-variance terms (Kendall uncertainty weighting).
    """

    task_order: tuple[str, str, str, str] = ("hs", "tp", "dir", "dp")

    def __init__(
        self,
        base_regression_loss: str = "huber",
        huber_delta: float = 1.0,
        smooth_l1_beta: float = 1.0,
        direction_pair_indices: Sequence[Tuple[int, int]] = ((2, 3), (4, 5)),
        direction_kappa: Dict[str, float] | None = None,
        kappa_trainable: bool = False,
        include_log_i0: bool = True,
        initial_uncertainty: Dict[str, float] | None = None,
        uncertainty_trainable: bool = True,
        task_weights: Dict[str, float] | None = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.reduction = _validate_reduction(reduction)
        self.include_log_i0 = bool(include_log_i0)

        pairs = list(direction_pair_indices)
        if len(pairs) < 2:
            raise ValueError(
                "CoastalKendallVonMisesLoss requires two direction pairs for dir and dp. "
                f"Received: {pairs}"
            )
        self.dir_pair = (int(pairs[0][0]), int(pairs[0][1]))
        self.dp_pair = (int(pairs[1][0]), int(pairs[1][1]))

        self.scalar_loss = _build_scalar_loss(
            name=base_regression_loss,
            reduction="none",
            huber_delta=float(huber_delta),
            smooth_l1_beta=float(smooth_l1_beta),
        )

        default_kappa = {"dir": 4.0, "dp": 4.0}
        direction_kappa = direction_kappa or {}
        dir_kappa = float(direction_kappa.get("dir", default_kappa["dir"]))
        dp_kappa = float(direction_kappa.get("dp", default_kappa["dp"]))
        kappa_values = torch.tensor([dir_kappa, dp_kappa], dtype=torch.float32)
        if kappa_trainable:
            self.kappa = nn.Parameter(kappa_values)
        else:
            self.register_buffer("kappa", kappa_values)

        default_uncertainty = {name: 1.0 for name in self.task_order}
        initial_uncertainty = initial_uncertainty or {}
        init_log_vars = []
        for name in self.task_order:
            sigma = float(initial_uncertainty.get(name, default_uncertainty[name]))
            sigma = max(sigma, 1e-6)
            init_log_vars.append(math.log(sigma * sigma))
        log_var_values = torch.tensor(init_log_vars, dtype=torch.float32)
        if uncertainty_trainable:
            self.log_vars = nn.Parameter(log_var_values)
        else:
            self.register_buffer("log_vars", log_var_values)

        tw = task_weights or {}
        self.task_weights = {
            "hs": float(tw.get("hs", 1.0)),
            "tp": float(tw.get("tp", 1.0)),
            "dir": float(tw.get("dir", tw.get("direction", 1.0))),
            "dp": float(tw.get("dp", 1.0)),
        }

    @staticmethod
    def _per_sample(loss_tensor: torch.Tensor) -> torch.Tensor:
        if loss_tensor.ndim == 0:
            return loss_tensor.unsqueeze(0)
        if loss_tensor.ndim == 1:
            return loss_tensor
        return loss_tensor.reshape(loss_tensor.shape[0], -1).mean(dim=1)

    @staticmethod
    def _cosine_distance_from_pair(
        pred: torch.Tensor,
        target: torch.Tensor,
        pair_indices: Tuple[int, int],
    ) -> torch.Tensor:
        sin_idx, cos_idx = pair_indices

        pred_vec = torch.stack((pred[..., sin_idx], pred[..., cos_idx]), dim=-1)
        true_vec = torch.stack((target[..., sin_idx], target[..., cos_idx]), dim=-1)

        pred_unit = pred_vec / torch.clamp(pred_vec.norm(dim=-1, keepdim=True), min=1e-8)
        true_unit = true_vec / torch.clamp(true_vec.norm(dim=-1, keepdim=True), min=1e-8)

        cos_sim = torch.sum(pred_unit * true_unit, dim=-1)
        cos_sim = torch.clamp(cos_sim, min=-1.0, max=1.0)

        # Cosine distance is guaranteed non-negative and bounded in [0, 2].
        return 1.0 - cos_sim

    def _component_losses(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        n_features = int(pred.shape[-1])
        for sin_idx, cos_idx in (self.dir_pair, self.dp_pair):
            if sin_idx >= n_features or cos_idx >= n_features:
                raise ValueError(
                    "CoastalKendallVonMisesLoss direction pairs are out of range for model output. "
                    f"output_dim={n_features}, dir_pair={self.dir_pair}, dp_pair={self.dp_pair}"
                )

        hs_loss = self._per_sample(self.scalar_loss(pred[..., 0:1], target[..., 0:1]))
        tp_loss = self._per_sample(self.scalar_loss(pred[..., 1:2], target[..., 1:2]))

        dir_loss = self._per_sample(self._cosine_distance_from_pair(pred, target, self.dir_pair))
        dp_loss = self._per_sample(self._cosine_distance_from_pair(pred, target, self.dp_pair))

        return {
            "hs": hs_loss,
            "tp": tp_loss,
            "dir": dir_loss,
            "dp": dp_loss,
        }

    def components(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        losses = self._component_losses(pred, target)
        out = {name: losses[name].mean().detach() for name in self.task_order}
        out.update(
            {
                "log_var_hs": self.log_vars[0].detach(),
                "log_var_tp": self.log_vars[1].detach(),
                "log_var_dir": self.log_vars[2].detach(),
                "log_var_dp": self.log_vars[3].detach(),
                "kappa_dir": torch.clamp(self.kappa[0], min=1e-4, max=200.0).detach(),
                "kappa_dp": torch.clamp(self.kappa[1], min=1e-4, max=200.0).detach(),
            }
        )
        return out

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        losses = self._component_losses(pred, target)
        log_vars = self.log_vars.clamp(-4.0, 4.0)

        total = 0.0
        for i, name in enumerate(self.task_order):
            task_loss = losses[name]
            min_loss = float(task_loss.min().detach().cpu())
            assert bool(torch.all(task_loss >= 0.0)), f"{name} loss negative: {min_loss:.6g}"
            weight = self.task_weights[name]
            precision = torch.exp(-log_vars[i])
            total = total + weight * (precision * task_loss + log_vars[i])

        total = total + 0.01 * log_vars.pow(2).sum()

        if self.reduction == "none":
            return total
        if self.reduction == "sum":
            return total.sum()
        return total.mean()


class WeightedMultiTaskLoss(nn.Module):
    """Weighted multitask loss for [Hs, Tp, Sin_dir, Cos_dir, Sin_dp, Cos_dp] outputs.

    Loss terms:
    - Hs regression loss
    - Tp regression loss
    - Direction angular loss (from configured direction sin/cos pairs)

        Optional asymmetric extreme penalty:
        - Applied only to Hs loss where target Hs exceeds a threshold and prediction
            underestimates target Hs.
    """

    def __init__(
        self,
        hs_weight: float = 1.0,
        tp_weight: float = 1.0,
        direction_weight: float = 1.0,
        base_regression_loss: str = "mse",
        huber_delta: float = 1.0,
        smooth_l1_beta: float = 1.0,
        hs_extreme_threshold: float | None = None,
        hs_extreme_penalty: float = 1.0,
        direction_pair_indices: Sequence[Tuple[int, int]] = ((2, 3), (4, 5)),
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.hs_weight = float(hs_weight)
        self.tp_weight = float(tp_weight)
        self.direction_weight = float(direction_weight)
        self.hs_extreme_threshold = (
            None if hs_extreme_threshold is None else float(hs_extreme_threshold)
        )
        self.hs_extreme_penalty = float(hs_extreme_penalty)
        self.reduction = _validate_reduction(reduction)

        # Build component losses in "none" mode; this class owns final reduction.
        self.scalar_loss = _build_scalar_loss(
            name=base_regression_loss,
            reduction="none",
            huber_delta=float(huber_delta),
            smooth_l1_beta=float(smooth_l1_beta),
        )
        self.angular_loss = AngularLoss(
            pair_indices=direction_pair_indices,
            reduction="none",
            squared=True,
        )

    @staticmethod
    def _per_sample(loss_tensor: torch.Tensor) -> torch.Tensor:
        """Collapse loss tensor to one scalar per sample."""
        if loss_tensor.ndim == 0:
            return loss_tensor.unsqueeze(0)
        if loss_tensor.ndim == 1:
            return loss_tensor
        return loss_tensor.reshape(loss_tensor.shape[0], -1).mean(dim=1)

    def _apply_hs_extreme_penalty(
        self,
        hs_raw_loss: torch.Tensor,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        if self.hs_extreme_threshold is None or self.hs_extreme_penalty == 1.0:
            return hs_raw_loss

        hs_target = target[..., 0:1]
        hs_pred = pred[..., 0:1]
        extreme_under_mask = (hs_target > self.hs_extreme_threshold) & (hs_pred < hs_target)
        if not torch.any(extreme_under_mask):
            return hs_raw_loss

        penalty = torch.full_like(hs_raw_loss, self.hs_extreme_penalty)
        return hs_raw_loss * torch.where(extreme_under_mask, penalty, torch.ones_like(hs_raw_loss))

    def components(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Return detached component losses for logging/debugging."""
        hs_raw_loss = self.scalar_loss(pred[..., 0:1], target[..., 0:1])
        hs_raw_loss = self._apply_hs_extreme_penalty(hs_raw_loss, pred=pred, target=target)
        hs_loss = self._per_sample(hs_raw_loss)
        tp_loss = self._per_sample(self.scalar_loss(pred[..., 1:2], target[..., 1:2]))
        dir_loss = self._per_sample(self.angular_loss(pred, target))
        return {
            "hs": hs_loss.mean().detach(),
            "tp": tp_loss.mean().detach(),
            "direction": dir_loss.mean().detach(),
        }

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        hs_raw_loss = self.scalar_loss(pred[..., 0:1], target[..., 0:1])
        hs_raw_loss = self._apply_hs_extreme_penalty(hs_raw_loss, pred=pred, target=target)
        hs_loss = self._per_sample(hs_raw_loss)
        tp_loss = self._per_sample(self.scalar_loss(pred[..., 1:2], target[..., 1:2]))
        dir_loss = self._per_sample(self.angular_loss(pred, target))

        total = (
            self.hs_weight * hs_loss + self.tp_weight * tp_loss + self.direction_weight * dir_loss
        )

        if self.reduction == "none":
            return total
        if self.reduction == "sum":
            return total.sum()
        return total.mean()


class BlueprintMultiTaskLoss(nn.Module):
    """Blueprint-aligned multitask objective for coastal transformer training.

    - Energy heads (Hs/Tp): Huber loss with delta=0.5 by default.
    - Direction heads (Dir/Dp): bounded cosine-distance loss.
    - Direction losses are weighted by true unscaled wave energy (Hs^power).
    """

    def __init__(
        self,
        output_columns: Sequence[str],
        direction_pair_indices: Sequence[Tuple[int, int]] = ((2, 3), (4, 5)),
        huber_delta: float = 0.5,
        task_weights: Dict[str, float] | None = None,
        energy_weight_power: float = 2.0,
        energy_weight_epsilon: float = 0.0,
        normalize_energy_weights: bool = True,
        target_scaler_meta: dict | None = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.reduction = _validate_reduction(reduction)
        self.output_columns = list(
            output_columns or ["hs", "tp", "dir_sin", "dir_cos", "dp_sin", "dp_cos"]
        )
        self.huber_delta = float(huber_delta)
        self.energy_weight_power = float(energy_weight_power)
        self.energy_weight_epsilon = float(energy_weight_epsilon)
        self.normalize_energy_weights = bool(normalize_energy_weights)

        index_lut = {str(col).strip().lower(): i for i, col in enumerate(self.output_columns)}
        self.hs_idx = int(index_lut.get("hs", 0))
        self.tp_idx = int(index_lut.get("tp", 1))

        pairs = [(int(i), int(j)) for i, j in direction_pair_indices]
        if len(pairs) < 2:
            raise ValueError(
                "BlueprintMultiTaskLoss requires two direction pairs for dir and dp. "
                f"Received: {pairs}"
            )
        self.dir_pair = pairs[0]
        self.dp_pair = pairs[1]

        tw = task_weights or {}
        self.task_weights = {
            "hs": float(tw.get("hs", 1.0)),
            "tp": float(tw.get("tp", 1.0)),
            "dir": float(tw.get("dir", tw.get("direction", 1.0))),
            "dp": float(tw.get("dp", 1.0)),
        }

        self._scaler_stats = _extract_output_scaler_stats(
            scaler_meta=target_scaler_meta,
            output_columns=self.output_columns,
        )

    def _inverse_column(self, tensor: torch.Tensor, column_idx: int) -> torch.Tensor:
        stats = self._scaler_stats.get(int(column_idx), None)
        if stats is None:
            return tensor
        mean, scale = stats
        return (tensor * scale) + mean

    def _cosine_distance(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        pair: Tuple[int, int],
    ) -> torch.Tensor:
        sin_idx, cos_idx = pair

        p_sin = self._inverse_column(pred[..., sin_idx], sin_idx)
        p_cos = self._inverse_column(pred[..., cos_idx], cos_idx)
        t_sin = self._inverse_column(target[..., sin_idx], sin_idx)
        t_cos = self._inverse_column(target[..., cos_idx], cos_idx)

        p_vec = torch.stack((p_sin, p_cos), dim=-1)
        t_vec = torch.stack((t_sin, t_cos), dim=-1)

        p_unit = F.normalize(p_vec, dim=-1, eps=1e-8)
        t_unit = F.normalize(t_vec, dim=-1, eps=1e-8)

        cos_sim = torch.sum(p_unit * t_unit, dim=-1)
        cos_sim = torch.clamp(cos_sim, min=-1.0, max=1.0)
        return 1.0 - cos_sim

    def _wave_energy_weights(self, target: torch.Tensor) -> torch.Tensor:
        hs_true = self._inverse_column(target[..., self.hs_idx], self.hs_idx)
        energy = torch.clamp(hs_true, min=0.0).pow(self.energy_weight_power)
        energy = energy + self.energy_weight_epsilon

        if self.normalize_energy_weights:
            denom = torch.clamp(energy.mean().detach(), min=1e-8)
            energy = energy / denom
        return energy

    def task_losses(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        hs_loss = F.huber_loss(
            pred[..., self.hs_idx],
            target[..., self.hs_idx],
            delta=self.huber_delta,
            reduction="mean",
        )
        tp_loss = F.huber_loss(
            pred[..., self.tp_idx],
            target[..., self.tp_idx],
            delta=self.huber_delta,
            reduction="mean",
        )

        energy_w = self._wave_energy_weights(target)
        dir_dist = self._cosine_distance(pred, target, self.dir_pair)
        dp_dist = self._cosine_distance(pred, target, self.dp_pair)
        dir_loss = (dir_dist * energy_w).mean()
        dp_loss = (dp_dist * energy_w).mean()

        return {
            "hs": hs_loss,
            "tp": tp_loss,
            "dir": dir_loss,
            "dp": dp_loss,
        }

    def components(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        losses = self.task_losses(pred, target)
        return {k: v.detach() for k, v in losses.items()}

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        losses = self.task_losses(pred, target)
        total = (
            self.task_weights["hs"] * losses["hs"]
            + self.task_weights["tp"] * losses["tp"]
            + self.task_weights["dir"] * losses["dir"]
            + self.task_weights["dp"] * losses["dp"]
        )

        if self.reduction == "none":
            return torch.stack([losses["hs"], losses["tp"], losses["dir"], losses["dp"]])
        if self.reduction == "sum":
            return total
        return total


class BlueprintHybridLoss(nn.Module):
    """Hybrid objective for scaled Hs regression plus soft-class wave targets.

    The loss expects model outputs in the form:
    - hs: [B]
    - tp_log_probs: [B, num_tp_bins]
    - dir_log_probs: [B, num_dp_bins]
    - dp_log_probs: [B, num_dp_bins]

    And targets in the form returned by PointCentricWindowDataset:
    - hs: scaled regression target
    - tp_soft / dir_soft / dp_soft: probability distributions
    - tp_value / dir_value / dp_value: recovered physical targets (optional)
    """

    task_order: tuple[str, str, str, str] = ("hs", "tp", "dir", "dp")

    def __init__(
        self,
        huber_delta: float = 0.5,
        label_smoothing_sigma: float = 0.8,
        tp_bin_centers: Sequence[float] | None = None,
        dp_bin_centers: Sequence[float] | None = None,
        task_weights: Dict[str, float] | None = None,
        target_scaler_meta: dict | None = None,
        energy_weight_power: float = 2.0,
        log_components: bool = True,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.reduction = _validate_reduction(reduction)
        self.huber_delta = float(huber_delta)
        self.label_smoothing_sigma = float(label_smoothing_sigma)
        self.energy_weight_power = float(energy_weight_power)
        self.log_components = bool(log_components)
        self.last_components: Dict[str, float] = {}

        tp_source = (
            tp_bin_centers
            if tp_bin_centers is not None
            else np.linspace(0.0, 25.0, 32, dtype=np.float32)
        )
        dp_source = (
            dp_bin_centers
            if dp_bin_centers is not None
            else np.linspace(0.0, 360.0, 36, endpoint=False, dtype=np.float32)
        )
        tp_centers = torch.as_tensor(tp_source, dtype=torch.float32)
        dp_centers = torch.as_tensor(dp_source, dtype=torch.float32)
        self.register_buffer("tp_bin_centers", tp_centers, persistent=False)
        self.register_buffer("dp_bin_centers", dp_centers, persistent=False)

        tw = task_weights or {}
        self.task_weights = {
            "hs": float(tw.get("hs", 1.0)),
            "tp": float(tw.get("tp", 1.0)),
            "dir": float(tw.get("dir", tw.get("direction", 1.0))),
            "dp": float(tw.get("dp", 1.0)),
        }

        self._hs_scaler_stats = _extract_output_scaler_stats(
            scaler_meta=target_scaler_meta,
            output_columns=["hs"],
        )

    def _hs_physical(self, hs_scaled: torch.Tensor) -> torch.Tensor:
        stats = self._hs_scaler_stats.get(0)
        if stats is None:
            return torch.clamp(hs_scaled, min=0.0)
        mean, scale = stats
        return (hs_scaled * scale) + mean

    @staticmethod
    def _reduce_per_sample(loss_tensor: torch.Tensor) -> torch.Tensor:
        if loss_tensor.ndim == 0:
            return loss_tensor.unsqueeze(0)
        if loss_tensor.ndim == 1:
            return loss_tensor
        return loss_tensor.reshape(loss_tensor.shape[0], -1).mean(dim=1)

    def _classification_loss_per_sample(
        self,
        log_probs: torch.Tensor,
        target_probs: torch.Tensor,
        energy_w: torch.Tensor | None = None,
    ) -> torch.Tensor:
        per_sample = F.kl_div(log_probs, target_probs, reduction="none").sum(dim=-1)
        if energy_w is not None:
            per_sample = per_sample * energy_w
        return per_sample

    def _classification_loss(
        self,
        log_probs: torch.Tensor,
        target_probs: torch.Tensor,
        energy_w: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self._classification_loss_per_sample(log_probs, target_probs, energy_w).mean()

    @staticmethod
    def _to_float(value: torch.Tensor | float) -> float:
        if torch.is_tensor(value):
            return float(value.detach().cpu())
        return float(value)

    def _build_component_snapshot(
        self,
        losses: Dict[str, torch.Tensor],
        per_sample_losses: Dict[str, torch.Tensor] | None = None,
    ) -> Dict[str, float]:
        total = sum(self.task_weights[name] * losses[name] for name in self.task_order)
        snapshot: Dict[str, float] = {"total_loss": self._to_float(total)}

        for name in self.task_order:
            snapshot[f"{name}_loss"] = self._to_float(losses[name])
            if per_sample_losses is not None and name in per_sample_losses:
                snapshot[f"{name}_loss_per_sample"] = self._to_float(
                    self._reduce_per_sample(per_sample_losses[name]).mean()
                )

        return snapshot

    def _store_component_snapshot(
        self,
        losses: Dict[str, torch.Tensor],
        per_sample_losses: Dict[str, torch.Tensor] | None = None,
        total_override: torch.Tensor | None = None,
    ) -> None:
        if not self.log_components:
            self.last_components = {}
            return

        snapshot = self._build_component_snapshot(losses, per_sample_losses=per_sample_losses)
        if total_override is not None:
            snapshot["total_loss"] = self._to_float(total_override)
        self.last_components = snapshot

    def task_loss_details(
        self,
        pred: dict,
        target: dict,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        hs_per_sample = F.huber_loss(
            pred["hs"].reshape(-1),
            target["hs"].reshape(-1),
            delta=self.huber_delta,
            reduction="none",
        )
        hs_loss = hs_per_sample.mean()

        hs_physical = self._hs_physical(target["hs"].reshape(-1))
        energy_w = torch.clamp(hs_physical, min=0.0).pow(self.energy_weight_power)
        if torch.isfinite(energy_w).any():
            energy_w = energy_w / torch.clamp(energy_w.mean().detach(), min=1e-8)

        tp_per_sample = self._classification_loss_per_sample(
            pred["tp_log_probs"],
            target["tp_soft"],
            None,
        )
        dir_per_sample = self._classification_loss_per_sample(
            pred["dir_log_probs"],
            target["dir_soft"],
            energy_w,
        )
        dp_per_sample = self._classification_loss_per_sample(
            pred["dp_log_probs"],
            target["dp_soft"],
            energy_w,
        )

        losses = {
            "hs": hs_loss,
            "tp": tp_per_sample.mean(),
            "dir": dir_per_sample.mean(),
            "dp": dp_per_sample.mean(),
        }
        per_sample_losses = {
            "hs": hs_per_sample,
            "tp": tp_per_sample,
            "dir": dir_per_sample,
            "dp": dp_per_sample,
        }
        return losses, per_sample_losses

    def task_losses(self, pred: dict, target: dict) -> Dict[str, torch.Tensor]:
        losses, _ = self.task_loss_details(pred, target)
        return losses

    def record_task_loss_components(
        self,
        task_losses: Dict[str, torch.Tensor],
        per_sample_losses: Dict[str, torch.Tensor] | None = None,
        total_override: torch.Tensor | None = None,
    ) -> None:
        self._store_component_snapshot(
            losses=task_losses,
            per_sample_losses=per_sample_losses,
            total_override=total_override,
        )

    def forward(self, pred: dict, target: dict) -> torch.Tensor:
        losses, per_sample_losses = self.task_loss_details(pred, target)
        total = (
            self.task_weights["hs"] * losses["hs"]
            + self.task_weights["tp"] * losses["tp"]
            + self.task_weights["dir"] * losses["dir"]
            + self.task_weights["dp"] * losses["dp"]
        )
        self._store_component_snapshot(
            losses, per_sample_losses=per_sample_losses, total_override=total
        )

        if self.reduction == "none":
            return torch.stack([losses["hs"], losses["tp"], losses["dir"], losses["dp"]])
        if self.reduction == "sum":
            return total
        return total


class TransferHybridLoss(nn.Module):
    """Transfer-target loss with optional reconstructed physical-space supervision."""

    task_order: tuple[str, str, str, str] = ("hs", "tp", "dir", "dp")

    def __init__(
        self,
        transfer_scaler_meta: dict | None = None,
        task_weights: Dict[str, float] | None = None,
        physical_loss_weight: float = 1.0,
        transfer_loss_weight: float = 0.5,
        huber_delta: float = 0.5,
        target_mode: str = "physical_and_transfer",
        transfer_representation: str = "legacy",
        residual_config: dict | None = None,
        compute_physical_loss_on_reconstruction: bool = True,
        tp_min: float = 0.5,
        tp_max: float = 30.0,
        log_components: bool = True,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.reduction = _validate_reduction(reduction)
        self.transfer_scaler_meta = transfer_scaler_meta or {}
        self.physical_loss_weight = float(physical_loss_weight)
        self.transfer_loss_weight = float(transfer_loss_weight)
        self.huber_delta = float(huber_delta)
        self.target_mode = str(target_mode or "physical_and_transfer").strip().lower()
        self.transfer_representation = str(transfer_representation or "legacy").strip().lower()
        self.residual_config = dict(residual_config or {})
        self.compute_physical_loss_on_reconstruction = bool(compute_physical_loss_on_reconstruction)
        self.tp_min = float(tp_min)
        self.tp_max = float(tp_max)
        self.log_components = bool(log_components)
        self.last_components: Dict[str, float] = {}
        if self.target_mode not in {"transfer", "physical_and_transfer"}:
            raise ValueError(
                "TransferHybridLoss target_mode must be one of: transfer, physical_and_transfer; "
                f"got '{self.target_mode}'"
            )
        if self.transfer_representation not in {"legacy", "residual_correction"}:
            raise ValueError(
                "TransferHybridLoss transfer_representation must be one of: legacy, residual_correction; "
                f"got '{self.transfer_representation}'"
            )
        if self.tp_min <= 0.0 or self.tp_max <= self.tp_min:
            raise ValueError(
                f"tp_min/tp_max must satisfy 0 < tp_min < tp_max, got tp_min={self.tp_min}, tp_max={self.tp_max}"
            )

        tw = task_weights or {}
        self.task_weights = {
            "hs": float(tw.get("hs", 1.0)),
            "tp": float(tw.get("tp", 1.0)),
            "dir": float(tw.get("dir", tw.get("direction", 1.0))),
            "dp": float(tw.get("dp", 1.0)),
        }

        self._transfer_scaler_stats = _extract_output_scaler_stats(
            scaler_meta=self.transfer_scaler_meta,
            output_columns=["log_hs_ratio", "tp_delta"],
        )

    def _inverse_transfer_column(self, tensor: torch.Tensor, column_idx: int) -> torch.Tensor:
        stats = self._transfer_scaler_stats.get(int(column_idx), None)
        if stats is None:
            return tensor
        mean, scale = stats
        return (tensor * scale) + mean

    def _reconstruct_physical(self, pred: dict, target: dict) -> torch.Tensor:
        reference = target["reference"].reshape(-1, 4)
        if self.transfer_representation == "residual_correction":
            log_hs_ratio = pred["log_hs_ratio"].reshape(-1)
            tp_delta = pred["tp_delta"].reshape(-1)
            dir_delta = pred["dir_delta_deg"].reshape(-1)
            dp_delta = pred["dp_delta_deg"].reshape(-1)

            ref_hs = reference[:, 0]
            ref_tp = reference[:, 1]
            ref_dir = reference[:, 2]
            ref_dp = reference[:, 3]
            pred_hs = ref_hs * torch.exp(log_hs_ratio)
            pred_tp = ref_tp + tp_delta
            pred_dir = _torch_circular_add_deg(ref_dir, dir_delta)
            pred_dp = _torch_circular_add_deg(ref_dp, dp_delta)
            return torch.stack([pred_hs, pred_tp, pred_dir, pred_dp], dim=-1)

        log_hs_ratio = self._inverse_transfer_column(pred["log_hs_ratio"].reshape(-1), 0)
        tp_delta = self._inverse_transfer_column(pred["tp_delta"].reshape(-1), 1)
        dir_delta = pred["dir_delta_deg"].reshape(-1)
        dp_delta = pred["dp_delta_deg"].reshape(-1)

        ref_hs = reference[:, 0]
        ref_tp = reference[:, 1]
        ref_dir = reference[:, 2]
        ref_dp = reference[:, 3]

        pred_hs = ref_hs * torch.exp(log_hs_ratio)
        pred_tp = torch.clamp(ref_tp + tp_delta, min=self.tp_min, max=self.tp_max)
        pred_dir = _torch_circular_add_deg(ref_dir, dir_delta)
        pred_dp = _torch_circular_add_deg(ref_dp, dp_delta)
        return torch.stack([pred_hs, pred_tp, pred_dir, pred_dp], dim=-1)

    @staticmethod
    def _to_float(value: torch.Tensor | float) -> float:
        if torch.is_tensor(value):
            return float(value.detach().cpu())
        return float(value)

    def task_loss_details(
        self,
        pred: dict,
        target: dict,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        use_physical_supervision = (
            self.target_mode == "physical_and_transfer"
            and self.compute_physical_loss_on_reconstruction
        )
        if self.transfer_representation == "residual_correction":
            transfer_raw = target["transfer"].reshape(-1, 4)
            transfer_hs = F.huber_loss(
                pred["log_hs_ratio"].reshape(-1),
                transfer_raw[:, 0],
                delta=self.huber_delta,
                reduction="none",
            )
            transfer_tp = F.huber_loss(
                pred["tp_delta"].reshape(-1),
                transfer_raw[:, 1],
                delta=self.huber_delta,
                reduction="none",
            )
            transfer_dir = _torch_circular_loss_deg(
                pred["dir_delta_deg"].reshape(-1),
                transfer_raw[:, 2],
            )
            transfer_dp = _torch_circular_loss_deg(
                pred["dp_delta_deg"].reshape(-1),
                transfer_raw[:, 3],
            )
        else:
            transfer_hs = F.huber_loss(
                pred["log_hs_ratio"].reshape(-1),
                target["log_hs_ratio"].reshape(-1),
                delta=self.huber_delta,
                reduction="none",
            )
            transfer_tp = F.huber_loss(
                pred["tp_delta"].reshape(-1),
                target["tp_delta"].reshape(-1),
                delta=self.huber_delta,
                reduction="none",
            )
            transfer_dir = _torch_circular_loss_deg(
                pred["dir_delta_deg"].reshape(-1),
                target["dir_delta_deg"].reshape(-1),
            )
            transfer_dp = _torch_circular_loss_deg(
                pred["dp_delta_deg"].reshape(-1),
                target["dp_delta_deg"].reshape(-1),
            )

        if use_physical_supervision:
            reconstructed = self._reconstruct_physical(pred, target)
            physical = target["physical"].reshape(-1, 4)
            physical_hs = F.huber_loss(
                reconstructed[:, 0],
                physical[:, 0],
                delta=self.huber_delta,
                reduction="none",
            )
            physical_tp = F.huber_loss(
                reconstructed[:, 1],
                physical[:, 1],
                delta=self.huber_delta,
                reduction="none",
            )
            physical_dir = _torch_circular_loss_deg(reconstructed[:, 2], physical[:, 2])
            physical_dp = _torch_circular_loss_deg(reconstructed[:, 3], physical[:, 3])
            physical_weight = self.physical_loss_weight
        else:
            physical_hs = torch.zeros_like(transfer_hs)
            physical_tp = torch.zeros_like(transfer_tp)
            physical_dir = torch.zeros_like(transfer_dir)
            physical_dp = torch.zeros_like(transfer_dp)
            physical_weight = 0.0

        task_losses = {
            "hs": (physical_weight * physical_hs.mean())
            + (self.transfer_loss_weight * transfer_hs.mean()),
            "tp": (physical_weight * physical_tp.mean())
            + (self.transfer_loss_weight * transfer_tp.mean()),
            "dir": (physical_weight * physical_dir.mean())
            + (self.transfer_loss_weight * transfer_dir.mean()),
            "dp": (physical_weight * physical_dp.mean())
            + (self.transfer_loss_weight * transfer_dp.mean()),
        }
        per_sample_losses = {
            "hs": (physical_weight * physical_hs) + (self.transfer_loss_weight * transfer_hs),
            "tp": (physical_weight * physical_tp) + (self.transfer_loss_weight * transfer_tp),
            "dir": (physical_weight * physical_dir) + (self.transfer_loss_weight * transfer_dir),
            "dp": (physical_weight * physical_dp) + (self.transfer_loss_weight * transfer_dp),
            "transfer_hs": transfer_hs,
            "transfer_tp": transfer_tp,
            "transfer_dir": transfer_dir,
            "transfer_dp": transfer_dp,
            "physical_hs": physical_hs,
            "physical_tp": physical_tp,
            "physical_dir": physical_dir,
            "physical_dp": physical_dp,
        }
        return task_losses, per_sample_losses

    def _store_snapshot(
        self,
        task_losses: Dict[str, torch.Tensor],
        per_sample_losses: Dict[str, torch.Tensor],
        total_override: torch.Tensor | None = None,
    ) -> None:
        if not self.log_components:
            self.last_components = {}
            return

        snapshot = {
            "hs_loss": self._to_float(task_losses["hs"]),
            "tp_loss": self._to_float(task_losses["tp"]),
            "dir_loss": self._to_float(task_losses["dir"]),
            "dp_loss": self._to_float(task_losses["dp"]),
            "transfer_hs_loss": self._to_float(per_sample_losses["transfer_hs"].mean()),
            "transfer_tp_loss": self._to_float(per_sample_losses["transfer_tp"].mean()),
            "transfer_dir_loss": self._to_float(per_sample_losses["transfer_dir"].mean()),
            "transfer_dp_loss": self._to_float(per_sample_losses["transfer_dp"].mean()),
            "physical_hs_loss": self._to_float(per_sample_losses["physical_hs"].mean()),
            "physical_tp_loss": self._to_float(per_sample_losses["physical_tp"].mean()),
            "physical_dir_loss": self._to_float(per_sample_losses["physical_dir"].mean()),
            "physical_dp_loss": self._to_float(per_sample_losses["physical_dp"].mean()),
            "transfer_loss": self._to_float(
                per_sample_losses["transfer_hs"].mean()
                + per_sample_losses["transfer_tp"].mean()
                + per_sample_losses["transfer_dir"].mean()
                + per_sample_losses["transfer_dp"].mean()
            ),
            "physical_loss": self._to_float(
                per_sample_losses["physical_hs"].mean()
                + per_sample_losses["physical_tp"].mean()
                + per_sample_losses["physical_dir"].mean()
                + per_sample_losses["physical_dp"].mean()
            ),
        }
        total = (
            self.task_weights["hs"] * task_losses["hs"]
            + self.task_weights["tp"] * task_losses["tp"]
            + self.task_weights["dir"] * task_losses["dir"]
            + self.task_weights["dp"] * task_losses["dp"]
        )
        snapshot["total_loss"] = self._to_float(
            total_override if total_override is not None else total
        )
        self.last_components = snapshot

    def task_losses(self, pred: dict, target: dict) -> Dict[str, torch.Tensor]:
        losses, _ = self.task_loss_details(pred, target)
        return losses

    def record_task_loss_components(
        self,
        task_losses: Dict[str, torch.Tensor],
        per_sample_losses: Dict[str, torch.Tensor] | None = None,
        total_override: torch.Tensor | None = None,
    ) -> None:
        if per_sample_losses is None:
            per_sample_losses = {}
        self._store_snapshot(task_losses, per_sample_losses, total_override=total_override)

    def forward(self, pred: dict, target: dict) -> torch.Tensor:
        task_losses, per_sample_losses = self.task_loss_details(pred, target)
        total = (
            self.task_weights["hs"] * task_losses["hs"]
            + self.task_weights["tp"] * task_losses["tp"]
            + self.task_weights["dir"] * task_losses["dir"]
            + self.task_weights["dp"] * task_losses["dp"]
        )
        self._store_snapshot(task_losses, per_sample_losses, total_override=total)
        if self.reduction == "none":
            return torch.stack(
                [task_losses["hs"], task_losses["tp"], task_losses["dir"], task_losses["dp"]]
            )
        if self.reduction == "sum":
            return total
        return total


def build_loss_from_config(config: dict) -> nn.Module:
    """Build loss module from `training` config.

    Supports both:
    - legacy flat keys: training.loss_type, training.huber_delta, training.loss_weights
    - richer nested keys under: training.loss
    """
    try:
        from config_resolution import resolve_config
    except Exception:
        from src.config_resolution import resolve_config

    config = resolve_config(config)
    train_cfg = config.get("training", {}) or {}
    data_cfg = config.get("data", {}) or {}
    targets_cfg = resolve_targets_config(data_cfg)
    loss_cfg = train_cfg.get("loss", {}) or {}
    output_columns = data_cfg.get("output_columns", None)

    legacy_loss_type_raw = train_cfg.get("loss_type", None)
    nested_loss_type_raw = loss_cfg.get("type", None)
    if legacy_loss_type_raw is not None and nested_loss_type_raw is not None:
        legacy_norm = str(legacy_loss_type_raw).strip().lower()
        nested_norm = str(nested_loss_type_raw).strip().lower()
        if legacy_norm != nested_norm:
            raise ValueError(
                "Conflicting loss settings detected: "
                f"training.loss_type='{legacy_loss_type_raw}' and training.loss.type='{nested_loss_type_raw}'. "
                "Please set one canonical value (recommended: training.loss.type)."
            )

    loss_type = str(loss_cfg.get("type", train_cfg.get("loss_type", "mse"))).lower()
    default_reduction = str(loss_cfg.get("reduction", "mean")).lower()

    top_huber_delta = float(train_cfg.get("huber_delta", 1.0))
    top_base = str(train_cfg.get("base_regression_loss", "mse")).lower()
    top_weights = train_cfg.get("loss_weights", {}) or {}

    mse_cfg = loss_cfg.get("mse", {}) or {}
    huber_cfg = loss_cfg.get("huber", {}) or {}
    smooth_l1_cfg = loss_cfg.get("smooth_l1", {}) or {}
    angular_cfg = loss_cfg.get("angular", {}) or {}
    weighted_cfg = loss_cfg.get("weighted_multitask", {}) or {}
    coastal_cfg = loss_cfg.get("coastal_multitask", {}) or {}
    blueprint_cfg = loss_cfg.get("blueprint_multitask", {}) or {}
    hybrid_cfg = loss_cfg.get("blueprint_hybrid", {}) or {}

    huber_delta = float(huber_cfg.get("delta", top_huber_delta))
    smooth_l1_beta = float(smooth_l1_cfg.get("beta", 1.0))

    if loss_type == "mse":
        return MSELoss(reduction=str(mse_cfg.get("reduction", default_reduction)))

    if loss_type in {"mae", "l1"}:
        mae_cfg = loss_cfg.get("mae", {}) or {}
        return MAELoss(reduction=str(mae_cfg.get("reduction", default_reduction)))

    if loss_type == "huber":
        return HuberLoss(
            delta=huber_delta,
            reduction=str(huber_cfg.get("reduction", default_reduction)),
        )

    if loss_type in {"smooth_l1", "smoothl1"}:
        return SmoothL1Loss(
            beta=smooth_l1_beta,
            reduction=str(smooth_l1_cfg.get("reduction", default_reduction)),
        )

    if loss_type == "angular":
        pair_indices = _parse_pair_indices(
            angular_cfg.get("pair_indices"),
            output_columns=output_columns,
        )
        return AngularLoss(
            pair_indices=pair_indices,
            reduction=str(angular_cfg.get("reduction", default_reduction)),
            squared=bool(angular_cfg.get("squared", True)),
        )

    if loss_type == "weighted_multitask":
        weights_cfg = weighted_cfg.get("weights", top_weights) or {}
        pair_indices = _parse_pair_indices(
            weighted_cfg.get("direction_pair_indices", angular_cfg.get("pair_indices")),
            output_columns=output_columns,
        )
        hs_extreme_threshold = weighted_cfg.get("hs_extreme_threshold", None)
        return WeightedMultiTaskLoss(
            hs_weight=_get_weight(weights_cfg, aliases=("hs",), default=1.0),
            tp_weight=_get_weight(weights_cfg, aliases=("tp",), default=1.0),
            direction_weight=_get_weight(weights_cfg, aliases=("direction", "dir"), default=1.0),
            base_regression_loss=str(weighted_cfg.get("base_regression_loss", top_base)).lower(),
            huber_delta=float(weighted_cfg.get("huber_delta", huber_delta)),
            smooth_l1_beta=float(weighted_cfg.get("smooth_l1_beta", smooth_l1_beta)),
            hs_extreme_threshold=None
            if hs_extreme_threshold is None
            else float(hs_extreme_threshold),
            hs_extreme_penalty=float(weighted_cfg.get("hs_extreme_penalty", 1.0)),
            direction_pair_indices=pair_indices,
            reduction=str(weighted_cfg.get("reduction", default_reduction)),
        )

    if loss_type in {"coastal_multitask", "coastal_kendall_von_mises", "kendall_von_mises"}:
        pair_indices = _parse_pair_indices(
            coastal_cfg.get("direction_pair_indices", angular_cfg.get("pair_indices")),
            output_columns=output_columns,
        )

        raw_kappa = coastal_cfg.get("direction_kappa", {})
        if isinstance(raw_kappa, (int, float)):
            kappa_cfg = {"dir": float(raw_kappa), "dp": float(raw_kappa)}
        elif isinstance(raw_kappa, (list, tuple)) and len(raw_kappa) >= 2:
            kappa_cfg = {"dir": float(raw_kappa[0]), "dp": float(raw_kappa[1])}
        elif isinstance(raw_kappa, dict):
            kappa_cfg = {
                "dir": float(raw_kappa.get("dir", 4.0)),
                "dp": float(raw_kappa.get("dp", 4.0)),
            }
        else:
            kappa_cfg = {"dir": 4.0, "dp": 4.0}

        raw_uncertainty = coastal_cfg.get("initial_uncertainty", {})
        if isinstance(raw_uncertainty, (int, float)):
            init_uncertainty = {
                "hs": float(raw_uncertainty),
                "tp": float(raw_uncertainty),
                "dir": float(raw_uncertainty),
                "dp": float(raw_uncertainty),
            }
        elif isinstance(raw_uncertainty, dict):
            init_uncertainty = {
                "hs": float(raw_uncertainty.get("hs", 1.0)),
                "tp": float(raw_uncertainty.get("tp", 1.0)),
                "dir": float(raw_uncertainty.get("dir", 1.0)),
                "dp": float(raw_uncertainty.get("dp", 1.0)),
            }
        else:
            init_uncertainty = {"hs": 1.0, "tp": 1.0, "dir": 1.0, "dp": 1.0}

        task_weights_cfg = coastal_cfg.get("task_weights", {}) or {}
        if not isinstance(task_weights_cfg, dict):
            task_weights_cfg = {}

        return CoastalKendallVonMisesLoss(
            base_regression_loss=str(coastal_cfg.get("base_regression_loss", top_base)).lower(),
            huber_delta=float(coastal_cfg.get("huber_delta", huber_delta)),
            smooth_l1_beta=float(coastal_cfg.get("smooth_l1_beta", smooth_l1_beta)),
            direction_pair_indices=pair_indices,
            direction_kappa=kappa_cfg,
            kappa_trainable=bool(coastal_cfg.get("kappa_trainable", False)),
            include_log_i0=bool(coastal_cfg.get("include_log_i0", True)),
            initial_uncertainty=init_uncertainty,
            uncertainty_trainable=bool(coastal_cfg.get("uncertainty_trainable", True)),
            task_weights=task_weights_cfg,
            reduction=str(coastal_cfg.get("reduction", default_reduction)),
        )

    if loss_type in {"blueprint_multitask", "coastal_blueprint"}:
        pair_indices = _parse_pair_indices(
            blueprint_cfg.get("direction_pair_indices", angular_cfg.get("pair_indices")),
            output_columns=output_columns,
        )
        weights_cfg = blueprint_cfg.get("weights", top_weights) or {}
        point_centric_dir = str((config.get("data", {}) or {}).get("point_centric_dir", ""))
        scaler_meta = _load_target_scaler_from_metadata(point_centric_dir)

        return BlueprintMultiTaskLoss(
            output_columns=list(
                output_columns or ["hs", "tp", "dir_sin", "dir_cos", "dp_sin", "dp_cos"]
            ),
            direction_pair_indices=pair_indices,
            huber_delta=float(blueprint_cfg.get("huber_delta", 0.5)),
            task_weights={
                "hs": _get_weight(weights_cfg, aliases=("hs",), default=1.0),
                "tp": _get_weight(weights_cfg, aliases=("tp",), default=1.0),
                "dir": _get_weight(weights_cfg, aliases=("dir", "direction"), default=1.0),
                "dp": _get_weight(weights_cfg, aliases=("dp",), default=1.0),
            },
            energy_weight_power=float(blueprint_cfg.get("energy_weight_power", 2.0)),
            energy_weight_epsilon=float(blueprint_cfg.get("energy_weight_epsilon", 0.0)),
            normalize_energy_weights=bool(blueprint_cfg.get("normalize_energy_weights", True)),
            target_scaler_meta=scaler_meta,
            reduction=str(blueprint_cfg.get("reduction", default_reduction)),
        )

    if loss_type in {"blueprint_hybrid", "hybrid_blueprint", "hybrid_multitask"}:
        point_centric_dir = str((config.get("data", {}) or {}).get("point_centric_dir", ""))
        if str(targets_cfg.get("mode", "physical")) != "physical":
            transfer_scaler_meta = _load_transfer_target_scaler_from_metadata(point_centric_dir)
            return TransferHybridLoss(
                transfer_scaler_meta=transfer_scaler_meta,
                task_weights={
                    "hs": _get_weight(
                        hybrid_cfg.get("weights", top_weights) or {}, aliases=("hs",), default=1.0
                    ),
                    "tp": _get_weight(
                        hybrid_cfg.get("weights", top_weights) or {}, aliases=("tp",), default=1.0
                    ),
                    "dir": _get_weight(
                        hybrid_cfg.get("weights", top_weights) or {},
                        aliases=("dir", "direction"),
                        default=1.0,
                    ),
                    "dp": _get_weight(
                        hybrid_cfg.get("weights", top_weights) or {}, aliases=("dp",), default=1.0
                    ),
                },
                physical_loss_weight=float(targets_cfg.get("physical_loss_weight", 1.0)),
                transfer_loss_weight=float(targets_cfg.get("transfer_loss_weight", 0.5)),
                huber_delta=float(hybrid_cfg.get("huber_delta", huber_delta)),
                target_mode=str(targets_cfg.get("mode", "physical_and_transfer")),
                transfer_representation=str(targets_cfg.get("transfer_representation", "legacy")),
                residual_config=targets_cfg.get("residual_correction", {}) or {},
                compute_physical_loss_on_reconstruction=bool(
                    loss_cfg.get("compute_physical_loss_on_reconstruction", True)
                ),
                tp_min=float(targets_cfg.get("tp_min", 0.5)),
                tp_max=float(targets_cfg.get("tp_max", 30.0)),
                log_components=bool(loss_cfg.get("log_components", True)),
                reduction=str(hybrid_cfg.get("reduction", default_reduction)),
            )

        num_tp_bins = int(hybrid_cfg.get("num_tp_bins", 32))
        num_dp_bins = int(hybrid_cfg.get("num_dp_bins", 36))
        tp_bin_range = tuple(hybrid_cfg.get("tp_bin_range", [0.0, 25.0]))
        dp_bin_range = tuple(hybrid_cfg.get("dp_bin_range", [0.0, 360.0]))
        tp_centers = np.linspace(tp_bin_range[0], tp_bin_range[1], num_tp_bins, dtype=np.float32)
        dp_centers = np.linspace(
            dp_bin_range[0], dp_bin_range[1], num_dp_bins + 1, dtype=np.float32
        )[:-1]
        scaler_meta = _load_target_scaler_from_metadata(point_centric_dir)

        return BlueprintHybridLoss(
            huber_delta=float(hybrid_cfg.get("huber_delta", huber_delta)),
            label_smoothing_sigma=float(
                hybrid_cfg.get("label_smoothing_sigma", loss_cfg.get("label_smoothing_sigma", 0.8))
            ),
            tp_bin_centers=tp_centers,
            dp_bin_centers=dp_centers,
            task_weights={
                "hs": _get_weight(
                    hybrid_cfg.get("weights", top_weights) or {}, aliases=("hs",), default=1.0
                ),
                "tp": _get_weight(
                    hybrid_cfg.get("weights", top_weights) or {}, aliases=("tp",), default=1.0
                ),
                "dir": _get_weight(
                    hybrid_cfg.get("weights", top_weights) or {},
                    aliases=("dir", "direction"),
                    default=1.0,
                ),
                "dp": _get_weight(
                    hybrid_cfg.get("weights", top_weights) or {}, aliases=("dp",), default=1.0
                ),
            },
            target_scaler_meta=scaler_meta,
            energy_weight_power=float(hybrid_cfg.get("energy_weight_power", 2.0)),
            log_components=bool(loss_cfg.get("log_components", True)),
            reduction=str(hybrid_cfg.get("reduction", default_reduction)),
        )

    raise ValueError(
        f"Unsupported loss type '{loss_type}'. "
        "Expected one of: mse, mae, huber, smooth_l1, angular, weighted_multitask, coastal_multitask, blueprint_multitask, blueprint_hybrid"
    )
