"""Compute physical-unit accuracy metrics for coastal-wave predictions.

Scalar wave-height and period errors are evaluated directly.  Direction and
peak-direction use sine/cosine pairs internally, then circular differences in
degrees, so errors crossing north remain physically meaningful.
"""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np


TARGET_METRIC_SPECS = (
    {
        "target": "hs",
        "mse_key": "hs_mse",
        "rmse_key": "hs_rmse",
        "bias_key": "hs_bias",
        "pearson_key": "hs_pearson_r",
        "r2_key": "hs_r2",
        "unit": "m",
        "mse_unit": "m2",
    },
    {
        "target": "tp",
        "mse_key": "tp_mse",
        "rmse_key": "tp_rmse",
        "bias_key": "tp_bias",
        "pearson_key": "tp_pearson_r",
        "r2_key": "tp_r2",
        "unit": "s",
        "mse_unit": "s2",
    },
    {
        "target": "dir",
        "mse_key": "dir_mse",
        "rmse_key": "dir_rmse_deg",
        "bias_key": "dir_bias_deg",
        "pearson_key": "dir_pearson_r",
        "r2_key": "dir_r2",
        "unit": "deg",
        "mse_unit": "rad2",
    },
    {
        "target": "dp",
        "mse_key": "dp_mse",
        "rmse_key": "dp_rmse_deg",
        "bias_key": "dp_bias_deg",
        "pearson_key": "dp_pearson_r",
        "r2_key": "dp_r2",
        "unit": "deg",
        "mse_unit": "rad2",
    },
)


def _finite_pair_mask(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    true_arr = np.asarray(y_true, dtype=float)
    pred_arr = np.asarray(y_pred, dtype=float)
    return np.isfinite(true_arr) & np.isfinite(pred_arr)


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean squared error."""
    true_arr = np.asarray(y_true, dtype=float)
    pred_arr = np.asarray(y_pred, dtype=float)
    mask = _finite_pair_mask(true_arr, pred_arr)
    if not np.any(mask):
        return float("nan")
    diff = pred_arr[mask] - true_arr[mask]
    return float(np.mean(diff**2))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error."""
    mse_value = mse(y_true, y_pred)
    if not np.isfinite(mse_value):
        return float("nan")
    return float(np.sqrt(mse_value))


def bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean signed prediction error."""
    true_arr = np.asarray(y_true, dtype=float)
    pred_arr = np.asarray(y_pred, dtype=float)
    mask = _finite_pair_mask(true_arr, pred_arr)
    if not np.any(mask):
        return float("nan")
    return float(np.mean(pred_arr[mask] - true_arr[mask]))


def pearson_r(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Pearson correlation coefficient."""
    true_arr = np.asarray(y_true, dtype=float)
    pred_arr = np.asarray(y_pred, dtype=float)
    mask = _finite_pair_mask(true_arr, pred_arr)
    if int(np.count_nonzero(mask)) < 2:
        return float("nan")
    true_arr = true_arr[mask]
    pred_arr = pred_arr[mask]
    if np.std(true_arr) == 0.0 or np.std(pred_arr) == 0.0:
        return float("nan")
    return float(np.corrcoef(true_arr, pred_arr)[0, 1])


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = _finite_pair_mask(y_true, y_pred)
    if not np.any(mask):
        return 0.0
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    ss_res = np.sum((y_true - y_pred) ** 2)
    mean_true = np.mean(y_true)
    ss_tot = np.sum((y_true - mean_true) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1.0 - (ss_res / ss_tot))


def _direction_delta_deg(
    true_sin: np.ndarray,
    true_cos: np.ndarray,
    pred_sin: np.ndarray,
    pred_cos: np.ndarray,
) -> np.ndarray:
    true_ang = np.arctan2(true_sin, true_cos)
    pred_ang = np.arctan2(pred_sin, pred_cos)
    delta = np.arctan2(np.sin(pred_ang - true_ang), np.cos(pred_ang - true_ang))
    return np.degrees(delta)


def _direction_delta_rad(
    true_sin: np.ndarray,
    true_cos: np.ndarray,
    pred_sin: np.ndarray,
    pred_cos: np.ndarray,
) -> np.ndarray:
    true_ang = np.arctan2(true_sin, true_cos)
    pred_ang = np.arctan2(pred_sin, pred_cos)
    return np.arctan2(np.sin(pred_ang - true_ang), np.cos(pred_ang - true_ang))


def direction_mse_rad2(
    true_sin: np.ndarray,
    true_cos: np.ndarray,
    pred_sin: np.ndarray,
    pred_cos: np.ndarray,
) -> float:
    """Directional MSE in wrapped angle space, measured in radians squared."""
    true_sin = np.asarray(true_sin, dtype=float)
    true_cos = np.asarray(true_cos, dtype=float)
    pred_sin = np.asarray(pred_sin, dtype=float)
    pred_cos = np.asarray(pred_cos, dtype=float)
    mask = (
        np.isfinite(true_sin)
        & np.isfinite(true_cos)
        & np.isfinite(pred_sin)
        & np.isfinite(pred_cos)
    )
    if not np.any(mask):
        return float("nan")
    delta = _direction_delta_rad(true_sin[mask], true_cos[mask], pred_sin[mask], pred_cos[mask])
    return float(np.mean(delta**2))


def direction_rmse_deg(
    true_sin: np.ndarray,
    true_cos: np.ndarray,
    pred_sin: np.ndarray,
    pred_cos: np.ndarray,
) -> float:
    """Directional RMSE in degrees using wrapped angular difference."""
    true_sin = np.asarray(true_sin, dtype=float)
    true_cos = np.asarray(true_cos, dtype=float)
    pred_sin = np.asarray(pred_sin, dtype=float)
    pred_cos = np.asarray(pred_cos, dtype=float)
    mask = (
        np.isfinite(true_sin)
        & np.isfinite(true_cos)
        & np.isfinite(pred_sin)
        & np.isfinite(pred_cos)
    )
    if not np.any(mask):
        return float("nan")
    delta_deg = _direction_delta_deg(true_sin[mask], true_cos[mask], pred_sin[mask], pred_cos[mask])
    return float(np.sqrt(np.mean(delta_deg**2)))


def direction_bias_deg(
    true_sin: np.ndarray,
    true_cos: np.ndarray,
    pred_sin: np.ndarray,
    pred_cos: np.ndarray,
) -> float:
    """Directional bias in degrees using wrapped angular difference."""
    true_sin = np.asarray(true_sin, dtype=float)
    true_cos = np.asarray(true_cos, dtype=float)
    pred_sin = np.asarray(pred_sin, dtype=float)
    pred_cos = np.asarray(pred_cos, dtype=float)
    mask = (
        np.isfinite(true_sin)
        & np.isfinite(true_cos)
        & np.isfinite(pred_sin)
        & np.isfinite(pred_cos)
    )
    if not np.any(mask):
        return float("nan")
    delta_deg = _direction_delta_deg(true_sin[mask], true_cos[mask], pred_sin[mask], pred_cos[mask])
    return float(np.mean(delta_deg))


def direction_pearson_r(
    true_sin: np.ndarray,
    true_cos: np.ndarray,
    pred_sin: np.ndarray,
    pred_cos: np.ndarray,
) -> float:
    """Directional Pearson correlation as the mean of sin/cos correlations."""
    sin_corr = pearson_r(true_sin, pred_sin)
    cos_corr = pearson_r(true_cos, pred_cos)
    return float(0.5 * (sin_corr + cos_corr))


def direction_r2(
    true_sin: np.ndarray,
    true_cos: np.ndarray,
    pred_sin: np.ndarray,
    pred_cos: np.ndarray,
) -> float:
    """Circular direction R2 computed in angle space.

    Uses wrapped angular residuals for the error term and circular variance
    around the circular mean for the total variance term:

    R2 = 1 - sum(wrap(pred-true)^2) / sum(wrap(true-mu)^2)
    """
    true_sin = np.asarray(true_sin, dtype=float)
    true_cos = np.asarray(true_cos, dtype=float)
    pred_sin = np.asarray(pred_sin, dtype=float)
    pred_cos = np.asarray(pred_cos, dtype=float)

    mask = (
        np.isfinite(true_sin)
        & np.isfinite(true_cos)
        & np.isfinite(pred_sin)
        & np.isfinite(pred_cos)
    )
    if not np.any(mask):
        return 0.0

    true_ang = np.arctan2(true_sin[mask], true_cos[mask])
    pred_ang = np.arctan2(pred_sin[mask], pred_cos[mask])

    # Residual angles wrapped into [-pi, pi].
    delta = np.arctan2(np.sin(pred_ang - true_ang), np.cos(pred_ang - true_ang))
    ss_res = float(np.sum(delta**2))

    # Circular mean of true angles and wrapped deviations from it.
    mu = float(np.arctan2(np.mean(np.sin(true_ang)), np.mean(np.cos(true_ang))))
    centered = np.arctan2(np.sin(true_ang - mu), np.cos(true_ang - mu))
    ss_tot = float(np.sum(centered**2))

    if ss_tot <= 0.0:
        return 0.0
    return float(1.0 - (ss_res / ss_tot))


def compute_wave_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute aggregate physical metrics for canonical six-channel outputs.

    Input rows must be ordered ``[Hs, Tp, sin(dir), cos(dir), sin(dp),
    cos(dp)]`` and must already be in physical units.  Non-finite paired
    values are excluded from each metric independently.
    """
    hs_true, tp_true = y_true[:, 0], y_true[:, 1]
    hs_pred, tp_pred = y_pred[:, 0], y_pred[:, 1]

    dir_sin_true, dir_cos_true = y_true[:, 2], y_true[:, 3]
    dir_sin_pred, dir_cos_pred = y_pred[:, 2], y_pred[:, 3]

    dp_sin_true, dp_cos_true = y_true[:, 4], y_true[:, 5]
    dp_sin_pred, dp_cos_pred = y_pred[:, 4], y_pred[:, 5]

    dir_mse = direction_mse_rad2(dir_sin_true, dir_cos_true, dir_sin_pred, dir_cos_pred)
    dir_rmse = direction_rmse_deg(dir_sin_true, dir_cos_true, dir_sin_pred, dir_cos_pred)
    dir_bias = direction_bias_deg(dir_sin_true, dir_cos_true, dir_sin_pred, dir_cos_pred)
    dir_pearson = direction_pearson_r(dir_sin_true, dir_cos_true, dir_sin_pred, dir_cos_pred)
    dir_r2_val = direction_r2(dir_sin_true, dir_cos_true, dir_sin_pred, dir_cos_pred)

    dp_mse = direction_mse_rad2(dp_sin_true, dp_cos_true, dp_sin_pred, dp_cos_pred)
    dp_rmse = direction_rmse_deg(dp_sin_true, dp_cos_true, dp_sin_pred, dp_cos_pred)
    dp_bias = direction_bias_deg(dp_sin_true, dp_cos_true, dp_sin_pred, dp_cos_pred)
    dp_pearson = direction_pearson_r(dp_sin_true, dp_cos_true, dp_sin_pred, dp_cos_pred)
    dp_r2_val = direction_r2(dp_sin_true, dp_cos_true, dp_sin_pred, dp_cos_pred)

    out = {
        "hs_mse": mse(hs_true, hs_pred),
        "hs_rmse": rmse(hs_true, hs_pred),
        "hs_bias": bias(hs_true, hs_pred),
        "hs_pearson_r": pearson_r(hs_true, hs_pred),
        "hs_r2": r2_score(hs_true, hs_pred),
        "tp_mse": mse(tp_true, tp_pred),
        "tp_rmse": rmse(tp_true, tp_pred),
        "tp_bias": bias(tp_true, tp_pred),
        "tp_pearson_r": pearson_r(tp_true, tp_pred),
        "tp_r2": r2_score(tp_true, tp_pred),
        "dir_mse": dir_mse,
        "dir_rmse_deg": dir_rmse,
        "dir_bias_deg": dir_bias,
        "dir_pearson_r": dir_pearson,
        "dir_r2": dir_r2_val,
        "dp_mse": dp_mse,
        "dp_rmse_deg": dp_rmse,
        "dp_bias_deg": dp_bias,
        "dp_pearson_r": dp_pearson,
        "dp_r2": dp_r2_val,
        "direction_rmse_deg": float(0.5 * (dir_rmse + dp_rmse)),
        "direction_r2": float(0.5 * (dir_r2_val + dp_r2_val)),
    }
    return out


def compute_target_metric_rows(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    split: str,
    aggregation: str,
    site: str,
    sample_count: int | None = None,
    site_count: int | None = None,
) -> list[dict[str, object]]:
    """Convert aggregate wave metrics into one row per target."""
    metrics = compute_wave_metrics(y_true=y_true, y_pred=y_pred)
    rows: list[dict[str, object]] = []
    for spec in TARGET_METRIC_SPECS:
        rows.append(
            {
                "split": str(split),
                "aggregation": str(aggregation),
                "site": str(site),
                "target": str(spec["target"]),
                "mse": float(metrics[spec["mse_key"]]),
                "rmse": float(metrics[spec["rmse_key"]]),
                "bias": float(metrics[spec["bias_key"]]),
                "pearson_r": float(metrics[spec["pearson_key"]]),
                "r2": float(metrics[spec["r2_key"]]),
                "mse_unit": str(spec["mse_unit"]),
                "rmse_unit": str(spec["unit"]),
                "bias_unit": str(spec["unit"]),
                "sample_count": None if sample_count is None else int(sample_count),
                "site_count": None if site_count is None else int(site_count),
            }
        )
    return rows


def compute_split_target_metric_rows(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    site_names: Sequence[str],
    *,
    split: str,
    aggregation: str,
    site: str,
) -> list[dict[str, object]]:
    """Compute one aggregate row per target from all samples in a split."""
    site_arr = np.asarray(site_names, dtype=str)
    if site_arr.shape[0] != y_true.shape[0] or y_true.shape[0] != y_pred.shape[0]:
        raise ValueError(
            "Split-level metric computation requires site_names, y_true, and y_pred "
            "to have the same sample length."
        )
    return compute_target_metric_rows(
        y_true=y_true,
        y_pred=y_pred,
        split=split,
        aggregation=aggregation,
        site=site,
        sample_count=int(y_true.shape[0]),
        site_count=int(len(set(site_arr.tolist()))),
    )


def compute_site_target_metric_rows(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    site_names: Sequence[str],
    *,
    split: str,
) -> list[dict[str, object]]:
    """Compute one metric row per target for every site in a split."""
    site_arr = np.asarray(site_names, dtype=str)
    if site_arr.shape[0] != y_true.shape[0] or y_true.shape[0] != y_pred.shape[0]:
        raise ValueError(
            "Per-site metric computation requires site_names, y_true, and y_pred "
            "to have the same sample length."
        )

    rows: list[dict[str, object]] = []
    for site in sorted(set(site_arr.tolist())):
        mask = site_arr == site
        sample_count = int(np.count_nonzero(mask))
        if sample_count == 0:
            continue
        rows.extend(
            compute_target_metric_rows(
                y_true=y_true[mask],
                y_pred=y_pred[mask],
                split=split,
                aggregation="site",
                site=site,
                sample_count=sample_count,
                site_count=1,
            )
        )
    return rows


def average_target_metric_rows(
    rows: Sequence[dict[str, object]],
    *,
    split: str,
    aggregation: str,
    site: str,
) -> list[dict[str, object]]:
    """Average target rows across sites, preserving one row per target."""
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["target"]), []).append(dict(row))

    averaged_rows: list[dict[str, object]] = []
    for spec in TARGET_METRIC_SPECS:
        target = str(spec["target"])
        bucket = grouped.get(target, [])
        if not bucket:
            continue

        sample_total = sum(int(item.get("sample_count") or 0) for item in bucket)
        averaged_rows.append(
            {
                "split": str(split),
                "aggregation": str(aggregation),
                "site": str(site),
                "target": target,
                "mse": float(np.mean([float(item["mse"]) for item in bucket])),
                "rmse": float(np.mean([float(item["rmse"]) for item in bucket])),
                "bias": float(np.mean([float(item["bias"]) for item in bucket])),
                "pearson_r": float(np.mean([float(item["pearson_r"]) for item in bucket])),
                "r2": float(np.mean([float(item["r2"]) for item in bucket])),
                "mse_unit": str(spec["mse_unit"]),
                "rmse_unit": str(spec["unit"]),
                "bias_unit": str(spec["unit"]),
                "sample_count": int(sample_total),
                "site_count": int(len(bucket)),
            }
        )
    return averaged_rows
