"""Aggregate route and directional-ray records into the static feature contract."""

from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd
from .ray_caster import (
    COMPASS_SECTORS,
    POROSITY_RADII_M,
    porosity_column_name,
    sector_feature_column_names,
)

RAY_SECTOR_COLUMNS = sector_feature_column_names()
SECTOR_ANGLE_DEG = {
    sector: idx * (360.0 / len(COMPASS_SECTORS)) for idx, sector in enumerate(COMPASS_SECTORS)
}
SECTOR_WIDTH_DEG = 360.0 / len(COMPASS_SECTORS)
POROSITY_COLUMNS = [porosity_column_name(radius_m) for radius_m in POROSITY_RADII_M]
FETCH_ANISOTROPY_COLUMNS = [
    "fetch_max_over_mean",
    "fetch_std_m",
    "fetch_cv",
    "fetch_directional_entropy",
    "fetch_resultant_length",
    "open_sector_fraction",
    "closed_sector_fraction",
    "open_sector_width_deg",
    "dominant_fetch_direction_sin",
    "dominant_fetch_direction_cos",
]
FJORDNESS_COMPONENT_COLUMNS = [
    "fjordness_land_blocking_component",
    "fjordness_low_porosity_component",
    "fjordness_closed_sector_component",
    "fjordness_anisotropy_component",
    "fjordness_low_fetch_component",
    "fjordness_route_complexity_component",
]
FJORDNESS_COLUMNS = [
    *FJORDNESS_COMPONENT_COLUMNS,
    "fjordness_score",
    "site_regime_open",
    "site_regime_transition",
    "site_regime_fjord",
]
PATH_RATIO_COLUMNS = [
    "path_direct_distance_m",
    "path_tortuosity_ratio",
    "bottleneck_to_path_ratio",
    "bottleneck_to_fetch_ratio",
    "funneling_log",
]
OPEN_FETCH_THRESHOLD_M = 5_000.0
CLOSED_FETCH_THRESHOLD_M = 500.0
FJORDNESS_OPEN_MAX = 0.35
FJORDNESS_TRANSITION_MAX = 0.65
DEFAULT_BREAKING_ENABLED = True
DEFAULT_BREAKING_GAMMA = 0.78


MASTER_COLUMNS = [
    "site_name",
    "site_lat",
    "site_lon",
    "site_x",
    "site_y",
    "site_row",
    "site_col",
    "reachable",
    "path_length_m",
    "path_point_count",
    "snap_distance_m",
    "path_bottleneck_m",
    "funneling_ratio",
    "choke_out_ratio",
    "static_tortuosity_sum",
    "static_signed_curvature_deg",
    "static_net_deflection_deg",
    "static_final_approach_deg",
    *POROSITY_COLUMNS,
    "static_dist_to_coast_m",
    "static_local_depth_m",
    "local_depth_m",
    "local_breaking_hs_cap",
    "local_breaking_cap_valid",
    "static_nearest_shore_steepness",
    "static_nearest_shore_normal_deg",
    "ray_fetch_min_m",
    "ray_fetch_mean_m",
    "ray_fetch_max_m",
    *RAY_SECTOR_COLUMNS,
    *FETCH_ANISOTROPY_COLUMNS,
    *FJORDNESS_COLUMNS,
    *PATH_RATIO_COLUMNS,
    "ray_hit_land_fraction",
    "ray_count",
]


def _safe_divide(
    numerator: pd.Series | np.ndarray, denominator: pd.Series | np.ndarray
) -> np.ndarray:
    """Divide with NaN where denominator is zero or non-finite."""
    num = np.asarray(numerator, dtype=np.float64)
    den = np.asarray(denominator, dtype=np.float64)
    out = np.full(np.broadcast(num, den).shape, np.nan, dtype=np.float64)
    valid = np.isfinite(num) & np.isfinite(den) & (np.abs(den) > 0.0)
    out[valid] = num[valid] / den[valid]
    return out


def _extract_sector_matrix(df: pd.DataFrame, prefix: str, suffix: str = "") -> np.ndarray:
    cols = [f"{prefix}_{sector}{suffix}" for sector in COMPASS_SECTORS]
    return df[cols].to_numpy(dtype=np.float64, copy=True)


def _circular_run_width(values: np.ndarray) -> float:
    """Return largest contiguous circular run width in degrees."""
    mask = np.asarray(values, dtype=bool)
    if mask.size == 0:
        return float("nan")
    if np.all(mask):
        return float(mask.size * SECTOR_WIDTH_DEG)
    doubled = np.concatenate([mask.astype(np.int64), mask.astype(np.int64)])
    best = 0
    current = 0
    for value in doubled:
        if value:
            current += 1
            best = max(best, current)
        else:
            current = 0
    best = min(best, mask.size)
    return float(best * SECTOR_WIDTH_DEG)


def add_fetch_anisotropy_features(master_df: pd.DataFrame) -> pd.DataFrame:
    """Add static fetch-shape summary features from sector-wise fetch columns."""
    out = master_df.copy()
    fetch_matrix = _extract_sector_matrix(out, "ray_fetch", "_m")
    row_mean = np.nanmean(fetch_matrix, axis=1)
    row_std = np.nanstd(fetch_matrix, axis=1)
    row_max = np.nanmax(fetch_matrix, axis=1)
    row_sum = np.nansum(fetch_matrix, axis=1)

    out["fetch_max_over_mean"] = _safe_divide(row_max, row_mean)
    out["fetch_std_m"] = row_std
    out["fetch_cv"] = _safe_divide(row_std, row_mean)

    probs = np.full_like(fetch_matrix, np.nan, dtype=np.float64)
    valid_sum = np.isfinite(row_sum) & (row_sum > 0.0)
    probs[valid_sum, :] = fetch_matrix[valid_sum, :] / row_sum[valid_sum, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        entropy = -np.nansum(np.where(probs > 0.0, probs * np.log(probs), 0.0), axis=1)
    max_entropy = np.log(float(len(COMPASS_SECTORS)))
    out["fetch_directional_entropy"] = np.where(
        np.isfinite(entropy) & (max_entropy > 0.0),
        np.clip(entropy / max_entropy, 0.0, 1.0),
        np.nan,
    )

    sector_angles_rad = np.deg2rad(
        np.asarray([SECTOR_ANGLE_DEG[s] for s in COMPASS_SECTORS], dtype=np.float64)
    )
    unit_x = np.sin(sector_angles_rad)
    unit_y = np.cos(sector_angles_rad)
    weighted_x = np.nansum(fetch_matrix * unit_x[None, :], axis=1)
    weighted_y = np.nansum(fetch_matrix * unit_y[None, :], axis=1)
    resultant = np.hypot(weighted_x, weighted_y)
    out["fetch_resultant_length"] = np.clip(_safe_divide(resultant, row_sum), 0.0, 1.0)

    open_mask = np.where(np.isfinite(fetch_matrix), fetch_matrix >= OPEN_FETCH_THRESHOLD_M, False)
    closed_mask = np.where(
        np.isfinite(fetch_matrix), fetch_matrix <= CLOSED_FETCH_THRESHOLD_M, False
    )
    out["open_sector_fraction"] = np.mean(open_mask, axis=1)
    out["closed_sector_fraction"] = np.mean(closed_mask, axis=1)
    out["open_sector_width_deg"] = [_circular_run_width(mask_row) for mask_row in open_mask]

    dominant_idx = np.nanargmax(np.where(np.isfinite(fetch_matrix), fetch_matrix, -np.inf), axis=1)
    dominant_angles_deg = np.asarray(
        [SECTOR_ANGLE_DEG[COMPASS_SECTORS[int(idx)]] for idx in dominant_idx], dtype=np.float64
    )
    invalid_dominant = ~np.isfinite(row_max)
    dominant_angles_deg[invalid_dominant] = np.nan
    dominant_angles_rad = np.deg2rad(dominant_angles_deg)
    out["dominant_fetch_direction_sin"] = np.sin(dominant_angles_rad)
    out["dominant_fetch_direction_cos"] = np.cos(dominant_angles_rad)
    return out


def add_path_geometry_ratio_features(
    master_df: pd.DataFrame, routes_df: pd.DataFrame
) -> pd.DataFrame:
    """Add path geometry ratios derived from routed polylines."""
    out = master_df.copy()
    route_lookup = routes_df.set_index("site_name")
    direct_distance = np.full(out.shape[0], np.nan, dtype=np.float64)

    for idx, site_name in enumerate(out["site_name"].astype(str)):
        if site_name not in route_lookup.index:
            continue
        path_xy = route_lookup.at[site_name, "path_xy"]
        if not isinstance(path_xy, list) or len(path_xy) < 2:
            continue
        try:
            path_xy_arr = np.asarray(path_xy, dtype=np.float64)
        except Exception:
            continue
        if path_xy_arr.ndim != 2 or path_xy_arr.shape[1] != 2:
            continue
        start_xy = path_xy_arr[0]
        end_xy = path_xy_arr[-1]
        if np.all(np.isfinite(start_xy)) and np.all(np.isfinite(end_xy)):
            direct_distance[idx] = float(np.hypot(end_xy[0] - start_xy[0], end_xy[1] - start_xy[1]))

    out["path_direct_distance_m"] = direct_distance
    out["path_tortuosity_ratio"] = _safe_divide(out["path_length_m"], out["path_direct_distance_m"])
    out["bottleneck_to_path_ratio"] = _safe_divide(out["path_bottleneck_m"], out["path_length_m"])
    out["bottleneck_to_fetch_ratio"] = _safe_divide(
        out["path_bottleneck_m"], out["ray_fetch_mean_m"]
    )
    funneling_numeric = pd.to_numeric(out["funneling_ratio"], errors="coerce").to_numpy(
        dtype=np.float64, copy=False
    )
    out["funneling_log"] = np.where(
        np.isfinite(funneling_numeric),
        np.log1p(np.maximum(funneling_numeric, 0.0)),
        np.nan,
    )
    return out


def _minmax_component(values: pd.Series | np.ndarray, inverse: bool = False) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if inverse:
        arr = np.where(np.isfinite(arr), 1.0 / np.maximum(arr, 1e-6), np.nan)
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros(arr.shape, dtype=np.float64)
    fill_value = float(np.nanmedian(arr[finite]))
    arr = np.where(finite, arr, fill_value)
    min_value = float(np.min(arr))
    max_value = float(np.max(arr))
    if max_value <= min_value:
        return np.zeros(arr.shape, dtype=np.float64)
    scaled = (arr - min_value) / (max_value - min_value)
    return np.clip(scaled, 0.0, 1.0)


def add_fjordness_features(master_df: pd.DataFrame) -> pd.DataFrame:
    """Add fjordness score and one-hot site regimes."""
    out = master_df.copy()

    land_blocking = _minmax_component(out["ray_hit_land_fraction"])
    low_porosity = _minmax_component(
        1.0 - pd.to_numeric(out["static_porosity_5km"], errors="coerce")
    )
    closed_sector = 0.5 * (
        _minmax_component(out["closed_sector_fraction"])
        + _minmax_component(1.0 - pd.to_numeric(out["open_sector_fraction"], errors="coerce"))
    )
    anisotropy = _minmax_component(out["fetch_max_over_mean"])
    low_fetch = _minmax_component(out["ray_fetch_mean_m"], inverse=True)
    route_complexity = np.mean(
        np.column_stack(
            [
                _minmax_component(out["static_tortuosity_sum"]),
                _minmax_component(out["funneling_log"]),
                _minmax_component(out["path_bottleneck_m"], inverse=True),
            ]
        ),
        axis=1,
    )

    out["fjordness_land_blocking_component"] = land_blocking
    out["fjordness_low_porosity_component"] = low_porosity
    out["fjordness_closed_sector_component"] = closed_sector
    out["fjordness_anisotropy_component"] = anisotropy
    out["fjordness_low_fetch_component"] = low_fetch
    out["fjordness_route_complexity_component"] = route_complexity

    component_matrix = out[FJORDNESS_COMPONENT_COLUMNS].to_numpy(dtype=np.float64, copy=False)
    out["fjordness_score"] = np.clip(np.nanmean(component_matrix, axis=1), 0.0, 1.0)
    out["site_regime_open"] = (out["fjordness_score"] < FJORDNESS_OPEN_MAX).astype(np.int64)
    out["site_regime_transition"] = (
        (out["fjordness_score"] >= FJORDNESS_OPEN_MAX)
        & (out["fjordness_score"] < FJORDNESS_TRANSITION_MAX)
    ).astype(np.int64)
    out["site_regime_fjord"] = (out["fjordness_score"] >= FJORDNESS_TRANSITION_MAX).astype(np.int64)
    return out


def add_local_breaking_features(
    master_df: pd.DataFrame,
    *,
    enabled: bool,
    gamma: float,
) -> pd.DataFrame:
    """Add local depth-limited wave-breaking cap columns."""
    out = master_df.copy()
    depth = pd.to_numeric(out.get("local_depth_m", np.nan), errors="coerce").to_numpy(
        dtype=np.float64, copy=False
    )
    valid = np.isfinite(depth) & (depth > 0.0)

    if "local_breaking_cap_valid" in out.columns:
        input_valid = pd.to_numeric(
            out["local_breaking_cap_valid"],
            errors="coerce",
        ).to_numpy(dtype=np.float64, copy=False)
        valid &= np.isfinite(input_valid) & (input_valid > 0.0)

    out["local_breaking_cap_valid"] = valid.astype(np.int64)
    if enabled:
        out["local_breaking_hs_cap"] = np.where(valid, depth * float(gamma), np.nan)
    else:
        out["local_breaking_hs_cap"] = np.nan
    return out


def log_local_breaking_summary(master_df: pd.DataFrame) -> None:
    depth = pd.to_numeric(master_df["local_depth_m"], errors="coerce").to_numpy(
        dtype=np.float64, copy=False
    )
    cap = pd.to_numeric(master_df["local_breaking_hs_cap"], errors="coerce").to_numpy(
        dtype=np.float64, copy=False
    )
    valid = (
        pd.to_numeric(master_df["local_breaking_cap_valid"], errors="coerce")
        .fillna(0)
        .astype(int)
        .to_numpy()
    )
    site_names = master_df["site_name"].astype(str).tolist()

    finite_depth = depth[np.isfinite(depth)]
    finite_cap = cap[np.isfinite(cap)]
    invalid_sites = [site for site, flag in zip(site_names, valid) if int(flag) == 0]

    if finite_depth.size:
        print(
            "Local depth summary (m): "
            f"min={float(np.nanmin(finite_depth)):.3f} "
            f"median={float(np.nanmedian(finite_depth)):.3f} "
            f"max={float(np.nanmax(finite_depth)):.3f}"
        )
    else:
        print("Local depth summary (m): no finite values")

    if finite_cap.size:
        print(
            "Local breaking cap summary (m): "
            f"min={float(np.nanmin(finite_cap)):.3f} "
            f"median={float(np.nanmedian(finite_cap)):.3f} "
            f"max={float(np.nanmax(finite_cap)):.3f}"
        )
    else:
        print("Local breaking cap summary (m): no finite values")

    print(f"Invalid local breaking caps: {len(invalid_sites)}")
    if invalid_sites:
        print("Invalid local breaking cap sites: " + ", ".join(sorted(invalid_sites)))


def validate_master_static_features(master_df: pd.DataFrame) -> None:
    """Validate merged master static output for duplicated or non-physical values."""
    if master_df.columns.duplicated().any():
        duplicates = master_df.columns[master_df.columns.duplicated()].tolist()
        raise ValueError(f"Duplicate static feature columns detected: {duplicates}")

    required = [
        *POROSITY_COLUMNS,
        *FETCH_ANISOTROPY_COLUMNS,
        *FJORDNESS_COLUMNS,
        *PATH_RATIO_COLUMNS,
    ]
    required.extend([f"ray_min_depth_{sector}_m" for sector in COMPASS_SECTORS])
    required.extend(["local_depth_m", "local_breaking_hs_cap", "local_breaking_cap_valid"])
    missing = [col for col in required if col not in master_df.columns]
    if missing:
        raise ValueError(f"master_static_features.csv is missing required new columns: {missing}")

    numeric_df = master_df.select_dtypes(include=[np.number])
    inf_columns = [
        col
        for col in numeric_df.columns
        if np.isinf(numeric_df[col].to_numpy(dtype=np.float64)).any()
    ]
    if inf_columns:
        raise ValueError(f"Non-finite static feature values detected in columns: {inf_columns}")

    min_depth_cols = [f"ray_min_depth_{sector}_m" for sector in COMPASS_SECTORS]
    for col in min_depth_cols:
        values = pd.to_numeric(master_df[col], errors="coerce").to_numpy(
            dtype=np.float64, copy=False
        )
        if np.isfinite(values).any() and float(np.nanmin(values)) < 0.0:
            raise ValueError(f"Negative minimum depth encountered in column '{col}'")

    fjordness = pd.to_numeric(master_df["fjordness_score"], errors="coerce").to_numpy(
        dtype=np.float64, copy=False
    )
    if np.isfinite(fjordness).any():
        if float(np.nanmin(fjordness)) < 0.0 or float(np.nanmax(fjordness)) > 1.0:
            raise ValueError("fjordness_score must remain within [0, 1]")

    regime_sum = master_df[["site_regime_open", "site_regime_transition", "site_regime_fjord"]].sum(
        axis=1
    )
    if not np.allclose(regime_sum.to_numpy(dtype=np.float64, copy=False), 1.0):
        raise ValueError("Site regime one-hot columns must sum to 1 for every site")

    local_depth = pd.to_numeric(master_df["local_depth_m"], errors="coerce").to_numpy(
        dtype=np.float64, copy=False
    )
    if np.isfinite(local_depth).any() and float(np.nanmin(local_depth)) < 0.0:
        raise ValueError("local_depth_m must be non-negative for finite entries")

    cap_valid = pd.to_numeric(master_df["local_breaking_cap_valid"], errors="coerce").to_numpy(
        dtype=np.float64, copy=False
    )
    finite_cap_valid = cap_valid[np.isfinite(cap_valid)]
    if finite_cap_valid.size:
        unique_flags = sorted(set(int(v) for v in np.unique(finite_cap_valid)))
        if any(v not in (0, 1) for v in unique_flags):
            raise ValueError("local_breaking_cap_valid must contain only 0/1 values")


def build_ray_site_summary(ray_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-ray output into one static row per nearshore site."""
    if "site_name" not in ray_df.columns:
        raise KeyError("ray_features.csv must contain a 'site_name' column")

    work = ray_df.copy()
    expected_ray_columns = [
        *POROSITY_COLUMNS,
        "static_dist_to_coast_m",
        "static_local_depth_m",
        "local_depth_m",
        "local_breaking_cap_valid",
        "static_nearest_shore_steepness",
        "static_nearest_shore_normal_deg",
        "fetch_m",
        *RAY_SECTOR_COLUMNS,
    ]
    for col in expected_ray_columns:
        if col not in work.columns:
            work[col] = np.nan

    if "hit_land" not in work.columns:
        work["hit_land"] = np.nan

    hit_land_numeric = pd.to_numeric(work["hit_land"], errors="coerce")
    unresolved = hit_land_numeric.isna()
    if unresolved.any():
        mapped = (
            work.loc[unresolved, "hit_land"]
            .astype(str)
            .str.strip()
            .str.lower()
            .map({"true": 1.0, "false": 0.0})
        )
        hit_land_numeric.loc[unresolved] = mapped
    work["hit_land_numeric"] = hit_land_numeric

    grouped = work.groupby("site_name", sort=False)
    agg_spec: dict[str, tuple[str, str]] = {
        "static_dist_to_coast_m": ("static_dist_to_coast_m", "first"),
        "static_local_depth_m": ("static_local_depth_m", "first"),
        "local_depth_m": ("local_depth_m", "first"),
        "local_breaking_cap_valid": ("local_breaking_cap_valid", "first"),
        "static_nearest_shore_steepness": ("static_nearest_shore_steepness", "first"),
        "static_nearest_shore_normal_deg": ("static_nearest_shore_normal_deg", "first"),
        "ray_fetch_min_m": ("fetch_m", "min"),
        "ray_fetch_mean_m": ("fetch_m", "mean"),
        "ray_fetch_max_m": ("fetch_m", "max"),
        "ray_hit_land_fraction": ("hit_land_numeric", "mean"),
    }
    for col in POROSITY_COLUMNS:
        agg_spec[col] = (col, "first")
    for col in RAY_SECTOR_COLUMNS:
        agg_spec[col] = (col, "first")

    summary = grouped.agg(**agg_spec).reset_index()

    count_df = grouped.size().reset_index(name="ray_count")
    summary = summary.merge(count_df, on="site_name", how="left", validate="one_to_one")
    return summary


def build_master_static_features(
    routes_df: pd.DataFrame,
    ray_df: pd.DataFrame,
    *,
    breaking_enabled: bool,
    breaking_gamma: float,
) -> pd.DataFrame:
    """Merge routing and ray outputs into one row per nearshore site."""
    route_defaults: dict[str, Any] = {
        "site_name": "",
        "site_lat": np.nan,
        "site_lon": np.nan,
        "site_x": np.nan,
        "site_y": np.nan,
        "site_row": np.nan,
        "site_col": np.nan,
        "reachable": False,
        "path_length_m": np.nan,
        "path_point_count": np.nan,
        "snap_distance_m": np.nan,
        "path_bottleneck_m": np.nan,
        "funneling_ratio": np.nan,
        "choke_out_ratio": np.nan,
        "static_tortuosity_sum": np.nan,
        "static_signed_curvature_deg": np.nan,
        "static_net_deflection_deg": np.nan,
        "static_final_approach_deg": np.nan,
    }
    for col, default in route_defaults.items():
        if col not in routes_df.columns:
            routes_df[col] = default

    routes = routes_df[
        [
            "site_name",
            "site_lat",
            "site_lon",
            "site_x",
            "site_y",
            "site_row",
            "site_col",
            "reachable",
            "path_length_m",
            "path_point_count",
            "snap_distance_m",
            "path_bottleneck_m",
            "funneling_ratio",
            "choke_out_ratio",
            "static_tortuosity_sum",
            "static_signed_curvature_deg",
            "static_net_deflection_deg",
            "static_final_approach_deg",
        ]
    ].copy()

    routes["reachable"] = routes["reachable"].fillna(False).astype(bool)
    for col in (
        "site_lat",
        "site_lon",
        "site_x",
        "site_y",
        "path_length_m",
        "snap_distance_m",
        "path_bottleneck_m",
        "funneling_ratio",
        "choke_out_ratio",
        "static_tortuosity_sum",
        "static_signed_curvature_deg",
        "static_net_deflection_deg",
        "static_final_approach_deg",
    ):
        routes[col] = pd.to_numeric(routes[col], errors="coerce")

    for col in ("site_row", "site_col", "path_point_count"):
        routes[col] = pd.to_numeric(routes[col], errors="coerce").astype("Int64")

    ray_summary = build_ray_site_summary(ray_df)
    master_df = routes.merge(ray_summary, on="site_name", how="left", validate="one_to_one")
    master_df = add_fetch_anisotropy_features(master_df)
    master_df = add_path_geometry_ratio_features(master_df, routes_df=routes_df)
    master_df = add_fjordness_features(master_df)
    master_df = add_local_breaking_features(
        master_df,
        enabled=breaking_enabled,
        gamma=breaking_gamma,
    )

    for col in MASTER_COLUMNS:
        if col not in master_df.columns:
            master_df[col] = np.nan

    master_df = master_df[MASTER_COLUMNS].reset_index(drop=True)
    master_df["ray_count"] = pd.to_numeric(master_df["ray_count"], errors="coerce").astype("Int64")
    validate_master_static_features(master_df)
    return master_df
