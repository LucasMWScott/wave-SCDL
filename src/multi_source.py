"""Helpers for deterministic multi-source NORA3 selection and geometry features."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd


from coastal_wave.common.sources import (
    _try_float as _try_float,
    is_wave_source_name as is_wave_source_name,
    great_circle_distance_m as great_circle_distance_m,
    initial_bearing_deg as initial_bearing_deg,
    inverse_distance_weights as inverse_distance_weights,
    resolve_multi_source_config as resolve_multi_source_config,
    build_k_nearest_source_metadata as build_k_nearest_source_metadata,
    save_source_metadata_json as save_source_metadata_json,
    summarize_nearest_distances_km as summarize_nearest_distances_km,
)

EARTH_RADIUS_M = 6_371_000.0
_SECTOR_LABELS: Tuple[str, ...] = (
    "N",
    "NNE",
    "NE",
    "ENE",
    "E",
    "ESE",
    "SE",
    "SSE",
    "S",
    "SSW",
    "SW",
    "WSW",
    "W",
    "WNW",
    "NW",
    "NNW",
)
_SECTOR_ANGLES_DEG = np.arange(0.0, 360.0, 360.0 / len(_SECTOR_LABELS), dtype=np.float64)


def build_source_geometry_array(
    source_metadata: dict,
    max_distance_km_error: float,
) -> Tuple[np.ndarray, List[str]]:
    """Build `[N_sites, K, Dg]` geometry tensor from source metadata."""
    target_sites = list(source_metadata.get("target_sites", []) or [])
    distances_all = np.asarray(source_metadata.get("distances_m", []), dtype=np.float32)
    bearings_all = np.asarray(source_metadata.get("bearings_deg", []), dtype=np.float32)
    weights_all = np.asarray(source_metadata.get("weights", []), dtype=np.float32)

    if (
        distances_all.ndim != 2
        or bearings_all.shape != distances_all.shape
        or weights_all.shape != distances_all.shape
    ):
        raise ValueError("Invalid source metadata shapes for geometry array construction")
    n_sites, k_nearest = distances_all.shape
    if n_sites != len(target_sites):
        raise ValueError("target_sites length does not match source-metadata array shape")

    scale_m = max(float(max_distance_km_error) * 1000.0, 1.0)
    bearing_rad = np.deg2rad(bearings_all.astype(np.float64))
    distance_norm = np.clip(distances_all / scale_m, 0.0, 1.0)
    bearing_sin = np.sin(bearing_rad).astype(np.float32)
    bearing_cos = np.cos(bearing_rad).astype(np.float32)

    rank_one_hot = np.zeros((n_sites, k_nearest, k_nearest), dtype=np.float32)
    for rank_idx in range(k_nearest):
        rank_one_hot[:, rank_idx, rank_idx] = 1.0

    geometry = np.concatenate(
        [
            distance_norm[..., None].astype(np.float32),
            bearing_sin[..., None],
            bearing_cos[..., None],
            weights_all[..., None].astype(np.float32),
            rank_one_hot,
        ],
        axis=-1,
    )
    feature_names = [
        "distance_m_norm",
        "bearing_sin",
        "bearing_cos",
        "inverse_distance_weight",
        *[f"source_rank_{rank_idx + 1}" for rank_idx in range(k_nearest)],
    ]
    return geometry.astype(np.float32, copy=False), feature_names


def load_optional_routes_dataframe(routing_path: str | Path | None) -> pd.DataFrame | None:
    if not routing_path:
        return None
    path = Path(routing_path)
    if not path.exists():
        return None
    try:
        import pickle

        with path.open("rb") as fh:
            payload = pickle.load(fh)
    except Exception:
        return None

    routes = payload.get("routes") if isinstance(payload, dict) else None
    if isinstance(routes, pd.DataFrame):
        return routes.copy()
    return None


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    num = np.asarray(numerator, dtype=np.float64)
    den = np.asarray(denominator, dtype=np.float64)
    out = np.full(np.broadcast(num, den).shape, np.nan, dtype=np.float64)
    valid = np.isfinite(num) & np.isfinite(den) & (np.abs(den) > 0.0)
    out[valid] = num[valid] / den[valid]
    return out


def circular_interp(values: Sequence[float], query_deg: np.ndarray) -> np.ndarray:
    vals = np.asarray(values, dtype=np.float64)
    query = np.mod(np.asarray(query_deg, dtype=np.float64), 360.0)
    extended_angles = np.concatenate([_SECTOR_ANGLES_DEG, [_SECTOR_ANGLES_DEG[0] + 360.0]])
    extended_vals = np.concatenate([vals, [vals[0]]])
    return np.interp(query, extended_angles, extended_vals)


def _sector_columns(prefix: str, suffix: str = "") -> List[str]:
    return [f"{prefix}_{sector}{suffix}" for sector in _SECTOR_LABELS]


def build_site_local_directional_features(
    static_row: pd.Series,
    direction_queries: Dict[str, np.ndarray],
) -> pd.DataFrame:
    """Build site-centered directional exposure features for named direction arrays."""
    fetch_values = static_row[_sector_columns("ray_fetch", "_m")].to_numpy(
        dtype=np.float64, copy=True
    )
    slope_values = static_row[_sector_columns("ray_max_slope")].to_numpy(
        dtype=np.float64, copy=True
    )
    fetch_max = float(
        pd.to_numeric(pd.Series([static_row.get("ray_fetch_max_m")]), errors="coerce").iloc[0]
    )

    out: Dict[str, np.ndarray] = {}
    for label, query_deg in direction_queries.items():
        query = np.asarray(query_deg, dtype=np.float64)
        if query.ndim != 1:
            raise ValueError(f"Direction query '{label}' must be 1D, got shape {query.shape}")
        fetch = circular_interp(fetch_values, query)
        slope = circular_interp(slope_values, query)
        blocking = np.clip(1.0 - _safe_divide(fetch, np.full_like(fetch, fetch_max)), 0.0, 1.0)
        out[f"fetch_at_{label}_direction_m"] = fetch
        out[f"blocking_at_{label}_direction"] = blocking
        out[f"slope_at_{label}_direction"] = slope
    return pd.DataFrame(out)
