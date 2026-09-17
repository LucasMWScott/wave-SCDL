#!/usr/bin/env python3
"""Build the reusable point-centric dataset consumed by model runs.

Raw offshore and nearshore time series are aligned by site and timestamp,
converted into dynamic histories and target/reference tensors, then joined to
static geometry and optional bathymetry patches.  The module fits every
normalizer on the configured training subset and writes the resulting state
alongside the arrays so evaluation and inference can reproduce it exactly.
"""

from __future__ import annotations

import glob
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    from config_loader import read_yaml_config
except Exception:
    from src.config_loader import read_yaml_config

try:
    from runtime_paths import normalize_runtime_config_paths
except Exception:
    from src.runtime_paths import normalize_runtime_config_paths

try:
    from ablation import (
        DEFAULT_ABLATION_CONFIG_PATH,
        load_ablation_config,
        resolve_raw_static_ablation,
    )
    from preprocess.build_bathy_patches import build_bathymetry_patch_dataset
    from data_pipeline import (
        resolve_site_split_config,
        resolve_static_ablation_config_path,
        warn_if_legacy_validation_mode,
    )
    from multi_source import (
        build_k_nearest_source_metadata,
        build_site_local_directional_features,
        build_source_geometry_array,
        great_circle_distance_m,
        resolve_multi_source_config,
        save_source_metadata_json,
        summarize_nearest_distances_km,
    )
    from preprocessing.transfer_targets import (
        PHYSICAL_TARGET_NAMES,
        REFERENCE_TARGET_NAMES,
        TRANSFER_SCALER_COLUMNS,
        TRANSFER_TARGET_NAMES,
        build_transfer_targets,
        circular_weighted_mean_deg,
        resolve_targets_config,
        validate_target_name_block,
    )
except Exception:
    from src.ablation import (
        DEFAULT_ABLATION_CONFIG_PATH,
        load_ablation_config,
        resolve_raw_static_ablation,
    )
    from src.preprocess.build_bathy_patches import build_bathymetry_patch_dataset
    from src.data_pipeline import (
        resolve_site_split_config,
        resolve_static_ablation_config_path,
        warn_if_legacy_validation_mode,
    )
    from src.multi_source import (
        build_k_nearest_source_metadata,
        build_site_local_directional_features,
        build_source_geometry_array,
        great_circle_distance_m,
        resolve_multi_source_config,
        save_source_metadata_json,
        summarize_nearest_distances_km,
    )
    from src.preprocessing.transfer_targets import (
        PHYSICAL_TARGET_NAMES,
        REFERENCE_TARGET_NAMES,
        TRANSFER_SCALER_COLUMNS,
        TRANSFER_TARGET_NAMES,
        build_transfer_targets,
        circular_weighted_mean_deg,
        resolve_targets_config,
        validate_target_name_block,
    )


from src.preprocessing.timeseries import (
    _build_split_indices as _build_split_indices,
    _filename_matches_site_name as _filename_matches_site_name,
    _normalize_timestamp_naive as _normalize_timestamp_naive,
    _parse_date_range_bound as _parse_date_range_bound,
    _parse_time_index as _parse_time_index,
    _resolve_and_apply_date_range as _resolve_and_apply_date_range,
    find_param_files as find_param_files,
    load_site_timeseries as load_site_timeseries,
)

_NORMALIZATION_MODULE = None
_DEFAULT_STATIC_SITE_METADATA_COLUMNS: Tuple[str, ...] = (
    "site_lon",
    "site_lat",
    "site_x",
    "site_y",
    "site_row",
    "site_col",
    "reachable",
)
_EXPECTED_TARGET_FEATURE_ORDER: List[str] = ["hs", "tp", "dir_sin", "dir_cos", "dp_sin", "dp_cos"]


def read_yaml(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    return read_yaml_config(p)


def _resolve_runtime_source_geometry_features_flag(train_cfg: dict) -> bool:
    data_cfg = (train_cfg.get("data", {}) or {}) if isinstance(train_cfg, dict) else {}
    model_cfg = (train_cfg.get("model", {}) or {}) if isinstance(train_cfg, dict) else {}
    coastal_cfg = (
        (model_cfg.get("coastal_transformer", {}) or {}) if isinstance(model_cfg, dict) else {}
    )
    multi_source_cfg = (
        (coastal_cfg.get("multi_source", {}) or {}) if isinstance(coastal_cfg, dict) else {}
    )

    if "use_geometry_features" in multi_source_cfg:
        return bool(multi_source_cfg.get("use_geometry_features", False))
    if "use_geometry" in data_cfg:
        return bool(data_cfg.get("use_geometry", False))
    return False


def _remove_stale_optional_artifact(path: Path) -> None:
    if path.exists():
        path.unlink()
        logging.info("Removed stale optional artifact: %s", path)


def _load_normalization_module():
    """Return the canonical normalization implementation."""
    from src.preprocessing import normalize

    return normalize


def _safe_glob(params_dir: str, pattern: str) -> List[str]:
    pat = os.path.join(params_dir, "**", pattern)
    return sorted(
        [p for p in glob.glob(pat, recursive=True) if "Zone.Identifier" not in os.path.basename(p)]
    )


def _legacy_site_holdout_keys_present(train_cfg: dict) -> List[str]:
    data_cfg = (train_cfg.get("data", {}) or {}) if isinstance(train_cfg, dict) else {}
    split_cfg = (train_cfg.get("split", {}) or {}) if isinstance(train_cfg, dict) else {}

    present: List[str] = []
    if isinstance(data_cfg, dict) and "holdout_sites" in data_cfg:
        present.append("data.holdout_sites")
    if isinstance(split_cfg, dict) and "norac_holdout" in split_cfg:
        present.append("split.norac_holdout")
    return present


def _raise_if_legacy_site_holdout_keys_present(train_cfg: dict) -> None:
    present = _legacy_site_holdout_keys_present(train_cfg)
    if not present:
        return
    raise ValueError(
        "Legacy site holdout keys are no longer supported: "
        f"{present}. Use data.validation_sites for held-out validation sites "
        "and data.test_sites for final test sites."
    )


def encode_circular_columns(
    df: pd.DataFrame, columns: List[str], input_degrees: bool = True, drop_original: bool = True
) -> pd.DataFrame:
    norm_mod = _load_normalization_module()
    return norm_mod.encode_cyclical_columns(
        df,
        columns,
        input_degrees=input_degrees,
        drop_original=drop_original,
    )


def apply_nora3_wave_direction_offset(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Shift NORA3 wave-direction columns by +180 degrees in-place.

    NORA3 wave-direction parameters use the opposite directional convention
    relative to the downstream encoding used by this project. Wind-direction
    columns are already aligned and must remain unchanged.
    """
    out = df.copy()
    for col in columns:
        name = str(col)
        low = name.lower()
        if name not in out.columns:
            continue
        if "wind" in low and ("dir" in low or "direction" in low):
            continue
        if low not in {"pdir", "thq", "thq_sea", "thq_swell"}:
            continue
        values = pd.to_numeric(out[name], errors="coerce").astype(float)
        out[name] = np.mod(values + 180.0, 360.0)
    return out


def _resolve_training_sites_for_normalization(
    all_sites: List[str],
    data_cfg: dict,
    split_cfg: dict,
) -> List[str]:
    """Resolve nearshore sites allowed to contribute scaler-fit statistics.

    Training normalization excludes holdout/test sites so scaler statistics are
    derived only from sites used for model training.
    """
    _raise_if_legacy_site_holdout_keys_present({"data": data_cfg, "split": split_cfg})
    requested_train = [str(s) for s in (data_cfg.get("train_sites", []) or [])]
    test_site_set = {str(s) for s in (data_cfg.get("test_sites", []) or [])}

    if requested_train:
        requested_set = set(requested_train)
        selected = [s for s in all_sites if s in requested_set]
    else:
        selected = list(all_sites)

    selected = [s for s in selected if s not in test_site_set]
    if not selected:
        raise ValueError(
            "No nearshore sites available for normalization fit after applying "
            "train_sites and data.test_sites exclusions."
        )
    return selected


def _fit_scaler(values: np.ndarray, method: str, feature_range: Tuple[float, float]) -> dict:
    x = np.asarray(values, dtype=float)
    if x.ndim == 1:
        x = x.reshape(-1, 1)

    def _col_stats(arr: np.ndarray):
        n_cols = arr.shape[1]
        mean = np.zeros(n_cols, dtype=float)
        std = np.ones(n_cols, dtype=float)
        minv = np.zeros(n_cols, dtype=float)
        maxv = np.ones(n_cols, dtype=float)
        for j in range(n_cols):
            col = arr[:, j]
            valid = col[~np.isnan(col)]
            if valid.size == 0:
                # default stats for all-NaN columns (leave transformed values as NaN)
                mean[j] = 0.0
                std[j] = 1.0
                minv[j] = 0.0
                maxv[j] = 1.0
                continue
            mean[j] = float(np.mean(valid))
            s = float(np.std(valid))
            std[j] = 1.0 if s == 0.0 else s
            minv[j] = float(np.min(valid))
            maxv[j] = float(np.max(valid))
            if maxv[j] == minv[j]:
                maxv[j] = minv[j] + 1.0
        return mean, std, minv, maxv

    mean, std, minv, maxv = _col_stats(x)

    if method == "zscore":
        return {"method": method, "mean": mean, "std": std}

    if method == "minmax":
        a, b = feature_range
        return {"method": method, "min": minv, "max": maxv, "feature_range": (a, b)}

    # fallback to zscore for unsupported methods
    return {"method": "zscore", "mean": mean, "std": std}


def _apply_scaler(values: np.ndarray, scaler: dict) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    if x.ndim == 1:
        x = x.reshape(-1, 1)

    method = scaler.get("method", "zscore")
    if method == "zscore":
        return (x - scaler["mean"]) / scaler["std"]

    if method == "minmax":
        minv = scaler["min"]
        maxv = scaler["max"]
        a, b = scaler.get("feature_range", (0.0, 1.0))
        span = maxv - minv
        span[span == 0] = 1.0
        y = (x - minv) / span
        return y * (b - a) + a

    return x


def _normalize_magnitude_columns(
    df: pd.DataFrame,
    magnitude_cols: List[str],
    train_idx: np.ndarray,
    method: str,
    feature_range: Tuple[float, float],
) -> Tuple[pd.DataFrame, dict]:
    out = df.copy()
    if not magnitude_cols:
        return out, {"method": method, "columns": []}

    train_vals = out.iloc[train_idx][magnitude_cols].values
    scaler = _fit_scaler(train_vals, method, feature_range)
    out.loc[:, magnitude_cols] = _apply_scaler(out[magnitude_cols].values, scaler)
    scaler["columns"] = list(magnitude_cols)
    return out, scaler


def _safe_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in name)


def _try_float(value: object) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    if np.isnan(out) or np.isinf(out):
        return None
    return out


def _trailing_site_index(name: str) -> int | None:
    tail = str(name).rsplit("_", 1)[-1]
    if tail.isdigit():
        return int(tail)
    return None


def _build_wave_to_wind_site_map(sites_cfg: dict, wave_sites: List[str]) -> Dict[str, str]:
    """Map each wave-grid offshore site to a wind-grid offshore site.

    Mapping preference:
    1) same trailing site index (e.g. nora3_grid_10 -> nora3_wind_grid_10)
    2) nearest wind-grid site by lat/lon
    3) first wind-grid site as a final fallback
    """
    offshore_entries = sites_cfg.get("offshore_sites") or []
    offshore_meta: Dict[str, dict] = {
        str(s.get("name")): s for s in offshore_entries if s.get("name") is not None
    }

    wind_sites = [name for name in offshore_meta if "wind" in name.lower()]
    if not wind_sites:
        return {}

    wind_by_idx: Dict[int, str] = {}
    for ws in wind_sites:
        idx = _trailing_site_index(ws)
        if idx is not None:
            wind_by_idx[idx] = ws

    mapping: Dict[str, str] = {}
    for wave in wave_sites:
        chosen: str | None = None

        idx = _trailing_site_index(wave)
        if idx is not None and idx in wind_by_idx:
            chosen = wind_by_idx[idx]

        if chosen is None:
            wave_meta = offshore_meta.get(wave, {})
            wave_lat = _try_float(wave_meta.get("lat"))
            wave_lon = _try_float(wave_meta.get("lon"))

            best_dist = float("inf")
            for ws in wind_sites:
                ws_meta = offshore_meta.get(ws, {})
                ws_lat = _try_float(ws_meta.get("lat"))
                ws_lon = _try_float(ws_meta.get("lon"))
                if wave_lat is None or wave_lon is None or ws_lat is None or ws_lon is None:
                    continue
                dist = (wave_lat - ws_lat) ** 2 + (wave_lon - ws_lon) ** 2
                if dist < best_dist:
                    best_dist = dist
                    chosen = ws

        if chosen is None:
            chosen = wind_sites[0]

        mapping[wave] = chosen

    return mapping


def _site_lookup(entries: List[dict]) -> Dict[str, dict]:
    return {str(entry.get("name")): entry for entry in entries if entry.get("name") is not None}


def _is_wind_site_name(name: object) -> bool:
    return "wind" in str(name or "").strip().lower()


def _resolve_wind_feature_toggles(data_cfg: dict) -> Tuple[bool, bool]:
    return (
        bool(data_cfg.get("use_offshore_wind_features", True)),
        bool(data_cfg.get("use_local_wind_features", True)),
    )


def _build_nearest_local_wind_metadata(
    nearshore_entries: List[dict],
    offshore_entries: List[dict],
) -> Tuple[Dict[str, dict], dict]:
    wind_candidates = []
    for entry in offshore_entries:
        name = str(entry.get("name", "")).strip()
        lat = _try_float(entry.get("lat"))
        lon = _try_float(entry.get("lon"))
        if not name or lat is None or lon is None or not _is_wind_site_name(name):
            continue
        wind_candidates.append(
            {
                "name": name,
                "lat": float(lat),
                "lon": float(lon),
            }
        )
    if not wind_candidates:
        raise ValueError(
            "No candidate local wind points found from sites.yaml offshore_sites entries"
        )

    mapping_by_site: Dict[str, dict] = {}
    records: List[dict] = []
    distances_m: List[float] = []

    for nearshore in nearshore_entries:
        site_name = str(nearshore.get("name", "")).strip()
        site_lat = _try_float(nearshore.get("lat"))
        site_lon = _try_float(nearshore.get("lon"))
        if not site_name or site_lat is None or site_lon is None:
            raise ValueError(f"Nearshore site is missing valid name/lat/lon: {nearshore}")

        best_candidate = None
        best_distance_m = float("inf")
        for candidate in wind_candidates:
            distance_m = great_circle_distance_m(
                site_lat,
                site_lon,
                candidate["lat"],
                candidate["lon"],
            )
            if distance_m < best_distance_m:
                best_distance_m = float(distance_m)
                best_candidate = candidate

        if best_candidate is None:
            raise ValueError(
                f"Failed to resolve nearest local wind point for nearshore site '{site_name}'"
            )

        record = {
            "norac_point_id": site_name,
            "norac_point_name": site_name,
            "norac_lat": float(site_lat),
            "norac_lon": float(site_lon),
            "local_wind_point_id": str(best_candidate["name"]),
            "local_wind_point_name": str(best_candidate["name"]),
            "local_wind_lat": float(best_candidate["lat"]),
            "local_wind_lon": float(best_candidate["lon"]),
            "distance_m": float(best_distance_m),
        }
        mapping_by_site[site_name] = record
        records.append(record)
        distances_m.append(float(best_distance_m))

    distance_array = np.asarray(distances_m, dtype=np.float64)
    diagnostics = {
        "num_norac_points": int(len(records)),
        "num_candidate_local_wind_points": int(len(wind_candidates)),
        "max_nearest_local_wind_distance_m": float(np.nanmax(distance_array))
        if distance_array.size
        else float("nan"),
        "median_nearest_local_wind_distance_m": float(np.nanmedian(distance_array))
        if distance_array.size
        else float("nan"),
        "example_local_wind_assignments": records[:5],
        "candidate_local_wind_points": [str(item["name"]) for item in wind_candidates],
        "assignments": records,
    }
    return mapping_by_site, diagnostics


def _resolve_wave_site_for_nearshore(
    nearshore_entry: dict,
    offshore_lookup: Dict[str, dict],
    available_wave_sites: List[str],
) -> str | None:
    """Select the offshore wave site used for site-specific dynamic geometry."""
    candidates = [
        str(name)
        for name in (nearshore_entry.get("paired_offshore") or [])
        if str(name) in available_wave_sites and "wind" not in str(name).lower()
    ]
    if not candidates:
        candidates = list(available_wave_sites)
    if not candidates:
        return None

    near_lat = _try_float(nearshore_entry.get("lat"))
    near_lon = _try_float(nearshore_entry.get("lon"))
    if near_lat is None or near_lon is None:
        return candidates[0]

    best_site = candidates[0]
    best_dist = float("inf")
    for site_name in candidates:
        meta = offshore_lookup.get(site_name, {})
        off_lat = _try_float(meta.get("lat"))
        off_lon = _try_float(meta.get("lon"))
        if off_lat is None or off_lon is None:
            continue
        dist = (near_lat - off_lat) ** 2 + (near_lon - off_lon) ** 2
        if dist < best_dist:
            best_dist = dist
            best_site = site_name
    return best_site


def _load_master_static_frame(static_csv: str, nearshore_sites: List[str]) -> pd.DataFrame:
    """Load master static CSV and align rows to nearshore site order."""
    if not static_csv or not os.path.exists(static_csv):
        raise FileNotFoundError(
            "Static feature CSV not found. Expected master static file at "
            "data/processed/master_static_features.csv (or configured static_features.out_path)."
        )

    sdf = pd.read_csv(static_csv)
    if "site_name" not in sdf.columns:
        raise ValueError(
            f"Static CSV {static_csv} must contain a 'site_name' column. "
            "This pipeline expects master_static_features.csv schema."
        )

    sdf = sdf.copy()
    sdf["site_name"] = sdf["site_name"].astype(str)
    if sdf["site_name"].duplicated().any():
        dupes = sorted(
            sdf.loc[sdf["site_name"].duplicated(), "site_name"].astype(str).unique().tolist()
        )
        raise ValueError(f"Duplicate static rows found for sites: {dupes}")

    missing_sites = sorted(set(nearshore_sites) - set(sdf["site_name"].tolist()))
    if missing_sites:
        raise ValueError(
            "master_static_features.csv is missing rows for nearshore sites: "
            f"{missing_sites}. Confirm site names against configs/sites.yaml."
        )

    return sdf.set_index("site_name").reindex(nearshore_sites).reset_index()


def _safe_divide_array(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    num = np.asarray(numerator, dtype=np.float64)
    den = np.asarray(denominator, dtype=np.float64)
    out = np.full(np.broadcast(num, den).shape, np.nan, dtype=np.float64)
    valid = np.isfinite(num) & np.isfinite(den) & (np.abs(den) > 0.0)
    out[valid] = num[valid] / den[valid]
    return out


def _circular_distance_deg(values_deg: np.ndarray, reference_deg: np.ndarray) -> np.ndarray:
    delta = np.abs(
        np.asarray(values_deg, dtype=np.float64) - np.asarray(reference_deg, dtype=np.float64)
    )
    return np.minimum(delta, 360.0 - delta)


def _circular_interp(
    values: np.ndarray, query_deg: np.ndarray, sector_angles_deg: np.ndarray
) -> np.ndarray:
    angles = np.asarray(sector_angles_deg, dtype=np.float64)
    vals = np.asarray(values, dtype=np.float64)
    query = np.mod(np.asarray(query_deg, dtype=np.float64), 360.0)
    extended_angles = np.concatenate([angles, [angles[0] + 360.0]])
    extended_vals = np.concatenate([vals, [vals[0]]])
    return np.interp(query, extended_angles, extended_vals)


def _resolve_site_dynamic_wave_source_map(
    sites_cfg: dict,
    available_wave_sites: List[str],
) -> Dict[str, str]:
    nearshore_lookup = _site_lookup(list(sites_cfg.get("nearshore_sites") or []))
    offshore_lookup = _site_lookup(list(sites_cfg.get("offshore_sites") or []))
    mapping: Dict[str, str] = {}
    for site_name, nearshore_entry in nearshore_lookup.items():
        chosen = _resolve_wave_site_for_nearshore(
            nearshore_entry=nearshore_entry,
            offshore_lookup=offshore_lookup,
            available_wave_sites=available_wave_sites,
        )
        if chosen is not None:
            mapping[site_name] = chosen
    return mapping


def _build_site_dynamic_sidecar(
    master_static_df: pd.DataFrame,
    prepared_offshore_by_site: Dict[str, pd.DataFrame],
    local_wind_by_site: Dict[str, pd.DataFrame],
    aligned_index: pd.Index,
    site_to_wave_site: Dict[str, str],
    input_degrees: bool,
    include_local_wind_features: bool = True,
) -> Tuple[Dict[str, pd.DataFrame], List[str], List[str]]:
    """Build per-site dynamic geometry interaction features."""
    sector_angles_deg = np.arange(0.0, 360.0, 360.0 / 16.0, dtype=np.float64)
    sector_fetch_cols = [
        f"ray_fetch_{sector}_m"
        for sector in (
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
    ]
    sector_slope_cols = [
        f"ray_max_slope_{sector}"
        for sector in (
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
    ]
    sector_laplacian_cols = [
        f"ray_max_laplacian_{sector}"
        for sector in (
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
    ]
    sector_min_depth_cols = [
        f"ray_min_depth_{sector}_m"
        for sector in (
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
    ]

    feature_names = [
        "wave_fetch_aligned_m",
        "wave_fetch_aligned_ratio",
        "wave_slope_aligned",
        "wave_laplacian_aligned",
        "wave_min_depth_aligned_m",
        "wave_blocked_sector_fraction_pm30",
        "wave_open_sector_fraction_pm30",
    ]
    if include_local_wind_features:
        feature_names.extend(
            [
                "local_wind_fetch_aligned_m",
                "local_wind_fetch_aligned_ratio",
                "local_windsea_proxy_u2_fetch",
                "local_windsea_proxy_u2_fetch_ratio",
                "local_windsea_proxy_3h_mean",
                "local_windsea_proxy_6h_mean",
                "local_windsea_proxy_12h_mean",
                "local_wind_speed_10m",
                "local_wind_dir_sin",
                "local_wind_dir_cos",
            ]
        )

    sitewise_frames: Dict[str, pd.DataFrame] = {}
    warnings: List[str] = []
    static_lookup = master_static_df.set_index("site_name")

    for site_name in master_static_df["site_name"].astype(str):
        wave_site = site_to_wave_site.get(site_name)
        offshore_df = prepared_offshore_by_site.get(wave_site or "")
        local_wind_df = local_wind_by_site.get(site_name) if include_local_wind_features else None
        if offshore_df is None or offshore_df.empty:
            warnings.append(
                f"site dynamic sidecar missing offshore source for nearshore site: {site_name}"
            )
            sitewise_frames[site_name] = pd.DataFrame(
                index=aligned_index, columns=feature_names, dtype=float
            )
            continue
        if include_local_wind_features and (local_wind_df is None or local_wind_df.empty):
            warnings.append(
                f"site dynamic sidecar missing local wind source for nearshore site: {site_name}"
            )
            sitewise_frames[site_name] = pd.DataFrame(
                index=aligned_index, columns=feature_names, dtype=float
            )
            continue

        static_row = static_lookup.loc[site_name]
        fetch_values = static_row[sector_fetch_cols].to_numpy(dtype=np.float64, copy=True)
        slope_values = static_row[sector_slope_cols].to_numpy(dtype=np.float64, copy=True)
        lap_values = static_row[sector_laplacian_cols].to_numpy(dtype=np.float64, copy=True)
        min_depth_values = static_row[sector_min_depth_cols].to_numpy(dtype=np.float64, copy=True)
        fetch_max = float(
            pd.to_numeric(pd.Series([static_row.get("ray_fetch_max_m")]), errors="coerce").iloc[0]
        )

        def _aligned_numeric_series(column_name: str) -> np.ndarray:
            if column_name in offshore_df.columns:
                series = pd.to_numeric(offshore_df[column_name], errors="coerce")
            else:
                series = pd.Series(np.nan, index=offshore_df.index, dtype=float)
            return series.reindex(aligned_index).to_numpy(dtype=np.float64)

        wave_dir = _aligned_numeric_series("thq")
        if include_local_wind_features:
            assert local_wind_df is not None
            local_wind_dir = pd.to_numeric(
                local_wind_df.reindex(aligned_index)["wind_direction_10m"]
                if "wind_direction_10m" in local_wind_df.columns
                else pd.Series(np.nan, index=aligned_index),
                errors="coerce",
            ).to_numpy(dtype=np.float64)
            local_wind_speed = pd.to_numeric(
                local_wind_df.reindex(aligned_index)["wind_speed_10m"]
                if "wind_speed_10m" in local_wind_df.columns
                else pd.Series(np.nan, index=aligned_index),
                errors="coerce",
            ).to_numpy(dtype=np.float64)
        else:
            local_wind_dir = np.full(len(aligned_index), np.nan, dtype=np.float64)
            local_wind_speed = np.full(len(aligned_index), np.nan, dtype=np.float64)

        wave_fetch_aligned = _circular_interp(fetch_values, wave_dir, sector_angles_deg)
        wave_slope_aligned = _circular_interp(slope_values, wave_dir, sector_angles_deg)
        wave_laplacian_aligned = _circular_interp(lap_values, wave_dir, sector_angles_deg)
        wave_min_depth_aligned = _circular_interp(min_depth_values, wave_dir, sector_angles_deg)
        wave_fetch_ratio = np.clip(
            _safe_divide_array(wave_fetch_aligned, np.full_like(wave_fetch_aligned, fetch_max)),
            0.0,
            1.0,
        )
        if include_local_wind_features:
            local_wind_fetch_aligned = _circular_interp(
                fetch_values, local_wind_dir, sector_angles_deg
            )
            local_wind_fetch_ratio = np.clip(
                _safe_divide_array(
                    local_wind_fetch_aligned, np.full_like(local_wind_fetch_aligned, fetch_max)
                ),
                0.0,
                1.0,
            )
        else:
            local_wind_fetch_aligned = np.full(len(aligned_index), np.nan, dtype=np.float64)
            local_wind_fetch_ratio = np.full(len(aligned_index), np.nan, dtype=np.float64)

        angle_distance_wave = _circular_distance_deg(sector_angles_deg[None, :], wave_dir[:, None])
        wave_window = angle_distance_wave <= 30.0
        window_count = np.maximum(np.sum(wave_window, axis=1), 1)
        open_hits = np.sum(
            (
                np.where(np.isfinite(fetch_values), fetch_values[None, :] >= 5_000.0, False)
                & wave_window
            ),
            axis=1,
        )
        blocked_hits = np.sum(
            (
                np.where(np.isfinite(fetch_values), fetch_values[None, :] <= 500.0, False)
                & wave_window
            ),
            axis=1,
        )
        open_sector_fraction = open_hits / window_count
        blocked_sector_fraction = blocked_hits / window_count

        if include_local_wind_features:
            local_wind_fetch_km = np.where(
                np.isfinite(local_wind_fetch_aligned), local_wind_fetch_aligned / 1_000.0, np.nan
            )
            local_wind_u2 = np.where(
                np.isfinite(local_wind_speed), np.square(np.maximum(local_wind_speed, 0.0)), np.nan
            )
            local_windsea_proxy = local_wind_u2 * local_wind_fetch_km
            local_windsea_proxy_ratio = local_wind_u2 * np.where(
                np.isfinite(local_wind_fetch_ratio),
                local_wind_fetch_ratio,
                np.nan,
            )
            local_wind_features = _prepare_local_wind_sequence_frame(
                local_wind_df.reindex(aligned_index),
                input_degrees=input_degrees,
            )
        else:
            local_windsea_proxy = np.full(len(aligned_index), np.nan, dtype=np.float64)
            local_windsea_proxy_ratio = np.full(len(aligned_index), np.nan, dtype=np.float64)
            local_wind_features = pd.DataFrame(index=aligned_index)

        frame = pd.DataFrame(
            {
                "wave_fetch_aligned_m": wave_fetch_aligned,
                "wave_fetch_aligned_ratio": wave_fetch_ratio,
                "wave_slope_aligned": wave_slope_aligned,
                "wave_laplacian_aligned": wave_laplacian_aligned,
                "wave_min_depth_aligned_m": wave_min_depth_aligned,
                "wave_blocked_sector_fraction_pm30": np.clip(blocked_sector_fraction, 0.0, 1.0),
                "wave_open_sector_fraction_pm30": np.clip(open_sector_fraction, 0.0, 1.0),
                "local_wind_fetch_aligned_m": local_wind_fetch_aligned,
                "local_wind_fetch_aligned_ratio": local_wind_fetch_ratio,
                "local_windsea_proxy_u2_fetch": local_windsea_proxy,
                "local_windsea_proxy_u2_fetch_ratio": local_windsea_proxy_ratio,
            },
            index=aligned_index,
        )
        proxy_series = pd.Series(local_windsea_proxy, index=aligned_index, dtype=float)
        if include_local_wind_features:
            frame["local_windsea_proxy_3h_mean"] = (
                proxy_series.rolling("3h", min_periods=1).mean().to_numpy(dtype=np.float64)
            )
            frame["local_windsea_proxy_6h_mean"] = (
                proxy_series.rolling("6h", min_periods=1).mean().to_numpy(dtype=np.float64)
            )
            frame["local_windsea_proxy_12h_mean"] = (
                proxy_series.rolling("12h", min_periods=1).mean().to_numpy(dtype=np.float64)
            )
            for col in ("local_wind_speed_10m", "local_wind_dir_sin", "local_wind_dir_cos"):
                frame[col] = local_wind_features[col].to_numpy(dtype=np.float64, copy=False)
        frame = frame.reindex(columns=feature_names)
        sitewise_frames[site_name] = frame

    return sitewise_frames, feature_names, warnings


def _prepare_local_wind_sequence_frame(
    local_wind_df: pd.DataFrame,
    input_degrees: bool,
) -> pd.DataFrame:
    frame = local_wind_df.copy()
    if "wind_speed_10m" not in frame.columns:
        frame["wind_speed_10m"] = np.nan
    if "wind_direction_10m" not in frame.columns:
        frame["wind_direction_10m"] = np.nan
    frame = frame[["wind_speed_10m", "wind_direction_10m"]]
    encoded = encode_circular_columns(
        frame,
        ["wind_direction_10m"],
        input_degrees=input_degrees,
        drop_original=True,
    )
    out = pd.DataFrame(index=frame.index)
    out["local_wind_speed_10m"] = pd.to_numeric(encoded.get("wind_speed_10m"), errors="coerce")
    out["local_wind_dir_sin"] = pd.to_numeric(
        encoded.get("wind_direction_10m_sin"), errors="coerce"
    )
    out["local_wind_dir_cos"] = pd.to_numeric(
        encoded.get("wind_direction_10m_cos"), errors="coerce"
    )
    return out


def _build_site_local_wind_sequence_sidecar(
    local_wind_by_site: Dict[str, pd.DataFrame],
    aligned_index: pd.Index,
    input_degrees: bool,
) -> Tuple[Dict[str, pd.DataFrame], List[str], List[str]]:
    feature_names = [
        "local_wind_speed_10m",
        "local_wind_dir_sin",
        "local_wind_dir_cos",
    ]
    sitewise_frames: Dict[str, pd.DataFrame] = {}
    warnings: List[str] = []
    for site_name, local_wind_df in sorted(local_wind_by_site.items()):
        if local_wind_df is None or local_wind_df.empty:
            warnings.append(
                f"site local-wind sidecar missing local wind source for nearshore site: {site_name}"
            )
            sitewise_frames[site_name] = pd.DataFrame(
                index=aligned_index, columns=feature_names, dtype=float
            )
            continue
        frame = _prepare_local_wind_sequence_frame(
            local_wind_df.reindex(aligned_index),
            input_degrees=input_degrees,
        ).reindex(columns=feature_names)
        sitewise_frames[site_name] = frame
    return sitewise_frames, feature_names, warnings


def _resolve_static_csv(preprocess_cfg: dict) -> str | None:
    static_cfg = preprocess_cfg.get("static_features", {}) if preprocess_cfg else {}
    configured = static_cfg.get("master_out_path") or static_cfg.get("out_path")
    candidates = [
        configured,
        "data/processed/master_static_features.csv",
        "geometric_builder/data/processed/master_static_features.csv",
        "data/processed/static_features.csv",
        "experiments/static_features.csv",
    ]
    for c in candidates:
        if c and os.path.exists(c):
            return c
    return None


def _resolve_static_excluded_columns(data_cfg: dict) -> List[str]:
    """Resolve static columns to drop before static transformer fitting.

    Config knobs under ``data``:
    - ``static_ignore_site_metadata`` (bool)
    - ``static_ignore_columns`` (list[str])
    """
    out: List[str] = []

    if bool(data_cfg.get("static_ignore_site_metadata", False)):
        out.extend(_DEFAULT_STATIC_SITE_METADATA_COLUMNS)

    out.extend([str(c) for c in (data_cfg.get("static_ignore_columns", []) or [])])

    deduped: List[str] = []
    seen = set()
    for col in out:
        c = str(col).strip()
        if not c or c == "site_name":
            continue
        if c not in seen:
            deduped.append(c)
            seen.add(c)
    return deduped


def _build_point_centric_physics_payload(
    master_static_df: pd.DataFrame,
    target_sites: List[str],
) -> dict:
    """Build raw per-site physics payload aligned to point-centric target sites."""
    required_cols = [
        "site_name",
        "local_depth_m",
        "local_breaking_hs_cap",
        "local_breaking_cap_valid",
    ]
    for col in required_cols:
        if col not in master_static_df.columns:
            if col == "site_name":
                raise KeyError("master_static_features.csv is missing required 'site_name' column")
            master_static_df[col] = np.nan

    work = master_static_df[required_cols].copy()
    work["site_name"] = work["site_name"].astype(str)
    work = work.drop_duplicates(subset=["site_name"], keep="first").set_index("site_name")
    aligned = work.reindex(target_sites)

    local_depth = pd.to_numeric(aligned["local_depth_m"], errors="coerce").to_numpy(
        dtype=np.float32, copy=False
    )
    local_cap = pd.to_numeric(aligned["local_breaking_hs_cap"], errors="coerce").to_numpy(
        dtype=np.float32, copy=False
    )
    local_valid = pd.to_numeric(aligned["local_breaking_cap_valid"], errors="coerce").fillna(0.0)
    local_valid = np.where(local_valid > 0.0, 1.0, 0.0).astype(np.float32, copy=False)

    return {
        "target_sites": np.array(target_sites, dtype=str),
        "local_depth_m": local_depth,
        "local_breaking_hs_cap": local_cap,
        "local_breaking_cap_valid": local_valid,
    }


def _prepare_multisource_feature_frame(
    offshore_df: pd.DataFrame,
    static_row: pd.Series | None,
    local_wind_df: pd.DataFrame | None,
    offshore_vars: List[str],
    direction_vars: set[str],
    auto_dynamic_direction_vars: set[str],
    input_degrees: bool,
    include_local_direction_features: bool = True,
    include_local_wind_features: bool = True,
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """Prepare one source feature frame with local directional exposure features."""
    frame = offshore_df.copy()
    for var_name in offshore_vars:
        if var_name not in frame.columns:
            frame[var_name] = np.nan
    frame = frame[offshore_vars]

    local_direction_frame = pd.DataFrame(index=frame.index)
    if include_local_wind_features:
        if local_wind_df is None:
            raise ValueError(
                "Local wind dataframe is required for multi-source local wind feature preparation"
            )
        local_wind_frame = _prepare_local_wind_sequence_frame(
            local_wind_df.reindex(frame.index),
            input_degrees=input_degrees,
        )
    else:
        local_wind_frame = pd.DataFrame(index=frame.index)
    if include_local_direction_features:
        if static_row is None:
            raise ValueError("Static row is required when include_local_direction_features=True")

        direction_queries = {}
        for label, column_name in (
            ("swell", "thq_swell"),
            ("windwave", "thq_sea"),
        ):
            if column_name in frame.columns:
                direction_queries[label] = pd.to_numeric(
                    frame[column_name], errors="coerce"
                ).to_numpy(dtype=np.float64)
            else:
                direction_queries[label] = np.full(len(frame), np.nan, dtype=np.float64)
        if include_local_wind_features:
            assert local_wind_df is not None
            direction_queries["local_wind"] = pd.to_numeric(
                local_wind_df.reindex(frame.index)["wind_direction_10m"]
                if "wind_direction_10m" in local_wind_df.columns
                else pd.Series(np.nan, index=frame.index),
                errors="coerce",
            ).to_numpy(dtype=np.float64)

        local_direction_frame = build_site_local_directional_features(
            static_row=static_row,
            direction_queries=direction_queries,
        )
        if not local_direction_frame.empty and len(local_direction_frame) == len(frame.index):
            local_direction_frame.index = frame.index

    site_circular = [
        var_name
        for var_name in offshore_vars
        if (var_name in direction_vars or var_name in auto_dynamic_direction_vars)
    ]
    encoded = encode_circular_columns(
        frame, site_circular, input_degrees=input_degrees, drop_original=True
    )

    keep_cols: List[str] = []
    circular_cols: List[str] = []
    for var_name in offshore_vars:
        if var_name in direction_vars:
            sin_col = f"{var_name}_sin"
            cos_col = f"{var_name}_cos"
            if sin_col not in encoded.columns:
                encoded[sin_col] = np.nan
            if cos_col not in encoded.columns:
                encoded[cos_col] = np.nan
            keep_cols.extend([sin_col, cos_col])
            circular_cols.extend([sin_col, cos_col])
        else:
            if var_name not in encoded.columns:
                encoded[var_name] = np.nan
            keep_cols.append(var_name)

    out = encoded[keep_cols].copy()
    if not local_direction_frame.empty:
        local_direction_frame = local_direction_frame.reindex(out.index)
        for col in local_direction_frame.columns:
            out[col] = local_direction_frame[col].to_numpy(dtype=np.float64, copy=False)
    if include_local_wind_features:
        local_wind_frame = local_wind_frame.reindex(out.index)
        for col in ("local_wind_speed_10m", "local_wind_dir_sin", "local_wind_dir_cos"):
            out[col] = local_wind_frame[col].to_numpy(dtype=np.float64, copy=False)
    feature_names = list(out.columns)
    circular_cols.extend(
        [col for col in out.columns if col.endswith("_sin") or col.endswith("_cos")]
    )
    return out, feature_names, sorted(set(circular_cols))


def _build_multisource_dynamic_artifacts(
    source_metadata: dict,
    prepared_offshore_by_site: Dict[str, pd.DataFrame],
    local_wind_by_site: Dict[str, pd.DataFrame],
    master_static_df: pd.DataFrame | None,
    aligned_index: pd.Index,
    offshore_vars: List[str],
    direction_vars: set[str],
    auto_dynamic_direction_vars: set[str],
    input_degrees: bool,
    norm_mod,
    method: str,
    method_cfg: dict,
    feature_range: Tuple[float, float],
    split_fit_idx: np.ndarray,
    normalization_train_sites: List[str],
    max_distance_km_error: float,
    use_source_geometry_features: bool,
    include_local_direction_features: bool,
    include_local_wind_features: bool,
    scaler_override: dict | None = None,
) -> Tuple[np.ndarray, List[str], np.ndarray | None, List[str], dict]:
    """Build site-indexed multi-source dynamic tensors and geometry arrays."""
    target_sites = [str(site) for site in (source_metadata.get("target_sites", []) or [])]
    static_lookup = None
    if include_local_direction_features:
        if master_static_df is None:
            raise ValueError(
                "master_static_df is required when include_local_direction_features=True for multi-source artifacts"
            )
        static_lookup = master_static_df.set_index("site_name")
    source_feature_names: List[str] = []
    source_circular_features: List[str] = []
    tensors_by_site: List[np.ndarray] = []

    for site_idx, site_name in enumerate(target_sites):
        static_row = None
        if include_local_direction_features:
            assert static_lookup is not None
            if site_name not in static_lookup.index:
                raise ValueError(
                    f"Static feature table is missing site '{site_name}' for multi-source preprocessing"
                )
            static_row = static_lookup.loc[site_name]
        local_wind_df = local_wind_by_site.get(site_name) if include_local_wind_features else None
        if include_local_wind_features and local_wind_df is None:
            raise ValueError(
                f"Local wind mapping is missing target site '{site_name}' for multi-source preprocessing"
            )
        source_names = [str(name) for name in source_metadata["source_names"][site_idx]]
        site_tensors: List[np.ndarray] = []

        for source_name in source_names:
            offshore_df = prepared_offshore_by_site.get(source_name)
            if offshore_df is None or offshore_df.empty:
                raise ValueError(
                    f"Multi-source preprocessing missing offshore frame for source '{source_name}' at site '{site_name}'"
                )
            frame = offshore_df.reindex(aligned_index)
            prepared_frame, prepared_names, circular_cols = _prepare_multisource_feature_frame(
                offshore_df=frame,
                static_row=static_row,
                local_wind_df=local_wind_df,
                offshore_vars=offshore_vars,
                direction_vars=direction_vars,
                auto_dynamic_direction_vars=auto_dynamic_direction_vars,
                input_degrees=input_degrees,
                include_local_direction_features=include_local_direction_features,
                include_local_wind_features=include_local_wind_features,
            )
            if not source_feature_names:
                source_feature_names = list(prepared_names)
            elif list(prepared_names) != list(source_feature_names):
                raise ValueError("Inconsistent multi-source feature ordering across sources/sites")
            source_circular_features.extend(list(circular_cols))
            site_tensors.append(prepared_frame.to_numpy(dtype=np.float32, copy=True))

        tensors_by_site.append(np.stack(site_tensors, axis=1))

    x_dynamic_sources = np.stack(tensors_by_site, axis=0).astype(np.float32, copy=False)
    if x_dynamic_sources.ndim != 4:
        raise ValueError(
            f"Expected X_dynamic_sources to be 4D, got shape {x_dynamic_sources.shape}"
        )

    scale_cols = (
        norm_mod.select_dynamic_columns_for_scaling(source_feature_names)
        if hasattr(norm_mod, "select_dynamic_columns_for_scaling")
        else [
            name
            for name in source_feature_names
            if not (str(name).endswith("_sin") or str(name).endswith("_cos"))
        ]
    )
    scale_indices = [
        source_feature_names.index(name) for name in scale_cols if name in source_feature_names
    ]
    if scale_indices:
        train_site_idx = [
            target_sites.index(site_name)
            for site_name in normalization_train_sites
            if site_name in target_sites
        ]
        if not train_site_idx:
            raise ValueError("No training sites available for multi-source scaler fitting")

        # Fit over bounded time chunks.  The previous implementation created
        # a second full source tensor and flattened it before fitting, which
        # could add multiple gigabytes of peak memory during recovery.
        def _source_train_chunks():
            for site_idx in train_site_idx:
                site_values = x_dynamic_sources[int(site_idx), split_fit_idx, :, :]
                for start in range(0, len(site_values), 256):
                    yield site_values[start : start + 256, :, scale_indices].reshape(
                        -1, len(scale_indices)
                    )

        stream_stats = getattr(norm_mod, "compute_stats_stream", None)
        if scaler_override is not None:
            scaler = dict(scaler_override)
        elif callable(stream_stats):
            scaler = stream_stats(
                _source_train_chunks(), method, feature_range=tuple(feature_range)
            )
        else:
            raise RuntimeError(
                "Normalization module lacks required compute_stats_stream implementation"
            )
        x_dynamic_sources[:, :, :, scale_indices] = norm_mod.apply_normalization(
            x_dynamic_sources[:, :, :, scale_indices].reshape(-1, len(scale_indices)),
            scaler,
            method,
        ).reshape(
            x_dynamic_sources.shape[0],
            x_dynamic_sources.shape[1],
            x_dynamic_sources.shape[2],
            len(scale_indices),
        )
        scaler["columns"] = list(scale_cols)
    else:
        scaler = {"method": method, "columns": []}

    if use_source_geometry_features:
        source_geometry, source_geometry_feature_names = build_source_geometry_array(
            source_metadata=source_metadata,
            max_distance_km_error=max_distance_km_error,
        )
    else:
        source_geometry = None
        source_geometry_feature_names = []

    route_feature_names = [
        str(name)
        for name in source_feature_names
        if str(name).startswith("fetch_at_")
        or str(name).startswith("blocking_at_")
        or str(name).startswith("slope_at_")
    ]
    geometry_meta = {
        "route_feature_names": route_feature_names,
        "route_features_available": bool(route_feature_names),
        "circular_features": sorted(set(source_circular_features)),
        "source_dynamic_scaler": {
            "method": scaler.get("method", method),
            "columns": scaler.get("columns", []),
        },
        "source_dynamic_scaler_stats": scaler,
        "source_feature_names": list(source_feature_names),
        "source_geometry_feature_names": list(source_geometry_feature_names),
    }
    return (
        x_dynamic_sources,
        source_feature_names,
        source_geometry,
        source_geometry_feature_names,
        geometry_meta,
    )


def _build_static_vectors(
    static_csv: str,
    nearshore_sites: List[str],
    normalization_train_sites: List[str],
    feature_range: Tuple[float, float],
    ablation_config_path: str = DEFAULT_ABLATION_CONFIG_PATH,
    ignore_columns: List[str] | None = None,
    strict_required: bool = True,
) -> Tuple[Dict[str, np.ndarray], List[str], List[str], dict, dict, pd.DataFrame]:
    """Load and transform site-level static vectors from master_static_features.csv.

    The static ColumnTransformer is fitted strictly on training sites only.
    """
    sdf = _load_master_static_frame(static_csv=static_csv, nearshore_sites=nearshore_sites)

    requested_ignored = [str(c) for c in (ignore_columns or []) if str(c).strip()]
    applied_ignored = [c for c in requested_ignored if c in sdf.columns and c != "site_name"]
    if applied_ignored:
        sdf = sdf.drop(columns=applied_ignored)
        logging.info(
            "Dropped %d static columns from model feature set: %s",
            len(applied_ignored),
            applied_ignored,
        )

    norm_mod = _load_normalization_module()
    raw_feature_candidates = [c for c in sdf.columns if c != "site_name"]
    pre_ablation_groups = norm_mod.build_master_static_feature_groups(
        raw_feature_candidates,
        strict_required=strict_required,
    )
    pre_ablation_raw_map = norm_mod.build_raw_to_transformed_feature_map(pre_ablation_groups)

    ablation_cfg = load_ablation_config(ablation_config_path)
    ablation_summary = resolve_raw_static_ablation(raw_feature_candidates, ablation_cfg)
    matched_raw_to_drop = [
        c for c in ablation_summary.get("matched_raw_features", []) if c in sdf.columns
    ]
    if ablation_summary.get("enabled", False) and matched_raw_to_drop:
        sdf = sdf.drop(columns=matched_raw_to_drop)
        logging.info(
            "Applied static ablation to %d raw columns before static transformer fitting",
            len(matched_raw_to_drop),
        )
    for warning in ablation_summary.get("warnings", []):
        logging.warning("%s", warning)

    static_artifacts = norm_mod.fit_master_static_feature_transformer(
        master_df=sdf,
        train_sites=normalization_train_sites,
        site_column="site_name",
        feature_range=feature_range,
        strict_required=strict_required,
        intentionally_missing_columns=ablation_summary.get("matched_raw_features", []),
    )

    transformed = norm_mod.transform_master_static_features(
        master_df=sdf,
        artifacts=static_artifacts,
        site_column="site_name",
    )

    static_feature_names = [c for c in transformed.columns if c != "site_name"]
    circular_labels = [c for c in static_feature_names if c.endswith("_sin") or c.endswith("_cos")]

    vectors: Dict[str, np.ndarray] = {}
    for _, row in transformed.iterrows():
        site = str(row["site_name"])
        vectors[site] = row[static_feature_names].to_numpy(dtype=float, copy=True)

    scaler_meta = static_artifacts.to_metadata()
    scaler_meta["source_csv"] = str(static_csv)
    scaler_meta["excluded_columns_requested"] = requested_ignored
    scaler_meta["excluded_columns_applied"] = applied_ignored
    scaler_meta["ablation"] = ablation_summary
    scaler_meta["raw_to_transformed_feature_map_before_ablation"] = pre_ablation_raw_map
    return vectors, static_feature_names, circular_labels, scaler_meta, ablation_summary, sdf


def _normalize_static_vectors(
    vectors: Dict[str, np.ndarray],
    labels: List[str],
    method: str,
    feature_range: Tuple[float, float],
) -> Tuple[Dict[str, np.ndarray], dict]:
    if not vectors:
        return vectors, {"method": method, "columns": []}

    ordered_sites = list(vectors.keys())
    mat = np.vstack([vectors[s] for s in ordered_sites])

    magnitude_idx = [
        i for i, lab in enumerate(labels) if not (lab.endswith("_sin") or lab.endswith("_cos"))
    ]
    if not magnitude_idx:
        return vectors, {"method": method, "columns": []}

    scaler = _fit_scaler(mat[:, magnitude_idx], method, feature_range)
    mat[:, magnitude_idx] = _apply_scaler(mat[:, magnitude_idx], scaler)

    out: Dict[str, np.ndarray] = {}
    for i, site in enumerate(ordered_sites):
        out[site] = mat[i, :]

    scaler["columns"] = [labels[i] for i in magnitude_idx]
    return out, scaler


def _build_physical_target_frames(
    nearshore_sites: List[str],
    aligned_index: pd.Index,
    norac_dir: str,
    nearshore_vars: List[str],
    datetime_col: str,
) -> Dict[str, pd.DataFrame]:
    frames: Dict[str, pd.DataFrame] = {}
    for near in nearshore_sites:
        ndf = load_site_timeseries(near, norac_dir, datetime_col=datetime_col)
        if ndf.empty:
            logging.warning("No nearshore timeseries found for %s", near)
            frames[near] = pd.DataFrame(
                index=aligned_index,
                columns=list(PHYSICAL_TARGET_NAMES),
                dtype=float,
            )
            continue

        for raw_col in PHYSICAL_TARGET_NAMES:
            if raw_col not in ndf.columns:
                ndf[raw_col] = np.nan

        frames[near] = (
            ndf[list(PHYSICAL_TARGET_NAMES)]
            .rename(columns={"dir": "dir", "dp": "dp"})
            .reindex(aligned_index)
            .astype(float)
        )
    return frames


def _build_legacy_encoded_target_frames(
    y_physical_by_site: Dict[str, pd.DataFrame],
    input_degrees: bool,
) -> Tuple[Dict[str, pd.DataFrame], List[str], List[str]]:
    encoded_by_site: Dict[str, pd.DataFrame] = {}
    target_feature_names: List[str] = []
    circular_target_features: List[str] = []

    for site_name, frame in y_physical_by_site.items():
        encoded = encode_circular_columns(
            frame,
            ["dir", "dp"],
            input_degrees=input_degrees,
            drop_original=True,
        )
        keep_cols = ["hs", "tp", "dir_sin", "dir_cos", "dp_sin", "dp_cos"]
        for col in keep_cols:
            if col not in encoded.columns:
                encoded[col] = np.nan
        encoded = encoded[keep_cols]
        if list(encoded.columns) != _EXPECTED_TARGET_FEATURE_ORDER:
            raise ValueError(
                "Encoded target feature order must be exactly "
                f"{_EXPECTED_TARGET_FEATURE_ORDER}, got {list(encoded.columns)}"
            )
        encoded_by_site[site_name] = encoded
        if not target_feature_names:
            target_feature_names = list(keep_cols)
            circular_target_features = [
                c for c in keep_cols if c.endswith("_sin") or c.endswith("_cos")
            ]

    return encoded_by_site, target_feature_names, circular_target_features


def _require_reference_columns(
    frame: pd.DataFrame, columns: List[str], site_name: str, label: str
) -> pd.DataFrame:
    missing = [col for col in columns if col not in frame.columns]
    if missing:
        raise ValueError(
            f"Missing offshore reference columns {missing} for site '{site_name}' while building {label} transfer targets"
        )
    return frame[columns].astype(float)


def _partition_reference_from_frame(frame: pd.DataFrame, site_name: str) -> np.ndarray:
    part = _require_reference_columns(
        frame,
        ["hs_swell", "tp_swell", "thq_swell", "hs_sea", "tp_sea", "thq_sea"],
        site_name=site_name,
        label="weighted_partitioned",
    )
    swell_energy = np.square(part["hs_swell"].to_numpy(dtype=np.float64))
    sea_energy = np.square(part["hs_sea"].to_numpy(dtype=np.float64))
    total_energy = swell_energy + sea_energy
    safe_total = np.where(total_energy <= 0.0, 1.0, total_energy)
    tp = (
        (swell_energy * part["tp_swell"].to_numpy(dtype=np.float64))
        + (sea_energy * part["tp_sea"].to_numpy(dtype=np.float64))
    ) / safe_total
    direction_stack = np.stack(
        [
            part["thq_swell"].to_numpy(dtype=np.float64),
            part["thq_sea"].to_numpy(dtype=np.float64),
        ],
        axis=1,
    )
    weight_stack = np.stack([swell_energy, sea_energy], axis=1)
    direction = circular_weighted_mean_deg(
        direction_stack, np.where(weight_stack > 0.0, weight_stack, 0.0), axis=1
    )
    hs = np.sqrt(np.clip(total_energy, a_min=0.0, a_max=None))
    return np.column_stack([hs, tp, direction, direction]).astype(np.float64, copy=False)


def _reference_matrix_for_source_frame(
    frame: pd.DataFrame,
    source_name: str,
    transfer_reference: str,
) -> np.ndarray:
    ref = str(transfer_reference).strip().lower()
    if ref in {"nearest_bulk", "weighted_bulk"}:
        cols = _require_reference_columns(
            frame, ["hs", "tp", "Pdir", "thq"], site_name=source_name, label=ref
        )
        return cols.to_numpy(dtype=np.float64, copy=True)
    if ref in {"nearest_swell", "weighted_swell"}:
        cols = _require_reference_columns(
            frame, ["hs_swell", "tp_swell", "thq_swell"], site_name=source_name, label=ref
        )
        arr = cols.to_numpy(dtype=np.float64, copy=True)
        return np.column_stack([arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 2]]).astype(
            np.float64, copy=False
        )
    if ref == "weighted_partitioned":
        return _partition_reference_from_frame(frame, site_name=source_name)
    raise ValueError(f"Unsupported transfer_reference '{transfer_reference}'")


def _build_transfer_references_by_site(
    target_sites: List[str],
    aligned_index: pd.Index,
    prepared_offshore_by_site: Dict[str, pd.DataFrame],
    source_metadata: dict | None,
    site_to_wave_site: Dict[str, str],
    transfer_reference: str,
) -> Tuple[Dict[str, pd.DataFrame], str]:
    reference_mode = str(transfer_reference).strip().lower()
    refs: Dict[str, pd.DataFrame] = {}

    if reference_mode in {"weighted_bulk", "weighted_swell", "weighted_partitioned"}:
        if source_metadata is None:
            raise ValueError(
                "targets.transfer_reference requires multi-source preprocessing artifacts, "
                f"but source metadata is missing for '{reference_mode}'"
            )

        meta_sites = [str(site) for site in (source_metadata.get("target_sites", []) or [])]
        site_to_meta_idx = {site: idx for idx, site in enumerate(meta_sites)}
        for site_name in target_sites:
            if site_name not in site_to_meta_idx:
                raise ValueError(
                    f"Multi-source metadata is missing target site '{site_name}' for transfer references"
                )
            meta_idx = site_to_meta_idx[site_name]
            source_names = [str(name) for name in source_metadata["source_names"][meta_idx]]
            source_weights = np.asarray(source_metadata["weights"][meta_idx], dtype=np.float64)
            if source_weights.ndim != 1 or source_weights.size != len(source_names):
                raise ValueError(
                    f"Invalid source weights for site '{site_name}': expected {len(source_names)} entries, "
                    f"got shape {source_weights.shape}"
                )
            weight_sum = float(np.sum(source_weights))
            if not np.isfinite(weight_sum) or abs(weight_sum - 1.0) > 1e-5:
                raise ValueError(
                    f"Source weights must exist and sum to 1 for site '{site_name}', got sum={weight_sum}"
                )

            matrices = []
            for source_name in source_names:
                frame = prepared_offshore_by_site.get(source_name)
                if frame is None or frame.empty:
                    raise ValueError(
                        f"Missing offshore frame for weighted transfer source '{source_name}' at site '{site_name}'"
                    )
                matrices.append(
                    _reference_matrix_for_source_frame(
                        frame.reindex(aligned_index),
                        source_name=source_name,
                        transfer_reference=reference_mode,
                    )
                )
            stacked = np.stack(matrices, axis=1)
            scalar_weights = np.broadcast_to(source_weights.reshape(1, -1), stacked[:, :, 0].shape)
            ref_hs = np.sum(stacked[:, :, 0] * scalar_weights, axis=1)
            ref_tp = np.sum(stacked[:, :, 1] * scalar_weights, axis=1)
            ref_dir = circular_weighted_mean_deg(stacked[:, :, 2], scalar_weights, axis=1)
            ref_dp = circular_weighted_mean_deg(stacked[:, :, 3], scalar_weights, axis=1)
            refs[site_name] = pd.DataFrame(
                np.column_stack([ref_hs, ref_tp, ref_dir, ref_dp]),
                index=aligned_index,
                columns=list(REFERENCE_TARGET_NAMES),
            )
        return refs, "multi_source_weighted"

    for site_name in target_sites:
        wave_site = site_to_wave_site.get(site_name)
        if not wave_site:
            raise ValueError(
                f"Single-source transfer reference requires site_to_wave_site mapping for site '{site_name}'"
            )
        frame = prepared_offshore_by_site.get(wave_site)
        if frame is None or frame.empty:
            raise ValueError(
                f"Missing offshore frame for mapped wave site '{wave_site}' while building transfer references for '{site_name}'"
            )
        ref_matrix = _reference_matrix_for_source_frame(
            frame.reindex(aligned_index),
            source_name=wave_site,
            transfer_reference=reference_mode,
        )
        refs[site_name] = pd.DataFrame(
            ref_matrix,
            index=aligned_index,
            columns=list(REFERENCE_TARGET_NAMES),
        )
    return refs, "legacy_nearest"


def build_point_centric_dataset(
    sites_yaml: str = "configs/sites.yaml",
    training_config: str = "configs/training.yaml",
    preprocess_config: str = "configs/preprocess.yaml",
    out_dir: str | None = None,
) -> Dict[str, str]:
    """Create one complete, aligned prepared-data directory.

    Reads site, training, and preprocess settings; discovers raw parameter
    files; fits training-only transforms; and writes every array and metadata
    sidecar required by ``train`` and ``evaluate``.  Returns the paths of the
    generated artifacts for programmatic callers.
    """
    train_cfg = normalize_runtime_config_paths(
        read_yaml(training_config), config_path=training_config
    )
    _raise_if_legacy_site_holdout_keys_present(train_cfg)
    prep_cfg = read_yaml(preprocess_config)

    data_cfg = train_cfg.get("data", {})
    static_ablation_config_path = resolve_static_ablation_config_path(train_cfg)
    multi_source_cfg = resolve_multi_source_config(data_cfg)
    targets_cfg = resolve_targets_config(data_cfg)
    use_static_features = bool(data_cfg.get("use_static_features", True))
    use_bathymetry = bool(data_cfg.get("use_bathymetry", False))
    use_offshore_wind_features, use_local_wind_features = _resolve_wind_feature_toggles(data_cfg)
    use_source_geometry_features = _resolve_runtime_source_geometry_features_flag(train_cfg)
    resolved_sites_yaml = str(sites_yaml)
    if str(sites_yaml) == "configs/sites.yaml" and data_cfg.get("sites_config"):
        resolved_sites_yaml = str(data_cfg.get("sites_config"))
    sites_cfg = read_yaml(resolved_sites_yaml)

    backend = sites_cfg.get("backend", {})
    nora3_dir = backend.get("nora3_params_dir", "data/raw/nora3/params")
    norac_dir = backend.get("norac_params_dir", "data/raw/norac/params")

    offshore_vars = list(data_cfg.get("offshore_vars") or [])
    nearshore_vars = list(data_cfg.get("nearshore_vars") or [])
    direction_vars = set(data_cfg.get("direction_vars") or [])
    if not use_offshore_wind_features:
        offshore_vars = [name for name in offshore_vars if not str(name).startswith("wind_")]
        direction_vars = {name for name in direction_vars if str(name) != "wind_direction_10m"}

    if not offshore_vars:
        raise ValueError(
            "training config is missing data.offshore_vars or it is empty. "
            "Check configs/training.yaml and YAML indentation under the data: section."
        )
    if not nearshore_vars:
        raise ValueError(
            "training config is missing data.nearshore_vars or it is empty. "
            "Check configs/training.yaml and YAML indentation under the data: section."
        )

    if "dp" in nearshore_vars:
        direction_vars.add("dp")

    split_cfg = train_cfg.get("split", {})
    datetime_col = split_cfg.get("datetime_column", "time")
    train_frac = float(split_cfg.get("train", 0.7))
    val_frac = float(split_cfg.get("val", 0.15))

    norm_cfg = train_cfg.get("normalization", {})
    method = norm_cfg.get("default_method", "minmax")
    method_cfg = (norm_cfg.get("methods") or {}).get(method, {})
    feature_range = tuple(method_cfg.get("feature_range", (0.0, 1.0)))

    circ_cfg = prep_cfg.get("circular", {})
    input_degrees = bool(circ_cfg.get("input_degrees", True))
    norm_mod = _load_normalization_module()

    base_out = (
        out_dir
        or prep_cfg.get("paths", {}).get("processed_dir")
        or split_cfg.get("out_dir")
        or "data/processed"
    )
    outp = Path(base_out)
    outp.mkdir(parents=True, exist_ok=True)

    nearshore_sites = [
        str(s.get("name")) for s in (sites_cfg.get("nearshore_sites", []) or []) if s.get("name")
    ]
    nearshore_entries = [s for s in (sites_cfg.get("nearshore_sites", []) or []) if s.get("name")]
    canonical_target_sites = list(nearshore_sites)
    split_info = resolve_site_split_config(nearshore_sites, train_cfg)
    warn_if_legacy_validation_mode(split_info)
    normalization_train_sites = list(split_info.get("train_sites", []) or [])
    logging.info(
        "Resolved site splits | mode=%s | train=%d val=%d test=%d",
        split_info.get("validation_mode", "legacy-temporal"),
        len(split_info.get("train_sites", []) or []),
        len(split_info.get("val_sites", []) or []),
        len(split_info.get("test_sites", []) or []),
    )
    logging.info("Train sites: %s", split_info.get("train_sites", []))
    logging.info("Validation sites: %s", split_info.get("val_sites", []))
    logging.info("Test sites: %s", split_info.get("test_sites", []))
    train_site_subsampling_summary = split_info.get("train_site_subsampling", {}) or {}
    if bool(train_site_subsampling_summary.get("enabled", False)):
        logging.info(
            "Train site subsampling | enabled=%s applied=%s fraction=%.6f seed=%d before=%d after=%d removed=%d",
            bool(train_site_subsampling_summary.get("enabled", False)),
            bool(train_site_subsampling_summary.get("applied", False)),
            float(train_site_subsampling_summary.get("fraction", 1.0)),
            int(train_site_subsampling_summary.get("seed", 42)),
            int(
                train_site_subsampling_summary.get(
                    "before_count", len(split_info.get("train_sites", []) or [])
                )
            ),
            int(
                train_site_subsampling_summary.get(
                    "after_count", len(split_info.get("train_sites", []) or [])
                )
            ),
            int(train_site_subsampling_summary.get("removed_count", 0)),
        )

    offshore_sites_all = [
        str(s.get("name")) for s in (sites_cfg.get("offshore_sites", []) or []) if s.get("name")
    ]
    offshore_entries = [s for s in (sites_cfg.get("offshore_sites", []) or []) if s.get("name")]
    if use_local_wind_features:
        local_wind_metadata_by_site, local_wind_diagnostics = _build_nearest_local_wind_metadata(
            nearshore_entries=nearshore_entries,
            offshore_entries=offshore_entries,
        )
    else:
        local_wind_metadata_by_site = {}
        local_wind_diagnostics = {
            "enabled": False,
            "num_norac_points": int(len(nearshore_entries)),
            "num_candidate_local_wind_points": 0,
            "max_nearest_local_wind_distance_m": float("nan"),
            "median_nearest_local_wind_distance_m": float("nan"),
            "example_local_wind_assignments": [],
            "candidate_local_wind_points": [],
            "assignments": [],
        }

    if not offshore_sites_all:
        raise ValueError("No offshore sites discovered from sites.yaml")

    # Wind-grid points are used upstream to build/interpolate wind fields and
    # should not be direct model input points in the point-centric tensors.
    offshore_sites = [s for s in offshore_sites_all if "wind" not in str(s).lower()]
    excluded_offshore_wind_sites = [s for s in offshore_sites_all if s not in offshore_sites]
    if excluded_offshore_wind_sites:
        logging.info(
            "Excluding %d offshore wind sites from model inputs: %s",
            len(excluded_offshore_wind_sites),
            excluded_offshore_wind_sites,
        )

    if not offshore_sites:
        raise ValueError(
            "No non-wind offshore sites remain after excluding wind-grid points. "
            "Check paired_offshore entries in sites.yaml."
        )
    logging.info(
        "Preprocess startup | target_sites=%d candidate_wave_sources=%d multi_source_enabled=%s k_nearest=%d target_mode=%s transfer_reference=%s",
        len(nearshore_sites),
        len(offshore_sites),
        bool(multi_source_cfg.get("enabled", False)),
        int(multi_source_cfg.get("k_nearest", 3)),
        str(targets_cfg.get("mode", "physical")),
        str(targets_cfg.get("transfer_reference", "nearest_bulk")),
    )
    logging.info(
        "Artifact switches | static=%s bathymetry=%s source_geometry=%s offshore_wind=%s local_wind=%s strict_dynamic_only=%s",
        use_static_features,
        use_bathymetry,
        use_source_geometry_features,
        use_offshore_wind_features,
        use_local_wind_features,
        (not use_static_features and not use_bathymetry and not use_source_geometry_features),
    )
    if use_local_wind_features:
        logging.info(
            "Nearest local wind mapping | norac_points=%d candidate_local_wind_points=%d max_distance_m=%.3f median_distance_m=%.3f",
            int(local_wind_diagnostics.get("num_norac_points", 0)),
            int(local_wind_diagnostics.get("num_candidate_local_wind_points", 0)),
            float(local_wind_diagnostics.get("max_nearest_local_wind_distance_m", float("nan"))),
            float(local_wind_diagnostics.get("median_nearest_local_wind_distance_m", float("nan"))),
        )
        for example in list(local_wind_diagnostics.get("example_local_wind_assignments", []) or [])[
            :5
        ]:
            logging.info(
                "Local wind assignment example | %s -> %s | distance_m=%.3f",
                str(example.get("norac_point_name", "")),
                str(example.get("local_wind_point_name", "")),
                float(example.get("distance_m", float("nan"))),
            )
    else:
        logging.info("Local wind features disabled via data.use_local_wind_features=false")

    if use_offshore_wind_features:
        wave_to_wind_site = _build_wave_to_wind_site_map(
            sites_cfg=sites_cfg, wave_sites=offshore_sites
        )
        if wave_to_wind_site:
            unique_wind_sources = sorted(set(wave_to_wind_site.values()))
            logging.info(
                "Using %d wind-grid source sites to populate wind variables for %d wave-grid sites",
                len(unique_wind_sources),
                len(offshore_sites),
            )
        else:
            logging.warning(
                "No wind-grid sites found in sites configuration. "
                "Wind variables in offshore inputs may become NaN."
            )
    else:
        wave_to_wind_site = {}
        logging.info("Offshore wind features disabled via data.use_offshore_wind_features=false")

    # Step 1: Build X_dynamic with strict temporal alignment across all offshore points.
    offshore_frames: List[pd.DataFrame] = []
    offshore_feature_names: List[str] = []
    circular_dynamic_features: List[str] = []
    wind_like_vars = [v for v in offshore_vars if str(v).startswith("wind_")]
    auto_dynamic_direction_vars = set(norm_mod.infer_dynamic_direction_columns(offshore_vars))
    wind_ts_cache: Dict[str, pd.DataFrame] = {}
    prepared_offshore_by_site: Dict[str, pd.DataFrame] = {}
    local_wind_by_site: Dict[str, pd.DataFrame] = {}

    for site in offshore_sites:
        site_df = load_site_timeseries(site, nora3_dir, datetime_col=datetime_col)
        if site_df.empty:
            logging.warning("No offshore timeseries found for %s", site)
            continue

        mapped_wind_site = wave_to_wind_site.get(site)
        if mapped_wind_site and wind_like_vars:
            if mapped_wind_site not in wind_ts_cache:
                wind_ts_cache[mapped_wind_site] = load_site_timeseries(
                    mapped_wind_site,
                    nora3_dir,
                    datetime_col=datetime_col,
                )

            wind_df = wind_ts_cache[mapped_wind_site]
            if wind_df.empty:
                logging.warning(
                    "No wind timeseries found for mapped wind site %s (wave site %s)",
                    mapped_wind_site,
                    site,
                )
            else:
                for wind_var in wind_like_vars:
                    if wind_var not in wind_df.columns:
                        continue
                    if wind_var in site_df.columns and not site_df[wind_var].isna().all():
                        continue
                    site_df[wind_var] = pd.to_numeric(wind_df[wind_var], errors="coerce").reindex(
                        site_df.index
                    )

        for v in offshore_vars:
            if v not in site_df.columns:
                site_df[v] = np.nan

        site_df = site_df[offshore_vars]
        site_df = apply_nora3_wave_direction_offset(site_df, offshore_vars)
        prepared_offshore_by_site[site] = site_df.copy()
        site_circular = [
            v for v in offshore_vars if (v in direction_vars or v in auto_dynamic_direction_vars)
        ]
        site_df = encode_circular_columns(
            site_df, site_circular, input_degrees=input_degrees, drop_original=True
        )

        keep_cols: List[str] = []
        for v in offshore_vars:
            if v in direction_vars:
                sin_col = f"{v}_sin"
                cos_col = f"{v}_cos"
                if sin_col not in site_df.columns:
                    site_df[sin_col] = np.nan
                if cos_col not in site_df.columns:
                    site_df[cos_col] = np.nan
                keep_cols.extend([sin_col, cos_col])
            else:
                if v not in site_df.columns:
                    site_df[v] = np.nan
                keep_cols.append(v)

        site_df = site_df[keep_cols]
        rename_map = {c: f"off_{site}_{c}" for c in keep_cols}
        site_df = site_df.rename(columns=rename_map)

        offshore_frames.append(site_df)
        offshore_feature_names.extend(list(site_df.columns))
        circular_dynamic_features.extend(
            [c for c in site_df.columns if c.endswith("_sin") or c.endswith("_cos")]
        )

    if not offshore_frames:
        raise ValueError("No offshore frames could be assembled")

    # Strict alignment: intersection of all offshore datetime indices.
    aligned_index = offshore_frames[0].index
    for f in offshore_frames[1:]:
        aligned_index = aligned_index.intersection(f.index)
    aligned_index = aligned_index.sort_values()

    if len(aligned_index) == 0:
        raise ValueError("No shared timestamps found across offshore points")

    aligned_index, date_range_meta = _resolve_and_apply_date_range(aligned_index, data_cfg)
    logging.info(
        "Date-range filter | enabled=%s requested=[%s, %s] resolved=[%s, %s] kept=%d dropped=%d",
        bool(date_range_meta.get("enabled", False)),
        date_range_meta.get("requested_start"),
        date_range_meta.get("requested_end"),
        date_range_meta.get("resolved_start"),
        date_range_meta.get("resolved_end"),
        int(date_range_meta.get("timestamp_count_after_filter", len(aligned_index))),
        int(date_range_meta.get("timestamp_count_dropped", 0)),
    )

    aligned_offshore = [f.reindex(aligned_index) for f in offshore_frames]
    x_dynamic_df = pd.concat(aligned_offshore, axis=1)
    if use_local_wind_features:
        for site_name, wind_meta in local_wind_metadata_by_site.items():
            local_wind_site_name = str(wind_meta.get("local_wind_point_name", "")).strip()
            if not local_wind_site_name:
                raise ValueError(
                    f"Local wind mapping is missing selected wind point for nearshore site '{site_name}'"
                )
            if local_wind_site_name not in wind_ts_cache:
                wind_ts_cache[local_wind_site_name] = load_site_timeseries(
                    local_wind_site_name,
                    nora3_dir,
                    datetime_col=datetime_col,
                )
            local_wind_df = wind_ts_cache[local_wind_site_name]
            if local_wind_df.empty:
                logging.warning(
                    "No local wind timeseries found for selected wind site %s (nearshore site %s)",
                    local_wind_site_name,
                    site_name,
                )
            local_wind_df = local_wind_df.copy()
            for wind_var in ("wind_speed_10m", "wind_direction_10m"):
                if wind_var not in local_wind_df.columns:
                    local_wind_df[wind_var] = np.nan
            local_wind_by_site[site_name] = local_wind_df[
                ["wind_speed_10m", "wind_direction_10m"]
            ].reindex(aligned_index)

    split_idx = _build_split_indices(len(aligned_index), train_frac, val_frac)
    temporal_holdout_active = bool(split_info.get("site_holdout_temporal_active", False))
    temporal_holdout_recent_fraction = split_info.get("site_holdout_temporal_recent_fraction", None)
    if temporal_holdout_active:
        recent_fraction = float(temporal_holdout_recent_fraction)
        recent_count = int(len(aligned_index) * recent_fraction)
        train_count = int(len(aligned_index) - recent_count)
        if recent_count < 1:
            raise ValueError(
                "split.site_holdout_temporal_mode='shared_recent' produced an empty recent holdout window. "
                f"Increase split.val/split.test or use more timestamps. n_timestamps={len(aligned_index)} "
                f"recent_fraction={recent_fraction:.12g}"
            )
        if train_count < 1:
            raise ValueError(
                "split.site_holdout_temporal_mode='shared_recent' produced an empty training window. "
                f"Reduce split.val/split.test. n_timestamps={len(aligned_index)} "
                f"recent_fraction={recent_fraction:.12g}"
            )
        recent_idx = np.arange(train_count, len(aligned_index), dtype=int)
        split_idx = {
            "train": np.arange(0, train_count, dtype=int),
            "val": recent_idx.copy(),
            "test": recent_idx.copy(),
        }
    split_fit_idx = (
        split_idx["train"]
        if temporal_holdout_active or not bool(split_info.get("validation_site_heldout", False))
        else np.arange(len(aligned_index), dtype=int)
    )
    # Defer magnitude normalization until we've collected target and static values
    # so that scalers can be computed from the training set across inputs+targets+statics.
    x_scaler = None

    # Step 2: Build physical targets and the legacy encoded target path.
    y_physical_df = _build_physical_target_frames(
        nearshore_sites=nearshore_sites,
        aligned_index=aligned_index,
        norac_dir=norac_dir,
        nearshore_vars=nearshore_vars,
        datetime_col=datetime_col,
    )
    y_targets_df, target_feature_names, circular_target_features = (
        _build_legacy_encoded_target_frames(
            y_physical_by_site=y_physical_df,
            input_degrees=input_degrees,
        )
    )

    if not target_feature_names:
        raise ValueError(
            "Failed to resolve target feature names from nearshore target preprocessing. "
            "Check data.nearshore_vars and confirm NORAC target files contain those columns."
        )

    # Step 3: Build X_static vectors from master_static_features.csv.
    static_csv = str(data_cfg.get("static_features_csv") or _resolve_static_csv(prep_cfg) or "")
    static_vectors: Dict[str, np.ndarray] = {}
    static_feature_names: List[str] = []
    circular_static_features: List[str] = []
    static_scaler: dict = {"method": method, "columns": []}
    static_ablation_summary: dict = {
        "enabled": False,
        "matched_raw_features": [],
        "matched_transformed_features": [],
        "warnings": [],
        "static_feature_count_before": 0,
        "static_feature_count_after": 0,
    }
    master_static_df: pd.DataFrame | None = None
    if use_static_features:
        static_excluded_columns = _resolve_static_excluded_columns(data_cfg)
        (
            static_vectors,
            static_feature_names,
            circular_static_features,
            static_scaler,
            static_ablation_summary,
            master_static_df,
        ) = _build_static_vectors(
            static_csv=static_csv,
            nearshore_sites=nearshore_sites,
            normalization_train_sites=normalization_train_sites,
            feature_range=feature_range,
            ablation_config_path=static_ablation_config_path,
            ignore_columns=static_excluded_columns,
            strict_required=True,
        )
    else:
        logging.info("Skipping static vector artifacts because data.use_static_features=false")

    # Step 3b: Build site-specific dynamic geometry interaction sidecar.
    site_to_wave_site = _resolve_site_dynamic_wave_source_map(
        sites_cfg=sites_cfg,
        available_wave_sites=sorted(prepared_offshore_by_site.keys()),
    )
    site_dynamic_frames: Dict[str, pd.DataFrame] = {}
    site_dynamic_feature_names: List[str] = []
    if use_static_features:
        assert master_static_df is not None
        site_dynamic_frames, site_dynamic_feature_names, site_dynamic_warnings = (
            _build_site_dynamic_sidecar(
                master_static_df=master_static_df,
                prepared_offshore_by_site=prepared_offshore_by_site,
                local_wind_by_site=local_wind_by_site,
                aligned_index=aligned_index,
                site_to_wave_site=site_to_wave_site,
                input_degrees=input_degrees,
                include_local_wind_features=use_local_wind_features,
            )
        )
        for warning in site_dynamic_warnings:
            logging.warning("%s", warning)
    elif use_local_wind_features:
        site_dynamic_frames, site_dynamic_feature_names, site_dynamic_warnings = (
            _build_site_local_wind_sequence_sidecar(
                local_wind_by_site=local_wind_by_site,
                aligned_index=aligned_index,
                input_degrees=input_degrees,
            )
        )
        for warning in site_dynamic_warnings:
            logging.warning("%s", warning)
        logging.info(
            "Built local-wind-only site-specific dynamic sidecar because static geometry is disabled"
        )
    else:
        logging.info(
            "Skipping site-specific dynamic sidecar because local wind features are disabled"
        )

    # Step 4: Dynamic magnitude normalization (fit on train timesteps except for
    # legacy site-heldout validation, which preserves the historical full-timeline fit).
    select_scale_cols = getattr(norm_mod, "select_dynamic_columns_for_scaling", None)
    if callable(select_scale_cols):
        dynamic_magnitude_cols = select_scale_cols(x_dynamic_df.columns)
    else:
        dynamic_magnitude_cols = [
            c for c in x_dynamic_df.columns if not (c.endswith("_sin") or c.endswith("_cos"))
        ]
    if dynamic_magnitude_cols:
        x_scaler = norm_mod.compute_stats(
            x_dynamic_df.iloc[split_fit_idx][dynamic_magnitude_cols],
            method,
            feature_range=tuple(feature_range),
            quantile_low=(method_cfg.get("quantile_range") or [25, 75])[0],
            quantile_high=(method_cfg.get("quantile_range") or [25, 75])[1],
        )
        x_dynamic_df.loc[:, dynamic_magnitude_cols] = norm_mod.apply_normalization(
            x_dynamic_df[dynamic_magnitude_cols],
            x_scaler,
            method,
        )
        x_scaler["columns"] = list(dynamic_magnitude_cols)
    else:
        x_scaler = {"method": method, "columns": []}

    if site_dynamic_frames:
        if callable(select_scale_cols):
            site_dynamic_cols_to_scale = list(select_scale_cols(site_dynamic_feature_names))
        else:
            site_dynamic_cols_to_scale = [
                name
                for name in site_dynamic_feature_names
                if not (str(name).endswith("_sin") or str(name).endswith("_cos"))
            ]
        training_site_set = set(normalization_train_sites)
        train_site_dynamic_rows = [
            site_dynamic_frames[site]
            .iloc[split_fit_idx][site_dynamic_cols_to_scale]
            .to_numpy(dtype=float)
            for site in site_dynamic_frames
            if site in training_site_set
        ]
        if site_dynamic_cols_to_scale and train_site_dynamic_rows:
            stacked_train_rows = np.vstack(train_site_dynamic_rows)
            site_dynamic_scaler = norm_mod.compute_stats(
                stacked_train_rows,
                method,
                feature_range=tuple(feature_range),
                quantile_low=(method_cfg.get("quantile_range") or [25, 75])[0],
                quantile_high=(method_cfg.get("quantile_range") or [25, 75])[1],
            )
            for site_name, frame in site_dynamic_frames.items():
                frame.loc[:, site_dynamic_cols_to_scale] = norm_mod.apply_normalization(
                    frame[site_dynamic_cols_to_scale],
                    site_dynamic_scaler,
                    method,
                )
            site_dynamic_scaler["columns"] = list(site_dynamic_cols_to_scale)
        else:
            site_dynamic_scaler = {"method": method, "columns": []}
    else:
        site_dynamic_scaler = {"method": method, "columns": []}

    source_metadata = None
    source_metadata_path = None
    source_dynamic_npz = None
    source_geometry_npz = None
    x_dynamic_sources = None
    source_feature_names: List[str] = []
    source_geometry = None
    source_geometry_feature_names: List[str] = []
    multi_source_meta: dict = {
        "enabled": bool(multi_source_cfg.get("enabled", False)),
        "route_features_available": False,
        "route_feature_names": [],
    }
    if bool(multi_source_cfg.get("enabled", False)):
        valid_offshore_entries = [
            entry
            for entry in offshore_entries
            if str(entry.get("name", "")) in prepared_offshore_by_site
        ]
        logging.info(
            "Valid multi-source candidates with assembled timeseries: %d",
            len(valid_offshore_entries),
        )
        source_metadata, source_mapping_warnings = build_k_nearest_source_metadata(
            nearshore_entries=nearshore_entries,
            offshore_entries=valid_offshore_entries,
            k_nearest=int(multi_source_cfg.get("k_nearest", 3)),
            source_type=str(multi_source_cfg.get("source_type", "nora3_wave")),
            weight_power=float(multi_source_cfg.get("weight_power", 1.0)),
            max_distance_km_warn=float(multi_source_cfg.get("max_distance_km_warn", 80.0)),
            max_distance_km_error=float(multi_source_cfg.get("max_distance_km_error", 150.0)),
            allow_padding=bool(multi_source_cfg.get("allow_padding", False)),
        )
        if list(source_metadata.get("target_sites", []) or []) != canonical_target_sites:
            raise AssertionError(
                "Multi-source metadata target_sites must preserve canonical nearshore site order: "
                f"expected={canonical_target_sites} got={list(source_metadata.get('target_sites', []) or [])}"
            )
        for warning in source_mapping_warnings:
            logging.warning("%s", warning)
        distance_summary = summarize_nearest_distances_km(source_metadata)
        logging.info(
            "Nearest-source distance summary (km) | min=%.3f median=%.3f max=%.3f",
            float(distance_summary.get("min_km", float("nan"))),
            float(distance_summary.get("median_km", float("nan"))),
            float(distance_summary.get("max_km", float("nan"))),
        )
        (
            x_dynamic_sources,
            source_feature_names,
            source_geometry,
            source_geometry_feature_names,
            multi_source_build_meta,
        ) = _build_multisource_dynamic_artifacts(
            source_metadata=source_metadata,
            prepared_offshore_by_site=prepared_offshore_by_site,
            local_wind_by_site=local_wind_by_site,
            master_static_df=master_static_df,
            aligned_index=aligned_index,
            offshore_vars=offshore_vars,
            direction_vars=direction_vars,
            auto_dynamic_direction_vars=auto_dynamic_direction_vars,
            input_degrees=input_degrees,
            norm_mod=norm_mod,
            method=method,
            method_cfg=method_cfg,
            feature_range=feature_range,
            split_fit_idx=split_fit_idx,
            normalization_train_sites=normalization_train_sites,
            max_distance_km_error=float(multi_source_cfg.get("max_distance_km_error", 150.0)),
            use_source_geometry_features=use_source_geometry_features,
            include_local_direction_features=use_static_features,
            include_local_wind_features=use_local_wind_features,
        )
        multi_source_meta.update(multi_source_build_meta)

    # Step 5: Target scaling with StandardScaler fit only on train split/sites.
    target_scaler = norm_mod.fit_target_standard_scaler(
        y_targets_by_site=y_targets_df,
        train_idx=split_fit_idx,
        train_sites=normalization_train_sites,
        feature_names=target_feature_names,
    )
    y_targets_df = norm_mod.transform_target_frames_by_site(
        y_targets_by_site=y_targets_df,
        scaler_meta=target_scaler,
        columns=target_feature_names,
    )

    physical_target_names = validate_target_name_block(
        PHYSICAL_TARGET_NAMES, "physical_target_names"
    )
    transfer_target_names: List[str] = []
    reference_target_names: List[str] = []
    y_reference_df: Dict[str, pd.DataFrame] = {}
    y_transfer_df: Dict[str, pd.DataFrame] = {}
    transfer_scaler: dict | None = None
    transfer_reference_source = ""

    if str(targets_cfg.get("mode", "physical")) != "physical":
        transfer_reference = str(targets_cfg.get("transfer_reference", "nearest_bulk"))
        if not bool(multi_source_cfg.get("enabled", False)) and transfer_reference not in {
            "nearest_bulk",
            "nearest_swell",
        }:
            raise ValueError(
                "Weighted transfer references require multi-source preprocessing artifacts. "
                f"Received targets.transfer_reference='{transfer_reference}' with data.multi_source.enabled=false."
            )

        y_reference_df, transfer_reference_source = _build_transfer_references_by_site(
            target_sites=canonical_target_sites,
            aligned_index=aligned_index,
            prepared_offshore_by_site=prepared_offshore_by_site,
            source_metadata=source_metadata,
            site_to_wave_site=site_to_wave_site,
            transfer_reference=transfer_reference,
        )
        reference_target_names = validate_target_name_block(
            REFERENCE_TARGET_NAMES, "reference_target_names"
        )

        for site_name in canonical_target_sites:
            physical_matrix = y_physical_df[site_name][physical_target_names].to_numpy(
                dtype=np.float64, copy=True
            )
            reference_matrix = y_reference_df[site_name][reference_target_names].to_numpy(
                dtype=np.float64, copy=True
            )
            transfer_matrix = build_transfer_targets(
                physical_matrix,
                reference_matrix,
                eps=float(targets_cfg.get("eps", 1e-3)),
            )
            y_transfer_df[site_name] = pd.DataFrame(
                transfer_matrix,
                index=aligned_index,
                columns=list(TRANSFER_TARGET_NAMES),
            )

        transfer_target_names = validate_target_name_block(
            TRANSFER_TARGET_NAMES, "transfer_target_names"
        )
        transfer_scaler = norm_mod.fit_target_standard_scaler(
            y_targets_by_site=y_transfer_df,
            train_idx=split_fit_idx,
            train_sites=normalization_train_sites,
            feature_names=list(TRANSFER_SCALER_COLUMNS),
        )
        transfer_scaler["feature_names"] = list(TRANSFER_SCALER_COLUMNS)
        transfer_scaler["unscaled_columns"] = [
            name for name in transfer_target_names if name not in TRANSFER_SCALER_COLUMNS
        ]

    # Convert to final numpy outputs.
    x_dynamic = x_dynamic_df.values.astype(np.float32)
    y_targets = {s: y_targets_df[s].values.astype(np.float32) for s in canonical_target_sites}
    y_physical = {
        s: y_physical_df[s][physical_target_names].to_numpy(dtype=np.float32, copy=True)
        for s in canonical_target_sites
    }
    y_reference = (
        {
            s: y_reference_df[s][reference_target_names].to_numpy(dtype=np.float32, copy=True)
            for s in canonical_target_sites
        }
        if y_reference_df
        else {}
    )
    y_transfer = (
        {
            s: y_transfer_df[s][transfer_target_names].to_numpy(dtype=np.float32, copy=True)
            for s in canonical_target_sites
        }
        if y_transfer_df
        else {}
    )
    x_static = {
        s: static_vectors[s].astype(np.float32)
        for s in canonical_target_sites
        if s in static_vectors
    }

    if use_local_wind_features:
        if set(local_wind_metadata_by_site.keys()) != set(canonical_target_sites):
            missing = sorted(set(canonical_target_sites) - set(local_wind_metadata_by_site.keys()))
            extra = sorted(set(local_wind_metadata_by_site.keys()) - set(canonical_target_sites))
            raise AssertionError(
                f"Local wind metadata/target site mismatch: missing={missing} extra={extra}"
            )
        local_wind_distances = [
            float(record.get("distance_m", float("nan")))
            for record in local_wind_metadata_by_site.values()
        ]
        if any((not np.isfinite(distance) or distance < 0.0) for distance in local_wind_distances):
            raise AssertionError(
                "Nearest local wind distances must be finite and non-negative for every NORAC point"
            )

    n_time = int(x_dynamic.shape[0])
    for site_name, arr in y_targets.items():
        if int(arr.shape[0]) != n_time:
            raise AssertionError(
                f"Y_targets[{site_name}] timestamp length mismatch: {arr.shape[0]} vs {n_time}"
            )
    for site_name, frame in site_dynamic_frames.items():
        if int(frame.shape[0]) != n_time:
            raise AssertionError(
                f"X_dynamic_sitewise[{site_name}] timestamp length mismatch: {frame.shape[0]} vs {n_time}"
            )
    if x_dynamic_sources is not None and int(x_dynamic_sources.shape[1]) != n_time:
        raise AssertionError(
            f"X_dynamic_sources timestamp length mismatch: {x_dynamic_sources.shape[1]} vs {n_time}"
        )
    if len(list(x_dynamic_df.columns)) != int(x_dynamic.shape[1]):
        raise AssertionError(
            "dynamic_feature_names count must equal X_dynamic feature dimension: "
            f"{len(list(x_dynamic_df.columns))} vs {x_dynamic.shape[1]}"
        )
    if x_dynamic_sources is not None and len(source_feature_names) != int(
        x_dynamic_sources.shape[-1]
    ):
        raise AssertionError(
            "source_feature_names count must equal X_dynamic_sources feature dimension: "
            f"{len(source_feature_names)} vs {x_dynamic_sources.shape[-1]}"
        )
    if use_local_wind_features and {
        "fetch_at_wind_direction_m",
        "blocking_at_wind_direction",
        "slope_at_wind_direction",
    }.intersection(set(source_feature_names)):
        raise AssertionError(
            "Engineered local wind source features must not use legacy ambiguous wind-direction names"
        )
    if site_dynamic_frames:
        legacy_sitewise_wind_names = {
            "wind_fetch_aligned_m",
            "wind_fetch_aligned_ratio",
            "windsea_proxy_u2_fetch",
            "windsea_proxy_u2_fetch_ratio",
            "windsea_proxy_3h_mean",
            "windsea_proxy_6h_mean",
            "windsea_proxy_12h_mean",
        }
        if use_local_wind_features and legacy_sitewise_wind_names.intersection(
            set(site_dynamic_feature_names)
        ):
            raise AssertionError(
                "Sitewise engineered local wind features must not use legacy offshore/interpolated wind names"
            )
        for site_name, frame in site_dynamic_frames.items():
            if len(site_dynamic_feature_names) != int(frame.shape[1]):
                raise AssertionError(
                    "site_dynamic_feature_names count must equal X_dynamic_sitewise feature dimension: "
                    f"{len(site_dynamic_feature_names)} vs {frame.shape[1]} for site={site_name}"
                )
    if x_static:
        for site_name, arr in x_static.items():
            if len(static_feature_names) != int(arr.shape[0]):
                raise AssertionError(
                    "static_feature_names count must equal X_static feature dimension: "
                    f"{len(static_feature_names)} vs {arr.shape[0]} for site={site_name}"
                )

    logging.info(
        "Point-centric sample counts | timesteps=%d target_sites=%d site_time_samples=%d",
        n_time,
        int(len(y_targets)),
        int(n_time * len(y_targets)),
    )
    logging.info(
        "Target tensor shapes | legacy=%s physical=%s transfer=%s reference=%s",
        tuple(next(iter(y_targets.values())).shape) if y_targets else (),
        tuple(next(iter(y_physical.values())).shape) if y_physical else (),
        tuple(next(iter(y_transfer.values())).shape) if y_transfer else (),
        tuple(next(iter(y_reference.values())).shape) if y_reference else (),
    )

    # Save arrays.
    dynamic_npz = outp / "point_centric_X_dynamic.npz"
    np.savez_compressed(
        str(dynamic_npz),
        X_dynamic=x_dynamic,
        timestamps=np.array([ts.isoformat() for ts in aligned_index], dtype=str),
        feature_names=np.array(list(x_dynamic_df.columns), dtype=str),
        offshore_sites=np.array(offshore_sites, dtype=str),
        train_idx=split_idx["train"],
        val_idx=split_idx["val"],
        test_idx=split_idx["test"],
    )

    y_payload = {
        "target_sites": np.array(canonical_target_sites, dtype=str),
        "target_feature_names": np.array(target_feature_names, dtype=str),
        "timestamps": np.array([ts.isoformat() for ts in aligned_index], dtype=str),
        "target_mode": np.array([str(targets_cfg.get("mode", "physical"))], dtype=str),
        "physical_target_names": np.array(physical_target_names, dtype=str),
    }
    if transfer_target_names:
        y_payload["transfer_target_names"] = np.array(transfer_target_names, dtype=str)
    if reference_target_names:
        y_payload["reference_target_names"] = np.array(reference_target_names, dtype=str)
    for site in canonical_target_sites:
        y_payload[f"Y__{_safe_name(site)}"] = y_targets[site]
        y_payload[f"Yphysical__{_safe_name(site)}"] = y_physical[site]
        if site in y_transfer:
            y_payload[f"Ytransfer__{_safe_name(site)}"] = y_transfer[site]
        if site in y_reference:
            y_payload[f"Yreference__{_safe_name(site)}"] = y_reference[site]

    targets_npz = outp / "point_centric_Y_targets.npz"
    np.savez_compressed(str(targets_npz), **y_payload)

    static_npz = None
    if x_static:
        static_payload = {
            "target_sites": np.array(
                [site for site in canonical_target_sites if site in x_static], dtype=str
            ),
            "static_feature_names": np.array(static_feature_names, dtype=str),
        }
        for site in canonical_target_sites:
            if site not in x_static:
                continue
            static_payload[f"Xstatic__{_safe_name(site)}"] = x_static[site]

        static_npz = outp / "point_centric_X_static.npz"
        np.savez_compressed(str(static_npz), **static_payload)

    physics_payload = None
    physics_npz = None
    if use_static_features and master_static_df is not None:
        physics_payload = _build_point_centric_physics_payload(
            master_static_df=master_static_df,
            target_sites=canonical_target_sites,
        )
        physics_npz = outp / "point_centric_physics.npz"
        np.savez_compressed(str(physics_npz), **physics_payload)

    dynamic_sitewise_npz = None
    if site_dynamic_frames:
        dynamic_sitewise_payload = {
            "target_sites": np.array(
                [site for site in canonical_target_sites if site in site_dynamic_frames], dtype=str
            ),
            "site_dynamic_feature_names": np.array(site_dynamic_feature_names, dtype=str),
            "timestamps": np.array([ts.isoformat() for ts in aligned_index], dtype=str),
        }
        for site_name in canonical_target_sites:
            if site_name not in site_dynamic_frames:
                continue
            dynamic_sitewise_payload[f"XdynamicSite__{_safe_name(site_name)}"] = (
                site_dynamic_frames[site_name].to_numpy(dtype=np.float32, copy=True)
            )

        dynamic_sitewise_npz = outp / "point_centric_X_dynamic_sitewise.npz"
        np.savez_compressed(str(dynamic_sitewise_npz), **dynamic_sitewise_payload)

    if source_metadata is not None and x_dynamic_sources is not None:
        source_metadata_path = outp / "point_centric_source_metadata.json"
        save_source_metadata_json(source_metadata, source_metadata_path)

        source_dynamic_npz = outp / "point_centric_X_dynamic_sources.npz"
        np.savez_compressed(
            str(source_dynamic_npz),
            X_dynamic_sources=x_dynamic_sources.astype(np.float32, copy=False),
            target_sites=np.array(source_metadata["target_sites"], dtype=str),
            timestamps=np.array([ts.isoformat() for ts in aligned_index], dtype=str),
            source_feature_names=np.array(source_feature_names, dtype=str),
            k_nearest=np.array([int(source_metadata["k_nearest"])], dtype=int),
        )

        if source_geometry is not None:
            source_geometry_npz = outp / "point_centric_source_geometry.npz"
            np.savez_compressed(
                str(source_geometry_npz),
                source_geometry=source_geometry.astype(np.float32, copy=False),
                target_sites=np.array(source_metadata["target_sites"], dtype=str),
                source_geometry_feature_names=np.array(source_geometry_feature_names, dtype=str),
                k_nearest=np.array([int(source_metadata["k_nearest"])], dtype=int),
            )
        logging.info(
            "Saved multi-source tensors | X_dynamic_sources=%s source_geometry=%s",
            tuple(x_dynamic_sources.shape),
            tuple(source_geometry.shape) if source_geometry is not None else "disabled",
        )

    bathy_npz = None
    bathy_info = {}
    if use_bathymetry:
        bathy_npz = outp / "point_centric_X_bathy.npz"
        bathy_info = build_bathymetry_patch_dataset(
            sites_yaml=resolved_sites_yaml,
            preprocess_cfg=prep_cfg,
            target_sites=canonical_target_sites,
            train_sites=normalization_train_sites,
            out_path=str(bathy_npz),
        )
    else:
        logging.info("Skipping bathymetry artifact because data.use_bathymetry=false")

    optional_artifacts = {
        "x_static": static_npz,
        "physics": physics_npz,
        "x_dynamic_sitewise": dynamic_sitewise_npz,
        "x_dynamic_sources": source_dynamic_npz,
        "source_geometry": source_geometry_npz,
        "source_metadata": source_metadata_path,
        "x_bathy": bathy_npz,
    }
    expected_optional_paths = {
        "x_static": outp / "point_centric_X_static.npz",
        "physics": outp / "point_centric_physics.npz",
        "x_dynamic_sitewise": outp / "point_centric_X_dynamic_sitewise.npz",
        "x_dynamic_sources": outp / "point_centric_X_dynamic_sources.npz",
        "source_geometry": outp / "point_centric_source_geometry.npz",
        "source_metadata": outp / "point_centric_source_metadata.json",
        "x_bathy": outp / "point_centric_X_bathy.npz",
    }
    for artifact_name, expected_path in expected_optional_paths.items():
        if optional_artifacts.get(artifact_name) is None:
            _remove_stale_optional_artifact(expected_path)

    train_site_set = set(split_info.get("train_sites", []) or [])
    val_site_set = set(split_info.get("val_sites", []) or [])
    test_site_set = set(split_info.get("test_sites", []) or [])
    site_identity_rows = []
    for entry in nearshore_entries:
        site_name = str(entry.get("name", "")).strip()
        if not site_name:
            continue
        if site_name in train_site_set:
            split_label = "train"
        elif site_name in val_site_set:
            split_label = "val"
        elif site_name in test_site_set:
            split_label = "test"
        else:
            split_label = "unassigned"
        site_identity_rows.append(
            {
                "site_name": site_name,
                "lat": _try_float(entry.get("lat")),
                "lon": _try_float(entry.get("lon")),
                "split": split_label,
            }
        )
    artifact_site_orders = {
        "y_targets": list(canonical_target_sites),
        "x_static": [site for site in canonical_target_sites if site in x_static],
        "x_dynamic_sitewise": [
            site for site in canonical_target_sites if site in site_dynamic_frames
        ],
        "x_dynamic_sources": list(source_metadata.get("target_sites", []) or [])
        if source_metadata is not None
        else [],
        "source_geometry": list(source_metadata.get("target_sites", []) or [])
        if source_metadata is not None
        else [],
        "physics": list(np.asarray(physics_payload["target_sites"]).astype(str))
        if physics_payload is not None
        else [],
        "x_bathy": list(np.asarray(bathy_info.get("target_sites", []) or []).astype(str))
        if bathy_info
        else [],
    }

    metadata = {
        "X_dynamic_shape": list(x_dynamic.shape),
        "Y_targets_shapes": {s: list(arr.shape) for s, arr in y_targets.items()},
        "Y_physical_shapes": {s: list(arr.shape) for s, arr in y_physical.items()},
        "Y_transfer_shapes": {s: list(arr.shape) for s, arr in y_transfer.items()},
        "Y_reference_shapes": {s: list(arr.shape) for s, arr in y_reference.items()},
        "X_static_shapes": {s: list(arr.shape) for s, arr in x_static.items()},
        "X_bathy_shape": list(bathy_info.get("shape", [])) if bathy_info else [],
        "X_dynamic_sitewise_shapes": {
            s: list(site_dynamic_frames[s].shape)
            for s in canonical_target_sites
            if s in site_dynamic_frames
        },
        "physics_shape": (
            {
                "target_sites": int(len(physics_payload["target_sites"])),
                "local_depth_m": list(np.asarray(physics_payload["local_depth_m"]).shape),
                "local_breaking_hs_cap": list(
                    np.asarray(physics_payload["local_breaking_hs_cap"]).shape
                ),
                "local_breaking_cap_valid": list(
                    np.asarray(physics_payload["local_breaking_cap_valid"]).shape
                ),
            }
            if physics_payload is not None
            else {}
        ),
        "X_dynamic_sources_shape": list(x_dynamic_sources.shape)
        if x_dynamic_sources is not None
        else [],
        "source_geometry_shape": list(source_geometry.shape) if source_geometry is not None else [],
        "sites_yaml": resolved_sites_yaml,
        "training_config": training_config,
        "preprocess_config": preprocess_config,
        "static_features_csv": static_csv if use_static_features else "",
        "offshore_sites": offshore_sites,
        "offshore_sites_all": offshore_sites_all,
        "excluded_offshore_wind_sites": excluded_offshore_wind_sites,
        "wind_feature_toggles": {
            "use_offshore_wind_features": bool(use_offshore_wind_features),
            "use_local_wind_features": bool(use_local_wind_features),
        },
        "wave_to_wind_site": wave_to_wind_site,
        "site_to_wave_site": site_to_wave_site,
        "local_wind_source_type": "nearest_sites_yaml_wind_point"
        if use_local_wind_features
        else "disabled",
        "local_wind_assignments": list(local_wind_diagnostics.get("assignments", []) or []),
        "local_wind_diagnostics": {
            "enabled": bool(use_local_wind_features),
            "num_norac_points": int(local_wind_diagnostics.get("num_norac_points", 0)),
            "num_candidate_local_wind_points": int(
                local_wind_diagnostics.get("num_candidate_local_wind_points", 0)
            ),
            "max_nearest_local_wind_distance_m": float(
                local_wind_diagnostics.get("max_nearest_local_wind_distance_m", float("nan"))
            ),
            "median_nearest_local_wind_distance_m": float(
                local_wind_diagnostics.get("median_nearest_local_wind_distance_m", float("nan"))
            ),
            "example_local_wind_assignments": list(
                local_wind_diagnostics.get("example_local_wind_assignments", []) or []
            ),
            "candidate_local_wind_points": list(
                local_wind_diagnostics.get("candidate_local_wind_points", []) or []
            ),
        },
        "nearshore_sites": list(canonical_target_sites),
        "point_identity": {
            "canonical_site_identifier": "configs/sites.yaml nearshore_sites[].name",
            "canonical_site_order": list(canonical_target_sites),
            "site_rows": site_identity_rows,
            "artifact_site_orders": artifact_site_orders,
            "sample_layout": {
                "timestamps_axis": "shared_global_time_index",
                "per_site_targets": "[time, feature] keyed by site name",
                "multi_source_tensor": "[site, time, source, feature] aligned to target_sites",
                "bathymetry_tensor": "[site, channel, y, x] aligned to target_sites",
            },
        },
        "dynamic_feature_names": list(x_dynamic_df.columns),
        "site_dynamic_feature_names": site_dynamic_feature_names,
        "source_feature_names": source_feature_names,
        "source_geometry_feature_names": source_geometry_feature_names,
        "target_feature_names": target_feature_names,
        "physical_target_names": physical_target_names,
        "transfer_target_names": transfer_target_names,
        "reference_target_names": reference_target_names,
        "static_feature_names": static_feature_names,
        "targets": {
            **targets_cfg,
            "reference_source": transfer_reference_source,
        },
        "date_range": date_range_meta,
        "sample_counts": {
            "timesteps": n_time,
            "target_sites": int(len(y_targets)),
            "site_time_samples": int(n_time * len(y_targets)),
        },
        "circular_features": {
            "dynamic": sorted(set(circular_dynamic_features)),
            "targets": sorted(set(circular_target_features)),
            "static": sorted(set(circular_static_features)),
            "multi_source_dynamic": list(multi_source_meta.get("circular_features", [])),
        },
        "normalization": {
            "dynamic_method": method,
            "feature_range": list(feature_range),
            "fit_sites": list(normalization_train_sites),
            "train_idx_count": int(len(split_fit_idx)),
            # Persist fitted values as well as the column lists.  Forward-only
            # inference must reuse these exact transformations and has no
            # target data from which it could safely refit them.
            "dynamic_scaler": {
                **x_scaler,
                "method": x_scaler.get("method", method),
                "columns": x_scaler.get("columns", []),
                "skipped_circular_columns": [
                    c for c in x_dynamic_df.columns if c.endswith("_sin") or c.endswith("_cos")
                ],
            },
            "site_dynamic_scaler": {
                **site_dynamic_scaler,
                "method": site_dynamic_scaler.get("method", method),
                "columns": site_dynamic_scaler.get("columns", []),
            },
            "source_dynamic_scaler": {
                **(multi_source_meta.get("source_dynamic_scaler_stats", {}) or {}),
                **multi_source_meta.get("source_dynamic_scaler", {"method": method, "columns": []}),
            },
            "target_scaler": target_scaler,
            "physical_target_scaler": target_scaler,
            "transfer_target_scaler": transfer_scaler or {},
            "static_scaler": static_scaler,
        },
        "multi_source": {
            **multi_source_cfg,
            **multi_source_meta,
        },
        "splits": {
            "mode": split_info.get("validation_mode", "legacy-temporal"),
            "validation_site_heldout": bool(split_info.get("validation_site_heldout", False)),
            "site_holdout_temporal_mode": str(
                split_info.get("site_holdout_temporal_mode", "legacy")
            ),
            "site_holdout_temporal_active": bool(
                split_info.get("site_holdout_temporal_active", False)
            ),
            "site_holdout_temporal_recent_fraction": split_info.get(
                "site_holdout_temporal_recent_fraction", None
            ),
            "site_holdout_temporal_train_fraction": float(
                split_info.get("site_holdout_temporal_train_fraction", train_frac)
            ),
            "requested_validation_sites": list(
                split_info.get("requested_validation_sites", []) or []
            ),
            "requested_test_sites": list(split_info.get("requested_test_sites", []) or []),
            "train_sites": list(split_info.get("train_sites", []) or []),
            "val_sites": list(split_info.get("val_sites", []) or []),
            "test_sites": list(split_info.get("test_sites", []) or []),
            "train_site_subsampling": dict(split_info.get("train_site_subsampling", {}) or {}),
        },
        "ablation": static_ablation_summary,
        "hybrid_targets": {
            "task_modes": {
                "hs": "regression",
                "tp": "classification",
                "dir": "classification",
                "dp": "classification",
            },
            "num_tp_bins": int(
                (train_cfg.get("model", {}) or {})
                .get("coastal_transformer", {})
                .get("num_tp_bins", 32)
            ),
            "num_dp_bins": int(
                (train_cfg.get("model", {}) or {})
                .get("coastal_transformer", {})
                .get("num_dp_bins", 36)
            ),
            "label_smoothing_sigma": float(
                (train_cfg.get("training", {}) or {})
                .get("loss", {})
                .get("label_smoothing_sigma", 0.8)
            ),
            "tp_bin_range": list(
                (
                    (train_cfg.get("training", {}) or {})
                    .get("loss", {})
                    .get("blueprint_hybrid", {})
                    .get("tp_bin_range", [0.0, 25.0])
                ),
            ),
            "dp_bin_range": list(
                (
                    (train_cfg.get("training", {}) or {})
                    .get("loss", {})
                    .get("blueprint_hybrid", {})
                    .get("dp_bin_range", [0.0, 360.0])
                ),
            ),
        },
        "bathymetry": bathy_info,
        "files": {
            "x_dynamic": str(dynamic_npz),
            "y_targets": str(targets_npz),
            "x_static": str(static_npz) if static_npz is not None else "",
            "physics": str(physics_npz) if physics_npz is not None else "",
            "x_bathy": str(bathy_npz) if bathy_npz is not None else "",
            "x_dynamic_sitewise": str(dynamic_sitewise_npz)
            if dynamic_sitewise_npz is not None
            else "",
            "x_dynamic_sources": str(source_dynamic_npz) if source_dynamic_npz is not None else "",
            "source_geometry": str(source_geometry_npz) if source_geometry_npz is not None else "",
            "source_metadata": str(source_metadata_path)
            if source_metadata_path is not None
            else "",
        },
    }

    metadata_path = outp / "point_centric_metadata.json"

    def _json_default(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")

    with metadata_path.open("w") as fh:
        json.dump(metadata, fh, indent=2, default=_json_default)

    logging.info("Saved point-centric outputs to %s", outp)
    return {
        "x_dynamic": str(dynamic_npz),
        "x_dynamic_sitewise": str(dynamic_sitewise_npz) if dynamic_sitewise_npz is not None else "",
        "y_targets": str(targets_npz),
        "x_static": str(static_npz) if static_npz is not None else "",
        "physics": str(physics_npz) if physics_npz is not None else "",
        "x_bathy": str(bathy_npz) if bathy_npz is not None else "",
        "metadata": str(metadata_path),
        "x_dynamic_sources": str(source_dynamic_npz) if source_dynamic_npz is not None else "",
        "source_geometry": str(source_geometry_npz) if source_geometry_npz is not None else "",
        "source_metadata": str(source_metadata_path) if source_metadata_path is not None else "",
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build point-centric coastal-transformer arrays")
    parser.add_argument("--sites", default="configs/sites.yaml", help="Path to sites.yaml")
    parser.add_argument(
        "--training-config", default="configs/training.yaml", help="Path to training config"
    )
    parser.add_argument(
        "--preprocess-config", default="configs/preprocess.yaml", help="Path to preprocess config"
    )
    parser.add_argument("--out-dir", default=None, help="Output directory")
    args = parser.parse_args()

    paths = build_point_centric_dataset(
        sites_yaml=args.sites,
        training_config=args.training_config,
        preprocess_config=args.preprocess_config,
        out_dir=args.out_dir,
    )
    print(json.dumps(paths, indent=2))
