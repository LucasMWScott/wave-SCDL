#!/usr/bin/env python3
"""Build per-site bathymetry patches for the point-centric coastal dataset."""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
from scipy import ndimage

try:
    from pyproj import Transformer
except Exception:
    Transformer = None


def read_yaml(path):
    """Read configuration paths relative to their declaring file."""
    from coastal_wave.common.config import read_config

    return read_config(path)


def _load_geometric_bathy_module():
    """Load the installed bathymetry preparation module."""
    from geometric_builder.src import generate_bathy_field

    return generate_bathy_field


def _unwrap_metadata(value) -> dict:
    if isinstance(value, dict):
        return value
    if isinstance(value, np.ndarray) and value.shape == ():
        try:
            item = value.item()
        except Exception:
            return {}
        return item if isinstance(item, dict) else {}
    return {}


def _ensure_full_bathy_grid(
    sites_yaml: str,
    preprocess_cfg: dict,
) -> Path:
    bathy_cfg = preprocess_cfg.get("bathy", {}) or {}
    full_path = Path(
        bathy_cfg.get("full_grid_path")
        or bathy_cfg.get("composite_path")
        or "data/processed/bathy/bathy_field_full.npz"
    )
    if full_path.exists():
        return full_path

    module = _load_geometric_bathy_module()
    out_dir = full_path.parent
    full_name = full_path.name
    subgrid_name = str(bathy_cfg.get("subgrid_name", "bathy_field_project_site.npz"))
    padding_m = float(bathy_cfg.get("padding_m", 3000.0))
    resolution_m = float(bathy_cfg.get("resolution_m", 50.0))
    epsg = int(bathy_cfg.get("epsg", 32633))
    bathy_dir = str(bathy_cfg.get("raw_dir", "data/raw/bathy"))

    logging.info("Bathymetry full-grid product missing; generating %s", full_path)
    module.run_generate_bathy_field(
        sites_path=str(sites_yaml),
        bathy_dir_path=bathy_dir,
        padding_m=padding_m,
        resolution_m=resolution_m,
        epsg=epsg,
        out_dir_path=str(out_dir),
        full_name=full_name,
        subgrid_name=subgrid_name,
    )
    if not full_path.exists():
        raise FileNotFoundError(
            f"Bathymetry grid generation completed but file is missing: {full_path}"
        )
    return full_path


def _load_bathy_grid(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    with np.load(path, allow_pickle=True) as npz:
        x = np.asarray(npz["x"], dtype=np.float32)
        y = np.asarray(npz["y"], dtype=np.float32)
        z = np.asarray(npz["z"], dtype=np.float32)
        land_mask = (
            np.asarray(npz["land_mask"], dtype=bool) if "land_mask" in npz else ~np.isfinite(z)
        )
        metadata = _unwrap_metadata(npz["metadata"]) if "metadata" in npz else {}
    if z.ndim != 2:
        raise ValueError(f"Expected 2D bathy grid, got shape {z.shape}")
    if land_mask.shape != z.shape:
        raise ValueError(
            f"Bathymetry land_mask shape mismatch: grid={z.shape}, land_mask={land_mask.shape}"
        )
    return x, y, z, land_mask, metadata


def _get_transformer(epsg: int):
    if Transformer is None:
        raise RuntimeError("pyproj is required to build bathymetry patches")
    return Transformer.from_crs(4326, int(epsg), always_xy=True)


def _normalize_water_depth(z: np.ndarray, metadata: dict) -> np.ndarray:
    vertical_convention = str(metadata.get("vertical_convention", "")).strip().lower()
    if "positive_down" in vertical_convention:
        return np.asarray(z, dtype=np.float32)
    if "signed_elevation" in vertical_convention:
        return (-np.asarray(z, dtype=np.float32)).astype(np.float32, copy=False)

    finite = np.asarray(z, dtype=np.float32)[np.isfinite(z)]
    if finite.size == 0:
        return np.asarray(z, dtype=np.float32)
    if float(np.mean(finite > 0.0)) > 0.8:
        return np.asarray(z, dtype=np.float32)
    return (-np.asarray(z, dtype=np.float32)).astype(np.float32, copy=False)


def _nearest_index(values: np.ndarray, target: float) -> int:
    if values.ndim != 1 or values.size == 0:
        raise ValueError("Grid coordinates must be non-empty 1D arrays")
    return int(np.argmin(np.abs(values - float(target))))


def _snap_to_wet_cell(
    row: int,
    col: int,
    wet_mask: np.ndarray,
    max_snap_cells: int,
) -> tuple[int, int]:
    if wet_mask[int(row), int(col)]:
        return int(row), int(col)

    best = None
    best_dist = float("inf")
    row = int(row)
    col = int(col)
    radius = max(0, int(max_snap_cells))
    for rr in range(max(0, row - radius), min(wet_mask.shape[0], row + radius + 1)):
        for cc in range(max(0, col - radius), min(wet_mask.shape[1], col + radius + 1)):
            if not wet_mask[rr, cc]:
                continue
            dist = math.hypot(rr - row, cc - col)
            if dist < best_dist:
                best = (rr, cc)
                best_dist = dist
    if best is None:
        return row, col
    return best


def _extract_patch(
    array: np.ndarray,
    center_row: int,
    center_col: int,
    patch_size: int,
    fill_value,
) -> np.ndarray:
    if patch_size < 1:
        raise ValueError("patch_size must be >= 1")
    half = patch_size // 2
    row0 = int(center_row) - half
    row1 = row0 + patch_size
    col0 = int(center_col) - half
    col1 = col0 + patch_size

    if array.ndim == 2:
        out = np.full((patch_size, patch_size), fill_value, dtype=array.dtype)
        src_row0 = max(0, row0)
        src_row1 = min(array.shape[0], row1)
        src_col0 = max(0, col0)
        src_col1 = min(array.shape[1], col1)
        dst_row0 = src_row0 - row0
        dst_row1 = dst_row0 + (src_row1 - src_row0)
        dst_col0 = src_col0 - col0
        dst_col1 = dst_col0 + (src_col1 - src_col0)
        out[dst_row0:dst_row1, dst_col0:dst_col1] = array[src_row0:src_row1, src_col0:src_col1]
        return out

    raise ValueError(f"Only 2D arrays are supported for patch extraction, got ndim={array.ndim}")


def _unwrap_scalar(value, default):
    if value is None:
        return default
    if isinstance(value, np.ndarray) and value.shape == ():
        try:
            return value.item()
        except Exception:
            return default
    return value


def _fit_standard_stats(values: np.ndarray) -> dict:
    arr = np.asarray(values, dtype=np.float32)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"method": "standard", "mean": 0.0, "std": 1.0}
    mean = float(np.mean(arr))
    std = float(np.std(arr))
    if not np.isfinite(std) or std <= 0.0:
        std = 1.0
    return {"method": "standard", "mean": mean, "std": std}


def _apply_standard_stats(values: np.ndarray, stats: dict) -> np.ndarray:
    mean = float(stats.get("mean", 0.0))
    std = float(stats.get("std", 1.0))
    if not np.isfinite(std) or std <= 0.0:
        std = 1.0
    return ((np.asarray(values, dtype=np.float32) - mean) / std).astype(np.float32, copy=False)


def _fit_unit_interval_depth_stats(values: np.ndarray) -> dict:
    arr = np.asarray(values, dtype=np.float32)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"method": "unit_interval_train_max", "train_max": 1.0}
    train_max = float(np.max(arr))
    if not np.isfinite(train_max) or train_max <= 0.0:
        train_max = 1.0
    return {"method": "unit_interval_train_max", "train_max": train_max}


def _apply_unit_interval_depth_stats(values: np.ndarray, stats: dict) -> np.ndarray:
    train_max = float(stats.get("train_max", 1.0))
    if not np.isfinite(train_max) or train_max <= 0.0:
        train_max = 1.0
    clipped = np.clip(np.asarray(values, dtype=np.float32), 0.0, train_max)
    return (clipped / train_max).astype(np.float32, copy=False)


def _fit_robust_stats(values: np.ndarray) -> dict:
    arr = np.asarray(values, dtype=np.float32)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"method": "robust", "median": 0.0, "p05": 0.0, "p95": 1.0, "scale": 1.0}
    median = float(np.nanmedian(arr))
    p05 = float(np.nanpercentile(arr, 5.0))
    p95 = float(np.nanpercentile(arr, 95.0))
    scale = float(p95 - p05)
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0
    return {"method": "robust", "median": median, "p05": p05, "p95": p95, "scale": scale}


def _apply_robust_stats(values: np.ndarray, stats: dict) -> np.ndarray:
    median = float(stats.get("median", 0.0))
    scale = float(stats.get("scale", 1.0))
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0
    return ((np.asarray(values, dtype=np.float32) - median) / scale).astype(np.float32, copy=False)


def _nearest_fill_water_values(field: np.ndarray, wet_mask: np.ndarray) -> np.ndarray:
    """Fill dry/invalid pixels using nearest wet-cell values for stable derivatives."""
    base = np.asarray(field, dtype=np.float32).copy()
    valid = np.asarray(wet_mask, dtype=bool) & np.isfinite(base)
    if not np.any(valid):
        return np.nan_to_num(base, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    filled = np.where(valid, base, np.nan).astype(np.float32, copy=False)
    indices = ndimage.distance_transform_edt(~valid, return_distances=False, return_indices=True)
    return filled[tuple(indices)].astype(np.float32, copy=False)


def build_bathymetry_patch_dataset(
    sites_yaml: str,
    preprocess_cfg: dict,
    target_sites: Sequence[str],
    train_sites: Sequence[str],
    out_path: str,
    normalization_reference: str | None = None,
) -> dict:
    """Build normalized, site-indexed bathymetry patches for model input.

    Statistics are fitted only on wet cells at training sites unless a
    ``normalization_reference`` is supplied.  The returned metadata records
    channel order, patch geometry, and the normalization state needed to reuse
    the patches with a checkpoint.
    """
    bathy_cfg = preprocess_cfg.get("bathy", {}) or {}
    version = str(bathy_cfg.get("version", "v1")).strip().lower()
    requested_channels = list(bathy_cfg.get("channels", []) or [])
    default_v2_channels = [
        "depth",
        "land_sea_mask",
        "slope_magnitude",
        "distance_to_land",
        "curvature_laplacian",
        "shallow_breaking_mask",
    ]
    if not requested_channels:
        requested_channels = (
            default_v2_channels if version == "v2" else ["depth_log1p_norm", "wet_mask"]
        )
    patch_size = int(bathy_cfg.get("patch_size", 128 if version == "v2" else 64))
    resolution_m = float(bathy_cfg.get("resolution_m", 50.0))
    depth_clip_m = float(bathy_cfg.get("depth_clip_m", 120.0))
    epsg = int(bathy_cfg.get("epsg", 32633))
    max_snap_cells = int(bathy_cfg.get("max_snap_cells", 2))
    shallow_breaking_depth_m = float(bathy_cfg.get("shallow_breaking_depth_m", 15.0))
    requested_in_channels = int(bathy_cfg.get("in_channels", len(requested_channels)))
    if requested_in_channels != len(requested_channels):
        raise ValueError(
            f"Bathymetry config mismatch: in_channels={requested_in_channels} but channels={requested_channels}"
        )
    if version not in {"v1", "v2"}:
        raise ValueError(f"Unsupported bathymetry version '{version}'. Expected 'v1' or 'v2'.")
    checkpoint_v2_channels = [
        "depth",
        "land_sea_mask",
        "slope_magnitude",
        "curvature_laplacian",
        "distance_to_land",
    ]
    if version == "v2" and list(requested_channels) not in (
        list(default_v2_channels),
        checkpoint_v2_channels,
    ):
        raise ValueError(
            "Bathymetry v2 currently expects channels="
            f"{default_v2_channels} or checkpoint-compatible {checkpoint_v2_channels}, got {requested_channels}"
        )

    full_grid_path = _ensure_full_bathy_grid(sites_yaml=sites_yaml, preprocess_cfg=preprocess_cfg)
    x_coord, y_coord, z_grid, land_mask, bathy_meta = _load_bathy_grid(full_grid_path)
    sites_cfg = read_yaml(sites_yaml)
    nearshore_lookup = {
        str(site.get("name")): site
        for site in (sites_cfg.get("nearshore_sites") or [])
        if site.get("name") is not None
    }

    missing_sites = sorted(set(str(site) for site in target_sites) - set(nearshore_lookup))
    if missing_sites:
        raise ValueError(f"Nearshore sites missing from {sites_yaml}: {missing_sites}")

    transformer = _get_transformer(epsg)
    water_depth_grid = _normalize_water_depth(z_grid, metadata=bathy_meta)
    water_mask_grid = (~land_mask) & np.isfinite(water_depth_grid) & (water_depth_grid > 0.0)
    filled_depth_grid = _nearest_fill_water_values(water_depth_grid, water_mask_grid)

    dy_m = float(abs(np.median(np.diff(y_coord)))) if y_coord.size > 1 else float(resolution_m)
    dx_m = float(abs(np.median(np.diff(x_coord)))) if x_coord.size > 1 else float(resolution_m)
    slope_y, slope_x = np.gradient(filled_depth_grid.astype(np.float32), dy_m, dx_m)
    slope_magnitude_grid = np.hypot(slope_x, slope_y).astype(np.float32, copy=False)
    distance_to_land_grid = ndimage.distance_transform_edt(
        water_mask_grid, sampling=(dy_m, dx_m)
    ).astype(np.float32, copy=False)
    curvature_laplacian_grid = ndimage.laplace(filled_depth_grid.astype(np.float32)).astype(
        np.float32, copy=False
    )
    shallow_breaking_mask_grid = (
        water_mask_grid
        & np.isfinite(water_depth_grid)
        & (water_depth_grid <= shallow_breaking_depth_m)
    ).astype(np.float32, copy=False)

    target_sites = [str(site) for site in target_sites]
    train_site_set = {str(site) for site in train_sites}
    raw_depth_patches = np.full(
        (len(target_sites), patch_size, patch_size), np.nan, dtype=np.float32
    )
    wet_mask_patches = np.zeros((len(target_sites), patch_size, patch_size), dtype=np.float32)
    slope_patches = np.full((len(target_sites), patch_size, patch_size), np.nan, dtype=np.float32)
    distance_to_land_patches = np.full(
        (len(target_sites), patch_size, patch_size), np.nan, dtype=np.float32
    )
    curvature_patches = np.full(
        (len(target_sites), patch_size, patch_size), np.nan, dtype=np.float32
    )
    shallow_breaking_patches = np.zeros(
        (len(target_sites), patch_size, patch_size), dtype=np.float32
    )
    centers: Dict[str, dict] = {}

    for site_idx, site_name in enumerate(target_sites):
        site_meta = nearshore_lookup[site_name]
        easting, northing = transformer.transform(float(site_meta["lon"]), float(site_meta["lat"]))
        center_col = _nearest_index(x_coord, easting)
        center_row = _nearest_index(y_coord, northing)
        snapped_row, snapped_col = _snap_to_wet_cell(
            row=center_row,
            col=center_col,
            wet_mask=water_mask_grid,
            max_snap_cells=max_snap_cells,
        )

        depth_patch = _extract_patch(
            water_depth_grid,
            center_row=snapped_row,
            center_col=snapped_col,
            patch_size=patch_size,
            fill_value=np.nan,
        )
        wet_patch = _extract_patch(
            water_mask_grid.astype(np.float32),
            center_row=snapped_row,
            center_col=snapped_col,
            patch_size=patch_size,
            fill_value=np.float32(0.0),
        )

        raw_depth_patches[site_idx] = depth_patch
        wet_mask_patches[site_idx] = wet_patch
        slope_patches[site_idx] = _extract_patch(
            slope_magnitude_grid,
            center_row=snapped_row,
            center_col=snapped_col,
            patch_size=patch_size,
            fill_value=np.nan,
        )
        distance_to_land_patches[site_idx] = _extract_patch(
            distance_to_land_grid,
            center_row=snapped_row,
            center_col=snapped_col,
            patch_size=patch_size,
            fill_value=np.nan,
        )
        curvature_patches[site_idx] = _extract_patch(
            curvature_laplacian_grid,
            center_row=snapped_row,
            center_col=snapped_col,
            patch_size=patch_size,
            fill_value=np.nan,
        )
        shallow_breaking_patches[site_idx] = _extract_patch(
            shallow_breaking_mask_grid,
            center_row=snapped_row,
            center_col=snapped_col,
            patch_size=patch_size,
            fill_value=np.float32(0.0),
        )
        centers[site_name] = {
            "requested_row": int(center_row),
            "requested_col": int(center_col),
            "snapped_row": int(snapped_row),
            "snapped_col": int(snapped_col),
        }

    train_depth_values_v1: List[np.ndarray] = []
    train_depth_values_v2: List[np.ndarray] = []
    train_slope_values: List[np.ndarray] = []
    train_distance_to_land_values: List[np.ndarray] = []
    train_curvature_values: List[np.ndarray] = []
    for site_idx, site_name in enumerate(target_sites):
        if site_name not in train_site_set:
            continue
        depth_patch = raw_depth_patches[site_idx]
        wet_patch = wet_mask_patches[site_idx] > 0.5
        valid = wet_patch & np.isfinite(depth_patch)
        if np.any(valid):
            clipped = np.clip(depth_patch[valid], 0.0, depth_clip_m)
            train_depth_values_v1.append(np.log1p(clipped).astype(np.float32, copy=False))
            train_depth_values_v2.append(clipped.astype(np.float32, copy=False))
        slope_patch = slope_patches[site_idx]
        valid_slope = wet_patch & np.isfinite(slope_patch)
        if np.any(valid_slope):
            train_slope_values.append(slope_patch[valid_slope].astype(np.float32, copy=False))
        dist_patch = distance_to_land_patches[site_idx]
        valid_dist = wet_patch & np.isfinite(dist_patch)
        if np.any(valid_dist):
            train_distance_to_land_values.append(
                dist_patch[valid_dist].astype(np.float32, copy=False)
            )
        curvature_patch = curvature_patches[site_idx]
        valid_curvature = wet_patch & np.isfinite(curvature_patch)
        if np.any(valid_curvature):
            train_curvature_values.append(
                curvature_patch[valid_curvature].astype(np.float32, copy=False)
            )

    if not train_depth_values_v1:
        raise RuntimeError("No finite wet-cell bathymetry values found on training sites")

    depth_log_stats = _fit_standard_stats(np.concatenate(train_depth_values_v1, axis=0))
    train_depth_values_v2_arr = np.concatenate(train_depth_values_v2, axis=0)
    depth_stats_v2 = (
        _fit_unit_interval_depth_stats(train_depth_values_v2_arr)
        if version == "v2"
        else _fit_standard_stats(train_depth_values_v2_arr)
    )
    slope_stats = _fit_robust_stats(
        np.concatenate(train_slope_values, axis=0)
        if train_slope_values
        else np.array([], dtype=np.float32)
    )
    distance_to_land_stats = _fit_robust_stats(
        np.concatenate(train_distance_to_land_values, axis=0)
        if train_distance_to_land_values
        else np.array([], dtype=np.float32)
    )
    curvature_stats = _fit_robust_stats(
        np.concatenate(train_curvature_values, axis=0)
        if train_curvature_values
        else np.array([], dtype=np.float32)
    )

    reference_path = Path(normalization_reference) if normalization_reference else None
    if reference_path is not None:
        if not reference_path.exists():
            raise FileNotFoundError(
                f"Bathymetry normalization reference not found: {reference_path}"
            )
        with np.load(reference_path, allow_pickle=True) as reference_npz:
            if "normalization_metadata" not in reference_npz:
                raise ValueError(
                    "Bathymetry normalization reference is missing 'normalization_metadata': "
                    f"{reference_path}"
                )
            reference_metadata = _unwrap_metadata(reference_npz["normalization_metadata"])
        if str(reference_metadata.get("version", "")).strip().lower() != version:
            raise ValueError(
                "Bathymetry normalization version mismatch: "
                f"reference={reference_metadata.get('version')!r}, requested={version!r}"
            )
        reference_channels = list(reference_metadata.get("channels", []) or [])
        if not set(requested_channels).issubset(set(reference_channels)):
            raise ValueError(
                "Bathymetry normalization channel mismatch between reference and requested patch build"
            )
        reference_stats = reference_metadata.get("stats", {}) or {}
        required_stats = {
            "depth_log1p_norm",
            "depth",
            "slope_magnitude",
            "distance_to_land",
            "curvature_laplacian",
        }
        missing_stats = sorted(required_stats - set(reference_stats))
        if missing_stats:
            raise ValueError(
                f"Bathymetry normalization reference is missing stats: {missing_stats}"
            )
        depth_log_stats = dict(reference_stats["depth_log1p_norm"])
        depth_stats_v2 = dict(reference_stats["depth"])
        slope_stats = dict(reference_stats["slope_magnitude"])
        distance_to_land_stats = dict(reference_stats["distance_to_land"])
        curvature_stats = dict(reference_stats["curvature_laplacian"])

    if version == "v2":
        x_bathy = np.zeros(
            (len(target_sites), len(requested_channels), patch_size, patch_size), dtype=np.float32
        )
    else:
        x_bathy = np.zeros((len(target_sites), 2, patch_size, patch_size), dtype=np.float32)

    for site_idx in range(len(target_sites)):
        depth_patch = raw_depth_patches[site_idx]
        wet_patch = wet_mask_patches[site_idx]
        safe_depth = np.clip(
            np.nan_to_num(depth_patch, nan=0.0, posinf=depth_clip_m, neginf=0.0), 0.0, depth_clip_m
        ).astype(np.float32, copy=False)

        if version == "v2":
            depth_norm = _apply_unit_interval_depth_stats(safe_depth, depth_stats_v2)
            depth_norm = np.where(wet_patch > 0.5, depth_norm, 0.0).astype(np.float32, copy=False)

            slope_norm = _apply_robust_stats(
                np.nan_to_num(slope_patches[site_idx], nan=0.0, posinf=0.0, neginf=0.0),
                slope_stats,
            )
            slope_norm = np.where(wet_patch > 0.5, slope_norm, 0.0).astype(np.float32, copy=False)

            distance_to_land_norm = _apply_robust_stats(
                np.nan_to_num(distance_to_land_patches[site_idx], nan=0.0, posinf=0.0, neginf=0.0),
                distance_to_land_stats,
            )
            distance_to_land_norm = np.where(wet_patch > 0.5, distance_to_land_norm, 0.0).astype(
                np.float32, copy=False
            )

            curvature_norm = _apply_robust_stats(
                np.nan_to_num(curvature_patches[site_idx], nan=0.0, posinf=0.0, neginf=0.0),
                curvature_stats,
            )
            curvature_norm = np.where(wet_patch > 0.5, curvature_norm, 0.0).astype(
                np.float32, copy=False
            )

            channels = {
                "depth": depth_norm,
                "land_sea_mask": wet_patch.astype(np.float32, copy=False),
                "slope_magnitude": slope_norm,
                "distance_to_land": distance_to_land_norm,
                "curvature_laplacian": curvature_norm,
                "shallow_breaking_mask": shallow_breaking_patches[site_idx].astype(
                    np.float32, copy=False
                ),
            }
            for channel_idx, channel_name in enumerate(requested_channels):
                x_bathy[site_idx, channel_idx, :, :] = channels[channel_name]
        else:
            depth_log = np.log1p(safe_depth).astype(np.float32, copy=False)
            depth_norm = _apply_standard_stats(depth_log, depth_log_stats)
            depth_norm = np.where(wet_patch > 0.5, depth_norm, 0.0).astype(np.float32, copy=False)
            x_bathy[site_idx, 0, :, :] = depth_norm
            x_bathy[site_idx, 1, :, :] = wet_patch.astype(np.float32, copy=False)

    normalization_metadata = {
        "version": version,
        "depth_clip_m": float(depth_clip_m),
        "shallow_breaking_depth_m": float(shallow_breaking_depth_m),
        "channels": list(requested_channels),
        "stats": {
            "depth_log1p_norm": depth_log_stats,
            "depth": depth_stats_v2,
            "slope_magnitude": slope_stats,
            "distance_to_land": distance_to_land_stats,
            "curvature_laplacian": curvature_stats,
        },
        "normalization_reference": str(reference_path) if reference_path is not None else None,
    }
    channel_names = (
        list(requested_channels) if version == "v2" else ["depth_log1p_norm", "wet_mask"]
    )
    if len(channel_names) != int(x_bathy.shape[1]):
        raise AssertionError("Bathymetry channel count does not match constructed tensor shape")

    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "X_bathy": x_bathy,
        "target_sites": np.asarray(target_sites, dtype=str),
        "channel_names": np.asarray(channel_names, dtype=str),
        "patch_size": np.asarray(patch_size, dtype=np.int32),
        "resolution_m": np.asarray(resolution_m, dtype=np.float32),
        "normalization_metadata": np.asarray(normalization_metadata, dtype=object),
    }
    if version == "v1":
        payload["train_depth_mean"] = np.asarray(float(depth_log_stats["mean"]), dtype=np.float32)
        payload["train_depth_std"] = np.asarray(float(depth_log_stats["std"]), dtype=np.float32)
    np.savez_compressed(str(out_file), **payload)

    return {
        "path": str(out_file),
        "shape": list(x_bathy.shape),
        "target_sites": list(target_sites),
        "channel_names": channel_names,
        "version": version,
        "patch_size": patch_size,
        "resolution_m": resolution_m,
        "depth_clip_m": depth_clip_m,
        "epsg": epsg,
        "full_grid_path": str(full_grid_path),
        "centers": centers,
        "normalization_metadata": normalization_metadata,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build per-site bathymetry patches")
    parser.add_argument("--sites", default="configs/sites.yaml", help="Path to sites.yaml")
    parser.add_argument(
        "--preprocess-config", default="configs/preprocess.yaml", help="Path to preprocess config"
    )
    parser.add_argument(
        "--target-sites-json", default=None, help="Optional JSON array of site names"
    )
    parser.add_argument(
        "--train-sites-json", default=None, help="Optional JSON array of training site names"
    )
    parser.add_argument("--out", required=True, help="Output point_centric_X_bathy.npz path")
    parser.add_argument(
        "--normalization-reference",
        default=None,
        help="Optional existing point_centric_X_bathy.npz whose normalization statistics will be reused",
    )
    args = parser.parse_args()

    preprocess_cfg = read_yaml(args.preprocess_config)
    sites_cfg = read_yaml(args.sites)
    target_sites = (
        json.loads(args.target_sites_json)
        if args.target_sites_json
        else [
            str(site.get("name"))
            for site in (sites_cfg.get("nearshore_sites") or [])
            if site.get("name")
        ]
    )
    train_sites = json.loads(args.train_sites_json) if args.train_sites_json else list(target_sites)
    info = build_bathymetry_patch_dataset(
        sites_yaml=args.sites,
        preprocess_cfg=preprocess_cfg,
        target_sites=target_sites,
        train_sites=train_sites,
        out_path=args.out,
        normalization_reference=args.normalization_reference,
    )
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
