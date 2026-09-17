#!/usr/bin/env python3
"""Directional ray-casting and static shoreline descriptors for nearshore sites."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
import yaml
from pyproj import CRS, Transformer
from scipy import ndimage
from scipy.spatial import cKDTree
from skimage.draw import disk


TARGET_EPSG = 32633
COMPASS_SECTORS = (
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
POROSITY_RADII_M = (
    500.0,
    1_000.0,
    2_000.0,
    5_000.0,
    10_000.0,
)


def porosity_column_name(radius_m: float) -> str:
    """Return canonical porosity column name for a search radius in meters."""
    radius_int = int(round(float(radius_m)))
    if radius_int >= 1_000:
        if radius_int % 1_000 == 0:
            return f"static_porosity_{radius_int // 1_000}km"
        return f"static_porosity_{radius_int / 1_000.0:g}km"
    return f"static_porosity_{radius_int}m"


def sector_feature_column_names() -> list[str]:
    """Return ordered directional feature columns for 16 compass sectors."""
    names: list[str] = []
    for sector in COMPASS_SECTORS:
        names.append(f"ray_fetch_{sector}_m")
    for sector in COMPASS_SECTORS:
        names.append(f"ray_max_slope_{sector}")
    for sector in COMPASS_SECTORS:
        names.append(f"ray_max_laplacian_{sector}")
    for sector in COMPASS_SECTORS:
        names.append(f"ray_min_depth_{sector}_m")
    return names


def resolve_project_root(start: Path) -> Path:
    """Use the selected working project or the editable project checkout."""
    import os

    explicit = os.environ.get("COASTAL_WAVE_PROJECT")
    if explicit:
        return Path(explicit).expanduser().resolve()
    for seed in (Path.cwd(), start.resolve()):
        for parent in (seed, *seed.parents):
            if (parent / "pyproject.toml").is_file():
                return parent
    raise FileNotFoundError("Set COASTAL_WAVE_PROJECT to the working project directory")


def resolve_from_root(project_root: Path, path_like: str) -> Path:
    """Resolve a path relative to cwd first, then project root, unless absolute."""
    path = Path(path_like)
    if path.is_absolute():
        return path
    cwd_path = Path.cwd() / path
    if cwd_path.exists():
        return cwd_path.resolve()
    return project_root / path


def load_nearshore_sites(sites_yaml: Path) -> pd.DataFrame:
    """Load nearshore site coordinates from YAML."""
    with sites_yaml.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}

    nearshore = pd.DataFrame(cfg.get("nearshore_sites", []))
    required_cols = {"name", "lat", "lon"}
    if nearshore.empty or not required_cols.issubset(nearshore.columns):
        raise ValueError("nearshore_sites in sites.yaml must include name/lat/lon")

    duplicate_nearshore = nearshore[nearshore["name"].duplicated(keep=False)]["name"].tolist()
    if duplicate_nearshore:
        duplicates = ", ".join(sorted(dict.fromkeys(str(name) for name in duplicate_nearshore)))
        raise ValueError(f"nearshore_sites contains duplicate site names: {duplicates}")

    return nearshore.copy()


def parse_npz_metadata(metadata_obj: Any) -> dict[str, Any]:
    """Normalize serialized metadata stored in NPZ payloads."""
    if isinstance(metadata_obj, np.ndarray):
        if metadata_obj.shape == () or metadata_obj.size == 1:
            parsed = metadata_obj.item()
            return parsed if isinstance(parsed, dict) else {}
        return {}
    return metadata_obj if isinstance(metadata_obj, dict) else {}


def load_bathy_grid(
    bathy_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, dict[str, Any]]:
    """Load x, y, z arrays from NPZ (preferred) or NetCDF bathymetry grid."""
    suffix = bathy_path.suffix.lower()
    if suffix == ".npz":
        with np.load(bathy_path, allow_pickle=True) as npz:
            missing = {key for key in ("x", "y", "z") if key not in npz}
            if missing:
                missing_txt = ", ".join(sorted(missing))
                raise KeyError(f"NPZ must contain x/y/z arrays, missing: {missing_txt}")

            x = np.asarray(npz["x"], dtype=np.float64)
            y = np.asarray(npz["y"], dtype=np.float64)
            z = np.asarray(npz["z"], dtype=np.float64)
            land_mask = np.asarray(npz["land_mask"], dtype=bool) if "land_mask" in npz else None
            attrs = parse_npz_metadata(npz["metadata"]) if "metadata" in npz else {}
    else:
        with xr.open_dataset(bathy_path) as ds:
            if "z" in ds.data_vars:
                z_da = ds["z"]
            elif "depth" in ds.data_vars:
                z_da = ds["depth"]
            else:
                raise KeyError("NetCDF must contain either a 'z' or 'depth' data variable")

            z = np.asarray(z_da.transpose("y", "x").to_numpy(), dtype=np.float64)
            x = np.asarray(z_da["x"].to_numpy(), dtype=np.float64)
            y = np.asarray(z_da["y"].to_numpy(), dtype=np.float64)
            land_mask = None
            attrs = dict(ds.attrs)

    if x.size > 1 and x[1] < x[0]:
        x = x[::-1]
        z = z[:, ::-1]
        if land_mask is not None:
            land_mask = land_mask[:, ::-1]
    if y.size > 1 and y[1] < y[0]:
        y = y[::-1]
        z = z[::-1, :]
        if land_mask is not None:
            land_mask = land_mask[::-1, :]

    return x, y, z, land_mask, attrs


def nearest_index(coord: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Return nearest coordinate index for each value using vectorized search."""
    if coord.size == 0:
        raise ValueError("coord is empty")
    if coord.size == 1:
        return np.zeros(values.shape, dtype=np.int64)

    idx = np.searchsorted(coord, values)
    idx = np.clip(idx, 1, coord.size - 1)
    left = coord[idx - 1]
    right = coord[idx]
    take_left = np.abs(values - left) <= np.abs(right - values)
    return idx - take_left.astype(np.int64)


def xy_to_rc(
    x_values: np.ndarray, y_values: np.ndarray, x_coord: np.ndarray, y_coord: np.ndarray
) -> np.ndarray:
    """Convert projected x/y coordinates to nearest row/col indices."""
    cols = nearest_index(x_coord, x_values)
    rows = nearest_index(y_coord, y_values)
    return np.column_stack((rows.astype(np.int64), cols.astype(np.int64)))


def snap_points_to_mask(
    points_rc: np.ndarray, valid_mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Snap points to nearest valid cell in a boolean mask."""
    valid_rc = np.argwhere(valid_mask)
    if valid_rc.size == 0:
        raise ValueError("No valid cells available for snapping")

    tree = cKDTree(valid_rc)
    distances, indices = tree.query(points_rc, k=1)
    snapped = valid_rc[indices]
    return snapped.astype(np.int64), distances.astype(np.float64)


def fill_nans_with_nearest(values: np.ndarray) -> np.ndarray:
    """Fill NaNs using nearest finite-neighbor assignment in index space."""
    finite = np.isfinite(values)
    if finite.all():
        return values.copy()
    if not finite.any():
        raise ValueError("Bathymetry grid contains no finite values")

    _, nearest_idx = ndimage.distance_transform_edt(
        ~finite,
        return_distances=True,
        return_indices=True,
    )
    filled = values[tuple(nearest_idx)]
    return filled.astype(np.float64)


def compute_base_matrices(
    z_grid: np.ndarray,
    dx: float,
    dy: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute filled elevation, gradients, slope magnitude, and Laplacian matrices."""
    z_filled = fill_nans_with_nearest(z_grid)

    dz_dy, dz_dx = np.gradient(z_filled, dy, dx, edge_order=2)
    slope = np.hypot(dz_dx, dz_dy)

    kernel_x = np.array([[0.0, 0.0, 0.0], [1.0, -2.0, 1.0], [0.0, 0.0, 0.0]], dtype=np.float64) / (
        dx * dx
    )
    kernel_y = np.array([[0.0, 1.0, 0.0], [0.0, -2.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64) / (
        dy * dy
    )
    laplacian = ndimage.convolve(z_filled, kernel_x, mode="nearest") + ndimage.convolve(
        z_filled,
        kernel_y,
        mode="nearest",
    )

    return z_filled, dz_dx, dz_dy, slope, laplacian


def to_compass_degrees(vec_x: np.ndarray | float, vec_y: np.ndarray | float) -> np.ndarray:
    """Convert Cartesian vectors (x east, y north) to compass degrees."""
    angle_math = np.degrees(np.arctan2(vec_y, vec_x))
    return np.mod(90.0 - angle_math, 360.0)


def angle_to_sector_labels(angle_deg: np.ndarray) -> np.ndarray:
    """Map compass angles, 0=N clockwise, to compass sector labels."""
    wrapped = np.mod(angle_deg.astype(np.float64), 360.0)
    sector_width = 360.0 / len(COMPASS_SECTORS)
    sector_idx = np.floor(np.mod(wrapped + sector_width / 2.0, 360.0) / sector_width).astype(
        np.int64
    )
    labels = np.asarray(COMPASS_SECTORS, dtype=object)
    return labels[sector_idx]


def build_sector_site_features(ray_df: pd.DataFrame) -> pd.DataFrame:
    """Build per-site directional summaries from per-ray records."""
    if ray_df.empty:
        return pd.DataFrame(columns=["site_name", *sector_feature_column_names()])

    work = ray_df[
        ["site_name", "angle_deg", "fetch_m", "max_slope", "max_laplacian", "min_depth_m"]
    ].copy()
    work["sector"] = angle_to_sector_labels(
        pd.to_numeric(work["angle_deg"], errors="coerce").to_numpy()
    )

    grouped = (
        work.groupby(["site_name", "sector"], sort=False)
        .agg(
            ray_fetch_m=("fetch_m", "mean"),
            ray_max_slope=("max_slope", "max"),
            ray_max_laplacian=("max_laplacian", "max"),
            ray_min_depth_m=("min_depth_m", "min"),
        )
        .reset_index()
    )

    sites = pd.Index(ray_df["site_name"].dropna().astype(str).unique(), name="site_name")
    sector_index = pd.Index(COMPASS_SECTORS, name="sector")

    fetch_wide = (
        grouped.pivot(index="site_name", columns="sector", values="ray_fetch_m")
        .reindex(index=sites, columns=sector_index)
        .rename(columns={sector: f"ray_fetch_{sector}_m" for sector in COMPASS_SECTORS})
    )
    slope_wide = (
        grouped.pivot(index="site_name", columns="sector", values="ray_max_slope")
        .reindex(index=sites, columns=sector_index)
        .rename(columns={sector: f"ray_max_slope_{sector}" for sector in COMPASS_SECTORS})
    )
    lap_wide = (
        grouped.pivot(index="site_name", columns="sector", values="ray_max_laplacian")
        .reindex(index=sites, columns=sector_index)
        .rename(columns={sector: f"ray_max_laplacian_{sector}" for sector in COMPASS_SECTORS})
    )
    depth_wide = (
        grouped.pivot(index="site_name", columns="sector", values="ray_min_depth_m")
        .reindex(index=sites, columns=sector_index)
        .rename(columns={sector: f"ray_min_depth_{sector}_m" for sector in COMPASS_SECTORS})
    )

    merged = (
        fetch_wide.join(slope_wide, how="outer")
        .join(lap_wide, how="outer")
        .join(depth_wide, how="outer")
        .reset_index()
    )
    for col in sector_feature_column_names():
        if col not in merged.columns:
            merged[col] = np.nan
    return merged[["site_name", *sector_feature_column_names()]]


def elevation_to_positive_depth(
    elevation_values: np.ndarray, vertical_convention: str
) -> np.ndarray:
    """Convert sampled water-column values to positive water depth in meters."""
    values = np.asarray(elevation_values, dtype=np.float64)
    if vertical_convention == "positive_down_depth":
        depth = values
    else:
        depth = -values
    depth = np.where(np.isfinite(depth), np.maximum(depth, 0.0), np.nan)
    return depth.astype(np.float64, copy=False)


def compute_static_site_features(
    snapped_site_rc: np.ndarray,
    land_mask: np.ndarray,
    z_grid: np.ndarray,
    z_filled: np.ndarray,
    dz_dx: np.ndarray,
    dz_dy: np.ndarray,
    dx: float,
    dy: float,
    vertical_convention: str,
    porosity_radii_m: tuple[float, ...] = POROSITY_RADII_M,
) -> dict[str, np.ndarray]:
    """Compute per-site static porosity and nearest-shoreline descriptors."""
    water_mask = ~land_mask
    dist_to_coast_m, nearest_land_idx = ndimage.distance_transform_edt(
        water_mask,
        sampling=(dy, dx),
        return_distances=True,
        return_indices=True,
    )

    site_rows = snapped_site_rc[:, 0].astype(np.int64, copy=False)
    site_cols = snapped_site_rc[:, 1].astype(np.int64, copy=False)

    nearest_land_rows = nearest_land_idx[0, site_rows, site_cols].astype(np.int64, copy=False)
    nearest_land_cols = nearest_land_idx[1, site_rows, site_cols].astype(np.int64, copy=False)

    nearest_dz_dx = dz_dx[nearest_land_rows, nearest_land_cols]
    nearest_dz_dy = dz_dy[nearest_land_rows, nearest_land_cols]
    nearest_steepness = np.hypot(nearest_dz_dx, nearest_dz_dy)
    nearest_normal_deg = to_compass_degrees(nearest_dz_dx, nearest_dz_dy)

    avg_spacing_m = float((abs(dx) + abs(dy)) * 0.5)
    if avg_spacing_m <= 0.0:
        avg_spacing_m = 1.0
    porosity_by_radius: dict[str, np.ndarray] = {
        porosity_column_name(radius_m): np.full(site_rows.shape[0], np.nan, dtype=np.float64)
        for radius_m in porosity_radii_m
    }
    radius_cells_by_key = {
        porosity_column_name(radius_m): max(1, int(np.round(float(radius_m) / avg_spacing_m)))
        for radius_m in porosity_radii_m
    }

    for idx, (row, col) in enumerate(zip(site_rows, site_cols)):
        for key, radius_cells in radius_cells_by_key.items():
            rr, cc = disk((int(row), int(col)), radius=radius_cells, shape=land_mask.shape)
            if rr.size:
                porosity_by_radius[key][idx] = float(np.mean(water_mask[rr, cc]))

    static_local_depth_m = z_grid[site_rows, site_cols].astype(np.float64, copy=False)
    missing_local_depth = ~np.isfinite(static_local_depth_m)
    if np.any(missing_local_depth):
        static_local_depth_m = static_local_depth_m.copy()
        static_local_depth_m[missing_local_depth] = z_filled[
            site_rows[missing_local_depth],
            site_cols[missing_local_depth],
        ]

    local_depth_m = elevation_to_positive_depth(static_local_depth_m, vertical_convention)
    site_on_land = land_mask[site_rows, site_cols].astype(bool, copy=False)
    local_breaking_cap_valid = (
        (~site_on_land) & np.isfinite(local_depth_m) & (local_depth_m > 0.0)
    ).astype(np.int64, copy=False)

    out = {
        "static_dist_to_coast_m": dist_to_coast_m[site_rows, site_cols].astype(
            np.float64, copy=False
        ),
        "static_local_depth_m": static_local_depth_m.astype(np.float64, copy=False),
        "local_depth_m": local_depth_m.astype(np.float64, copy=False),
        "local_breaking_cap_valid": local_breaking_cap_valid,
        "static_nearest_shore_steepness": nearest_steepness.astype(np.float64, copy=False),
        "static_nearest_shore_normal_deg": nearest_normal_deg.astype(np.float64, copy=False),
    }
    out.update(porosity_by_radius)
    return out


def infer_water_land_masks(z_grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, str]:
    """Infer water and land masks from bathymetry sign convention."""
    finite = np.isfinite(z_grid)
    finite_values = z_grid[finite]
    if finite_values.size == 0:
        raise ValueError("Bathymetry grid has no finite values")

    positive_ratio = float(np.mean(finite_values > 0.0))
    if positive_ratio > 0.9:
        land_mask = (~finite) | (z_grid <= 0.0)
        water_mask = ~land_mask
        return water_mask, land_mask, "positive_down_depth"

    land_mask = (~finite) | (z_grid >= 0.0)
    water_mask = ~land_mask
    return water_mask, land_mask, "signed_elevation"


def serialize_profile(values: np.ndarray, digits: int = 4) -> str:
    """Serialize a 1D array to compact JSON for CSV export."""
    return json.dumps(np.round(values.astype(np.float64), digits).tolist(), separators=(",", ":"))


def fractional_index_to_coord(coord: np.ndarray, fractional_idx: float) -> float:
    """Map a fractional index to coordinate value using 1D linear interpolation."""
    grid_idx = np.arange(coord.size, dtype=np.float64)
    return float(np.interp(fractional_idx, grid_idx, coord))


def cast_single_ray(
    site_name: str,
    site_row: int,
    site_col: int,
    site_x: float,
    site_y: float,
    angle_deg: float,
    sample_dist_m: np.ndarray,
    land_mask: np.ndarray,
    z_filled: np.ndarray,
    slope: np.ndarray,
    laplacian: np.ndarray,
    x_coord: np.ndarray,
    y_coord: np.ndarray,
    dx: float,
    dy: float,
    vertical_convention: str,
) -> dict[str, Any]:
    """Cast one directional ray and compute fetch + profile statistics."""
    theta = np.deg2rad(angle_deg)
    ray_rows = site_row + (sample_dist_m / dy) * np.cos(theta)
    ray_cols = site_col + (sample_dist_m / dx) * np.sin(theta)

    in_bounds = (
        (ray_rows >= 0.0)
        & (ray_rows <= (land_mask.shape[0] - 1))
        & (ray_cols >= 0.0)
        & (ray_cols <= (land_mask.shape[1] - 1))
    )
    if not np.any(in_bounds):
        return {
            "site_name": site_name,
            "site_x": float(site_x),
            "site_y": float(site_y),
            "site_row": int(site_row),
            "site_col": int(site_col),
            "angle_deg": float(angle_deg),
            "fetch_m": float("nan"),
            "hit_land": False,
            "endpoint_x": float(site_x),
            "endpoint_y": float(site_y),
            "n_samples_water": 0,
            "max_slope": float("nan"),
            "mean_slope": float("nan"),
            "mean_laplacian": float("nan"),
            "max_laplacian": float("nan"),
            "max_abs_laplacian": float("nan"),
            "min_elevation": float("nan"),
            "mean_elevation": float("nan"),
            "min_depth_m": float("nan"),
            "distance_profile_m": "[]",
            "elevation_profile_m": "[]",
            "gradient_profile": "[]",
            "laplacian_profile": "[]",
        }

    ray_rows = ray_rows[in_bounds]
    ray_cols = ray_cols[in_bounds]
    dist = sample_dist_m[in_bounds]

    land_hit = (
        ndimage.map_coordinates(
            land_mask.astype(np.float32),
            [ray_rows, ray_cols],
            order=0,
            mode="nearest",
        )
        >= 0.5
    )

    if np.any(land_hit):
        first_land_idx = int(np.argmax(land_hit))
        hit_land = True
        fetch_m = float(dist[first_land_idx])
        water_stop = max(first_land_idx, 1)
        endpoint_row = float(ray_rows[first_land_idx])
        endpoint_col = float(ray_cols[first_land_idx])
    else:
        hit_land = False
        fetch_m = float(dist[-1])
        water_stop = len(dist)
        endpoint_row = float(ray_rows[-1])
        endpoint_col = float(ray_cols[-1])

    water_rows = ray_rows[:water_stop]
    water_cols = ray_cols[:water_stop]
    water_dist = dist[:water_stop]

    elev_profile = ndimage.map_coordinates(
        z_filled,
        [water_rows, water_cols],
        order=1,
        mode="nearest",
    )
    slope_profile = ndimage.map_coordinates(
        slope,
        [water_rows, water_cols],
        order=1,
        mode="nearest",
    )
    lap_profile = ndimage.map_coordinates(
        laplacian,
        [water_rows, water_cols],
        order=1,
        mode="nearest",
    )
    depth_profile = elevation_to_positive_depth(elev_profile, vertical_convention)

    endpoint_x = fractional_index_to_coord(x_coord, endpoint_col)
    endpoint_y = fractional_index_to_coord(y_coord, endpoint_row)

    return {
        "site_name": site_name,
        "site_x": float(site_x),
        "site_y": float(site_y),
        "site_row": int(site_row),
        "site_col": int(site_col),
        "angle_deg": float(angle_deg),
        "fetch_m": float(fetch_m),
        "hit_land": bool(hit_land),
        "endpoint_x": float(endpoint_x),
        "endpoint_y": float(endpoint_y),
        "n_samples_water": int(water_dist.size),
        "max_slope": float(np.max(slope_profile)) if slope_profile.size else float("nan"),
        "mean_slope": float(np.mean(slope_profile)) if slope_profile.size else float("nan"),
        "mean_laplacian": float(np.mean(lap_profile)) if lap_profile.size else float("nan"),
        "max_laplacian": float(np.max(lap_profile)) if lap_profile.size else float("nan"),
        "max_abs_laplacian": float(np.max(np.abs(lap_profile)))
        if lap_profile.size
        else float("nan"),
        "min_elevation": float(np.min(elev_profile)) if elev_profile.size else float("nan"),
        "mean_elevation": float(np.mean(elev_profile)) if elev_profile.size else float("nan"),
        "min_depth_m": float(np.min(depth_profile)) if depth_profile.size else float("nan"),
        "distance_profile_m": serialize_profile(water_dist, digits=2),
        "elevation_profile_m": serialize_profile(elev_profile, digits=3),
        "gradient_profile": serialize_profile(slope_profile, digits=5),
        "laplacian_profile": serialize_profile(lap_profile, digits=6),
    }


def run_ray_casting(
    project_root: Path,
    sites_rel: str,
    bathy_rel: str,
    output_rel: str,
    target_epsg: int,
    angle_step_deg: float,
    ray_step_m: float,
    max_ray_m: float | None,
) -> Path:
    """Execute ray-casting feature extraction and save CSV."""
    sites_path = resolve_from_root(project_root, sites_rel)
    bathy_path = resolve_from_root(project_root, bathy_rel)
    output_path = resolve_from_root(project_root, output_rel)

    nearshore_df = load_nearshore_sites(sites_path)
    x_coord, y_coord, z_grid, land_mask_from_file, grid_attrs = load_bathy_grid(bathy_path)

    grid_epsg = grid_attrs.get("target_epsg", grid_attrs.get("epsg"))
    if grid_epsg is not None and int(grid_epsg) != int(target_epsg):
        raise ValueError(
            f"Bathymetry EPSG mismatch: expected EPSG:{target_epsg}, found EPSG:{int(grid_epsg)}"
        )

    transformer = Transformer.from_crs(
        CRS.from_epsg(4326), CRS.from_epsg(target_epsg), always_xy=True
    )
    nearshore_x, nearshore_y = transformer.transform(
        nearshore_df["lon"].to_numpy(dtype=np.float64),
        nearshore_df["lat"].to_numpy(dtype=np.float64),
    )
    nearshore_df = nearshore_df.assign(x=nearshore_x, y=nearshore_y)

    site_rc = xy_to_rc(nearshore_df["x"].to_numpy(), nearshore_df["y"].to_numpy(), x_coord, y_coord)

    if land_mask_from_file is not None:
        if land_mask_from_file.shape != z_grid.shape:
            raise ValueError(
                "land_mask shape mismatch: "
                f"expected {z_grid.shape}, found {land_mask_from_file.shape}"
            )
        land_mask = land_mask_from_file.astype(bool, copy=False)
        water_mask = ~land_mask
        vertical_convention = str(grid_attrs.get("vertical_convention", "from_land_mask"))
        land_mask_source = "npz_land_mask"
    else:
        water_mask, land_mask, vertical_convention = infer_water_land_masks(z_grid)
        land_mask_source = "inferred_from_z"
    snapped_site_rc, snap_idx = snap_points_to_mask(site_rc, water_mask)

    dx = float(np.median(np.diff(x_coord))) if x_coord.size > 1 else 1.0
    dy = float(np.median(np.diff(y_coord))) if y_coord.size > 1 else 1.0

    z_filled, dz_dx, dz_dy, slope, laplacian = compute_base_matrices(z_grid, dx=dx, dy=dy)
    site_static = compute_static_site_features(
        snapped_site_rc=snapped_site_rc,
        land_mask=land_mask,
        z_grid=z_grid,
        z_filled=z_filled,
        dz_dx=dz_dx,
        dz_dy=dz_dy,
        dx=dx,
        dy=dy,
        vertical_convention=vertical_convention,
        porosity_radii_m=POROSITY_RADII_M,
    )

    if max_ray_m is None:
        max_ray_m = float(np.hypot((x_coord.size - 1) * dx, (y_coord.size - 1) * dy))

    sample_dist_m = np.arange(0.0, max_ray_m + ray_step_m, ray_step_m, dtype=np.float64)
    angles = np.arange(0.0, 360.0, angle_step_deg, dtype=np.float64)

    records: list[dict[str, Any]] = []
    for idx, site in nearshore_df.reset_index(drop=True).iterrows():
        row = int(snapped_site_rc[idx, 0])
        col = int(snapped_site_rc[idx, 1])
        snapped_x = float(x_coord[col])
        snapped_y = float(y_coord[row])

        for angle in angles:
            record = cast_single_ray(
                site_name=str(site["name"]),
                site_row=row,
                site_col=col,
                site_x=snapped_x,
                site_y=snapped_y,
                angle_deg=float(angle),
                sample_dist_m=sample_dist_m,
                land_mask=land_mask,
                z_filled=z_filled,
                slope=slope,
                laplacian=laplacian,
                x_coord=x_coord,
                y_coord=y_coord,
                dx=dx,
                dy=dy,
                vertical_convention=vertical_convention,
            )
            record["site_lat"] = float(site["lat"])
            record["site_lon"] = float(site["lon"])
            record["site_snap_distance_m"] = float(snap_idx[idx] * np.hypot(dx, dy))
            for porosity_key, porosity_values in site_static.items():
                if porosity_key.startswith("static_porosity_"):
                    record[porosity_key] = float(porosity_values[idx])
            record["static_dist_to_coast_m"] = float(site_static["static_dist_to_coast_m"][idx])
            record["static_local_depth_m"] = float(site_static["static_local_depth_m"][idx])
            record["local_depth_m"] = float(site_static["local_depth_m"][idx])
            record["local_breaking_cap_valid"] = int(site_static["local_breaking_cap_valid"][idx])
            record["static_nearest_shore_steepness"] = float(
                site_static["static_nearest_shore_steepness"][idx]
            )
            record["static_nearest_shore_normal_deg"] = float(
                site_static["static_nearest_shore_normal_deg"][idx]
            )
            records.append(record)

    ray_df = pd.DataFrame(records).sort_values(["site_name", "angle_deg"]).reset_index(drop=True)
    sector_site_df = build_sector_site_features(ray_df)
    if not sector_site_df.empty:
        ray_df = ray_df.merge(sector_site_df, on="site_name", how="left", validate="many_to_one")
    else:
        for col in sector_feature_column_names():
            ray_df[col] = np.nan

    ray_df.insert(0, "land_mask_source", land_mask_source)
    ray_df.insert(0, "vertical_convention", vertical_convention)
    ray_df.insert(0, "target_epsg", int(target_epsg))

    if ray_df.columns.duplicated().any():
        duplicates = ray_df.columns[ray_df.columns.duplicated()].tolist()
        raise ValueError(f"Duplicate ray feature columns detected: {duplicates}")

    min_depth_cols = [
        c for c in ray_df.columns if c.startswith("ray_min_depth_") and c.endswith("_m")
    ]
    for col in min_depth_cols:
        values = pd.to_numeric(ray_df[col], errors="coerce").to_numpy(dtype=np.float64, copy=False)
        if np.isfinite(values).any() and float(np.nanmin(values)) < 0.0:
            raise ValueError(f"Non-physical negative ray minimum depth detected in column '{col}'")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ray_df.to_csv(output_path, index=False)

    print(f"Saved ray features: {output_path}")
    print(f"Rows exported: {len(ray_df)}")
    print(f"Ray angles per site: {len(angles)}")
    print(
        "Fetch range (m): "
        f"{ray_df['fetch_m'].min(skipna=True):.2f} to {ray_df['fetch_m'].max(skipna=True):.2f}"
    )
    return output_path


def parse_args() -> argparse.Namespace:
    """Build CLI parser for standalone script execution."""
    parser = argparse.ArgumentParser(description="Directional ray-casting geometry features")
    parser.add_argument("--sites", default="configs/sites.yaml", help="Path to sites YAML")
    parser.add_argument(
        "--bathy",
        "--bathy-nc",
        dest="bathy",
        default="data/processed/bathy/bathy_field_project_site.npz",
        help="Path to structured bathymetry file (NPZ preferred; NetCDF also supported)",
    )
    parser.add_argument(
        "--output",
        default="data/processed/ray_features.csv",
        help="Path to output ray feature CSV",
    )
    parser.add_argument(
        "--epsg",
        type=int,
        default=TARGET_EPSG,
        help="Projected EPSG expected for geometry extraction",
    )
    parser.add_argument(
        "--angle-step-deg",
        type=float,
        default=5.625,
        help="Angular step in degrees for starburst rays",
    )
    parser.add_argument(
        "--ray-step-m",
        type=float,
        default=50.0,
        help="Sampling interval along each ray (meters)",
    )
    parser.add_argument(
        "--max-ray-m",
        type=float,
        default=None,
        help="Optional max ray distance (meters). Defaults to full-grid diagonal.",
    )
    return parser.parse_args()


def main() -> None:
    """Run directional ray-casting end-to-end."""
    args = parse_args()
    project_root = resolve_project_root(Path(__file__).resolve())
    run_ray_casting(
        project_root=project_root,
        sites_rel=args.sites,
        bathy_rel=args.bathy,
        output_rel=args.output,
        target_epsg=args.epsg,
        angle_step_deg=args.angle_step_deg,
        ray_step_m=args.ray_step_m,
        max_ray_m=args.max_ray_m,
    )


if __name__ == "__main__":
    main()
