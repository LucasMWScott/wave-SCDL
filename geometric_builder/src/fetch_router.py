#!/usr/bin/env python3
"""Boundary-curtain routing for coastal wave downscaling features."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
import yaml
from pyproj import CRS, Transformer
from scipy import ndimage
from scipy.spatial import cKDTree
from skimage.draw import line_nd
from skimage.graph import MCP_Geometric
from skimage.measure import approximate_polygon


TARGET_EPSG = 32633
BASE_WATER_COST = 1.0
BOTTLENECK_WEIGHT = 500.0
BOTTLENECK_EPSILON_M = 1.0
PATH_SMOOTH_TOLERANCE_CELLS = 1.5


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


def format_path_for_metadata(path: Path, project_root: Path) -> str:
    """Return a stable display path relative to project root when possible."""
    try:
        return str(path.relative_to(project_root))
    except ValueError:
        return str(path)


def load_sites_yaml(sites_yaml: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load offshore and nearshore sites from YAML."""
    with sites_yaml.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}

    offshore = pd.DataFrame(cfg.get("offshore_sites", []))
    nearshore = pd.DataFrame(cfg.get("nearshore_sites", []))

    required_cols = {"name", "lat", "lon"}
    if offshore.empty or not required_cols.issubset(offshore.columns):
        raise ValueError("offshore_sites in sites.yaml must include name/lat/lon")
    if nearshore.empty or not required_cols.issubset(nearshore.columns):
        raise ValueError("nearshore_sites in sites.yaml must include name/lat/lon")

    duplicate_offshore = offshore[offshore["name"].duplicated(keep=False)]["name"].tolist()
    if duplicate_offshore:
        duplicates = ", ".join(sorted(dict.fromkeys(str(name) for name in duplicate_offshore)))
        raise ValueError(f"offshore_sites contains duplicate site names: {duplicates}")

    duplicate_nearshore = nearshore[nearshore["name"].duplicated(keep=False)]["name"].tolist()
    if duplicate_nearshore:
        duplicates = ", ".join(sorted(dict.fromkeys(str(name) for name in duplicate_nearshore)))
        raise ValueError(f"nearshore_sites contains duplicate site names: {duplicates}")

    return offshore.copy(), nearshore.copy()


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

            if "x" not in z_da.coords or "y" not in z_da.coords:
                raise KeyError("Bathymetry variable must carry x/y coordinates")

            z = np.asarray(z_da.transpose("y", "x").to_numpy(), dtype=np.float64)
            x = np.asarray(z_da["x"].to_numpy(), dtype=np.float64)
            y = np.asarray(z_da["y"].to_numpy(), dtype=np.float64)
            land_mask = None
            attrs = dict(ds.attrs)

    # Ensure monotonically increasing coordinates for index<->metric conversion.
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
    if coord.ndim != 1:
        raise ValueError("coord must be 1D")
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
    """Convert projected coordinates to nearest raster row/col indices."""
    cols = nearest_index(x_coord, x_values)
    rows = nearest_index(y_coord, y_values)
    return np.column_stack((rows.astype(np.int64), cols.astype(np.int64)))


def build_curtain(
    offshore_rc: np.ndarray, grid_shape: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    """Build a straight linear offshore curtain between endpoint offshore points."""
    if offshore_rc.shape[0] < 2:
        raise ValueError("At least two offshore points are required to build a curtain")

    start = offshore_rc[0].astype(np.int64)
    end = offshore_rc[-1].astype(np.int64)

    if np.array_equal(start, end):
        delta = offshore_rc[:, None, :] - offshore_rc[None, :, :]
        dist2 = np.sum(delta.astype(np.float64) ** 2, axis=2)
        i, j = np.unravel_index(np.argmax(dist2), dist2.shape)
        start = offshore_rc[int(i)].astype(np.int64)
        end = offshore_rc[int(j)].astype(np.int64)
        if np.array_equal(start, end):
            raise ValueError(
                "Offshore points collapse to one grid cell; cannot build linear curtain"
            )

    curtain_rc = np.vstack(line_nd(tuple(start), tuple(end), endpoint=True)).T
    in_bounds = (
        (curtain_rc[:, 0] >= 0)
        & (curtain_rc[:, 0] < grid_shape[0])
        & (curtain_rc[:, 1] >= 0)
        & (curtain_rc[:, 1] < grid_shape[1])
    )
    curtain_rc = curtain_rc[in_bounds]

    # Keep first occurrence order while removing duplicates.
    _, first_idx = np.unique(curtain_rc, axis=0, return_index=True)
    curtain_rc = curtain_rc[np.sort(first_idx)]

    curtain_mask = np.zeros(grid_shape, dtype=bool)
    curtain_mask[curtain_rc[:, 0], curtain_rc[:, 1]] = True
    return curtain_rc, curtain_mask


def mask_ghost_offshore_water(
    water_mask: np.ndarray,
    curtain_mask: np.ndarray,
    x_coord: np.ndarray,
    y_coord: np.ndarray,
    offshore_xy: np.ndarray,
    nearshore_xy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Mask offshore-side water using a curtain-aligned geometric half-plane.

    We estimate the dominant offshore curtain direction with PCA/SVD over
    offshore points, construct the normal direction, and remove water on the
    opposite side of the line from the nearshore centroid.
    """
    if water_mask.shape != curtain_mask.shape:
        raise ValueError("water_mask and curtain_mask must have identical shape")
    if offshore_xy.shape[0] < 2:
        raise ValueError("At least two offshore points are required for curtain-side masking")

    center = offshore_xy.mean(axis=0)
    centered = offshore_xy - center
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    tangent = vh[0]
    normal = np.array([-tangent[1], tangent[0]], dtype=np.float64)

    nearshore_center = nearshore_xy.mean(axis=0)
    nearshore_sign = float(np.sign(np.dot(nearshore_center - center, normal)))
    if nearshore_sign == 0.0:
        nearshore_sign = 1.0

    xx, yy = np.meshgrid(x_coord, y_coord)
    signed = (xx - center[0]) * normal[0] + (yy - center[1]) * normal[1]
    offshore_side = (signed * nearshore_sign) < 0.0

    offshore_ghost = water_mask & offshore_side & (~curtain_mask)
    routing_water = water_mask & (~offshore_ghost)
    routing_water |= curtain_mask & water_mask
    return routing_water, offshore_ghost


def snap_points_to_mask(
    points_rc: np.ndarray, valid_mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Snap points to the nearest valid mask cell (row/col index space)."""
    valid_rc = np.argwhere(valid_mask)
    if valid_rc.size == 0:
        raise ValueError("No valid cells available for snapping")

    tree = cKDTree(valid_rc)
    distances, indices = tree.query(points_rc, k=1)
    snapped = valid_rc[indices]
    return snapped.astype(np.int64), distances.astype(np.float64)


def path_length_m(path_rc: np.ndarray, dx: float, dy: float) -> float:
    """Compute polyline length in meters from row/col path."""
    if path_rc.shape[0] < 2:
        return 0.0
    d_rc = np.diff(path_rc.astype(np.float64), axis=0)
    segment_m = np.hypot(d_rc[:, 0] * dy, d_rc[:, 1] * dx)
    return float(segment_m.sum())


def rc_path_to_xy(path_rc: np.ndarray, x_coord: np.ndarray, y_coord: np.ndarray) -> np.ndarray:
    """Map row/col path to projected x/y polyline coordinates."""
    return np.column_stack((x_coord[path_rc[:, 1]], y_coord[path_rc[:, 0]])).astype(np.float64)


def curtain_rc_to_xy(
    curtain_rc: np.ndarray, x_coord: np.ndarray, y_coord: np.ndarray
) -> np.ndarray:
    """Map curtain row/col vertices to projected x/y coordinates."""
    if curtain_rc.size == 0:
        return np.zeros((0, 2), dtype=np.float64)
    return np.column_stack((x_coord[curtain_rc[:, 1]], y_coord[curtain_rc[:, 0]])).astype(
        np.float64
    )


def polyline_direct_distance_m(path_xy: np.ndarray) -> float:
    """Return straight-line endpoint distance for a path polyline."""
    if path_xy.ndim != 2 or path_xy.shape[0] < 2 or path_xy.shape[1] != 2:
        return float("nan")
    start_xy = path_xy[0]
    end_xy = path_xy[-1]
    if not (np.all(np.isfinite(start_xy)) and np.all(np.isfinite(end_xy))):
        return float("nan")
    return float(np.hypot(end_xy[0] - start_xy[0], end_xy[1] - start_xy[1]))


def safe_tortuosity_ratio(length_m: float, direct_distance_m: float) -> float:
    """Return `length/direct_distance` when finite and valid."""
    if not np.isfinite(length_m) or not np.isfinite(direct_distance_m) or direct_distance_m <= 0.0:
        return float("nan")
    return float(length_m / direct_distance_m)


def to_compass_degrees(delta_x: float, delta_y: float) -> float:
    """Convert Cartesian vector components to compass bearing in degrees."""
    angle_math = np.degrees(np.arctan2(delta_y, delta_x))
    return float(np.mod(90.0 - angle_math, 360.0))


def orient_path_rc_toward_site(path_rc: np.ndarray, site_rc: tuple[int, int]) -> np.ndarray:
    """Orient a path so its final vertex is nearest the nearshore site index."""
    if path_rc.shape[0] < 2:
        return path_rc.astype(np.int64, copy=True)

    site = np.asarray(site_rc, dtype=np.float64)
    start_dist = np.hypot(path_rc[0, 0] - site[0], path_rc[0, 1] - site[1])
    end_dist = np.hypot(path_rc[-1, 0] - site[0], path_rc[-1, 1] - site[1])
    if end_dist <= start_dist:
        return path_rc.astype(np.int64, copy=True)
    return path_rc[::-1].astype(np.int64, copy=True)


def smooth_path_xy(path_xy: np.ndarray, tolerance_m: float) -> np.ndarray:
    """Simplify raster-staircase path geometry while preserving endpoints."""
    if path_xy.shape[0] < 3 or tolerance_m <= 0.0:
        return path_xy.astype(np.float64, copy=True)

    smoothed = np.asarray(
        approximate_polygon(path_xy, tolerance=float(tolerance_m)), dtype=np.float64
    )
    if smoothed.ndim != 2 or smoothed.shape[1] != 2 or smoothed.shape[0] < 2:
        return path_xy.astype(np.float64, copy=True)

    smoothed[0] = path_xy[0]
    smoothed[-1] = path_xy[-1]

    delta = np.diff(smoothed, axis=0)
    keep = np.concatenate(([True], np.any(np.abs(delta) > 0.0, axis=1)))
    smoothed = smoothed[keep]
    if smoothed.shape[0] < 2:
        return path_xy.astype(np.float64, copy=True)
    return smoothed


def terminal_segment_start(path_xy: np.ndarray, final_segment_m: float) -> np.ndarray:
    """Get the start point of the last ``final_segment_m`` segment ending at the coast."""
    if path_xy.shape[0] < 2:
        return path_xy[0].astype(np.float64, copy=True)

    seg_xy = np.diff(path_xy, axis=0)
    seg_len = np.hypot(seg_xy[:, 0], seg_xy[:, 1])
    total_m = float(seg_len.sum())
    if total_m <= 0.0:
        return path_xy[0].astype(np.float64, copy=True)

    remaining = float(min(final_segment_m, total_m))
    for seg_idx in range(path_xy.shape[0] - 2, -1, -1):
        p0 = path_xy[seg_idx]
        p1 = path_xy[seg_idx + 1]
        vec = p1 - p0
        seg_m = float(np.hypot(vec[0], vec[1]))
        if seg_m <= 0.0:
            continue
        if remaining <= seg_m:
            alpha = (seg_m - remaining) / seg_m
            return p0 + alpha * vec
        remaining -= seg_m

    return path_xy[0].astype(np.float64, copy=True)


def compute_path_width_features(
    path_rc: np.ndarray,
    site_rc: tuple[int, int],
    distance_to_coast_m: np.ndarray,
) -> tuple[float, float, float]:
    """Compute bottleneck and width-ratio features along a path."""
    if path_rc.shape[0] == 0:
        return float("nan"), float("nan"), float("nan")

    path_width_m = 2.0 * distance_to_coast_m[path_rc[:, 0], path_rc[:, 1]]
    if path_width_m.size == 0:
        return float("nan"), float("nan"), float("nan")

    local_site_width_m = float(2.0 * distance_to_coast_m[int(site_rc[0]), int(site_rc[1])])
    path_bottleneck_m = float(np.min(path_width_m))
    path_max_width_m = float(np.max(path_width_m))

    if local_site_width_m <= 0.0:
        return path_bottleneck_m, float("nan"), float("nan")

    funneling_ratio = float(path_max_width_m / local_site_width_m)
    choke_out_ratio = float(path_bottleneck_m / local_site_width_m)
    return path_bottleneck_m, funneling_ratio, choke_out_ratio


def compute_path_angle_features(
    path_xy: np.ndarray,
    final_segment_m: float = 500.0,
) -> tuple[float, float, float, float]:
    """Compute path-angle descriptors from an offshore->nearshore polyline.

    Let ``theta_i`` be the heading angle (radians) of segment ``i`` along the
    smoothed path. Wrapped turn increments are

    ``delta_i = wrap(theta_i - theta_{i-1})``

    where wrapping is implemented via ``atan2(sin(.), cos(.))`` to keep
    increments in ``[-pi, pi]``. We export:

    - ``static_tortuosity_sum`` = ``sum_i |delta_i|`` (degrees)
    - ``static_signed_curvature_deg`` = ``sum_i delta_i`` (degrees)
    - net and final-approach compass bearings (degrees)
    """
    if path_xy.shape[0] < 2:
        return float("nan"), float("nan"), float("nan"), float("nan")

    seg_xy = np.diff(path_xy, axis=0)
    seg_len = np.hypot(seg_xy[:, 0], seg_xy[:, 1])

    nonzero = seg_len > 0.0
    if np.any(nonzero):
        seg_bearing_math = np.arctan2(seg_xy[nonzero, 1], seg_xy[nonzero, 0])
        if seg_bearing_math.size >= 2:
            turn = np.arctan2(
                np.sin(np.diff(seg_bearing_math)),
                np.cos(np.diff(seg_bearing_math)),
            )
            turn_deg = np.degrees(turn)
            tortuosity_sum_deg = float(np.abs(turn_deg).sum())
            signed_curvature_deg = float(turn_deg.sum())
        else:
            tortuosity_sum_deg = 0.0
            signed_curvature_deg = 0.0
    else:
        tortuosity_sum_deg = 0.0
        signed_curvature_deg = 0.0

    net_dx = float(path_xy[-1, 0] - path_xy[0, 0])
    net_dy = float(path_xy[-1, 1] - path_xy[0, 1])
    if net_dx == 0.0 and net_dy == 0.0:
        net_deflection_deg = float("nan")
    else:
        net_deflection_deg = to_compass_degrees(net_dx, net_dy)

    if float(seg_len.sum()) <= 0.0:
        final_approach_deg = float("nan")
    else:
        start_point = terminal_segment_start(path_xy=path_xy, final_segment_m=final_segment_m)
        final_dx = float(path_xy[-1, 0] - start_point[0])
        final_dy = float(path_xy[-1, 1] - start_point[1])
        if final_dx == 0.0 and final_dy == 0.0:
            final_approach_deg = float("nan")
        else:
            final_approach_deg = to_compass_degrees(final_dx, final_dy)

    return tortuosity_sum_deg, signed_curvature_deg, net_deflection_deg, final_approach_deg


def _route_record_from_solution(
    *,
    site: pd.Series,
    nearshore_rc: np.ndarray,
    nearshore_snapped_rc: np.ndarray,
    idx: int,
    reachable: bool,
    path_rc: np.ndarray | None,
    path_xy: np.ndarray | None,
    path_xy_smooth: np.ndarray | None,
    dx: float,
    dy: float,
    distance_to_coast_m: np.ndarray,
) -> dict[str, Any]:
    if reachable and path_rc is not None and path_xy is not None and path_xy_smooth is not None:
        length_m = path_length_m(path_rc, dx=dx, dy=dy)
        path_bottleneck_m, funneling_ratio, choke_out_ratio = compute_path_width_features(
            path_rc=path_rc,
            site_rc=tuple(int(v) for v in nearshore_snapped_rc[idx]),
            distance_to_coast_m=distance_to_coast_m,
        )
        (
            static_tortuosity_sum,
            static_signed_curvature_deg,
            static_net_deflection_deg,
            static_final_approach_deg,
        ) = compute_path_angle_features(path_xy=path_xy_smooth, final_segment_m=500.0)
        path_rc_list = [tuple(int(v) for v in rc) for rc in path_rc.tolist()]
        path_xy_list = [tuple(float(v) for v in xy) for xy in path_xy.tolist()]
        direct_distance_m = polyline_direct_distance_m(path_xy)
        tortuosity_ratio = safe_tortuosity_ratio(length_m, direct_distance_m)
    else:
        length_m = float("nan")
        path_bottleneck_m = float("nan")
        funneling_ratio = float("nan")
        choke_out_ratio = float("nan")
        static_tortuosity_sum = float("nan")
        static_signed_curvature_deg = float("nan")
        static_net_deflection_deg = float("nan")
        static_final_approach_deg = float("nan")
        path_rc_list = []
        path_xy_list = []
        direct_distance_m = float("nan")
        tortuosity_ratio = float("nan")

    return {
        "site_name": str(site["name"]),
        "site_lat": float(site["lat"]),
        "site_lon": float(site["lon"]),
        "site_x": float(site["x"]),
        "site_y": float(site["y"]),
        "site_row": int(nearshore_snapped_rc[idx, 0]),
        "site_col": int(nearshore_snapped_rc[idx, 1]),
        "path_length_m": float(length_m),
        "path_point_count": int(len(path_rc_list)),
        "snap_distance_m": float(
            np.hypot(
                (nearshore_snapped_rc[idx, 0] - nearshore_rc[idx, 0]) * dy,
                (nearshore_snapped_rc[idx, 1] - nearshore_rc[idx, 1]) * dx,
            )
        ),
        "reachable": bool(reachable),
        "path_bottleneck_m": float(path_bottleneck_m),
        "funneling_ratio": float(funneling_ratio),
        "choke_out_ratio": float(choke_out_ratio),
        "static_tortuosity_sum": float(static_tortuosity_sum),
        "static_signed_curvature_deg": float(static_signed_curvature_deg),
        "static_net_deflection_deg": float(static_net_deflection_deg),
        "static_final_approach_deg": float(static_final_approach_deg),
        "path_rc": path_rc_list,
        "path_xy": path_xy_list,
        "route_direct_distance_m": float(direct_distance_m),
        "route_tortuosity_ratio": float(tortuosity_ratio),
    }


def _build_route_qa_record(
    *,
    route_record: dict[str, Any],
    route_mode: str,
    k_sources: int,
    selected_sources: list[dict[str, Any]] | None,
    curtain_xy: np.ndarray,
    curtain_length_m: float,
) -> dict[str, Any]:
    qa_record = {
        "site_name": str(route_record.get("site_name", "")),
        "site_lon": float(route_record.get("site_lon", np.nan)),
        "site_lat": float(route_record.get("site_lat", np.nan)),
        "route_mode": str(route_mode),
        "k_sources": int(k_sources),
        "local_curtain_start_x": float(curtain_xy[0, 0])
        if curtain_xy.shape[0] >= 1
        else float("nan"),
        "local_curtain_start_y": float(curtain_xy[0, 1])
        if curtain_xy.shape[0] >= 1
        else float("nan"),
        "local_curtain_end_x": float(curtain_xy[-1, 0])
        if curtain_xy.shape[0] >= 1
        else float("nan"),
        "local_curtain_end_y": float(curtain_xy[-1, 1])
        if curtain_xy.shape[0] >= 1
        else float("nan"),
        "local_curtain_length_m": float(curtain_length_m),
        "route_valid": bool(route_record.get("reachable", False)),
        "route_length_m": float(route_record.get("path_length_m", np.nan)),
        "route_direct_distance_m": float(route_record.get("route_direct_distance_m", np.nan)),
        "route_tortuosity_ratio": float(route_record.get("route_tortuosity_ratio", np.nan)),
        "snap_distance_m": float(route_record.get("snap_distance_m", np.nan)),
        "path_bottleneck_m": float(route_record.get("path_bottleneck_m", np.nan)),
        "funneling_ratio": float(route_record.get("funneling_ratio", np.nan)),
        "static_final_approach_deg": float(route_record.get("static_final_approach_deg", np.nan)),
        "static_net_deflection_deg": float(route_record.get("static_net_deflection_deg", np.nan)),
        "static_signed_curvature_deg": float(
            route_record.get("static_signed_curvature_deg", np.nan)
        ),
    }
    max_sources = max(3, int(k_sources))
    chosen = list(selected_sources or [])
    for rank_idx in range(max_sources):
        if rank_idx < len(chosen):
            item = chosen[rank_idx]
            qa_record[f"selected_source_{rank_idx}"] = str(item.get("name", ""))
            qa_record[f"source{rank_idx}_distance_m"] = float(item.get("distance_m", np.nan))
            qa_record[f"source{rank_idx}_bearing_deg"] = float(item.get("bearing_deg", np.nan))
        else:
            qa_record[f"selected_source_{rank_idx}"] = ""
            qa_record[f"source{rank_idx}_distance_m"] = float("nan")
            qa_record[f"source{rank_idx}_bearing_deg"] = float("nan")
    return qa_record


def _route_single_site_with_local_curtain(
    *,
    site: pd.Series,
    nearshore_site_rc: np.ndarray,
    x_coord: np.ndarray,
    y_coord: np.ndarray,
    water_mask: np.ndarray,
    land_mask: np.ndarray,
    dx: float,
    dy: float,
    path_smoothing_tolerance_m: float,
    bottleneck_weight: float,
    bottleneck_epsilon_m: float,
    curtain_rc: np.ndarray,
    curtain_mask: np.ndarray,
    offshore_xy_local: np.ndarray,
) -> dict[str, Any]:
    site_xy = np.asarray([[float(site["x"]), float(site["y"])]], dtype=np.float64)
    routing_water, offshore_ghost = mask_ghost_offshore_water(
        water_mask=water_mask,
        curtain_mask=curtain_mask,
        x_coord=x_coord,
        y_coord=y_coord,
        offshore_xy=offshore_xy_local,
        nearshore_xy=site_xy,
    )
    if not np.any(routing_water):
        routing_water = water_mask.copy()
        offshore_ghost = np.zeros_like(water_mask)

    curtain_start_mask = curtain_mask & routing_water
    if not np.any(curtain_start_mask):
        snapped_starts, _ = snap_points_to_mask(curtain_rc, routing_water)
        curtain_start_mask = np.zeros_like(routing_water)
        curtain_start_mask[snapped_starts[:, 0], snapped_starts[:, 1]] = True

    original_site_rc = np.asarray([nearshore_site_rc], dtype=np.int64)
    nearshore_snapped_rc, _ = snap_points_to_mask(original_site_rc, routing_water)

    cc_labels, _ = ndimage.label(routing_water, structure=np.ones((3, 3), dtype=np.int8))
    start_labels = np.unique(cc_labels[curtain_start_mask])
    nearshore_labels = np.unique(cc_labels[nearshore_snapped_rc[:, 0], nearshore_snapped_rc[:, 1]])
    shared_labels = np.intersect1d(
        start_labels[start_labels > 0], nearshore_labels[nearshore_labels > 0]
    )
    if shared_labels.size == 0:
        routing_water = water_mask.copy()
        offshore_ghost = np.zeros_like(water_mask)
        curtain_start_mask = curtain_mask & routing_water
        if not np.any(curtain_start_mask):
            snapped_starts, _ = snap_points_to_mask(curtain_rc, routing_water)
            curtain_start_mask = np.zeros_like(routing_water)
            curtain_start_mask[snapped_starts[:, 0], snapped_starts[:, 1]] = True
        nearshore_snapped_rc, _ = snap_points_to_mask(original_site_rc, routing_water)

    cost_matrix, distance_to_coast_m = build_dynamic_cost_matrix(
        routing_water=routing_water,
        land_mask=land_mask,
        dy=dy,
        dx=dx,
        bottleneck_weight=bottleneck_weight,
        bottleneck_epsilon_m=bottleneck_epsilon_m,
        base_water_cost=BASE_WATER_COST,
    )
    mcp = MCP_Geometric(cost_matrix, sampling=(dy, dx), fully_connected=True)
    starts = [tuple(rc) for rc in np.argwhere(curtain_start_mask)]
    cumulative_costs, _ = mcp.find_costs(starts=starts)
    target_rc = tuple(int(v) for v in nearshore_snapped_rc[0])
    reachable = np.isfinite(cumulative_costs[target_rc])

    path_rc: np.ndarray | None = None
    path_xy: np.ndarray | None = None
    path_xy_smooth: np.ndarray | None = None
    if reachable:
        path_rc = np.asarray(mcp.traceback(target_rc), dtype=np.int64)
        path_rc = orient_path_rc_toward_site(path_rc=path_rc, site_rc=target_rc)
        path_xy = rc_path_to_xy(path_rc, x_coord, y_coord)
        path_xy_smooth = smooth_path_xy(path_xy=path_xy, tolerance_m=path_smoothing_tolerance_m)

    return _route_record_from_solution(
        site=site,
        nearshore_rc=original_site_rc,
        nearshore_snapped_rc=nearshore_snapped_rc,
        idx=0,
        reachable=bool(reachable),
        path_rc=path_rc,
        path_xy=path_xy,
        path_xy_smooth=path_xy_smooth,
        dx=dx,
        dy=dy,
        distance_to_coast_m=distance_to_coast_m,
    )


def infer_water_land_masks(z_grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, str]:
    """Infer water/land masks from bathymetry sign convention.

    Land is defined by thresholding finite cells (z <= 0 for positive-down
    depth, z >= 0 for signed elevation), with NaNs always treated as land.
    """
    finite_mask = np.isfinite(z_grid)
    finite_values = z_grid[finite_mask]
    if finite_values.size == 0:
        raise ValueError("Bathymetry grid has no finite values")

    positive_ratio = float(np.mean(finite_values > 0.0))
    if positive_ratio > 0.9:
        land_mask = (~finite_mask) | (z_grid <= 0.0)
        water_mask = ~land_mask
        return water_mask, land_mask, "positive_down_depth"

    land_mask = (~finite_mask) | (z_grid >= 0.0)
    water_mask = ~land_mask
    return water_mask, land_mask, "signed_elevation"


def build_dynamic_cost_matrix(
    routing_water: np.ndarray,
    land_mask: np.ndarray,
    dy: float,
    dx: float,
    bottleneck_weight: float,
    bottleneck_epsilon_m: float,
    base_water_cost: float = BASE_WATER_COST,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a bottleneck-aware routing cost surface.

    Distance-to-coast is computed from the water domain defined by ``land_mask``.
    Routing remains constrained by ``routing_water`` (land + offshore ghost water
    stay at infinite cost).
    """
    if bottleneck_weight < 0.0:
        raise ValueError("bottleneck_weight must be >= 0")
    if bottleneck_epsilon_m <= 0.0:
        raise ValueError("bottleneck_epsilon_m must be > 0")
    if base_water_cost <= 0.0:
        raise ValueError("base_water_cost must be > 0")

    water_mask = ~land_mask
    distance_to_coast_m = ndimage.distance_transform_edt(water_mask, sampling=(dy, dx))

    cost_matrix = np.full(routing_water.shape, np.inf, dtype=np.float64)
    water_penalty = bottleneck_weight / (distance_to_coast_m[routing_water] + bottleneck_epsilon_m)
    cost_matrix[routing_water] = base_water_cost + water_penalty
    return cost_matrix, distance_to_coast_m


def run_fetch_routing(
    project_root: Path,
    sites_rel: str,
    bathy_rel: str,
    output_rel: str,
    target_epsg: int,
    bottleneck_weight: float,
    bottleneck_epsilon_m: float,
    route_mode: str = "global",
    route_source_metadata: dict[str, Any] | None = None,
) -> Path:
    """Execute boundary-curtain routing pipeline and export feature pickle."""
    sites_path = resolve_from_root(project_root, sites_rel)
    bathy_path = resolve_from_root(project_root, bathy_rel)
    output_path = resolve_from_root(project_root, output_rel)

    offshore_df, nearshore_df = load_sites_yaml(sites_path)
    x_coord, y_coord, z_grid, land_mask_from_file, grid_attrs = load_bathy_grid(bathy_path)

    grid_epsg = grid_attrs.get("target_epsg", grid_attrs.get("epsg"))
    if grid_epsg is not None and int(grid_epsg) != int(target_epsg):
        raise ValueError(
            f"Bathymetry EPSG mismatch: expected EPSG:{target_epsg}, found EPSG:{int(grid_epsg)}"
        )

    transformer = Transformer.from_crs(
        CRS.from_epsg(4326), CRS.from_epsg(target_epsg), always_xy=True
    )

    offshore_x, offshore_y = transformer.transform(
        offshore_df["lon"].to_numpy(dtype=np.float64),
        offshore_df["lat"].to_numpy(dtype=np.float64),
    )
    nearshore_x, nearshore_y = transformer.transform(
        nearshore_df["lon"].to_numpy(dtype=np.float64),
        nearshore_df["lat"].to_numpy(dtype=np.float64),
    )

    offshore_df = offshore_df.assign(x=offshore_x, y=offshore_y)
    nearshore_df = nearshore_df.assign(x=nearshore_x, y=nearshore_y)

    offshore_rc = xy_to_rc(
        offshore_df["x"].to_numpy(), offshore_df["y"].to_numpy(), x_coord, y_coord
    )
    nearshore_rc = xy_to_rc(
        nearshore_df["x"].to_numpy(), nearshore_df["y"].to_numpy(), x_coord, y_coord
    )

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

    dx = float(np.median(np.diff(x_coord))) if x_coord.size > 1 else 1.0
    dy = float(np.median(np.diff(y_coord))) if y_coord.size > 1 else 1.0
    avg_spacing_m = max(1.0, 0.5 * (abs(dx) + abs(dy)))
    path_smoothing_tolerance_m = PATH_SMOOTH_TOLERANCE_CELLS * avg_spacing_m
    route_mode = str(route_mode or "global").strip().lower()
    if route_mode not in {"global", "local_k_nearest"}:
        raise ValueError(f"route_mode must be one of: global, local_k_nearest; got '{route_mode}'")

    route_records: list[dict[str, Any]] = []
    qa_records: list[dict[str, Any]] = []
    debug_curtains: list[dict[str, Any]] = []

    if route_mode == "global":
        offshore_order = np.argsort(offshore_df["x"].to_numpy())
        curtain_rc, curtain_mask = build_curtain(offshore_rc[offshore_order], z_grid.shape)

        offshore_xy = offshore_df[["x", "y"]].to_numpy(dtype=np.float64)
        nearshore_xy = nearshore_df[["x", "y"]].to_numpy(dtype=np.float64)
        routing_water, offshore_ghost = mask_ghost_offshore_water(
            water_mask=water_mask,
            curtain_mask=curtain_mask,
            x_coord=x_coord,
            y_coord=y_coord,
            offshore_xy=offshore_xy,
            nearshore_xy=nearshore_xy,
        )

        if not np.any(routing_water):
            routing_water = water_mask.copy()
            offshore_ghost = np.zeros_like(water_mask)

        curtain_start_mask = curtain_mask & routing_water
        if not np.any(curtain_start_mask):
            snapped_starts, _ = snap_points_to_mask(curtain_rc, routing_water)
            curtain_start_mask = np.zeros_like(routing_water)
            curtain_start_mask[snapped_starts[:, 0], snapped_starts[:, 1]] = True

        nearshore_snapped_rc, _ = snap_points_to_mask(nearshore_rc, routing_water)

        cc_labels, _ = ndimage.label(routing_water, structure=np.ones((3, 3), dtype=np.int8))
        start_labels = np.unique(cc_labels[curtain_start_mask])
        nearshore_labels = np.unique(
            cc_labels[nearshore_snapped_rc[:, 0], nearshore_snapped_rc[:, 1]]
        )
        shared_labels = np.intersect1d(
            start_labels[start_labels > 0], nearshore_labels[nearshore_labels > 0]
        )
        if shared_labels.size == 0:
            routing_water = water_mask.copy()
            offshore_ghost = np.zeros_like(water_mask)
            curtain_start_mask = curtain_mask & routing_water
            if not np.any(curtain_start_mask):
                snapped_starts, _ = snap_points_to_mask(curtain_rc, routing_water)
                curtain_start_mask = np.zeros_like(routing_water)
                curtain_start_mask[snapped_starts[:, 0], snapped_starts[:, 1]] = True
            nearshore_snapped_rc, _ = snap_points_to_mask(nearshore_rc, routing_water)

        cost_matrix, distance_to_coast_m = build_dynamic_cost_matrix(
            routing_water=routing_water,
            land_mask=land_mask,
            dy=dy,
            dx=dx,
            bottleneck_weight=bottleneck_weight,
            bottleneck_epsilon_m=bottleneck_epsilon_m,
            base_water_cost=BASE_WATER_COST,
        )
        mcp = MCP_Geometric(cost_matrix, sampling=(dy, dx), fully_connected=True)
        starts = [tuple(rc) for rc in np.argwhere(curtain_start_mask)]
        cumulative_costs, _ = mcp.find_costs(starts=starts)

        curtain_xy = curtain_rc_to_xy(curtain_rc, x_coord, y_coord)
        curtain_length_m = path_length_m(curtain_rc, dx=dx, dy=dy)
        debug_curtains.append(
            {
                "site_name": "__global__",
                "curtain_rc": [tuple(int(v) for v in rc) for rc in curtain_rc.tolist()],
                "curtain_xy": [tuple(float(v) for v in xy) for xy in curtain_xy.tolist()],
            }
        )

        for idx, site in nearshore_df.reset_index(drop=True).iterrows():
            target_rc = tuple(int(v) for v in nearshore_snapped_rc[idx])
            reachable = np.isfinite(cumulative_costs[target_rc])
            path_rc: np.ndarray | None = None
            path_xy: np.ndarray | None = None
            path_xy_smooth: np.ndarray | None = None
            if reachable:
                path_rc = np.asarray(mcp.traceback(target_rc), dtype=np.int64)
                path_rc = orient_path_rc_toward_site(path_rc=path_rc, site_rc=target_rc)
                path_xy = rc_path_to_xy(path_rc, x_coord, y_coord)
                path_xy_smooth = smooth_path_xy(
                    path_xy=path_xy, tolerance_m=path_smoothing_tolerance_m
                )

            route_record = _route_record_from_solution(
                site=site,
                nearshore_rc=nearshore_rc,
                nearshore_snapped_rc=nearshore_snapped_rc,
                idx=idx,
                reachable=bool(reachable),
                path_rc=path_rc,
                path_xy=path_xy,
                path_xy_smooth=path_xy_smooth,
                dx=dx,
                dy=dy,
                distance_to_coast_m=distance_to_coast_m,
            )
            route_records.append(route_record)
            qa_records.append(
                _build_route_qa_record(
                    route_record=route_record,
                    route_mode=route_mode,
                    k_sources=0,
                    selected_sources=None,
                    curtain_xy=curtain_xy,
                    curtain_length_m=curtain_length_m,
                )
            )

        metadata = {
            "target_epsg": int(target_epsg),
            "route_mode": route_mode,
            "vertical_convention": vertical_convention,
            "land_mask_source": land_mask_source,
            "cost_model": "base_water_cost + bottleneck_weight / (distance_to_coast_m + bottleneck_epsilon_m)",
            "base_water_cost": float(BASE_WATER_COST),
            "bottleneck_weight": float(bottleneck_weight),
            "bottleneck_epsilon_m": float(bottleneck_epsilon_m),
            "grid_shape": tuple(int(v) for v in z_grid.shape),
            "dx_m": float(dx),
            "dy_m": float(dy),
            "land_pixel_count": int(land_mask.sum()),
            "water_pixel_count": int(water_mask.sum()),
            "sites_yaml": format_path_for_metadata(sites_path, project_root),
            "bathy_file": format_path_for_metadata(bathy_path, project_root),
            "offshore_point_count": int(offshore_df.shape[0]),
            "nearshore_point_count": int(nearshore_df.shape[0]),
            "curtain_pixel_count": int(curtain_rc.shape[0]),
            "ghost_water_pixel_count": int(offshore_ghost.sum()),
            "routing_water_pixel_count": int(routing_water.sum()),
            "distance_to_coast_m_min": float(distance_to_coast_m[routing_water].min()),
            "distance_to_coast_m_max": float(distance_to_coast_m[routing_water].max()),
            "routing_cost_min": float(cost_matrix[routing_water].min()),
            "routing_cost_max": float(cost_matrix[routing_water].max()),
            "path_width_definition": "2 * distance_to_coast_m",
            "final_approach_segment_m": 500.0,
            "path_smoothing_tolerance_m": float(path_smoothing_tolerance_m),
        }
    else:
        if not route_source_metadata:
            raise ValueError("local_k_nearest routing requires route_source_metadata")
        target_sites = [str(v) for v in route_source_metadata.get("target_sites", []) or []]
        source_names_all = route_source_metadata.get("source_names", []) or []
        source_lons_all = route_source_metadata.get("source_lons", []) or []
        source_lats_all = route_source_metadata.get("source_lats", []) or []
        distances_all = route_source_metadata.get("distances_m", []) or []
        bearings_all = route_source_metadata.get("bearings_deg", []) or []
        k_nearest = int(route_source_metadata.get("k_nearest", 0) or 0)
        if not target_sites or not source_names_all:
            raise ValueError("route_source_metadata is missing target site selections")

        source_lookup = {}
        for site_name, names, lons, lats, distances, bearings in zip(
            target_sites,
            source_names_all,
            source_lons_all,
            source_lats_all,
            distances_all,
            bearings_all,
        ):
            source_lookup[str(site_name)] = [
                {
                    "name": str(name),
                    "lon": float(lon),
                    "lat": float(lat),
                    "distance_m": float(distance),
                    "bearing_deg": float(bearing),
                }
                for name, lon, lat, distance, bearing in zip(names, lons, lats, distances, bearings)
            ]

        offshore_by_name = {
            str(row["name"]): row for _, row in offshore_df.reset_index(drop=True).iterrows()
        }
        for idx, site in nearshore_df.reset_index(drop=True).iterrows():
            site_name = str(site["name"])
            selected_sources = source_lookup.get(site_name, [])
            if len(selected_sources) < 2:
                raise ValueError(
                    f"local_k_nearest routing requires at least two sources for site '{site_name}', "
                    f"got {len(selected_sources)}"
                )

            missing_sources = [
                item["name"] for item in selected_sources if item["name"] not in offshore_by_name
            ]
            if missing_sources:
                raise ValueError(
                    f"Selected offshore sources missing from sites.yaml for site '{site_name}': {missing_sources}"
                )

            selected_offshore_df = pd.DataFrame(
                [offshore_by_name[item["name"]] for item in selected_sources]
            ).reset_index(drop=True)
            selected_offshore_rc = xy_to_rc(
                selected_offshore_df["x"].to_numpy(),
                selected_offshore_df["y"].to_numpy(),
                x_coord,
                y_coord,
            )
            selected_order = np.argsort(selected_offshore_df["x"].to_numpy())
            curtain_rc = build_curtain(selected_offshore_rc[selected_order], z_grid.shape)[0]
            curtain_mask = np.zeros(z_grid.shape, dtype=bool)
            curtain_mask[curtain_rc[:, 0], curtain_rc[:, 1]] = True
            offshore_xy_local = selected_offshore_df[["x", "y"]].to_numpy(dtype=np.float64)
            route_record = _route_single_site_with_local_curtain(
                site=site,
                nearshore_site_rc=nearshore_rc[idx],
                x_coord=x_coord,
                y_coord=y_coord,
                water_mask=water_mask,
                land_mask=land_mask,
                dx=dx,
                dy=dy,
                path_smoothing_tolerance_m=path_smoothing_tolerance_m,
                bottleneck_weight=bottleneck_weight,
                bottleneck_epsilon_m=bottleneck_epsilon_m,
                curtain_rc=curtain_rc,
                curtain_mask=curtain_mask,
                offshore_xy_local=offshore_xy_local,
            )
            route_records.append(route_record)

            curtain_xy = curtain_rc_to_xy(curtain_rc, x_coord, y_coord)
            curtain_length_m = path_length_m(curtain_rc, dx=dx, dy=dy)
            qa_records.append(
                _build_route_qa_record(
                    route_record=route_record,
                    route_mode=route_mode,
                    k_sources=k_nearest,
                    selected_sources=selected_sources,
                    curtain_xy=curtain_xy,
                    curtain_length_m=curtain_length_m,
                )
            )
            debug_curtains.append(
                {
                    "site_name": site_name,
                    "curtain_rc": [tuple(int(v) for v in rc) for rc in curtain_rc.tolist()],
                    "curtain_xy": [tuple(float(v) for v in xy) for xy in curtain_xy.tolist()],
                    "selected_sources": selected_sources,
                }
            )

        metadata = {
            "target_epsg": int(target_epsg),
            "route_mode": route_mode,
            "vertical_convention": vertical_convention,
            "land_mask_source": land_mask_source,
            "cost_model": "base_water_cost + bottleneck_weight / (distance_to_coast_m + bottleneck_epsilon_m)",
            "base_water_cost": float(BASE_WATER_COST),
            "bottleneck_weight": float(bottleneck_weight),
            "bottleneck_epsilon_m": float(bottleneck_epsilon_m),
            "grid_shape": tuple(int(v) for v in z_grid.shape),
            "dx_m": float(dx),
            "dy_m": float(dy),
            "land_pixel_count": int(land_mask.sum()),
            "water_pixel_count": int(water_mask.sum()),
            "sites_yaml": format_path_for_metadata(sites_path, project_root),
            "bathy_file": format_path_for_metadata(bathy_path, project_root),
            "offshore_point_count": int(offshore_df.shape[0]),
            "nearshore_point_count": int(nearshore_df.shape[0]),
            "curtain_pixel_count": int(
                sum(len(item.get("curtain_rc", [])) for item in debug_curtains)
            ),
            "path_width_definition": "2 * distance_to_coast_m",
            "final_approach_segment_m": 500.0,
            "path_smoothing_tolerance_m": float(path_smoothing_tolerance_m),
            "k_nearest": int(k_nearest),
        }

    routes_df = pd.DataFrame(route_records).reset_index(drop=True)
    qa_df = pd.DataFrame(qa_records).reset_index(drop=True) if qa_records else pd.DataFrame()
    if route_mode == "global" and debug_curtains:
        legacy_curtain_rc = debug_curtains[0].get("curtain_rc", [])
        legacy_curtain_xy = debug_curtains[0].get("curtain_xy", [])
    else:
        legacy_curtain_rc = []
        legacy_curtain_xy = []

    payload: dict[str, Any] = {
        "metadata": metadata,
        "curtain_rc": legacy_curtain_rc,
        "curtain_xy": legacy_curtain_xy,
        "routes": routes_df,
        "route_curtain_qa": qa_df,
        "route_curtains": debug_curtains,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Saved routing features: {output_path}")
    print(f"Route curtain mode: {route_mode}")
    print(f"Reachable nearshore sites: {int(routes_df['reachable'].sum())} / {len(routes_df)}")
    print(
        "Path length range (m): "
        f"{routes_df['path_length_m'].min(skipna=True):.2f} to {routes_df['path_length_m'].max(skipna=True):.2f}"
    )
    return output_path


def parse_args() -> argparse.Namespace:
    """Build CLI parser for standalone script execution."""
    parser = argparse.ArgumentParser(description="Boundary-curtain routing feature extraction")
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
        default="data/processed/routing_features.pkl",
        help="Path to output routing pickle",
    )
    parser.add_argument(
        "--epsg",
        type=int,
        default=TARGET_EPSG,
        help="Projected EPSG expected for routing grid",
    )
    parser.add_argument(
        "--bottleneck-weight",
        type=float,
        default=BOTTLENECK_WEIGHT,
        help="Penalty strength for narrow channels (higher avoids bottlenecks more strongly)",
    )
    parser.add_argument(
        "--bottleneck-epsilon-m",
        type=float,
        default=BOTTLENECK_EPSILON_M,
        help="Small positive stabilizer in meters for inverse-distance penalty",
    )
    return parser.parse_args()


def main() -> None:
    """Run boundary-curtain routing end-to-end."""
    args = parse_args()
    project_root = resolve_project_root(Path(__file__).resolve())
    run_fetch_routing(
        project_root=project_root,
        sites_rel=args.sites,
        bathy_rel=args.bathy,
        output_rel=args.output,
        target_epsg=args.epsg,
        bottleneck_weight=args.bottleneck_weight,
        bottleneck_epsilon_m=args.bottleneck_epsilon_m,
    )


if __name__ == "__main__":
    main()
