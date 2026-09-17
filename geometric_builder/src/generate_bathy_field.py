#!/usr/bin/env python3
"""Generate full and site-local bathymetry grids from .xyz tiles."""

from __future__ import annotations

import argparse
import glob
import logging
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import yaml

try:
    from pyproj import Transformer
except Exception:
    Transformer = None


def read_sites_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        sites = yaml.safe_load(fh) or {}

    for key in ("nearshore_sites", "offshore_sites"):
        entries = sites.get(key) or []
        names = [
            str(site.get("name")) for site in entries if isinstance(site, dict) and "name" in site
        ]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"{key} contains duplicate site names: {', '.join(duplicates)}")
    return sites


def find_bathy_dir(candidates: Optional[list] = None) -> Optional[str]:
    if candidates is None:
        candidates = [
            "data/bathy",
            "data/raw/bathy",
            "data/raw/bathy/dybedata",
            "data/bathy/dybdedata",
            "data/bathy/dybedata",
            "data/raw/bathy/dybdedata",
        ]
    for path in candidates:
        if os.path.isdir(path):
            return path
    return None


def latlon_to_utm(
    xs_lon: np.ndarray,
    ys_lat: np.ndarray,
    epsg: int = 32633,
) -> Tuple[np.ndarray, np.ndarray]:
    if Transformer is None:
        raise RuntimeError("pyproj is required for lat/lon to UTM conversion")
    transformer = Transformer.from_crs(4326, epsg, always_xy=True)
    easting, northing = transformer.transform(xs_lon, ys_lat)
    return np.asarray(easting), np.asarray(northing)


def compute_bbox_from_sites(
    sites: dict,
    padding_m: float,
    epsg: int = 32633,
) -> Tuple[float, float, float, float]:
    all_sites = []
    all_sites += sites.get("nearshore_sites") or []
    all_sites += sites.get("nearshore") or []
    all_sites += sites.get("offshore_sites") or []

    if not all_sites:
        raise ValueError("No nearshore/offshore sites found in sites.yaml to derive bbox")

    coords = [(site["lon"], site["lat"]) for site in all_sites if ("lon" in site and "lat" in site)]
    if not coords:
        raise ValueError("No site coordinates found in sites.yaml to derive bbox")

    lons = np.array([c[0] for c in coords], dtype=float)
    lats = np.array([c[1] for c in coords], dtype=float)
    eastings, northings = latlon_to_utm(lons, lats, epsg=epsg)

    min_x = float(eastings.min()) - padding_m
    max_x = float(eastings.max()) + padding_m
    min_y = float(northings.min()) - padding_m
    max_y = float(northings.max()) + padding_m
    return min_x, max_x, min_y, max_y


def list_xyz_files(bathy_dir: str) -> list:
    patterns = [os.path.join(bathy_dir, "*.xyz"), os.path.join(bathy_dir, "*.XYZ")]
    files = []
    for pattern in patterns:
        for file_path in sorted(glob.glob(pattern)):
            if ":Zone.Identifier" in file_path:
                continue
            files.append(file_path)
    return sorted(set(files))


def read_bathy_tiles(
    bathy_files: list,
    bbox: Optional[Tuple[float, float, float, float]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs_all = []
    ys_all = []
    zs_all = []

    for file_path in bathy_files:
        try:
            data = np.loadtxt(file_path, ndmin=2)
            if data.size == 0 or data.shape[1] < 3:
                continue

            x_vals = data[:, 0]
            y_vals = data[:, 1]
            z_vals = data[:, 2]

            if bbox is not None:
                min_x, max_x, min_y, max_y = bbox
                mask = (x_vals >= min_x) & (x_vals <= max_x) & (y_vals >= min_y) & (y_vals <= max_y)
                if not np.any(mask):
                    continue
                x_vals = x_vals[mask]
                y_vals = y_vals[mask]
                z_vals = z_vals[mask]

            xs_all.append(x_vals)
            ys_all.append(y_vals)
            zs_all.append(z_vals)
        except Exception as exc:
            logging.warning("Failed to read %s: %s", file_path, exc)

    if not xs_all:
        return np.array([]), np.array([]), np.array([])

    xs = np.concatenate(xs_all)
    ys = np.concatenate(ys_all)
    zs = np.concatenate(zs_all)
    return xs, ys, zs


def compute_bbox_from_points(xs: np.ndarray, ys: np.ndarray) -> Tuple[float, float, float, float]:
    if xs.size == 0 or ys.size == 0:
        raise ValueError("Cannot compute bbox from empty point arrays")
    return float(xs.min()), float(xs.max()), float(ys.min()), float(ys.max())


def align_bbox_to_resolution(
    bbox: Tuple[float, float, float, float],
    resolution: float,
) -> Tuple[float, float, float, float]:
    min_x, max_x, min_y, max_y = bbox
    aligned_min_x = float(np.floor(min_x / resolution) * resolution)
    aligned_max_x = float(np.ceil(max_x / resolution) * resolution)
    aligned_min_y = float(np.floor(min_y / resolution) * resolution)
    aligned_max_y = float(np.ceil(max_y / resolution) * resolution)
    return aligned_min_x, aligned_max_x, aligned_min_y, aligned_max_y


def grid_points(
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
    bbox: Tuple[float, float, float, float],
    resolution: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Tuple[float, float, float, float]]:
    min_x, max_x, min_y, max_y = align_bbox_to_resolution(bbox, resolution)

    nx = int(round((max_x - min_x) / resolution)) + 1
    ny = int(round((max_y - min_y) / resolution)) + 1
    if nx <= 0 or ny <= 0:
        raise ValueError("Invalid grid size computed")

    x_coords = min_x + np.arange(nx, dtype=float) * resolution
    y_coords = min_y + np.arange(ny, dtype=float) * resolution

    sum_grid = np.zeros((ny, nx), dtype=np.float64)
    count_grid = np.zeros((ny, nx), dtype=np.int64)

    ix = np.rint((xs - min_x) / resolution).astype(np.int64)
    iy = np.rint((ys - min_y) / resolution).astype(np.int64)
    valid = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)

    np.add.at(sum_grid, (iy[valid], ix[valid]), zs[valid])
    np.add.at(count_grid, (iy[valid], ix[valid]), 1)

    with np.errstate(invalid="ignore", divide="ignore"):
        mean_grid = np.where(count_grid > 0, sum_grid / count_grid, np.nan)

    return x_coords, y_coords, mean_grid, count_grid, (min_x, max_x, min_y, max_y)


def build_land_mask(z_grid: np.ndarray) -> Tuple[np.ndarray, str]:
    """Build a boolean land mask from gridded bathymetry/elevation.

    The convention is inferred from finite-cell sign:
    - positive-down depth: land is ``z <= 0`` or NaN
    - signed elevation: land is ``z >= 0`` or NaN
    """
    finite = np.isfinite(z_grid)
    finite_values = z_grid[finite]

    if finite_values.size == 0:
        return np.ones_like(z_grid, dtype=bool), "unknown"

    positive_ratio = float(np.mean(finite_values > 0.0))
    if positive_ratio > 0.9:
        convention = "positive_down_depth"
        land_mask = (~finite) | (z_grid <= 0.0)
    else:
        convention = "signed_elevation"
        land_mask = (~finite) | (z_grid >= 0.0)

    return land_mask.astype(bool), convention


def extract_subgrid(
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    z_grid: np.ndarray,
    bbox: Tuple[float, float, float, float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    min_x, max_x, min_y, max_y = bbox
    x_mask = (x_coords >= min_x) & (x_coords <= max_x)
    y_mask = (y_coords >= min_y) & (y_coords <= max_y)

    if not np.any(x_mask) or not np.any(y_mask):
        raise ValueError("Computed subgrid is empty. Check EPSG, site coordinates, and padding.")

    return x_coords[x_mask], y_coords[y_mask], z_grid[np.ix_(y_mask, x_mask)], x_mask, y_mask


def bbox_to_dict(bbox: Tuple[float, float, float, float]) -> dict:
    min_x, max_x, min_y, max_y = bbox
    return {
        "min_x": float(min_x),
        "max_x": float(max_x),
        "min_y": float(min_y),
        "max_y": float(max_y),
    }


def save_grid(
    out_path: str,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    z_grid: np.ndarray,
    land_mask: np.ndarray,
    metadata: dict,
    sample_count: Optional[np.ndarray] = None,
) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    payload = {
        "x": x_coords,
        "y": y_coords,
        "z": z_grid,
        "land_mask": land_mask.astype(bool),
        "metadata": np.array(metadata, dtype=object),
    }
    if sample_count is not None:
        payload["sample_count"] = sample_count.astype(np.int32)
    np.savez_compressed(out_path, **payload)
    logging.info("Saved bathymetry product: %s", out_path)


def run_generate_bathy_field(
    sites_path: str,
    bathy_dir_path: str,
    padding_m: float,
    resolution_m: float,
    epsg: int,
    out_dir_path: str,
    full_name: str,
    subgrid_name: str,
) -> tuple[Path, Path]:
    """Execute full + site-subgrid bathymetry generation and save NPZ outputs."""
    sites = read_sites_yaml(sites_path)

    bathy_dir = bathy_dir_path
    if not os.path.isdir(bathy_dir):
        fallback_dir = find_bathy_dir()
        if fallback_dir is None:
            raise FileNotFoundError(
                f"Bathymetry directory not found: {bathy_dir}. Provide a valid bathy directory."
            )
        logging.warning(
            "Requested bathy dir not found (%s). Falling back to %s", bathy_dir, fallback_dir
        )
        bathy_dir = fallback_dir

    files = list_xyz_files(bathy_dir)
    if not files:
        raise FileNotFoundError(f"No .xyz files found in {bathy_dir}")

    logging.info("Reading %d bathymetry tiles from %s", len(files), bathy_dir)
    xs, ys, zs = read_bathy_tiles(files)
    if xs.size == 0:
        raise RuntimeError("No readable bathymetry points found in input tiles")

    full_bbox_raw = compute_bbox_from_points(xs, ys)
    x_full, y_full, z_full, count_full, full_bbox = grid_points(
        xs,
        ys,
        zs,
        full_bbox_raw,
        resolution_m,
    )
    logging.info("Full field grid shape: (ny=%d, nx=%d)", z_full.shape[0], z_full.shape[1])

    full_land_mask, vertical_convention = build_land_mask(z_full)
    logging.info(
        "Land-mask pixels (full): %d land / %d water",
        int(full_land_mask.sum()),
        int((~full_land_mask).sum()),
    )

    site_bbox = compute_bbox_from_sites(sites, padding_m, epsg=epsg)
    x_sub, y_sub, z_sub, x_mask, y_mask = extract_subgrid(x_full, y_full, z_full, site_bbox)
    count_sub = count_full[np.ix_(y_mask, x_mask)]
    sub_land_mask = full_land_mask[np.ix_(y_mask, x_mask)]
    logging.info("Site subgrid shape: (ny=%d, nx=%d)", z_sub.shape[0], z_sub.shape[1])

    os.makedirs(out_dir_path, exist_ok=True)
    full_out = os.path.join(out_dir_path, full_name)
    sub_out = os.path.join(out_dir_path, subgrid_name)

    common_metadata = {
        "epsg": int(epsg),
        "resolution_m": float(resolution_m),
        "sites_yaml": os.path.abspath(sites_path),
        "bathy_dir": os.path.abspath(bathy_dir),
        "source_tile_count": len(files),
        "source_tiles": [os.path.basename(file_path) for file_path in files],
    }
    full_metadata = {
        **common_metadata,
        "product": "full_2d_field",
        "vertical_convention": vertical_convention,
        "land_mask_rule": "land = NaN or dry-side by inferred vertical convention",
        "bbox": bbox_to_dict(full_bbox),
        "shape": {"ny": int(z_full.shape[0]), "nx": int(z_full.shape[1])},
        "land_pixel_count": int(full_land_mask.sum()),
        "water_pixel_count": int((~full_land_mask).sum()),
    }
    sub_metadata = {
        **common_metadata,
        "product": "site_subgrid",
        "vertical_convention": vertical_convention,
        "land_mask_rule": "land = NaN or dry-side by inferred vertical convention",
        "bbox": bbox_to_dict(site_bbox),
        "padding_m": float(padding_m),
        "shape": {"ny": int(z_sub.shape[0]), "nx": int(z_sub.shape[1])},
        "land_pixel_count": int(sub_land_mask.sum()),
        "water_pixel_count": int((~sub_land_mask).sum()),
    }

    save_grid(
        full_out,
        x_full,
        y_full,
        z_full,
        full_land_mask,
        full_metadata,
        sample_count=count_full,
    )
    save_grid(
        sub_out,
        x_sub,
        y_sub,
        z_sub,
        sub_land_mask,
        sub_metadata,
        sample_count=count_sub,
    )

    logging.info("Finished generating bathymetry products")
    logging.info("Full field: %s", full_out)
    logging.info("Site subgrid: %s", sub_out)
    return Path(full_out), Path(sub_out)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate full + site-local bathymetry fields from .xyz tiles"
    )
    parser.add_argument("--sites", default="configs/sites.yaml", help="Path to sites.yaml")
    parser.add_argument(
        "--bathy-dir",
        default="data/bathy",
        help="Directory with .xyz bathymetry tiles",
    )
    parser.add_argument(
        "--padding",
        type=float,
        default=1_000.0,
        help="Site subgrid padding around all configured coordinates (meters)",
    )
    parser.add_argument("--resolution", type=float, default=50.0, help="Grid resolution in meters")
    parser.add_argument(
        "--epsg",
        type=int,
        default=32633,
        help="Projected EPSG of bathymetry and target site coordinates",
    )
    parser.add_argument(
        "--out-dir",
        default="data/processed/bathy",
        help="Output directory for generated bathymetry products",
    )
    parser.add_argument(
        "--full-name",
        default="bathy_field_full.npz",
        help="Filename for full 2D bathymetry field",
    )
    parser.add_argument(
        "--subgrid-name",
        default="bathy_field_project_site.npz",
        help="Filename for site-focused subgrid",
    )
    args = parser.parse_args()

    run_generate_bathy_field(
        sites_path=args.sites,
        bathy_dir_path=args.bathy_dir,
        padding_m=args.padding,
        resolution_m=args.resolution,
        epsg=args.epsg,
        out_dir_path=args.out_dir,
        full_name=args.full_name,
        subgrid_name=args.subgrid_name,
    )


if __name__ == "__main__":
    main()
