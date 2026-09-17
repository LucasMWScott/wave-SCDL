"""Shared helpers for multi-source QA and analysis notebooks."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import contextily as ctx
except Exception:
    ctx = None

try:
    from pyproj import Transformer
except Exception:
    Transformer = None

try:
    import xarray as xr
except Exception:
    xr = None


HELPERS_DIR = Path(__file__).resolve().parent
REPO_ROOT = HELPERS_DIR.parent


def read_yaml(path):
    """Read YAML with inherited, config-relative paths."""
    from src.config_loader import read_yaml_config

    return read_yaml_config(path)


def _resolve_from_anchor(path_like: str | Path, anchor: str | Path | None = None) -> Path:
    """Resolve a path relative to cwd first, then an optional anchor file/dir."""
    path = Path(path_like)
    if path.is_absolute():
        return path.resolve()

    candidates = [(Path.cwd() / path).resolve()]

    if anchor is not None:
        anchor_path = Path(anchor).resolve()
        anchor_dir = anchor_path.parent if anchor_path.suffix else anchor_path
        candidates.append((anchor_dir / path).resolve())

    candidates.append((HELPERS_DIR / path).resolve())
    candidates.append((REPO_ROOT / path).resolve())

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return candidates[0]


def resolve_point_centric_dir(config_path: str | Path = "../configs/training.yaml") -> Path:
    cfg_path = _resolve_from_anchor(config_path)
    cfg = read_yaml(cfg_path)
    point_centric_dir = str(((cfg.get("data", {}) or {}).get("point_centric_dir", ""))).strip()
    if not point_centric_dir:
        raise ValueError("Config does not define data.point_centric_dir")
    return _resolve_from_anchor(point_centric_dir, anchor=cfg_path)


def resolve_sites_config_path(config_path: str | Path = "../configs/training.yaml") -> Path:
    cfg_path = _resolve_from_anchor(config_path)
    cfg = read_yaml(cfg_path)
    sites_config = str(((cfg.get("data", {}) or {}).get("sites_config", ""))).strip()
    if not sites_config:
        raise ValueError("Config does not define data.sites_config")
    return _resolve_from_anchor(sites_config, anchor=cfg_path)


def load_metadata(point_centric_dir: str | Path) -> dict:
    path = _resolve_from_anchor(point_centric_dir) / "point_centric_metadata.json"
    with path.open("r") as fh:
        return json.load(fh) or {}


def load_source_metadata(point_centric_dir: str | Path) -> dict:
    path = _resolve_from_anchor(point_centric_dir) / "point_centric_source_metadata.json"
    if not path.exists():
        return {}
    with path.open("r") as fh:
        return json.load(fh) or {}


def load_route_curtain_qa(
    path: str | Path = "../data/processed/route_curtain_qa.csv",
) -> pd.DataFrame:
    csv_path = _resolve_from_anchor(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing route curtain QA CSV: {csv_path}")
    return pd.read_csv(csv_path)


def load_routing_payload(path: str | Path = "../data/processed/routing_features.pkl") -> dict:
    pkl_path = _resolve_from_anchor(path)
    if not pkl_path.exists():
        raise FileNotFoundError(f"Missing routing payload: {pkl_path}")
    with pkl_path.open("rb") as fh:
        return pickle.load(fh)


def load_sites_config(path: str | Path = "../configs/sites.yaml") -> dict:
    return read_yaml(_resolve_from_anchor(path))


def load_static_features(
    path: str | Path = "../data/processed/master_static_features.csv",
) -> pd.DataFrame:
    csv_path = _resolve_from_anchor(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing static features CSV: {csv_path}")
    return pd.read_csv(csv_path)


def load_npz(path: str | Path) -> Any:
    return np.load(_resolve_from_anchor(path), allow_pickle=True)


def load_optional_npz(path: str | Path) -> Any | None:
    path = _resolve_from_anchor(path)
    if not path.exists():
        return None
    return np.load(path, allow_pickle=True)


def load_bathy_artifact(point_centric_dir: str | Path) -> Any:
    path = _resolve_from_anchor(point_centric_dir) / "point_centric_X_bathy.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing bathymetry artifact: {path}")
    return np.load(path, allow_pickle=True)


def discover_prediction_files(results_dir: str | Path) -> dict[str, Path]:
    base = _resolve_from_anchor(results_dir)
    discovered: dict[str, Path] = {}
    for split in ("val", "test"):
        csv_path = base / f"predictions_{split}.csv"
        nc_path = base / f"predictions_{split}.nc"
        if csv_path.exists():
            discovered[f"{split}_csv"] = csv_path
        if nc_path.exists():
            discovered[f"{split}_nc"] = nc_path
    return discovered


def load_prediction_table(results_dir: str | Path, split: str = "val") -> pd.DataFrame:
    base = _resolve_from_anchor(results_dir)
    csv_path = base / f"predictions_{split}.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    nc_path = base / f"predictions_{split}.nc"
    if nc_path.exists():
        if xr is None:
            raise RuntimeError("xarray is required to load NetCDF prediction files")
        with xr.open_dataset(nc_path) as ds:
            return ds.to_dataframe().reset_index()
    raise FileNotFoundError(f"No prediction file found for split='{split}' under {base}")


def metadata_shape_table(metadata: dict) -> pd.DataFrame:
    rows = []
    for key, value in metadata.items():
        if key.endswith("_shape") or key.endswith("_shapes"):
            rows.append({"key": key, "value": value})
    return pd.DataFrame(rows)


def print_artifact_summary(point_centric_dir: str | Path) -> None:
    base = _resolve_from_anchor(point_centric_dir)
    metadata = load_metadata(base)
    source_metadata = load_source_metadata(base)
    print("Point-centric dir:", base)
    print("Nearshore sites:", len(metadata.get("nearshore_sites", []) or []))
    print("Dynamic shape:", metadata.get("X_dynamic_shape"))
    print("Dynamic sources shape:", metadata.get("X_dynamic_sources_shape"))
    print("Source geometry shape:", metadata.get("source_geometry_shape"))
    print("Bathy shape:", metadata.get("X_bathy_shape"))
    print("Multi-source enabled:", ((metadata.get("multi_source", {}) or {}).get("enabled", False)))
    if source_metadata:
        print("K nearest:", source_metadata.get("k_nearest"))
        print("First target site:", (source_metadata.get("target_sites", []) or [None])[0])


def results_dir_from_config(config_path: str | Path = "../configs/training.yaml") -> Path:
    cfg_path = _resolve_from_anchor(config_path)
    cfg = read_yaml(cfg_path)
    output_dir = str(((cfg.get("logging", {}) or {}).get("output_dir", ""))).strip()
    if not output_dir:
        raise ValueError("Config does not define logging.output_dir")
    return _resolve_from_anchor(output_dir, anchor=cfg_path)


def resolve_split_sets(
    metadata: dict | None,
    config_path: str | Path = "../configs/training.yaml",
    prefer: str = "metadata",
) -> dict[str, set[str]]:
    """Resolve train/val/test site sets from metadata or config."""
    metadata = metadata or {}
    splits = (metadata.get("splits", {}) or {}) if isinstance(metadata, dict) else {}
    metadata_sets = {
        "train": set(str(v) for v in (splits.get("train_sites", []) or [])),
        "val": set(str(v) for v in (splits.get("val_sites", []) or [])),
        "test": set(str(v) for v in (splits.get("test_sites", []) or [])),
    }

    cfg = read_yaml(config_path)
    data_cfg = cfg.get("data", {}) or {}
    config_sets = {
        "train": set(str(v) for v in (data_cfg.get("train_sites", []) or [])),
        "val": set(
            str(v)
            for v in (data_cfg.get("validation_sites", []) or data_cfg.get("val_sites", []) or [])
        ),
        "test": set(str(v) for v in (data_cfg.get("test_sites", []) or [])),
    }
    metadata_has_values = any(metadata_sets.values())
    config_has_values = any(config_sets.values())

    if prefer not in {"metadata", "config"}:
        raise ValueError("prefer must be either 'metadata' or 'config'")

    if prefer == "config":
        return config_sets if config_has_values else metadata_sets

    return metadata_sets if metadata_has_values else config_sets


def attach_split_column(
    df: pd.DataFrame,
    split_sets: dict[str, set[str]],
    site_col: str = "name",
) -> pd.DataFrame:
    """Attach a `split` column using train/val/test site sets."""
    out = df.copy()
    site_names = out[site_col].astype(str)
    out["split"] = np.where(
        site_names.isin(split_sets.get("test", set())),
        "test",
        np.where(site_names.isin(split_sets.get("val", set())), "val", "train"),
    )
    return out


def require_web_mercator_tools() -> None:
    if Transformer is None:
        raise RuntimeError(
            "pyproj is required for satellite notebooks. Install pyproj to continue."
        )


def add_web_mercator_columns(
    df: pd.DataFrame,
    lon_col: str = "lon",
    lat_col: str = "lat",
    x_col: str = "x_3857",
    y_col: str = "y_3857",
) -> pd.DataFrame:
    """Project lon/lat columns to EPSG:3857 for satellite basemaps."""
    require_web_mercator_tools()
    out = df.copy()
    transformer = Transformer.from_crs(4326, 3857, always_xy=True)
    x_vals, y_vals = transformer.transform(
        out[lon_col].astype(float).to_numpy(dtype=float),
        out[lat_col].astype(float).to_numpy(dtype=float),
    )
    out[x_col] = np.asarray(x_vals, dtype=float)
    out[y_col] = np.asarray(y_vals, dtype=float)
    return out


def add_world_imagery_basemap(ax, zoom: int = 10) -> None:
    """Add Esri World Imagery when contextily is available."""
    if ctx is None:
        print("contextily not available; plotting without satellite basemap.")
        return
    try:
        ctx.add_basemap(
            ax,
            crs="EPSG:3857",
            source=ctx.providers.Esri.WorldImagery,
            zoom=zoom,
            attribution=False,
        )
    except Exception as exc:
        print(f"Satellite basemap skipped: {exc}")
