#!/usr/bin/env python3
"""Run bathymetry preparation, routing, ray casting, and static aggregation."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import generate_bathy_field
from .fetch_router import BOTTLENECK_EPSILON_M, BOTTLENECK_WEIGHT, TARGET_EPSG, run_fetch_routing
from .ray_caster import (
    run_ray_casting,
)


from .features import (
    RAY_SECTOR_COLUMNS as RAY_SECTOR_COLUMNS,
    SECTOR_ANGLE_DEG as SECTOR_ANGLE_DEG,
    SECTOR_WIDTH_DEG as SECTOR_WIDTH_DEG,
    POROSITY_COLUMNS as POROSITY_COLUMNS,
    FETCH_ANISOTROPY_COLUMNS as FETCH_ANISOTROPY_COLUMNS,
    FJORDNESS_COMPONENT_COLUMNS as FJORDNESS_COMPONENT_COLUMNS,
    FJORDNESS_COLUMNS as FJORDNESS_COLUMNS,
    PATH_RATIO_COLUMNS as PATH_RATIO_COLUMNS,
    OPEN_FETCH_THRESHOLD_M as OPEN_FETCH_THRESHOLD_M,
    CLOSED_FETCH_THRESHOLD_M as CLOSED_FETCH_THRESHOLD_M,
    FJORDNESS_OPEN_MAX as FJORDNESS_OPEN_MAX,
    FJORDNESS_TRANSITION_MAX as FJORDNESS_TRANSITION_MAX,
    DEFAULT_BREAKING_ENABLED as DEFAULT_BREAKING_ENABLED,
    DEFAULT_BREAKING_GAMMA as DEFAULT_BREAKING_GAMMA,
    MASTER_COLUMNS as MASTER_COLUMNS,
    _safe_divide as _safe_divide,
    _extract_sector_matrix as _extract_sector_matrix,
    _circular_run_width as _circular_run_width,
    add_fetch_anisotropy_features as add_fetch_anisotropy_features,
    add_path_geometry_ratio_features as add_path_geometry_ratio_features,
    _minmax_component as _minmax_component,
    add_fjordness_features as add_fjordness_features,
    add_local_breaking_features as add_local_breaking_features,
    log_local_breaking_summary as log_local_breaking_summary,
    validate_master_static_features as validate_master_static_features,
    build_ray_site_summary as build_ray_site_summary,
    build_master_static_features as build_master_static_features,
)


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


def _read_yaml(path):
    """Read configuration paths relative to their declaring file."""
    from coastal_wave.common.config import read_config

    return read_config(path)


def _load_multi_source_module(project_root: Path):
    """Use the same deterministic source selection as ML preprocessing."""
    from coastal_wave.common import sources

    return sources


def _resolve_breaking_config(
    project_root: Path,
    preprocess_config_rel: str,
) -> tuple[bool, float, str]:
    cfg_path = resolve_from_root(project_root, preprocess_config_rel)
    if not cfg_path.exists():
        return DEFAULT_BREAKING_ENABLED, DEFAULT_BREAKING_GAMMA, str(cfg_path)

    payload = _read_yaml(cfg_path)

    static_cfg = payload.get("static_features", {}) or {}
    breaking_cfg = static_cfg.get("breaking", {}) or {}
    enabled = bool(breaking_cfg.get("enabled", DEFAULT_BREAKING_ENABLED))
    gamma = float(breaking_cfg.get("gamma", DEFAULT_BREAKING_GAMMA))
    if not np.isfinite(gamma) or gamma <= 0.0:
        raise ValueError(f"static_features.breaking.gamma must be finite and > 0, got {gamma}")
    return enabled, gamma, str(cfg_path)


def _resolve_route_curtain_config(
    project_root: Path,
    preprocess_config_rel: str,
) -> tuple[dict[str, Any], str]:
    cfg_path = resolve_from_root(project_root, preprocess_config_rel)
    payload = _read_yaml(cfg_path) if cfg_path.exists() else {}
    static_cfg = payload.get("static_features", {}) or {}
    route_cfg = static_cfg.get("route_curtain", {}) or {}
    resolved = {
        "mode": str(route_cfg.get("mode", "global")).strip().lower() or "global",
        "use_multi_source_config": bool(route_cfg.get("use_multi_source_config", False)),
        "padding_m": float(route_cfg.get("padding_m", 2000.0)),
        "qa_out_path": str(route_cfg.get("qa_out_path", "data/processed/route_curtain_qa.csv")),
    }
    if resolved["mode"] not in {"global", "local_k_nearest"}:
        raise ValueError(
            "static_features.route_curtain.mode must be one of: global, local_k_nearest; "
            f"got '{resolved['mode']}'"
        )
    return resolved, str(cfg_path)


def _build_route_source_metadata(
    *,
    project_root: Path,
    sites_rel: str,
    training_config_rel: str,
    route_cfg: dict[str, Any],
) -> tuple[dict[str, Any] | None, dict[str, Any], list[str]]:
    if str(route_cfg.get("mode", "global")) != "local_k_nearest":
        return None, {}, []

    if not bool(route_cfg.get("use_multi_source_config", False)):
        raise ValueError(
            "local_k_nearest route curtains require static_features.route_curtain.use_multi_source_config=true "
            "so static routing cannot drift from dynamic multi-source selection."
        )

    training_cfg_path = resolve_from_root(project_root, training_config_rel)
    if not training_cfg_path.exists():
        raise FileNotFoundError(
            f"Missing training config for multi-source route curtain: {training_cfg_path}"
        )
    training_cfg = _read_yaml(training_cfg_path)

    sites_path = resolve_from_root(project_root, sites_rel)
    sites_payload = _read_yaml(sites_path)
    nearshore_entries = list(sites_payload.get("nearshore_sites", []) or [])
    offshore_entries = list(sites_payload.get("offshore_sites", []) or [])

    multi_source_module = _load_multi_source_module(project_root)
    multi_source_cfg = multi_source_module.resolve_multi_source_config(
        training_cfg.get("data", {}) or {}
    )
    if not bool(multi_source_cfg.get("enabled", False)):
        raise ValueError(
            "local_k_nearest route curtains require data.multi_source.enabled=true in the training config."
        )

    source_metadata, warnings = multi_source_module.build_k_nearest_source_metadata(
        nearshore_entries=nearshore_entries,
        offshore_entries=offshore_entries,
        k_nearest=int(multi_source_cfg.get("k_nearest", 3)),
        source_type=str(multi_source_cfg.get("source_type", "nora3_wave")),
        weight_power=float(multi_source_cfg.get("weight_power", 1.0)),
        max_distance_km_warn=float(multi_source_cfg.get("max_distance_km_warn", 80.0)),
        max_distance_km_error=float(multi_source_cfg.get("max_distance_km_error", 150.0)),
        allow_padding=bool(multi_source_cfg.get("allow_padding", False)),
    )
    return source_metadata, multi_source_cfg, warnings


def load_routing_payload(routing_path: Path) -> dict[str, Any]:
    """Load the full routing pickle payload."""
    with routing_path.open("rb") as handle:
        payload: dict[str, Any] = pickle.load(handle)
    return payload


def load_routing_routes(routing_path: Path) -> pd.DataFrame:
    """Load route feature table from routing pickle payload."""
    payload = load_routing_payload(routing_path)
    routes = payload.get("routes")
    if not isinstance(routes, pd.DataFrame):
        raise TypeError(
            "routing_features.pkl does not contain a pandas DataFrame under key 'routes'"
        )
    return routes.copy()


def run_pipeline(
    project_root: Path,
    sites_rel: str,
    preprocess_config_rel: str,
    training_config_rel: str,
    bathy_dir_rel: str,
    bathy_rel: str | None,
    bathy_out_dir_rel: str,
    full_name: str,
    subgrid_name: str,
    routing_output_rel: str,
    ray_output_rel: str,
    master_output_rel: str,
    target_epsg: int,
    padding_m: float,
    resolution_m: float,
    bottleneck_weight: float,
    bottleneck_epsilon_m: float,
    angle_step_deg: float,
    ray_step_m: float,
    max_ray_m: float | None,
) -> Path:
    """Run local preprocessing pipeline and export master static feature table."""
    sites_path = resolve_from_root(project_root, sites_rel)
    bathy_dir_path = resolve_from_root(project_root, bathy_dir_rel)
    bathy_out_dir_path = resolve_from_root(project_root, bathy_out_dir_rel)
    breaking_enabled, breaking_gamma, breaking_cfg_path = _resolve_breaking_config(
        project_root=project_root,
        preprocess_config_rel=preprocess_config_rel,
    )
    print(
        "Breaking config | "
        f"path={breaking_cfg_path} enabled={breaking_enabled} gamma={breaking_gamma:.5f}"
    )
    route_cfg, route_cfg_path = _resolve_route_curtain_config(
        project_root=project_root,
        preprocess_config_rel=preprocess_config_rel,
    )
    route_source_metadata, route_multi_source_cfg, route_source_warnings = (
        _build_route_source_metadata(
            project_root=project_root,
            sites_rel=sites_rel,
            training_config_rel=training_config_rel,
            route_cfg=route_cfg,
        )
    )
    effective_padding_m = float(route_cfg.get("padding_m", padding_m))
    print(
        "Route curtain config | "
        f"path={route_cfg_path} mode={route_cfg.get('mode')} "
        f"use_multi_source_config={bool(route_cfg.get('use_multi_source_config', False))} "
        f"padding_m={effective_padding_m:.1f}"
    )
    if route_multi_source_cfg:
        print(
            "Route curtain multi-source | "
            f"k_nearest={int(route_multi_source_cfg.get('k_nearest', 0))} "
            f"source_type={route_multi_source_cfg.get('source_type', 'nora3_wave')}"
        )
    for warning in route_source_warnings:
        print(f"WARNING: {warning}")

    _, generated_subgrid_path = generate_bathy_field.run_generate_bathy_field(
        sites_path=str(sites_path),
        bathy_dir_path=str(bathy_dir_path),
        padding_m=effective_padding_m,
        resolution_m=resolution_m,
        epsg=target_epsg,
        out_dir_path=str(bathy_out_dir_path),
        full_name=full_name,
        subgrid_name=subgrid_name,
    )

    generated_subgrid_rel = format_path_for_metadata(generated_subgrid_path, project_root)
    routing_bathy_rel = bathy_rel or generated_subgrid_rel

    routing_path = run_fetch_routing(
        project_root=project_root,
        sites_rel=sites_rel,
        bathy_rel=routing_bathy_rel,
        output_rel=routing_output_rel,
        target_epsg=target_epsg,
        bottleneck_weight=bottleneck_weight,
        bottleneck_epsilon_m=bottleneck_epsilon_m,
        route_mode=str(route_cfg.get("mode", "global")),
        route_source_metadata=route_source_metadata,
    )
    ray_path = run_ray_casting(
        project_root=project_root,
        sites_rel=sites_rel,
        bathy_rel=routing_bathy_rel,
        output_rel=ray_output_rel,
        target_epsg=target_epsg,
        angle_step_deg=angle_step_deg,
        ray_step_m=ray_step_m,
        max_ray_m=max_ray_m,
    )

    routing_payload = load_routing_payload(routing_path)
    routes_df = load_routing_routes(routing_path)
    route_qa_df = routing_payload.get("route_curtain_qa")
    if route_qa_df is None:
        route_qa_df = pd.DataFrame()
    elif not isinstance(route_qa_df, pd.DataFrame):
        raise TypeError("routing_features.pkl contains a non-DataFrame route_curtain_qa payload")
    route_qa_path = resolve_from_root(
        project_root, str(route_cfg.get("qa_out_path", "data/processed/route_curtain_qa.csv"))
    )
    route_qa_path.parent.mkdir(parents=True, exist_ok=True)
    route_qa_df.to_csv(route_qa_path, index=False)
    route_lengths = pd.to_numeric(
        route_qa_df.get("route_length_m", pd.Series(dtype=float)), errors="coerce"
    ).to_numpy(dtype=np.float64)
    finite_route_lengths = route_lengths[np.isfinite(route_lengths)]
    valid_routes = (
        int(
            pd.to_numeric(route_qa_df.get("route_valid", pd.Series(dtype=float)), errors="coerce")
            .fillna(0)
            .astype(int)
            .sum()
        )
        if not route_qa_df.empty
        else 0
    )
    invalid_routes = int(len(route_qa_df) - valid_routes) if not route_qa_df.empty else 0
    print(f"Route QA CSV: {route_qa_path}")
    print(f"Routed NORAC sites: {len(route_qa_df)} | valid={valid_routes} invalid={invalid_routes}")
    if finite_route_lengths.size:
        print(
            "Route length summary (m): "
            f"min={float(np.nanmin(finite_route_lengths)):.2f} "
            f"median={float(np.nanmedian(finite_route_lengths)):.2f} "
            f"max={float(np.nanmax(finite_route_lengths)):.2f}"
        )
    distance_cols = [
        col
        for col in route_qa_df.columns
        if col.startswith("source") and col.endswith("_distance_m")
    ]
    for col in sorted(
        distance_cols, key=lambda name: int(name.split("_")[0].replace("source", ""))
    ):
        vals = pd.to_numeric(route_qa_df[col], errors="coerce").to_numpy(dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        print(
            f"{col} summary (m): "
            f"min={float(np.nanmin(vals)):.2f} "
            f"median={float(np.nanmedian(vals)):.2f} "
            f"max={float(np.nanmax(vals)):.2f}"
        )

    ray_df = pd.read_csv(ray_path)
    master_df = build_master_static_features(
        routes_df=routes_df,
        ray_df=ray_df,
        breaking_enabled=breaking_enabled,
        breaking_gamma=breaking_gamma,
    )
    log_local_breaking_summary(master_df)

    master_path = resolve_from_root(project_root, master_output_rel)
    master_path.parent.mkdir(parents=True, exist_ok=True)
    master_df.to_csv(master_path, index=False)

    print(f"Saved master static features: {master_path}")
    print(f"Rows exported: {len(master_df)}")
    print(f"Columns exported: {len(master_df.columns)}")
    return master_path


def parse_args() -> argparse.Namespace:
    """Build CLI parser for local end-to-end feature generation."""
    parser = argparse.ArgumentParser(description="Run full local geometry feature pipeline")
    parser.add_argument("--sites", default="configs/sites.yaml", help="Path to site YAML")
    parser.add_argument(
        "--preprocess-config",
        default="configs/preprocess.yaml",
        help="Path to preprocess YAML (for static_features.breaking config)",
    )
    parser.add_argument(
        "--training-config",
        default="configs/training.yaml",
        help="Path to training YAML (used for local_k_nearest route curtain multi-source selection)",
    )
    parser.add_argument(
        "--bathy-dir", default="data/bathy", help="Directory with source XYZ bathymetry tiles"
    )
    parser.add_argument(
        "--bathy",
        default=None,
        help="Optional bathymetry file for fetch/ray (defaults to generated site subgrid)",
    )
    parser.add_argument(
        "--bathy-out-dir",
        default="data/processed/bathy",
        help="Output directory for generated bathymetry NPZ files",
    )
    parser.add_argument(
        "--full-name", default="bathy_field_full.npz", help="Generated full bathy filename"
    )
    parser.add_argument(
        "--subgrid-name",
        default="bathy_field_project_site.npz",
        help="Generated site subgrid filename",
    )
    parser.add_argument(
        "--routing-output",
        default="data/processed/routing_features.pkl",
        help="Routing output pickle path",
    )
    parser.add_argument(
        "--ray-output",
        default="data/processed/ray_features.csv",
        help="Ray feature output CSV path",
    )
    parser.add_argument(
        "--master-output",
        default="data/processed/master_static_features.csv",
        help="Merged master static feature CSV path",
    )
    parser.add_argument("--epsg", type=int, default=TARGET_EPSG, help="Projected CRS EPSG code")
    parser.add_argument(
        "--padding", type=float, default=1_000.0, help="Site-padding for bathy subgrid (m)"
    )
    parser.add_argument("--resolution", type=float, default=50.0, help="Bathy grid resolution (m)")
    parser.add_argument(
        "--bottleneck-weight",
        type=float,
        default=BOTTLENECK_WEIGHT,
        help="Routing bottleneck penalty weight",
    )
    parser.add_argument(
        "--bottleneck-epsilon-m",
        type=float,
        default=BOTTLENECK_EPSILON_M,
        help="Routing bottleneck penalty epsilon in meters",
    )
    parser.add_argument(
        "--angle-step-deg",
        type=float,
        default=10.0,
        help="Angular step for ray-casting (degrees)",
    )
    parser.add_argument(
        "--ray-step-m", type=float, default=50.0, help="Distance step along each ray (m)"
    )
    parser.add_argument(
        "--max-ray-m",
        type=float,
        default=None,
        help="Optional maximum ray distance (m). Defaults to grid diagonal.",
    )
    return parser.parse_args()


def main() -> None:
    """Run full local feature orchestration pipeline."""
    args = parse_args()
    project_root = resolve_project_root(Path(__file__).resolve())
    run_pipeline(
        project_root=project_root,
        sites_rel=args.sites,
        preprocess_config_rel=args.preprocess_config,
        training_config_rel=args.training_config,
        bathy_dir_rel=args.bathy_dir,
        bathy_rel=args.bathy,
        bathy_out_dir_rel=args.bathy_out_dir,
        full_name=args.full_name,
        subgrid_name=args.subgrid_name,
        routing_output_rel=args.routing_output,
        ray_output_rel=args.ray_output,
        master_output_rel=args.master_output,
        target_epsg=args.epsg,
        padding_m=args.padding,
        resolution_m=args.resolution,
        bottleneck_weight=args.bottleneck_weight,
        bottleneck_epsilon_m=args.bottleneck_epsilon_m,
        angle_step_deg=args.angle_step_deg,
        ray_step_m=args.ray_step_m,
        max_ray_m=args.max_ray_m,
    )


if __name__ == "__main__":
    main()
