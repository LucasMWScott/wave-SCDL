"""Reusable explainability helpers for coastal transformer runs."""

from __future__ import annotations

import json
import importlib
import pickle
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader

try:
    import shap  # type: ignore
except Exception:
    shap = None

try:
    import umap  # type: ignore
except Exception:
    umap = None

try:
    import xarray as xr
except Exception:
    xr = None

from src.data_pipeline import (
    PointCentricArrays,
    PointCentricWindowDataset,
    _is_static_tensor_standardized,
    _load_target_scaler_metadata,
    _load_transfer_target_scaler_metadata,
    _override_split_indices,
    _resolve_runtime_source_geometry_features_flag,
    apply_runtime_static_ablation,
    build_bin_centers,
    build_split_dataloader,
    load_point_centric_arrays,
    resolve_output_indices,
    resolve_sequence_length,
    resolve_static_ablation_config_path,
)
from src.evaluate import (
    _extract_state_dict,
    _load_checkpoint_compat,
    _load_state_dict_best_effort,
)
from src.config_resolution import resolve_config
from src.independent_target_mode import load_model_from_training_metadata
from src.models import build_model_from_config
from src.preprocessing.transfer_targets import resolve_targets_config
from src.transfer_runtime import load_transfer_scaler_stats

from . import explainability_plotting as _plotting

# Long-lived notebook kernels can hold an older plotting module object across edits.
# Reload once here so newly added helpers are available before we bind local aliases.
_REQUIRED_PLOTTING_HELPERS = (
    "plot_top_static_shap_beeswarm",
    "plot_sitewise_error_vs_analog_distance_grid",
)
if any(not hasattr(_plotting, helper_name) for helper_name in _REQUIRED_PLOTTING_HELPERS):
    _plotting = importlib.reload(_plotting)

plot_ale_curve = _plotting.plot_ale_curve
plot_attention_by_head = _plotting.plot_attention_by_head
plot_attention_regime_comparison = _plotting.plot_attention_regime_comparison
plot_bathy_attribution_map = _plotting.plot_bathy_attribution_map
plot_branch_ablation_bars = _plotting.plot_branch_ablation_bars
plot_grouped_shap_importance_heatmap = _plotting.plot_grouped_shap_importance_heatmap
plot_good_bad_attribution_delta = _plotting.plot_good_bad_attribution_delta
plot_grouped_shap_bar = _plotting.plot_grouped_shap_bar
plot_grouped_shap_distribution = _plotting.plot_grouped_shap_distribution
plot_top_static_shap_beeswarm = _plotting.plot_top_static_shap_beeswarm
plot_static_shap_beeswarm = _plotting.plot_static_shap_beeswarm
plot_ale_with_histogram = _plotting.plot_ale_with_histogram
save_figure_variants = _plotting.save_figure_variants
plot_static_embedding_space = _plotting.plot_static_embedding_space
plot_sitewise_error_vs_analog_distance_grid = _plotting.plot_sitewise_error_vs_analog_distance_grid
plot_temporal_feature_lag_heatmap = _plotting.plot_temporal_feature_lag_heatmap
plot_temporal_source_lag_heatmap = _plotting.plot_temporal_source_lag_heatmap


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HEADS = ("hs", "tp", "dir", "dp")
DEFAULT_STATIC_GROUP_PATTERNS: dict[str, tuple[str, ...]] = {
    "local_site_geometry": ("local_depth", "dist_to_coast", "nearest_shore", "snap_distance"),
    "ray_fetches": (
        "ray_fetch",
        "fetch_mean",
        "fetch_max",
        "fetch_min",
        "fetch_std",
        "fetch_cv",
        "resultant",
        "open_sector_width",
        "open_sector_fraction",
        "dominant_fetch",
    ),
    "ray_depths_bathymetry": ("ray_min_depth", "breaking", "depth_profile", "shallow", "deep"),
    "ray_slopes_curvature": ("ray_max_slope", "ray_max_laplacian", "curvature", "steepness"),
    "path_route_geometry": ("path_", "route_", "tortuosity", "deflection", "approach"),
    "bottleneck_funneling": ("bottleneck", "funneling", "choke"),
    "porosity_openness": (
        "porosity",
        "closed_sector",
        "land_block",
        "land_hit",
        "island",
        "openness",
    ),
    "fjordness_regime": ("fjordness", "site_regime"),
    "source_geometry": ("source_",),
    "shoreline_geometry": ("shore", "coast"),
}
DEFAULT_DYNAMIC_GROUP_PATTERNS: dict[str, tuple[str, ...]] = {
    "wave_height_energy": ("hs", "energy"),
    "period_frequency": ("tp", "tm1", "tm2", "tmp", "freq", "period"),
    "wave_direction": ("pdir", "thq", "dir_", "direction"),
    "wind": ("wind_speed", "wind_direction"),
    "swell_partition": ("swell", "sea"),
    "local_wind": ("local_wind",),
    "tide_current": ("tide", "current"),
    "source_geometry_dynamic": ("fetch_at_", "blocking_at_", "slope_at_"),
    "seasonality": ("time_sin", "time_cos"),
}
DEFAULT_EXPERT_REGIME_GROUP_PATTERNS: dict[str, tuple[str, ...]] = {
    "swell_proxy": (
        "hs_swell",
        "tp_swell",
        "thq_swell",
        "fetch_at_swell_direction",
        "blocking_at_swell_direction",
        "slope_at_swell_direction",
    ),
    "local_wind_proxy": (
        "local_wind_speed_10m",
        "local_wind_dir",
        "fetch_at_local_wind_direction",
        "blocking_at_local_wind_direction",
        "slope_at_local_wind_direction",
    ),
    "windsea_proxy": (
        "hs_sea",
        "tp_sea",
        "thq_sea",
        "wind_speed_10m",
        "wind_direction_10m",
        "fetch_at_windwave_direction",
        "blocking_at_windwave_direction",
        "slope_at_windwave_direction",
    ),
    "background_wave": (
        "hs",
        "tp",
        "tm1",
        "tm2",
        "tmp",
        "pdir",
        "thq",
    ),
    "seasonality": ("time_sin", "time_cos"),
}

TARGET_DISPLAY_NAMES: dict[str, str] = {
    "hs": "Hs",
    "tp": "Tp",
    "dir": "Dir",
    "dp": "Dp",
}

STATIC_SHAP_TARGET_DISPLAY_NAMES: dict[str, str] = {
    "hs": "Hs",
    "tp": "Tp",
    "dir": "Mean direction",
    "dp": "Peak direction",
}

STATIC_GROUP_DISPLAY_NAMES: dict[str, str] = {
    "local_site_geometry": "Local site geometry",
    "ray_fetches": "Fetch geometry",
    "ray_depths_bathymetry": "Bathymetry exposure",
    "ray_slopes_curvature": "Slope and curvature",
    "path_route_geometry": "Path-route geometry",
    "bottleneck_funneling": "Bottleneck and funneling",
    "porosity_openness": "Porosity and openness",
    "fjordness_regime": "Fjordness and regime",
    "source_geometry": "Source geometry",
    "shoreline_geometry": "Shoreline geometry",
    "misc_static": "Other static features",
}

STATIC_FEATURE_DISPLAY_NAMES: dict[str, str] = {
    "local_depth_m": "Local depth",
    "static_local_depth_m": "Nearshore depth",
    "ray_fetch_mean_m": "Mean ray fetch",
    "ray_fetch_max_m": "Maximum ray fetch",
    "ray_fetch_min_m": "Minimum ray fetch",
    "open_sector_width_deg": "Open-sector width",
    "open_sector_fraction": "Open-sector fraction",
    "closed_sector_fraction": "Closed-sector fraction",
    "path_length_m": "Path length",
    "path_direct_distance_m": "Path direct distance",
    "path_tortuosity_ratio": "Path tortuosity ratio",
    "path_bottleneck_m": "Path bottleneck width",
    "bottleneck_to_path_ratio": "Bottleneck-to-path ratio",
    "bottleneck_to_fetch_ratio": "Bottleneck-to-fetch ratio",
    "funneling_ratio": "Funneling ratio",
    "funneling_log": "Funneling index",
    "fetch_resultant_length": "Fetch resultant length",
    "fetch_max_over_mean": "Fetch max-over-mean ratio",
    "fetch_std_m": "Fetch standard deviation",
    "fetch_cv": "Fetch coefficient of variation",
    "fetch_directional_entropy": "Fetch directional entropy",
    "ray_hit_land_fraction": "Ray land-hit fraction",
    "static_porosity_1km": "Porosity within 1 km",
    "static_porosity_500m": "Porosity within 500 m",
    "static_porosity_2km": "Porosity within 2 km",
    "local_breaking_hs_cap": "Local breaking-wave height cap",
}

STATIC_FEATURE_PLOT_METADATA: dict[str, dict[str, Any]] = {
    "local_depth_m": {"label": "Local depth", "unit": "m", "display_unit": "m", "scale": 1.0},
    "static_local_depth_m": {
        "label": "Nearshore depth",
        "unit": "m",
        "display_unit": "m",
        "scale": 1.0,
    },
    "ray_fetch_mean_m": {
        "label": "Mean ray fetch",
        "unit": "m",
        "display_unit": "km",
        "scale": 1e-3,
    },
    "ray_fetch_max_m": {
        "label": "Maximum ray fetch",
        "unit": "m",
        "display_unit": "km",
        "scale": 1e-3,
    },
    "ray_fetch_min_m": {
        "label": "Minimum ray fetch",
        "unit": "m",
        "display_unit": "km",
        "scale": 1e-3,
    },
    "open_sector_width_deg": {
        "label": "Open-sector width",
        "unit": "degrees",
        "display_unit": "degrees",
        "scale": 1.0,
    },
    "path_length_m": {"label": "Path length", "unit": "m", "display_unit": "km", "scale": 1e-3},
    "snap_distance_m": {"label": "Snap distance", "unit": "m", "display_unit": "km", "scale": 1e-3},
    "path_bottleneck_m": {
        "label": "Path bottleneck width",
        "unit": "m",
        "display_unit": "km",
        "scale": 1e-3,
    },
    "static_dist_to_coast_m": {
        "label": "Distance to coast",
        "unit": "m",
        "display_unit": "km",
        "scale": 1e-3,
    },
}

ALE_TARGET_TITLES: dict[str, str] = {
    "hs": "predicted $H_s$",
    "tp": "predicted $T_p$",
    "dir": "directional transfer",
    "dp": "peak-direction transfer",
}

ALE_TARGET_Y_LABELS: dict[str, str] = {
    "hs": "ALE in predicted $H_s$ (m)",
    "tp": "ALE in predicted $T_p$ (s)",
    "dir": "ALE in directional transfer (degrees)",
    "dp": "ALE in peak-direction transfer (degrees)",
}

DEFAULT_STATIC_ALE_FEATURE_SELECTIONS: dict[str, tuple[str, ...]] = {
    "hs": (
        "ray_fetch_mean_m",
        "ray_fetch_max_m",
        "local_depth_m",
        "open_sector_width_deg",
        "path_length_m",
        "funneling_ratio",
        "static_porosity_1km",
    ),
    "tp": (
        "ray_fetch_mean_m",
        "ray_fetch_max_m",
        "local_depth_m",
        "open_sector_width_deg",
        "path_length_m",
    ),
    "dir": (
        "open_sector_width_deg",
        "ray_fetch_mean_m",
        "path_length_m",
        "local_depth_m",
    ),
    "dp": (
        "open_sector_width_deg",
        "ray_fetch_max_m",
        "ray_fetch_mean_m",
        "local_depth_m",
    ),
}


@dataclass
class ResultsBundle:
    """Resolved metadata and artifact locations for one results folder."""

    results_dir: Path
    config_path: Path | None
    checkpoint_path: Path | None
    point_centric_dir: Path | None
    training_metadata: dict
    observed_model_io: dict
    config: dict
    prediction_files: dict[str, Path]
    explainability_dir: Path
    cache_dir: Path


@dataclass
class ExplainabilityModelBundle:
    """Loaded model plus runtime context for explainability methods."""

    bundle: ResultsBundle
    model: torch.nn.Module
    device: torch.device
    arrays: PointCentricArrays
    loader: DataLoader
    dataset: PointCentricWindowDataset
    transfer_scaler_stats: dict[int, tuple[float, float]]
    target_scaler_meta: dict | None
    split: str


def _read_json(path: str | Path) -> dict:
    with Path(path).open("r") as fh:
        return json.load(fh) or {}


def _read_yaml(path: str | Path) -> dict:
    with Path(path).open("r") as fh:
        return yaml.safe_load(fh) or {}


def _resolve_path(path_like: str | Path | None, *, anchor: str | Path | None = None) -> Path | None:
    if path_like in {None, ""}:
        return None
    path = Path(path_like).expanduser()
    if path.is_absolute():
        return path.resolve()
    candidates: list[Path] = [(Path.cwd() / path).resolve(), (REPO_ROOT / path).resolve()]
    if anchor is not None:
        anchor_path = Path(anchor).resolve()
        base = anchor_path if anchor_path.is_dir() else anchor_path.parent
        candidates.insert(1, (base / path).resolve())
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _discover_prediction_files(results_dir: Path) -> dict[str, Path]:
    discovered: dict[str, Path] = {}
    for split in ("train", "val", "test"):
        csv_path = results_dir / f"predictions_{split}.csv"
        nc_path = results_dir / f"predictions_{split}.nc"
        if csv_path.exists():
            discovered[f"{split}_csv"] = csv_path
        if nc_path.exists():
            discovered[f"{split}_nc"] = nc_path
    return discovered


def _ensure_prediction_angles(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for prefix in ("dir", "dp"):
        sin_col = f"pred_{prefix}_sin"
        cos_col = f"pred_{prefix}_cos"
        if (
            sin_col in out.columns
            and cos_col in out.columns
            and f"pred_{prefix}_deg" not in out.columns
        ):
            out[f"pred_{prefix}_deg"] = (
                np.degrees(np.arctan2(out[sin_col], out[cos_col])) + 360.0
            ) % 360.0
        sin_col = f"target_{prefix}_sin"
        cos_col = f"target_{prefix}_cos"
        if (
            sin_col in out.columns
            and cos_col in out.columns
            and f"target_{prefix}_deg" not in out.columns
        ):
            out[f"target_{prefix}_deg"] = (
                np.degrees(np.arctan2(out[sin_col], out[cos_col])) + 360.0
            ) % 360.0
    return out


def _prediction_frame_for_split(results_dir: Path, split: str) -> pd.DataFrame:
    csv_path = results_dir / f"predictions_{split}.csv"
    if csv_path.exists():
        return _ensure_prediction_angles(pd.read_csv(csv_path))
    nc_path = results_dir / f"predictions_{split}.nc"
    if nc_path.exists():
        if xr is None:
            raise RuntimeError("xarray is required to load NetCDF prediction files")
        with xr.open_dataset(nc_path) as ds:
            return _ensure_prediction_angles(ds.to_dataframe().reset_index())
    raise FileNotFoundError(f"No predictions_{split}.csv or .nc found under {results_dir}")


def _results_dir_label(results_dir: str | Path) -> Path:
    resolved = _resolve_path(results_dir)
    if resolved is None:
        raise ValueError("results_dir is required")
    return resolved


def _metadata_point_centric_dir(training_metadata: Mapping[str, Any]) -> str | None:
    runtime_cfg = (
        (training_metadata.get("config") or {}) if isinstance(training_metadata, Mapping) else {}
    )
    data_cfg = (runtime_cfg.get("data") or {}) if isinstance(runtime_cfg, Mapping) else {}
    raw = data_cfg.get("point_centric_dir")
    return str(raw).strip() if raw else None


def _metadata_config_path(training_metadata: Mapping[str, Any]) -> str | None:
    runtime = (
        (training_metadata.get("runtime") or {}) if isinstance(training_metadata, Mapping) else {}
    )
    raw = runtime.get("config_path")
    return str(raw).strip() if raw else None


def _metadata_checkpoint_path(training_metadata: Mapping[str, Any]) -> str | None:
    runtime = (
        (training_metadata.get("runtime") or {}) if isinstance(training_metadata, Mapping) else {}
    )
    raw = runtime.get("checkpoint_path")
    return str(raw).strip() if raw else None


def load_results_bundle(
    results_dir: str | Path,
    training_config_path: str | Path | None = None,
    point_centric_dir: str | Path | None = None,
) -> ResultsBundle:
    """Load the canonical result artifacts for one run."""
    resolved_results_dir = _results_dir_label(results_dir)
    if not resolved_results_dir.exists():
        raise FileNotFoundError(f"Missing RESULTS_DIR: {resolved_results_dir}")

    training_meta_path = resolved_results_dir / "training_run_metadata.json"
    if not training_meta_path.exists():
        raise FileNotFoundError(f"Missing training_run_metadata.json under {resolved_results_dir}")
    training_metadata = _read_json(training_meta_path)

    observed_model_io_path = resolved_results_dir / "observed_model_io.json"
    observed_model_io = (
        _read_json(observed_model_io_path) if observed_model_io_path.exists() else {}
    )

    resolved_config_path = _resolve_path(
        training_config_path or _metadata_config_path(training_metadata),
        anchor=resolved_results_dir,
    )
    config = (
        _read_yaml(resolved_config_path)
        if resolved_config_path and resolved_config_path.exists()
        else (training_metadata.get("config") or {})
    )

    resolved_point_centric_dir = _resolve_path(
        point_centric_dir or _metadata_point_centric_dir(training_metadata),
        anchor=resolved_config_path or resolved_results_dir,
    )
    resolved_checkpoint_path = _resolve_path(
        _metadata_checkpoint_path(training_metadata),
        anchor=resolved_results_dir,
    )
    if resolved_checkpoint_path is None or not resolved_checkpoint_path.exists():
        checkpoints = sorted(
            resolved_results_dir.glob("*.pt"), key=lambda path: path.stat().st_mtime, reverse=True
        )
        resolved_checkpoint_path = checkpoints[0] if checkpoints else None

    explainability_dir = resolved_results_dir / "explainability"
    cache_dir = explainability_dir / "cache"
    prediction_files = _discover_prediction_files(resolved_results_dir)
    return ResultsBundle(
        results_dir=resolved_results_dir,
        config_path=resolved_config_path,
        checkpoint_path=resolved_checkpoint_path,
        point_centric_dir=resolved_point_centric_dir,
        training_metadata=training_metadata,
        observed_model_io=observed_model_io,
        config=config,
        prediction_files=prediction_files,
        explainability_dir=explainability_dir,
        cache_dir=cache_dir,
    )


def resolve_target_heads(bundle: ResultsBundle | Mapping[str, Any]) -> list[str]:
    """Resolve the active target heads for the loaded run."""
    if isinstance(bundle, ResultsBundle):
        config = bundle.config
        observed = bundle.observed_model_io
    else:
        config = dict(bundle)
        observed = {}

    targets_cfg = resolve_targets_config((config.get("data") or {}))
    if str(targets_cfg.get("mode", "physical")).strip().lower() in {
        "physical",
        "physical_and_transfer",
        "transfer",
    }:
        return list(DEFAULT_HEADS)

    inputs = (
        ((observed.get("targets") or {}).get("tensors") or {})
        if isinstance(observed, Mapping)
        else {}
    )
    available = [name for name in DEFAULT_HEADS if name in inputs]
    return available or list(DEFAULT_HEADS)


def resolve_active_branches(bundle: ResultsBundle) -> dict[str, Any]:
    """Infer which branches and modalities are active for a run."""
    config = bundle.config or {}
    data_cfg = config.get("data", {}) or {}
    model_cfg = (config.get("model", {}) or {}).get("coastal_transformer", {}) or {}
    multi_cfg = model_cfg.get("multi_source", {}) or {}
    observed_inputs = (
        (bundle.observed_model_io.get("inputs_passed_to_model") or {})
        if bundle.observed_model_io
        else {}
    )

    static_enabled = bool(model_cfg.get("use_static", data_cfg.get("use_static_features", True)))
    bathy_enabled = bool(data_cfg.get("use_bathymetry", False)) and bool(
        (model_cfg.get("bathy", {}) or {}).get("enabled", True)
    )
    source_geometry_enabled = bool(
        multi_cfg.get("use_geometry_features", data_cfg.get("use_geometry", False))
    )
    multi_source_enabled = bool((data_cfg.get("multi_source", {}) or {}).get("enabled", False))
    sequence_length = int(resolve_sequence_length(data_cfg, default=24))

    source_count = None
    x_dynamic_sources = (
        observed_inputs.get("x_dynamic_sources", {}) if isinstance(observed_inputs, Mapping) else {}
    )
    if (
        x_dynamic_sources.get("present")
        and isinstance(x_dynamic_sources.get("shape"), list)
        and len(x_dynamic_sources["shape"]) >= 3
    ):
        source_count = int(x_dynamic_sources["shape"][2])

    return {
        "static": static_enabled,
        "bathy": bathy_enabled,
        "source_geometry": source_geometry_enabled,
        "multi_source": multi_source_enabled,
        "sequence_length": sequence_length,
        "num_sources": source_count,
        "sequence_encoder_type": str(
            (bundle.observed_model_io.get("model_flags") or {}).get(
                "sequence_encoder_type", "unknown"
            )
        ),
        "decoder_type": str(
            (bundle.observed_model_io.get("model_flags") or {}).get(
                "decoder_type", "cross_attention"
            )
        ),
        "target_heads": resolve_target_heads(bundle),
        "target_mode": str(resolve_targets_config(data_cfg).get("mode", "physical")),
    }


def build_branch_availability_report(bundle: ResultsBundle) -> pd.DataFrame:
    branches = resolve_active_branches(bundle)
    rows = []
    for key in ("static", "bathy", "source_geometry", "multi_source"):
        enabled = bool(branches.get(key, False))
        rows.append(
            {
                "branch": key,
                "enabled": enabled,
                "note": "available" if enabled else f"not applicable: branch disabled in this run",
            }
        )
    rows.append(
        {
            "branch": "target_mode",
            "enabled": True,
            "note": str(branches.get("target_mode", "unknown")),
        }
    )
    rows.append(
        {
            "branch": "sequence_length",
            "enabled": True,
            "note": str(branches.get("sequence_length", "unknown")),
        }
    )
    return pd.DataFrame(rows)


def load_prediction_frame(
    bundle_or_dir: ResultsBundle | str | Path, split: str = "val"
) -> pd.DataFrame:
    """Load predictions for one split and derive circular degree columns."""
    bundle = (
        bundle_or_dir
        if isinstance(bundle_or_dir, ResultsBundle)
        else load_results_bundle(bundle_or_dir)
    )
    if split == "all":
        frames = []
        for candidate in ("train", "val", "test"):
            key = f"{candidate}_csv"
            nc_key = f"{candidate}_nc"
            if key in bundle.prediction_files or nc_key in bundle.prediction_files:
                frame = _prediction_frame_for_split(bundle.results_dir, candidate)
                frame["split"] = candidate
                frames.append(frame)
        if not frames:
            raise FileNotFoundError(f"No prediction exports found under {bundle.results_dir}")
        return pd.concat(frames, ignore_index=True)
    return _prediction_frame_for_split(bundle.results_dir, split)


def _extract_feature_names(observed_model_io: dict, key: str) -> list[str]:
    payload = (observed_model_io.get("inputs_passed_to_model") or {}).get(key) or {}
    return [str(item) for item in (payload.get("feature_names") or [])]


def load_feature_metadata(bundle: ResultsBundle) -> dict[str, Any]:
    """Load feature-name metadata from the observed manifest or point-centric arrays."""
    metadata = {
        "static_feature_names": _extract_feature_names(bundle.observed_model_io, "x_static"),
        "dynamic_feature_names": _extract_feature_names(bundle.observed_model_io, "x_dynamic"),
        "source_feature_names": _extract_feature_names(
            bundle.observed_model_io, "x_dynamic_sources"
        ),
        "source_geometry_feature_names": _extract_feature_names(
            bundle.observed_model_io, "source_geometry"
        ),
        "bathy_channel_names": _extract_feature_names(bundle.observed_model_io, "x_bathy"),
    }
    if any(metadata.values()):
        return metadata
    if bundle.point_centric_dir is None:
        return metadata
    arrays = load_point_centric_arrays(str(bundle.point_centric_dir))
    return {
        "static_feature_names": list(getattr(arrays, "static_feature_names", []) or []),
        "dynamic_feature_names": list(getattr(arrays, "dynamic_feature_names", []) or []),
        "source_feature_names": list(getattr(arrays, "source_feature_names", []) or []),
        "source_geometry_feature_names": list(
            getattr(arrays, "source_geometry_feature_names", []) or []
        ),
        "bathy_channel_names": list(getattr(arrays, "bathy_channel_names", []) or []),
    }


def _load_arrays(bundle: ResultsBundle) -> PointCentricArrays:
    if bundle.point_centric_dir is None:
        raise ValueError("Could not infer point-centric directory from results artifacts")
    config = _resolved_runtime_config(bundle)
    arrays = load_point_centric_arrays(
        str(bundle.point_centric_dir),
        bathy_channels=((config.get("model", {}) or {}).get("coastal_transformer", {}) or {})
        .get("bathy", {})
        .get("channels"),
        bathy_in_channels=(
            ((config.get("model", {}) or {}).get("coastal_transformer", {}) or {}).get("bathy", {})
            or {}
        ).get("in_channels"),
    )
    return apply_runtime_static_ablation(arrays, resolve_static_ablation_config_path(config))


def _resolved_runtime_config(bundle: ResultsBundle) -> dict:
    """Return a config copy with repo-relative runtime paths resolved to absolute paths.

    Notebook kernels often start in `notebooks/`, while the training/evaluation
    pipeline typically runs from the repo root. Normalizing these paths here
    keeps dataset/scaler lookups aligned with the saved run artifacts.
    """
    config = resolve_config(deepcopy(bundle.config or {}))
    data_cfg = config.setdefault("data", {})
    logging_cfg = config.setdefault("logging", {})

    if bundle.point_centric_dir is not None:
        data_cfg["point_centric_dir"] = str(bundle.point_centric_dir)

    static_features_csv = data_cfg.get("static_features_csv")
    resolved_static = _resolve_path(
        static_features_csv, anchor=bundle.config_path or bundle.results_dir
    )
    if resolved_static is not None:
        data_cfg["static_features_csv"] = str(resolved_static)

    sites_config = data_cfg.get("sites_config")
    resolved_sites = _resolve_path(sites_config, anchor=bundle.config_path or bundle.results_dir)
    if resolved_sites is not None:
        data_cfg["sites_config"] = str(resolved_sites)

    static_ablation_config = data_cfg.get("static_ablation_config")
    resolved_ablation = _resolve_path(
        static_ablation_config, anchor=bundle.config_path or bundle.results_dir
    )
    if resolved_ablation is not None:
        data_cfg["static_ablation_config"] = str(resolved_ablation)

    logging_cfg["output_dir"] = str(bundle.results_dir)
    if bundle.checkpoint_path is not None:
        logging_cfg["checkpoint_name"] = str(bundle.checkpoint_path.name)

    return config


def _build_all_split_dataset(
    bundle: ResultsBundle,
    arrays: PointCentricArrays,
    config: dict,
) -> tuple[DataLoader, PointCentricWindowDataset]:
    train_cfg = config.get("training", {}) or {}
    data_cfg = config.get("data", {}) or {}
    arrays_for_ds = _override_split_indices(
        arrays, "train", np.arange(len(arrays.timestamps), dtype=int)
    )
    target_mode = str(resolve_targets_config(data_cfg).get("mode", "physical"))
    output_columns = data_cfg.get(
        "output_columns", ["hs", "tp", "dir_sin", "dir_cos", "dp_sin", "dp_cos"]
    )
    output_indices = (
        resolve_output_indices(arrays.target_feature_names, output_columns)
        if target_mode == "physical"
        else []
    )
    loss_cfg = train_cfg.get("loss", {}) or {}
    hybrid_cfg = loss_cfg.get("blueprint_hybrid", {}) or {}
    loss_type = str(loss_cfg.get("type", train_cfg.get("loss_type", "mse"))).strip().lower()
    if target_mode == "physical" and loss_type in {
        "blueprint_hybrid",
        "hybrid_blueprint",
        "hybrid_multitask",
    }:
        hybrid_task_config = {
            "tp_bin_centers": build_bin_centers(
                0.0, 25.0, int(hybrid_cfg.get("num_tp_bins", 32)), circular=False
            ),
            "dp_bin_centers": build_bin_centers(
                0.0, 360.0, int(hybrid_cfg.get("num_dp_bins", 36)), circular=True
            ),
            "label_smoothing_sigma": float(hybrid_cfg.get("label_smoothing_sigma", 0.8)),
        }
    else:
        hybrid_task_config = {}

    dataset = PointCentricWindowDataset(
        arrays=arrays_for_ds,
        split_name="train",
        seq_len=resolve_sequence_length(data_cfg, default=24),
        sites=list(arrays.target_sites),
        output_indices=output_indices,
        target_mode=target_mode,
        target_columns=data_cfg.get(
            "target_columns", data_cfg.get("output_columns", ["hs", "tp", "dir", "dp"])
        ),
        target_scaler_meta=_load_target_scaler_metadata(str(bundle.point_centric_dir))
        if bundle.point_centric_dir
        else None,
        transfer_target_scaler_meta=_load_transfer_target_scaler_metadata(
            str(bundle.point_centric_dir)
        )
        if bundle.point_centric_dir
        else None,
        hybrid_task_config=hybrid_task_config,
        return_concat_dynamic_static=bool(data_cfg.get("return_concat_dynamic_static", True)),
        use_multi_source=bool((data_cfg.get("multi_source", {}) or {}).get("enabled", False)),
        use_source_geometry=_resolve_runtime_source_geometry_features_flag(config),
        use_bathymetry=bool(data_cfg.get("use_bathymetry", False)),
        bathymetry_shuffle_mode=str(data_cfg.get("bathymetry_shuffle", "none")),
        bathymetry_train_jitter_cells=int(data_cfg.get("bathymetry_train_jitter_cells", 0)),
        bathymetry_noise_std=float(data_cfg.get("bathymetry_noise_std", 0.0)),
        random_seed=int(train_cfg.get("seed", 42)),
        sample_filter_cfg=(data_cfg.get("sample_filter", {}) or {}),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(train_cfg.get("batch_size", 64)),
        shuffle=False,
        num_workers=int(data_cfg.get("num_workers", 0)),
        pin_memory=bool(data_cfg.get("pin_memory", False)),
        drop_last=False,
    )
    return loader, dataset


def load_split_dataset(
    bundle_or_dir: ResultsBundle | str | Path,
    split: str = "val",
) -> tuple[PointCentricArrays, DataLoader, PointCentricWindowDataset]:
    """Load the point-centric arrays and one dataset split for explainability."""
    bundle = (
        bundle_or_dir
        if isinstance(bundle_or_dir, ResultsBundle)
        else load_results_bundle(bundle_or_dir)
    )
    arrays = _load_arrays(bundle)
    runtime_config = _resolved_runtime_config(bundle)
    if split == "all":
        loader, dataset = _build_all_split_dataset(bundle, arrays, runtime_config)
        return arrays, loader, dataset
    loader, dataset = build_split_dataloader(
        arrays, runtime_config, split_name=split, shuffle=False
    )
    return arrays, loader, dataset


def load_model_for_explainability(
    results_dir: str | Path,
    *,
    split: str = "val",
    device: str = "cuda",
    training_config_path: str | Path | None = None,
    point_centric_dir: str | Path | None = None,
) -> ExplainabilityModelBundle:
    """Load model, arrays, and split data for explainability analysis."""
    bundle = load_results_bundle(
        results_dir,
        training_config_path=training_config_path,
        point_centric_dir=point_centric_dir,
    )
    arrays, loader, dataset = load_split_dataset(bundle, split=split)
    runtime_config = _resolved_runtime_config(bundle)
    runtime_device = torch.device(
        "cuda" if str(device).lower() == "cuda" and torch.cuda.is_available() else "cpu"
    )
    sample = dataset[0]
    output_dim = 4 if isinstance(sample["y"], dict) else int(sample["y"].shape[-1])
    dynamic_input_dim = int(sample["x_dynamic"].shape[-1]) if "x_dynamic" in sample else 0
    source_dynamic_input_dim = (
        int(sample["x_dynamic_sources"].shape[-1]) if "x_dynamic_sources" in sample else None
    )
    source_geometry_input_dim = (
        int(sample["source_geometry"].shape[-1]) if "source_geometry" in sample else None
    )

    def _build_model() -> torch.nn.Module:
        return build_model_from_config(
            config=runtime_config,
            dynamic_input_dim=dynamic_input_dim,
            static_input_dim=int(sample["x_static"].shape[-1]) if "x_static" in sample else 0,
            output_dim=output_dim,
            dynamic_feature_names=getattr(arrays, "dynamic_feature_names", None),
            source_dynamic_input_dim=source_dynamic_input_dim,
            source_geometry_input_dim=source_geometry_input_dim,
            source_feature_names=getattr(arrays, "source_feature_names", None),
        ).to(runtime_device)

    model = load_model_from_training_metadata(
        training_metadata=bundle.training_metadata,
        build_model=_build_model,
        checkpoint_loader=lambda path: _load_checkpoint_compat(path, runtime_device),
        state_dict_getter=_extract_state_dict,
        state_dict_loader=lambda target_model, state_dict: _load_state_dict_best_effort(
            target_model, state_dict
        ),
    )
    model.eval()
    transfer_scaler_stats = load_transfer_scaler_stats(
        str(bundle.point_centric_dir) if bundle.point_centric_dir else None
    )
    target_scaler_meta = (
        _load_target_scaler_metadata(str(bundle.point_centric_dir))
        if bundle.point_centric_dir
        else None
    )
    return ExplainabilityModelBundle(
        bundle=bundle,
        model=model,
        device=runtime_device,
        arrays=arrays,
        loader=loader,
        dataset=dataset,
        transfer_scaler_stats=transfer_scaler_stats,
        target_scaler_meta=target_scaler_meta,
        split=split,
    )


def circular_error_deg(
    y_true: np.ndarray | torch.Tensor, y_pred: np.ndarray | torch.Tensor
) -> np.ndarray | torch.Tensor:
    """Return wrapped signed prediction error in degrees."""
    if torch.is_tensor(y_true) or torch.is_tensor(y_pred):
        true = y_true if torch.is_tensor(y_true) else torch.as_tensor(y_true)
        pred = (
            y_pred
            if torch.is_tensor(y_pred)
            else torch.as_tensor(y_pred, device=true.device, dtype=true.dtype)
        )
        return ((pred - true + 180.0) % 360.0) - 180.0
    return ((np.asarray(y_pred) - np.asarray(y_true) + 180.0) % 360.0) - 180.0


def _site_metric_target_columns(target: str) -> tuple[str, str]:
    key = str(target).strip().lower()
    if key in {"dir", "dp"}:
        return f"target_{key}_deg", f"pred_{key}_deg"
    return f"target_{key}", f"pred_{key}"


def _compute_sitewise_normalized_rmse(
    prediction_frame: pd.DataFrame,
    *,
    targets: Sequence[str] = DEFAULT_HEADS,
) -> pd.DataFrame:
    if prediction_frame.empty or "site" not in prediction_frame.columns:
        return pd.DataFrame(
            columns=[
                "site",
                "split",
                "target",
                "rmse",
                "normalized_rmse",
                "target_std",
                "sample_count",
            ]
        )

    frame = prediction_frame.copy()
    if "split" not in frame.columns:
        frame["split"] = "all"
    rows: list[dict[str, Any]] = []
    for target in [str(item) for item in targets]:
        target_col, pred_col = _site_metric_target_columns(target)
        if target_col not in frame.columns or pred_col not in frame.columns:
            continue
        subset = (
            frame[["site", "split", target_col, pred_col]]
            .dropna(subset=[target_col, pred_col])
            .copy()
        )
        if subset.empty:
            continue
        for (site_name, split_name), group_df in subset.groupby(
            ["site", "split"], dropna=False, sort=False
        ):
            target_values = pd.to_numeric(group_df[target_col], errors="coerce")
            pred_values = pd.to_numeric(group_df[pred_col], errors="coerce")
            valid_mask = target_values.notna() & pred_values.notna()
            if not bool(valid_mask.any()):
                continue
            y_true = target_values.loc[valid_mask].to_numpy(dtype=float)
            y_pred = pred_values.loc[valid_mask].to_numpy(dtype=float)
            if target in {"dir", "dp"}:
                errors = np.asarray(circular_error_deg(y_true, y_pred), dtype=float)
            else:
                errors = np.asarray(y_pred - y_true, dtype=float)
            if errors.size == 0:
                continue
            rmse = float(np.sqrt(np.mean(np.square(errors))))
            target_std = float(np.std(y_true, ddof=0))
            rows.append(
                {
                    "site": str(site_name),
                    "split": str(split_name),
                    "target": str(target),
                    "rmse": rmse,
                    "normalized_rmse": rmse / target_std
                    if np.isfinite(target_std) and target_std > 0.0
                    else float("nan"),
                    "target_std": target_std,
                    "sample_count": int(errors.size),
                }
            )
    if not rows:
        return pd.DataFrame(
            columns=[
                "site",
                "split",
                "target",
                "rmse",
                "normalized_rmse",
                "target_std",
                "sample_count",
            ]
        )
    return pd.DataFrame(rows).sort_values(["target", "split", "site"]).reset_index(drop=True)


def _compute_training_analogue_distances_from_static(
    arrays: PointCentricArrays,
    *,
    train_sites: Sequence[str],
    eval_sites: Sequence[str] | None = None,
    k_nearest: int = 5,
    exclude_self_for_training: bool = True,
) -> pd.DataFrame:
    x_static = getattr(arrays, "x_static", {})
    if not isinstance(x_static, Mapping) or not x_static:
        return pd.DataFrame(
            columns=[
                "site",
                "nearest_training_site",
                "nearest_training_distance",
                "mean_training_analogue_distance",
                "k_nearest_analogues",
                "static_feature_count",
            ]
        )

    site_names = [str(site) for site in x_static.keys()]
    site_to_vector = {
        str(site): np.asarray(vector, dtype=np.float32).reshape(-1)
        for site, vector in x_static.items()
    }
    train_site_list = [str(site) for site in train_sites if str(site) in site_to_vector]
    eval_site_list = [
        str(site) for site in (eval_sites or site_names) if str(site) in site_to_vector
    ]
    if not train_site_list or not eval_site_list:
        return pd.DataFrame(
            columns=[
                "site",
                "nearest_training_site",
                "nearest_training_distance",
                "mean_training_analogue_distance",
                "k_nearest_analogues",
                "static_feature_count",
            ]
        )

    train_matrix_raw = np.vstack([site_to_vector[site] for site in train_site_list]).astype(
        float, copy=False
    )
    train_means = np.nanmean(train_matrix_raw, axis=0)
    train_stds = np.nanstd(train_matrix_raw, axis=0, ddof=0)
    usable_mask = np.isfinite(train_means) & np.isfinite(train_stds) & (train_stds > 0.0)
    if not bool(np.any(usable_mask)):
        return pd.DataFrame(
            columns=[
                "site",
                "nearest_training_site",
                "nearest_training_distance",
                "mean_training_analogue_distance",
                "k_nearest_analogues",
                "static_feature_count",
            ]
        )

    train_matrix = (train_matrix_raw[:, usable_mask] - train_means[usable_mask]) / train_stds[
        usable_mask
    ]
    rows: list[dict[str, Any]] = []
    k = max(1, int(k_nearest))
    for site_name in eval_site_list:
        row_vec = site_to_vector[site_name].astype(float, copy=False)
        row_vec = (row_vec[usable_mask] - train_means[usable_mask]) / train_stds[usable_mask]
        if not np.isfinite(row_vec).all():
            continue
        candidate_sites = list(train_site_list)
        candidate_matrix = train_matrix.copy()
        if exclude_self_for_training and site_name in train_site_list and len(candidate_sites) > 1:
            keep_mask = np.asarray(
                [candidate != site_name for candidate in candidate_sites], dtype=bool
            )
            candidate_sites = [
                candidate for candidate, keep in zip(candidate_sites, keep_mask) if keep
            ]
            candidate_matrix = candidate_matrix[keep_mask]
        if candidate_matrix.size == 0:
            continue
        distances = np.sqrt(np.sum(np.square(candidate_matrix - row_vec), axis=1))
        finite_mask = np.isfinite(distances)
        if not bool(np.any(finite_mask)):
            continue
        candidate_matrix = candidate_matrix[finite_mask]
        distances = distances[finite_mask]
        candidate_sites = [
            candidate for candidate, keep in zip(candidate_sites, finite_mask) if keep
        ]
        order = np.argsort(distances, kind="mergesort")
        k_idx = order[: min(k, len(order))]
        nearest_idx = int(order[0])
        rows.append(
            {
                "site": site_name,
                "nearest_training_site": str(candidate_sites[nearest_idx]),
                "nearest_training_distance": float(distances[nearest_idx]),
                "mean_training_analogue_distance": float(np.mean(distances[k_idx])),
                "k_nearest_analogues": int(len(k_idx)),
                "static_feature_count": int(np.sum(usable_mask)),
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "site",
                "nearest_training_site",
                "nearest_training_distance",
                "mean_training_analogue_distance",
                "k_nearest_analogues",
                "static_feature_count",
            ]
        )
    return pd.DataFrame(rows).sort_values(["site"]).reset_index(drop=True)


def compute_sitewise_static_analogue_error_table(
    results_dir: str | Path,
    *,
    targets: Sequence[str] = DEFAULT_HEADS,
    k_nearest: int = 5,
    exclude_self_for_training: bool = True,
    training_config_path: str | Path | None = None,
    point_centric_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Compute per-site normalized RMSE against mean distance to training static analogues."""
    bundle = load_results_bundle(
        results_dir,
        training_config_path=training_config_path,
        point_centric_dir=point_centric_dir,
    )
    prediction_frame = load_prediction_frame(bundle, split="all")
    site_metrics = _compute_sitewise_normalized_rmse(prediction_frame, targets=targets)
    if site_metrics.empty:
        return site_metrics

    arrays, _, _ = load_split_dataset(bundle, split="all")
    train_sites: list[str] = []
    if "train_csv" in bundle.prediction_files or "train_nc" in bundle.prediction_files:
        train_sites = (
            load_prediction_frame(bundle, split="train")
            .get("site", pd.Series(dtype=str))
            .astype(str)
            .dropna()
            .drop_duplicates()
            .tolist()
        )
    elif (
        hasattr(arrays, "split_idx")
        and isinstance(arrays.split_idx, Mapping)
        and "train" in arrays.split_idx
    ):
        train_idx = np.asarray(arrays.split_idx.get("train", []), dtype=int)
        target_sites = np.asarray(getattr(arrays, "target_sites", []), dtype=object)
        if train_idx.size and target_sites.size:
            valid_idx = train_idx[(train_idx >= 0) & (train_idx < target_sites.size)]
            if valid_idx.size:
                train_sites = (
                    pd.Series(target_sites[valid_idx], dtype="object")
                    .astype(str)
                    .dropna()
                    .drop_duplicates()
                    .tolist()
                )
    if not train_sites:
        return site_metrics
    analog_distances = _compute_training_analogue_distances_from_static(
        arrays,
        train_sites=train_sites,
        eval_sites=site_metrics["site"].astype(str).drop_duplicates().tolist(),
        k_nearest=int(k_nearest),
        exclude_self_for_training=exclude_self_for_training,
    )
    if analog_distances.empty:
        return site_metrics
    merged = site_metrics.merge(analog_distances, on="site", how="left")
    return merged.sort_values(
        ["target", "split", "mean_training_analogue_distance", "site"]
    ).reset_index(drop=True)


def _to_device_tensor(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _to_device_tensor(item, device) for key, item in value.items()}
    return value


def _sample_to_device(sample: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: _to_device_tensor(value, device) for key, value in dict(sample).items()}


def _clone_tensor(value: torch.Tensor | None) -> torch.Tensor | None:
    return None if value is None else value.detach().clone()


def _detach_cpu_if_tensor(value: Any) -> Any:
    return value.detach().cpu() if torch.is_tensor(value) else value


def safe_forward(
    model: torch.nn.Module,
    batch: Mapping[str, Any],
    *,
    return_attention: bool = False,
    return_diagnostics: bool = False,
) -> dict[str, torch.Tensor]:
    """Run the model on a dataset sample or batch with optional diagnostics."""
    output = model(
        batch.get("x_dynamic"),
        batch.get("x_static"),
        x_bathy=batch.get("x_bathy"),
        x_dynamic_sources=batch.get("x_dynamic_sources"),
        source_geometry=batch.get("source_geometry"),
        return_attention=return_attention,
        return_diagnostics=return_diagnostics,
    )
    if not isinstance(output, dict):
        raise TypeError("Explainability helpers expect model forward to return a mapping")
    return output


def _recover_physical_targets(target: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    if "physical" in target:
        physical = target["physical"].reshape(-1, 4)
        return {
            "hs": physical[:, 0],
            "tp": physical[:, 1],
            "dir": physical[:, 2],
            "dp": physical[:, 3],
        }
    hs = target["hs"].reshape(-1)
    tp = (
        target.get("tp_value", target["tp_soft"].argmax(dim=-1)).reshape(-1).to(dtype=torch.float32)
    )
    direction = (
        target.get("dir_value", target["dir_soft"].argmax(dim=-1))
        .reshape(-1)
        .to(dtype=torch.float32)
    )
    dp = (
        target.get("dp_value", target["dp_soft"].argmax(dim=-1)).reshape(-1).to(dtype=torch.float32)
    )
    return {"hs": hs, "tp": tp, "dir": direction, "dp": dp}


def recover_physical_predictions(
    output: Mapping[str, torch.Tensor],
    *,
    target: Mapping[str, Any] | None = None,
    transfer_scaler_stats: Mapping[int, tuple[float, float]] | None = None,
    tp_min: float = 0.5,
    tp_max: float = 30.0,
) -> dict[str, torch.Tensor]:
    """Recover physical head predictions for any supported target mode."""
    if "hs" in output and "tp_pred" in output and "dir_pred" in output and "dp_pred" in output:
        return {
            "hs": output["hs"].reshape(-1),
            "tp": output["tp_pred"].reshape(-1),
            "dir": output["dir_pred"].reshape(-1),
            "dp": output["dp_pred"].reshape(-1),
        }

    if "log_hs_ratio" not in output:
        raise ValueError(
            "Output does not contain a known physical or transfer prediction structure"
        )
    if target is None or "reference" not in target:
        raise ValueError("Transfer-mode physical recovery requires target['reference']")

    reference = target["reference"].reshape(-1, 4)
    residual_correction_mode = any(
        key in output
        for key in ("raw_log_hs_ratio", "raw_tp_delta", "raw_dir_delta_deg", "raw_dp_delta_deg")
    )
    log_hs_ratio = output["log_hs_ratio"].reshape(-1)
    tp_delta = output["tp_delta"].reshape(-1)
    if not residual_correction_mode:
        stats = dict(transfer_scaler_stats or {})
        if 0 in stats:
            mean, scale = stats[0]
            log_hs_ratio = (log_hs_ratio * float(scale)) + float(mean)
        if 1 in stats:
            mean, scale = stats[1]
            tp_delta = (tp_delta * float(scale)) + float(mean)
    pred_hs = reference[:, 0] * torch.exp(log_hs_ratio)
    pred_tp = reference[:, 1] + tp_delta
    if not residual_correction_mode:
        pred_tp = torch.clamp(pred_tp, min=float(tp_min), max=float(tp_max))
    pred_dir = (reference[:, 2] + output["dir_delta_deg"].reshape(-1)) % 360.0
    pred_dp = (reference[:, 3] + output["dp_delta_deg"].reshape(-1)) % 360.0
    return {"hs": pred_hs, "tp": pred_tp, "dir": pred_dir, "dp": pred_dp}


def _resolve_head_log_probs(output: Mapping[str, torch.Tensor], head: str) -> torch.Tensor | None:
    mapping = {
        "tp": "tp_log_probs",
        "dir": "dir_log_probs",
        "dp": "dp_log_probs",
    }
    return output.get(mapping.get(head, ""))


def resolve_quantity_tensor(
    output: Mapping[str, torch.Tensor],
    target: Mapping[str, Any],
    *,
    head: str,
    quantity: str,
    transfer_scaler_stats: Mapping[int, tuple[float, float]] | None = None,
    selected_class_idx: int | None = None,
    tp_min: float = 0.5,
    tp_max: float = 30.0,
) -> torch.Tensor:
    """Resolve one explainability scalar for a head and quantity choice."""
    quantity_key = str(quantity).strip().lower()
    physical_pred = recover_physical_predictions(
        output,
        target=target,
        transfer_scaler_stats=transfer_scaler_stats,
        tp_min=tp_min,
        tp_max=tp_max,
    )
    if quantity_key == "physical_prediction":
        return physical_pred[head]

    physical_true = _recover_physical_targets(target)
    if quantity_key == "absolute_error":
        if head in {"dir", "dp"}:
            return torch.abs(circular_error_deg(physical_true[head], physical_pred[head]))
        return torch.abs(physical_pred[head] - physical_true[head])
    if quantity_key == "signed_error":
        if head in {"dir", "dp"}:
            return circular_error_deg(physical_true[head], physical_pred[head])
        return physical_pred[head] - physical_true[head]
    if quantity_key == "entropy":
        log_probs = _resolve_head_log_probs(output, head)
        if log_probs is None:
            raise ValueError(f"Entropy is not defined for head '{head}'")
        probs = torch.exp(log_probs)
        return -torch.sum(probs * log_probs, dim=-1)
    if quantity_key == "selected_logit":
        log_probs = _resolve_head_log_probs(output, head)
        if log_probs is None:
            raise ValueError(f"Selected logit is not defined for head '{head}'")
        if selected_class_idx is None:
            selected = torch.argmax(log_probs, dim=-1)
        else:
            selected = torch.full(
                (log_probs.size(0),),
                int(selected_class_idx),
                device=log_probs.device,
                dtype=torch.long,
            )
        return log_probs.gather(1, selected.unsqueeze(-1)).squeeze(-1)
    raise ValueError(f"Unsupported quantity: {quantity}")


def _bin_series(values: pd.Series, n_bins: int = 5, *, prefix: str = "bin") -> pd.Series:
    clean = pd.to_numeric(values, errors="coerce")
    if clean.notna().sum() < 2:
        return pd.Series([f"{prefix}_0"] * len(values), index=values.index)
    try:
        bins = pd.qcut(
            clean.rank(method="first"), q=min(n_bins, clean.notna().sum()), duplicates="drop"
        )
        return bins.astype(str).fillna(f"{prefix}_0")
    except Exception:
        return pd.Series([f"{prefix}_0"] * len(values), index=values.index)


def sample_explainability_subset(
    frame: pd.DataFrame,
    *,
    max_samples: int = 512,
    random_seed: int = 42,
    use_stratified_sampling: bool = True,
    strata: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Sample a reproducible subset, optionally stratified over derived bins."""
    if len(frame) <= max_samples:
        return frame.copy()

    sampled = frame.copy()
    if "target_hs_bin" not in sampled.columns and "target_hs" in sampled.columns:
        sampled["target_hs_bin"] = _bin_series(sampled["target_hs"], prefix="target_hs")
    if "target_tp_bin" not in sampled.columns and "target_tp" in sampled.columns:
        sampled["target_tp_bin"] = _bin_series(sampled["target_tp"], prefix="target_tp")

    strata_cols = [col for col in (strata or []) if col in sampled.columns]
    if (not use_stratified_sampling) or (not strata_cols):
        return sampled.sample(n=max_samples, random_state=random_seed).sort_index()

    grouped = sampled.groupby(strata_cols, dropna=False, sort=False)
    group_keys = list(grouped.groups.keys())
    if not group_keys:
        return sampled.sample(n=max_samples, random_state=random_seed).sort_index()

    base_per_group = max(1, max_samples // len(group_keys))
    pieces = []
    for _, group_df in grouped:
        take = min(len(group_df), base_per_group)
        pieces.append(group_df.sample(n=take, random_state=random_seed))
    out = pd.concat(pieces).drop_duplicates()
    remaining = max_samples - len(out)
    if remaining > 0:
        leftovers = sampled.drop(index=out.index, errors="ignore")
        if len(leftovers) > 0:
            out = pd.concat(
                [out, leftovers.sample(n=min(remaining, len(leftovers)), random_state=random_seed)]
            )
    return out.iloc[:max_samples].sort_index()


def _iter_sample_index_batches(
    indices: Sequence[int], batch_size: int | None
) -> Iterable[list[int]]:
    if not indices:
        return
    if batch_size is None:
        yield [int(idx) for idx in indices]
        return
    size = max(1, int(batch_size))
    for start in range(0, len(indices), size):
        yield [int(idx) for idx in indices[start : start + size]]


def _match_feature_group(
    feature_name: str,
    patterns: Mapping[str, Sequence[str]],
) -> str | None:
    lower = str(feature_name).strip().lower()
    for group_name, tokens in patterns.items():
        for token in tokens:
            token_lower = str(token).strip().lower()
            if lower == token_lower or token_lower in lower:
                return str(group_name)
    return None


def build_feature_groups(
    feature_names: Sequence[str],
    *,
    overrides: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, Any]:
    """Group static feature names by editable pattern rules."""
    patterns = dict(DEFAULT_STATIC_GROUP_PATTERNS)
    if overrides:
        patterns.update({str(key): tuple(value) for key, value in overrides.items()})

    groups: dict[str, list[str]] = {group: [] for group in patterns}
    unmatched: list[str] = []
    feature_to_group: dict[str, str] = {}
    for name in [str(item) for item in feature_names]:
        group = _match_feature_group(name, patterns)
        if group is None:
            unmatched.append(name)
            groups.setdefault("misc_static", []).append(name)
            feature_to_group[name] = "misc_static"
            continue
        groups.setdefault(group, []).append(name)
        feature_to_group[name] = group
    groups = {name: values for name, values in groups.items() if values}
    return {"groups": groups, "unmatched": unmatched, "feature_to_group": feature_to_group}


def build_dynamic_feature_groups(
    feature_names: Sequence[str],
    *,
    overrides: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, Any]:
    """Group dynamic/source feature names by editable pattern rules."""
    patterns = dict(DEFAULT_DYNAMIC_GROUP_PATTERNS)
    if overrides:
        patterns.update({str(key): tuple(value) for key, value in overrides.items()})

    groups: dict[str, list[str]] = {group: [] for group in patterns}
    unmatched: list[str] = []
    feature_to_group: dict[str, str] = {}
    for name in [str(item) for item in feature_names]:
        group = _match_feature_group(name, patterns)
        if group is None:
            unmatched.append(name)
            groups.setdefault("misc_dynamic", []).append(name)
            feature_to_group[name] = "misc_dynamic"
            continue
        groups.setdefault(group, []).append(name)
        feature_to_group[name] = group
    groups = {name: values for name, values in groups.items() if values}
    return {"groups": groups, "unmatched": unmatched, "feature_to_group": feature_to_group}


def build_expert_regime_feature_groups(
    feature_names: Sequence[str],
    *,
    overrides: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, Any]:
    """Group dynamic source features into swell / wind-sea / local-wind regime families."""
    patterns = dict(DEFAULT_EXPERT_REGIME_GROUP_PATTERNS)
    if overrides:
        patterns.update({str(key): tuple(value) for key, value in overrides.items()})

    groups: dict[str, list[str]] = {group: [] for group in patterns}
    unmatched: list[str] = []
    feature_to_group: dict[str, str] = {}
    for name in [str(item) for item in feature_names]:
        group = _match_feature_group(name, patterns)
        if group is None:
            unmatched.append(name)
            groups.setdefault("other_dynamic", []).append(name)
            feature_to_group[name] = "other_dynamic"
            continue
        groups.setdefault(group, []).append(name)
        feature_to_group[name] = group
    groups = {name: values for name, values in groups.items() if values}
    return {"groups": groups, "unmatched": unmatched, "feature_to_group": feature_to_group}


def _ensure_cache_dir(bundle: ResultsBundle) -> Path:
    bundle.cache_dir.mkdir(parents=True, exist_ok=True)
    return bundle.cache_dir


def _cache_file(bundle: ResultsBundle, name: str, suffix: str = ".pkl") -> Path:
    return _ensure_cache_dir(bundle) / f"{name}{suffix}"


def _write_cache(bundle: ResultsBundle, name: str, payload: Any) -> Path:
    path = _cache_file(bundle, name)
    with path.open("wb") as fh:
        pickle.dump(payload, fh)
    return path


def _read_cache(bundle: ResultsBundle, name: str) -> Any | None:
    path = _cache_file(bundle, name)
    if not path.exists():
        return None
    with path.open("rb") as fh:
        return pickle.load(fh)


def _collate_values(values: Sequence[Any]) -> Any:
    first = values[0]
    if torch.is_tensor(first):
        return torch.stack(list(values))
    if isinstance(first, Mapping):
        keys = {key for value in values for key in value.keys()}
        return {
            key: _collate_values([value[key] for value in values if key in value]) for key in keys
        }
    return list(values)


def _batchify_samples(runner: ExplainabilityModelBundle, indices: Sequence[int]) -> dict[str, Any]:
    samples = [_sample_to_device(runner.dataset[int(idx)], runner.device) for idx in indices]
    keys = {key for sample in samples for key in sample.keys()}
    batch: dict[str, Any] = {}
    for key in keys:
        values = [sample[key] for sample in samples if key in sample]
        if not values:
            continue
        batch[key] = _collate_values(values)
    return batch


def _tensor_mean_baseline(values: torch.Tensor) -> torch.Tensor:
    dims = list(range(values.ndim))
    keep_shape = [1] * values.ndim
    return values.mean(dim=0, keepdim=True).expand_as(values).reshape(values.shape)


def _mean_dynamic_context(
    runner: ExplainabilityModelBundle, background_indices: Sequence[int]
) -> dict[str, torch.Tensor | None]:
    batch = _batchify_samples(runner, background_indices)
    context = {}
    for key in ("x_dynamic", "x_dynamic_sources", "source_geometry", "x_bathy"):
        value = batch.get(key)
        if torch.is_tensor(value):
            context[key] = value.mean(dim=0, keepdim=True)
        else:
            context[key] = None
    return context


def _evaluate_static_context_average(
    runner: ExplainabilityModelBundle,
    x_static_matrix: np.ndarray,
    *,
    head: str,
    quantity: str,
    background_indices: Sequence[int],
) -> np.ndarray:
    context = _mean_dynamic_context(runner, background_indices)
    reference_batch = _batchify_samples(runner, background_indices)
    target_batch = reference_batch.get("y")
    if not isinstance(target_batch, Mapping):
        raise ValueError("Context-averaged static explainability requires batched target mappings")
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for row in np.asarray(x_static_matrix, dtype=np.float32):
            batch_size = int(len(background_indices))
            static_tensor = (
                torch.as_tensor(row, device=runner.device).reshape(1, -1).repeat(batch_size, 1)
            )
            batch: dict[str, Any] = {"x_static": static_tensor}
            for key, value in context.items():
                if value is None:
                    batch[key] = None
                    continue
                batch[key] = value.repeat(batch_size, *([1] * (value.ndim - 1)))
            output = safe_forward(runner.model, batch)
            quantity_tensor = resolve_quantity_tensor(
                output,
                target_batch,
                head=head,
                quantity=quantity,
                transfer_scaler_stats=runner.transfer_scaler_stats,
                tp_min=float(
                    resolve_targets_config(runner.bundle.config.get("data", {}) or {}).get(
                        "tp_min", 0.5
                    )
                ),
                tp_max=float(
                    resolve_targets_config(runner.bundle.config.get("data", {}) or {}).get(
                        "tp_max", 30.0
                    )
                ),
            )
            outputs.append(quantity_tensor.mean().reshape(1).detach().cpu().numpy())
    return np.asarray(outputs, dtype=np.float32).reshape(-1)


def compute_context_averaged_static_shap(
    runner: ExplainabilityModelBundle,
    *,
    head: str,
    quantity: str,
    sample_indices: Sequence[int],
    background_indices: Sequence[int],
    cache_key: str | None = None,
) -> dict[str, Any]:
    """Compute context-averaged static SHAP or a documented fallback approximation."""
    cache_name = cache_key or f"static_shap_{runner.split}_{head}_{quantity}"
    cached = _read_cache(runner.bundle, cache_name)
    if cached is not None:
        return cached

    sample_batch = _batchify_samples(runner, sample_indices)
    background_batch = _batchify_samples(runner, background_indices)
    if "x_static" not in sample_batch:
        result = {
            "method": "unavailable",
            "reason": "static branch disabled",
            "values": pd.DataFrame(),
        }
        _write_cache(runner.bundle, cache_name, result)
        return result

    x_static = sample_batch["x_static"].detach().cpu().numpy()
    background_static = background_batch["x_static"].detach().cpu().numpy()
    feature_names = list(
        getattr(runner.arrays, "static_feature_names", [])
        or [f"static_{idx}" for idx in range(x_static.shape[1])]
    )

    if shap is not None:
        try:

            def predictor(x_array: np.ndarray) -> np.ndarray:
                return _evaluate_static_context_average(
                    runner,
                    x_array,
                    head=head,
                    quantity=quantity,
                    background_indices=background_indices,
                )

            explainer = shap.KernelExplainer(
                predictor, background_static[: min(len(background_static), 32)]
            )
            shap_values = np.asarray(
                explainer.shap_values(x_static, nsamples=min(128, x_static.shape[0] * 2)),
                dtype=np.float32,
            )
            expected_value = getattr(explainer, "expected_value", None)
            if expected_value is None:
                base_values = None
            else:
                expected_array = np.asarray(expected_value, dtype=np.float32).reshape(-1)
                if expected_array.size == 1:
                    base_values = np.repeat(expected_array, repeats=x_static.shape[0]).astype(
                        np.float32, copy=False
                    )
                else:
                    base_values = expected_array.astype(np.float32, copy=False)
            method = "kernel_shap"
            note = None
        except Exception as exc:
            shap_values = None
            base_values = None
            method = "kernel_shap_failed"
            note = f"Direct SHAP path failed and fell back to baseline surrogate: {exc}"
    else:
        shap_values = None
        base_values = None
        method = "baseline_surrogate"
        note = "shap is unavailable; used baseline surrogate"

    if shap_values is None:
        baseline = background_static.mean(axis=0, keepdims=True)
        baseline_preds = _evaluate_static_context_average(
            runner,
            np.repeat(baseline, repeats=x_static.shape[0], axis=0),
            head=head,
            quantity=quantity,
            background_indices=background_indices,
        )
        point_preds = _evaluate_static_context_average(
            runner,
            x_static,
            head=head,
            quantity=quantity,
            background_indices=background_indices,
        )
        delta = (point_preds - baseline_preds).reshape(-1, 1)
        denom = np.where(np.abs(x_static - baseline) < 1e-6, 1.0, np.abs(x_static - baseline))
        weights = np.abs(x_static - baseline) / np.clip(
            np.sum(np.abs(x_static - baseline), axis=1, keepdims=True), 1e-6, None
        )
        shap_values = delta * weights / np.where(np.isfinite(denom), 1.0, 1.0)
        if method != "kernel_shap_failed":
            method = "baseline_surrogate"

    values_df = pd.DataFrame(shap_values, columns=feature_names)
    payload = {
        "method": method,
        "values": values_df,
        "sample_indices": list(sample_indices),
        "base_values": base_values,
        "note": note,
    }
    _write_cache(runner.bundle, cache_name, payload)
    return payload


def compute_static_grouped_shap(
    shap_payload: Mapping[str, Any],
    feature_groups: Mapping[str, Any],
) -> dict[str, pd.DataFrame]:
    """Aggregate per-feature SHAP values to editable feature groups."""
    values_df: pd.DataFrame = shap_payload["values"]
    groups = dict(feature_groups.get("groups", {}))
    rows = []
    beeswarm_rows = []
    for group_name, columns in groups.items():
        present = [column for column in columns if column in values_df.columns]
        if not present:
            continue
        grouped = values_df[present].sum(axis=1)
        rows.append(
            {
                "group": group_name,
                "mean_abs_value": float(np.abs(grouped).mean()),
                "mean_signed_value": float(grouped.mean()),
                "n_features": int(len(present)),
            }
        )
        beeswarm_rows.extend(
            [{"group": group_name, "value": float(item)} for item in grouped.to_numpy(dtype=float)]
        )
    return {
        "summary": pd.DataFrame(rows)
        .sort_values("mean_abs_value", ascending=False)
        .reset_index(drop=True),
        "beeswarm": pd.DataFrame(beeswarm_rows),
    }


def _target_display_name(head: str) -> str:
    return TARGET_DISPLAY_NAMES.get(str(head), str(head).upper())


def _group_display_name(group_name: str) -> str:
    return STATIC_GROUP_DISPLAY_NAMES.get(
        str(group_name), str(group_name).replace("_", " ").title()
    )


def _infer_feature_unit(feature_name: str) -> str | None:
    if feature_name.endswith("_m"):
        return "m"
    if feature_name.endswith("_deg"):
        return "deg"
    if feature_name.endswith("_km"):
        return "km"
    return None


def _feature_display_name(feature_name: str, *, include_units: bool = True) -> str:
    base = STATIC_FEATURE_DISPLAY_NAMES.get(str(feature_name))
    if base is None:
        clean = str(feature_name).replace("_", " ").strip()
        clean = clean.replace(" hs ", " Hs ").replace(" tp ", " Tp ")
        base = clean[:1].upper() + clean[1:]
    unit = _infer_feature_unit(str(feature_name))
    if include_units and unit is not None and f"({unit})" not in base:
        return f"{base} ({unit})"
    return base


def _resolve_static_output_dirs(bundle: ResultsBundle) -> dict[str, Path]:
    root = bundle.results_dir / "static_explainability"
    paths = {
        "root": root,
        "shap": root / "shap",
        "ale": root / "ale",
        "tables": root / "tables",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def _fit_monotonic_feature_mapping(
    transformed_values: np.ndarray,
    raw_values: np.ndarray,
) -> Callable[[np.ndarray], np.ndarray] | None:
    valid_mask = np.isfinite(transformed_values) & np.isfinite(raw_values)
    if int(valid_mask.sum()) < 3:
        return None
    transformed = np.asarray(transformed_values[valid_mask], dtype=float)
    raw = np.asarray(raw_values[valid_mask], dtype=float)
    order = np.argsort(transformed, kind="mergesort")
    transformed = transformed[order]
    raw = raw[order]

    unique_x, inverse = np.unique(transformed, return_inverse=True)
    if unique_x.size < 2:
        return None
    mean_y = np.zeros(unique_x.size, dtype=float)
    counts = np.zeros(unique_x.size, dtype=float)
    for idx, value in enumerate(raw):
        mean_y[inverse[idx]] += value
        counts[inverse[idx]] += 1.0
    mean_y = mean_y / np.clip(counts, 1.0, None)

    diffs = np.diff(mean_y)
    monotonic_increasing = np.all(diffs >= -1e-9)
    monotonic_decreasing = np.all(diffs <= 1e-9)
    if not monotonic_increasing and not monotonic_decreasing:
        return None
    if monotonic_decreasing:
        unique_x = unique_x[::-1]
        mean_y = mean_y[::-1]

    def mapper(values: np.ndarray) -> np.ndarray:
        arr = np.asarray(values, dtype=float)
        return np.interp(arr, unique_x, mean_y)

    return mapper


def _resolve_feature_axis_metadata(
    sampled_frame: pd.DataFrame | None,
    sample_indices: Sequence[int],
    *,
    feature_name: str,
    transformed_values: np.ndarray,
    arrays: PointCentricArrays,
) -> dict[str, Any]:
    is_standardized = _is_static_tensor_standardized(arrays)
    metadata = {
        "feature_label": _feature_display_name(feature_name, include_units=not is_standardized),
        "physical_units_used": False,
        "normalization_status": "normalized_model_input"
        if is_standardized
        else "model_input_units",
        "map_to_physical": None,
        "raw_values": None,
    }
    if sampled_frame is None or feature_name not in sampled_frame.columns:
        return metadata

    raw_series = sampled_frame.loc[list(sample_indices), feature_name]
    raw_values = pd.to_numeric(raw_series, errors="coerce").to_numpy(dtype=float)
    mapper = _fit_monotonic_feature_mapping(np.asarray(transformed_values, dtype=float), raw_values)
    if mapper is None:
        return metadata

    metadata["feature_label"] = _feature_display_name(feature_name, include_units=True)
    metadata["physical_units_used"] = True
    metadata["normalization_status"] = "inverse_mapped_to_physical_units"
    metadata["map_to_physical"] = mapper
    metadata["raw_values"] = raw_values
    return metadata


def _static_shap_target_display_name(head: str) -> str:
    return STATIC_SHAP_TARGET_DISPLAY_NAMES.get(str(head), _target_display_name(str(head)))


def _select_target_payload(candidate: Any, *, head: str) -> Any:
    if isinstance(candidate, Mapping):
        if str(head) in candidate:
            return candidate[str(head)]
        if head in candidate:
            return candidate[head]
    if isinstance(candidate, (list, tuple)) and candidate:
        head_idx = list(DEFAULT_HEADS).index(str(head)) if str(head) in DEFAULT_HEADS else None
        if head_idx is not None and len(candidate) == len(DEFAULT_HEADS):
            return candidate[head_idx]
        if len(candidate) == 1:
            return candidate[0]
    return candidate


def _coerce_feature_name_order(
    names: Sequence[str] | None,
    *,
    expected_features: int | None,
) -> list[str]:
    feature_names = [str(name) for name in (names or [])]
    if feature_names:
        return feature_names
    if expected_features is None:
        raise ValueError("Feature names are required when the feature dimension cannot be inferred")
    return [f"static_{idx}" for idx in range(int(expected_features))]


def _extract_2d_static_matrix_from_array(
    values: np.ndarray,
    *,
    head: str,
    expected_samples: int | None,
    expected_features: int,
) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        if array.size != expected_features:
            raise ValueError(
                f"1D SHAP array length {array.size} does not match expected features {expected_features}"
            )
        return array.reshape(1, -1)
    if array.ndim == 2:
        if array.shape[1] == expected_features:
            return array
        if array.shape[0] == expected_features:
            return array.T
        raise ValueError(
            f"2D SHAP array shape {array.shape} does not match expected feature dimension {expected_features}"
        )
    if array.ndim != 3:
        raise ValueError(f"Unsupported SHAP array shape {array.shape}; expected 2D or 3D")

    feature_axes = [
        idx for idx, size in enumerate(array.shape) if int(size) == int(expected_features)
    ]
    if len(feature_axes) != 1:
        raise ValueError(
            f"Could not determine feature axis for SHAP array shape {array.shape} and expected feature count {expected_features}"
        )
    feature_axis = feature_axes[0]

    target_axes = [
        idx
        for idx, size in enumerate(array.shape)
        if idx != feature_axis and int(size) == len(DEFAULT_HEADS)
    ]
    if len(target_axes) != 1:
        raise ValueError(
            f"Could not determine target axis for SHAP array shape {array.shape}; "
            f"expected one non-feature axis of size {len(DEFAULT_HEADS)}"
        )
    target_axis = target_axes[0]
    if str(head) not in DEFAULT_HEADS:
        raise ValueError(f"Unknown target head '{head}'")
    array = np.take(array, indices=list(DEFAULT_HEADS).index(str(head)), axis=target_axis)

    remaining_axes = [idx for idx in range(3) if idx != target_axis]
    feature_axis_after_take = remaining_axes.index(feature_axis)
    sample_axes = [idx for idx in range(array.ndim) if idx != feature_axis_after_take]
    if expected_samples is not None:
        matching_sample_axes = [
            idx for idx in sample_axes if int(array.shape[idx]) == int(expected_samples)
        ]
        if len(matching_sample_axes) == 1:
            sample_axis = matching_sample_axes[0]
        else:
            sample_axis = sample_axes[0]
    else:
        sample_axis = sample_axes[0]

    matrix = np.moveaxis(array, (sample_axis, feature_axis_after_take), (0, 1))
    if matrix.ndim != 2:
        matrix = np.squeeze(matrix)
    if matrix.ndim != 2 or matrix.shape[1] != expected_features:
        raise ValueError(
            f"Failed to extract a 2D SHAP matrix from shape {values.shape}; got {matrix.shape}"
        )
    return np.asarray(matrix, dtype=float)


def _extract_static_base_values(
    shap_payload: Mapping[str, Any],
    *,
    head: str,
    n_samples: int,
) -> np.ndarray | None:
    raw_base = shap_payload.get("base_values")
    if raw_base is None:
        raw_base = shap_payload.get("expected_value")
    if raw_base is None:
        return None

    selected = _select_target_payload(raw_base, head=head)
    base_array = np.asarray(selected, dtype=float)
    if base_array.ndim == 0:
        return np.repeat(float(base_array), repeats=n_samples).astype(float, copy=False)
    if base_array.ndim == 1:
        if base_array.size == 1:
            return np.repeat(float(base_array[0]), repeats=n_samples).astype(float, copy=False)
        if base_array.size == n_samples:
            return base_array.astype(float, copy=False)
        if base_array.size == len(DEFAULT_HEADS) and str(head) in DEFAULT_HEADS:
            return np.repeat(
                float(base_array[list(DEFAULT_HEADS).index(str(head))]), repeats=n_samples
            ).astype(float, copy=False)
    raise ValueError(
        "Could not align cached base values to the SHAP sample dimension; "
        f"got shape {base_array.shape} for {n_samples} samples"
    )


def extract_static_shap_matrix(
    shap_payload: Mapping[str, Any],
    *,
    head: str,
    feature_names: Sequence[str] | None,
    expected_samples: int | None,
) -> dict[str, Any]:
    """Normalize cached/current SHAP payloads to a signed 2D sample-feature matrix."""
    payload_values = _select_target_payload(shap_payload.get("values"), head=head)
    if payload_values is None:
        raise ValueError("SHAP payload does not contain values")

    if isinstance(payload_values, pd.DataFrame):
        values_df = payload_values.copy()
        if feature_names:
            missing = [name for name in feature_names if name not in values_df.columns]
            if missing:
                raise ValueError(
                    f"SHAP DataFrame is missing expected static features: {missing[:8]}"
                )
            values_df = values_df.loc[:, list(feature_names)]
        matrix = values_df.to_numpy(dtype=float)
        resolved_feature_names = [str(name) for name in values_df.columns]
    else:
        resolved_feature_names = _coerce_feature_name_order(feature_names, expected_features=None)
        matrix = _extract_2d_static_matrix_from_array(
            np.asarray(payload_values, dtype=float),
            head=head,
            expected_samples=expected_samples,
            expected_features=len(resolved_feature_names),
        )

    if feature_names:
        resolved_feature_names = [str(name) for name in feature_names]
    if matrix.ndim != 2:
        raise ValueError(f"Resolved SHAP matrix must be 2D, got shape {matrix.shape}")
    if expected_samples is not None and matrix.shape[0] != int(expected_samples):
        raise ValueError(
            f"Resolved SHAP sample dimension {matrix.shape[0]} does not match expected {expected_samples}"
        )
    if matrix.shape[1] != len(resolved_feature_names):
        raise ValueError(
            f"Resolved SHAP feature dimension {matrix.shape[1]} does not match feature name count {len(resolved_feature_names)}"
        )

    base_values = _extract_static_base_values(shap_payload, head=head, n_samples=matrix.shape[0])
    sample_indices = [int(idx) for idx in (shap_payload.get("sample_indices") or [])]
    if sample_indices and len(sample_indices) != matrix.shape[0]:
        raise ValueError(
            f"Cached sample_indices length {len(sample_indices)} does not match SHAP sample dimension {matrix.shape[0]}"
        )
    return {
        "matrix": np.asarray(matrix, dtype=float),
        "feature_names": resolved_feature_names,
        "base_values": base_values,
        "sample_indices": sample_indices,
    }


def _build_transformed_to_raw_static_map(arrays: PointCentricArrays) -> dict[str, str]:
    meta = ((arrays.metadata or {}).get("normalization", {}) or {}).get("static_scaler", {}) or {}
    raw_map = meta.get("raw_to_transformed_feature_map_before_ablation") or meta.get(
        "raw_to_transformed_feature_map", {}
    )
    transformed_to_raw: dict[str, str] = {}
    for raw_name, transformed_names in dict(raw_map or {}).items():
        names = [str(name) for name in (transformed_names or [])]
        if len(names) == 1:
            transformed_to_raw.setdefault(names[0], str(raw_name))
    return transformed_to_raw


def recover_static_feature_values(
    runner: ExplainabilityModelBundle,
    *,
    sampled_frame: pd.DataFrame,
    sample_indices: Sequence[int],
    feature_names: Sequence[str],
) -> dict[str, Any]:
    """Recover aligned static feature values, preferring raw values when unambiguous."""
    ordered_indices = [int(idx) for idx in sample_indices]
    feature_list = [str(name) for name in feature_names]
    batch = _batchify_samples(runner, ordered_indices)
    x_static = batch["x_static"].detach().cpu().numpy().astype(float, copy=False)

    if x_static.ndim != 2:
        raise ValueError(f"x_static must be 2D, got shape {x_static.shape}")
    if x_static.shape[0] != len(ordered_indices):
        raise ValueError(
            f"x_static sample dimension {x_static.shape[0]} does not match sampled index count {len(ordered_indices)}"
        )
    if x_static.shape[1] != len(feature_list):
        raise ValueError(
            f"x_static feature dimension {x_static.shape[1]} does not match feature name count {len(feature_list)}"
        )

    aligned_frame = sampled_frame.loc[ordered_indices]
    if list(aligned_frame.index) != ordered_indices:
        raise ValueError(
            "Sample order mismatch between SHAP sample indices and static feature rows"
        )

    values = np.asarray(x_static, dtype=float).copy()
    value_sources = np.array(["standardized"] * len(feature_list), dtype=object)
    raw_feature_map = _build_transformed_to_raw_static_map(runner.arrays)

    for feature_idx, feature_name in enumerate(feature_list):
        raw_name = raw_feature_map.get(feature_name)
        if raw_name is None or raw_name not in aligned_frame.columns:
            continue
        raw_values = pd.to_numeric(aligned_frame[raw_name], errors="coerce").to_numpy(dtype=float)
        if raw_values.shape[0] != values.shape[0]:
            raise ValueError(
                f"Recovered raw feature '{raw_name}' length {raw_values.shape[0]} does not match sample count {values.shape[0]}"
            )
        values[:, feature_idx] = raw_values
        value_sources[feature_idx] = "raw"

    return {
        "values": values,
        "value_sources": value_sources.tolist(),
    }


def _ale_target_title(head: str) -> str:
    return ALE_TARGET_TITLES.get(str(head), _target_display_name(str(head)))


def _ale_target_y_label(head: str, representation: str | None = None) -> str:
    rep = str(representation or "").strip().lower()
    if head in {"dir", "dp"} and rep in {"predicted_direction_deg", "predicted_peak_direction_deg"}:
        if head == "dir":
            return "ALE in predicted mean direction (degrees)"
        return "ALE in predicted peak direction (degrees)"
    return ALE_TARGET_Y_LABELS.get(str(head), f"ALE in {_target_display_name(str(head))}")


def _resolve_ale_split_sequence(
    bundle: ResultsBundle,
    ale_site_set: str,
) -> tuple[list[str], list[str]]:
    requested = str(ale_site_set).strip().lower()
    warnings: list[str] = []
    if requested not in {"test", "val", "heldout"}:
        raise ValueError("ale_site_set must be one of: test, val, heldout")

    available_splits: list[str] = []
    for split_name in ("val", "test"):
        try:
            _prediction_frame_for_split(bundle.results_dir, split_name)
            available_splits.append(split_name)
        except Exception:
            continue

    if requested == "heldout":
        splits = [name for name in ("val", "test") if name in available_splits]
        if not splits:
            raise FileNotFoundError("No held-out prediction exports were found for ALE analysis")
        return splits, warnings

    if requested in available_splits:
        return [requested], warnings

    fallback = (
        "val" if "val" in available_splits else ("test" if "test" in available_splits else None)
    )
    if fallback is None:
        raise FileNotFoundError(
            "No validation or test prediction exports were found for ALE analysis"
        )
    warnings.append(
        f"ALE site set '{requested}' was unavailable; falling back to '{fallback}' for held-out ALE analysis."
    )
    return [fallback], warnings


def _dataset_alignment_frame(runner: ExplainabilityModelBundle, *, split_name: str) -> pd.DataFrame:
    samples = list(getattr(runner.dataset, "samples", []) or [])
    if not samples:
        return pd.DataFrame(
            columns=["dataset_index", "ale_split", "site", "time_index", "timestamp"]
        )
    timestamps = runner.arrays.timestamps
    rows = [
        {
            "dataset_index": int(idx),
            "ale_split": str(split_name),
            "site": str(site),
            "time_index": int(timestep),
            "timestamp": str(timestamps[int(timestep)]),
        }
        for idx, (site, timestep) in enumerate(samples)
    ]
    return pd.DataFrame(rows)


def build_static_ale_analysis_frame(
    bundle: ResultsBundle,
    *,
    device: str,
    training_config_path: str | Path | None,
    point_centric_dir: str | Path | None,
    ale_site_set: str = "test",
) -> dict[str, Any]:
    """Build an ALE-specific, sample-aligned held-out frame plus runner map."""
    split_sequence, warnings = _resolve_ale_split_sequence(bundle, ale_site_set)
    runner_map: dict[str, ExplainabilityModelBundle] = {}
    frames: list[pd.DataFrame] = []

    for split_name in split_sequence:
        split_runner = load_model_for_explainability(
            bundle.results_dir,
            split=split_name,
            device=device,
            training_config_path=training_config_path,
            point_centric_dir=point_centric_dir,
        )
        runner_map[str(split_name)] = split_runner
        dataset_meta = _dataset_alignment_frame(split_runner, split_name=str(split_name))
        prediction_frame = _attach_static_features_to_predictions(
            bundle, load_prediction_frame(bundle, split=split_name)
        ).reset_index(drop=True)
        if len(dataset_meta) != len(prediction_frame):
            raise ValueError(
                f"ALE alignment failed for split '{split_name}': dataset rows={len(dataset_meta)} vs prediction rows={len(prediction_frame)}"
            )
        if not (
            dataset_meta["site"].astype(str).to_numpy().tolist()
            == prediction_frame["site"].astype(str).to_numpy().tolist()
            and dataset_meta["time_index"].astype(int).to_numpy().tolist()
            == prediction_frame["time_index"].astype(int).to_numpy().tolist()
            and dataset_meta["timestamp"].astype(str).to_numpy().tolist()
            == prediction_frame["timestamp"].astype(str).to_numpy().tolist()
        ):
            raise ValueError(
                f"ALE alignment failed for split '{split_name}': dataset metadata and prediction exports are not row-aligned"
            )
        merged = pd.concat(
            [
                dataset_meta.reset_index(drop=True),
                prediction_frame.drop(
                    columns=[
                        col
                        for col in ("split", "site", "time_index", "timestamp")
                        if col in prediction_frame.columns
                    ]
                ).reset_index(drop=True),
            ],
            axis=1,
        )
        frames.append(merged)

    analysis_frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return {
        "frame": analysis_frame,
        "runners": runner_map,
        "splits": split_sequence,
        "warnings": warnings,
    }


def sample_site_balanced_ale_subset(
    analysis_frame: pd.DataFrame,
    *,
    samples_per_site: int = 32,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Sample an equal number of temporal rows per site for ALE."""
    if analysis_frame.empty:
        return analysis_frame.copy()
    grouped = analysis_frame.groupby("site", sort=True, dropna=False)
    site_sizes = grouped.size()
    if site_sizes.empty:
        return analysis_frame.head(0).copy()
    effective_count = int(min(max(1, samples_per_site), int(site_sizes.min())))
    rng = np.random.default_rng(int(random_seed))
    sampled_parts: list[pd.DataFrame] = []
    for site_name, site_df in grouped:
        positions = np.sort(
            rng.choice(site_df.index.to_numpy(dtype=int), size=effective_count, replace=False)
        )
        sampled_parts.append(analysis_frame.loc[positions].copy())
    sampled = (
        pd.concat(sampled_parts, ignore_index=False)
        .sort_values(["site", "time_index", "timestamp"])
        .reset_index(drop=True)
    )
    sampled["site_weight"] = 1.0 / float(sampled["site"].nunique())
    sampled["within_site_weight"] = 1.0 / float(effective_count)
    sampled["effective_samples_per_site"] = int(effective_count)
    return sampled


def _batchify_ale_frame_rows(
    runner_map: Mapping[str, ExplainabilityModelBundle],
    rows: pd.DataFrame,
) -> dict[str, Any]:
    samples: list[dict[str, Any]] = []
    for item in rows.itertuples(index=False):
        split_name = str(getattr(item, "ale_split"))
        dataset_index = int(getattr(item, "dataset_index"))
        samples.append(
            _sample_to_device(
                runner_map[split_name].dataset[dataset_index], runner_map[split_name].device
            )
        )
    keys = {key for sample in samples for key in sample.keys()}
    batch: dict[str, Any] = {}
    for key in keys:
        values = [sample[key] for sample in samples if key in sample]
        if values:
            batch[key] = _collate_values(values)
    return batch


def _resolve_plot_metadata(
    feature_name: str,
    *,
    raw_feature_name: str | None,
    normalization_status: str,
) -> dict[str, Any]:
    metadata_key = str(raw_feature_name or feature_name)
    meta = dict(STATIC_FEATURE_PLOT_METADATA.get(metadata_key, {}))
    label = str(meta.get("label") or _feature_display_name(metadata_key, include_units=False))
    if str(normalization_status).strip().lower() == "standardized_model_input":
        return {
            "display_name": label,
            "display_unit": "standardized model input",
            "feature_label": f"{label} (standardized model input)",
            "feature_unit": "standardized",
            "display_scale": 1.0,
        }

    base_unit = str(
        meta.get("display_unit") or meta.get("unit") or _infer_feature_unit(metadata_key) or ""
    ).strip()
    scale = float(meta.get("scale", 1.0))
    if not base_unit:
        base_unit = "model input units"
    return {
        "display_name": label,
        "display_unit": base_unit,
        "feature_label": f"{label} ({base_unit})",
        "feature_unit": base_unit,
        "display_scale": scale,
    }


def recover_static_feature_plot_values(
    analysis_frame: pd.DataFrame,
    *,
    feature_name: str,
    transformed_values: np.ndarray,
    arrays: PointCentricArrays,
) -> dict[str, Any]:
    """Recover site-level plotting values distinct from model-input transformed values."""
    frame = analysis_frame.reset_index(drop=True)
    transformed = np.asarray(transformed_values, dtype=float).reshape(-1)
    if transformed.shape[0] != len(frame):
        raise ValueError(
            f"Transformed feature values length {transformed.shape[0]} does not match analysis-frame rows {len(frame)}"
        )

    raw_feature_name = _build_transformed_to_raw_static_map(arrays).get(str(feature_name))
    normalization_status = (
        "standardized_model_input"
        if _is_static_tensor_standardized(arrays)
        else "model_input_units"
    )
    plot_values = transformed.copy()

    if raw_feature_name is not None and raw_feature_name in frame.columns:
        raw_values = pd.to_numeric(frame[raw_feature_name], errors="coerce").to_numpy(dtype=float)
        if np.isfinite(raw_values).sum() >= 3:
            plot_values = raw_values
            normalization_status = "inverse_mapped_to_physical_units"

    plot_meta = _resolve_plot_metadata(
        str(feature_name),
        raw_feature_name=raw_feature_name,
        normalization_status=normalization_status,
    )
    plot_values = np.asarray(plot_values, dtype=float) * float(plot_meta["display_scale"])

    site_frame = (
        pd.DataFrame(
            {
                "site": frame["site"].astype(str).to_numpy(),
                "plot_value": plot_values,
                "transformed_value": transformed,
            }
        )
        .groupby("site", as_index=False)
        .agg(
            plot_value=("plot_value", "first"),
            transformed_value=("transformed_value", "first"),
        )
    )
    if not np.all(np.isfinite(site_frame["plot_value"].to_numpy(dtype=float))):
        site_frame = site_frame.loc[
            np.isfinite(site_frame["plot_value"].to_numpy(dtype=float))
        ].reset_index(drop=True)

    plot_to_transformed = _fit_monotonic_feature_mapping(
        site_frame["plot_value"].to_numpy(dtype=float),
        site_frame["transformed_value"].to_numpy(dtype=float),
    )
    if plot_to_transformed is None:
        if normalization_status == "inverse_mapped_to_physical_units":
            plot_values = transformed.copy()
            normalization_status = "standardized_model_input"
            plot_meta = _resolve_plot_metadata(
                str(feature_name),
                raw_feature_name=None,
                normalization_status=normalization_status,
            )
            site_frame = (
                pd.DataFrame(
                    {
                        "site": frame["site"].astype(str).to_numpy(),
                        "plot_value": plot_values,
                        "transformed_value": transformed,
                    }
                )
                .groupby("site", as_index=False)
                .agg(
                    plot_value=("plot_value", "first"),
                    transformed_value=("transformed_value", "first"),
                )
            )
            plot_to_transformed = lambda values: np.asarray(values, dtype=float)
        else:
            plot_to_transformed = lambda values: np.asarray(values, dtype=float)

    return {
        "plot_values": plot_values,
        "site_frame": site_frame,
        "plot_to_transformed": plot_to_transformed,
        "raw_feature_name": raw_feature_name,
        "normalization_status": normalization_status,
        **plot_meta,
    }


def _assign_interval_ids(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    bin_ids = np.full(values.shape[0], -1, dtype=int)
    for idx, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
        if idx == len(edges) - 2:
            mask = (values >= float(left)) & (values <= float(right))
        else:
            mask = (values >= float(left)) & (values < float(right))
        bin_ids[mask] = int(idx)
    return bin_ids


def _merge_sparse_site_bins(
    edges: np.ndarray,
    values: np.ndarray,
    *,
    min_unique_sites_per_bin: int,
) -> np.ndarray:
    working = np.asarray(edges, dtype=float)
    while working.size >= 4:
        bin_ids = _assign_interval_ids(values, working)
        counts = np.array([(bin_ids == idx).sum() for idx in range(working.size - 1)], dtype=int)
        bad = np.where(counts < int(min_unique_sites_per_bin))[0]
        if bad.size == 0:
            return working
        idx = int(bad[0])
        if idx == 0:
            working = np.delete(working, 1)
        elif idx == counts.size - 1:
            working = np.delete(working, -2)
        else:
            merge_right = counts[idx + 1] <= counts[idx - 1]
            working = np.delete(working, idx + 1 if merge_right else idx)
    return working


def build_adaptive_site_bins(
    site_frame: pd.DataFrame,
    *,
    feature_name: str,
    target_bins: int = 6,
    min_unique_sites_per_bin: int = 5,
) -> dict[str, Any]:
    """Build adaptive site-level bins for ALE or classify a feature as discrete."""
    if site_frame.empty:
        return {"analysis_kind": "unsupported", "reason": "no_valid_site_values"}

    values = site_frame["plot_value"].to_numpy(dtype=float)
    unique_values = np.unique(values[np.isfinite(values)])
    if unique_values.size < 2:
        return {"analysis_kind": "unsupported", "reason": "single_unique_site_value"}
    if unique_values.size < 3:
        return {
            "analysis_kind": "discrete",
            "reason": "low_cardinality_feature",
            "levels": unique_values,
        }

    quantiles = np.unique(
        np.quantile(
            unique_values,
            np.linspace(0.0, 1.0, num=max(3, int(target_bins) + 1)),
        )
    )
    if quantiles.size < 4:
        return {
            "analysis_kind": "discrete",
            "reason": "low_cardinality_feature",
            "levels": unique_values,
        }

    edges = _merge_sparse_site_bins(
        quantiles,
        values,
        min_unique_sites_per_bin=int(min_unique_sites_per_bin),
    )
    if edges.size < 4:
        return {
            "analysis_kind": "discrete",
            "reason": "low_cardinality_feature",
            "levels": unique_values,
        }
    if not np.all(np.diff(edges) > 0):
        return {"analysis_kind": "unsupported", "reason": "non_increasing_bin_edges"}

    bin_ids = _assign_interval_ids(values, edges)
    counts = np.array([(bin_ids == idx).sum() for idx in range(edges.size - 1)], dtype=int)
    if np.any(counts < int(min_unique_sites_per_bin)):
        return {"analysis_kind": "unsupported", "reason": "insufficient_bin_support_after_merge"}
    return {
        "analysis_kind": "continuous",
        "reason": "continuous_bins",
        "edges": edges,
        "bin_ids": bin_ids,
        "site_counts": counts,
    }


def resolve_ale_response_tensor(
    output: Mapping[str, torch.Tensor],
    target: Mapping[str, Any],
    *,
    head: str,
    transfer_scaler_stats: Mapping[int, tuple[float, float]] | None = None,
    tp_min: float = 0.5,
    tp_max: float = 30.0,
) -> tuple[torch.Tensor, str]:
    """Resolve the physically interpretable ALE response for one target head."""
    physical_pred = recover_physical_predictions(
        output,
        target=target,
        transfer_scaler_stats=transfer_scaler_stats,
        tp_min=tp_min,
        tp_max=tp_max,
    )
    if head in {"hs", "tp"}:
        representation = "predicted_hs_m" if head == "hs" else "predicted_tp_s"
        return physical_pred[head], representation

    reference = target.get("reference") if isinstance(target, Mapping) else None
    if torch.is_tensor(reference):
        reference_row = reference.reshape(-1, 4)
        ref_idx = 2 if str(head) == "dir" else 3
        response = circular_error_deg(reference_row[:, ref_idx], physical_pred[head])
        representation = (
            "directional_transfer_deg" if str(head) == "dir" else "peak_direction_transfer_deg"
        )
        return response, representation

    representation = (
        "predicted_direction_deg" if str(head) == "dir" else "predicted_peak_direction_deg"
    )
    return physical_pred[head], representation


def _ale_interval_difference(
    lower_response: torch.Tensor,
    upper_response: torch.Tensor,
    *,
    head: str,
    representation: str,
) -> np.ndarray:
    if str(head) in {"dir", "dp"} or "direction" in str(representation):
        return circular_error_deg(
            lower_response.detach().cpu().numpy().reshape(-1),
            upper_response.detach().cpu().numpy().reshape(-1),
        ).astype(float)
    return (
        upper_response.detach().cpu().numpy().reshape(-1)
        - lower_response.detach().cpu().numpy().reshape(-1)
    ).astype(float)


def _bootstrap_centered_curves(
    site_effect_tables: Sequence[pd.DataFrame],
    level_weights: np.ndarray,
    *,
    bootstrap_repeats: int,
    random_seed: int,
    min_sites_per_bin: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    bin_count = len(site_effect_tables)
    if bootstrap_repeats <= 0 or bin_count == 0:
        nan = np.full(bin_count, np.nan, dtype=float)
        return nan, nan, np.zeros(bin_count, dtype=int)

    prepared_tables = []
    for table in site_effect_tables:
        values = pd.to_numeric(table["site_effect"], errors="coerce").to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if values.size < max(1, int(min_sites_per_bin)):
            nan = np.full(bin_count, np.nan, dtype=float)
            return nan, nan, np.zeros(bin_count, dtype=int)
        prepared_tables.append(values)

    rng = np.random.default_rng(int(random_seed))
    curves: list[np.ndarray] = []
    for _ in range(int(bootstrap_repeats)):
        interval_means = np.array(
            [
                float(np.mean(rng.choice(values, size=values.size, replace=True)))
                for values in prepared_tables
            ],
            dtype=float,
        )
        curve = np.cumsum(interval_means)
        curve = _weighted_center(curve, np.asarray(level_weights, dtype=float))
        curves.append(curve)

    curve_array = np.vstack(curves).astype(float, copy=False)
    valid_counts = np.full(bin_count, curve_array.shape[0], dtype=int)
    return (
        np.nanpercentile(curve_array, 2.5, axis=0),
        np.nanpercentile(curve_array, 97.5, axis=0),
        valid_counts,
    )


def _ale_reliability_flag(
    *,
    analysis_kind: str,
    reliable_bins: int,
    min_unique_sites: int,
    normalized_range: float,
    max_ci_width: float,
    effect_range: float,
    fraction_bins_excluding_zero: float,
) -> str:
    if analysis_kind == "unsupported":
        return "insufficient bin support"
    if analysis_kind == "discrete":
        return "low-cardinality feature"
    if reliable_bins < 3 or min_unique_sites < 5:
        return "insufficient bin support"
    if not np.isfinite(normalized_range) or normalized_range < 0.1:
        return "weak effect"
    if (
        normalized_range >= 0.25
        and fraction_bins_excluding_zero >= 0.5
        and np.isfinite(effect_range)
        and effect_range > 0
        and np.isfinite(max_ci_width)
        and max_ci_width <= effect_range
    ):
        return "strong and well-supported"
    return "moderate"


def _build_ale_reliability_row(
    ale_df: pd.DataFrame,
    *,
    head: str,
    feature_name: str,
    target_std: float,
    skip_reason: str | None = None,
) -> dict[str, Any]:
    analysis_kind = (
        str(ale_df.get("analysis_kind", pd.Series(["unsupported"])).iloc[0]).strip().lower()
    )
    feature_label = str(ale_df.get("feature_label", pd.Series([feature_name])).iloc[0])
    feature_unit = str(ale_df.get("feature_unit", pd.Series([""])).iloc[0])
    output_representation = str(ale_df.get("output_representation", pd.Series([""])).iloc[0])
    min_unique_sites = (
        int(pd.to_numeric(ale_df.get("n_unique_sites", pd.Series([0])), errors="coerce").min())
        if not ale_df.empty
        else 0
    )
    reliable_bins = int(len(ale_df))
    effect_range = (
        float(np.nanmax(ale_df["ale_value"]) - np.nanmin(ale_df["ale_value"]))
        if len(ale_df)
        else np.nan
    )
    normalized_range = (
        float(effect_range / target_std) if np.isfinite(target_std) and target_std > 0 else np.nan
    )
    ci_width = (
        pd.to_numeric(ale_df["upper_confidence"], errors="coerce")
        - pd.to_numeric(ale_df["lower_confidence"], errors="coerce")
        if {"upper_confidence", "lower_confidence"}.issubset(ale_df.columns)
        else pd.Series(dtype=float)
    )
    ci_values = ci_width.to_numpy(dtype=float) if len(ci_width) else np.array([], dtype=float)
    finite_ci = ci_values[np.isfinite(ci_values)]
    max_ci_width = float(np.max(finite_ci)) if finite_ci.size else np.nan
    excludes_zero = (
        (
            (pd.to_numeric(ale_df["lower_confidence"], errors="coerce") > 0)
            | (pd.to_numeric(ale_df["upper_confidence"], errors="coerce") < 0)
        ).astype(float)
        if {"upper_confidence", "lower_confidence"}.issubset(ale_df.columns)
        else pd.Series(dtype=float)
    )
    fraction_bins_excluding_zero = (
        float(np.nanmean(excludes_zero.to_numpy(dtype=float))) if len(excludes_zero) else np.nan
    )
    feature_min = (
        float(np.nanmin(pd.to_numeric(ale_df["bin_lower"], errors="coerce")))
        if "bin_lower" in ale_df.columns and len(ale_df)
        else np.nan
    )
    feature_max = (
        float(np.nanmax(pd.to_numeric(ale_df["bin_upper"], errors="coerce")))
        if "bin_upper" in ale_df.columns and len(ale_df)
        else np.nan
    )
    reliability_flag = _ale_reliability_flag(
        analysis_kind=analysis_kind,
        reliable_bins=reliable_bins,
        min_unique_sites=min_unique_sites,
        normalized_range=normalized_range,
        max_ci_width=max_ci_width,
        effect_range=effect_range,
        fraction_bins_excluding_zero=0.0
        if not np.isfinite(fraction_bins_excluding_zero)
        else fraction_bins_excluding_zero,
    )
    if skip_reason:
        reliability_flag = (
            "low-cardinality feature"
            if "cardinality" in str(skip_reason)
            else "insufficient bin support"
        )
    return {
        "target": str(head),
        "feature": str(feature_name),
        "feature_label": feature_label,
        "feature_unit": feature_unit,
        "analysis_kind": analysis_kind,
        "feature_range_min": feature_min,
        "feature_range_max": feature_max,
        "n_unique_sites": int(ale_df["n_unique_sites"].max())
        if "n_unique_sites" in ale_df.columns and len(ale_df)
        else 0,
        "reliable_bins": int(reliable_bins),
        "total_ale_range": effect_range,
        "normalized_ale_range": normalized_range,
        "max_confidence_interval_width": max_ci_width,
        "fraction_bins_excluding_zero": fraction_bins_excluding_zero,
        "minimum_unique_site_count": min_unique_sites,
        "output_representation": output_representation,
        "reliability_flag": reliability_flag,
        "skip_reason": skip_reason,
    }


def _ale_target_standard_deviation(
    analysis_frame: pd.DataFrame,
    *,
    head: str,
) -> float:
    if str(head) == "hs" and "target_hs" in analysis_frame.columns:
        series = pd.to_numeric(analysis_frame["target_hs"], errors="coerce")
    elif str(head) == "tp" and "target_tp" in analysis_frame.columns:
        series = pd.to_numeric(analysis_frame["target_tp"], errors="coerce")
    elif str(head) == "dir":
        source_col = (
            "target_transfer_dir_delta_deg"
            if "target_transfer_dir_delta_deg" in analysis_frame.columns
            else "target_dir_deg"
        )
        series = pd.to_numeric(analysis_frame[source_col], errors="coerce")
    else:
        source_col = (
            "target_transfer_dp_delta_deg"
            if "target_transfer_dp_delta_deg" in analysis_frame.columns
            else "target_dp_deg"
        )
        series = pd.to_numeric(analysis_frame[source_col], errors="coerce")
    values = series.to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    return float(np.std(values)) if values.size else float("nan")


def _static_feature_value_label(
    shap_matrix: np.ndarray,
    value_sources: Sequence[str],
    *,
    max_display: int = 20,
) -> str:
    ranked = np.argsort(np.mean(np.abs(shap_matrix), axis=0))[::-1][
        : int(min(max_display, shap_matrix.shape[1]))
    ]
    display_sources = [str(value_sources[idx]) for idx in ranked]
    if display_sources and all(source == "raw" for source in display_sources):
        return "Feature value"
    if display_sources and all(source != "raw" for source in display_sources):
        return "Standardized value"
    return "Feature value"


def _static_shap_quantity_axis_target(head: str, quantity: str) -> str:
    quantity_key = str(quantity).strip().lower()
    if quantity_key == "physical_prediction":
        mapping = {
            "hs": "predicted Hs",
            "tp": "predicted Tp",
            "dir": "predicted mean direction",
            "dp": "predicted peak direction",
        }
        return mapping.get(str(head), f"predicted {_static_shap_target_display_name(str(head))}")
    if quantity_key == "absolute_error":
        return f"absolute error in {_static_shap_target_display_name(str(head))}"
    if quantity_key == "signed_error":
        return f"signed error in {_static_shap_target_display_name(str(head))}"
    if quantity_key == "entropy":
        return f"{_static_shap_target_display_name(str(head))} predictive entropy"
    if quantity_key == "selected_logit":
        return f"{_static_shap_target_display_name(str(head))} selected logit"
    return f"{_static_shap_target_display_name(str(head))} ({str(quantity)})"


def _static_shap_figure_title(head: str, quantity: str) -> str:
    quantity_key = str(quantity).strip().lower()
    target_name = _static_shap_target_display_name(str(head))
    if quantity_key == "physical_prediction":
        return f"Static SHAP: {target_name}"
    if quantity_key == "absolute_error":
        return f"Static SHAP: {target_name} absolute error"
    if quantity_key == "signed_error":
        return f"Static SHAP: {target_name} signed error"
    if quantity_key == "entropy":
        return f"Static SHAP: {target_name} entropy"
    if quantity_key == "selected_logit":
        return f"Static SHAP: {target_name} selected logit"
    return f"Static SHAP: {target_name}"


def _select_top_static_shap_features(
    shap_matrix: np.ndarray,
    feature_matrix: np.ndarray,
    feature_names: Sequence[str],
    *,
    max_display: int = 10,
) -> dict[str, Any]:
    if shap_matrix.ndim != 2:
        raise ValueError(
            f"SHAP values must be 2D after target selection, got shape {shap_matrix.shape}"
        )
    if feature_matrix.ndim != 2:
        raise ValueError(f"Feature values must be 2D, got shape {feature_matrix.shape}")
    if shap_matrix.shape[0] != feature_matrix.shape[0]:
        raise ValueError(
            f"SHAP values and feature values must share the same sample dimension, got {shap_matrix.shape[0]} and {feature_matrix.shape[0]}"
        )
    if shap_matrix.shape[1] != feature_matrix.shape[1]:
        raise ValueError(
            f"SHAP values and feature values must share the same feature dimension, got {shap_matrix.shape[1]} and {feature_matrix.shape[1]}"
        )
    if shap_matrix.shape[1] != len(feature_names):
        raise ValueError(
            f"Feature-name length {len(feature_names)} does not match feature dimension {shap_matrix.shape[1]}"
        )

    importance = np.mean(np.abs(shap_matrix), axis=0)
    ranking = np.argsort(importance)[::-1]
    top_indices = ranking[: int(min(max_display, len(ranking)))]
    selected_shap = np.asarray(shap_matrix[:, top_indices], dtype=float)
    selected_features = np.asarray(feature_matrix[:, top_indices], dtype=float)
    if not np.allclose(selected_shap, shap_matrix[:, top_indices], equal_nan=True):
        raise ValueError(
            "Selected SHAP values must remain the original signed per-feature values without transformation"
        )
    if selected_shap.shape != selected_features.shape:
        raise ValueError(
            f"Selected SHAP and feature matrices must align exactly, got {selected_shap.shape} and {selected_features.shape}"
        )
    return {
        "indices": top_indices.astype(int),
        "selected_shap": selected_shap,
        "selected_feature_values": selected_features,
        "selected_feature_names": [str(feature_names[idx]) for idx in top_indices],
        "importance": np.asarray(importance[top_indices], dtype=float),
    }


def _filter_static_shap_display_samples_by_percentile(
    shap_matrix: np.ndarray,
    feature_matrix: np.ndarray,
    *,
    base_values: np.ndarray | None = None,
    percentile_range: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray, dict[str, float] | None]:
    if percentile_range is None:
        return (
            np.asarray(shap_matrix, dtype=float),
            np.asarray(feature_matrix, dtype=float),
            None if base_values is None else np.asarray(base_values, dtype=float),
            np.ones(np.asarray(shap_matrix).shape[0], dtype=bool),
            None,
        )

    lower_pct, upper_pct = (float(percentile_range[0]), float(percentile_range[1]))
    if not (0.0 <= lower_pct < upper_pct <= 100.0):
        raise ValueError(
            f"static_shap_display_percentile_range must satisfy 0 <= lower < upper <= 100; got {percentile_range}"
        )

    shap_array = np.asarray(shap_matrix, dtype=float)
    feature_array = np.asarray(feature_matrix, dtype=float)
    if shap_array.shape != feature_array.shape:
        raise ValueError(
            f"Display-filter SHAP and feature matrices must have identical shape, got {shap_array.shape} and {feature_array.shape}"
        )

    sample_strength = np.max(np.abs(shap_array), axis=1)
    finite_nonzero = sample_strength[np.isfinite(sample_strength) & (sample_strength > 0.0)]
    if finite_nonzero.size < 3:
        return (
            shap_array,
            feature_array,
            base_values,
            np.ones(shap_array.shape[0], dtype=bool),
            None,
        )

    lower_cut = float(np.percentile(finite_nonzero, lower_pct))
    upper_cut = float(np.percentile(finite_nonzero, upper_pct))
    row_mask = (
        np.isfinite(sample_strength)
        & (sample_strength >= lower_cut)
        & (sample_strength <= upper_cut)
    )
    if not np.any(row_mask):
        row_mask = np.ones(shap_array.shape[0], dtype=bool)
        return shap_array, feature_array, base_values, row_mask, None

    filtered_base = None
    if base_values is not None:
        base_array = np.asarray(base_values, dtype=float).reshape(-1)
        if base_array.shape[0] != shap_array.shape[0]:
            raise ValueError(
                f"base_values length {base_array.shape[0]} does not match sample count {shap_array.shape[0]}"
            )
        filtered_base = base_array[row_mask]

    return (
        shap_array[row_mask],
        feature_array[row_mask],
        filtered_base,
        row_mask,
        {
            "lower_percentile": lower_pct,
            "upper_percentile": upper_pct,
            "lower_cut": lower_cut,
            "upper_cut": upper_cut,
            "kept_samples": float(np.sum(row_mask)),
        },
    )


def _drop_non_finite_static_rows(
    shap_matrix: np.ndarray,
    feature_matrix: np.ndarray,
    *,
    base_values: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray]:
    shap_finite = np.all(np.isfinite(shap_matrix), axis=1)
    feature_finite = np.all(np.isfinite(feature_matrix), axis=1)
    mask = shap_finite & feature_finite
    if base_values is not None:
        base_array = np.asarray(base_values, dtype=float).reshape(-1)
        if base_array.size == 1:
            base_array = np.repeat(base_array, repeats=shap_matrix.shape[0]).astype(
                float, copy=False
            )
        if base_array.shape[0] != shap_matrix.shape[0]:
            raise ValueError(
                f"base_values length {base_array.shape[0]} does not match sample count {shap_matrix.shape[0]}"
            )
        mask &= np.isfinite(base_array)
        base_values = base_array[mask]
    return shap_matrix[mask], feature_matrix[mask], base_values, mask


def _build_top_static_shap_importance_table(
    *,
    head: str,
    top_feature_names: Sequence[str],
    top_importance: Sequence[float],
    feature_groups: Mapping[str, Any],
    value_sources: Sequence[str],
    raw_feature_map: Mapping[str, str],
) -> tuple[pd.DataFrame, dict[str, str], list[str]]:
    feature_to_group = {
        str(key): str(value)
        for key, value in dict(feature_groups.get("feature_to_group", {})).items()
    }
    readable_name_map: dict[str, str] = {}
    standardized_features: list[str] = []
    rows: list[dict[str, Any]] = []

    for rank, (feature_name, mean_abs_shap, value_source) in enumerate(
        zip(top_feature_names, top_importance, value_sources),
        start=1,
    ):
        internal_name = str(feature_name)
        source = str(value_source)
        raw_feature_name = raw_feature_map.get(internal_name) if source == "raw" else None
        normalization_status = (
            "inverse_mapped_to_physical_units" if source == "raw" else "standardized_model_input"
        )
        plot_meta = _resolve_plot_metadata(
            internal_name,
            raw_feature_name=raw_feature_name,
            normalization_status=normalization_status,
        )
        readable_name_map[internal_name] = str(plot_meta["display_name"])
        if source != "raw":
            standardized_features.append(internal_name)
        rows.append(
            {
                "target": str(head),
                "rank": int(rank),
                "feature": internal_name,
                "mean_abs_shap": float(mean_abs_shap),
                "feature_group": feature_to_group.get(internal_name, "misc_static"),
                "display_unit": str(plot_meta["display_unit"]),
            }
        )
    return pd.DataFrame(rows), readable_name_map, standardized_features


def _weighted_center(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    centered = np.asarray(values, dtype=float).copy()
    if centered.size == 0:
        return centered
    weight_sum = float(np.sum(weights))
    offset = (
        float(np.sum(centered * weights) / weight_sum)
        if weight_sum > 0
        else float(np.mean(centered))
    )
    return centered - offset


def _bootstrap_ale_curves(
    effect_samples: Sequence[np.ndarray],
    sample_counts: np.ndarray,
    *,
    bootstrap_repeats: int,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if bootstrap_repeats <= 0 or not effect_samples:
        nan = np.full(len(effect_samples), np.nan, dtype=float)
        return nan, nan

    rng = np.random.default_rng(random_seed)
    boot_curves = np.full((int(bootstrap_repeats), len(effect_samples)), np.nan, dtype=float)
    for repeat_idx in range(int(bootstrap_repeats)):
        interval_effects = []
        for effects in effect_samples:
            if effects.size == 0:
                interval_effects.append(np.nan)
                continue
            sampled = rng.choice(effects, size=effects.size, replace=True)
            interval_effects.append(float(np.mean(sampled)))
        curve = np.cumsum(np.asarray(interval_effects, dtype=float))
        curve = _weighted_center(curve, sample_counts)
        boot_curves[repeat_idx] = curve
    return (
        np.nanpercentile(boot_curves, 2.5, axis=0),
        np.nanpercentile(boot_curves, 97.5, axis=0),
    )


def _subset_batch_rows(batch: Mapping[str, Any], row_mask: np.ndarray) -> dict[str, Any]:
    subset: dict[str, Any] = {}
    for key, value in dict(batch).items():
        if torch.is_tensor(value):
            subset[key] = value[row_mask]
        elif isinstance(value, dict):
            subset[key] = {
                inner_key: inner_value[row_mask] if torch.is_tensor(inner_value) else inner_value
                for inner_key, inner_value in value.items()
            }
        else:
            subset[key] = value
    return subset


def _default_selected_static_ale_features(
    feature_names: Sequence[str],
    *,
    head: str,
    limit: int = 4,
) -> list[str]:
    available = set(str(name) for name in feature_names)
    desired = [
        feature
        for feature in DEFAULT_STATIC_ALE_FEATURE_SELECTIONS.get(head, ())
        if feature in available
    ]
    if desired:
        return desired[:limit]
    fallback = [
        feature
        for feature in (
            "local_depth_m",
            "ray_fetch_mean_m",
            "ray_fetch_max_m",
            "open_sector_width_deg",
            "path_length_m",
            "path_tortuosity_ratio",
            "funneling_ratio",
            "static_porosity_1km",
        )
        if feature in available
    ]
    return fallback[:limit]


def _masked_batch(batch: dict[str, Any], key: str, value: torch.Tensor | None) -> dict[str, Any]:
    cloned = dict(batch)
    cloned[key] = value
    return cloned


def compute_group_permutation_importance(
    runner: ExplainabilityModelBundle,
    *,
    head: str,
    quantity: str,
    sample_indices: Sequence[int],
    feature_groups: Mapping[str, Any],
    group_type: str = "static",
    random_seed: int = 42,
) -> pd.DataFrame:
    """Permutation importance for grouped static or dynamic features."""
    rng = np.random.default_rng(random_seed)
    batch = _batchify_samples(runner, sample_indices)
    if group_type == "static":
        tensor_key = "x_static"
        feature_names = list(getattr(runner.arrays, "static_feature_names", []) or [])
    elif group_type == "dynamic":
        tensor_key = "x_dynamic"
        feature_names = list(getattr(runner.arrays, "dynamic_feature_names", []) or [])
    else:
        tensor_key = "x_dynamic_sources"
        feature_names = list(getattr(runner.arrays, "source_feature_names", []) or [])

    input_tensor = batch.get(tensor_key)
    if not torch.is_tensor(input_tensor):
        return pd.DataFrame(columns=["group", "mean_absolute_change"])

    baseline_output = safe_forward(runner.model, batch)
    baseline_quantity = resolve_quantity_tensor(
        baseline_output,
        batch["y"],
        head=head,
        quantity=quantity,
        transfer_scaler_stats=runner.transfer_scaler_stats,
    ).detach()

    rows = []
    for group_name, columns in dict(feature_groups.get("groups", {})).items():
        indices = [feature_names.index(name) for name in columns if name in feature_names]
        if not indices:
            continue
        permuted = _clone_tensor(input_tensor)
        if permuted is None:
            continue
        order = rng.permutation(permuted.size(0))
        if permuted.ndim == 2:
            permuted[:, indices] = permuted[order][:, indices]
        elif permuted.ndim == 3:
            permuted[:, :, indices] = permuted[order][:, :, indices]
        else:
            permuted[:, :, :, indices] = permuted[order][:, :, :, indices]
        permuted_output = safe_forward(runner.model, _masked_batch(batch, tensor_key, permuted))
        permuted_quantity = resolve_quantity_tensor(
            permuted_output,
            batch["y"],
            head=head,
            quantity=quantity,
            transfer_scaler_stats=runner.transfer_scaler_stats,
        ).detach()
        rows.append(
            {
                "group": group_name,
                "mean_absolute_change": float(
                    torch.mean(torch.abs(permuted_quantity - baseline_quantity)).cpu()
                ),
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values("mean_absolute_change", ascending=False)
        .reset_index(drop=True)
    )


def compute_static_ale(
    runner_map: Mapping[str, ExplainabilityModelBundle],
    *,
    head: str,
    analysis_frame: pd.DataFrame,
    feature_name: str,
    bins: int = 6,
    min_unique_sites_per_bin: int = 5,
    bootstrap_repeats: int = 500,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Site-balanced first-order ALE over held-out sites."""
    if analysis_frame.empty:
        return pd.DataFrame()

    primary_runner = next(iter(runner_map.values()))
    feature_names = list(getattr(primary_runner.arrays, "static_feature_names", []) or [])
    if feature_name not in feature_names:
        return pd.DataFrame()
    feature_idx = feature_names.index(feature_name)

    batch = _batchify_ale_frame_rows(runner_map, analysis_frame)
    x_static = batch["x_static"].detach().clone()
    transformed_values = x_static[:, feature_idx].detach().cpu().numpy().astype(float, copy=False)
    feature_payload = recover_static_feature_plot_values(
        analysis_frame,
        feature_name=str(feature_name),
        transformed_values=transformed_values,
        arrays=primary_runner.arrays,
    )

    site_frame = feature_payload["site_frame"].copy()
    bin_payload = build_adaptive_site_bins(
        site_frame,
        feature_name=str(feature_name),
        target_bins=int(bins),
        min_unique_sites_per_bin=int(min_unique_sites_per_bin),
    )
    if bin_payload.get("analysis_kind") != "continuous":
        return pd.DataFrame()

    plot_values = np.asarray(feature_payload["plot_values"], dtype=float)
    transformed_values = np.asarray(transformed_values, dtype=float)
    edges = np.asarray(bin_payload["edges"], dtype=float)
    site_bin_lookup = (
        site_frame.assign(bin_id=np.asarray(bin_payload["bin_ids"], dtype=int))
        .set_index("site")["bin_id"]
        .to_dict()
    )
    row_bin_ids = (
        analysis_frame["site"].astype(str).map(site_bin_lookup).fillna(-1).astype(int).to_numpy()
    )

    rows: list[dict[str, Any]] = []
    site_effect_tables: list[pd.DataFrame] = []
    output_representation: str | None = None

    for bin_idx, (left_plot, right_plot) in enumerate(zip(edges[:-1], edges[1:])):
        row_mask = row_bin_ids == int(bin_idx)
        if not np.any(row_mask):
            continue
        subset_rows = analysis_frame.loc[row_mask].reset_index(drop=True)
        subset_batch = _batchify_ale_frame_rows(runner_map, subset_rows)
        lower = subset_batch["x_static"].detach().clone()
        upper = subset_batch["x_static"].detach().clone()
        plot_to_transformed = feature_payload["plot_to_transformed"]
        left_transformed = float(
            np.asarray(
                plot_to_transformed(np.asarray([left_plot], dtype=float)), dtype=float
            ).reshape(-1)[0]
        )
        right_transformed = float(
            np.asarray(
                plot_to_transformed(np.asarray([right_plot], dtype=float)), dtype=float
            ).reshape(-1)[0]
        )
        lower[:, feature_idx] = left_transformed
        upper[:, feature_idx] = right_transformed

        any_runner = next(iter(runner_map.values()))
        lower_output = safe_forward(
            any_runner.model, _masked_batch(subset_batch, "x_static", lower)
        )
        upper_output = safe_forward(
            any_runner.model, _masked_batch(subset_batch, "x_static", upper)
        )
        lower_response, representation = resolve_ale_response_tensor(
            lower_output,
            subset_batch["y"],
            head=str(head),
            transfer_scaler_stats=any_runner.transfer_scaler_stats,
        )
        upper_response, _ = resolve_ale_response_tensor(
            upper_output,
            subset_batch["y"],
            head=str(head),
            transfer_scaler_stats=any_runner.transfer_scaler_stats,
        )
        output_representation = str(representation)
        local_effects = _ale_interval_difference(
            lower_response,
            upper_response,
            head=str(head),
            representation=str(representation),
        )
        effect_df = pd.DataFrame(
            {
                "site": subset_rows["site"].astype(str).to_numpy(),
                "local_effect": local_effects,
            }
        )
        site_effects = (
            effect_df.groupby("site", as_index=False)["local_effect"]
            .mean()
            .rename(columns={"local_effect": "site_effect"})
        )
        if len(site_effects) < int(min_unique_sites_per_bin):
            continue
        site_effect_tables.append(site_effects.copy())
        rows.append(
            {
                "bin_id": int(bin_idx),
                "bin_lower": float(left_plot),
                "bin_upper": float(right_plot),
                "bin_center": float((left_plot + right_plot) * 0.5),
                "bin_left_transformed": float(left_transformed),
                "bin_right_transformed": float(right_transformed),
                "bin_center_transformed": float((left_transformed + right_transformed) * 0.5),
                "interval_effect": float(site_effects["site_effect"].mean()),
                "n_unique_sites": int(site_effects["site"].nunique()),
                "n_temporal_samples": int(len(effect_df)),
            }
        )

    if len(rows) < 3:
        return pd.DataFrame()

    ale_df = pd.DataFrame(rows).sort_values("bin_id").reset_index(drop=True)
    if not np.all(np.diff(ale_df["bin_lower"].to_numpy(dtype=float)) >= 0):
        raise ValueError(f"ALE bin edges are not ordered for feature '{feature_name}'")
    level_weights = ale_df["n_unique_sites"].to_numpy(dtype=float)
    ale_curve = np.cumsum(ale_df["interval_effect"].to_numpy(dtype=float))
    ale_curve = _weighted_center(ale_curve, level_weights)
    lower_ci, upper_ci, valid_bootstrap = _bootstrap_centered_curves(
        site_effect_tables,
        level_weights,
        bootstrap_repeats=int(bootstrap_repeats),
        random_seed=int(random_seed),
        min_sites_per_bin=int(min_unique_sites_per_bin),
    )
    if not np.isfinite(np.average(ale_curve, weights=level_weights)):
        raise ValueError(f"ALE centering failed for feature '{feature_name}'")

    ale_df["target"] = str(head)
    ale_df["feature"] = str(feature_name)
    ale_df["feature_label"] = str(feature_payload["feature_label"])
    ale_df["feature_unit"] = str(feature_payload["feature_unit"])
    ale_df["ale_value"] = ale_curve
    ale_df["lower_confidence"] = lower_ci
    ale_df["upper_confidence"] = upper_ci
    ale_df["normalization_status"] = str(feature_payload["normalization_status"])
    ale_df["output_representation"] = str(output_representation or "")
    ale_df["n_valid_bootstrap_replicates"] = valid_bootstrap
    ale_df["weighted_mean_ale"] = float(np.average(ale_curve, weights=level_weights))
    ale_df["analysis_kind"] = "continuous"
    return ale_df[
        [
            "target",
            "feature",
            "feature_label",
            "feature_unit",
            "analysis_kind",
            "bin_lower",
            "bin_upper",
            "bin_center",
            "ale_value",
            "lower_confidence",
            "upper_confidence",
            "n_unique_sites",
            "n_temporal_samples",
            "n_valid_bootstrap_replicates",
            "output_representation",
            "normalization_status",
            "bin_left_transformed",
            "bin_right_transformed",
            "bin_center_transformed",
            "weighted_mean_ale",
        ]
    ]


def compute_discrete_static_effect(
    runner_map: Mapping[str, ExplainabilityModelBundle],
    *,
    head: str,
    analysis_frame: pd.DataFrame,
    feature_name: str,
    bootstrap_repeats: int = 500,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Discrete effect comparison for low-cardinality static features."""
    if analysis_frame.empty:
        return pd.DataFrame()
    primary_runner = next(iter(runner_map.values()))
    feature_names = list(getattr(primary_runner.arrays, "static_feature_names", []) or [])
    if feature_name not in feature_names:
        return pd.DataFrame()
    feature_idx = feature_names.index(feature_name)
    batch = _batchify_ale_frame_rows(runner_map, analysis_frame)
    x_static = batch["x_static"].detach().clone()
    transformed_values = x_static[:, feature_idx].detach().cpu().numpy().astype(float, copy=False)
    feature_payload = recover_static_feature_plot_values(
        analysis_frame,
        feature_name=str(feature_name),
        transformed_values=transformed_values,
        arrays=primary_runner.arrays,
    )
    site_frame = feature_payload["site_frame"].copy()
    level_values = np.unique(site_frame["plot_value"].to_numpy(dtype=float))
    if level_values.size < 2:
        return pd.DataFrame()

    plot_to_transformed = feature_payload["plot_to_transformed"]
    site_level_counts = (
        site_frame.groupby("plot_value", as_index=False)["site"]
        .nunique()
        .rename(columns={"site": "n_unique_sites"})
        .sort_values("plot_value")
        .reset_index(drop=True)
    )
    level_weights = site_level_counts["n_unique_sites"].to_numpy(dtype=float)
    rows: list[dict[str, Any]] = []
    site_effect_tables: list[pd.DataFrame] = []
    output_representation: str | None = None
    any_runner = next(iter(runner_map.values()))

    for level_value in site_level_counts["plot_value"].to_numpy(dtype=float):
        perturbed = x_static.detach().clone()
        transformed_level = float(
            np.asarray(
                plot_to_transformed(np.asarray([level_value], dtype=float)), dtype=float
            ).reshape(-1)[0]
        )
        perturbed[:, feature_idx] = transformed_level
        output = safe_forward(any_runner.model, _masked_batch(batch, "x_static", perturbed))
        response, representation = resolve_ale_response_tensor(
            output,
            batch["y"],
            head=str(head),
            transfer_scaler_stats=any_runner.transfer_scaler_stats,
        )
        output_representation = str(representation)
        response_np = response.detach().cpu().numpy().reshape(-1).astype(float)
        site_response = (
            pd.DataFrame(
                {"site": analysis_frame["site"].astype(str).to_numpy(), "response": response_np}
            )
            .groupby("site", as_index=False)["response"]
            .mean()
            .rename(columns={"response": "site_effect"})
        )
        site_effect_tables.append(site_response.copy())
        observed_sites = int(
            site_level_counts.loc[
                site_level_counts["plot_value"] == level_value, "n_unique_sites"
            ].iloc[0]
        )
        rows.append(
            {
                "bin_lower": float(level_value),
                "bin_upper": float(level_value),
                "bin_center": float(level_value),
                "bin_left_transformed": float(transformed_level),
                "bin_right_transformed": float(transformed_level),
                "bin_center_transformed": float(transformed_level),
                "level_response": float(site_response["site_effect"].mean()),
                "n_unique_sites": observed_sites,
                "n_temporal_samples": int(
                    observed_sites * int(analysis_frame["effective_samples_per_site"].iloc[0])
                ),
            }
        )

    discrete_df = pd.DataFrame(rows).sort_values("bin_center").reset_index(drop=True)
    centered = _weighted_center(discrete_df["level_response"].to_numpy(dtype=float), level_weights)
    lower_ci, upper_ci, valid_bootstrap = _bootstrap_centered_curves(
        site_effect_tables,
        level_weights,
        bootstrap_repeats=int(bootstrap_repeats),
        random_seed=int(random_seed),
        min_sites_per_bin=1,
    )
    discrete_df["target"] = str(head)
    discrete_df["feature"] = str(feature_name)
    discrete_df["feature_label"] = str(feature_payload["feature_label"])
    discrete_df["feature_unit"] = str(feature_payload["feature_unit"])
    discrete_df["analysis_kind"] = "discrete"
    discrete_df["ale_value"] = centered
    discrete_df["lower_confidence"] = lower_ci
    discrete_df["upper_confidence"] = upper_ci
    discrete_df["n_valid_bootstrap_replicates"] = valid_bootstrap
    discrete_df["output_representation"] = str(output_representation or "")
    discrete_df["normalization_status"] = str(feature_payload["normalization_status"])
    discrete_df["weighted_mean_ale"] = float(np.average(centered, weights=level_weights))
    return discrete_df[
        [
            "target",
            "feature",
            "feature_label",
            "feature_unit",
            "analysis_kind",
            "bin_lower",
            "bin_upper",
            "bin_center",
            "ale_value",
            "lower_confidence",
            "upper_confidence",
            "n_unique_sites",
            "n_temporal_samples",
            "n_valid_bootstrap_replicates",
            "output_representation",
            "normalization_status",
            "bin_left_transformed",
            "bin_right_transformed",
            "bin_center_transformed",
            "weighted_mean_ale",
        ]
    ]


def compute_static_counterfactual_curves(
    runner_map: Mapping[str, ExplainabilityModelBundle],
    *,
    head: str,
    analysis_frame: pd.DataFrame,
    feature_name: str,
    num_steps: int = 9,
) -> pd.DataFrame:
    """Counterfactual response curve for one static feature."""
    if analysis_frame.empty:
        return pd.DataFrame(columns=["feature_value", "response"])
    primary_runner = next(iter(runner_map.values()))
    feature_names = list(getattr(primary_runner.arrays, "static_feature_names", []) or [])
    if feature_name not in feature_names:
        return pd.DataFrame(columns=["feature_value", "response"])
    feature_idx = feature_names.index(feature_name)
    batch = _batchify_ale_frame_rows(runner_map, analysis_frame)
    x_static = batch["x_static"].detach().clone()
    transformed_values = x_static[:, feature_idx].detach().cpu().numpy().astype(float, copy=False)
    feature_payload = recover_static_feature_plot_values(
        analysis_frame,
        feature_name=str(feature_name),
        transformed_values=transformed_values,
        arrays=primary_runner.arrays,
    )
    plot_values = np.asarray(feature_payload["plot_values"], dtype=float)
    grid = np.linspace(
        float(np.nanpercentile(plot_values, 5)),
        float(np.nanpercentile(plot_values, 95)),
        num=max(3, num_steps),
    )
    rows = []
    plot_to_transformed = feature_payload["plot_to_transformed"]
    for point in grid:
        perturbed = x_static.clone()
        transformed_point = float(
            np.asarray(plot_to_transformed(np.asarray([point], dtype=float)), dtype=float).reshape(
                -1
            )[0]
        )
        perturbed[:, feature_idx] = transformed_point
        response, representation = resolve_ale_response_tensor(
            safe_forward(primary_runner.model, _masked_batch(batch, "x_static", perturbed)),
            batch["y"],
            head=head,
            transfer_scaler_stats=primary_runner.transfer_scaler_stats,
        )
        rows.append(
            {
                "feature_value": float(point),
                "response": float(torch.mean(response).detach().cpu()),
                "output_representation": str(representation),
                "feature_unit": str(feature_payload["feature_unit"]),
                "feature_label": str(feature_payload["feature_label"]),
            }
        )
    return pd.DataFrame(rows)


def _integrated_gradients_for_tensor(
    runner: ExplainabilityModelBundle,
    batch: dict[str, Any],
    *,
    input_key: str,
    head: str,
    quantity: str,
    baseline: torch.Tensor | None = None,
    steps: int = 16,
) -> torch.Tensor:
    original = batch[input_key]
    if not torch.is_tensor(original):
        raise ValueError(f"Batch does not contain tensor '{input_key}'")
    base = (
        torch.zeros_like(original)
        if baseline is None
        else baseline.to(device=original.device, dtype=original.dtype)
    )
    total_grads = torch.zeros_like(original)
    for alpha in torch.linspace(0.0, 1.0, steps=steps, device=original.device):
        interpolated = base + alpha * (original - base)
        interpolated.requires_grad_(True)
        working = dict(batch)
        working[input_key] = interpolated
        runner.model.zero_grad(set_to_none=True)
        output = safe_forward(runner.model, working)
        quantity_tensor = resolve_quantity_tensor(
            output,
            batch["y"],
            head=head,
            quantity=quantity,
            transfer_scaler_stats=runner.transfer_scaler_stats,
        )
        quantity_tensor.sum().backward()
        grad = interpolated.grad
        if grad is None:
            raise RuntimeError(f"Failed to compute gradients for '{input_key}'")
        total_grads = total_grads + grad.detach()
    return (original - base) * (total_grads / float(steps))


def compute_temporal_integrated_gradients(
    runner: ExplainabilityModelBundle,
    *,
    head: str,
    quantity: str,
    sample_indices: Sequence[int],
    steps: int = 16,
    batch_size: int | None = None,
) -> dict[str, Any]:
    """Integrated gradients over dynamic tensors."""
    attribution_chunks: list[torch.Tensor] = []
    input_key: str | None = None
    for batch_indices in _iter_sample_index_batches(sample_indices, batch_size):
        batch = _batchify_samples(runner, batch_indices)
        batch_input_key = "x_dynamic_sources" if "x_dynamic_sources" in batch else "x_dynamic"
        attr = _integrated_gradients_for_tensor(
            runner,
            batch,
            input_key=batch_input_key,
            head=head,
            quantity=quantity,
            steps=steps,
        )
        attribution_chunks.append(attr.detach().cpu())
        if input_key is None:
            input_key = batch_input_key
    if input_key is None or not attribution_chunks:
        raise ValueError("No temporal inputs found for integrated gradients")
    attributions = (
        torch.cat(attribution_chunks, dim=0)
        if len(attribution_chunks) > 1
        else attribution_chunks[0]
    )
    feature_names = (
        list(getattr(runner.arrays, "source_feature_names", []) or [])
        if input_key == "x_dynamic_sources"
        else list(getattr(runner.arrays, "dynamic_feature_names", []) or [])
    )
    return {"input_key": input_key, "attributions": attributions, "feature_names": feature_names}


def compute_temporal_occlusion(
    runner: ExplainabilityModelBundle,
    *,
    head: str,
    quantity: str,
    sample_indices: Sequence[int],
    window_sizes: Sequence[int] = (1, 3, 6, 12, 24, 48),
    batch_size: int | None = None,
) -> pd.DataFrame:
    """Lag/window occlusion importance for the temporal branch."""
    totals: dict[int, dict[str, float]] = {}
    for batch_indices in _iter_sample_index_batches(sample_indices, batch_size):
        batch = _batchify_samples(runner, batch_indices)
        input_key = "x_dynamic_sources" if "x_dynamic_sources" in batch else "x_dynamic"
        input_tensor = batch[input_key]
        with torch.inference_mode():
            baseline_output = safe_forward(runner.model, batch)
            baseline_quantity = resolve_quantity_tensor(
                baseline_output,
                batch["y"],
                head=head,
                quantity=quantity,
                transfer_scaler_stats=runner.transfer_scaler_stats,
            ).detach()

            seq_len = int(input_tensor.size(1))
            for window in window_sizes:
                width = min(int(window), seq_len)
                occluded = _clone_tensor(input_tensor)
                assert occluded is not None
                if input_tensor.ndim == 4:
                    occluded[:, seq_len - width :, :, :] = 0.0
                else:
                    occluded[:, seq_len - width :, :] = 0.0
                quantity_tensor = resolve_quantity_tensor(
                    safe_forward(runner.model, _masked_batch(batch, input_key, occluded)),
                    batch["y"],
                    head=head,
                    quantity=quantity,
                    transfer_scaler_stats=runner.transfer_scaler_stats,
                ).detach()
                delta = torch.abs(quantity_tensor - baseline_quantity)
                stats = totals.setdefault(width, {"sum": 0.0, "count": 0.0})
                stats["sum"] += float(delta.sum().cpu())
                stats["count"] += float(delta.numel())
    rows = [
        {
            "window_size": int(width),
            "mean_absolute_change": float(stats["sum"] / max(1.0, stats["count"])),
        }
        for width, stats in sorted(totals.items())
    ]
    return pd.DataFrame(rows)


def compute_source_occlusion(
    runner: ExplainabilityModelBundle,
    *,
    head: str,
    quantity: str,
    sample_indices: Sequence[int],
    batch_size: int | None = None,
) -> pd.DataFrame:
    """Occlude one source at a time in multi-source runs."""
    totals: dict[int, dict[str, float]] = {}
    saw_sources = False
    for batch_indices in _iter_sample_index_batches(sample_indices, batch_size):
        batch = _batchify_samples(runner, batch_indices)
        input_tensor = batch.get("x_dynamic_sources")
        if not torch.is_tensor(input_tensor):
            continue
        saw_sources = True
        with torch.inference_mode():
            baseline_output = safe_forward(runner.model, batch)
            baseline_quantity = resolve_quantity_tensor(
                baseline_output,
                batch["y"],
                head=head,
                quantity=quantity,
                transfer_scaler_stats=runner.transfer_scaler_stats,
            ).detach()
            for source_idx in range(int(input_tensor.size(2))):
                occluded = _clone_tensor(input_tensor)
                assert occluded is not None
                occluded[:, :, source_idx, :] = 0.0
                quantity_tensor = resolve_quantity_tensor(
                    safe_forward(runner.model, _masked_batch(batch, "x_dynamic_sources", occluded)),
                    batch["y"],
                    head=head,
                    quantity=quantity,
                    transfer_scaler_stats=runner.transfer_scaler_stats,
                ).detach()
                delta = torch.abs(quantity_tensor - baseline_quantity)
                stats = totals.setdefault(source_idx, {"sum": 0.0, "count": 0.0})
                stats["sum"] += float(delta.sum().cpu())
                stats["count"] += float(delta.numel())
    if not saw_sources:
        return pd.DataFrame(columns=["source_index", "mean_absolute_change"])
    rows = [
        {
            "source_index": int(source_idx),
            "mean_absolute_change": float(stats["sum"] / max(1.0, stats["count"])),
        }
        for source_idx, stats in sorted(totals.items())
    ]
    return pd.DataFrame(rows)


def compute_feature_family_occlusion(
    runner: ExplainabilityModelBundle,
    *,
    head: str,
    quantity: str,
    sample_indices: Sequence[int],
    feature_groups: Mapping[str, Any],
    batch_size: int | None = None,
) -> pd.DataFrame:
    """Occlude dynamic feature families one group at a time."""
    totals: dict[str, dict[str, float]] = {}
    saw_temporal_input = False
    for batch_indices in _iter_sample_index_batches(sample_indices, batch_size):
        batch = _batchify_samples(runner, batch_indices)
        input_key = "x_dynamic_sources" if "x_dynamic_sources" in batch else "x_dynamic"
        input_tensor = batch.get(input_key)
        if not torch.is_tensor(input_tensor):
            continue
        saw_temporal_input = True
        feature_names = (
            list(getattr(runner.arrays, "source_feature_names", []) or [])
            if input_key == "x_dynamic_sources"
            else list(getattr(runner.arrays, "dynamic_feature_names", []) or [])
        )
        with torch.inference_mode():
            baseline_output = safe_forward(runner.model, batch)
            baseline_quantity = resolve_quantity_tensor(
                baseline_output,
                batch["y"],
                head=head,
                quantity=quantity,
                transfer_scaler_stats=runner.transfer_scaler_stats,
            ).detach()
            for group_name, columns in dict(feature_groups.get("groups", {})).items():
                indices = [feature_names.index(name) for name in columns if name in feature_names]
                if not indices:
                    continue
                occluded = _clone_tensor(input_tensor)
                assert occluded is not None
                if occluded.ndim == 3:
                    occluded[:, :, indices] = 0.0
                else:
                    occluded[:, :, :, indices] = 0.0
                quantity_tensor = resolve_quantity_tensor(
                    safe_forward(runner.model, _masked_batch(batch, input_key, occluded)),
                    batch["y"],
                    head=head,
                    quantity=quantity,
                    transfer_scaler_stats=runner.transfer_scaler_stats,
                ).detach()
                delta = torch.abs(quantity_tensor - baseline_quantity)
                stats = totals.setdefault(str(group_name), {"sum": 0.0, "count": 0.0})
                stats["sum"] += float(delta.sum().cpu())
                stats["count"] += float(delta.numel())
    if not saw_temporal_input:
        return pd.DataFrame(columns=["group", "mean_absolute_change"])
    rows = [
        {
            "group": group_name,
            "mean_absolute_change": float(stats["sum"] / max(1.0, stats["count"])),
        }
        for group_name, stats in totals.items()
    ]
    return (
        pd.DataFrame(rows)
        .sort_values("mean_absolute_change", ascending=False)
        .reset_index(drop=True)
    )


def _feature_group_indices(
    feature_names: Sequence[str],
    feature_groups: Mapping[str, Any],
) -> dict[str, list[int]]:
    feature_list = [str(name) for name in feature_names]
    indices: dict[str, list[int]] = {}
    for group_name, columns in dict(feature_groups.get("groups", {})).items():
        idx = [feature_list.index(name) for name in columns if name in feature_list]
        if idx:
            indices[str(group_name)] = idx
    return indices


def _shares_from_values(values: pd.Series) -> pd.Series:
    total = float(values.sum())
    if not np.isfinite(total) or total <= 0.0:
        return pd.Series(np.zeros(len(values), dtype=float), index=values.index)
    return values / total


def compute_grouped_temporal_integrated_gradients_per_sample(
    runner: ExplainabilityModelBundle,
    *,
    head: str,
    quantity: str,
    sample_indices: Sequence[int],
    feature_groups: Mapping[str, Any],
    steps: int = 16,
) -> pd.DataFrame:
    """Aggregate absolute temporal integrated gradients by feature family for each sample."""
    payload = compute_temporal_integrated_gradients(
        runner,
        head=head,
        quantity=quantity,
        sample_indices=sample_indices,
        steps=steps,
    )
    attributions = payload["attributions"]
    if torch.is_tensor(attributions):
        attr = attributions.detach().cpu().numpy()
    else:
        attr = np.asarray(attributions, dtype=float)
    feature_names = [str(name) for name in payload.get("feature_names", [])]
    group_indices = _feature_group_indices(feature_names, feature_groups)
    rows: list[dict[str, Any]] = []
    if attr.ndim not in {3, 4}:
        raise ValueError(f"Expected 3D or 4D attribution tensor, got shape {tuple(attr.shape)}")
    for sample_pos, sample_index in enumerate(sample_indices):
        sample_attr = np.abs(attr[sample_pos])
        totals: dict[str, float] = {}
        for group_name, indices in group_indices.items():
            if sample_attr.ndim == 3:
                value = float(sample_attr[:, :, indices].sum())
            else:
                value = float(sample_attr[:, indices].sum())
            totals[group_name] = value
        share_series = _shares_from_values(pd.Series(totals, dtype=float))
        for group_name, absolute_value in totals.items():
            rows.append(
                {
                    "sample_index": int(sample_index),
                    "head": str(head),
                    "group": str(group_name),
                    "absolute_importance": float(absolute_value),
                    "share": float(share_series.get(group_name, 0.0)),
                    "quantity": str(quantity),
                    "method": "integrated_gradients",
                }
            )
    return pd.DataFrame(rows)


def compute_grouped_feature_family_occlusion_per_sample(
    runner: ExplainabilityModelBundle,
    *,
    head: str,
    quantity: str,
    sample_indices: Sequence[int],
    feature_groups: Mapping[str, Any],
) -> pd.DataFrame:
    """Aggregate per-sample temporal feature-family occlusion deltas."""
    batch = _batchify_samples(runner, sample_indices)
    input_key = "x_dynamic_sources" if "x_dynamic_sources" in batch else "x_dynamic"
    input_tensor = batch.get(input_key)
    if not torch.is_tensor(input_tensor):
        return pd.DataFrame(
            columns=[
                "sample_index",
                "head",
                "group",
                "absolute_change",
                "share",
                "quantity",
                "method",
            ]
        )
    feature_names = (
        list(getattr(runner.arrays, "source_feature_names", []) or [])
        if input_key == "x_dynamic_sources"
        else list(getattr(runner.arrays, "dynamic_feature_names", []) or [])
    )
    group_indices = _feature_group_indices(feature_names, feature_groups)
    baseline_output = safe_forward(runner.model, batch)
    baseline_quantity = resolve_quantity_tensor(
        baseline_output,
        batch["y"],
        head=head,
        quantity=quantity,
        transfer_scaler_stats=runner.transfer_scaler_stats,
    ).detach()
    per_group_changes: dict[str, np.ndarray] = {}
    for group_name, indices in group_indices.items():
        occluded = _clone_tensor(input_tensor)
        assert occluded is not None
        if occluded.ndim == 3:
            occluded[:, :, indices] = 0.0
        else:
            occluded[:, :, :, indices] = 0.0
        quantity_tensor = resolve_quantity_tensor(
            safe_forward(runner.model, _masked_batch(batch, input_key, occluded)),
            batch["y"],
            head=head,
            quantity=quantity,
            transfer_scaler_stats=runner.transfer_scaler_stats,
        )
        per_group_changes[group_name] = (
            torch.abs(quantity_tensor - baseline_quantity).detach().cpu().numpy()
        )

    rows: list[dict[str, Any]] = []
    for sample_pos, sample_index in enumerate(sample_indices):
        totals = {
            group_name: float(values[sample_pos])
            for group_name, values in per_group_changes.items()
        }
        share_series = _shares_from_values(pd.Series(totals, dtype=float))
        for group_name, absolute_change in totals.items():
            rows.append(
                {
                    "sample_index": int(sample_index),
                    "head": str(head),
                    "group": str(group_name),
                    "absolute_change": float(absolute_change),
                    "share": float(share_series.get(group_name, 0.0)),
                    "quantity": str(quantity),
                    "method": "feature_family_occlusion",
                }
            )
    return pd.DataFrame(rows)


def join_prediction_static_and_attribution_summaries(
    bundle: ResultsBundle,
    sampled_prediction_frame: pd.DataFrame,
    *,
    grouped_ig: pd.DataFrame | None = None,
    grouped_occlusion: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Attach static features and grouped attribution summaries to sampled predictions."""
    enriched = _attach_static_features_to_predictions(bundle, sampled_prediction_frame.copy())
    enriched = enriched.copy()
    enriched["sample_index"] = enriched.index.astype(int)
    result = enriched.reset_index(drop=True)

    def _pivot(summary_df: pd.DataFrame, value_col: str, prefix: str) -> pd.DataFrame:
        if summary_df is None or summary_df.empty:
            return pd.DataFrame(columns=["sample_index"])
        pivot = summary_df.pivot_table(
            index="sample_index",
            columns=["head", "group"],
            values=value_col,
            aggfunc="first",
        )
        pivot.columns = [f"{prefix}_{head}_{group}" for head, group in pivot.columns]
        return pivot.reset_index()

    merged = result
    for payload_df, value_col, prefix in (
        (grouped_ig, "share", "ig_share"),
        (grouped_occlusion, "share", "occ_share"),
    ):
        wide = _pivot(payload_df, value_col=value_col, prefix=prefix)
        if wide.empty:
            continue
        merged = merged.merge(wide, on=["sample_index"], how="left")
    return merged


def compute_proxy_regime_scores_and_labels(
    frame: pd.DataFrame,
    *,
    primary_head: str = "hs",
    dominance_margin: float = 0.10,
) -> pd.DataFrame:
    """Compute attribution-first swell vs local-wind-sea regime labels from grouped shares."""
    head_value = str(primary_head)
    head_frame = frame.copy()
    required_columns = [
        f"ig_share_{head_value}_swell_proxy",
        f"ig_share_{head_value}_windsea_proxy",
        f"ig_share_{head_value}_local_wind_proxy",
        f"occ_share_{head_value}_swell_proxy",
        f"occ_share_{head_value}_windsea_proxy",
        f"occ_share_{head_value}_local_wind_proxy",
    ]
    for column in required_columns:
        if column not in head_frame.columns:
            head_frame[column] = 0.0
    head_frame["local_windsea_score"] = 0.5 * (
        head_frame[f"ig_share_{head_value}_windsea_proxy"].fillna(0.0)
        + head_frame[f"ig_share_{head_value}_local_wind_proxy"].fillna(0.0)
    ) + 0.5 * (
        head_frame[f"occ_share_{head_value}_windsea_proxy"].fillna(0.0)
        + head_frame[f"occ_share_{head_value}_local_wind_proxy"].fillna(0.0)
    )
    head_frame["swell_score"] = 0.5 * head_frame[f"ig_share_{head_value}_swell_proxy"].fillna(
        0.0
    ) + 0.5 * head_frame[f"occ_share_{head_value}_swell_proxy"].fillna(0.0)
    head_frame["dominance_delta"] = head_frame["local_windsea_score"] - head_frame["swell_score"]
    head_frame["dominant_regime"] = np.where(
        head_frame["dominance_delta"] > float(dominance_margin),
        "local_windsea",
        np.where(head_frame["dominance_delta"] < -float(dominance_margin), "swell", "mixed"),
    )
    head_frame["primary_head"] = head_value
    return head_frame


def summarize_site_regime_dominance(
    sample_frame: pd.DataFrame,
    *,
    site_col: str = "site",
    regime_col: str = "dominant_regime",
) -> pd.DataFrame:
    """Summarize sample-level regime labels into site-level regime shares and dominant class."""
    if sample_frame.empty:
        return pd.DataFrame(
            columns=[
                site_col,
                "sample_count",
                "swell_fraction",
                "local_windsea_fraction",
                "mixed_fraction",
                "mean_swell_score",
                "mean_local_windsea_score",
                "site_dominant_regime",
            ]
        )
    counts = (
        sample_frame.groupby([site_col, regime_col], dropna=False)
        .size()
        .rename("count")
        .reset_index()
    )
    pivot = counts.pivot(index=site_col, columns=regime_col, values="count").fillna(0.0)
    for column in ("swell", "local_windsea", "mixed"):
        if column not in pivot.columns:
            pivot[column] = 0.0
    pivot = pivot.reset_index()
    pivot["sample_count"] = pivot[["swell", "local_windsea", "mixed"]].sum(axis=1)
    for column in ("swell", "local_windsea", "mixed"):
        pivot[f"{column}_fraction"] = np.where(
            pivot["sample_count"] > 0,
            pivot[column] / pivot["sample_count"],
            0.0,
        )
    score_summary = (
        sample_frame.groupby(site_col, dropna=False)
        .agg(
            mean_swell_score=("swell_score", "mean"),
            mean_local_windsea_score=("local_windsea_score", "mean"),
        )
        .reset_index()
    )
    out = pivot.merge(score_summary, on=site_col, how="left")
    out["site_dominant_regime"] = np.select(
        [
            out["local_windsea_fraction"] > out["swell_fraction"],
            out["swell_fraction"] > out["local_windsea_fraction"],
        ],
        ["local_windsea", "swell"],
        default="mixed",
    )
    ordered_cols = [
        site_col,
        "sample_count",
        "swell_fraction",
        "local_windsea_fraction",
        "mixed_fraction",
        "mean_swell_score",
        "mean_local_windsea_score",
        "site_dominant_regime",
    ]
    for column in sample_frame.columns:
        if column in ordered_cols or column == regime_col:
            continue
        if column not in out.columns and column in {
            "fjordness_score",
            "open_sector_fraction",
            "closed_sector_fraction",
        }:
            summary = sample_frame.groupby(site_col, dropna=False)[column].mean().reset_index()
            out = out.merge(summary, on=site_col, how="left")
    extra_cols = [col for col in out.columns if col not in ordered_cols]
    return (
        out[ordered_cols + extra_cols]
        .sort_values(["site_dominant_regime", site_col])
        .reset_index(drop=True)
    )


def extract_cross_attention_maps(
    runner: ExplainabilityModelBundle,
    *,
    sample_indices: Sequence[int],
    batch_size: int | None = None,
) -> dict[str, Any]:
    """Extract cross-attention maps and context-token diagnostics."""
    attention_chunks: list[torch.Tensor] = []
    source_attention_chunks: list[torch.Tensor] = []
    context_tokens_chunks: list[torch.Tensor] = []
    dynamic_tokens_chunks: list[torch.Tensor] = []
    task_tokens_chunks: list[torch.Tensor] = []
    static_token_chunks: list[torch.Tensor] = []
    bathy_tokens_chunks: list[torch.Tensor] = []
    bathy_summary_chunks: list[torch.Tensor] = []
    context_token_types = None

    for batch_indices in _iter_sample_index_batches(sample_indices, batch_size):
        batch = _batchify_samples(runner, batch_indices)
        with torch.inference_mode():
            output = safe_forward(
                runner.model, batch, return_attention=True, return_diagnostics=True
            )
        attention = output.get("cross_attention_weights")
        if attention is None:
            decoder_type = str(getattr(runner.model, "decoder_type", "unknown"))
            raise RuntimeError(
                "Model did not return cross_attention_weights because decoder.type="
                f"{decoder_type}. Cross-attention maps are available only when decoder.type=cross_attention."
            )
        attention_chunks.append(attention.detach().cpu())
        source_attention = output.get("source_attention_weights")
        if torch.is_tensor(source_attention):
            source_attention_chunks.append(source_attention.detach().cpu())
        diagnostics = output.get("diagnostics") or {}
        if context_token_types is None:
            context_token_types = diagnostics.get("context_token_types")
        for key, chunks in (
            ("context_tokens", context_tokens_chunks),
            ("dynamic_tokens", dynamic_tokens_chunks),
            ("task_tokens", task_tokens_chunks),
            ("static_token", static_token_chunks),
            ("bathy_tokens", bathy_tokens_chunks),
            ("bathy_summary", bathy_summary_chunks),
        ):
            value = diagnostics.get(key)
            if torch.is_tensor(value):
                chunks.append(value.detach().cpu())

    def _concat_or_none(chunks: list[torch.Tensor]) -> torch.Tensor | None:
        if not chunks:
            return None
        return torch.cat(chunks, dim=0) if len(chunks) > 1 else chunks[0]

    return {
        "cross_attention_weights": _concat_or_none(attention_chunks),
        "source_attention_weights": _concat_or_none(source_attention_chunks),
        "context_token_types": context_token_types,
        "context_tokens": _concat_or_none(context_tokens_chunks),
        "dynamic_tokens": _concat_or_none(dynamic_tokens_chunks),
        "task_tokens": _concat_or_none(task_tokens_chunks),
        "static_token": _concat_or_none(static_token_chunks),
        "bathy_tokens": _concat_or_none(bathy_tokens_chunks),
        "bathy_summary": _concat_or_none(bathy_summary_chunks),
        "task_order": list(getattr(runner.model, "task_order", DEFAULT_HEADS)),
    }


def reconstruct_context_token_index_map(
    runner: ExplainabilityModelBundle,
    attention_payload: Mapping[str, Any],
    *,
    sample_indices: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Reconstruct a readable context-token index map from runtime shapes and coarse diagnostics."""
    attention = attention_payload["cross_attention_weights"]
    if torch.is_tensor(attention):
        context_length = int(attention.shape[-1])
    else:
        context_length = int(np.asarray(attention).shape[-1])

    sample_index = int(sample_indices[0]) if sample_indices else 0
    sample = runner.dataset[sample_index]
    branches = resolve_active_branches(runner.bundle)
    dynamic_count = (
        int(sample["x_dynamic"].shape[0])
        if torch.is_tensor(sample.get("x_dynamic"))
        else int(branches.get("sequence_length") or 0)
    )
    static_count = int(
        bool(getattr(runner.model, "use_static_features", False))
        and torch.is_tensor(sample.get("x_static"))
    )
    bathy_count = 0
    if (
        bool(getattr(runner.model, "use_bathymetry", False))
        and getattr(runner.model, "bathy_encoder", None) is not None
    ):
        bathy_count = int(getattr(runner.model.bathy_encoder, "num_tokens", 0))

    rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    context_token_types = [
        str(item) for item in (attention_payload.get("context_token_types") or ())
    ]
    token_map_source = "reconstructed_from_model_and_sample_shapes"
    expected_context_length = dynamic_count + static_count + bathy_count
    num_sources = branches.get("num_sources")

    def _base_row(token_index: int, raw_token_type: str | None = None) -> dict[str, Any]:
        return {
            "token_index": int(token_index),
            "token_type": raw_token_type or "unknown token",
            "raw_token_type": raw_token_type,
            "timestep_index": np.nan,
            "lag": np.nan,
            "source_index": np.nan,
            "token_subindex": np.nan,
            "uncertain": False,
        }

    if context_token_types:
        token_map_source = "reconstructed_from_context_token_types_and_runtime_shapes"
        warnings.append(
            "Token map unknown: model diagnostics expose coarse context_token_types but not a full token index map; "
            "timestep and token-role annotations below are reconstructed."
        )
        if len(context_token_types) != context_length:
            warnings.append(
                f"context_token_types length ({len(context_token_types)}) does not match extracted context length ({context_length})."
            )
        dynamic_seen = 0
        static_seen = 0
        bathy_seen = 0
        source_seen = 0
        summary_seen = 0
        for token_index, raw_token_type in enumerate(context_token_types[:context_length]):
            token_key = str(raw_token_type).strip().lower()
            row = _base_row(token_index, raw_token_type)
            if token_key == "dynamic":
                row["token_type"] = "dynamic timestep"
                row["timestep_index"] = int(dynamic_seen)
                if dynamic_count > 0:
                    row["lag"] = int(max(dynamic_count - 1 - dynamic_seen, 0))
                dynamic_seen += 1
            elif token_key == "static":
                row["token_type"] = "static token"
                row["token_subindex"] = int(static_seen)
                static_seen += 1
            elif token_key == "bathy":
                row["token_type"] = "bathy token"
                row["token_subindex"] = int(bathy_seen)
                bathy_seen += 1
            elif token_key.startswith("source"):
                row["token_type"] = "source token"
                row["token_subindex"] = int(source_seen)
                if num_sources is None or source_seen < int(num_sources):
                    row["source_index"] = int(source_seen)
                source_seen += 1
            elif "summary" in token_key:
                row["token_type"] = "summary token"
                row["token_subindex"] = int(summary_seen)
                summary_seen += 1
            else:
                row["uncertain"] = True
                warnings.append(
                    f"Unrecognized context token type '{raw_token_type}' at token index {token_index}; annotations may be incomplete."
                )
            rows.append(row)
        if dynamic_seen not in {0, dynamic_count}:
            warnings.append(
                f"Observed {dynamic_seen} dynamic tokens in diagnostics, expected {dynamic_count}."
            )
        if static_seen not in {0, static_count}:
            warnings.append(
                f"Observed {static_seen} static tokens in diagnostics, expected {static_count}."
            )
        if bathy_seen not in {0, bathy_count}:
            warnings.append(
                f"Observed {bathy_seen} bathy tokens in diagnostics, expected {bathy_count}."
            )
    else:
        warnings.append(
            "Token map unknown: no context_token_types were returned by the model; reconstructing expected layout from model/config."
        )
        token_index = 0
        for timestep_index in range(dynamic_count):
            row = _base_row(token_index, "dynamic")
            row["token_type"] = "dynamic timestep"
            row["timestep_index"] = int(timestep_index)
            row["lag"] = int(max(dynamic_count - 1 - timestep_index, 0))
            rows.append(row)
            token_index += 1
        if static_count:
            row = _base_row(token_index, "static")
            row["token_type"] = "static token"
            row["token_subindex"] = 0
            rows.append(row)
            token_index += 1
        for bathy_index in range(bathy_count):
            row = _base_row(token_index, "bathy")
            row["token_type"] = "bathy token"
            row["token_subindex"] = int(bathy_index)
            rows.append(row)
            token_index += 1

    if len(rows) < context_length:
        warnings.append(
            f"Reconstructed token map covers {len(rows)} tokens but extracted attention uses {context_length}; appending unknown placeholders."
        )
        for token_index in range(len(rows), context_length):
            row = _base_row(token_index)
            row["uncertain"] = True
            rows.append(row)
    elif len(rows) > context_length:
        warnings.append(
            f"Reconstructed token map produced {len(rows)} tokens but extracted attention uses {context_length}; truncating to match attention."
        )
        rows = rows[:context_length]

    if expected_context_length != context_length:
        warnings.append(
            f"Expected context length from model/config is {expected_context_length}, but extracted attention uses {context_length} tokens."
        )

    token_map = pd.DataFrame(rows)
    return {
        "token_map": token_map,
        "warnings": list(dict.fromkeys(warnings)),
        "token_map_source": token_map_source,
        "expected_context_length": int(expected_context_length),
        "actual_context_length": int(context_length),
    }


def aggregate_attention_by_head(
    attention_payload: Mapping[str, Any],
    *,
    reduce_batch: bool = True,
) -> pd.DataFrame:
    """Aggregate attention to task-head summaries."""
    attention = attention_payload["cross_attention_weights"]
    if torch.is_tensor(attention):
        array = attention.detach().cpu().numpy()
    else:
        array = np.asarray(attention)
    if reduce_batch:
        array = array.mean(axis=0)
    rows = []
    task_order = attention_payload.get("task_order", list(DEFAULT_HEADS))
    for head_idx in range(array.shape[-2]):
        rows.append(
            {
                "task_head": task_order[head_idx]
                if head_idx < len(task_order)
                else f"head_{head_idx}",
                "mean_attention": float(array[..., head_idx, :].mean()),
            }
        )
    return pd.DataFrame(rows)


def compute_branch_ablation_importance(
    runner: ExplainabilityModelBundle,
    *,
    head: str,
    quantity: str,
    sample_indices: Sequence[int],
    batch_size: int | None = None,
) -> pd.DataFrame:
    """Inference-time branch ablation importance without retraining."""
    branch_totals: dict[str, dict[str, float]] = {}
    branch_map = {
        "dynamic": "x_dynamic",
        "dynamic_sources": "x_dynamic_sources",
        "static": "x_static",
        "source_geometry": "source_geometry",
        "bathy": "x_bathy",
    }
    for batch_indices in _iter_sample_index_batches(sample_indices, batch_size):
        batch = _batchify_samples(runner, batch_indices)
        with torch.inference_mode():
            baseline_output = safe_forward(runner.model, batch)
            baseline_quantity = resolve_quantity_tensor(
                baseline_output,
                batch["y"],
                head=head,
                quantity=quantity,
                transfer_scaler_stats=runner.transfer_scaler_stats,
            ).detach()
            for branch_name, tensor_key in branch_map.items():
                tensor = batch.get(tensor_key)
                if not torch.is_tensor(tensor):
                    continue
                ablated = torch.zeros_like(tensor)
                ablated_output = safe_forward(
                    runner.model, _masked_batch(batch, tensor_key, ablated)
                )
                ablated_quantity = resolve_quantity_tensor(
                    ablated_output,
                    batch["y"],
                    head=head,
                    quantity=quantity,
                    transfer_scaler_stats=runner.transfer_scaler_stats,
                ).detach()
                delta = torch.abs(ablated_quantity - baseline_quantity)
                stats = branch_totals.setdefault(branch_name, {"sum": 0.0, "count": 0.0})
                stats["sum"] += float(delta.sum().cpu())
                stats["count"] += float(delta.numel())
    rows = []
    for branch_name, stats in branch_totals.items():
        count = max(1.0, float(stats["count"]))
        rows.append(
            {
                "branch": branch_name,
                "mean_absolute_change": float(stats["sum"] / count),
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values("mean_absolute_change", ascending=False)
        .reset_index(drop=True)
    )


def compute_static_dynamic_interactions(
    prediction_frame: pd.DataFrame,
    *,
    feature_x: str,
    feature_y: str,
    response_col: str,
    bins: int = 5,
) -> pd.DataFrame:
    """Compute a simple 2D binned interaction surface from exported predictions/features."""
    if (
        feature_x not in prediction_frame.columns
        or feature_y not in prediction_frame.columns
        or response_col not in prediction_frame.columns
    ):
        return pd.DataFrame(columns=["x_bin", "y_bin", "response"])
    df = prediction_frame[[feature_x, feature_y, response_col]].dropna().copy()
    if df.empty:
        return df
    df["x_bin"] = pd.qcut(
        df[feature_x].rank(method="first"), q=min(bins, len(df)), duplicates="drop"
    ).astype(str)
    df["y_bin"] = pd.qcut(
        df[feature_y].rank(method="first"), q=min(bins, len(df)), duplicates="drop"
    ).astype(str)
    return (
        df.groupby(["x_bin", "y_bin"], as_index=False)[response_col]
        .mean()
        .rename(columns={response_col: "response"})
    )


def compute_modality_contribution_summary(
    branch_importance: pd.DataFrame,
    *,
    head: str,
    quantity: str,
) -> pd.DataFrame:
    """Convert branch ablations into relative modality contribution shares."""
    df = branch_importance.copy()
    total = float(df["mean_absolute_change"].sum()) if len(df) else 0.0
    df["relative_share"] = 0.0 if total <= 0.0 else df["mean_absolute_change"] / total
    df["head"] = head
    df["quantity"] = quantity
    return df


def compute_bathy_integrated_gradients(
    runner: ExplainabilityModelBundle,
    *,
    head: str,
    quantity: str,
    sample_indices: Sequence[int],
    steps: int = 16,
    batch_size: int | None = None,
) -> dict[str, Any]:
    """Integrated gradients over the bathymetry patch branch."""
    attribution_chunks: list[torch.Tensor] = []
    mean_absolute_map_sum: torch.Tensor | None = None
    mean_absolute_map_count = 0.0
    saw_bathy = False
    for batch_indices in _iter_sample_index_batches(sample_indices, batch_size):
        batch = _batchify_samples(runner, batch_indices)
        if "x_bathy" not in batch:
            continue
        saw_bathy = True
        attr = _integrated_gradients_for_tensor(
            runner,
            batch,
            input_key="x_bathy",
            head=head,
            quantity=quantity,
            steps=steps,
        )
        attr_cpu = attr.detach().cpu()
        attribution_chunks.append(attr_cpu)
        chunk_abs_map = attr_cpu.abs().sum(dim=(0, 1))
        mean_absolute_map_sum = (
            chunk_abs_map
            if mean_absolute_map_sum is None
            else mean_absolute_map_sum + chunk_abs_map
        )
        mean_absolute_map_count += float(attr_cpu.shape[0] * attr_cpu.shape[1])
    if not saw_bathy:
        return {"available": False, "attributions": None, "mean_absolute_map": None}
    attributions = (
        torch.cat(attribution_chunks, dim=0)
        if len(attribution_chunks) > 1
        else attribution_chunks[0]
    )
    mean_absolute_map = None
    if mean_absolute_map_sum is not None and mean_absolute_map_count > 0:
        mean_absolute_map = mean_absolute_map_sum / float(mean_absolute_map_count)
    return {"available": True, "attributions": attributions, "mean_absolute_map": mean_absolute_map}


def compute_bathy_occlusion_maps(
    runner: ExplainabilityModelBundle,
    *,
    head: str,
    quantity: str,
    sample_indices: Sequence[int],
    patch_stride: int = 8,
    batch_size: int | None = None,
) -> dict[str, Any]:
    """Occlude bathymetry patches in coarse windows."""
    occlusion_sum_map: torch.Tensor | None = None
    total_samples = 0
    saw_bathy = False
    for batch_indices in _iter_sample_index_batches(sample_indices, batch_size):
        batch = _batchify_samples(runner, batch_indices)
        bathy = batch.get("x_bathy")
        if not torch.is_tensor(bathy):
            continue
        saw_bathy = True
        with torch.inference_mode():
            baseline_output = safe_forward(runner.model, batch)
            baseline_quantity = resolve_quantity_tensor(
                baseline_output,
                batch["y"],
                head=head,
                quantity=quantity,
                transfer_scaler_stats=runner.transfer_scaler_stats,
            ).detach()
            batch_map = torch.zeros(
                (bathy.size(-2), bathy.size(-1)), device=bathy.device, dtype=bathy.dtype
            )
            for row in range(0, bathy.size(-2), patch_stride):
                for col in range(0, bathy.size(-1), patch_stride):
                    occluded = _clone_tensor(bathy)
                    assert occluded is not None
                    occluded[:, :, row : row + patch_stride, col : col + patch_stride] = 0.0
                    quantity_tensor = resolve_quantity_tensor(
                        safe_forward(runner.model, _masked_batch(batch, "x_bathy", occluded)),
                        batch["y"],
                        head=head,
                        quantity=quantity,
                        transfer_scaler_stats=runner.transfer_scaler_stats,
                    ).detach()
                    delta = torch.mean(torch.abs(quantity_tensor - baseline_quantity))
                    batch_map[row : row + patch_stride, col : col + patch_stride] = delta
        batch_weight = len(batch_indices)
        batch_map_cpu = batch_map.detach().cpu()
        occlusion_sum_map = (
            batch_map_cpu * batch_weight
            if occlusion_sum_map is None
            else occlusion_sum_map + (batch_map_cpu * batch_weight)
        )
        total_samples += batch_weight
    if not saw_bathy:
        return {"available": False, "map": None}
    assert occlusion_sum_map is not None
    return {"available": True, "map": occlusion_sum_map / max(1, total_samples)}


def compute_error_attributions(
    runner: ExplainabilityModelBundle,
    *,
    head: str,
    sample_indices: Sequence[int],
    feature_groups: Mapping[str, Any],
) -> dict[str, Any]:
    """Convenience wrapper for error-focused explainability."""
    shap_payload = compute_context_averaged_static_shap(
        runner,
        head=head,
        quantity="absolute_error",
        sample_indices=sample_indices,
        background_indices=sample_indices[: min(len(sample_indices), 32)],
        cache_key=f"error_static_shap_{runner.split}_{head}",
    )
    return {
        "static_grouped": compute_static_grouped_shap(shap_payload, feature_groups),
        "dynamic_occlusion": compute_temporal_occlusion(
            runner,
            head=head,
            quantity="absolute_error",
            sample_indices=sample_indices,
        ),
    }


def rank_failure_sites(
    prediction_frame: pd.DataFrame,
    *,
    high_hs_threshold: float = 1.5,
) -> pd.DataFrame:
    """Rank sites by configurable failure severity metrics."""
    if "site" not in prediction_frame.columns:
        return pd.DataFrame()
    rows = []
    for site, site_df in prediction_frame.groupby("site"):
        hs_rmse = (
            float(np.sqrt(np.mean(np.square(site_df["pred_hs"] - site_df["target_hs"]))))
            if {"pred_hs", "target_hs"} <= set(site_df.columns)
            else np.nan
        )
        hs_bias = (
            float(np.mean(site_df["pred_hs"] - site_df["target_hs"]))
            if {"pred_hs", "target_hs"} <= set(site_df.columns)
            else np.nan
        )
        high_mask = (
            site_df["target_hs"] >= float(high_hs_threshold)
            if "target_hs" in site_df.columns
            else pd.Series(False, index=site_df.index)
        )
        high_hs_bias = (
            float(
                np.mean((site_df.loc[high_mask, "pred_hs"] - site_df.loc[high_mask, "target_hs"]))
            )
            if high_mask.any()
            else np.nan
        )
        dir_rmse = (
            float(
                np.sqrt(
                    np.mean(
                        np.square(
                            circular_error_deg(site_df["target_dir_deg"], site_df["pred_dir_deg"])
                        )
                    )
                )
            )
            if {"target_dir_deg", "pred_dir_deg"} <= set(site_df.columns)
            else np.nan
        )
        dp_rmse = (
            float(
                np.sqrt(
                    np.mean(
                        np.square(
                            circular_error_deg(site_df["target_dp_deg"], site_df["pred_dp_deg"])
                        )
                    )
                )
            )
            if {"target_dp_deg", "pred_dp_deg"} <= set(site_df.columns)
            else np.nan
        )
        rows.append(
            {
                "site": site,
                "sample_count": int(len(site_df)),
                "hs_rmse": hs_rmse,
                "hs_bias": hs_bias,
                "high_hs_bias": high_hs_bias,
                "dir_rmse_deg": dir_rmse,
                "dp_rmse_deg": dp_rmse,
            }
        )
    ranked = pd.DataFrame(rows)
    if ranked.empty:
        return ranked
    ranked["severity_score"] = (
        ranked[["hs_rmse", "dir_rmse_deg", "dp_rmse_deg"]].fillna(0.0).sum(axis=1)
    )
    ranked["failure_type"] = np.where(
        ranked["high_hs_bias"] < -0.1,
        "amplitude underprediction",
        np.where(ranked["hs_bias"] > 0.1, "amplitude overprediction", "mixed / other"),
    )
    return ranked.sort_values("severity_score", ascending=False).reset_index(drop=True)


def compare_good_vs_bad_samples(
    values: pd.DataFrame,
    *,
    score_col: str,
    group_col: str,
    value_col: str,
    good_quantile: float = 0.25,
    bad_quantile: float = 0.75,
) -> pd.DataFrame:
    """Compare grouped attribution values for good and bad samples."""
    if values.empty or score_col not in values.columns:
        return pd.DataFrame(columns=[group_col, "good_mean", "bad_mean", "delta"])
    good_cut = values[score_col].quantile(float(good_quantile))
    bad_cut = values[score_col].quantile(float(bad_quantile))
    good = values.loc[values[score_col] <= good_cut].groupby(group_col)[value_col].mean()
    bad = values.loc[values[score_col] >= bad_cut].groupby(group_col)[value_col].mean()
    merged = (
        pd.concat([good.rename("good_mean"), bad.rename("bad_mean")], axis=1)
        .fillna(0.0)
        .reset_index()
    )
    merged["delta"] = merged["bad_mean"] - merged["good_mean"]
    return merged.sort_values("delta", ascending=False).reset_index(drop=True)


def compare_good_vs_bad_sites(
    ranked_sites: pd.DataFrame,
    *,
    value_cols: Sequence[str] = ("hs_rmse", "hs_bias", "dir_rmse_deg", "dp_rmse_deg"),
    top_n: int = 5,
) -> pd.DataFrame:
    """Compare mean metrics between the best and worst ranked sites."""
    if ranked_sites.empty:
        return pd.DataFrame(columns=["metric", "good_mean", "bad_mean", "delta"])
    good = ranked_sites.tail(min(top_n, len(ranked_sites)))
    bad = ranked_sites.head(min(top_n, len(ranked_sites)))
    rows = []
    for metric in value_cols:
        if metric not in ranked_sites.columns:
            continue
        good_mean = float(good[metric].mean())
        bad_mean = float(bad[metric].mean())
        rows.append(
            {
                "metric": metric,
                "good_mean": good_mean,
                "bad_mean": bad_mean,
                "delta": bad_mean - good_mean,
            }
        )
    return pd.DataFrame(rows).sort_values("delta", ascending=False).reset_index(drop=True)


def explain_single_site(
    prediction_frame: pd.DataFrame,
    ranked_sites: pd.DataFrame,
    *,
    point_name: str,
) -> dict[str, Any]:
    """Generate a concise rule-based diagnosis for one site."""
    site_df = prediction_frame.loc[prediction_frame["site"].astype(str) == str(point_name)].copy()
    if site_df.empty:
        return {"site": point_name, "summary": "Site not found in the selected prediction frame."}
    row = ranked_sites.loc[ranked_sites["site"].astype(str) == str(point_name)]
    severity = row.iloc[0].to_dict() if len(row) else {}
    primary_failure = severity.get("failure_type", "mixed / other")
    evidence = []
    if "hs_bias" in severity and np.isfinite(severity["hs_bias"]):
        evidence.append(f"Hs bias = {severity['hs_bias']:.3f}")
    if "dir_rmse_deg" in severity and np.isfinite(severity["dir_rmse_deg"]):
        evidence.append(f"Dir RMSE = {severity['dir_rmse_deg']:.1f} deg")
    if "dp_rmse_deg" in severity and np.isfinite(severity["dp_rmse_deg"]):
        evidence.append(f"Dp RMSE = {severity['dp_rmse_deg']:.1f} deg")
    summary = (
        f"{point_name}: Primary failure: {primary_failure}. "
        f"Evidence: {'; '.join(evidence) if evidence else 'insufficient exported metrics'}."
    )
    return {
        "site": point_name,
        "summary": summary,
        "metrics": severity,
        "sample_count": int(len(site_df)),
    }


def _write_note(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _printable_run_summary(bundle: ResultsBundle, split: str, sample_count: int) -> pd.DataFrame:
    branches = resolve_active_branches(bundle)
    features = load_feature_metadata(bundle)
    rows = [
        {"field": "results_dir", "value": str(bundle.results_dir)},
        {"field": "checkpoint_loaded", "value": str(bundle.checkpoint_path)},
        {
            "field": "model_architecture",
            "value": str(((bundle.config.get("model", {}) or {}).get("architecture", "unknown"))),
        },
        {
            "field": "active_branches",
            "value": ", ".join(
                [
                    name
                    for name in ("static", "multi_source", "source_geometry", "bathy")
                    if branches.get(name)
                ]
            )
            or "dynamic_only",
        },
        {
            "field": "dynamic_mode",
            "value": "multi_source" if branches.get("multi_source") else "single_source",
        },
        {"field": "sequence_length", "value": branches.get("sequence_length")},
        {"field": "number_of_sources", "value": branches.get("num_sources")},
        {
            "field": "available_dynamic_feature_names",
            "value": ", ".join(features.get("dynamic_feature_names", [])[:12]),
        },
        {
            "field": "available_static_feature_names",
            "value": ", ".join(features.get("static_feature_names", [])[:12]),
        },
        {"field": "bathy_enabled", "value": branches.get("bathy")},
        {"field": "source_geometry_enabled", "value": branches.get("source_geometry")},
        {"field": "active_target_heads", "value": ", ".join(branches.get("target_heads", []))},
        {"field": "split_analyzed", "value": split},
        {"field": "number_of_samples_used", "value": sample_count},
        {
            "field": "ablation_status",
            "value": str(getattr(_load_arrays(bundle), "ablation_summary", None)),
        },
    ]
    return pd.DataFrame(rows)


def _attach_static_features_to_predictions(
    bundle: ResultsBundle, prediction_frame: pd.DataFrame
) -> pd.DataFrame:
    if bundle.point_centric_dir is None:
        return prediction_frame
    static_csv = (
        ((bundle.config.get("data", {}) or {}).get("static_features_csv"))
        if bundle.config
        else None
    )
    resolved_csv = (
        _resolve_path(static_csv, anchor=bundle.config_path or bundle.results_dir)
        if static_csv
        else None
    )
    if resolved_csv is None or not resolved_csv.exists():
        return prediction_frame
    static_df = pd.read_csv(resolved_csv)
    join_key = (
        "site"
        if "site" in static_df.columns
        else (
            "site_name"
            if "site_name" in static_df.columns
            else ("name" if "name" in static_df.columns else None)
        )
    )
    if join_key is None or "site" not in prediction_frame.columns:
        return prediction_frame
    if join_key != "site":
        static_df = static_df.rename(columns={join_key: "site"})
    return prediction_frame.merge(static_df, on="site", how="left", suffixes=("", "_static"))


def _compute_group_order(grouped_outputs: Mapping[str, dict[str, pd.DataFrame]]) -> list[str]:
    rows = []
    for head, payload in grouped_outputs.items():
        summary = payload.get("summary", pd.DataFrame())
        if summary.empty:
            continue
        subset = summary[["group", "mean_abs_value"]].copy()
        subset["target"] = str(head)
        rows.append(subset)
    if not rows:
        return []
    combined = pd.concat(rows, ignore_index=True)
    ranking = (
        combined.groupby("group", as_index=False)["mean_abs_value"]
        .mean()
        .sort_values("mean_abs_value", ascending=False)
    )
    return ranking["group"].astype(str).tolist()


def _build_grouped_shap_importance_table(
    grouped_outputs: Mapping[str, dict[str, pd.DataFrame]],
    *,
    group_order: Sequence[str],
) -> pd.DataFrame:
    rows = []
    for head, payload in grouped_outputs.items():
        summary = payload.get("summary", pd.DataFrame()).copy()
        if summary.empty:
            continue
        total = float(summary["mean_abs_value"].sum())
        summary["target"] = str(head)
        summary["feature_group"] = summary["group"].astype(str)
        summary["normalized_mean_abs_shap"] = (
            summary["mean_abs_value"] / total if total > 0 else 0.0
        )
        summary = summary.sort_values("mean_abs_value", ascending=False).reset_index(drop=True)
        summary["rank_within_target"] = np.arange(1, len(summary) + 1, dtype=int)
        rows.append(
            summary[
                [
                    "target",
                    "feature_group",
                    "mean_abs_value",
                    "normalized_mean_abs_shap",
                    "rank_within_target",
                ]
            ]
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "target",
                "feature_group",
                "mean_abs_shap",
                "normalized_mean_abs_shap",
                "rank_within_target",
            ]
        )
    table = pd.concat(rows, ignore_index=True).rename(columns={"mean_abs_value": "mean_abs_shap"})
    if group_order:
        table["feature_group"] = pd.Categorical(
            table["feature_group"], categories=list(group_order), ordered=True
        )
        table = table.sort_values(["feature_group", "target"]).reset_index(drop=True)
        table["feature_group"] = table["feature_group"].astype(str)
    return table


def _static_feature_distribution_summary(
    sampled_frame: pd.DataFrame,
    sample_indices: Sequence[int],
    *,
    target: str,
    feature_name: str,
    feature_label: str,
    physical_units_used: bool,
    normalization_status: str,
) -> dict[str, Any] | None:
    if feature_name not in sampled_frame.columns:
        return None
    series = pd.to_numeric(
        sampled_frame.loc[list(sample_indices), feature_name], errors="coerce"
    ).dropna()
    if series.empty:
        return None
    values = series.to_numpy(dtype=float)
    return {
        "target": str(target),
        "feature": str(feature_name),
        "feature_label": str(feature_label),
        "minimum": float(np.min(values)),
        "percentile_1": float(np.percentile(values, 1)),
        "percentile_5": float(np.percentile(values, 5)),
        "median": float(np.median(values)),
        "percentile_95": float(np.percentile(values, 95)),
        "percentile_99": float(np.percentile(values, 99)),
        "maximum": float(np.max(values)),
        "valid_sample_count": int(values.size),
        "physical_units_used": bool(physical_units_used),
        "normalization_status": str(normalization_status),
    }


def run_static_explainability_analysis(
    *,
    results_dir: str | Path,
    split: str = "val",
    device: str = "cuda",
    max_explain_samples: int = 512,
    background_samples: int = 128,
    random_seed: int = 42,
    targets_to_analyze: Sequence[str] = DEFAULT_HEADS,
    explain_quantity: str = "physical_prediction",
    use_stratified_sampling: bool = True,
    sampling_strata: Sequence[str] = ("site", "target_hs_bin", "target_tp_bin"),
    static_group_overrides: Mapping[str, Sequence[str]] | None = None,
    training_config_path: str | Path | None = None,
    point_centric_dir: str | Path | None = None,
    ale_feature_selections: Mapping[str, Sequence[str]] | None = None,
    ale_bins: int = 6,
    ale_min_bin_samples: int = 10,
    ale_percentile_range: tuple[float, float] = (1.0, 99.0),
    ale_bootstrap_repeats: int = 500,
    ale_site_set: str = "test",
    ale_samples_per_site: int = 32,
    ale_min_unique_sites_per_bin: int = 5,
    ale_random_seed: int | None = None,
    static_shap_display_percentile_range: tuple[float, float] | None = None,
    sitewise_k_nearest_analogues: int = 5,
) -> dict[str, Any]:
    runner = load_model_for_explainability(
        results_dir,
        split=split,
        device=device,
        training_config_path=training_config_path,
        point_centric_dir=point_centric_dir,
    )
    prediction_frame = _attach_static_features_to_predictions(
        runner.bundle, load_prediction_frame(runner.bundle, split=split)
    )
    sampled = sample_explainability_subset(
        prediction_frame,
        max_samples=max_explain_samples,
        random_seed=random_seed,
        use_stratified_sampling=use_stratified_sampling,
        strata=sampling_strata,
    )
    background = sample_explainability_subset(
        prediction_frame,
        max_samples=background_samples,
        random_seed=random_seed + 1,
        use_stratified_sampling=use_stratified_sampling,
        strata=sampling_strata,
    )
    output_dirs = _resolve_static_output_dirs(runner.bundle)
    run_dir = output_dirs["root"]
    summary_df = _printable_run_summary(runner.bundle, split, len(sampled))
    summary_df.to_csv(output_dirs["tables"] / "run_summary.csv", index=False)

    features = load_feature_metadata(runner.bundle)
    feature_groups = build_feature_groups(
        features.get("static_feature_names", []), overrides=static_group_overrides
    )
    pd.DataFrame({"unmatched_feature": feature_groups["unmatched"]}).to_csv(
        output_dirs["tables"] / "unmatched_static_features.csv", index=False
    )
    rows = []
    for group_name, feature_names in feature_groups["groups"].items():
        for feature_name in feature_names:
            rows.append(
                {
                    "group": str(group_name),
                    "group_display_name": _group_display_name(str(group_name)),
                    "feature": str(feature_name),
                    "feature_display_name": _feature_display_name(
                        str(feature_name), include_units=True
                    ),
                    "matched_pattern": str(feature_name) not in feature_groups["unmatched"],
                }
            )
    pd.DataFrame(rows).sort_values(["group", "feature"]).reset_index(drop=True).to_csv(
        output_dirs["tables"] / "static_feature_groups.csv",
        index=False,
    )

    outputs: dict[str, Any] = {
        "run_summary": summary_df,
        "feature_groups": feature_groups,
        "output_dirs": output_dirs,
        "warnings": [],
    }
    if not resolve_active_branches(runner.bundle).get("static", False):
        _write_note(
            run_dir / "README.txt",
            "Static branch disabled in this run. Static SHAP/ALE sections were skipped.",
        )
        return outputs

    sample_indices = sampled.index.to_list()
    background_indices = background.index.to_list()
    group_name_map = {
        group: _group_display_name(group) for group in feature_groups.get("groups", {})
    }
    grouped_outputs: dict[str, dict[str, pd.DataFrame]] = {}
    shap_figure_paths: list[str] = []
    ale_figure_paths: list[str] = []
    primary_shap_paths: list[str] = []
    exported_csv_paths: list[str] = [
        str(output_dirs["tables"] / "run_summary.csv"),
        str(output_dirs["tables"] / "static_feature_groups.csv"),
        str(output_dirs["tables"] / "unmatched_static_features.csv"),
    ]
    skipped_ale_features: list[dict[str, Any]] = []

    for head in targets_to_analyze:
        shap_payload = compute_context_averaged_static_shap(
            runner,
            head=head,
            quantity=explain_quantity,
            sample_indices=sample_indices,
            background_indices=background_indices,
            cache_key=f"static_shap_{split}_{head}_{explain_quantity}",
        )
        grouped = compute_static_grouped_shap(shap_payload, feature_groups)
        perm = compute_group_permutation_importance(
            runner,
            head=head,
            quantity=explain_quantity,
            sample_indices=sample_indices,
            feature_groups=feature_groups,
            group_type="static",
            random_seed=random_seed,
        )
        grouped["summary"]["target"] = str(head)
        grouped["summary"]["group_display_name"] = (
            grouped["summary"]["group"].map(group_name_map).fillna(grouped["summary"]["group"])
        )
        grouped["summary"]["calculation"] = (
            "Grouped SHAP values sum signed feature-level SHAP contributions within each static feature group for each sample."
        )
        grouped["beeswarm"]["target"] = str(head)
        grouped["beeswarm"]["group_display_name"] = (
            grouped["beeswarm"]["group"].map(group_name_map).fillna(grouped["beeswarm"]["group"])
        )
        grouped_outputs[str(head)] = {
            "summary": grouped["summary"].copy(),
        }

        extracted = extract_static_shap_matrix(
            shap_payload,
            head=str(head),
            feature_names=list(getattr(runner.arrays, "static_feature_names", []) or []),
            expected_samples=len(sample_indices),
        )
        shap_sample_indices = extracted["sample_indices"] or [int(idx) for idx in sample_indices]
        if extracted["sample_indices"] and shap_sample_indices != [
            int(idx) for idx in sample_indices
        ]:
            raise ValueError(
                f"Cached SHAP sample order for target '{head}' does not match the current sampled subset. "
                "Delete the cache or rerun with matching sample selection."
            )
        feature_payload = recover_static_feature_values(
            runner,
            sampled_frame=sampled,
            sample_indices=shap_sample_indices,
            feature_names=extracted["feature_names"],
        )
        shap_matrix = np.asarray(extracted["matrix"], dtype=float)
        feature_matrix = np.asarray(feature_payload["values"], dtype=float)
        if shap_matrix.shape != feature_matrix.shape:
            raise ValueError(
                f"SHAP matrix shape {shap_matrix.shape} does not match feature-value matrix shape {feature_matrix.shape} for target '{head}'"
            )
        if shap_matrix.shape[1] != len(extracted["feature_names"]):
            raise ValueError(
                f"Feature name count {len(extracted['feature_names'])} does not match SHAP feature dimension {shap_matrix.shape[1]} for target '{head}'"
            )

        filtered_shap, filtered_features, filtered_base, finite_mask = _drop_non_finite_static_rows(
            shap_matrix,
            feature_matrix,
            base_values=extracted["base_values"],
        )
        if not np.any(finite_mask):
            raise ValueError(
                f"All static SHAP rows became non-finite after validation for target '{head}'"
            )

        top_feature_payload = _select_top_static_shap_features(
            filtered_shap,
            filtered_features,
            extracted["feature_names"],
            max_display=10,
        )
        top_feature_names = list(top_feature_payload["selected_feature_names"])
        top_value_sources = [
            str(feature_payload["value_sources"][idx]) for idx in top_feature_payload["indices"]
        ]
        raw_feature_map = _build_transformed_to_raw_static_map(runner.arrays)
        top10_table, readable_name_map, standardized_top_features = (
            _build_top_static_shap_importance_table(
                head=str(head),
                top_feature_names=top_feature_names,
                top_importance=top_feature_payload["importance"],
                feature_groups=feature_groups,
                value_sources=top_value_sources,
                raw_feature_map=raw_feature_map,
            )
        )
        top10_csv_path = output_dirs["tables"] / f"{head}_static_shap_top10.csv"
        top10_table.to_csv(top10_csv_path, index=False)
        exported_csv_paths.append(str(top10_csv_path))

        top_feature_value_label = _static_feature_value_label(
            top_feature_payload["selected_shap"],
            top_value_sources,
            max_display=10,
        )
        display_shap, display_features, display_base, display_mask, display_filter_meta = (
            _filter_static_shap_display_samples_by_percentile(
                top_feature_payload["selected_shap"],
                top_feature_payload["selected_feature_values"],
                base_values=filtered_base,
                percentile_range=static_shap_display_percentile_range,
            )
        )
        top_beeswarm_paths = plot_top_static_shap_beeswarm(
            display_shap,
            display_features,
            top_feature_names,
            _static_shap_target_display_name(str(head)),
            output_dirs["shap"] / f"{head}_static_shap_top10_beeswarm",
            base_values=display_base,
            max_display=10,
            readable_name_map=readable_name_map,
            x_label="SHAP value",
            title=_static_shap_figure_title(str(head), explain_quantity),
            feature_value_label=top_feature_value_label,
        )
        shap_figure_paths.extend(str(path) for path in top_beeswarm_paths)
        primary_shap_paths.extend(str(path) for path in top_beeswarm_paths)
        if standardized_top_features:
            outputs["warnings"].append(
                f"{head}: top static SHAP beeswarm used standardized model-input values for color mapping where raw values could not be recovered ({', '.join(standardized_top_features[:5])}{'...' if len(standardized_top_features) > 5 else ''})."
            )
        if display_filter_meta is not None:
            outputs["warnings"].append(
                f"{head}: top static SHAP beeswarm display kept {int(display_filter_meta['kept_samples'])} of {int(top_feature_payload['selected_shap'].shape[0])} samples using max-|SHAP| percentiles {display_filter_meta['lower_percentile']:.0f}-{display_filter_meta['upper_percentile']:.0f}."
            )

        group_order_for_target = grouped["summary"]["group"].astype(str).tolist()
        grouped_values = grouped["beeswarm"].copy()
        grouped_values = grouped_values[
            np.isfinite(pd.to_numeric(grouped_values["value"], errors="coerce"))
        ].copy()
        grouped_values["value"] = pd.to_numeric(grouped_values["value"], errors="coerce")
        fig = plot_grouped_shap_distribution(
            grouped_values,
            group_order=group_order_for_target,
            display_name_map=group_name_map,
            title=f"Grouped static SHAP contributions: {_static_shap_target_display_name(str(head))}",
            x_label=f"Grouped SHAP value (impact on predicted {_static_shap_target_display_name(str(head))})",
        )
        fig.tight_layout()
        beeswarm_paths = save_figure_variants(
            fig, output_dirs["shap"] / f"{head}_static_shap_beeswarm", close=True
        )
        shap_figure_paths.extend(str(path) for path in beeswarm_paths)
        primary_shap_paths.extend(str(path) for path in beeswarm_paths)

        grouped["summary"].to_csv(output_dirs["tables"] / f"grouped_shap_{head}.csv", index=False)
        grouped["beeswarm"].to_csv(
            output_dirs["tables"] / f"grouped_shap_distribution_{head}.csv", index=False
        )
        perm.to_csv(output_dirs["tables"] / f"grouped_permutation_{head}.csv", index=False)
        exported_csv_paths.extend(
            [
                str(output_dirs["tables"] / f"grouped_shap_{head}.csv"),
                str(output_dirs["tables"] / f"grouped_shap_distribution_{head}.csv"),
                str(output_dirs["tables"] / f"grouped_permutation_{head}.csv"),
            ]
        )
        fig = plot_grouped_shap_bar(
            grouped["summary"],
            title=f"Grouped static SHAP importance: {_target_display_name(head)}",
        )
        saved = save_figure_variants(
            fig, output_dirs["shap"] / f"grouped_shap_bar_{head}", close=True
        )
        shap_figure_paths.extend(str(path) for path in saved)
        fig = plot_branch_ablation_bars(
            perm.rename(columns={"group": "branch"}),
            x_col="branch",
            y_col="mean_absolute_change",
            title=f"Grouped permutation importance: {_target_display_name(head)}",
        )
        saved = save_figure_variants(
            fig, output_dirs["shap"] / f"grouped_permutation_{head}", close=True
        )
        shap_figure_paths.extend(str(path) for path in saved)
        outputs[head] = {
            "grouped_shap": grouped["summary"],
            "permutation": perm,
            "static_shap_beeswarm_paths": [str(path) for path in beeswarm_paths],
            "static_shap_top10_beeswarm_paths": [str(path) for path in top_beeswarm_paths],
            "static_shap_top10_table_path": str(top10_csv_path),
            "static_shap_valid_sample_count": int(len(sample_indices)),
            "static_shap_top10_display_sample_count": int(display_shap.shape[0]),
            "static_shap_top10_display_percentile_range": static_shap_display_percentile_range,
            "static_shap_feature_value_label": "Grouped SHAP plot; feature-value color scale omitted because groups contain multiple features.",
            "static_shap_top10_feature_value_label": top_feature_value_label,
        }

    group_order = _compute_group_order(grouped_outputs)
    output_dirs["root"].mkdir(parents=True, exist_ok=True)
    combined_shap_paths: list[str] = list(primary_shap_paths)

    importance_table = _build_grouped_shap_importance_table(
        grouped_outputs, group_order=group_order
    )
    importance_csv_path = output_dirs["tables"] / "grouped_static_shap_importance.csv"
    importance_table.to_csv(importance_csv_path, index=False)
    exported_csv_paths.append(str(importance_csv_path))
    if not importance_table.empty and group_order:
        fig = plot_grouped_shap_importance_heatmap(
            importance_table,
            group_order=group_order,
            target_order=[str(head) for head in targets_to_analyze if str(head) in grouped_outputs],
            display_name_map=group_name_map,
            target_display_map={
                str(head): _target_display_name(str(head)) for head in targets_to_analyze
            },
            title="Static grouped SHAP importance summary",
        )
        saved = save_figure_variants(
            fig, output_dirs["shap"] / "grouped_static_shap_importance_summary", close=True
        )
        shap_figure_paths.extend(str(path) for path in saved)

    all_ale_rows: list[pd.DataFrame] = []
    ale_reliability_rows: list[dict[str, Any]] = []
    distribution_rows: list[dict[str, Any]] = []
    selected_ale_paths: list[str] = []
    available_static_features = list(features.get("static_feature_names", []))
    user_selected = {
        str(key): [str(item) for item in value]
        for key, value in dict(ale_feature_selections or {}).items()
    }
    effective_ale_seed = int(random_seed if ale_random_seed is None else ale_random_seed)
    ale_bundle = build_static_ale_analysis_frame(
        runner.bundle,
        device=device,
        training_config_path=training_config_path,
        point_centric_dir=point_centric_dir,
        ale_site_set=str(ale_site_set),
    )
    for warning in ale_bundle.get("warnings", []):
        outputs["warnings"].append(warning)
    ale_analysis_frame = sample_site_balanced_ale_subset(
        ale_bundle["frame"],
        samples_per_site=int(ale_samples_per_site),
        random_seed=effective_ale_seed,
    )
    ale_runners = ale_bundle["runners"]
    outputs["ale_dataset_summary"] = {
        "ale_site_set": str(ale_site_set),
        "resolved_ale_splits": list(ale_bundle.get("splits", [])),
        "selected_rows": int(len(ale_analysis_frame)),
        "selected_unique_sites": int(ale_analysis_frame["site"].nunique())
        if not ale_analysis_frame.empty
        else 0,
        "effective_samples_per_site": int(ale_analysis_frame["effective_samples_per_site"].iloc[0])
        if not ale_analysis_frame.empty
        else 0,
    }
    ale_preview_batch = (
        _batchify_ale_frame_rows(ale_runners, ale_analysis_frame)
        if not ale_analysis_frame.empty
        else None
    )

    for head in targets_to_analyze:
        selected_features = [
            feature
            for feature in user_selected.get(
                str(head),
                _default_selected_static_ale_features(available_static_features, head=str(head)),
            )
            if feature in available_static_features
        ]
        head_ale_frames: list[pd.DataFrame] = []
        target_std = _ale_target_standard_deviation(ale_analysis_frame, head=str(head))

        for feature_name in selected_features:
            primary_ale_runner = next(iter(ale_runners.values()))
            if ale_preview_batch is None:
                continue
            preview_values = (
                ale_preview_batch["x_static"][:, available_static_features.index(str(feature_name))]
                .detach()
                .cpu()
                .numpy()
                .astype(float, copy=False)
            )
            preview_feature = recover_static_feature_plot_values(
                ale_analysis_frame,
                feature_name=str(feature_name),
                transformed_values=preview_values,
                arrays=primary_ale_runner.arrays,
            )
            bin_preview = build_adaptive_site_bins(
                preview_feature["site_frame"],
                feature_name=str(feature_name),
                target_bins=int(ale_bins),
                min_unique_sites_per_bin=int(ale_min_unique_sites_per_bin),
            )

            if bin_preview.get("analysis_kind") == "continuous":
                ale_df = compute_static_ale(
                    ale_runners,
                    head=str(head),
                    analysis_frame=ale_analysis_frame,
                    feature_name=str(feature_name),
                    bins=int(ale_bins),
                    min_unique_sites_per_bin=int(ale_min_unique_sites_per_bin),
                    bootstrap_repeats=int(ale_bootstrap_repeats),
                    random_seed=effective_ale_seed,
                )
                skip_reason = None
            elif bin_preview.get("analysis_kind") == "discrete":
                ale_df = compute_discrete_static_effect(
                    ale_runners,
                    head=str(head),
                    analysis_frame=ale_analysis_frame,
                    feature_name=str(feature_name),
                    bootstrap_repeats=int(ale_bootstrap_repeats),
                    random_seed=effective_ale_seed,
                )
                skip_reason = str(bin_preview.get("reason", "low_cardinality_feature"))
            else:
                ale_df = pd.DataFrame()
                skip_reason = str(bin_preview.get("reason", "insufficient_bin_support"))

            if ale_df.empty:
                warning = {
                    "target": str(head),
                    "feature": str(feature_name),
                    "reason": skip_reason or "insufficient_valid_samples_or_bins",
                }
                skipped_ale_features.append(warning)
                outputs["warnings"].append(warning)
                distribution_rows.append(
                    {
                        "target": str(head),
                        "feature": str(feature_name),
                        "feature_label": str(preview_feature["feature_label"]),
                        "feature_unit": str(preview_feature["feature_unit"]),
                        "minimum": float(np.nanmin(preview_feature["site_frame"]["plot_value"]))
                        if not preview_feature["site_frame"].empty
                        else np.nan,
                        "maximum": float(np.nanmax(preview_feature["site_frame"]["plot_value"]))
                        if not preview_feature["site_frame"].empty
                        else np.nan,
                        "valid_sample_count": int(len(ale_analysis_frame)),
                        "n_unique_sites": int(preview_feature["site_frame"]["site"].nunique()),
                        "normalization_status": str(preview_feature["normalization_status"]),
                    }
                )
                ale_reliability_rows.append(
                    {
                        "target": str(head),
                        "feature": str(feature_name),
                        "feature_label": str(preview_feature["feature_label"]),
                        "feature_unit": str(preview_feature["feature_unit"]),
                        "analysis_kind": str(bin_preview.get("analysis_kind", "unsupported")),
                        "feature_range_min": float(
                            np.nanmin(preview_feature["site_frame"]["plot_value"])
                        )
                        if not preview_feature["site_frame"].empty
                        else np.nan,
                        "feature_range_max": float(
                            np.nanmax(preview_feature["site_frame"]["plot_value"])
                        )
                        if not preview_feature["site_frame"].empty
                        else np.nan,
                        "n_unique_sites": int(preview_feature["site_frame"]["site"].nunique()),
                        "reliable_bins": 0,
                        "total_ale_range": np.nan,
                        "normalized_ale_range": np.nan,
                        "max_confidence_interval_width": np.nan,
                        "fraction_bins_excluding_zero": np.nan,
                        "minimum_unique_site_count": 0,
                        "output_representation": "",
                        "reliability_flag": "low-cardinality feature"
                        if "cardinality" in str(skip_reason)
                        else "insufficient bin support",
                        "skip_reason": skip_reason,
                    }
                )
                print(f"[static ALE] Skipping {head}:{feature_name} because {warning['reason']}.")
                continue

            all_ale_rows.append(ale_df.copy())
            head_ale_frames.append(ale_df.copy())
            ale_path = output_dirs["tables"] / f"ale_{head}_{feature_name}.csv"
            ale_df.to_csv(ale_path, index=False)
            exported_csv_paths.append(str(ale_path))

            distribution_rows.append(
                {
                    "target": str(head),
                    "feature": str(feature_name),
                    "feature_label": str(ale_df["feature_label"].iloc[0]),
                    "feature_unit": str(ale_df["feature_unit"].iloc[0]),
                    "minimum": float(
                        np.nanmin(pd.to_numeric(ale_df["bin_lower"], errors="coerce"))
                    ),
                    "maximum": float(
                        np.nanmax(pd.to_numeric(ale_df["bin_upper"], errors="coerce"))
                    ),
                    "valid_sample_count": int(ale_df["n_temporal_samples"].sum()),
                    "n_unique_sites": int(ale_df["n_unique_sites"].max()),
                    "normalization_status": str(ale_df["normalization_status"].iloc[0]),
                }
            )
            ale_reliability_rows.append(
                _build_ale_reliability_row(
                    ale_df,
                    head=str(head),
                    feature_name=str(feature_name),
                    target_std=target_std,
                    skip_reason=skip_reason,
                )
            )

            title = f"ALE of {str(ale_df['feature_label'].iloc[0]).split(' (')[0]} on {_ale_target_title(str(head))}"
            fig = plot_ale_with_histogram(
                ale_df,
                feature_label=str(ale_df["feature_label"].iloc[0]),
                title=title,
                y_label=_ale_target_y_label(
                    str(head), str(ale_df["output_representation"].iloc[0])
                ),
                count_label="Number of sites",
            )
            saved = save_figure_variants(
                fig, output_dirs["ale"] / f"ale_{head}_{feature_name}", close=True
            )
            ale_figure_paths.extend(str(path) for path in saved)

            cf_df = compute_static_counterfactual_curves(
                ale_runners,
                head=str(head),
                analysis_frame=ale_analysis_frame,
                feature_name=str(feature_name),
            )
            cf_path = output_dirs["tables"] / f"counterfactual_{head}_{feature_name}.csv"
            cf_df.to_csv(cf_path, index=False)
            exported_csv_paths.append(str(cf_path))

        if head_ale_frames:
            fig = plt.figure(figsize=(10.5, max(4.8, 4.8 * len(head_ale_frames))))
            grid = fig.add_gridspec(
                len(head_ale_frames) * 2,
                1,
                height_ratios=[4.0, 1.1] * len(head_ale_frames),
                hspace=0.35,
            )
            for idx, ale_df in enumerate(head_ale_frames):
                ax_main = fig.add_subplot(grid[idx * 2, 0])
                ax_hist = fig.add_subplot(grid[idx * 2 + 1, 0], sharex=ax_main)
                plot_ale_with_histogram(
                    ale_df,
                    feature_label=str(ale_df["feature_label"].iloc[0]),
                    title=f"{str(ale_df['feature_label'].iloc[0]).split(' (')[0]}",
                    y_label=_ale_target_y_label(
                        str(head), str(ale_df["output_representation"].iloc[0])
                    ),
                    count_label="Number of sites",
                    ax_main=ax_main,
                    ax_hist=ax_hist,
                )
            fig.suptitle(
                f"Selected ALE examples for {_target_display_name(head)}"
                + (" (bootstrap CI disabled)" if int(ale_bootstrap_repeats) <= 0 else ""),
                fontsize=14,
            )
            fig.tight_layout()
            saved = save_figure_variants(
                fig, output_dirs["ale"] / f"ale_{head}_selected", close=True
            )
            selected_ale_paths.extend(str(path) for path in saved)
            ale_figure_paths.extend(str(path) for path in saved)

    ale_values_csv_path = output_dirs["tables"] / "ale_values_all_features.csv"
    if all_ale_rows:
        ale_all_features_df = pd.concat(all_ale_rows, ignore_index=True)
    else:
        ale_all_features_df = pd.DataFrame(
            columns=[
                "target",
                "feature",
                "bin_center",
                "ale_value",
                "lower_confidence",
                "upper_confidence",
                "n_unique_sites",
                "n_temporal_samples",
                "normalization_status",
            ]
        )
    ale_all_features_df.to_csv(ale_values_csv_path, index=False)
    exported_csv_paths.append(str(ale_values_csv_path))

    ale_distribution_csv_path = output_dirs["tables"] / "ale_feature_distributions.csv"
    pd.DataFrame(distribution_rows).to_csv(ale_distribution_csv_path, index=False)
    exported_csv_paths.append(str(ale_distribution_csv_path))

    ale_reliability_df = (
        pd.DataFrame(ale_reliability_rows).sort_values(["target", "feature"]).reset_index(drop=True)
        if ale_reliability_rows
        else pd.DataFrame()
    )
    ale_reliability_csv_path = output_dirs["tables"] / "ale_reliability_summary.csv"
    ale_reliability_df.to_csv(ale_reliability_csv_path, index=False)
    exported_csv_paths.append(str(ale_reliability_csv_path))
    outputs["ale_reliability_summary"] = ale_reliability_df

    site_names = list(getattr(runner.arrays, "x_static", {}).keys())
    if site_names and getattr(runner.model, "use_static_features", False):
        static_matrix = np.vstack(
            [np.asarray(runner.arrays.x_static[site], dtype=np.float32) for site in site_names]
        )
        with torch.no_grad():
            embeddings = (
                runner.model.static_encoder(torch.as_tensor(static_matrix, device=runner.device))
                .detach()
                .cpu()
                .numpy()
            )
        pca = PCA(n_components=2, random_state=random_seed)
        pca_coords = pca.fit_transform(embeddings)
        embed_df = pd.DataFrame(
            {"site": site_names, "x": pca_coords[:, 0], "y": pca_coords[:, 1], "method": "pca"}
        )
        if umap is not None and len(site_names) >= 4:
            reducer = umap.UMAP(n_components=2, random_state=random_seed)
            umap_coords = reducer.fit_transform(embeddings)
            embed_df = pd.concat(
                [
                    embed_df,
                    pd.DataFrame(
                        {
                            "site": site_names,
                            "x": umap_coords[:, 0],
                            "y": umap_coords[:, 1],
                            "method": "umap",
                        }
                    ),
                ],
                ignore_index=True,
            )
        embed_df.to_csv(output_dirs["tables"] / "static_embeddings.csv", index=False)
        exported_csv_paths.append(str(output_dirs["tables"] / "static_embeddings.csv"))
        for method, sub_df in embed_df.groupby("method"):
            fig = plot_static_embedding_space(sub_df, title=f"Static embedding space ({method})")
            saved = save_figure_variants(
                fig, output_dirs["shap"] / f"static_embedding_{method}", close=True
            )
            shap_figure_paths.extend(str(path) for path in saved)
        outputs["embeddings"] = embed_df
        sitewise_analog_df = compute_sitewise_static_analogue_error_table(
            runner.bundle.results_dir,
            targets=targets_to_analyze,
            k_nearest=sitewise_k_nearest_analogues,
            exclude_self_for_training=True,
            training_config_path=training_config_path,
            point_centric_dir=point_centric_dir,
        )
        if (
            not sitewise_analog_df.empty
            and sitewise_analog_df["mean_training_analogue_distance"].notna().any()
        ):
            analogue_csv_path = output_dirs["tables"] / "sitewise_static_analogue_error.csv"
            sitewise_analog_df.to_csv(analogue_csv_path, index=False)
            exported_csv_paths.append(str(analogue_csv_path))
            fig = plot_sitewise_error_vs_analog_distance_grid(
                sitewise_analog_df,
                title="Site-wise normalized RMSE vs training analogue distance",
            )
            saved = save_figure_variants(
                fig,
                output_dirs["shap"] / "sitewise_normalized_rmse_vs_training_analogue_distance",
                close=True,
            )
            shap_figure_paths.extend(str(path) for path in saved)
            outputs["sitewise_static_analogue_error"] = sitewise_analog_df

    artifact_summary = {
        "resolved_run_directory": str(runner.bundle.results_dir),
        "resolved_output_directory": str(output_dirs["root"]),
        "shap_figure_count": len(shap_figure_paths),
        "ale_figure_count": len(ale_figure_paths),
        "skipped_feature_count": len(skipped_ale_features),
        "main_combined_figures": combined_shap_paths + selected_ale_paths,
        "exported_csv_tables": exported_csv_paths,
    }
    outputs["artifact_summary"] = artifact_summary
    outputs["skipped_ale_features"] = skipped_ale_features
    print("Static explainability outputs")
    print(f"- Resolved run directory: {runner.bundle.results_dir}")
    print(f"- Resolved output directory: {output_dirs['root']}")
    print(f"- SHAP figures saved: {len(shap_figure_paths)}")
    print(f"- ALE figures saved: {len(ale_figure_paths)}")
    print(f"- Skipped ALE features: {len(skipped_ale_features)}")
    print(f"- Main combined figures: {combined_shap_paths + selected_ale_paths}")
    print(f"- Exported CSV tables: {exported_csv_paths}")
    return outputs


def run_temporal_explainability_analysis(
    *,
    results_dir: str | Path,
    split: str = "val",
    device: str = "cuda",
    max_explain_samples: int = 256,
    random_seed: int = 42,
    targets_to_analyze: Sequence[str] = DEFAULT_HEADS,
    explain_quantity: str = "physical_prediction",
    dynamic_group_overrides: Mapping[str, Sequence[str]] | None = None,
    explain_batch_size: int | None = 16,
    temporal_ig_steps: int = 8,
    attention_max_samples: int = 16,
    attention_batch_size: int | None = None,
) -> dict[str, Any]:
    runner = load_model_for_explainability(results_dir, split=split, device=device)
    prediction_frame = load_prediction_frame(runner.bundle, split=split)
    sampled = sample_explainability_subset(
        prediction_frame, max_samples=max_explain_samples, random_seed=random_seed
    )
    run_dir = runner.bundle.explainability_dir / "analysis_06_temporal_explainability"
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_df = _printable_run_summary(runner.bundle, split, len(sampled))
    summary_df.to_csv(run_dir / "run_summary.csv", index=False)

    dynamic_names = load_feature_metadata(runner.bundle).get(
        "source_feature_names"
        if resolve_active_branches(runner.bundle).get("multi_source")
        else "dynamic_feature_names",
        [],
    )
    dynamic_groups = build_dynamic_feature_groups(dynamic_names, overrides=dynamic_group_overrides)
    pd.DataFrame({"unmatched_feature": dynamic_groups["unmatched"]}).to_csv(
        run_dir / "unmatched_dynamic_features.csv", index=False
    )
    outputs: dict[str, Any] = {"run_summary": summary_df, "dynamic_groups": dynamic_groups}
    sample_indices = sampled.index.to_list()
    for head in targets_to_analyze:
        ig_payload = compute_temporal_integrated_gradients(
            runner,
            head=head,
            quantity=explain_quantity,
            sample_indices=sample_indices,
            steps=temporal_ig_steps,
            batch_size=explain_batch_size,
        )
        ig_tensor = ig_payload["attributions"].numpy()
        np.save(run_dir / f"integrated_gradients_{head}.npy", ig_tensor)
        if ig_tensor.ndim == 4:
            feature_lag = np.abs(ig_tensor).mean(axis=(0, 2)).T
            source_lag = np.abs(ig_tensor).mean(axis=(0, 3)).T
        else:
            feature_lag = np.abs(ig_tensor).mean(axis=0).T
            source_lag = None
        plot_temporal_feature_lag_heatmap(
            feature_lag,
            y_labels=ig_payload["feature_names"],
            title=f"Temporal integrated gradients: {head}",
            out_path=run_dir / f"feature_lag_ig_{head}.png",
        )
        if source_lag is not None:
            plot_temporal_source_lag_heatmap(
                source_lag,
                y_labels=[f"source_{idx}" for idx in range(source_lag.shape[0])],
                title=f"Source x lag integrated gradients: {head}",
                out_path=run_dir / f"source_lag_ig_{head}.png",
            )
        occlusion_df = compute_temporal_occlusion(
            runner,
            head=head,
            quantity=explain_quantity,
            sample_indices=sample_indices,
            batch_size=explain_batch_size,
        )
        occlusion_df.to_csv(run_dir / f"temporal_occlusion_{head}.csv", index=False)
        family_df = compute_feature_family_occlusion(
            runner,
            head=head,
            quantity=explain_quantity,
            sample_indices=sample_indices,
            feature_groups=dynamic_groups,
            batch_size=explain_batch_size,
        )
        family_df.to_csv(run_dir / f"feature_family_occlusion_{head}.csv", index=False)
        if resolve_active_branches(runner.bundle).get("multi_source"):
            source_df = compute_source_occlusion(
                runner,
                head=head,
                quantity=explain_quantity,
                sample_indices=sample_indices,
                batch_size=explain_batch_size,
            )
            source_df.to_csv(run_dir / f"source_occlusion_{head}.csv", index=False)
        try:
            attention_payload = extract_cross_attention_maps(
                runner,
                sample_indices=sample_indices[
                    : min(len(sample_indices), max(1, int(attention_max_samples)))
                ],
                batch_size=attention_batch_size
                if attention_batch_size is not None
                else explain_batch_size,
            )
            attention_tensor = (
                attention_payload["cross_attention_weights"].numpy().mean(axis=0).mean(axis=1)
            )
            plot_attention_by_head(
                attention_tensor,
                title=f"Cross attention by head: {head}",
                out_path=run_dir / f"attention_by_head_{head}.png",
            )
            pd.DataFrame(aggregate_attention_by_head(attention_payload)).to_csv(
                run_dir / f"attention_summary_{head}.csv", index=False
            )
        except Exception as exc:
            _write_note(
                run_dir / f"attention_warning_{head}.txt", f"Attention maps unavailable: {exc}"
            )
        outputs[head] = {"occlusion": occlusion_df, "feature_family_occlusion": family_df}
    return outputs


def run_cross_branch_explainability_analysis(
    *,
    results_dir: str | Path,
    split: str = "val",
    device: str = "cuda",
    max_explain_samples: int = 256,
    random_seed: int = 42,
    targets_to_analyze: Sequence[str] = DEFAULT_HEADS,
    explain_quantity: str = "physical_prediction",
    results_dirs: Mapping[str, str] | None = None,
    explain_batch_size: int | None = 16,
    bathy_max_explain_samples: int = 8,
    bathy_ig_steps: int = 8,
    bathy_occlusion_stride: int = 16,
    bathy_batch_size: int | None = None,
) -> dict[str, Any]:
    runner = load_model_for_explainability(results_dir, split=split, device=device)
    prediction_frame = _attach_static_features_to_predictions(
        runner.bundle, load_prediction_frame(runner.bundle, split=split)
    )
    sampled = sample_explainability_subset(
        prediction_frame, max_samples=max_explain_samples, random_seed=random_seed
    )
    run_dir = runner.bundle.explainability_dir / "analysis_07_cross_branch_explainability"
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_df = _printable_run_summary(runner.bundle, split, len(sampled))
    summary_df.to_csv(run_dir / "run_summary.csv", index=False)

    outputs: dict[str, Any] = {"run_summary": summary_df}
    sample_indices = sampled.index.to_list()
    for head in targets_to_analyze:
        branch_df = compute_branch_ablation_importance(
            runner,
            head=head,
            quantity=explain_quantity,
            sample_indices=sample_indices,
            batch_size=explain_batch_size,
        )
        branch_df.to_csv(run_dir / f"branch_ablation_{head}.csv", index=False)
        plot_branch_ablation_bars(
            branch_df,
            title=f"Branch ablation importance: {head}",
            out_path=run_dir / f"branch_ablation_{head}.png",
        )
        contribution_df = compute_modality_contribution_summary(
            branch_df, head=head, quantity=explain_quantity
        )
        contribution_df.to_csv(run_dir / f"modality_contribution_{head}.csv", index=False)
        outputs[head] = contribution_df

    interaction_candidates = [
        ("target_hs", "open_sector_width_deg", "pred_hs"),
        ("target_hs", "ray_fetch_mean_m", "pred_hs"),
        ("target_hs", "local_depth_m", "pred_hs"),
        ("pred_tp", "path_length_m", "pred_tp"),
    ]
    interaction_tables = []
    for feature_x, feature_y, response_col in interaction_candidates:
        table = compute_static_dynamic_interactions(
            prediction_frame, feature_x=feature_x, feature_y=feature_y, response_col=response_col
        )
        if table.empty:
            continue
        safe_name = f"{feature_x}__{feature_y}__{response_col}".replace("/", "_")
        table.to_csv(run_dir / f"interaction_{safe_name}.csv", index=False)
        interaction_tables.append(
            table.assign(feature_x=feature_x, feature_y=feature_y, response_col=response_col)
        )
    if interaction_tables:
        outputs["interactions"] = pd.concat(interaction_tables, ignore_index=True)

    if resolve_active_branches(runner.bundle).get("bathy"):
        bathy_sample_indices = sample_indices[
            : min(len(sample_indices), max(1, int(bathy_max_explain_samples)))
        ]
        resolved_bathy_batch_size = (
            bathy_batch_size if bathy_batch_size is not None else explain_batch_size
        )
        for head in targets_to_analyze:
            bathy_ig = compute_bathy_integrated_gradients(
                runner,
                head=head,
                quantity=explain_quantity,
                sample_indices=bathy_sample_indices,
                steps=bathy_ig_steps,
                batch_size=resolved_bathy_batch_size,
            )
            bathy_occ = compute_bathy_occlusion_maps(
                runner,
                head=head,
                quantity=explain_quantity,
                sample_indices=bathy_sample_indices,
                patch_stride=bathy_occlusion_stride,
                batch_size=resolved_bathy_batch_size,
            )
            if bathy_ig.get("available"):
                mean_absolute_map = bathy_ig.get("mean_absolute_map")
                if torch.is_tensor(mean_absolute_map):
                    plot_bathy_attribution_map(
                        mean_absolute_map.numpy(),
                        title=f"Bathy integrated gradients: {head}",
                        out_path=run_dir / f"bathy_ig_{head}.png",
                    )
            if bathy_occ.get("available"):
                plot_bathy_attribution_map(
                    bathy_occ["map"].numpy(),
                    title=f"Bathy occlusion: {head}",
                    out_path=run_dir / f"bathy_occlusion_{head}.png",
                )
    else:
        _write_note(
            run_dir / "bathy_note.txt",
            "Bathymetry branch disabled in this run. Bathy explainability sections were skipped.",
        )

    if results_dirs:
        rows = []
        for label, path in results_dirs.items():
            other_bundle = load_results_bundle(path)
            metrics_path = other_bundle.results_dir / "metrics" / "metrics_validation.csv"
            if not metrics_path.exists():
                continue
            metrics_df = pd.read_csv(metrics_path)
            metrics_df["run_label"] = label
            rows.append(metrics_df)
        if rows:
            comparison = pd.concat(rows, ignore_index=True)
            comparison.to_csv(run_dir / "multi_run_metrics_comparison.csv", index=False)
            outputs["multi_run_comparison"] = comparison
    return outputs


def run_failure_mode_explainability_analysis(
    *,
    results_dir: str | Path,
    split: str = "val",
    device: str = "cuda",
    max_explain_samples: int = 256,
    random_seed: int = 42,
    point_name: str = "norac_grid_131",
) -> dict[str, Any]:
    runner = load_model_for_explainability(results_dir, split=split, device=device)
    prediction_frame = _attach_static_features_to_predictions(
        runner.bundle, load_prediction_frame(runner.bundle, split=split)
    )
    sampled = sample_explainability_subset(
        prediction_frame, max_samples=max_explain_samples, random_seed=random_seed
    )
    run_dir = runner.bundle.explainability_dir / "analysis_08_failure_mode_explainability"
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_df = _printable_run_summary(runner.bundle, split, len(sampled))
    summary_df.to_csv(run_dir / "run_summary.csv", index=False)

    ranked = rank_failure_sites(prediction_frame)
    ranked.to_csv(run_dir / "ranked_failure_sites.csv", index=False)
    site_delta = compare_good_vs_bad_sites(ranked)
    site_delta.to_csv(run_dir / "good_vs_bad_sites.csv", index=False)
    feature_groups = build_feature_groups(
        load_feature_metadata(runner.bundle).get("static_feature_names", [])
    )
    sample_indices = sampled.index.to_list()
    error_payload = compute_error_attributions(
        runner, head="hs", sample_indices=sample_indices, feature_groups=feature_groups
    )
    error_payload["static_grouped"]["summary"].to_csv(
        run_dir / "error_static_grouped_shap_hs.csv", index=False
    )
    error_payload["dynamic_occlusion"].to_csv(
        run_dir / "error_dynamic_occlusion_hs.csv", index=False
    )

    site_summary = explain_single_site(prediction_frame, ranked, point_name=point_name)
    _write_note(run_dir / f"{point_name}_diagnosis.md", site_summary["summary"])
    compare_df = compare_good_vs_bad_samples(
        error_payload["static_grouped"]["beeswarm"].assign(score=0.0),
        score_col="score",
        group_col="group",
        value_col="value",
    )
    compare_df.to_csv(run_dir / "good_bad_attribution_delta.csv", index=False)
    if not compare_df.empty:
        plot_good_bad_attribution_delta(
            compare_df,
            title="Good vs bad attribution delta",
            out_path=run_dir / "good_bad_attribution_delta.png",
        )

    return {
        "run_summary": summary_df,
        "ranked_sites": ranked,
        "good_vs_bad_sites": site_delta,
        "site_summary": site_summary,
    }
