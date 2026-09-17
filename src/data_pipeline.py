"""Load prepared coastal-wave artifacts and expose them to PyTorch.

This module is the boundary between preprocessing and model execution.  It
reads the point-centric NPZ/JSON products, validates their feature contracts,
selects train/validation/test windows, and constructs the datasets and data
loaders consumed by training and evaluation.  It never fits preprocessing
statistics: those are stored with the prepared artifacts and reused here.
"""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch.utils.data._utils.collate import default_collate

try:
    from ablation import (
        DEFAULT_ABLATION_CONFIG_PATH,
        build_transformed_static_group_manifest,
        load_ablation_config,
        resolve_transformed_static_ablation,
        validate_transformed_static_group_manifest,
    )
    from preprocessing.transfer_targets import (
        PHYSICAL_TARGET_NAMES,
        TRANSFER_SCALER_COLUMNS,
        resolve_targets_config,
        validate_target_name_block,
    )
except Exception:
    from src.ablation import (
        DEFAULT_ABLATION_CONFIG_PATH,
        build_transformed_static_group_manifest,
        load_ablation_config,
        resolve_transformed_static_ablation,
        validate_transformed_static_group_manifest,
    )
    from src.preprocessing.transfer_targets import (
        PHYSICAL_TARGET_NAMES,
        TRANSFER_SCALER_COLUMNS,
        resolve_targets_config,
        validate_target_name_block,
    )

try:
    from config_resolution import resolve_config
except Exception:
    from src.config_resolution import resolve_config


logger = logging.getLogger(__name__)

_SEASONAL_PERIOD_DAYS = np.float32(365.25)
_NORMALIZATION_MODULE = None
_LEGACY_VALIDATION_WARNING_EMITTED = False
_EXPECTED_TARGET_FEATURE_ORDER = ["hs", "tp", "dir_sin", "dir_cos", "dp_sin", "dp_cos"]
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SITE_HOLDOUT_TEMPORAL_MODES = {"legacy", "shared_recent"}


def _resolve_repo_relative_path(path_like: str | Path) -> Path:
    """Resolve repo-relative config paths robustly across scripts and notebooks."""
    path = Path(path_like).expanduser()
    if path.is_absolute():
        return path.resolve()

    cwd_candidate = (Path.cwd() / path).resolve()
    if cwd_candidate.exists():
        return cwd_candidate

    repo_candidate = (_REPO_ROOT / path).resolve()
    if repo_candidate.exists():
        return repo_candidate

    return cwd_candidate


def _resolve_runtime_source_geometry_features_flag(config: dict) -> bool:
    data_cfg = (config.get("data", {}) or {}) if isinstance(config, dict) else {}
    model_cfg = (config.get("model", {}) or {}) if isinstance(config, dict) else {}
    coastal_cfg = (
        (model_cfg.get("coastal_transformer", {}) or {}) if isinstance(model_cfg, dict) else {}
    )
    multi_source_cfg = (
        (coastal_cfg.get("multi_source", {}) or {}) if isinstance(coastal_cfg, dict) else {}
    )

    return bool(multi_source_cfg.get("use_geometry_features", False))


def resolve_static_ablation_config_path(config: dict | None) -> str:
    """Resolve the runtime static ablation config path from training config."""
    data_cfg = (config or {}).get("data", {}) or {}
    configured = data_cfg.get("static_ablation_config")
    if configured in (None, ""):
        return DEFAULT_ABLATION_CONFIG_PATH
    return str(configured)


def _load_normalization_module():
    """Return the canonical normalization implementation."""
    from src.preprocessing import normalize

    return normalize


@dataclass
class PointCentricArrays:
    """Container for point-centric arrays loaded from disk."""

    x_dynamic: np.ndarray
    x_dynamic_sources: np.ndarray | None
    x_dynamic_sitewise: Dict[str, np.ndarray]
    source_geometry: np.ndarray | None
    y_targets: Dict[str, np.ndarray]
    y_physical: Dict[str, np.ndarray]
    y_transfer: Dict[str, np.ndarray]
    y_reference: Dict[str, np.ndarray]
    x_static: Dict[str, np.ndarray]
    x_bathy: np.ndarray | None
    local_depth_m: np.ndarray | None
    local_breaking_hs_cap: np.ndarray | None
    local_breaking_cap_valid: np.ndarray | None
    target_sites: List[str]
    target_mode: str
    dynamic_feature_names: List[str]
    source_feature_names: List[str]
    site_dynamic_feature_names: List[str]
    source_geometry_feature_names: List[str]
    target_feature_names: List[str]
    physical_target_names: List[str]
    transfer_target_names: List[str]
    reference_target_names: List[str]
    static_feature_names: List[str]
    bathy_channel_names: List[str]
    bathy_site_to_index: Dict[str, int]
    bathy_patch_size: int | None
    bathy_resolution_m: float | None
    bathy_normalization_metadata: dict
    timestamps: np.ndarray
    split_idx: Dict[str, np.ndarray]
    metadata: dict
    ablation_summary: dict | None = None


def _safe_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in name)


def _npz_string_list(npz, key: str) -> List[str]:
    if key not in npz:
        return []
    return npz[key].astype(str).tolist()


def _unwrap_npz_object(value) -> dict:
    if isinstance(value, dict):
        return value
    if isinstance(value, np.ndarray) and value.shape == ():
        try:
            item = value.item()
        except Exception:
            return {}
        return item if isinstance(item, dict) else {}
    return {}


def _resolve_bathy_channel_selection(
    available_channel_names: Sequence[str],
    available_channel_count: int,
    requested_channels: Sequence[str] | None = None,
    requested_in_channels: int | None = None,
) -> Tuple[List[int], List[str]]:
    """Resolve bathymetry channel indices to keep at runtime."""
    available_names = [str(name) for name in (available_channel_names or [])]
    total_channels = int(available_channel_count)
    if total_channels < 1:
        raise ValueError(f"Expected bathymetry channel count >= 1, got {total_channels}")

    selected_names = [str(name) for name in (requested_channels or [])]
    in_channels = None if requested_in_channels is None else int(requested_in_channels)

    if in_channels is not None and in_channels < 1:
        raise ValueError(
            f"model.coastal_transformer.bathy.in_channels must be >= 1, got {in_channels}"
        )

    # Allow users to reduce channel count by setting in_channels while leaving
    # the full canonical channels list in config.
    if in_channels is not None and selected_names:
        if in_channels > len(selected_names):
            raise ValueError(
                "Bathymetry config mismatch: "
                f"in_channels={in_channels} exceeds configured channels list length={len(selected_names)}."
            )
        selected_names = selected_names[:in_channels]

    if selected_names:
        if not available_names:
            raise ValueError(
                "Bathymetry channel names were requested in config, but point_centric_X_bathy.npz "
                "does not include channel_names metadata."
            )
        available_lookup = {name: idx for idx, name in enumerate(available_names)}
        missing = [name for name in selected_names if name not in available_lookup]
        if missing:
            raise ValueError(
                "Requested bathymetry channel(s) not found in point_centric_X_bathy.npz: "
                f"{missing}. Available: {available_names}"
            )
        indices = [int(available_lookup[name]) for name in selected_names]
        return indices, selected_names

    if in_channels is None:
        indices = list(range(total_channels))
    else:
        if in_channels > total_channels:
            raise ValueError(
                "Bathymetry config/data channel mismatch: "
                f"model.coastal_transformer.bathy.in_channels={in_channels} "
                f"but point_centric_X_bathy.npz contains only {total_channels} channel(s)."
            )
        indices = list(range(in_channels))

    if available_names:
        names = [available_names[idx] for idx in indices]
    else:
        names = [f"channel_{idx}" for idx in indices]
    return indices, names


def _build_temporal_seasonality(timestamps: np.ndarray) -> np.ndarray:
    """Encode timestamps as cyclical day-of-year features.

    For each timestep with day-of-year ``d`` we compute

    ``time_sin = sin(2*pi*d/365.25)`` and
    ``time_cos = cos(2*pi*d/365.25)``.

    The 365.25 denominator keeps phase continuity across leap years while
    preserving the circular embedding used by sequence models.
    """
    ts = np.asarray(timestamps)
    if ts.ndim != 1:
        raise ValueError(f"Expected 1D timestamps array, got shape {ts.shape}")

    try:
        ts64 = ts.astype("datetime64[ns]")
    except Exception as exc:
        raise ValueError(
            "Failed to parse timestamps into datetime64[ns] for seasonality encoding"
        ) from exc

    if np.isnat(ts64).any():
        bad_count = int(np.count_nonzero(np.isnat(ts64)))
        raise ValueError(
            f"Found {bad_count} invalid timestamp(s) while computing seasonality features"
        )

    day_start = ts64.astype("datetime64[D]")
    year_start = ts64.astype("datetime64[Y]").astype("datetime64[D]")
    day_of_year = (day_start - year_start).astype(np.int32) + 1

    phase = (2.0 * np.pi * day_of_year.astype(np.float32)) / _SEASONAL_PERIOD_DAYS
    return np.column_stack((np.sin(phase), np.cos(phase))).astype(np.float32, copy=False)


def _column_median_fill_values(
    matrix: np.ndarray, ref_rows: np.ndarray | None = None
) -> np.ndarray:
    """Compute per-column median fill values from finite entries.

    Uses `ref_rows` (for example training indices) when provided.
    Falls back to all rows and then to 0.0 if no finite values exist.
    """
    arr = np.asarray(matrix, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array for fill-value computation, got shape {arr.shape}")

    if ref_rows is not None and ref_rows.size > 0:
        ref = arr[np.asarray(ref_rows, dtype=int), :]
    else:
        ref = arr

    n_cols = arr.shape[1]
    fill = np.zeros(n_cols, dtype=np.float32)

    for j in range(n_cols):
        col = ref[:, j]
        valid = col[np.isfinite(col)]
        if valid.size > 0:
            fill[j] = np.float32(np.median(valid))
            continue

        # Fallback to all rows if train rows had no finite values.
        valid_all = arr[np.isfinite(arr[:, j]), j]
        fill[j] = np.float32(np.median(valid_all)) if valid_all.size > 0 else 0.0

    return fill


def _replace_non_finite_2d(
    matrix: np.ndarray,
    label: str,
    ref_rows: np.ndarray | None = None,
) -> np.ndarray:
    """Replace NaN/Inf in a 2D array using per-column median imputation."""
    arr = np.asarray(matrix, dtype=np.float32).copy()
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array for {label}, got shape {arr.shape}")

    bad = ~np.isfinite(arr)
    bad_count = int(np.count_nonzero(bad))
    if bad_count == 0:
        return arr

    fill = _column_median_fill_values(arr, ref_rows=ref_rows)
    rows, cols = np.where(bad)
    arr[rows, cols] = fill[cols]

    logger.warning("Replaced %d non-finite values in %s", bad_count, label)
    return arr


def _replace_non_finite_4d_lastdim(
    tensor: np.ndarray,
    label: str,
    ref_rows: np.ndarray | None = None,
) -> np.ndarray:
    """Replace NaN/Inf in a `[N, T, K, D]` tensor using per-feature medians."""
    arr = np.asarray(tensor, dtype=np.float32).copy()
    if arr.ndim != 4:
        raise ValueError(f"Expected 4D array for {label}, got shape {arr.shape}")

    bad = ~np.isfinite(arr)
    bad_count = int(np.count_nonzero(bad))
    if bad_count == 0:
        return arr

    n_features = arr.shape[-1]
    if ref_rows is not None and ref_rows.size > 0:
        selected = arr[:, np.asarray(ref_rows, dtype=int), :, :].reshape(-1, n_features)
    else:
        selected = arr.reshape(-1, n_features)
    fill = _column_median_fill_values(selected, ref_rows=None)
    feature_indices = np.where(bad)[3]
    arr[bad] = fill[feature_indices]
    logger.warning("Replaced %d non-finite values in %s", bad_count, label)
    return arr


def _replace_non_finite_static_vectors(x_static: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Replace NaN/Inf in static vectors using feature-wise medians across sites."""
    if not x_static:
        return x_static

    ordered_sites = list(x_static.keys())
    mat = np.vstack([np.asarray(x_static[s], dtype=np.float32) for s in ordered_sites])
    if mat.ndim != 2:
        raise ValueError(f"Expected 2D stacked static matrix, got shape {mat.shape}")

    bad = ~np.isfinite(mat)
    bad_count = int(np.count_nonzero(bad))
    if bad_count == 0:
        return {site: mat[i, :] for i, site in enumerate(ordered_sites)}

    fill = _column_median_fill_values(mat, ref_rows=None)
    rows, cols = np.where(bad)
    mat[rows, cols] = fill[cols]

    logger.warning("Replaced %d non-finite values in X_static", bad_count)
    return {site: mat[i, :] for i, site in enumerate(ordered_sites)}


def _legacy_site_holdout_keys_present(config: dict) -> List[str]:
    data_cfg = (config.get("data", {}) or {}) if isinstance(config, dict) else {}
    split_cfg = (config.get("split", {}) or {}) if isinstance(config, dict) else {}

    present: List[str] = []
    if isinstance(data_cfg, dict) and "holdout_sites" in data_cfg:
        present.append("data.holdout_sites")
    if isinstance(split_cfg, dict) and "norac_holdout" in split_cfg:
        present.append("split.norac_holdout")
    return present


def _raise_if_legacy_site_holdout_keys_present(config: dict) -> None:
    present = _legacy_site_holdout_keys_present(config)
    if not present:
        return
    raise ValueError(
        "Legacy site holdout keys are no longer supported: "
        f"{present}. Use data.validation_sites for held-out validation sites "
        "and data.test_sites for final test sites."
    )


def _validate_target_feature_contract(target_feature_names: Sequence[str]) -> List[str]:
    names = [str(name) for name in target_feature_names]
    if names != _EXPECTED_TARGET_FEATURE_ORDER:
        raise ValueError(
            f"Target feature order must be exactly {_EXPECTED_TARGET_FEATURE_ORDER}, got {names}"
        )
    return names


def apply_runtime_static_ablation(
    arrays: PointCentricArrays,
    ablation_path: str = DEFAULT_ABLATION_CONFIG_PATH,
) -> PointCentricArrays:
    """Re-apply static ablation to prebuilt point-centric arrays when metadata allows."""
    ablation_cfg = load_ablation_config(ablation_path)
    static_scaler_meta = (
        ((arrays.metadata.get("normalization", {}) or {}).get("static_scaler", {}) or {})
        if isinstance(arrays.metadata, dict)
        else {}
    )
    raw_map = static_scaler_meta.get(
        "raw_to_transformed_feature_map_before_ablation"
    ) or static_scaler_meta.get("raw_to_transformed_feature_map", {})
    summary = resolve_transformed_static_ablation(
        transformed_feature_names=arrays.static_feature_names,
        raw_to_transformed_map=raw_map,
        ablation_cfg=ablation_cfg,
    )
    validation_cfg = (
        (ablation_cfg.get("validation", {}) or {}) if isinstance(ablation_cfg, dict) else {}
    )
    if bool(validation_cfg.get("enabled", False)):
        group_manifest = build_transformed_static_group_manifest(
            transformed_feature_names=arrays.static_feature_names,
            raw_to_transformed_map=raw_map,
            ablation_cfg=ablation_cfg,
        )
        validate_transformed_static_group_manifest(group_manifest)
        summary["group_manifest"] = group_manifest
    before_count = len(arrays.static_feature_names)
    summary["static_feature_count_before"] = before_count

    if not summary.get("enabled", False):
        summary["static_feature_count_after"] = before_count
        arrays.ablation_summary = summary
        return arrays

    to_drop = [
        str(name)
        for name in summary.get("matched_transformed_features", [])
        if str(name) in arrays.static_feature_names
    ]
    if not to_drop:
        summary["static_feature_count_after"] = before_count
        arrays.ablation_summary = summary
        return arrays

    keep_indices = [
        idx for idx, name in enumerate(arrays.static_feature_names) if name not in set(to_drop)
    ]
    arrays.static_feature_names = [arrays.static_feature_names[idx] for idx in keep_indices]
    arrays.x_static = {
        site: np.asarray(values, dtype=np.float32)[keep_indices]
        for site, values in arrays.x_static.items()
    }
    summary["static_feature_count_after"] = len(arrays.static_feature_names)
    if bool(validation_cfg.get("enabled", False)):
        expected_after = int(before_count - len(to_drop))
        if summary["static_feature_count_after"] != expected_after:
            raise ValueError(
                "Static ablation group validation failed: resolved post-drop feature count "
                f"{summary['static_feature_count_after']} did not equal expected baseline-minus-drop count {expected_after}."
            )
    arrays.ablation_summary = summary
    return arrays


def _apply_train_site_subsampling(
    train_sites: Sequence[str], subsampling_cfg: dict
) -> tuple[list[str], dict]:
    before_sites = [str(site) for site in train_sites]
    summary = {
        "enabled": bool(subsampling_cfg.get("enabled", False)),
        "applied": False,
        "fraction": float(subsampling_cfg.get("fraction", 1.0)),
        "seed": int(subsampling_cfg.get("seed", 42)),
        "before_count": int(len(before_sites)),
        "after_count": int(len(before_sites)),
        "removed_count": 0,
    }
    if not summary["enabled"] or summary["fraction"] >= 1.0:
        return before_sites, summary
    if not before_sites:
        raise ValueError(
            "Train site subsampling is enabled but the training site set is empty before subsampling."
        )

    keep_count = max(1, int(np.floor(len(before_sites) * summary["fraction"])))
    keep_count = min(keep_count, len(before_sites))
    if keep_count == len(before_sites):
        return before_sites, summary

    rng = np.random.default_rng(summary["seed"])
    selected_positions = np.sort(
        rng.choice(len(before_sites), size=keep_count, replace=False).astype(int)
    )
    reduced_sites = [before_sites[int(pos)] for pos in selected_positions.tolist()]
    summary["applied"] = True
    summary["after_count"] = int(len(reduced_sites))
    summary["removed_count"] = int(summary["before_count"] - summary["after_count"])
    return reduced_sites, summary


def load_point_centric_arrays(
    data_dir: str,
    bathy_channels: Sequence[str] | None = None,
    bathy_in_channels: int | None = None,
) -> PointCentricArrays:
    """Load and validate the prepared arrays for one preprocessing run.

    Args:
        data_dir: Directory containing the point-centric NPZ files and
            metadata JSON.
        bathy_channels: Expected bathymetry channel names, when enabled.
        bathy_in_channels: Expected bathymetry channel count, when enabled.

    Returns:
        A single container holding dynamic, static, target, source-geometry,
        and optional bathymetry arrays, together with their saved schemas.
    """
    base = _resolve_repo_relative_path(data_dir)
    x_path = base / "point_centric_X_dynamic.npz"
    x_sources_path = base / "point_centric_X_dynamic_sources.npz"
    x_site_path = base / "point_centric_X_dynamic_sitewise.npz"
    y_path = base / "point_centric_Y_targets.npz"
    s_path = base / "point_centric_X_static.npz"
    p_path = base / "point_centric_physics.npz"
    b_path = base / "point_centric_X_bathy.npz"
    source_geometry_path = base / "point_centric_source_geometry.npz"

    if not x_path.exists():
        raise FileNotFoundError(f"Missing dynamic NPZ: {x_path}")
    if not y_path.exists():
        raise FileNotFoundError(f"Missing target NPZ: {y_path}")
    x_npz = np.load(x_path, allow_pickle=True)
    x_sources_npz = np.load(x_sources_path, allow_pickle=True) if x_sources_path.exists() else None
    x_site_npz = np.load(x_site_path, allow_pickle=True) if x_site_path.exists() else None
    y_npz = np.load(y_path, allow_pickle=True)
    s_npz = np.load(s_path, allow_pickle=True) if s_path.exists() else None
    p_npz = np.load(p_path, allow_pickle=True) if p_path.exists() else None
    b_npz = np.load(b_path, allow_pickle=True) if b_path.exists() else None
    source_geometry_npz = (
        np.load(source_geometry_path, allow_pickle=True) if source_geometry_path.exists() else None
    )
    metadata_path = base / "point_centric_metadata.json"
    metadata: dict = {}
    if metadata_path.exists():
        try:
            with metadata_path.open("r") as fh:
                metadata = json.load(fh) or {}
        except Exception:
            metadata = {}

    x_dynamic = x_npz["X_dynamic"].astype(np.float32)
    timestamps = x_npz["timestamps"].astype(str)
    split_idx = {
        "train": x_npz["train_idx"].astype(int),
        "val": x_npz["val_idx"].astype(int),
        "test": x_npz["test_idx"].astype(int),
    }

    train_idx = split_idx["train"]
    x_dynamic = _replace_non_finite_2d(
        x_dynamic,
        label="X_dynamic",
        ref_rows=train_idx,
    )

    target_sites = y_npz["target_sites"].astype(str).tolist()
    target_mode = str(y_npz["target_mode"][0]) if "target_mode" in y_npz else "physical"
    dynamic_feature_names = list(metadata.get("dynamic_feature_names", []) or [])
    source_feature_names = []
    source_geometry_feature_names = []
    site_dynamic_feature_names = []
    if x_site_npz is not None and "site_dynamic_feature_names" in x_site_npz:
        site_dynamic_feature_names = x_site_npz["site_dynamic_feature_names"].astype(str).tolist()
    if x_sources_npz is not None and "source_feature_names" in x_sources_npz:
        source_feature_names = x_sources_npz["source_feature_names"].astype(str).tolist()
    if source_geometry_npz is not None and "source_geometry_feature_names" in source_geometry_npz:
        source_geometry_feature_names = (
            source_geometry_npz["source_geometry_feature_names"].astype(str).tolist()
        )
    target_feature_names = _validate_target_feature_contract(
        _npz_string_list(y_npz, "target_feature_names")
    )
    physical_target_names = _npz_string_list(y_npz, "physical_target_names")
    transfer_target_names = _npz_string_list(y_npz, "transfer_target_names")
    reference_target_names = _npz_string_list(y_npz, "reference_target_names")
    if not physical_target_names:
        physical_target_names = list(PHYSICAL_TARGET_NAMES)
    static_feature_names = (
        s_npz["static_feature_names"].astype(str).tolist() if s_npz is not None else []
    )
    bathy_channel_names: List[str] = []
    bathy_site_to_index: Dict[str, int] = {}
    bathy_patch_size: int | None = None
    bathy_resolution_m: float | None = None
    bathy_normalization_metadata: dict = {}
    x_bathy: np.ndarray | None = None
    local_depth_m: np.ndarray | None = None
    local_breaking_hs_cap: np.ndarray | None = None
    local_breaking_cap_valid: np.ndarray | None = None
    x_dynamic_sources: np.ndarray | None = None
    source_geometry: np.ndarray | None = None
    if s_npz is not None and "target_sites" in s_npz:
        static_sites = s_npz["target_sites"].astype(str).tolist()
        if set(static_sites) != set(target_sites):
            missing = sorted(set(target_sites) - set(static_sites))
            extra = sorted(set(static_sites) - set(target_sites))
            raise ValueError(
                "point_centric_X_static.npz site mismatch with target_sites: "
                f"missing={missing} extra={extra}"
            )
    if x_site_npz is not None and "target_sites" in x_site_npz:
        site_dynamic_sites = x_site_npz["target_sites"].astype(str).tolist()
        if set(site_dynamic_sites) != set(target_sites):
            missing = sorted(set(target_sites) - set(site_dynamic_sites))
            extra = sorted(set(site_dynamic_sites) - set(target_sites))
            raise ValueError(
                "point_centric_X_dynamic_sitewise.npz site mismatch with target_sites: "
                f"missing={missing} extra={extra}"
            )
    if b_npz is not None:
        if "X_bathy" not in b_npz:
            raise KeyError(f"Missing X_bathy array in {b_path}")
        bathy_channel_names = (
            b_npz["channel_names"].astype(str).tolist() if "channel_names" in b_npz else []
        )
        x_bathy_np = b_npz["X_bathy"]
        if x_bathy_np.ndim != 4:
            raise ValueError(
                "Expected bathymetry tensor with shape [site, channel, y, x], "
                f"got shape {tuple(x_bathy_np.shape)} from {b_path}"
            )
        selected_channel_indices, selected_channel_names = _resolve_bathy_channel_selection(
            available_channel_names=bathy_channel_names,
            available_channel_count=int(x_bathy_np.shape[1]),
            requested_channels=bathy_channels,
            requested_in_channels=bathy_in_channels,
        )
        x_bathy_selected = x_bathy_np[:, selected_channel_indices, :, :]
        x_bathy = np.nan_to_num(
            x_bathy_selected.astype(np.float32),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        bathy_channel_names = selected_channel_names
        bathy_sites = b_npz["target_sites"].astype(str).tolist() if "target_sites" in b_npz else []
        bathy_site_to_index = {site: idx for idx, site in enumerate(bathy_sites)}
        if "patch_size" in b_npz:
            bathy_patch_size = int(np.asarray(b_npz["patch_size"]).reshape(-1)[0])
        if "resolution_m" in b_npz:
            bathy_resolution_m = float(np.asarray(b_npz["resolution_m"]).reshape(-1)[0])
        if "normalization_metadata" in b_npz:
            bathy_normalization_metadata = _unwrap_npz_object(b_npz["normalization_metadata"])
    if x_sources_npz is not None:
        if "X_dynamic_sources" not in x_sources_npz:
            raise KeyError(f"Missing X_dynamic_sources array in {x_sources_path}")
        source_target_sites = (
            x_sources_npz["target_sites"].astype(str).tolist()
            if "target_sites" in x_sources_npz
            else []
        )
        if source_target_sites:
            if set(source_target_sites) != set(target_sites):
                missing = sorted(set(target_sites) - set(source_target_sites))
                extra = sorted(set(source_target_sites) - set(target_sites))
                raise ValueError(
                    "point_centric_X_dynamic_sources.npz site mismatch with target_sites: "
                    f"missing={missing} extra={extra}"
                )
            source_order = np.asarray(
                [source_target_sites.index(site) for site in target_sites], dtype=int
            )
        else:
            source_order = np.arange(x_sources_npz["X_dynamic_sources"].shape[0], dtype=int)
        x_dynamic_sources = _replace_non_finite_4d_lastdim(
            x_sources_npz["X_dynamic_sources"].astype(np.float32)[source_order, ...],
            label="X_dynamic_sources",
            ref_rows=train_idx,
        )
    if source_geometry_npz is not None:
        if "source_geometry" not in source_geometry_npz:
            raise KeyError(f"Missing source_geometry array in {source_geometry_path}")
        source_geometry_sites = (
            source_geometry_npz["target_sites"].astype(str).tolist()
            if "target_sites" in source_geometry_npz
            else []
        )
        if source_geometry_sites:
            if set(source_geometry_sites) != set(target_sites):
                missing = sorted(set(target_sites) - set(source_geometry_sites))
                extra = sorted(set(source_geometry_sites) - set(target_sites))
                raise ValueError(
                    "point_centric_source_geometry.npz site mismatch with target_sites: "
                    f"missing={missing} extra={extra}"
                )
            geometry_order = np.asarray(
                [source_geometry_sites.index(site) for site in target_sites], dtype=int
            )
        else:
            geometry_order = np.arange(source_geometry_npz["source_geometry"].shape[0], dtype=int)
        source_geometry = np.nan_to_num(
            source_geometry_npz["source_geometry"].astype(np.float32)[geometry_order, ...],
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
    if p_npz is not None:
        physics_sites = (
            p_npz["target_sites"].astype(str).tolist() if "target_sites" in p_npz else []
        )
        if set(physics_sites) != set(target_sites):
            missing = sorted(set(target_sites) - set(physics_sites))
            extra = sorted(set(physics_sites) - set(target_sites))
            raise ValueError(
                "point_centric_physics.npz site mismatch with target_sites: "
                f"missing={missing} extra={extra}"
            )
        order = np.asarray([physics_sites.index(site) for site in target_sites], dtype=int)

        def _physics_vector(key: str) -> np.ndarray:
            if key not in p_npz:
                raise KeyError(f"Missing '{key}' in {p_path}")
            vector = np.asarray(p_npz[key], dtype=np.float32)
            if vector.ndim != 1:
                raise ValueError(f"Expected 1D '{key}' in {p_path}, got shape {vector.shape}")
            if vector.shape[0] != len(physics_sites):
                raise ValueError(
                    f"Length mismatch for '{key}' in {p_path}: {vector.shape[0]} vs {len(physics_sites)}"
                )
            return vector[order]

        local_depth_m = _physics_vector("local_depth_m")
        local_breaking_hs_cap = _physics_vector("local_breaking_hs_cap")
        local_breaking_cap_valid = _physics_vector("local_breaking_cap_valid")
        local_breaking_cap_valid = np.where(
            np.isfinite(local_breaking_cap_valid) & (local_breaking_cap_valid > 0.0),
            1.0,
            0.0,
        ).astype(np.float32, copy=False)

    y_targets: Dict[str, np.ndarray] = {}
    y_physical: Dict[str, np.ndarray] = {}
    y_transfer: Dict[str, np.ndarray] = {}
    y_reference: Dict[str, np.ndarray] = {}
    x_static: Dict[str, np.ndarray] = {}
    x_dynamic_sitewise: Dict[str, np.ndarray] = {}
    for site in target_sites:
        safe = _safe_name(site)
        y_key = f"Y__{safe}"
        y_physical_key = f"Yphysical__{safe}"
        y_transfer_key = f"Ytransfer__{safe}"
        y_reference_key = f"Yreference__{safe}"
        s_key = f"Xstatic__{safe}"
        x_site_key = f"XdynamicSite__{safe}"
        if y_key not in y_npz:
            raise KeyError(f"Missing target key '{y_key}' in {y_path}")
        y_targets[site] = _replace_non_finite_2d(
            y_npz[y_key].astype(np.float32),
            label=f"Y_targets[{site}]",
            ref_rows=train_idx,
        )
        if y_physical_key in y_npz:
            y_physical[site] = _replace_non_finite_2d(
                y_npz[y_physical_key].astype(np.float32),
                label=f"Y_physical[{site}]",
                ref_rows=train_idx,
            )
        if y_transfer_key in y_npz:
            y_transfer[site] = _replace_non_finite_2d(
                y_npz[y_transfer_key].astype(np.float32),
                label=f"Y_transfer[{site}]",
                ref_rows=train_idx,
            )
        if y_reference_key in y_npz:
            y_reference[site] = _replace_non_finite_2d(
                y_npz[y_reference_key].astype(np.float32),
                label=f"Y_reference[{site}]",
                ref_rows=train_idx,
            )
        if s_npz is not None:
            if s_key not in s_npz:
                raise KeyError(f"Missing static key '{s_key}' in {s_path}")
            x_static[site] = s_npz[s_key].astype(np.float32)
        if x_site_npz is not None and x_site_key in x_site_npz:
            x_dynamic_sitewise[site] = _replace_non_finite_2d(
                x_site_npz[x_site_key].astype(np.float32),
                label=f"X_dynamic_sitewise[{site}]",
                ref_rows=train_idx,
            )

    if not target_feature_names:
        raise ValueError(
            "Target feature names are empty. Check targets.mode and target column config."
        )
    if not y_physical:
        scaler_meta = (metadata.get("normalization", {}) or {}).get("target_scaler", {}) or {}
        for site in target_sites:
            legacy = np.asarray(y_targets[site], dtype=np.float32)
            if legacy.shape[1] < 6:
                raise ValueError(
                    "Legacy target artifacts without Yphysical__ keys require 6-column physical targets "
                    f"[hs, tp, dir_sin, dir_cos, dp_sin, dp_cos]; found shape {legacy.shape} for site '{site}'"
                )
            raw = legacy[:, :6]
            if scaler_meta:
                norm_mod = _load_normalization_module()
                try:
                    raw = norm_mod.inverse_transform_targets(
                        raw,
                        scaler_meta,
                        columns=["hs", "tp", "dir_sin", "dir_cos", "dp_sin", "dp_cos"],
                    ).astype(np.float32, copy=False)
                except Exception:
                    raw = raw.astype(np.float32, copy=False)
            dir_angle = np.degrees(np.arctan2(raw[:, 2], raw[:, 3])) % 360.0
            dp_angle = np.degrees(np.arctan2(raw[:, 4], raw[:, 5])) % 360.0
            y_physical[site] = np.column_stack([raw[:, 0], raw[:, 1], dir_angle, dp_angle]).astype(
                np.float32, copy=False
            )

    x_static = _replace_non_finite_static_vectors(x_static)

    if x_bathy is not None:
        missing_bathy_sites = sorted(set(target_sites) - set(bathy_site_to_index))
        if missing_bathy_sites:
            raise ValueError(
                f"Bathymetry NPZ is missing site patch(es) for target sites: {missing_bathy_sites}"
            )
    if x_dynamic_sources is not None:
        if x_dynamic_sources.shape[0] != len(target_sites):
            raise ValueError(
                "X_dynamic_sources first dimension must align with target_sites: "
                f"{x_dynamic_sources.shape[0]} vs {len(target_sites)}"
            )
        if x_dynamic_sources.shape[1] != len(timestamps):
            raise ValueError(
                "X_dynamic_sources time dimension must align with timestamps: "
                f"{x_dynamic_sources.shape[1]} vs {len(timestamps)}"
            )
    if source_geometry is not None and source_geometry.shape[0] != len(target_sites):
        raise ValueError(
            "source_geometry first dimension must align with target_sites: "
            f"{source_geometry.shape[0]} vs {len(target_sites)}"
        )

    return PointCentricArrays(
        x_dynamic=x_dynamic,
        x_dynamic_sources=x_dynamic_sources,
        x_dynamic_sitewise=x_dynamic_sitewise,
        source_geometry=source_geometry,
        y_targets=y_targets,
        x_static=x_static,
        x_bathy=x_bathy,
        local_depth_m=local_depth_m,
        local_breaking_hs_cap=local_breaking_hs_cap,
        local_breaking_cap_valid=local_breaking_cap_valid,
        target_sites=target_sites,
        target_mode=str(target_mode).strip().lower() or "physical",
        dynamic_feature_names=dynamic_feature_names,
        source_feature_names=source_feature_names,
        site_dynamic_feature_names=site_dynamic_feature_names,
        source_geometry_feature_names=source_geometry_feature_names,
        target_feature_names=target_feature_names,
        physical_target_names=list(physical_target_names),
        transfer_target_names=list(transfer_target_names),
        reference_target_names=list(reference_target_names),
        static_feature_names=static_feature_names,
        bathy_channel_names=bathy_channel_names,
        bathy_site_to_index=bathy_site_to_index,
        bathy_patch_size=bathy_patch_size,
        bathy_resolution_m=bathy_resolution_m,
        bathy_normalization_metadata=bathy_normalization_metadata,
        timestamps=timestamps,
        split_idx=split_idx,
        metadata=metadata,
        y_physical=y_physical,
        y_transfer=y_transfer,
        y_reference=y_reference,
    )


def resolve_output_indices(
    target_feature_names: Sequence[str],
    output_columns: Sequence[str],
) -> List[int]:
    """Map configured target column names to indices in `target_feature_names`."""
    indices: List[int] = []
    missing: List[str] = []
    for col in output_columns:
        if col not in target_feature_names:
            missing.append(col)
        else:
            indices.append(target_feature_names.index(col))

    if missing:
        raise ValueError(
            "Missing configured output columns in target features: "
            f"{missing}. Available: {list(target_feature_names)}"
        )
    return indices


def build_bin_centers(
    lower: float, upper: float, num_bins: int, circular: bool = False
) -> np.ndarray:
    """Build evenly spaced bin centers for regression-to-classification targets."""
    if num_bins < 1:
        raise ValueError(f"num_bins must be >= 1, got {num_bins}")
    lower = float(lower)
    upper = float(upper)
    if num_bins == 1:
        return np.array([(lower + upper) * 0.5], dtype=np.float32)
    if circular:
        return np.linspace(lower, upper, num_bins, endpoint=False, dtype=np.float32)
    return np.linspace(lower, upper, num_bins, dtype=np.float32)


def _angle_from_sin_cos(sin_val: float, cos_val: float) -> float:
    angle = float(np.degrees(np.arctan2(sin_val, cos_val)))
    return angle % 360.0


def select_sites(all_sites: Sequence[str], requested_sites: Sequence[str]) -> List[str]:
    """Return requested sites if provided, otherwise all sites."""
    if not requested_sites:
        return list(all_sites)
    req = set(requested_sites)
    found = [s for s in all_sites if s in req]
    missing = sorted(req - set(found))
    if missing:
        raise ValueError(f"Requested sites not found in dataset: {missing}")
    return found


def _ordered_unique_sites(values: Sequence[str]) -> List[str]:
    ordered: List[str] = []
    seen = set()
    for value in values:
        site = str(value)
        if site not in seen:
            ordered.append(site)
            seen.add(site)
    return ordered


def _validate_requested_sites(
    all_sites: Sequence[str],
    requested_sites: Sequence[str],
    config_label: str,
) -> List[str]:
    requested = _ordered_unique_sites(requested_sites)
    missing = sorted(set(requested) - set(str(site) for site in all_sites))
    if missing:
        raise ValueError(f"{config_label} contains site(s) not found in dataset: {missing}")
    return select_sites(all_sites, requested) if requested else []


def _override_split_indices(
    arrays: PointCentricArrays,
    split_name: str,
    indices: np.ndarray,
) -> PointCentricArrays:
    return PointCentricArrays(
        x_dynamic=arrays.x_dynamic,
        x_dynamic_sources=arrays.x_dynamic_sources,
        x_dynamic_sitewise=arrays.x_dynamic_sitewise,
        source_geometry=arrays.source_geometry,
        y_targets=arrays.y_targets,
        y_physical=arrays.y_physical,
        y_transfer=arrays.y_transfer,
        y_reference=arrays.y_reference,
        x_static=arrays.x_static,
        x_bathy=arrays.x_bathy,
        local_depth_m=arrays.local_depth_m,
        local_breaking_hs_cap=arrays.local_breaking_hs_cap,
        local_breaking_cap_valid=arrays.local_breaking_cap_valid,
        target_sites=arrays.target_sites,
        target_mode=arrays.target_mode,
        dynamic_feature_names=arrays.dynamic_feature_names,
        source_feature_names=arrays.source_feature_names,
        site_dynamic_feature_names=arrays.site_dynamic_feature_names,
        source_geometry_feature_names=arrays.source_geometry_feature_names,
        target_feature_names=arrays.target_feature_names,
        physical_target_names=arrays.physical_target_names,
        transfer_target_names=arrays.transfer_target_names,
        reference_target_names=arrays.reference_target_names,
        static_feature_names=arrays.static_feature_names,
        bathy_channel_names=arrays.bathy_channel_names,
        bathy_site_to_index=arrays.bathy_site_to_index,
        bathy_patch_size=arrays.bathy_patch_size,
        bathy_resolution_m=arrays.bathy_resolution_m,
        bathy_normalization_metadata=arrays.bathy_normalization_metadata,
        timestamps=arrays.timestamps,
        split_idx={**arrays.split_idx, split_name: np.asarray(indices, dtype=int)},
        metadata=arrays.metadata,
        ablation_summary=arrays.ablation_summary,
    )


def resolve_site_split_config(
    all_sites: Sequence[str],
    config: dict,
) -> Dict[str, object]:
    """Resolve non-overlapping train, validation, and test site sets.

    When `data.validation_sites` is configured, validation becomes a strict
    site-heldout split and train/val/test site sets are mutually exclusive.
    Otherwise the legacy temporal-validation behavior is preserved.
    """
    _raise_if_legacy_site_holdout_keys_present(config)

    all_sites_list = [str(site) for site in all_sites]
    data_cfg = config.get("data", {}) or {}
    train_site_subsampling_cfg = _resolve_train_site_subsampling_config(data_cfg)

    requested_train_sites = _validate_requested_sites(
        all_sites_list,
        data_cfg.get("train_sites", []) or [],
        "data.train_sites",
    )
    requested_legacy_val_sites = _validate_requested_sites(
        all_sites_list,
        data_cfg.get("val_sites", []) or [],
        "data.val_sites",
    )
    requested_validation_sites = _validate_requested_sites(
        all_sites_list,
        data_cfg.get("validation_sites", []) or [],
        "data.validation_sites",
    )
    requested_test_sites = _validate_requested_sites(
        all_sites_list,
        data_cfg.get("test_sites", []) or [],
        "data.test_sites",
    )
    temporal_holdout_cfg = _resolve_site_holdout_temporal_config(
        config,
        has_validation_sites=bool(requested_validation_sites),
        has_test_sites=bool(requested_test_sites),
    )
    test_site_set = set(requested_test_sites)
    validation_mode = "site-heldout" if requested_validation_sites else "legacy-temporal"
    validation_site_heldout = bool(requested_validation_sites)

    if validation_site_heldout:
        validation_set = set(requested_validation_sites)
        overlap = sorted(validation_set.intersection(test_site_set))
        if overlap:
            raise ValueError(
                "data.validation_sites overlaps data.test_sites: "
                f"{overlap}. Validation and test site sets must be mutually exclusive."
            )

        base_train_sites = requested_train_sites or list(all_sites_list)
        train_sites = [
            site
            for site in base_train_sites
            if site not in validation_set and site not in test_site_set
        ]
        val_sites = list(requested_validation_sites)
        test_sites = list(requested_test_sites)

        if not train_sites:
            raise ValueError(
                "No training sites remain after excluding data.validation_sites and data.test_sites. "
                "Adjust train_sites, validation_sites, or test_sites config."
            )
    else:
        base_train_sites = requested_train_sites or list(all_sites_list)
        train_sites = [site for site in base_train_sites if site not in test_site_set]
        if test_site_set and not train_sites:
            raise ValueError(
                f"No train sites remain after excluding data.test_sites {sorted(test_site_set)}. "
                "Adjust data.train_sites or data.test_sites."
            )

        base_val_sites = requested_legacy_val_sites or list(all_sites_list)
        val_sites = [site for site in base_val_sites if site not in test_site_set]
        if test_site_set and not val_sites:
            raise ValueError(
                f"No val sites remain after excluding data.test_sites {sorted(test_site_set)}. "
                "Adjust data.val_sites or data.test_sites."
            )

        test_sites = list(requested_test_sites)

    train_sites, train_site_subsampling_summary = _apply_train_site_subsampling(
        train_sites,
        train_site_subsampling_cfg,
    )

    return {
        "validation_mode": validation_mode,
        "validation_site_heldout": validation_site_heldout,
        "site_holdout_temporal_mode": str(temporal_holdout_cfg.get("mode", "legacy")),
        "site_holdout_temporal_active": bool(temporal_holdout_cfg.get("active", False)),
        "site_holdout_temporal_recent_fraction": temporal_holdout_cfg.get("recent_fraction", None),
        "site_holdout_temporal_train_fraction": float(
            temporal_holdout_cfg.get("train_fraction", 0.7)
        ),
        "requested_train_sites": list(requested_train_sites),
        "requested_legacy_val_sites": list(requested_legacy_val_sites),
        "requested_validation_sites": list(requested_validation_sites),
        "requested_test_sites": list(requested_test_sites),
        "train_sites": list(train_sites),
        "val_sites": list(val_sites),
        "test_sites": list(test_sites),
        "train_site_subsampling": train_site_subsampling_summary,
    }


def warn_if_legacy_validation_mode(split_info: Dict[str, object]) -> None:
    global _LEGACY_VALIDATION_WARNING_EMITTED
    if bool(split_info.get("validation_site_heldout", False)) or _LEGACY_VALIDATION_WARNING_EMITTED:
        return

    logger.warning(
        "Validation is using legacy temporal behavior because data.validation_sites is empty or missing. "
        "Configure data.validation_sites for unseen-site validation."
    )
    _LEGACY_VALIDATION_WARNING_EMITTED = True


def build_split_endpoints(split_indices: np.ndarray, seq_len: int) -> np.ndarray:
    """Build valid sequence endpoints for a split.

    Endpoint `t` is valid if all indices `[t-seq_len+1, ..., t]` are inside
    `split_indices`.
    """
    if seq_len < 1:
        raise ValueError("seq_len must be >= 1")

    idx = np.asarray(split_indices, dtype=int)
    if idx.size == 0:
        return idx

    idx_set = set(idx.tolist())
    endpoints = [t for t in idx.tolist() if all((t - k) in idx_set for k in range(seq_len))]
    return np.array(endpoints, dtype=int)


def resolve_sequence_length(data_cfg: dict, default: int = 24) -> int:
    """Return the configured sequence-window length."""
    if not isinstance(data_cfg, dict):
        return int(default)

    return int(data_cfg.get("sequence_window", default))


def _resolve_sample_filter_config(data_cfg: dict) -> dict:
    raw = (data_cfg.get("sample_filter", {}) or {}) if isinstance(data_cfg, dict) else {}
    apply_to_splits = [
        str(name).strip().lower() for name in (raw.get("apply_to_splits", ["train"]) or ["train"])
    ]
    apply_to_splits = [name for name in apply_to_splits if name]
    return {
        "enabled": bool(raw.get("enabled", False)),
        "apply_to_splits": apply_to_splits or ["train"],
        "hs_min": None if raw.get("hs_min", None) is None else float(raw.get("hs_min")),
        "tp_min": None if raw.get("tp_min", None) is None else float(raw.get("tp_min")),
        "match": str(raw.get("match", "any")).strip().lower(),
        "statistic": str(raw.get("statistic", "target_timestep")).strip().lower(),
    }


def _resolve_train_sample_subsampling_config(data_cfg: dict) -> dict:
    raw = (data_cfg.get("train_sample_subsampling", {}) or {}) if isinstance(data_cfg, dict) else {}
    cfg = {
        "enabled": bool(raw.get("enabled", False)),
        "fraction": float(raw.get("fraction", 1.0)),
        "seed": int(raw.get("seed", 42)),
        "mode": str(raw.get("mode", "per_site")).strip().lower(),
        "validate_sequence_continuity": bool(raw.get("validate_sequence_continuity", False)),
        "validation_samples": int(raw.get("validation_samples", 1000)),
    }
    if not (0.0 < cfg["fraction"] <= 1.0):
        raise ValueError("data.train_sample_subsampling.fraction must satisfy 0 < fraction <= 1")
    if cfg["mode"] not in {"global", "per_site"}:
        raise ValueError("data.train_sample_subsampling.mode must be one of: global, per_site")
    if cfg["validate_sequence_continuity"] and cfg["validation_samples"] < 1:
        raise ValueError(
            "data.train_sample_subsampling.validation_samples must be >= 1 when "
            "validate_sequence_continuity=true"
        )
    return cfg


def _resolve_train_site_subsampling_config(data_cfg: dict) -> dict:
    raw = (data_cfg.get("train_site_subsampling", {}) or {}) if isinstance(data_cfg, dict) else {}
    cfg = {
        "enabled": bool(raw.get("enabled", False)),
        "fraction": float(raw.get("fraction", 1.0)),
        "seed": int(raw.get("seed", 42)),
    }
    if not (0.0 < cfg["fraction"] <= 1.0):
        raise ValueError("data.train_site_subsampling.fraction must satisfy 0 < fraction <= 1")
    return cfg


def _resolve_site_holdout_temporal_config(
    config: dict,
    *,
    has_validation_sites: bool,
    has_test_sites: bool,
) -> dict:
    split_cfg = (config.get("split", {}) or {}) if isinstance(config, dict) else {}
    mode = str(split_cfg.get("site_holdout_temporal_mode", "legacy")).strip().lower() or "legacy"
    if mode not in _SITE_HOLDOUT_TEMPORAL_MODES:
        raise ValueError(
            "split.site_holdout_temporal_mode must be one of: "
            f"{sorted(_SITE_HOLDOUT_TEMPORAL_MODES)}"
        )

    train_fraction = float(split_cfg.get("train", 0.7))
    val_fraction = float(split_cfg.get("val", 0.15))
    test_fraction = float(split_cfg.get("test", 0.15))
    tolerance = 1e-9

    for name, value in (
        ("split.train", train_fraction),
        ("split.val", val_fraction),
        ("split.test", test_fraction),
    ):
        if not (0.0 <= value <= 1.0):
            raise ValueError(f"{name} must satisfy 0 <= value <= 1")

    active = bool(mode == "shared_recent" and (has_validation_sites or has_test_sites))
    recent_fraction = None
    if active:
        if has_validation_sites and has_test_sites:
            if not np.isclose(val_fraction, test_fraction, atol=tolerance, rtol=0.0):
                raise ValueError(
                    "split.site_holdout_temporal_mode='shared_recent' requires split.val == split.test "
                    "when both data.validation_sites and data.test_sites are configured."
                )
            recent_fraction = val_fraction
        elif has_validation_sites:
            recent_fraction = val_fraction
        else:
            recent_fraction = test_fraction

        expected_train_fraction = 1.0 - float(recent_fraction)
        if not np.isclose(train_fraction, expected_train_fraction, atol=tolerance, rtol=0.0):
            raise ValueError(
                "split.site_holdout_temporal_mode='shared_recent' requires split.train to equal "
                f"1 - recent_fraction ({expected_train_fraction:.12g}). "
                f"Received split.train={train_fraction:.12g}."
            )
    return {
        "mode": mode,
        "active": active,
        "recent_fraction": None if recent_fraction is None else float(recent_fraction),
        "train_fraction": float(train_fraction),
        "val_fraction": float(val_fraction),
        "test_fraction": float(test_fraction),
    }


def _load_target_scaler_metadata(point_centric_dir: str) -> dict | None:
    """Load target scaler metadata from point-centric metadata JSON when available."""
    metadata_path = Path(point_centric_dir) / "point_centric_metadata.json"
    if not metadata_path.exists():
        return None

    try:
        with metadata_path.open("r") as fh:
            payload = json.load(fh) or {}
    except Exception as exc:
        logger.warning("Failed to read %s: %s", metadata_path, exc)
        return None

    norm = payload.get("normalization", {}) or {}
    scaler = norm.get("target_scaler", {}) or {}
    if not scaler:
        return None
    return scaler


def _load_transfer_target_scaler_metadata(point_centric_dir: str) -> dict | None:
    """Load transfer-target scaler metadata from point-centric metadata JSON when available."""
    metadata_path = Path(point_centric_dir) / "point_centric_metadata.json"
    if not metadata_path.exists():
        return None

    try:
        with metadata_path.open("r") as fh:
            payload = json.load(fh) or {}
    except Exception as exc:
        logger.warning("Failed to read %s: %s", metadata_path, exc)
        return None

    norm = payload.get("normalization", {}) or {}
    scaler = norm.get("transfer_target_scaler", {}) or {}
    if not scaler:
        return None
    return scaler


def _is_static_tensor_standardized(arrays: PointCentricArrays) -> bool:
    meta = ((arrays.metadata or {}).get("normalization", {}) or {}).get("static_scaler", {}) or {}
    method = str(meta.get("method", "")).strip().lower()
    return method in {"column_transformer", "standard"}


def _resolve_static_group_indices(
    feature_names: Sequence[str],
    groups_cfg: dict | None,
) -> tuple[dict[str, list[int]], list[str]]:
    names = [str(name) for name in (feature_names or [])]
    groups = dict(groups_cfg or {})
    if not names or not groups:
        return {}, []

    group_indices: dict[str, list[int]] = {}
    warnings: list[str] = []
    for group_name, raw_spec in groups.items():
        spec = dict(raw_spec or {})
        prefixes = [str(item) for item in (spec.get("prefixes", []) or []) if str(item)]
        regexes = [str(item) for item in (spec.get("regexes", []) or []) if str(item)]
        matched: list[int] = []
        for idx, name in enumerate(names):
            lower = name.lower()
            if any(lower.startswith(prefix.lower()) for prefix in prefixes):
                matched.append(idx)
                continue
            if any(re.search(pattern, name) for pattern in regexes):
                matched.append(idx)
        if matched:
            group_indices[str(group_name)] = matched
        else:
            warnings.append(
                f"Static regularization group '{group_name}' matched no static feature columns."
            )
    return group_indices, warnings


class StaticFeatureRegularizationCollate:
    """Train-only collate wrapper for static-feature noise and group dropout."""

    def __init__(
        self,
        *,
        enabled: bool,
        split_name: str,
        static_dim: int,
        feature_names: Sequence[str],
        noise_cfg: dict | None,
        dropout_cfg: dict | None,
        group_indices: dict[str, list[int]] | None,
        static_tensor_standardized: bool,
    ) -> None:
        self.enabled = bool(enabled)
        self.split_name = str(split_name)
        self.static_dim = int(static_dim)
        self.feature_names = [str(name) for name in (feature_names or [])]
        self.noise_cfg = dict(noise_cfg or {})
        self.dropout_cfg = dict(dropout_cfg or {})
        self.group_indices = {
            str(k): [int(v) for v in values] for k, values in (group_indices or {}).items()
        }
        self.static_tensor_standardized = bool(static_tensor_standardized)

    def __call__(self, batch_list):
        batch = default_collate(batch_list)
        if not self.enabled or self.split_name != "train":
            return batch
        if "x_static" not in batch or not torch.is_tensor(batch["x_static"]):
            return batch

        x_static = batch["x_static"]
        if x_static.ndim != 2 or x_static.shape[-1] != self.static_dim:
            return batch

        x_static = x_static.clone()
        x_static = self._apply_noise(x_static)
        x_static = self._apply_group_dropout(x_static)
        batch["x_static"] = x_static

        if (
            "x_dynamic_static_concat" in batch
            and torch.is_tensor(batch["x_dynamic_static_concat"])
            and self.static_dim > 0
        ):
            concat = batch["x_dynamic_static_concat"].clone()
            concat[..., -self.static_dim :] = x_static.unsqueeze(1).expand(-1, concat.shape[1], -1)
            batch["x_dynamic_static_concat"] = concat
        return batch

    def _apply_noise(self, x_static: torch.Tensor) -> torch.Tensor:
        if not bool(self.noise_cfg.get("enabled", False)):
            return x_static
        if bool(self.noise_cfg.get("train_only", True)) and self.split_name != "train":
            return x_static
        std = float(self.noise_cfg.get("std", 0.0))
        if std <= 0.0:
            return x_static
        return x_static + torch.randn_like(x_static) * std

    def _apply_group_dropout(self, x_static: torch.Tensor) -> torch.Tensor:
        if not bool(self.dropout_cfg.get("enabled", False)):
            return x_static
        if bool(self.dropout_cfg.get("train_only", True)) and self.split_name != "train":
            return x_static
        if not self.group_indices:
            return x_static

        p = float(self.dropout_cfg.get("p", 0.0))
        if p <= 0.0:
            return x_static

        replacement = float(self.dropout_cfg.get("replacement_value", 0.0))
        if replacement == 0.0 and not self.static_tensor_standardized:
            logger.warning(
                "Static group dropout replacement_value=0.0 is being applied to a non-standardized static tensor. "
                "Zero may not correspond to the feature mean."
            )

        mode = str(self.dropout_cfg.get("mode", "per_sample")).strip().lower()
        num_samples = int(x_static.shape[0])
        for indices in self.group_indices.values():
            cols = torch.as_tensor(indices, device=x_static.device, dtype=torch.long)
            if mode == "per_batch":
                if bool(torch.rand(1, device=x_static.device) < p):
                    x_static[:, cols] = replacement
            else:
                mask = (torch.rand((num_samples, 1), device=x_static.device) < p).expand(
                    -1, len(indices)
                )
                replacement_tensor = torch.full(
                    (num_samples, len(indices)),
                    replacement,
                    dtype=x_static.dtype,
                    device=x_static.device,
                )
                x_static[:, cols] = torch.where(mask, replacement_tensor, x_static[:, cols])
        return x_static


def _target_scaler_stats_for_column(
    point_centric_dir: str,
    column_name: str,
) -> tuple[float, float] | None:
    """Return (mean, scale) for one target column from scaler metadata."""
    scaler = _load_target_scaler_metadata(point_centric_dir)
    if not scaler:
        return None

    feature_names = list(scaler.get("feature_names", []) or [])
    means = list(scaler.get("mean", []) or [])
    scales = list(scaler.get("scale", []) or [])
    if not feature_names:
        return None
    if len(feature_names) != len(means) or len(feature_names) != len(scales):
        return None

    lut = {str(name): i for i, name in enumerate(feature_names)}
    if column_name not in lut:
        return None

    idx = lut[column_name]
    scale = float(scales[idx])
    if scale == 0.0:
        scale = 1.0
    return float(means[idx]), scale


def _build_storm_weighted_sampler(
    arrays: PointCentricArrays,
    dataset: "PointCentricWindowDataset",
    config: dict,
    output_columns: Sequence[str],
    output_indices: Sequence[int],
    split_name: str,
) -> WeightedRandomSampler | None:
    """Create a train-only WeightedRandomSampler using Hs^2 sample weighting.

    We derive both scaled-space and inverse-transformed physical-space Hs weights,
    log their agreement, and default to physical-space weights when metadata allows.
    """
    if split_name != "train":
        return None

    train_cfg = config.get("training", {}) or {}
    sampler_cfg = train_cfg.get("sampler", {}) or {}
    enabled = bool(sampler_cfg.get("enabled", False))
    if not enabled:
        return None

    strategy = str(sampler_cfg.get("strategy", "hs_squared")).strip().lower()
    if strategy not in {"hs_squared", "hs2", "storm_hs_squared"}:
        raise ValueError(
            "training.sampler.strategy must be one of: hs_squared, hs2, storm_hs_squared"
        )

    point_centric_dir = str((config.get("data", {}) or {}).get("point_centric_dir", ""))
    use_physical_targets = (
        str(getattr(arrays, "target_mode", "physical")).strip().lower() != "physical"
    )
    if use_physical_targets:
        if not arrays.y_physical:
            raise ValueError(
                "Training sampler requires physical targets, but Y_physical is unavailable."
            )
        hs_unscaled = np.empty(len(dataset.samples), dtype=np.float64)
        for i, (site, t) in enumerate(dataset.samples):
            hs_unscaled[i] = float(arrays.y_physical[site][int(t), 0])
        hs_scaled = hs_unscaled.copy()
        has_inverse = True
    else:
        if "hs" not in output_columns:
            raise ValueError(
                "training.sampler.enabled=true requires 'hs' in data.output_columns "
                f"but got {list(output_columns)}"
            )

        hs_out_idx = list(output_columns).index("hs")
        hs_target_idx = int(output_indices[hs_out_idx])
        hs_target_name = str(arrays.target_feature_names[hs_target_idx])

        hs_scaled = np.empty(len(dataset.samples), dtype=np.float64)
        for i, (site, t) in enumerate(dataset.samples):
            hs_scaled[i] = float(arrays.y_targets[site][int(t), hs_target_idx])

        inv_stats = _target_scaler_stats_for_column(point_centric_dir, hs_target_name)
        hs_unscaled = hs_scaled.copy()
        has_inverse = inv_stats is not None
        if inv_stats is not None:
            mean, scale = inv_stats
            hs_unscaled = (hs_scaled * scale) + mean

    min_weight = float(sampler_cfg.get("min_weight", 1.0))
    if min_weight <= 0:
        raise ValueError("training.sampler.min_weight must be > 0")

    w_scaled = np.clip(np.square(hs_scaled), min_weight, None)
    w_unscaled = np.clip(np.square(hs_unscaled), min_weight, None)

    use_unscaled = bool(sampler_cfg.get("use_unscaled_hs", True)) and has_inverse
    weights = w_unscaled if use_unscaled else w_scaled

    min_target_tp_cfg = sampler_cfg.get("min_target_tp", None)
    if min_target_tp_cfg is not None:
        min_target_tp = float(min_target_tp_cfg)
        if use_physical_targets:
            tp_unscaled = np.empty(len(dataset.samples), dtype=np.float64)
            for i, (site, t) in enumerate(dataset.samples):
                tp_unscaled[i] = float(arrays.y_physical[site][int(t), 1])
        else:
            if "tp" not in output_columns:
                raise ValueError(
                    "training.sampler.min_target_tp requires 'tp' in data.output_columns "
                    f"but got {list(output_columns)}"
                )

            tp_out_idx = list(output_columns).index("tp")
            tp_target_idx = int(output_indices[tp_out_idx])
            tp_target_name = str(arrays.target_feature_names[tp_target_idx])

            tp_scaled = np.empty(len(dataset.samples), dtype=np.float64)
            for i, (site, t) in enumerate(dataset.samples):
                tp_scaled[i] = float(arrays.y_targets[site][int(t), tp_target_idx])

            tp_inv_stats = _target_scaler_stats_for_column(point_centric_dir, tp_target_name)
            if tp_inv_stats is None:
                raise ValueError(
                    "training.sampler.min_target_tp requires target scaler metadata for Tp "
                    f"column '{tp_target_name}'"
                )

            tp_mean, tp_scale = tp_inv_stats
            tp_unscaled = (tp_scaled * tp_scale) + tp_mean

        tp_mask = tp_unscaled > min_target_tp
        excluded = int(np.count_nonzero(~tp_mask))
        if excluded > 0:
            weights = np.asarray(weights, dtype=np.float64).copy()
            weights[~tp_mask] = 0.0

        kept = int(np.count_nonzero(tp_mask))
        if kept == 0:
            raise ValueError(
                "training.sampler.min_target_tp filtered all train samples; "
                "lower training.sampler.min_target_tp"
            )
        logger.info(
            "Applied sampler Tp filter | min_target_tp=%.3f | kept=%d | excluded=%d",
            min_target_tp,
            kept,
            excluded,
        )

    compare_signals = bool(sampler_cfg.get("compare_signals", True))
    if compare_signals:
        corr = np.nan
        if np.nanstd(w_scaled) > 0 and np.nanstd(w_unscaled) > 0:
            corr = float(np.corrcoef(w_scaled, w_unscaled)[0, 1])
        med_ratio = float(np.nanmedian(w_unscaled / np.clip(w_scaled, 1e-12, None)))
        logger.info(
            "Sampler diagnostics | hs_signal=%s | corr(unscaled^2,scaled^2)=%.4f | median_ratio=%.4f",
            "unscaled" if use_unscaled else "scaled",
            corr,
            med_ratio,
        )

    replacement = bool(sampler_cfg.get("replacement", True))
    num_samples = int(sampler_cfg.get("num_samples", len(dataset.samples)))
    if num_samples < 1:
        raise ValueError("training.sampler.num_samples must be >= 1")
    if float(np.sum(weights)) <= 0.0:
        raise ValueError(
            "Sampler weights sum to zero; check training.sampler settings (min_weight/min_target_tp)"
        )

    logger.info(
        "Using WeightedRandomSampler for train split | samples=%d | replacement=%s | min=%.4f median=%.4f max=%.4f",
        num_samples,
        replacement,
        float(np.nanmin(weights)),
        float(np.nanmedian(weights)),
        float(np.nanmax(weights)),
    )
    return WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=num_samples,
        replacement=replacement,
    )


def _build_site_balanced_sampler(
    arrays: PointCentricArrays,
    dataset: "PointCentricWindowDataset",
    config: dict,
) -> WeightedRandomSampler | None:
    train_cfg = config.get("training", {}) or {}
    sampler_cfg = train_cfg.get("sampler", {}) or {}
    site_cfg = sampler_cfg.get("site_balanced", {}) or {}
    if str(dataset.split_name).strip().lower() != "train":
        return None
    if not bool(site_cfg.get("train_only", True)):
        return None
    if not dataset.samples:
        return None

    site_to_positions: dict[str, list[int]] = {}
    for idx, (site, _t) in enumerate(dataset.samples):
        site_to_positions.setdefault(str(site), []).append(int(idx))

    active_sites = sorted(site_to_positions)
    missing_sites = sorted(set(dataset.sites) - set(active_sites))
    for site in missing_sites:
        logger.warning(
            "Site-balanced sampler skipped site '%s' because it has zero valid samples.", site
        )

    if not active_sites:
        raise ValueError("Site-balanced sampler found no valid training sites after filtering.")

    min_samples_per_site = max(1, int(site_cfg.get("min_samples_per_site", 1)))
    filtered = {
        site: idxs for site, idxs in site_to_positions.items() if len(idxs) >= min_samples_per_site
    }
    dropped = sorted(set(active_sites) - set(filtered))
    for site in dropped:
        logger.warning(
            "Site-balanced sampler skipped site '%s' because it has only %d sample(s), below min_samples_per_site=%d.",
            site,
            len(site_to_positions[site]),
            min_samples_per_site,
        )
    if not filtered:
        raise ValueError("Site-balanced sampler removed all sites via min_samples_per_site.")

    active_sites = sorted(filtered)
    counts = np.asarray([len(filtered[site]) for site in active_sites], dtype=np.int64)
    weights = np.zeros(len(dataset.samples), dtype=np.float64)
    combine_with_hs_weight = bool(site_cfg.get("combine_with_hs_weight", False))
    normalize_within_site = bool(site_cfg.get("normalize_within_site", True))
    hs_power = float(site_cfg.get("hs_power", 2.0))

    for site in active_sites:
        positions = np.asarray(filtered[site], dtype=np.int64)
        site_mass = np.full(len(positions), 1.0 / max(len(positions), 1), dtype=np.float64)
        if combine_with_hs_weight:
            hs_values = np.asarray(
                [
                    float(arrays.y_physical[site][int(dataset.samples[pos][1]), 0])
                    for pos in positions
                ],
                dtype=np.float64,
            )
            site_mass = np.clip(np.power(np.maximum(hs_values, 0.0), hs_power), 1e-12, None)
            if normalize_within_site:
                site_mass = site_mass / np.clip(np.sum(site_mass), 1e-12, None)
        weights[positions] = site_mass / max(len(active_sites), 1)

    if float(np.sum(weights)) <= 0.0:
        raise ValueError("Site-balanced sampler weights sum to zero.")

    replacement = bool(site_cfg.get("replacement", True))
    num_samples = int(sampler_cfg.get("num_samples", len(dataset.samples)))
    logger.info(
        "Using site-balanced sampler | training_sites=%d min/median/max_samples=%d/%d/%d combine_with_hs_weight=%s normalize_within_site=%s replacement=%s",
        len(active_sites),
        int(np.min(counts)),
        int(np.median(counts)),
        int(np.max(counts)),
        combine_with_hs_weight,
        normalize_within_site,
        replacement,
    )
    return WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=num_samples,
        replacement=replacement,
    )


def _build_training_sampler(
    arrays: PointCentricArrays,
    dataset: "PointCentricWindowDataset",
    config: dict,
    output_columns: Sequence[str],
    output_indices: Sequence[int],
    split_name: str,
) -> WeightedRandomSampler | None:
    if split_name != "train":
        return None

    sampler_cfg = (config.get("training", {}) or {}).get("sampler", {}) or {}
    if not bool(sampler_cfg.get("enabled", False)):
        return None

    strategy = str(sampler_cfg.get("strategy", "hs_squared")).strip().lower()
    if strategy == "site_balanced":
        return _build_site_balanced_sampler(arrays=arrays, dataset=dataset, config=config)
    if strategy in {"hs_squared", "hs2", "storm_hs_squared"}:
        return _build_storm_weighted_sampler(
            arrays=arrays,
            dataset=dataset,
            config=config,
            output_columns=output_columns,
            output_indices=output_indices,
            split_name=split_name,
        )
    raise ValueError(
        "training.sampler.strategy must be one of: hs_squared, hs2, storm_hs_squared, site_balanced"
    )


class PointCentricWindowDataset(Dataset):
    """Dataset over (site, time) samples for coastal-transformer models.

    Each sample contains:
        - `x_dynamic`: [seq_len, dynamic_dim + 2], where the `+2` are
            [`time_sin`, `time_cos`] from cyclical day-of-year encoding
    - `x_static`: [static_dim]
        - `x_dynamic_static_concat` (optional):
            [seq_len, dynamic_dim + 2 + static_dim]
    - `y`: [output_dim]
    """

    def __init__(
        self,
        arrays: PointCentricArrays,
        split_name: str,
        seq_len: int,
        sites: Sequence[str],
        output_indices: Sequence[int],
        target_mode: str = "physical",
        transfer_representation: str = "legacy",
        target_columns: Sequence[str] | None = None,
        target_scaler_meta: dict | None = None,
        transfer_target_scaler_meta: dict | None = None,
        hybrid_task_config: dict | None = None,
        return_concat_dynamic_static: bool = True,
        use_static_features: bool = True,
        use_multi_source: bool = False,
        use_source_geometry: bool = True,
        use_bathymetry: bool = False,
        include_breaking_physics: bool = False,
        bathymetry_shuffle_mode: str = "none",
        bathymetry_train_jitter_cells: int = 0,
        bathymetry_noise_std: float = 0.0,
        random_seed: int = 42,
        sample_filter_cfg: dict | None = None,
    ) -> None:
        if split_name not in arrays.split_idx:
            raise ValueError(
                f"Unknown split '{split_name}'. Expected one of {list(arrays.split_idx.keys())}"
            )

        self.arrays = arrays
        self.split_name = split_name
        self.seq_len = int(seq_len)
        self.sites = list(sites)
        self.output_indices = list(output_indices)
        self.target_mode = str(target_mode).strip().lower()
        self.transfer_representation = str(transfer_representation).strip().lower()
        self.target_columns = list(target_columns or [])
        self.target_scaler_meta = target_scaler_meta or {}
        self.transfer_target_scaler_meta = transfer_target_scaler_meta or {}
        self.hybrid_task_config = hybrid_task_config or {}
        self.return_concat_dynamic_static = bool(return_concat_dynamic_static)
        self.use_static_features = bool(use_static_features)
        self.use_multi_source = bool(use_multi_source)
        self.use_source_geometry = bool(use_source_geometry)
        self.use_bathymetry = bool(use_bathymetry)
        self.include_breaking_physics = bool(include_breaking_physics)
        self.bathymetry_shuffle_mode = str(bathymetry_shuffle_mode or "none").strip().lower()
        self.bathymetry_train_jitter_cells = max(0, int(bathymetry_train_jitter_cells))
        self.bathymetry_noise_std = max(0.0, float(bathymetry_noise_std))
        self.random_seed = int(random_seed)
        self.sample_filter_cfg = sample_filter_cfg or {}
        self.sample_filter_summary = {
            "enabled": bool((self.sample_filter_cfg or {}).get("enabled", False)),
            "applied": False,
            "split": self.split_name,
            "before_count": 0,
            "kept_count": 0,
            "excluded_count": 0,
            "hs_min": None,
            "tp_min": None,
            "match": None,
            "statistic": None,
        }
        self.train_sample_subsampling_summary = {
            "enabled": False,
            "applied": False,
            "mode": "per_site",
            "fraction": 1.0,
            "seed": 42,
            "before_count": 0,
            "after_count": 0,
            "removed_count": 0,
        }

        self.endpoints = build_split_endpoints(arrays.split_idx[split_name], self.seq_len)
        self.samples: List[Tuple[str, int]] = [
            (site, int(t)) for site in self.sites for t in self.endpoints.tolist()
        ]
        self.sample_filter_summary["before_count"] = int(len(self.samples))
        self.sample_filter_summary["kept_count"] = int(len(self.samples))
        self.train_sample_subsampling_summary["before_count"] = int(len(self.samples))
        self.train_sample_subsampling_summary["after_count"] = int(len(self.samples))
        self.time_seasonality = _build_temporal_seasonality(self.arrays.timestamps)
        if self.time_seasonality.shape[0] != self.arrays.x_dynamic.shape[0]:
            raise AssertionError(
                "Timestamp/feature length mismatch for seasonality encoding: "
                f"timestamps={self.time_seasonality.shape[0]} vs X_dynamic_rows={self.arrays.x_dynamic.shape[0]}"
            )

        self._tp_centers = np.asarray(
            self.hybrid_task_config.get("tp_bin_centers", []), dtype=np.float32
        )
        self._dp_centers = np.asarray(
            self.hybrid_task_config.get("dp_bin_centers", []), dtype=np.float32
        )
        self._label_smoothing_sigma = float(
            self.hybrid_task_config.get("label_smoothing_sigma", 0.8)
        )
        self._target_feature_lut = {
            str(name).strip().lower(): i for i, name in enumerate(self.arrays.target_feature_names)
        }
        self._transfer_feature_lut = {
            str(name).strip().lower(): i for i, name in enumerate(self.arrays.transfer_target_names)
        }
        self._physical_feature_lut = {
            str(name).strip().lower(): i for i, name in enumerate(self.arrays.physical_target_names)
        }
        self.site_to_index = {site: idx for idx, site in enumerate(self.arrays.target_sites)}
        self._bathy_shuffle_index: Dict[str, int] = {}
        if self.use_multi_source and self.arrays.x_dynamic_sources is None:
            raise ValueError(
                "data.multi_source.enabled=true requires point_centric_X_dynamic_sources.npz"
            )
        if (
            self.use_multi_source
            and self.use_source_geometry
            and self.arrays.source_geometry is None
        ):
            raise ValueError(
                "Multi-source geometry was requested but point_centric_source_geometry.npz is missing"
            )
        if self.target_mode not in {"physical", "transfer", "physical_and_transfer"}:
            raise ValueError(
                "targets.mode must be one of: physical, transfer, physical_and_transfer; "
                f"got '{self.target_mode}'"
            )
        if self.target_mode != "physical":
            if (
                not self.arrays.y_transfer
                or not self.arrays.y_reference
                or not self.arrays.y_physical
            ):
                raise ValueError(
                    "Transfer target mode requires Y_transfer, Y_reference, and Y_physical artifacts. "
                    "Re-run preprocessing with data.targets.mode set to transfer or physical_and_transfer."
                )
            validate_target_name_block(self.arrays.transfer_target_names, "transfer_target_names")
            validate_target_name_block(self.arrays.reference_target_names, "reference_target_names")
            validate_target_name_block(self.arrays.physical_target_names, "physical_target_names")

        if self.use_bathymetry:
            if self.arrays.x_bathy is None or not self.arrays.bathy_site_to_index:
                raise ValueError(
                    "data.use_bathymetry=true requires point_centric_X_bathy.npz with site-index mapping"
                )
            if self.bathymetry_shuffle_mode not in {"none", "fixed_within_split"}:
                raise ValueError("data.bathymetry_shuffle must be one of: none, fixed_within_split")
            if self.bathymetry_shuffle_mode == "fixed_within_split":
                rng = np.random.default_rng(
                    self.random_seed + sum(ord(ch) for ch in self.split_name)
                )
                perm = rng.permutation(len(self.sites))
                bathy_indices = [self.arrays.bathy_site_to_index[site] for site in self.sites]
                self._bathy_shuffle_index = {
                    site: int(bathy_indices[int(perm[idx])]) for idx, site in enumerate(self.sites)
                }

        self._apply_runtime_sample_filter()

    def _apply_runtime_sample_filter(self) -> None:
        cfg = dict(self.sample_filter_cfg or {})
        if not bool(cfg.get("enabled", False)):
            return

        apply_to_splits = [
            str(name).strip().lower() for name in (cfg.get("apply_to_splits", []) or [])
        ]
        if self.split_name.strip().lower() not in apply_to_splits:
            return

        hs_min = cfg.get("hs_min", None)
        tp_min = cfg.get("tp_min", None)
        if hs_min is None and tp_min is None:
            raise ValueError(
                "data.sample_filter.enabled=true requires at least one of "
                "data.sample_filter.hs_min or data.sample_filter.tp_min"
            )
        if not self.arrays.y_physical:
            raise ValueError(
                "Runtime sample filtering requires physical targets, but Y_physical is unavailable."
            )

        match_mode = str(cfg.get("match", "any")).strip().lower()
        if match_mode not in {"any", "all"}:
            raise ValueError("data.sample_filter.match must be one of: any, all")

        statistic = str(cfg.get("statistic", "target_timestep")).strip().lower()
        if statistic not in {"target_timestep", "max_over_window"}:
            raise ValueError(
                "data.sample_filter.statistic must be one of: target_timestep, max_over_window"
            )

        kept_samples: List[Tuple[str, int]] = []
        excluded_count = 0
        for site, t in self.samples:
            start = int(t) - self.seq_len + 1
            window = np.asarray(
                self.arrays.y_physical[site][start : int(t) + 1, :], dtype=np.float32
            )
            if window.ndim != 2 or window.shape[1] < 2:
                raise ValueError(
                    "Runtime sample filtering expects physical targets with at least [hs, tp] columns, "
                    f"got shape {window.shape} for site '{site}'"
                )

            if statistic == "max_over_window":
                hs_value = float(np.nanmax(window[:, 0]))
                tp_value = float(np.nanmax(window[:, 1]))
            else:
                hs_value = float(window[-1, 0])
                tp_value = float(window[-1, 1])

            checks: List[bool] = []
            if hs_min is not None:
                checks.append(np.isfinite(hs_value) and hs_value >= float(hs_min))
            if tp_min is not None:
                checks.append(np.isfinite(tp_value) and tp_value >= float(tp_min))

            keep = any(checks) if match_mode == "any" else all(checks)
            if keep:
                kept_samples.append((site, int(t)))
            else:
                excluded_count += 1

        if not kept_samples:
            raise ValueError(
                "data.sample_filter removed all samples for split "
                f"'{self.split_name}'. Lower thresholds or disable data.sample_filter."
            )

        self.sample_filter_summary = {
            "enabled": True,
            "applied": True,
            "split": self.split_name,
            "before_count": int(len(self.samples)),
            "kept_count": int(len(kept_samples)),
            "excluded_count": int(excluded_count),
            "hs_min": None if hs_min is None else float(hs_min),
            "tp_min": None if tp_min is None else float(tp_min),
            "match": match_mode,
            "statistic": statistic,
        }
        logger.info(
            "Applied runtime sample filter | split=%s statistic=%s match=%s hs_min=%s tp_min=%s kept=%d excluded=%d",
            self.split_name,
            statistic,
            match_mode,
            "None" if hs_min is None else f"{float(hs_min):.3f}",
            "None" if tp_min is None else f"{float(tp_min):.3f}",
            len(kept_samples),
            excluded_count,
        )
        self.samples = kept_samples

    @staticmethod
    def _shift_patch_zero_pad(patch: np.ndarray, shift_y: int, shift_x: int) -> np.ndarray:
        shifted = np.zeros_like(patch)
        src_y0 = max(0, -int(shift_y))
        src_y1 = (
            min(patch.shape[-2], patch.shape[-2] - int(shift_y))
            if shift_y >= 0
            else patch.shape[-2]
        )
        dst_y0 = max(0, int(shift_y))
        dst_y1 = dst_y0 + max(0, src_y1 - src_y0)

        src_x0 = max(0, -int(shift_x))
        src_x1 = (
            min(patch.shape[-1], patch.shape[-1] - int(shift_x))
            if shift_x >= 0
            else patch.shape[-1]
        )
        dst_x0 = max(0, int(shift_x))
        dst_x1 = dst_x0 + max(0, src_x1 - src_x0)

        if src_y1 > src_y0 and src_x1 > src_x0:
            shifted[..., dst_y0:dst_y1, dst_x0:dst_x1] = patch[..., src_y0:src_y1, src_x0:src_x1]
        return shifted

    def _get_bathy_patch(self, site: str) -> np.ndarray:
        if self.arrays.x_bathy is None:
            raise RuntimeError("Bathymetry patches requested but x_bathy is not loaded")
        site = str(site)
        bathy_index = self._bathy_shuffle_index.get(site, self.arrays.bathy_site_to_index[site])
        patch = np.array(self.arrays.x_bathy[int(bathy_index)], dtype=np.float32, copy=True)
        channel_names = [str(name) for name in (self.arrays.bathy_channel_names or [])]
        depth_stats = ((self.arrays.bathy_normalization_metadata or {}).get("stats", {}) or {}).get(
            "depth", {}
        ) or {}
        depth_is_unit_interval = (
            str(depth_stats.get("method", "")).strip().lower() == "unit_interval_train_max"
        )
        depth_idx = 0
        mask_idx = None
        for idx, name in enumerate(channel_names):
            if name in {"wet_mask", "land_sea_mask"}:
                mask_idx = idx
                break
        if mask_idx is None and patch.shape[0] >= 2:
            mask_idx = 1
        wet_mask = (
            patch[int(mask_idx)]
            if mask_idx is not None
            else np.ones_like(patch[depth_idx], dtype=np.float32)
        )

        if self.split_name == "train" and self.bathymetry_train_jitter_cells > 0:
            shift_y = int(
                np.random.randint(
                    -self.bathymetry_train_jitter_cells, self.bathymetry_train_jitter_cells + 1
                )
            )
            shift_x = int(
                np.random.randint(
                    -self.bathymetry_train_jitter_cells, self.bathymetry_train_jitter_cells + 1
                )
            )
            if shift_y != 0 or shift_x != 0:
                patch = self._shift_patch_zero_pad(patch, shift_y=shift_y, shift_x=shift_x)
                wet_mask = patch[int(mask_idx)] if mask_idx is not None else wet_mask

        if self.split_name == "train" and self.bathymetry_noise_std > 0.0:
            noise = np.random.normal(
                loc=0.0, scale=self.bathymetry_noise_std, size=patch[depth_idx].shape
            ).astype(np.float32)
            patch[depth_idx] = patch[depth_idx] + (noise * wet_mask)

        patch = np.where(np.isfinite(patch), patch, 0.0).astype(np.float32, copy=False)
        if depth_is_unit_interval:
            patch[depth_idx] = np.clip(patch[depth_idx], 0.0, 1.0).astype(np.float32, copy=False)
        if mask_idx is not None:
            patch[int(mask_idx)] = np.clip(patch[int(mask_idx)], 0.0, 1.0).astype(
                np.float32, copy=False
            )
        for idx, name in enumerate(channel_names):
            if name.endswith("_mask"):
                patch[idx] = np.clip(patch[idx], 0.0, 1.0).astype(np.float32, copy=False)
        return patch

    def _inverse_target_columns(self, values: np.ndarray, columns: Sequence[str]) -> np.ndarray:
        scaler_meta = self.target_scaler_meta or {}
        if not scaler_meta:
            return np.asarray(values, dtype=np.float32)
        norm_mod = _load_normalization_module()
        try:
            return norm_mod.inverse_transform_targets(values, scaler_meta, columns=columns).astype(
                np.float32, copy=False
            )
        except Exception:
            return np.asarray(values, dtype=np.float32)

    def generate_soft_targets(
        self,
        value: float,
        centers: np.ndarray,
        sigma: float,
        circular: bool = False,
    ) -> np.ndarray:
        if centers.size == 0:
            raise ValueError("Soft-target generation requires non-empty bin centers")
        sigma = max(float(sigma), 1e-6)
        value = float(value)
        if circular:
            diff = np.abs(value - centers)
            diff = np.minimum(diff, 360.0 - diff)
        else:
            diff = np.abs(value - centers)
        weights = np.exp(-0.5 * np.square(diff / sigma)).astype(np.float32)
        total = float(np.sum(weights))
        if not np.isfinite(total) or total <= 0.0:
            weights = np.zeros_like(weights)
            weights[int(np.argmin(diff))] = 1.0
            return weights
        return (weights / total).astype(np.float32, copy=False)

    def _digitize_value(self, value: float, centers: np.ndarray, circular: bool = False) -> int:
        if centers.size == 0:
            raise ValueError("Digitization requires non-empty bin centers")
        value = float(value)
        if circular:
            diff = np.abs(value - centers)
            diff = np.minimum(diff, 360.0 - diff)
        else:
            diff = np.abs(value - centers)
        return int(np.argmin(diff))

    def _scale_transfer_regression(self, transfer_row: np.ndarray) -> np.ndarray:
        values = np.asarray(transfer_row, dtype=np.float32).reshape(1, -1)
        if not self.transfer_target_scaler_meta:
            return values[0]

        norm_mod = _load_normalization_module()
        scaled = values.copy()
        scaled[:, : len(TRANSFER_SCALER_COLUMNS)] = norm_mod.transform_targets(
            values[:, : len(TRANSFER_SCALER_COLUMNS)],
            self.transfer_target_scaler_meta,
            columns=list(TRANSFER_SCALER_COLUMNS),
        ).astype(np.float32, copy=False)
        return scaled[0]

    def _build_transfer_targets(
        self,
        site: str,
        t: int,
    ) -> Dict[str, torch.Tensor]:
        transfer_row = np.asarray(self.arrays.y_transfer[site][t, :], dtype=np.float32)
        reference_row = np.asarray(self.arrays.y_reference[site][t, :], dtype=np.float32)
        physical_row = np.asarray(self.arrays.y_physical[site][t, :], dtype=np.float32)
        transfer_scaled = self._scale_transfer_regression(transfer_row)
        log_hs_target = (
            float(transfer_row[0])
            if self.transfer_representation == "residual_correction"
            else float(transfer_scaled[0])
        )
        tp_delta_target = (
            float(transfer_row[1])
            if self.transfer_representation == "residual_correction"
            else float(transfer_scaled[1])
        )

        return {
            "transfer": torch.as_tensor(transfer_row, dtype=torch.float32),
            "transfer_scaled": torch.as_tensor(transfer_scaled, dtype=torch.float32),
            "reference": torch.as_tensor(reference_row, dtype=torch.float32),
            "physical": torch.as_tensor(physical_row, dtype=torch.float32),
            "log_hs_ratio": torch.as_tensor(log_hs_target, dtype=torch.float32),
            "tp_delta": torch.as_tensor(tp_delta_target, dtype=torch.float32),
            "dir_delta_deg": torch.as_tensor(float(transfer_row[2]), dtype=torch.float32),
            "dp_delta_deg": torch.as_tensor(float(transfer_row[3]), dtype=torch.float32),
            "ref_hs": torch.as_tensor(float(reference_row[0]), dtype=torch.float32),
            "ref_tp": torch.as_tensor(float(reference_row[1]), dtype=torch.float32),
            "ref_dir": torch.as_tensor(float(reference_row[2]), dtype=torch.float32),
            "ref_dp": torch.as_tensor(float(reference_row[3]), dtype=torch.float32),
            "physical_hs": torch.as_tensor(float(physical_row[0]), dtype=torch.float32),
            "physical_tp": torch.as_tensor(float(physical_row[1]), dtype=torch.float32),
            "physical_dir": torch.as_tensor(float(physical_row[2]), dtype=torch.float32),
            "physical_dp": torch.as_tensor(float(physical_row[3]), dtype=torch.float32),
        }

    def _build_hybrid_targets(self, y_full: np.ndarray) -> Dict[str, torch.Tensor]:
        target_values = np.asarray(y_full, dtype=np.float32)
        if target_values.ndim != 1:
            raise ValueError(f"Expected 1D target row, got shape {target_values.shape}")

        hs_idx = self._target_feature_lut.get("hs")
        tp_idx = self._target_feature_lut.get("tp")
        dir_sin_idx = self._target_feature_lut.get("dir_sin")
        dir_cos_idx = self._target_feature_lut.get("dir_cos")
        dp_sin_idx = self._target_feature_lut.get("dp_sin")
        dp_cos_idx = self._target_feature_lut.get("dp_cos")
        if None in {hs_idx, tp_idx, dir_sin_idx, dir_cos_idx, dp_sin_idx, dp_cos_idx}:
            raise KeyError(
                "Target feature names must include hs, tp, dir_sin, dir_cos, dp_sin and dp_cos"
            )

        raw_targets = target_values[
            [hs_idx, tp_idx, dir_sin_idx, dir_cos_idx, dp_sin_idx, dp_cos_idx]
        ]
        raw_targets = self._inverse_target_columns(
            raw_targets.reshape(1, -1),
            ["hs", "tp", "dir_sin", "dir_cos", "dp_sin", "dp_cos"],
        )[0]

        hs_scaled = float(target_values[hs_idx])
        tp_value = float(raw_targets[1])
        dir_angle = _angle_from_sin_cos(float(raw_targets[2]), float(raw_targets[3]))
        dp_angle = _angle_from_sin_cos(float(raw_targets[4]), float(raw_targets[5]))

        tp_index = self._digitize_value(tp_value, self._tp_centers, circular=False)
        dir_index = self._digitize_value(dir_angle, self._dp_centers, circular=True)
        dp_index = self._digitize_value(dp_angle, self._dp_centers, circular=True)

        tp_soft = self.generate_soft_targets(
            tp_value, self._tp_centers, self._label_smoothing_sigma, circular=False
        )
        dir_soft = self.generate_soft_targets(
            dir_angle, self._dp_centers, self._label_smoothing_sigma, circular=True
        )
        dp_soft = self.generate_soft_targets(
            dp_angle, self._dp_centers, self._label_smoothing_sigma, circular=True
        )

        return {
            "hs": torch.as_tensor(hs_scaled, dtype=torch.float32),
            "tp_index": torch.as_tensor(tp_index, dtype=torch.long),
            "dir_index": torch.as_tensor(dir_index, dtype=torch.long),
            "dp_index": torch.as_tensor(dp_index, dtype=torch.long),
            "tp_soft": torch.as_tensor(tp_soft, dtype=torch.float32),
            "dir_soft": torch.as_tensor(dir_soft, dtype=torch.float32),
            "dp_soft": torch.as_tensor(dp_soft, dtype=torch.float32),
            "tp_value": torch.as_tensor(tp_value, dtype=torch.float32),
            "dir_value": torch.as_tensor(dir_angle, dtype=torch.float32),
            "dp_value": torch.as_tensor(dp_angle, dtype=torch.float32),
        }

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        site, t = self.samples[idx]
        start = t - self.seq_len + 1

        if start < 0:
            raise AssertionError(
                f"Invalid window start for idx={idx}, site={site}: start={start}, t={t}, seq_len={self.seq_len}"
            )

        time_features = self.time_seasonality[start : t + 1, :]
        if time_features.shape[0] != self.seq_len:
            raise AssertionError(
                "Seasonality window length mismatch: "
                f"expected {self.seq_len}, got {time_features.shape[0]} for site={site}, t={t}"
            )
        x_dynamic = None
        x_dynamic_sources = None
        if self.use_multi_source:
            assert self.arrays.x_dynamic_sources is not None
            site_idx = int(self.site_to_index[site])
            x_dynamic_sources_raw = self.arrays.x_dynamic_sources[site_idx, start : t + 1, :, :]
            if x_dynamic_sources_raw.shape[0] != self.seq_len:
                raise AssertionError(
                    "Dynamic source window length mismatch: "
                    f"expected {self.seq_len}, got {x_dynamic_sources_raw.shape[0]} for site={site}, t={t}"
                )
            time_broadcast = np.repeat(
                time_features[:, None, :], repeats=x_dynamic_sources_raw.shape[1], axis=1
            )
            x_dynamic_sources = np.concatenate(
                [x_dynamic_sources_raw, time_broadcast], axis=-1
            ).astype(np.float32, copy=False)
        else:
            x_dynamic_raw = self.arrays.x_dynamic[start : t + 1, :]
            if x_dynamic_raw.shape[0] != self.seq_len:
                raise AssertionError(
                    "Dynamic window length mismatch: "
                    f"expected {self.seq_len}, got {x_dynamic_raw.shape[0]} for site={site}, t={t}"
                )

            site_dynamic = self.arrays.x_dynamic_sitewise.get(site)
            if site_dynamic is not None and site_dynamic.size > 0:
                site_dynamic_window = site_dynamic[start : t + 1, :]
                if site_dynamic_window.shape[0] != self.seq_len:
                    raise AssertionError(
                        "Sitewise dynamic window length mismatch: "
                        f"expected {self.seq_len}, got {site_dynamic_window.shape[0]} for site={site}, t={t}"
                    )
                dynamic_core = np.concatenate([x_dynamic_raw, site_dynamic_window], axis=1)
            else:
                dynamic_core = x_dynamic_raw

            # Append bounded cyclical time-of-year channels at each timestep.
            x_dynamic = np.concatenate([dynamic_core, time_features], axis=1).astype(
                np.float32, copy=False
            )

        # Strict phase-alignment audit: y(t) must correspond to the final
        # timestep of the dynamic input window X_dynamic[t-seq_len+1 : t+1].
        last_dynamic_idx = start + self.seq_len - 1
        if int(last_dynamic_idx) != int(t):
            raise AssertionError(
                "Temporal misalignment detected: "
                f"target index={t}, last dynamic index={last_dynamic_idx}, site={site}"
            )

        target_timestamp = str(self.arrays.timestamps[t])
        last_dynamic_timestamp = str(self.arrays.timestamps[last_dynamic_idx])
        if target_timestamp != last_dynamic_timestamp:
            raise AssertionError(
                "Timestamp misalignment detected between y and X_dynamic[-1]: "
                f"site={site}, target={target_timestamp}, dynamic_last={last_dynamic_timestamp}"
            )

        x_static = self.arrays.x_static.get(site)
        has_static = (
            self.use_static_features and x_static is not None and np.asarray(x_static).size > 0
        )
        y_full = self.arrays.y_targets[site][t, :]
        if self.target_mode != "physical":
            y = self._build_transfer_targets(site=site, t=int(t))
        elif self.hybrid_task_config:
            y = self._build_hybrid_targets(y_full)
        else:
            y = y_full[self.output_indices]

        sample = {
            "y": y if isinstance(y, dict) else torch.as_tensor(y, dtype=torch.float32),
            "site": site,
            "site_name": site,
            "site_index": int(self.site_to_index[site]),
            "time_index": t,
            "timestamp": str(self.arrays.timestamps[t]),
        }
        if has_static:
            sample["x_static"] = torch.as_tensor(x_static, dtype=torch.float32)
        site_idx = int(self.site_to_index[site])
        if self.include_breaking_physics and self.arrays.local_breaking_hs_cap is not None:
            sample["local_breaking_hs_cap"] = torch.as_tensor(
                float(self.arrays.local_breaking_hs_cap[site_idx]),
                dtype=torch.float32,
            )
        if self.include_breaking_physics and self.arrays.local_breaking_cap_valid is not None:
            sample["local_breaking_cap_valid"] = torch.as_tensor(
                float(self.arrays.local_breaking_cap_valid[site_idx]),
                dtype=torch.float32,
            )
        if self.use_multi_source:
            assert x_dynamic_sources is not None
            sample["x_dynamic_sources"] = torch.as_tensor(x_dynamic_sources, dtype=torch.float32)
            if self.use_source_geometry:
                assert self.arrays.source_geometry is not None
                sample["source_geometry"] = torch.as_tensor(
                    self.arrays.source_geometry[int(self.site_to_index[site])],
                    dtype=torch.float32,
                )
        else:
            assert x_dynamic is not None
            sample["x_dynamic"] = torch.as_tensor(x_dynamic, dtype=torch.float32)

        if self.use_bathymetry:
            sample["x_bathy"] = torch.as_tensor(self._get_bathy_patch(site), dtype=torch.float32)

        if self.return_concat_dynamic_static and not self.use_multi_source and has_static:
            assert x_dynamic is not None
            assert x_static is not None
            static_tiled = np.repeat(x_static.reshape(1, -1), repeats=self.seq_len, axis=0)
            sample["x_dynamic_static_concat"] = torch.as_tensor(
                np.concatenate([x_dynamic, static_tiled], axis=1),
                dtype=torch.float32,
            )

        return sample


def _infer_expected_timestamp_spacing(timestamps: np.ndarray) -> np.timedelta64 | None:
    try:
        ts64 = np.asarray(timestamps).astype("datetime64[ns]")
    except Exception:
        return None
    if ts64.ndim != 1 or ts64.size < 2 or np.isnat(ts64).any():
        return None

    diffs = np.diff(ts64).astype("timedelta64[ns]").astype(np.int64)
    positive_diffs = diffs[diffs > 0]
    if positive_diffs.size == 0:
        return None

    unique_diffs, counts = np.unique(positive_diffs, return_counts=True)
    best_idx = int(np.argmax(counts))
    return np.timedelta64(int(unique_diffs[best_idx]), "ns")


def _raise_sequence_continuity_error(
    *,
    site: str,
    target_timestep: int,
    seq_len: int,
    actual_window_length: int,
    gap_location: str,
) -> None:
    raise ValueError(
        "Train sample subsampling sequence continuity validation failed: "
        f"site={site} target_timestep={target_timestep} expected_window_length={seq_len} "
        f"actual_window_length={actual_window_length} gap_location={gap_location}"
    )


def _validate_train_sample_sequence_continuity(
    dataset: PointCentricWindowDataset,
    subsampling_cfg: dict,
) -> None:
    if not bool(subsampling_cfg.get("validate_sequence_continuity", False)):
        return
    if not dataset.samples:
        return

    validation_samples = int(subsampling_cfg.get("validation_samples", 1000))
    sample_count = len(dataset.samples)
    if validation_samples >= sample_count:
        sample_positions = np.arange(sample_count, dtype=int)
    else:
        rng = np.random.default_rng(int(subsampling_cfg.get("seed", 42)) + 1)
        sample_positions = np.sort(
            rng.choice(sample_count, size=validation_samples, replace=False).astype(int)
        )

    expected_spacing = _infer_expected_timestamp_spacing(dataset.arrays.timestamps)
    timestamps_ns: np.ndarray | None = None
    expected_spacing_ns: int | None = None
    if expected_spacing is not None:
        try:
            timestamps_ns = np.asarray(dataset.arrays.timestamps).astype("datetime64[ns]")
            expected_spacing_ns = int(expected_spacing / np.timedelta64(1, "ns"))
        except Exception:
            timestamps_ns = None
            expected_spacing_ns = None

    for pos in sample_positions.tolist():
        site, target_timestep = dataset.samples[int(pos)]
        start = int(target_timestep) - dataset.seq_len + 1
        if start < 0:
            _raise_sequence_continuity_error(
                site=site,
                target_timestep=int(target_timestep),
                seq_len=int(dataset.seq_len),
                actual_window_length=max(0, int(target_timestep) + 1),
                gap_location=f"window_start_negative(start={start})",
            )

        window_indices = np.arange(start, int(target_timestep) + 1, dtype=int)
        if int(window_indices.size) != int(dataset.seq_len):
            _raise_sequence_continuity_error(
                site=site,
                target_timestep=int(target_timestep),
                seq_len=int(dataset.seq_len),
                actual_window_length=int(window_indices.size),
                gap_location="window_index_length_mismatch",
            )

        index_diffs = np.diff(window_indices)
        if index_diffs.size > 0 and not np.all(index_diffs == 1):
            gap_idx = int(np.flatnonzero(index_diffs != 1)[0])
            _raise_sequence_continuity_error(
                site=site,
                target_timestep=int(target_timestep),
                seq_len=int(dataset.seq_len),
                actual_window_length=int(window_indices.size),
                gap_location=(
                    f"index_gap_between_positions={gap_idx}->{gap_idx + 1} "
                    f"indices={int(window_indices[gap_idx])}->{int(window_indices[gap_idx + 1])}"
                ),
            )

        if timestamps_ns is not None and expected_spacing_ns is not None:
            window_timestamps = timestamps_ns[start : int(target_timestep) + 1]
            if int(window_timestamps.size) != int(dataset.seq_len):
                _raise_sequence_continuity_error(
                    site=site,
                    target_timestep=int(target_timestep),
                    seq_len=int(dataset.seq_len),
                    actual_window_length=int(window_timestamps.size),
                    gap_location="timestamp_window_length_mismatch",
                )
            timestamp_diffs = np.diff(window_timestamps).astype("timedelta64[ns]").astype(np.int64)
            if timestamp_diffs.size > 0 and not np.all(timestamp_diffs == expected_spacing_ns):
                gap_idx = int(np.flatnonzero(timestamp_diffs != expected_spacing_ns)[0])
                actual_spacing = int(timestamp_diffs[gap_idx])
                _raise_sequence_continuity_error(
                    site=site,
                    target_timestep=int(target_timestep),
                    seq_len=int(dataset.seq_len),
                    actual_window_length=int(window_timestamps.size),
                    gap_location=(
                        f"timestamp_gap_between_positions={gap_idx}->{gap_idx + 1} "
                        f"timestamps={str(window_timestamps[gap_idx])}->{str(window_timestamps[gap_idx + 1])} "
                        f"expected_spacing_ns={expected_spacing_ns} actual_spacing_ns={actual_spacing}"
                    ),
                )

        try:
            sample = dataset[int(pos)]
        except Exception as exc:
            _raise_sequence_continuity_error(
                site=site,
                target_timestep=int(target_timestep),
                seq_len=int(dataset.seq_len),
                actual_window_length=0,
                gap_location=f"dataset_getitem_error({type(exc).__name__}: {exc})",
            )

        if "x_dynamic" in sample:
            actual_seq_len = int(sample["x_dynamic"].shape[0])
        elif "x_dynamic_sources" in sample:
            actual_seq_len = int(sample["x_dynamic_sources"].shape[0])
        else:
            actual_seq_len = 0
        if actual_seq_len != int(dataset.seq_len):
            _raise_sequence_continuity_error(
                site=site,
                target_timestep=int(target_timestep),
                seq_len=int(dataset.seq_len),
                actual_window_length=actual_seq_len,
                gap_location="dataset_returned_sequence_length_mismatch",
            )
        if int(sample.get("time_index", -1)) != int(target_timestep):
            _raise_sequence_continuity_error(
                site=site,
                target_timestep=int(target_timestep),
                seq_len=int(dataset.seq_len),
                actual_window_length=actual_seq_len,
                gap_location=(
                    f"target_alignment_mismatch(returned_time_index={int(sample.get('time_index', -1))})"
                ),
            )

    logger.info(
        "Validated train sample subsampling sequence continuity | checked=%d seq_len=%d",
        int(len(sample_positions)),
        int(dataset.seq_len),
    )


def _apply_train_sample_subsampling(
    dataset: PointCentricWindowDataset,
    subsampling_cfg: dict,
) -> None:
    summary = {
        "enabled": bool(subsampling_cfg.get("enabled", False)),
        "applied": False,
        "mode": str(subsampling_cfg.get("mode", "per_site")),
        "fraction": float(subsampling_cfg.get("fraction", 1.0)),
        "seed": int(subsampling_cfg.get("seed", 42)),
        "before_count": int(len(dataset.samples)),
        "after_count": int(len(dataset.samples)),
        "removed_count": 0,
    }
    dataset.train_sample_subsampling_summary = summary

    if not summary["enabled"] or summary["fraction"] >= 1.0:
        return
    if str(dataset.split_name).strip().lower() != "train":
        return
    if not dataset.samples:
        raise ValueError(
            "Train sample subsampling is enabled but the training dataset has zero eligible samples "
            "before subsampling."
        )

    rng = np.random.default_rng(summary["seed"])
    sample_positions = np.arange(len(dataset.samples), dtype=int)
    selected_positions: np.ndarray

    if summary["mode"] == "global":
        keep_count = int(np.floor(len(dataset.samples) * summary["fraction"]))
        if keep_count < 1:
            raise ValueError(
                "Train sample subsampling removed all training samples. "
                f"mode=global fraction={summary['fraction']} before={len(dataset.samples)} after=0"
            )
        selected_positions = np.sort(
            rng.choice(sample_positions, size=keep_count, replace=False).astype(int)
        )
    else:
        site_to_positions: dict[str, list[int]] = defaultdict(list)
        for pos, (site, _target_timestep) in enumerate(dataset.samples):
            site_to_positions[str(site)].append(int(pos))

        selected_chunks: list[np.ndarray] = []
        before_counts: list[int] = []
        after_counts: list[int] = []
        for site in sorted(site_to_positions):
            positions = np.asarray(site_to_positions[site], dtype=int)
            before_count = int(positions.size)
            keep_count = max(1, int(np.floor(before_count * summary["fraction"])))
            keep_count = min(keep_count, before_count)
            before_counts.append(before_count)
            after_counts.append(keep_count)
            if keep_count == before_count:
                selected_chunks.append(positions)
            else:
                selected_chunks.append(
                    np.sort(rng.choice(positions, size=keep_count, replace=False).astype(int))
                )

        selected_positions = (
            np.sort(np.concatenate(selected_chunks)) if selected_chunks else np.array([], dtype=int)
        )
        summary.update(
            {
                "train_sites": int(len(site_to_positions)),
                "min_before": int(min(before_counts)) if before_counts else 0,
                "max_before": int(max(before_counts)) if before_counts else 0,
                "min_after": int(min(after_counts)) if after_counts else 0,
                "max_after": int(max(after_counts)) if after_counts else 0,
            }
        )

    if selected_positions.size < 1:
        raise ValueError(
            "Train sample subsampling removed all training samples. "
            f"mode={summary['mode']} fraction={summary['fraction']} before={len(dataset.samples)} after=0"
        )

    dataset.samples = [dataset.samples[int(pos)] for pos in selected_positions.tolist()]
    summary["applied"] = True
    summary["after_count"] = int(len(dataset.samples))
    summary["removed_count"] = int(summary["before_count"] - summary["after_count"])
    dataset.train_sample_subsampling_summary = summary

    logger.info(
        "Train sample subsampling: enabled=%s mode=%s fraction=%.6f seed=%d before=%d after=%d removed=%d",
        summary["enabled"],
        summary["mode"],
        summary["fraction"],
        summary["seed"],
        summary["before_count"],
        summary["after_count"],
        summary["removed_count"],
    )
    if summary["mode"] == "per_site":
        logger.info(
            "Train sample subsampling per-site: train_sites=%d min_before=%d max_before=%d min_after=%d max_after=%d",
            int(summary.get("train_sites", 0)),
            int(summary.get("min_before", 0)),
            int(summary.get("max_before", 0)),
            int(summary.get("min_after", 0)),
            int(summary.get("max_after", 0)),
        )

    _validate_train_sample_sequence_continuity(dataset, subsampling_cfg)


def build_split_dataloader(
    arrays: PointCentricArrays,
    config: dict,
    split_name: str,
    shuffle: bool,
    *,
    apply_train_sample_subsampling: bool = False,
) -> Tuple[DataLoader, PointCentricWindowDataset]:
    """Build a windowed PyTorch loader for one named data split.

    The loader applies only runtime operations, such as train-only static
    regularization, sampling, and bathymetry augmentation.  It does not alter
    the saved prepared arrays.
    """
    config = resolve_config(config)
    data_cfg = config.get("data", {})
    train_cfg = config.get("training", {})

    seq_len = resolve_sequence_length(data_cfg, default=24)
    return_concat_dynamic_static = bool(data_cfg.get("return_concat_dynamic_static", True))
    use_bathymetry = bool(data_cfg.get("use_bathymetry", False))
    requested_multi_source = bool((data_cfg.get("multi_source", {}) or {}).get("enabled", False))
    requested_source_geometry = _resolve_runtime_source_geometry_features_flag(config)
    bathymetry_shuffle_mode = str(data_cfg.get("bathymetry_shuffle", "none"))
    bathymetry_train_jitter_cells = int(data_cfg.get("bathymetry_train_jitter_cells", 0))
    bathymetry_noise_std = float(data_cfg.get("bathymetry_noise_std", 0.0))
    random_seed = int(train_cfg.get("seed", 42))
    targets_cfg = resolve_targets_config(data_cfg)
    breaking_cfg = (config.get("physics", {}) or {}).get("breaking", {}) or {}
    static_reg_cfg = data_cfg.get("static_regularization", {}) or {}
    sample_filter_cfg = _resolve_sample_filter_config(data_cfg)
    train_sample_subsampling_cfg = _resolve_train_sample_subsampling_config(data_cfg)
    target_mode = str(targets_cfg.get("mode", "physical"))
    transfer_representation = str(targets_cfg.get("transfer_representation", "legacy"))
    target_columns = data_cfg.get(
        "target_columns", data_cfg.get("output_columns", ["hs", "tp", "dir", "dp"])
    )
    output_columns = data_cfg.get(
        "output_columns",
        ["hs", "tp", "dir_sin", "dir_cos", "dp_sin", "dp_cos"],
    )
    output_indices = (
        resolve_output_indices(arrays.target_feature_names, output_columns)
        if target_mode == "physical"
        else []
    )

    point_centric_dir = str(data_cfg.get("point_centric_dir", ""))
    target_scaler_meta = (
        _load_target_scaler_metadata(point_centric_dir) if point_centric_dir else None
    )
    transfer_target_scaler_meta = (
        _load_transfer_target_scaler_metadata(point_centric_dir) if point_centric_dir else None
    )

    loss_cfg = train_cfg.get("loss", {}) or {}
    hybrid_cfg = loss_cfg.get("blueprint_hybrid", {}) or {}
    num_tp_bins = int(hybrid_cfg.get("num_tp_bins", 32))
    num_dp_bins = int(hybrid_cfg.get("num_dp_bins", 36))
    tp_bin_range = tuple(hybrid_cfg.get("tp_bin_range", [0.0, 25.0]))
    dp_bin_range = tuple(hybrid_cfg.get("dp_bin_range", [0.0, 360.0]))
    loss_type = str(loss_cfg.get("type", train_cfg.get("loss_type", "mse"))).strip().lower()
    if target_mode == "physical" and loss_type in {
        "blueprint_hybrid",
        "hybrid_blueprint",
        "hybrid_multitask",
    }:
        hybrid_task_config = {
            "tp_bin_centers": build_bin_centers(
                tp_bin_range[0], tp_bin_range[1], num_tp_bins, circular=False
            ),
            "dp_bin_centers": build_bin_centers(
                dp_bin_range[0], dp_bin_range[1], num_dp_bins, circular=True
            ),
            "label_smoothing_sigma": float(
                loss_cfg.get("label_smoothing_sigma", hybrid_cfg.get("label_smoothing_sigma", 0.8))
            ),
        }
    else:
        hybrid_task_config = {}

    split_info = resolve_site_split_config(arrays.target_sites, config)
    if split_name == "val":
        warn_if_legacy_validation_mode(split_info)
    temporal_holdout_active = bool(split_info.get("site_holdout_temporal_active", False))

    if bool(split_info.get("validation_site_heldout", False)):
        selected_sites = list(split_info.get(f"{split_name}_sites", []) or [])
        if temporal_holdout_active:
            selected_idx = np.asarray(
                arrays.split_idx.get(split_name, np.array([], dtype=int)), dtype=int
            )
            arrays_for_ds = _override_split_indices(arrays, split_name, selected_idx)
        else:
            full_idx = np.arange(len(arrays.timestamps), dtype=int)
            arrays_for_ds = _override_split_indices(arrays, split_name, full_idx)
    elif split_name == "test":
        selected_sites = list(split_info.get("test_sites", []) or [])
        if temporal_holdout_active:
            selected_idx = np.asarray(
                arrays.split_idx.get("test", np.array([], dtype=int)), dtype=int
            )
            arrays_for_ds = _override_split_indices(arrays, "test", selected_idx)
        else:
            full_idx = np.arange(len(arrays.timestamps), dtype=int)
            arrays_for_ds = _override_split_indices(arrays, "test", full_idx)
    else:
        selected_sites = list(split_info.get(f"{split_name}_sites", []) or [])
        arrays_for_ds = arrays
        test_sites = list(split_info.get("test_sites", []) or [])
        if split_name == "val" and test_sites and not temporal_holdout_active:
            val_idx = np.asarray(arrays.split_idx.get("val", np.array([], dtype=int)), dtype=int)
            test_idx = np.asarray(arrays.split_idx.get("test", np.array([], dtype=int)), dtype=int)
            combined = np.unique(np.concatenate([val_idx, test_idx]))
            arrays_for_ds = _override_split_indices(arrays, "val", combined)

    if requested_multi_source:
        if arrays_for_ds.x_dynamic_sources is None:
            raise ValueError(
                "data.multi_source.enabled=true requires multi-source point-centric artifacts. "
                "Re-run preprocessing with multi_source.enabled=true."
            )
        if requested_source_geometry and arrays_for_ds.source_geometry is None:
            raise ValueError(
                "Geometry features were requested for multi-source runtime, but point_centric_source_geometry.npz "
                "is missing. Re-run preprocessing with geometry enabled or disable geometry at runtime."
            )
        arrays_for_ds.metadata = {
            **(arrays_for_ds.metadata or {}),
            "multi_source": {
                **((arrays_for_ds.metadata or {}).get("multi_source", {}) or {}),
                "enabled": True,
            },
        }
    if target_mode != "physical" and not transfer_target_scaler_meta:
        raise ValueError(
            "Transfer target mode requires normalization.transfer_target_scaler in point_centric_metadata.json. "
            "Re-run preprocessing with data.targets.mode set to transfer or physical_and_transfer."
        )
    if use_bathymetry:
        if arrays_for_ds.x_bathy is None:
            raise ValueError("data.use_bathymetry=true requires point_centric_X_bathy.npz")
        bathy_cfg = (
            ((config.get("model", {}) or {}).get("coastal_transformer", {}) or {}).get("bathy", {})
        ) or {}
        expected_in_channels = int(bathy_cfg.get("in_channels", arrays_for_ds.x_bathy.shape[1]))
        expected_patch_size = int(bathy_cfg.get("patch_size", arrays_for_ds.x_bathy.shape[-1]))
        actual_channels = int(arrays_for_ds.x_bathy.shape[1])
        actual_patch_size = int(arrays_for_ds.x_bathy.shape[-1])
        if actual_channels != expected_in_channels:
            raise ValueError(
                "Bathymetry config/data channel mismatch: "
                f"model.coastal_transformer.bathy.in_channels={expected_in_channels} "
                f"but point_centric_X_bathy.npz contains {actual_channels} channel(s) "
                f"({arrays_for_ds.bathy_channel_names})."
            )
        if actual_patch_size != expected_patch_size:
            raise ValueError(
                "Bathymetry config/data patch-size mismatch: "
                f"model.coastal_transformer.bathy.patch_size={expected_patch_size} "
                f"but point_centric_X_bathy.npz contains patch_size={actual_patch_size}."
            )

    dataset = PointCentricWindowDataset(
        arrays=arrays_for_ds,
        split_name=split_name,
        seq_len=seq_len,
        sites=selected_sites,
        output_indices=output_indices,
        target_mode=target_mode,
        transfer_representation=transfer_representation,
        target_columns=target_columns,
        target_scaler_meta=target_scaler_meta,
        transfer_target_scaler_meta=transfer_target_scaler_meta,
        hybrid_task_config=hybrid_task_config,
        return_concat_dynamic_static=return_concat_dynamic_static,
        use_static_features=bool(data_cfg.get("use_static_features", True)),
        use_multi_source=requested_multi_source,
        use_source_geometry=requested_source_geometry,
        use_bathymetry=use_bathymetry,
        include_breaking_physics=bool(breaking_cfg.get("enabled", False)),
        bathymetry_shuffle_mode=bathymetry_shuffle_mode,
        bathymetry_train_jitter_cells=bathymetry_train_jitter_cells,
        bathymetry_noise_std=bathymetry_noise_std,
        random_seed=random_seed,
        sample_filter_cfg=sample_filter_cfg,
    )

    if apply_train_sample_subsampling and split_name == "train":
        _apply_train_sample_subsampling(dataset, train_sample_subsampling_cfg)

    sampler = _build_training_sampler(
        arrays=arrays_for_ds,
        dataset=dataset,
        config=config,
        output_columns=output_columns,
        output_indices=output_indices,
        split_name=split_name,
    )
    use_shuffle = bool(shuffle and sampler is None)
    if sampler is not None and shuffle:
        logger.info("Disabled DataLoader shuffle because WeightedRandomSampler is active.")

    static_group_indices: dict[str, list[int]] = {}
    static_group_warnings: list[str] = []
    if bool(static_reg_cfg.get("enabled", False)) and split_name == "train":
        dropout_cfg = static_reg_cfg.get("group_dropout", {}) or {}
        if dataset.arrays.static_feature_names:
            static_group_indices, static_group_warnings = _resolve_static_group_indices(
                dataset.arrays.static_feature_names,
                dropout_cfg.get("groups", {}) or {},
            )
        else:
            static_group_warnings.append(
                "Static group dropout requested, but static feature names are unavailable; skipping grouped dropout."
            )
        for warning in static_group_warnings:
            logger.warning(warning)
        logger.info(
            "Static regularization | enabled=%s noise_std=%.4f group_dropout_p=%.4f groups=%s",
            bool(static_reg_cfg.get("enabled", False)),
            float(((static_reg_cfg.get("noise", {}) or {}).get("std", 0.0))),
            float(((static_reg_cfg.get("group_dropout", {}) or {}).get("p", 0.0))),
            {name: len(indices) for name, indices in static_group_indices.items()},
        )

    collate_fn = StaticFeatureRegularizationCollate(
        enabled=bool(static_reg_cfg.get("enabled", False)),
        split_name=split_name,
        static_dim=len(dataset.arrays.static_feature_names),
        feature_names=dataset.arrays.static_feature_names,
        noise_cfg=static_reg_cfg.get("noise", {}) or {},
        dropout_cfg=static_reg_cfg.get("group_dropout", {}) or {},
        group_indices=static_group_indices,
        static_tensor_standardized=_is_static_tensor_standardized(dataset.arrays),
    )

    loader = DataLoader(
        dataset,
        batch_size=int(train_cfg.get("batch_size", 64)),
        shuffle=use_shuffle,
        sampler=sampler,
        num_workers=int(data_cfg.get("num_workers", 0)),
        pin_memory=bool(data_cfg.get("pin_memory", False)),
        drop_last=False,
        collate_fn=collate_fn,
    )
    return loader, dataset
