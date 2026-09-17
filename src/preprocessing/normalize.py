#!/usr/bin/env python3
"""Fit, serialize, and apply the transformations used by preprocessing.

Dynamic features, static geometry, target values, and circular directions use
different transformations.  This module owns those rules and their metadata
serialization.  Fitting belongs to preprocessing; training and evaluation
only restore and apply the stored state.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import yaml

try:
    import pandas as pd

    _HAS_PANDAS = True
except Exception:
    _HAS_PANDAS = False

try:
    from sklearn.base import BaseEstimator, TransformerMixin
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import (
        FunctionTransformer,
        MinMaxScaler,
        RobustScaler,
        StandardScaler,
    )

    _HAS_SKLEARN = True
except Exception:
    _HAS_SKLEARN = False

try:
    import torch

    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False


RAY_SECTORS: Tuple[str, ...] = (
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
POROSITY_COLUMNS: Tuple[str, ...] = (
    "static_porosity_500m",
    "static_porosity_1km",
    "static_porosity_2km",
    "static_porosity_5km",
    "static_porosity_10km",
)
FETCH_ANISOTROPY_COLUMNS: Tuple[str, ...] = (
    "fetch_max_over_mean",
    "fetch_std_m",
    "fetch_cv",
    "fetch_directional_entropy",
    "fetch_resultant_length",
    "open_sector_fraction",
    "closed_sector_fraction",
    "open_sector_width_deg",
    "dominant_fetch_direction_sin",
    "dominant_fetch_direction_cos",
)
FJORDNESS_COLUMNS: Tuple[str, ...] = (
    "fjordness_land_blocking_component",
    "fjordness_low_porosity_component",
    "fjordness_closed_sector_component",
    "fjordness_anisotropy_component",
    "fjordness_low_fetch_component",
    "fjordness_route_complexity_component",
    "fjordness_score",
    "site_regime_open",
    "site_regime_transition",
    "site_regime_fjord",
)
PATH_RATIO_COLUMNS: Tuple[str, ...] = (
    "path_tortuosity_ratio",
    "bottleneck_to_path_ratio",
    "bottleneck_to_fetch_ratio",
    "funneling_log",
)

STATIC_CYCLICAL_COLUMNS: Tuple[str, ...] = (
    "static_final_approach_deg",
    "static_net_deflection_deg",
    "static_nearest_shore_normal_deg",
    "static_signed_curvature_deg",
)

DYNAMIC_PREBOUNDED_COLUMNS: Tuple[str, ...] = (
    "time_sin",
    "time_cos",
)

STATIC_HEAVY_TAILED_COLUMNS: Tuple[str, ...] = (
    "path_length_m",
    "path_bottleneck_m",
    "path_direct_distance_m",
    *(f"ray_fetch_{sector}_m" for sector in RAY_SECTORS),
)

STATIC_SPATIAL_DERIVATIVE_COLUMNS: Tuple[str, ...] = (
    "static_local_depth_m",
    "static_nearest_shore_steepness",
    *(f"ray_max_slope_{sector}" for sector in RAY_SECTORS),
    *(f"ray_max_laplacian_{sector}" for sector in RAY_SECTORS),
    *(f"ray_min_depth_{sector}_m" for sector in RAY_SECTORS),
)

STATIC_RATIO_COLUMNS: Tuple[str, ...] = (
    "funneling_ratio",
    "choke_out_ratio",
    *POROSITY_COLUMNS,
    *FETCH_ANISOTROPY_COLUMNS,
    *FJORDNESS_COLUMNS,
    *PATH_RATIO_COLUMNS,
)

STATIC_NON_FEATURE_COLUMNS: Tuple[str, ...] = (
    "site_name",
    "site_lat",
    "site_lon",
    "site_x",
    "site_y",
    "site_row",
    "site_col",
)


@dataclass
class StaticTransformerArtifacts:
    """Fitted static-feature transformer and metadata."""

    transformer: ColumnTransformer
    raw_feature_columns: List[str]
    transformed_feature_columns: List[str]
    groups: Dict[str, List[str]]
    train_sites: List[str]
    feature_range: Tuple[float, float]
    raw_to_transformed_feature_map: Dict[str, List[str]]

    def to_metadata(self) -> dict:
        metadata = {
            "method": "column_transformer",
            "raw_feature_columns": list(self.raw_feature_columns),
            "transformed_feature_columns": list(self.transformed_feature_columns),
            "groups": {k: list(v) for k, v in self.groups.items()},
            "train_sites": list(self.train_sites),
            "feature_range": [float(self.feature_range[0]), float(self.feature_range[1])],
            "raw_to_transformed_feature_map": {
                k: list(v) for k, v in self.raw_to_transformed_feature_map.items()
            },
        }
        # Persist fitted ColumnTransformer state so preprocessing can be
        # restored without refitting against inference data.  The arrays are
        # deliberately JSON-native and retain sklearn's exact conventions.
        fitted = {}
        for name, transformer, columns in self.transformer.transformers_:
            if transformer == "drop" or transformer is None:
                continue
            pipe = transformer
            steps = getattr(pipe, "named_steps", {})
            group = {"columns": [str(c) for c in columns]}
            for step_name, step in steps.items():
                state = {}
                for attr in (
                    "statistics_",
                    "scale_",
                    "min_",
                    "data_min_",
                    "data_max_",
                    "center_",
                    "quantile_range",
                ):
                    value = getattr(step, attr, None)
                    if value is not None:
                        state[attr] = (
                            np.asarray(value).tolist() if isinstance(value, np.ndarray) else value
                        )
                if state:
                    state["class"] = step.__class__.__name__
                    group[step_name] = state
            fitted[str(name)] = group
        metadata["fitted_transformers"] = fitted
        metadata["serialization_version"] = 1
        return metadata


def restore_static_transformer_artifacts(
    metadata: Mapping[str, object],
) -> StaticTransformerArtifacts:
    """Restore a fitted static transformer from :meth:`to_metadata` output."""
    _require_sklearn()
    raw = [str(x) for x in metadata.get("raw_feature_columns", [])]
    groups = {str(k): [str(x) for x in v] for k, v in (metadata.get("groups", {}) or {}).items()}
    feature_range = tuple(
        float(x) for x in (metadata.get("feature_range", [0.0, 1.0]) or [0.0, 1.0])
    )
    transformer, _ = build_master_static_column_transformer(
        raw, feature_range=feature_range, strict_required=False
    )
    # Initialise ColumnTransformer bookkeeping (transformers_, output indices)
    # with a harmless row; all learned values are replaced from metadata below.
    if _HAS_PANDAS:
        transformer.fit(pd.DataFrame([{name: 0.0 for name in raw}]))
    else:
        raise ImportError("pandas is required to restore static transformer metadata")
    fitted = metadata.get("fitted_transformers", {}) or {}
    for name, pipe, _columns in transformer.transformers_:
        state = fitted.get(str(name), {})
        for step_name, step_state in state.items():
            if step_name == "columns" or step_name not in getattr(pipe, "named_steps", {}):
                continue
            step = pipe.named_steps[step_name]
            for attr, value in (step_state or {}).items():
                if attr == "class":
                    continue
                setattr(step, attr, np.asarray(value) if isinstance(value, list) else value)
        # Cyclical transformer only needs its input-width marker.
        cycle = getattr(pipe, "named_steps", {}).get("cycle")
        if cycle is not None and not hasattr(cycle, "n_input_features_"):
            cycle.fit(np.zeros((1, len(groups.get("cyclical", [])))))
    return StaticTransformerArtifacts(
        transformer=transformer,
        raw_feature_columns=raw,
        transformed_feature_columns=[
            str(x) for x in metadata.get("transformed_feature_columns", [])
        ],
        groups=groups,
        train_sites=[str(x) for x in metadata.get("train_sites", [])],
        feature_range=feature_range,
        raw_to_transformed_feature_map={
            str(k): [str(x) for x in v]
            for k, v in (metadata.get("raw_to_transformed_feature_map", {}) or {}).items()
        },
    )


# Backward-compatible generic normalization helpers


def load_config(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    with p.open() as fh:
        return yaml.safe_load(fh) or {}


def _to_list(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (list, tuple)):
        return [_to_list(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _to_list(v) for k, v in obj.items()}
    return obj


def load_input(path: str):
    p = Path(path)
    if p.suffix.lower() == ".csv":
        if not _HAS_PANDAS:
            arr = np.loadtxt(str(p), delimiter=",", skiprows=1)
            return arr, {"format": "csv_nopandas", "path": str(p)}
        df = pd.read_csv(str(p))
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        return df, {"format": "csv", "numeric_columns": numeric_cols, "path": str(p)}
    if p.suffix.lower() == ".npy":
        arr = np.load(str(p), allow_pickle=True)
        return arr, {"format": "npy", "path": str(p)}
    if p.suffix.lower() == ".npz":
        npz = np.load(str(p), allow_pickle=True)
        key = None
        for k in ("arr_0", "x", "z", "data"):
            if k in npz.files:
                key = k
                break
        if key is None:
            key = npz.files[0]
        return npz[key], {"format": "npz", "array_key": key, "path": str(p)}
    raise ValueError(f"Unsupported input extension: {p.suffix}")


def save_output(data, meta, out_path: str) -> None:
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if _HAS_PANDAS and isinstance(data, (pd.DataFrame,)):
        data.to_csv(str(p), index=False)
        return
    if p.suffix.lower() == ".csv":
        if _HAS_PANDAS and isinstance(data, np.ndarray):
            pd.DataFrame(data).to_csv(str(p), index=False)
            return
        np.savetxt(str(p), data, delimiter=",")
        return
    if p.suffix.lower() == ".npy":
        np.save(str(p), data)
        return
    if p.suffix.lower() == ".npz":
        np.savez_compressed(str(p), arr=data)
        return
    np.save(str(p.with_suffix(".npy")), data)


def compute_stats(
    arr,
    method: str,
    feature_range: Tuple[float, float] = (0.0, 1.0),
    quantile_low: float = 25,
    quantile_high: float = 75,
) -> dict:
    if hasattr(arr, "values") and _HAS_PANDAS:
        values = arr.values.astype(float)
    else:
        values = np.asarray(arr, dtype=float)
    if values.ndim == 1:
        values = values.reshape(-1, 1)

    if method == "zscore":
        mean = np.nanmean(values, axis=0)
        std = np.nanstd(values, axis=0)
        std[std == 0] = 1.0
        return {"mean": mean, "std": std}

    if method == "minmax":
        a, b = feature_range
        minv = np.nanmin(values, axis=0)
        maxv = np.nanmax(values, axis=0)
        rng = maxv - minv
        rng[rng == 0] = 1.0
        return {"min": minv, "max": maxv, "feature_range": (a, b)}

    if method == "robust":
        ql = np.nanpercentile(values, quantile_low, axis=0)
        qh = np.nanpercentile(values, quantile_high, axis=0)
        med = np.nanmedian(values, axis=0)
        iqr = qh - ql
        iqr[iqr == 0] = 1.0
        return {"median": med, "iqr": iqr}

    if method == "global_maxabs":
        m = np.nanmax(np.abs(values))
        if m == 0:
            m = 1.0
        return {"maxabs": float(m)}

    if method == "samplewise_l2":
        return {}

    raise ValueError(f"Unsupported normalization method: {method}")


def compute_stats_stream(
    chunks, method: str, feature_range: Tuple[float, float] = (0.0, 1.0)
) -> dict:
    """Compute scaler statistics from an iterable of bounded 2-D chunks."""
    count = None
    mean = None
    m2 = None
    minimum = maximum = None
    for chunk in chunks:
        values = np.asarray(chunk, dtype=float).reshape(-1, np.asarray(chunk).shape[-1])
        if count is None:
            n_features = values.shape[1]
            count = np.zeros(n_features, dtype=np.int64)
            mean = np.zeros(n_features, dtype=np.float64)
            m2 = np.zeros(n_features, dtype=np.float64)
            minimum = np.full(n_features, np.inf)
            maximum = np.full(n_features, -np.inf)
        for col in range(values.shape[1]):
            vals = values[:, col]
            vals = vals[np.isfinite(vals)]
            if not len(vals):
                continue
            n = len(vals)
            avg = float(vals.mean())
            ss = float(np.dot(vals - avg, vals - avg))
            old = int(count[col])
            total = old + n
            if old:
                delta = avg - mean[col]
                m2[col] += ss + delta * delta * old * n / total
                mean[col] += delta * n / total
            else:
                mean[col] = avg
                m2[col] = ss
            count[col] = total
            minimum[col] = min(minimum[col], float(vals.min()))
            maximum[col] = max(maximum[col], float(vals.max()))
    if count is None:
        # Preserve compute_stats' historical all-NaN behaviour for optional
        # source channels; downstream masks/fill handling remains unchanged.
        return (
            {"mean": np.array([np.nan]), "std": np.array([np.nan])}
            if method == "zscore"
            else {
                "min": np.array([np.nan]),
                "max": np.array([np.nan]),
                "feature_range": tuple(feature_range),
            }
        )
    empty = count == 0
    if method == "zscore":
        std = np.sqrt(m2 / count)
        std[std == 0] = 1.0
        mean[empty] = np.nan
        std[empty] = np.nan
        return {"mean": mean, "std": std}
    if method == "minmax":
        rng = maximum - minimum
        rng[rng == 0] = 1.0
        minimum[empty] = np.nan
        maximum[empty] = np.nan
        return {"min": minimum, "max": maximum, "feature_range": tuple(feature_range)}
    raise ValueError(
        "Streaming recovery supports zscore and minmax; retain existing robust artifacts"
    )


def apply_normalization(arr, stats: dict, method: str):
    if hasattr(arr, "values") and _HAS_PANDAS:
        df = arr.copy()
        X = df.values.astype(float)
        is_df = True
    else:
        X = np.asarray(arr, dtype=float)
        is_df = False

    if X.ndim == 1:
        X = X.reshape(-1, 1)

    if method == "zscore":
        mean = np.asarray(stats["mean"], dtype=float)
        std = np.asarray(stats["std"], dtype=float)
        std[std == 0] = 1.0
        X = (X - mean) / std
    elif method == "minmax":
        minv = np.asarray(stats["min"], dtype=float)
        maxv = np.asarray(stats["max"], dtype=float)
        a, b = stats.get("feature_range", (0.0, 1.0))
        denom = maxv - minv
        denom[denom == 0] = 1.0
        X = (X - minv) / denom
        X = X * (b - a) + a
    elif method == "robust":
        med = np.asarray(stats["median"], dtype=float)
        iqr = np.asarray(stats["iqr"], dtype=float)
        iqr[iqr == 0] = 1.0
        X = (X - med) / iqr
    elif method == "global_maxabs":
        m = float(stats["maxabs"])
        if m == 0:
            m = 1.0
        X = X / m
    elif method == "samplewise_l2":
        norms = np.linalg.norm(X, ord=2, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        X = X / norms
    else:
        raise ValueError(f"Unsupported normalization method: {method}")

    if is_df:
        df.iloc[:, :] = X
        return df
    return X


def save_stats(stats: dict, out_path: str) -> None:
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as fh:
        json.dump(_to_list(stats), fh, indent=2)


# New strict ColumnTransformer logic for master static features


def _require_sklearn() -> None:
    if not _HAS_SKLEARN:
        raise ImportError(
            "scikit-learn is required for strict ColumnTransformer normalization. "
            "Install it with `pip install scikit-learn`."
        )


def _ensure_pandas_dataframe(df) -> "pd.DataFrame":
    if not _HAS_PANDAS:
        raise ImportError("pandas is required for static feature normalization")
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pandas DataFrame, got {type(df)}")
    return df


def _available(columns: Sequence[str], candidates: Iterable[str]) -> List[str]:
    colset = set(columns)
    return [c for c in candidates if c in colset]


def _missing(columns: Sequence[str], required: Iterable[str]) -> List[str]:
    colset = set(columns)
    return sorted([c for c in required if c not in colset])


class CyclicalDegreesTransformer(BaseEstimator, TransformerMixin):
    """Convert degree columns to paired sin/cos channels."""

    def fit(self, X, y=None):
        arr = np.asarray(X, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        self.n_input_features_ = int(arr.shape[1])
        return self

    def transform(self, X):
        arr = np.asarray(X, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        radians = np.deg2rad(arr)
        return np.concatenate([np.sin(radians), np.cos(radians)], axis=1)

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            n_features = int(getattr(self, "n_input_features_", 0))
            input_features = [f"x{i}" for i in range(n_features)]
        out = []
        for name in input_features:
            out.append(f"{name}_sin")
            out.append(f"{name}_cos")
        return np.asarray(out, dtype=object)


def encode_cyclical_columns(
    df: "pd.DataFrame",
    columns: Sequence[str],
    *,
    input_degrees: bool = True,
    drop_original: bool = True,
) -> "pd.DataFrame":
    """Apply sine/cosine transform to angular columns in a DataFrame."""
    _ensure_pandas_dataframe(df)
    out = df.copy()
    for col in columns:
        if col not in out.columns:
            continue
        vals = pd.to_numeric(out[col], errors="coerce").astype(float)
        radians = np.deg2rad(vals) if input_degrees else vals
        out[f"{col}_sin"] = np.sin(radians)
        out[f"{col}_cos"] = np.cos(radians)
        if drop_original:
            out = out.drop(columns=[col])
    return out


def infer_dynamic_direction_columns(columns: Sequence[str]) -> List[str]:
    """Infer dynamic directional columns from naming patterns.

    This helper is intentionally broad to capture wind/wave direction variables
    from offshore feeds before sin/cos encoding.
    """
    detected: List[str] = []
    tokens = ("dir", "direction", "bearing", "heading", "thq")
    for col in columns:
        low = str(col).lower()
        if low.endswith("_sin") or low.endswith("_cos"):
            continue
        if any(tok in low for tok in tokens):
            detected.append(col)
    return detected


def select_dynamic_columns_for_scaling(columns: Sequence[str]) -> List[str]:
    """Return dynamic columns that should be passed to magnitude scalers.

    Mathematical rationale:
    Cyclical encodings live on the unit circle by construction,
    i.e. ``sin(phi), cos(phi) in [-1, 1]``. Re-scaling these channels would
    distort the geometry that preserves phase periodicity. Therefore, all
    `*_sin`/`*_cos` columns (including `time_sin`/`time_cos`) are excluded.
    """
    selected: List[str] = []
    prebounded = {c.lower() for c in DYNAMIC_PREBOUNDED_COLUMNS}

    for col in columns:
        name = str(col)
        low = name.lower()
        if low.endswith("_sin") or low.endswith("_cos"):
            continue
        if low in prebounded:
            continue
        selected.append(name)

    return selected


def _coerce_boolean_to_float(df: "pd.DataFrame") -> "pd.DataFrame":
    out = df.copy()
    for col in out.columns:
        if str(out[col].dtype) == "bool":
            out[col] = out[col].astype(float)
    return out


def select_static_feature_columns(master_df: "pd.DataFrame") -> List[str]:
    """Return numeric site-level static columns used as model features."""
    _ensure_pandas_dataframe(master_df)
    work = _coerce_boolean_to_float(master_df)
    numeric_cols = work.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in numeric_cols if c not in STATIC_NON_FEATURE_COLUMNS]


def _validate_static_group_assignments(
    feature_columns: Sequence[str],
    groups: Mapping[str, Sequence[str]],
) -> None:
    """Fail fast on duplicate raw feature names or overlapping group assignments."""
    columns = [str(col) for col in feature_columns]
    seen_columns: set[str] = set()
    duplicate_columns: list[str] = []
    for col in columns:
        if col in seen_columns and col not in duplicate_columns:
            duplicate_columns.append(col)
        seen_columns.add(col)
    if duplicate_columns:
        raise ValueError(f"Duplicate static feature columns detected: {sorted(duplicate_columns)}")

    assigned_owner: Dict[str, str] = {}
    overlaps: list[str] = []
    for group_name, group_columns in groups.items():
        for raw_name in group_columns:
            raw_key = str(raw_name)
            prior_owner = assigned_owner.get(raw_key)
            if prior_owner is not None and prior_owner != str(group_name):
                overlaps.append(f"{raw_key} ({prior_owner}, {group_name})")
                continue
            assigned_owner[raw_key] = str(group_name)
    if overlaps:
        raise ValueError(
            f"Static normalization assigned raw features to multiple groups: {sorted(overlaps)}"
        )


def build_master_static_feature_groups(
    feature_columns: Sequence[str],
    *,
    strict_required: bool,
    intentionally_missing_columns: Sequence[str] | None = None,
) -> Dict[str, List[str]]:
    """Resolve static feature groups used by the strict ColumnTransformer."""
    columns = [str(col) for col in feature_columns]
    intentionally_missing = {
        str(col) for col in (intentionally_missing_columns or []) if str(col).strip()
    }

    cyclical_cols = _available(columns, STATIC_CYCLICAL_COLUMNS)
    heavy_cols = _available(columns, STATIC_HEAVY_TAILED_COLUMNS)
    derivative_cols = _available(columns, STATIC_SPATIAL_DERIVATIVE_COLUMNS)
    ratio_cols = _available(columns, STATIC_RATIO_COLUMNS)

    if strict_required:
        required_columns = {
            *STATIC_CYCLICAL_COLUMNS,
            *STATIC_HEAVY_TAILED_COLUMNS,
            *STATIC_SPATIAL_DERIVATIVE_COLUMNS,
            *STATIC_RATIO_COLUMNS,
        }
        ignored_missing = sorted(
            col for col in intentionally_missing if col in required_columns and col not in columns
        )
        if ignored_missing:
            logging.info(
                "Static normalization: ignoring %d intentionally ablated columns.",
                len(ignored_missing),
            )
        missing_required = {
            "cyclical": [
                col
                for col in _missing(columns, STATIC_CYCLICAL_COLUMNS)
                if col not in intentionally_missing
            ],
            "heavy_tailed": [
                col
                for col in _missing(columns, STATIC_HEAVY_TAILED_COLUMNS)
                if col not in intentionally_missing
            ],
            "spatial_derivatives": [
                col
                for col in _missing(columns, STATIC_SPATIAL_DERIVATIVE_COLUMNS)
                if col not in intentionally_missing
            ],
            "ratios": [
                col
                for col in _missing(columns, STATIC_RATIO_COLUMNS)
                if col not in intentionally_missing
            ],
        }
        missing_required = {k: v for k, v in missing_required.items() if v}
        if missing_required:
            raise ValueError(
                "master_static_features.csv is missing required normalization columns: "
                f"{missing_required}"
            )

    grouped = set(cyclical_cols) | set(heavy_cols) | set(derivative_cols) | set(ratio_cols)
    other_numeric = [c for c in columns if c not in grouped]

    groups = {
        "cyclical": cyclical_cols,
        "heavy_tailed": heavy_cols,
        "spatial_derivatives": derivative_cols,
        "ratios": ratio_cols,
        "other_numeric": other_numeric,
    }
    _validate_static_group_assignments(columns, groups)
    return groups


def build_master_static_column_transformer(
    feature_columns: Sequence[str],
    *,
    feature_range: Tuple[float, float] = (0.0, 1.0),
    strict_required: bool = True,
    intentionally_missing_columns: Sequence[str] | None = None,
) -> Tuple[ColumnTransformer, Dict[str, List[str]]]:
    """Build a strict sklearn ColumnTransformer for static coastal geometry."""
    _require_sklearn()

    groups = build_master_static_feature_groups(
        feature_columns,
        strict_required=strict_required,
        intentionally_missing_columns=intentionally_missing_columns,
    )

    transformers = []

    if groups["cyclical"]:
        transformers.append(
            (
                "cyclical",
                Pipeline(
                    steps=[
                        ("cycle", CyclicalDegreesTransformer()),
                        ("impute", SimpleImputer(strategy="constant", fill_value=0.0)),
                    ]
                ),
                groups["cyclical"],
            )
        )

    if groups["heavy_tailed"]:
        transformers.append(
            (
                "heavy_tailed",
                Pipeline(
                    steps=[
                        ("impute", SimpleImputer(strategy="median")),
                        ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
                        ("scale", MinMaxScaler(feature_range=feature_range)),
                    ]
                ),
                groups["heavy_tailed"],
            )
        )

    if groups["spatial_derivatives"]:
        transformers.append(
            (
                "spatial_derivatives",
                Pipeline(
                    steps=[
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", RobustScaler()),
                    ]
                ),
                groups["spatial_derivatives"],
            )
        )

    if groups["ratios"]:
        transformers.append(
            (
                "ratios",
                Pipeline(
                    steps=[
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", MinMaxScaler(feature_range=feature_range)),
                    ]
                ),
                groups["ratios"],
            )
        )

    if groups["other_numeric"]:
        transformers.append(
            (
                "other_numeric",
                Pipeline(
                    steps=[
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", MinMaxScaler(feature_range=feature_range)),
                    ]
                ),
                groups["other_numeric"],
            )
        )

    if not transformers:
        raise ValueError("No static feature columns available for ColumnTransformer")

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0.0,
        verbose_feature_names_out=False,
    )
    return preprocessor, groups


def build_raw_to_transformed_feature_map(
    groups: Mapping[str, Sequence[str]],
) -> Dict[str, List[str]]:
    """Map raw static columns to transformed feature names deterministically."""
    mapping: Dict[str, List[str]] = {}
    cyclical = set(groups.get("cyclical", []))
    for raw_name in cyclical:
        mapping[str(raw_name)] = [f"{raw_name}_sin", f"{raw_name}_cos"]

    passthrough_groups = ("heavy_tailed", "spatial_derivatives", "ratios", "other_numeric")
    for group_name in passthrough_groups:
        for raw_name in groups.get(group_name, []):
            mapping[str(raw_name)] = [str(raw_name)]
    return mapping


def fit_master_static_feature_transformer(
    master_df: "pd.DataFrame",
    train_sites: Sequence[str],
    *,
    site_column: str = "site_name",
    feature_range: Tuple[float, float] = (0.0, 1.0),
    strict_required: bool = False,
    intentionally_missing_columns: Sequence[str] | None = None,
) -> StaticTransformerArtifacts:
    """Fit static-feature ColumnTransformer strictly on training sites only."""
    _require_sklearn()
    work = _coerce_boolean_to_float(_ensure_pandas_dataframe(master_df))

    if site_column not in work.columns:
        raise KeyError(f"Expected '{site_column}' in master static DataFrame")

    train_sites_set = {str(s) for s in train_sites}
    train_mask = work[site_column].astype(str).isin(train_sites_set)
    train_df = work.loc[train_mask].copy()
    if train_df.empty:
        raise ValueError("No rows left to fit static scaler after training-site filtering")

    raw_feature_columns = select_static_feature_columns(work)
    transformer, groups = build_master_static_column_transformer(
        raw_feature_columns,
        feature_range=feature_range,
        strict_required=strict_required,
        intentionally_missing_columns=intentionally_missing_columns,
    )

    transformer.fit(train_df[raw_feature_columns])
    transformed_columns = transformer.get_feature_names_out(raw_feature_columns).tolist()
    raw_to_transformed_feature_map = build_raw_to_transformed_feature_map(groups)

    return StaticTransformerArtifacts(
        transformer=transformer,
        raw_feature_columns=list(raw_feature_columns),
        transformed_feature_columns=list(transformed_columns),
        groups=groups,
        train_sites=sorted(train_sites_set),
        feature_range=(float(feature_range[0]), float(feature_range[1])),
        raw_to_transformed_feature_map=raw_to_transformed_feature_map,
    )


def transform_master_static_features(
    master_df: "pd.DataFrame",
    artifacts: StaticTransformerArtifacts,
    *,
    site_column: str = "site_name",
) -> "pd.DataFrame":
    """Apply fitted static transformer and return transformed frame with site ids."""
    work = _coerce_boolean_to_float(_ensure_pandas_dataframe(master_df))
    if site_column not in work.columns:
        raise KeyError(f"Expected '{site_column}' in master static DataFrame")

    transformed = artifacts.transformer.transform(work[artifacts.raw_feature_columns])
    out = pd.DataFrame(
        transformed,
        columns=artifacts.transformed_feature_columns,
        index=work.index,
    )
    out.insert(0, site_column, work[site_column].astype(str).values)
    return out


# Target StandardScaler utilities with inverse-transform support


def _sanitize_matrix_with_fill(
    matrix: np.ndarray,
    fill_values: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Replace non-finite values in matrix with column medians."""
    arr = np.asarray(matrix, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D matrix, got shape {arr.shape}")

    if fill_values is None:
        fill = np.nanmedian(arr, axis=0)
        fill = np.asarray(fill, dtype=float)
        fill[~np.isfinite(fill)] = 0.0
    else:
        fill = np.asarray(fill_values, dtype=float)
        if fill.shape[0] != arr.shape[1]:
            raise ValueError("fill_values length does not match matrix columns")

    bad = ~np.isfinite(arr)
    if np.any(bad):
        rows, cols = np.where(bad)
        arr[rows, cols] = fill[cols]
    return arr, fill


def fit_target_standard_scaler(
    y_targets_by_site: Mapping[str, "pd.DataFrame"],
    train_idx: np.ndarray,
    train_sites: Sequence[str],
    feature_names: Sequence[str],
) -> dict:
    """Fit target StandardScaler from train split only and return serializable metadata."""
    _require_sklearn()

    train_rows: List[np.ndarray] = []
    idx = np.asarray(train_idx, dtype=int)
    cols = list(feature_names)
    if len(cols) == 0:
        raise ValueError("feature_names for target scaler cannot be empty")

    for site in train_sites:
        if site not in y_targets_by_site:
            continue
        site_df = y_targets_by_site[site]
        site_df = _ensure_pandas_dataframe(site_df)
        missing_cols = [c for c in cols if c not in site_df.columns]
        if missing_cols:
            raise KeyError(f"Target columns missing for site '{site}': {missing_cols}")
        train_rows.append(site_df.iloc[idx][cols].to_numpy(dtype=float))

    if not train_rows:
        raise ValueError("No target rows available for target scaler fit")

    fit_matrix = np.vstack(train_rows)
    fit_matrix, fill_values = _sanitize_matrix_with_fill(fit_matrix)

    scaler = StandardScaler()
    scaler.fit(fit_matrix)

    return {
        "method": "standard",
        "feature_names": cols,
        "fill_values": fill_values.tolist(),
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "var": scaler.var_.tolist(),
    }


def _align_target_scaler_vectors(
    scaler_meta: Mapping[str, object],
    columns: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    feature_names = list(scaler_meta.get("feature_names", []) or [])
    if not feature_names:
        raise ValueError("target scaler metadata missing feature_names")

    mean = np.asarray(scaler_meta.get("mean", []), dtype=float)
    scale = np.asarray(scaler_meta.get("scale", []), dtype=float)
    fill_values = np.asarray(scaler_meta.get("fill_values", []), dtype=float)
    if not (len(feature_names) == len(mean) == len(scale) == len(fill_values)):
        raise ValueError("Inconsistent lengths in target scaler metadata")

    pos = {name: i for i, name in enumerate(feature_names)}
    missing = [c for c in columns if c not in pos]
    if missing:
        raise KeyError(f"Columns missing from target scaler metadata: {missing}")

    order = [pos[c] for c in columns]
    return mean[order], scale[order], fill_values[order]


def transform_targets(
    values: np.ndarray,
    scaler_meta: Mapping[str, object],
    *,
    columns: Sequence[str] | None = None,
) -> np.ndarray:
    """Apply target StandardScaler transform using stored metadata."""
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)

    cols = (
        list(columns) if columns is not None else list(scaler_meta.get("feature_names", []) or [])
    )
    if arr.shape[1] != len(cols):
        raise ValueError(
            "Target transform dimensionality mismatch: "
            f"array has {arr.shape[1]} cols but metadata expects {len(cols)}"
        )

    mean, scale, fill_values = _align_target_scaler_vectors(scaler_meta, cols)
    safe_scale = scale.copy()
    safe_scale[safe_scale == 0.0] = 1.0

    arr, _ = _sanitize_matrix_with_fill(arr, fill_values=fill_values)
    return (arr - mean) / safe_scale


def inverse_transform_targets(
    values: np.ndarray,
    scaler_meta: Mapping[str, object],
    *,
    columns: Sequence[str] | None = None,
) -> np.ndarray:
    """Invert target StandardScaler transform back to raw physical units."""
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)

    cols = (
        list(columns) if columns is not None else list(scaler_meta.get("feature_names", []) or [])
    )
    if arr.shape[1] != len(cols):
        raise ValueError(
            "Target inverse-transform dimensionality mismatch: "
            f"array has {arr.shape[1]} cols but metadata expects {len(cols)}"
        )

    mean, scale, _ = _align_target_scaler_vectors(scaler_meta, cols)
    safe_scale = scale.copy()
    safe_scale[safe_scale == 0.0] = 1.0
    return arr * safe_scale + mean


def transform_target_frames_by_site(
    y_targets_by_site: Mapping[str, "pd.DataFrame"],
    scaler_meta: Mapping[str, object],
    *,
    columns: Sequence[str] | None = None,
) -> Dict[str, "pd.DataFrame"]:
    """Apply target scaler metadata to each site DataFrame."""
    cols = (
        list(columns) if columns is not None else list(scaler_meta.get("feature_names", []) or [])
    )
    out: Dict[str, "pd.DataFrame"] = {}

    for site, frame in y_targets_by_site.items():
        df = _ensure_pandas_dataframe(frame)
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise KeyError(f"Target columns missing for site '{site}': {missing}")

        transformed = transform_targets(df[cols].to_numpy(dtype=float), scaler_meta, columns=cols)
        df_out = df.copy()
        df_out.loc[:, cols] = transformed
        out[site] = df_out

    return out


# CLI (legacy generic normalization kept intact)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/training.yaml")
    parser.add_argument("--method", default=None)
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("--stats-out", default=None)
    parser.add_argument("--to-torch", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    method = args.method or cfg.get("normalization", {}).get("default_method", "zscore")
    methods_cfg = cfg.get("normalization", {}).get("methods", {})
    method_conf = methods_cfg.get(method, {})

    data, meta = load_input(args.input)
    stats = compute_stats(
        data,
        method,
        feature_range=tuple(method_conf.get("feature_range", (0.0, 1.0))),
        quantile_low=(method_conf.get("quantile_range") or [25, 75])[0],
        quantile_high=(method_conf.get("quantile_range") or [25, 75])[1],
    )
    normed = apply_normalization(data, stats, method)

    inp = Path(args.input)
    outp = Path(args.out) if args.out else inp.with_name(inp.stem + f"_{method}_norm" + inp.suffix)
    save_output(normed, meta, str(outp))

    stats_out = args.stats_out or (
        Path(cfg.get("normalization", {}).get("stats_dir", "experiments/normalization_stats"))
        / (inp.stem + f"_{method}_stats.json")
    )
    save_stats(stats, str(stats_out))

    if args.to_torch:
        if not _HAS_TORCH:
            print("torch not available; skipping tensor save")
        else:
            t = torch.tensor(
                normed.values if (_HAS_PANDAS and isinstance(normed, pd.DataFrame)) else normed
            )
            torch.save(t, str(Path(stats_out).with_suffix(".pt")))


if __name__ == "__main__":
    main()
