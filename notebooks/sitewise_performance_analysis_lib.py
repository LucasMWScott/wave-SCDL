"""Helpers for the site-wise performance analysis notebook."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import colors as mcolors
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon as MplPolygon

from notebooks import multisource_notebook_helpers as nh
from src.diagnostics.explainability import (
    _ensure_prediction_angles,
    circular_error_deg,
    load_results_bundle,
)

try:
    import xarray as xr
except Exception:  # pragma: no cover - optional dependency in notebook use
    xr = None

try:
    from pyproj import Transformer
except Exception:  # pragma: no cover - optional dependency in notebook use
    Transformer = None

try:
    from scipy.spatial import Voronoi
except Exception:  # pragma: no cover - optional dependency in notebook use
    Voronoi = None

try:
    from shapely.geometry import Polygon, box, mapping
except Exception:  # pragma: no cover - optional dependency in notebook use
    Polygon = None
    box = None
    mapping = None

try:
    from IPython.display import Markdown, display
except Exception:  # pragma: no cover - notebook-only convenience
    Markdown = None

    def display(obj: Any) -> None:
        print(obj)


REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_DPI = 300
TARGET_ORDER = ["hs", "tp", "dir", "dp"]
DIRECTIONAL_TARGETS = {"dir", "dp"}
PERCENT_ERROR_THRESHOLDS = (5.0, 10.0, 20.0, 50.0)
TARGET_LABELS = {
    "hs": "Hs",
    "tp": "Tp",
    "dir": "Dir",
    "dp": "Dp",
}
TARGET_UNITS = {
    "hs": "m",
    "tp": "s",
    "dir": "deg",
    "dp": "deg",
}
STATIC_BAR_COLUMNS = [
    "fjordness_score",
    "open_sector_fraction",
    "ray_fetch_mean_m",
    "ray_fetch_max_m",
    "static_dist_to_coast_m",
    "local_depth_m",
    "path_bottleneck_m",
    "path_tortuosity_ratio",
]
TARGET_ALIASES = {
    "hs": "hs",
    "significant_wave_height": "hs",
    "target_hs": "hs",
    "pred_hs": "hs",
    "tp": "tp",
    "peak_period": "tp",
    "target_tp": "tp",
    "pred_tp": "tp",
    "dir": "dir",
    "direction": "dir",
    "direction_deg": "dir",
    "wave_direction": "dir",
    "target_dir_deg": "dir",
    "pred_dir_deg": "dir",
    "dp": "dp",
    "peak_direction": "dp",
    "peak_direction_deg": "dp",
    "dp_deg": "dp",
    "target_dp_deg": "dp",
    "pred_dp_deg": "dp",
}
EXISTING_METRIC_VARIABLE_MAP = {
    "hs": "hs",
    "tp": "tp",
    "direction_deg": "dir",
    "dir": "dir",
    "dir_deg": "dir",
    "dp_deg": "dp",
    "dp": "dp",
}
EXCLUDED_STATIC_PATTERNS = (
    "site_name",
    "site",
    "row",
    "col",
    "lat",
    "lon",
    "site_x",
    "site_y",
    "site_lat",
    "site_lon",
    "reachable",
    "ray_count",
    "site_regime_",
)


@dataclass
class AnalysisContext:
    run_results_dir: Path
    split: str
    requested_splits: list[str]
    model_name: str
    baseline_name: str | None
    baseline_results_dir: Path | None
    tables_dir: Path
    figures_dir: Path
    case_study_dir: Path
    k_nearest_analog: int
    bundle: Any
    training_metadata: dict[str, Any]
    predictions: pd.DataFrame
    prediction_artifact: Path | list[Path]
    split_sets: dict[str, set[str]]
    static_features: pd.DataFrame | None
    static_features_path: Path | None
    existing_metric_tables: dict[str, pd.DataFrame]
    baseline_predictions: pd.DataFrame | None = None
    baseline_artifact: Path | list[Path] | None = None
    baseline_join_keys: list[str] = field(default_factory=list)
    available_targets: list[str] = field(default_factory=list)
    reuse_log: list[str] = field(default_factory=list)
    warning_log: list[str] = field(default_factory=list)
    map_summary: dict[str, Any] = field(default_factory=dict)
    figure_outputs: dict[str, list[Path]] = field(default_factory=dict)


def setup_plotting() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "figure.dpi": 120,
            "savefig.dpi": DEFAULT_DPI,
        }
    )


def maybe_markdown(text: str) -> Any:
    if Markdown is not None:
        return Markdown(text)
    return text


def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")


def save_table(df: pd.DataFrame, path: str | Path) -> Path:
    out_path = Path(path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return out_path


def save_figure(fig: plt.Figure, path_stem: str | Path, *, pdf: bool = True) -> list[Path]:
    stem = Path(path_stem).resolve()
    stem.parent.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    png_path = stem.with_suffix(".png")
    fig.savefig(png_path, bbox_inches="tight")
    written.append(png_path)
    if pdf:
        pdf_path = stem.with_suffix(".pdf")
        fig.savefig(pdf_path, bbox_inches="tight")
        written.append(pdf_path)
    plt.close(fig)
    return written


def _read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _resolve_results_dir(path_like: str | Path) -> Path:
    path = Path(path_like).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (Path.cwd() / path).resolve()


def _canonical_target_name(raw_value: Any) -> str | None:
    token = re.sub(r"[^a-z0-9]+", "_", str(raw_value).strip().lower()).strip("_")
    return TARGET_ALIASES.get(token)


def _first_existing_column(columns: Iterable[str], candidates: Sequence[str]) -> str | None:
    lowered = {str(col).lower(): str(col) for col in columns}
    for candidate in candidates:
        match = lowered.get(candidate.lower())
        if match is not None:
            return match
    return None


def _discover_prediction_artifact(results_dir: Path, split: str) -> Path:
    direct_candidates = [
        results_dir / f"predictions_{split}.csv",
        results_dir / f"predictions_{split}.parquet",
        results_dir / f"predictions_{split}.nc",
    ]
    for candidate in direct_candidates:
        if candidate.exists():
            return candidate

    search_patterns = [
        f"*pred*{split}*.csv",
        f"*pred*{split}*.parquet",
        f"*pred*{split}*.nc",
        f"*{split}*pred*.csv",
        f"*{split}*pred*.parquet",
        f"*{split}*pred*.nc",
    ]
    for pattern in search_patterns:
        matches = sorted(results_dir.rglob(pattern))
        if matches:
            return matches[0]
    raise FileNotFoundError(
        f"Could not find prediction artifact for split '{split}' under {results_dir}"
    )


def _normalize_split_mode(split: str | None) -> tuple[str, list[str]]:
    raw = str(split or "val").strip().lower().replace("_", "+").replace("/", "+")
    compact = raw.replace(" ", "")
    if compact in {"val", "validation"}:
        return "val", ["val"]
    if compact in {"test"}:
        return "test", ["test"]
    if compact in {
        "both",
        "combined",
        "val+test",
        "test+val",
        "valtest",
        "testval",
        "all+eval",
        "alleval",
    }:
        return "val+test", ["val", "test"]
    raise ValueError("split must be one of: 'val', 'test', or 'val+test' / 'both'")


def _load_predictions_for_requested_splits(
    results_dir: Path, requested_splits: Sequence[str]
) -> tuple[pd.DataFrame, Path | list[Path]]:
    frames: list[pd.DataFrame] = []
    artifacts: list[Path] = []
    for split_name in requested_splits:
        artifact = _discover_prediction_artifact(results_dir, split_name)
        frame = standardize_prediction_frame(_load_raw_prediction_table(artifact), split=split_name)
        if "split" not in frame.columns or frame["split"].isna().all():
            frame["split"] = split_name
        else:
            frame["split"] = frame["split"].fillna(split_name).astype(str)
        frames.append(frame)
        artifacts.append(artifact)
    if not frames:
        raise FileNotFoundError(
            f"No prediction artifacts were found under {results_dir} for splits {list(requested_splits)}"
        )
    combined = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    combined = combined.sort_values(
        [col for col in ["split", "site", "timestamp", "time_index"] if col in combined.columns],
        kind="stable",
    ).reset_index(drop=True)
    return combined, artifacts if len(artifacts) > 1 else artifacts[0]


def _load_raw_prediction_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".nc":
        if xr is None:
            raise RuntimeError("xarray is required to read NetCDF prediction files")
        with xr.open_dataset(path) as ds:
            return ds.to_dataframe().reset_index()
    raise ValueError(f"Unsupported prediction artifact: {path}")


def _standardize_site_time_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    rename_map: dict[str, str] = {}
    site_col = _first_existing_column(
        out.columns, ["site", "site_id", "site_name", "name", "point_name"]
    )
    time_col = _first_existing_column(
        out.columns, ["timestamp", "time", "datetime", "valid_time", "date"]
    )
    index_col = _first_existing_column(
        out.columns, ["time_index", "timestep", "step", "sample_index"]
    )
    split_col = _first_existing_column(out.columns, ["split", "dataset_split"])
    if site_col is not None and site_col != "site":
        rename_map[site_col] = "site"
    if time_col is not None and time_col != "timestamp":
        rename_map[time_col] = "timestamp"
    if index_col is not None and index_col != "time_index":
        rename_map[index_col] = "time_index"
    if split_col is not None and split_col != "split":
        rename_map[split_col] = "split"
    if rename_map:
        out = out.rename(columns=rename_map)
    if "site" in out.columns:
        out["site"] = out["site"].astype(str)
    if "timestamp" in out.columns:
        parsed = pd.to_datetime(out["timestamp"], errors="coerce", utc=False)
        if parsed.notna().any():
            out["timestamp"] = parsed.dt.tz_localize(None).astype("string")
        else:
            out["timestamp"] = out["timestamp"].astype("string")
    if "time_index" in out.columns:
        out["time_index"] = pd.to_numeric(out["time_index"], errors="coerce").astype("Int64")
    return out


def _pivot_long_prediction_table(df: pd.DataFrame) -> pd.DataFrame:
    out = _standardize_site_time_columns(df)
    variable_col = _first_existing_column(out.columns, ["variable", "target", "metric", "head"])
    if variable_col is None:
        return out

    target_col = _first_existing_column(
        out.columns, ["target_value", "target", "observed", "true", "y_true"]
    )
    pred_col = _first_existing_column(out.columns, ["predicted", "prediction", "pred", "y_pred"])
    base_col = _first_existing_column(
        out.columns, ["baseline", "reference", "baseline_prediction", "reference_prediction"]
    )
    if target_col is None or pred_col is None:
        return out

    join_cols = [col for col in ["site", "timestamp", "time_index", "split"] if col in out.columns]
    if "site" not in join_cols:
        return out

    working = out.copy()
    working["_canonical_target"] = working[variable_col].map(_canonical_target_name)
    working = working.loc[working["_canonical_target"].notna()].copy()
    if working.empty:
        return out

    pieces: list[pd.DataFrame] = []
    for source_name, value_col in [
        ("target", target_col),
        ("pred", pred_col),
        ("baseline", base_col),
    ]:
        if value_col is None or value_col not in working.columns:
            continue
        pivot = (
            working[join_cols + ["_canonical_target", value_col]]
            .pivot_table(
                index=join_cols, columns="_canonical_target", values=value_col, aggfunc="first"
            )
            .rename(
                columns={
                    target: f"{source_name}_{target}"
                    for target in TARGET_ORDER
                    if target in working["_canonical_target"].unique()
                }
            )
        )
        pieces.append(pivot)

    if not pieces:
        return out

    combined = pd.concat(pieces, axis=1).reset_index()
    return combined


def _rename_prediction_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    rename_map: dict[str, str] = {}
    candidate_map = {
        "target_hs": ["target_hs", "observed_hs", "true_hs", "hs_target", "hs_true"],
        "pred_hs": ["pred_hs", "prediction_hs", "predicted_hs", "hs_pred"],
        "baseline_hs": ["baseline_hs", "reference_hs"],
        "target_tp": ["target_tp", "observed_tp", "true_tp", "tp_target", "tp_true"],
        "pred_tp": ["pred_tp", "prediction_tp", "predicted_tp", "tp_pred"],
        "baseline_tp": ["baseline_tp", "reference_tp"],
        "target_dir_deg": [
            "target_dir_deg",
            "target_direction_deg",
            "observed_dir_deg",
            "true_dir_deg",
            "target_dir",
        ],
        "pred_dir_deg": ["pred_dir_deg", "pred_direction_deg", "prediction_dir_deg", "pred_dir"],
        "baseline_dir_deg": ["baseline_dir_deg", "reference_dir_deg", "baseline_dir"],
        "target_dp_deg": [
            "target_dp_deg",
            "target_peak_direction_deg",
            "observed_dp_deg",
            "true_dp_deg",
            "target_dp",
        ],
        "pred_dp_deg": ["pred_dp_deg", "pred_peak_direction_deg", "prediction_dp_deg", "pred_dp"],
        "baseline_dp_deg": ["baseline_dp_deg", "reference_dp_deg", "baseline_dp"],
    }
    for canonical, candidates in candidate_map.items():
        match = _first_existing_column(out.columns, candidates)
        if match is not None and match != canonical:
            rename_map[match] = canonical
    if rename_map:
        out = out.rename(columns=rename_map)
    return out


def standardize_prediction_frame(df: pd.DataFrame, split: str | None = None) -> pd.DataFrame:
    out = _pivot_long_prediction_table(df)
    out = _standardize_site_time_columns(out)
    if "split" in out.columns and split:
        split_mask = out["split"].astype(str).str.lower() == str(split).lower()
        if split_mask.any():
            out = out.loc[split_mask].copy()
    out = _ensure_prediction_angles(out)
    out = _rename_prediction_columns(out)

    if "target_dir_deg" not in out.columns and {"target_dir_sin", "target_dir_cos"} <= set(
        out.columns
    ):
        out["target_dir_deg"] = (
            np.degrees(np.arctan2(out["target_dir_sin"], out["target_dir_cos"])) + 360.0
        ) % 360.0
    if "pred_dir_deg" not in out.columns and {"pred_dir_sin", "pred_dir_cos"} <= set(out.columns):
        out["pred_dir_deg"] = (
            np.degrees(np.arctan2(out["pred_dir_sin"], out["pred_dir_cos"])) + 360.0
        ) % 360.0
    if "target_dp_deg" not in out.columns and {"target_dp_sin", "target_dp_cos"} <= set(
        out.columns
    ):
        out["target_dp_deg"] = (
            np.degrees(np.arctan2(out["target_dp_sin"], out["target_dp_cos"])) + 360.0
        ) % 360.0
    if "pred_dp_deg" not in out.columns and {"pred_dp_sin", "pred_dp_cos"} <= set(out.columns):
        out["pred_dp_deg"] = (
            np.degrees(np.arctan2(out["pred_dp_sin"], out["pred_dp_cos"])) + 360.0
        ) % 360.0

    required_order = ["site", "timestamp", "time_index"]
    present = [col for col in required_order if col in out.columns]
    if present:
        out = out.sort_values(present, kind="stable").reset_index(drop=True)
    return out


def _available_targets(df: pd.DataFrame) -> list[str]:
    available: list[str] = []
    for target in TARGET_ORDER:
        target_col = (
            f"target_{target}" if target not in DIRECTIONAL_TARGETS else f"target_{target}_deg"
        )
        pred_col = f"pred_{target}" if target not in DIRECTIONAL_TARGETS else f"pred_{target}_deg"
        if target_col in df.columns and pred_col in df.columns:
            available.append(target)
    return available


def _metrics_dir(results_dir: Path) -> Path:
    return results_dir / "metrics"


def _load_existing_metric_tables(results_dir: Path) -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}
    for name in [
        "metrics_validation.csv",
        "metrics_all_sites_combined.csv",
        "summary_statistics_target_vs_predicted.csv",
    ]:
        path = _metrics_dir(results_dir) / name
        if not path.exists():
            path = results_dir / name
        if path.exists():
            try:
                tables[name] = pd.read_csv(path)
            except Exception:
                continue
    return tables


def _normalize_existing_metrics(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "variable" in out.columns:
        out["target"] = out["variable"].map(
            lambda value: EXISTING_METRIC_VARIABLE_MAP.get(str(value).strip().lower())
        )
    if "pearson" in out.columns and "pearson_r" not in out.columns:
        out["pearson_r"] = out["pearson"]
    if "count" in out.columns and "sample_count" not in out.columns:
        out["sample_count"] = out["count"]
    if "site" in out.columns:
        out["site"] = out["site"].astype(str)
    if "mse" in out.columns:
        out["mse"] = pd.to_numeric(out["mse"], errors="coerce")
    return out


def _group_sample_counts(
    predictions: pd.DataFrame, available_targets: Sequence[str]
) -> tuple[dict[str, int], dict[tuple[str, str], int]]:
    overall: dict[str, int] = {}
    sitewise: dict[tuple[str, str], int] = {}
    for target in available_targets:
        target_col = (
            f"target_{target}" if target not in DIRECTIONAL_TARGETS else f"target_{target}_deg"
        )
        pred_col = f"pred_{target}" if target not in DIRECTIONAL_TARGETS else f"pred_{target}_deg"
        valid = predictions[[target_col, pred_col]].dropna()
        overall[target] = int(len(valid))
        if valid.empty:
            continue
        group = predictions.loc[valid.index].groupby("site", dropna=False).size()
        for site_name, count in group.items():
            sitewise[(str(site_name), target)] = int(count)
    return overall, sitewise


def load_analysis_context(
    *,
    run_results_dir: str | Path,
    split: str = "val",
    model_name: str | None = None,
    baseline_name: str | None = None,
    baseline_results_dir: str | Path | None = None,
    tables_dir: str | Path | None = None,
    figures_dir: str | Path | None = None,
    case_study_top_n: int = 5,
    k_nearest_analog: int = 5,
) -> AnalysisContext:
    resolved_results_dir = _resolve_results_dir(run_results_dir)
    if not resolved_results_dir.exists():
        raise FileNotFoundError(f"Missing run results directory: {resolved_results_dir}")
    split_mode, requested_splits = _normalize_split_mode(split)

    bundle = load_results_bundle(resolved_results_dir)
    training_metadata = _read_json(resolved_results_dir / "training_run_metadata.json")
    predictions, prediction_artifact = _load_predictions_for_requested_splits(
        resolved_results_dir, requested_splits
    )
    available_targets = _available_targets(predictions)
    if not available_targets:
        raise ValueError(
            f"No supported targets found in prediction artifact: {prediction_artifact}"
        )

    split_sets = nh.resolve_split_sets(
        training_metadata,
        config_path=bundle.config_path or resolved_results_dir / "training_run_metadata.json",
    )
    static_path_raw = (
        ((bundle.config.get("data", {}) or {}).get("static_features_csv"))
        if bundle.config
        else None
    )
    static_path = (
        Path(static_path_raw).resolve()
        if static_path_raw
        else (REPO_ROOT / "data" / "processed" / "master_static_features.csv")
    )
    static_features: pd.DataFrame | None = None
    if static_path.exists():
        static_features = pd.read_csv(static_path)
        if "site_name" in static_features.columns and "site" not in static_features.columns:
            static_features = static_features.rename(columns={"site_name": "site"})
        elif "name" in static_features.columns and "site" not in static_features.columns:
            static_features = static_features.rename(columns={"name": "site"})
        if "site" in static_features.columns:
            static_features["site"] = static_features["site"].astype(str)
    else:
        static_path = None

    if tables_dir is None:
        tables_dir = resolved_results_dir / "sitewise_performance"
    else:
        tables_dir = Path(tables_dir).expanduser().resolve()
    if figures_dir is None:
        figures_dir = resolved_results_dir / "figures" / "sitewise_performance"
    else:
        figures_dir = Path(figures_dir).expanduser().resolve()
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    case_study_dir = figures_dir / "case_studies"
    case_study_dir.mkdir(parents=True, exist_ok=True)

    existing_metric_tables = _load_existing_metric_tables(resolved_results_dir)
    context = AnalysisContext(
        run_results_dir=resolved_results_dir,
        split=split_mode,
        requested_splits=list(requested_splits),
        model_name=model_name or resolved_results_dir.name,
        baseline_name=baseline_name,
        baseline_results_dir=Path(baseline_results_dir).expanduser().resolve()
        if baseline_results_dir
        else None,
        tables_dir=tables_dir,
        figures_dir=figures_dir,
        case_study_dir=case_study_dir,
        k_nearest_analog=int(max(1, k_nearest_analog)),
        bundle=bundle,
        training_metadata=training_metadata,
        predictions=predictions,
        prediction_artifact=prediction_artifact,
        split_sets=split_sets,
        static_features=static_features,
        static_features_path=static_path,
        existing_metric_tables=existing_metric_tables,
        available_targets=available_targets,
    )
    if isinstance(prediction_artifact, list):
        context.reuse_log.append(
            "Loaded prediction artifacts: " + ", ".join(str(path) for path in prediction_artifact)
        )
    else:
        context.reuse_log.append(f"Loaded prediction artifact: {prediction_artifact}")
    if static_path is not None:
        context.reuse_log.append(f"Loaded static features: {static_path}")
    if case_study_top_n:
        context.reuse_log.append(
            f"Case study selection will use top_n={int(case_study_top_n)} per requested group"
        )
    _attach_baseline_predictions(context)
    return context


def _resolve_baseline_results_dir(context: AnalysisContext) -> Path | None:
    if context.baseline_results_dir is not None:
        return context.baseline_results_dir
    if not context.baseline_name:
        return None
    candidate = context.run_results_dir.parent / str(context.baseline_name)
    if candidate.exists():
        return candidate.resolve()
    return None


def _attach_baseline_predictions(context: AnalysisContext) -> None:
    baseline_dir = _resolve_baseline_results_dir(context)
    if baseline_dir is None:
        context.warning_log.append(
            "No baseline run directory was resolved; baseline-dependent analysis will be skipped."
        )
        return
    if not baseline_dir.exists():
        context.warning_log.append(f"Baseline run directory does not exist: {baseline_dir}")
        return

    try:
        baseline_predictions, baseline_artifact = _load_predictions_for_requested_splits(
            baseline_dir, context.requested_splits
        )
    except Exception as exc:
        context.warning_log.append(
            f"Failed to load baseline predictions from {baseline_dir}: {exc}"
        )
        return

    main = context.predictions.copy()
    base = baseline_predictions.copy()
    join_keys = [
        col
        for col in ["split", "site", "timestamp", "time_index"]
        if col in main.columns and col in base.columns
    ]
    non_split_keys = [col for col in join_keys if col != "split"]
    if non_split_keys == ["site"]:
        context.warning_log.append(
            "Baseline alignment only had 'site' available; skipping baseline joins to avoid ambiguous matches."
        )
        return
    if "site" not in join_keys:
        context.warning_log.append(
            "Could not find a safe site/time key intersection for baseline alignment."
        )
        return

    keep_cols = join_keys.copy()
    for target in context.available_targets:
        source_col = f"pred_{target}" if target not in DIRECTIONAL_TARGETS else f"pred_{target}_deg"
        if source_col in base.columns:
            keep_cols.append(source_col)
    keep_cols = list(dict.fromkeys(keep_cols))
    base = base[keep_cols].rename(
        columns={
            f"pred_{target}"
            if target not in DIRECTIONAL_TARGETS
            else f"pred_{target}_deg": f"baseline_{target}"
            if target not in DIRECTIONAL_TARGETS
            else f"baseline_{target}_deg"
            for target in context.available_targets
        }
    )
    merged = main.merge(base, on=join_keys, how="left", validate="one_to_one")
    matched_rows = int(
        merged[[col for col in merged.columns if col.startswith("baseline_")]]
        .notna()
        .any(axis=1)
        .sum()
    )
    context.predictions = merged
    context.baseline_predictions = baseline_predictions
    context.baseline_artifact = baseline_artifact
    context.baseline_join_keys = join_keys
    artifact_label = (
        ", ".join(str(path) for path in baseline_artifact)
        if isinstance(baseline_artifact, list)
        else str(baseline_artifact)
    )
    context.reuse_log.append(
        f"Aligned baseline predictions from {artifact_label} using keys {join_keys}; matched {matched_rows:,} rows."
    )


def describe_context(context: AnalysisContext) -> pd.DataFrame:
    prediction_label = (
        ", ".join(str(path) for path in context.prediction_artifact)
        if isinstance(context.prediction_artifact, list)
        else str(context.prediction_artifact)
    )
    return pd.DataFrame(
        [
            {"field": "run_results_dir", "value": str(context.run_results_dir)},
            {"field": "prediction_artifact", "value": prediction_label},
            {"field": "split", "value": context.split},
            {"field": "requested_splits", "value": ", ".join(context.requested_splits)},
            {"field": "model_name", "value": context.model_name},
            {"field": "available_targets", "value": ", ".join(context.available_targets)},
            {
                "field": "static_features_path",
                "value": str(context.static_features_path)
                if context.static_features_path
                else "missing",
            },
            {
                "field": "baseline_results_dir",
                "value": str(_resolve_baseline_results_dir(context))
                if _resolve_baseline_results_dir(context)
                else "none",
            },
            {
                "field": "baseline_join_keys",
                "value": ", ".join(context.baseline_join_keys)
                if context.baseline_join_keys
                else "none",
            },
            {"field": "tables_dir", "value": str(context.tables_dir)},
            {"field": "figures_dir", "value": str(context.figures_dir)},
        ]
    )


def _safe_r2(y_true: pd.Series, y_pred: pd.Series) -> float:
    true = pd.to_numeric(y_true, errors="coerce").to_numpy(dtype=float)
    pred = pd.to_numeric(y_pred, errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(true) & np.isfinite(pred)
    if int(mask.sum()) < 2:
        return float("nan")
    true = true[mask]
    pred = pred[mask]
    ss_tot = float(np.sum((true - np.mean(true)) ** 2))
    if ss_tot <= 0.0:
        return float("nan")
    ss_res = float(np.sum((pred - true) ** 2))
    return 1.0 - (ss_res / ss_tot)


def _safe_corr(y_true: pd.Series, y_pred: pd.Series) -> float:
    frame = pd.DataFrame(
        {
            "true": pd.to_numeric(y_true, errors="coerce"),
            "pred": pd.to_numeric(y_pred, errors="coerce"),
        }
    ).dropna()
    if len(frame) < 2:
        return float("nan")
    if frame["true"].nunique() <= 1 or frame["pred"].nunique() <= 1:
        return float("nan")
    return float(frame["true"].corr(frame["pred"]))


def _scalar_errors(y_true: pd.Series, y_pred: pd.Series) -> np.ndarray:
    frame = pd.DataFrame(
        {
            "true": pd.to_numeric(y_true, errors="coerce"),
            "pred": pd.to_numeric(y_pred, errors="coerce"),
        }
    ).dropna()
    return (frame["pred"] - frame["true"]).to_numpy(dtype=float)


def _circular_errors(y_true: pd.Series, y_pred: pd.Series) -> np.ndarray:
    frame = pd.DataFrame(
        {
            "true": pd.to_numeric(y_true, errors="coerce"),
            "pred": pd.to_numeric(y_pred, errors="coerce"),
        }
    ).dropna()
    if frame.empty:
        return np.asarray([], dtype=float)
    return np.asarray(
        circular_error_deg(
            frame["true"].to_numpy(dtype=float), frame["pred"].to_numpy(dtype=float)
        ),
        dtype=float,
    )


def _scalar_percent_errors(
    y_true: pd.Series, y_pred: pd.Series, *, min_abs_target: float = 1e-6
) -> np.ndarray:
    frame = pd.DataFrame(
        {
            "true": pd.to_numeric(y_true, errors="coerce"),
            "pred": pd.to_numeric(y_pred, errors="coerce"),
        }
    ).dropna()
    if frame.empty:
        return np.asarray([], dtype=float)
    valid = frame["true"].abs() > float(min_abs_target)
    if not bool(valid.any()):
        return np.asarray([], dtype=float)
    safe = frame.loc[valid]
    return ((safe["pred"] - safe["true"]) / safe["true"] * 100.0).to_numpy(dtype=float)


def _target_columns(target: str, *, prefix: str = "pred") -> tuple[str, str]:
    if target in DIRECTIONAL_TARGETS:
        return f"target_{target}_deg", f"{prefix}_{target}_deg"
    return f"target_{target}", f"{prefix}_{target}"


def _mean_if_any(values: np.ndarray) -> float:
    return float(np.mean(values)) if values.size else float("nan")


def _rmse_from_errors(errors: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(errors)))) if errors.size else float("nan")


def _quantile(value: pd.Series, q: float) -> float:
    clean = pd.to_numeric(value, errors="coerce").dropna()
    if clean.empty:
        return float("nan")
    return float(clean.quantile(q))


def _compute_metric_row(
    df: pd.DataFrame,
    target: str,
    *,
    site: str | None = None,
    baseline_available: bool = False,
) -> dict[str, Any]:
    target_col, pred_col = _target_columns(target)
    if target_col not in df.columns or pred_col not in df.columns:
        return {}

    local = df[
        [target_col, pred_col]
        + [
            col
            for col in df.columns
            if col in {"site", "timestamp", "time_index"} or col.startswith("baseline_")
        ]
    ].copy()
    local = local.dropna(subset=[target_col, pred_col])
    errors = (
        _circular_errors(local[target_col], local[pred_col])
        if target in DIRECTIONAL_TARGETS
        else _scalar_errors(local[target_col], local[pred_col])
    )
    if errors.size == 0:
        return {}

    target_values = pd.to_numeric(local[target_col], errors="coerce")
    pred_values = pd.to_numeric(local[pred_col], errors="coerce")
    row: dict[str, Any] = {
        "site": site if site is not None else "overall",
        "target": target,
        "target_label": TARGET_LABELS[target],
        "sample_count": int(errors.size),
        "mse": float(np.mean(np.square(errors))),
        "rmse": _rmse_from_errors(errors),
        "mae": float(np.mean(np.abs(errors))),
        "bias": float(np.mean(errors)),
        "pearson_r": _safe_corr(target_values, pred_values),
        "r2": _safe_r2(target_values, pred_values),
        "target_mean": float(target_values.mean()),
        "target_std": float(target_values.std(ddof=0)),
        "target_p05": _quantile(target_values, 0.05),
        "target_p10": _quantile(target_values, 0.10),
        "target_p50": _quantile(target_values, 0.50),
        "target_p90": _quantile(target_values, 0.90),
        "target_p95": _quantile(target_values, 0.95),
        "target_p99": _quantile(target_values, 0.99),
        "pred_mean": float(pred_values.mean()),
        "pred_std": float(pred_values.std(ddof=0)),
        "circular_rmse_deg": _rmse_from_errors(errors)
        if target in DIRECTIONAL_TARGETS
        else float("nan"),
        "circular_mae_deg": float(np.mean(np.abs(errors)))
        if target in DIRECTIONAL_TARGETS
        else float("nan"),
        "pct_within_15deg": float(np.mean(np.abs(errors) <= 15.0) * 100.0)
        if target in DIRECTIONAL_TARGETS
        else float("nan"),
        "pct_within_30deg": float(np.mean(np.abs(errors) <= 30.0) * 100.0)
        if target in DIRECTIONAL_TARGETS
        else float("nan"),
        "pct_error_sample_count": 0,
        "mean_pct_error": float("nan"),
        "mean_abs_pct_error": float("nan"),
        "median_abs_pct_error": float("nan"),
        "p90_abs_pct_error": float("nan"),
        "pct_samples_abs_pct_error_le_5": float("nan"),
        "pct_samples_abs_pct_error_le_10": float("nan"),
        "pct_samples_abs_pct_error_le_20": float("nan"),
        "pct_samples_abs_pct_error_le_50": float("nan"),
        "scatter_index": float("nan"),
        "normalized_rmse": float("nan"),
        "high_energy_rmse": float("nan"),
        "baseline_mse": float("nan"),
        "baseline_rmse": float("nan"),
        "baseline_mae": float("nan"),
        "skill": float("nan"),
        "rmse_improvement_vs_baseline": float("nan"),
    }

    observed_std = float(target_values.std(ddof=0))
    if observed_std > 0.0:
        row["normalized_rmse"] = row["rmse"] / observed_std
    if target not in DIRECTIONAL_TARGETS:
        pct_errors = _scalar_percent_errors(target_values, pred_values)
        if pct_errors.size:
            abs_pct_errors = np.abs(pct_errors)
            row["pct_error_sample_count"] = int(pct_errors.size)
            row["mean_pct_error"] = float(np.mean(pct_errors))
            row["mean_abs_pct_error"] = float(np.mean(abs_pct_errors))
            row["median_abs_pct_error"] = float(np.median(abs_pct_errors))
            row["p90_abs_pct_error"] = float(np.quantile(abs_pct_errors, 0.90))
            row["pct_samples_abs_pct_error_le_5"] = float(np.mean(abs_pct_errors <= 5.0) * 100.0)
            row["pct_samples_abs_pct_error_le_10"] = float(np.mean(abs_pct_errors <= 10.0) * 100.0)
            row["pct_samples_abs_pct_error_le_20"] = float(np.mean(abs_pct_errors <= 20.0) * 100.0)
            row["pct_samples_abs_pct_error_le_50"] = float(np.mean(abs_pct_errors <= 50.0) * 100.0)
    if target == "hs":
        observed_mean = float(target_values.mean())
        if observed_mean != 0.0 and np.isfinite(observed_mean):
            row["scatter_index"] = row["rmse"] / observed_mean
        hs_p90 = float(target_values.quantile(0.90))
        high_mask = target_values >= hs_p90
        if bool(high_mask.any()):
            high_errors = _scalar_errors(target_values.loc[high_mask], pred_values.loc[high_mask])
            row["high_energy_rmse"] = _rmse_from_errors(high_errors)

    if baseline_available:
        baseline_col = (
            f"baseline_{target}" if target not in DIRECTIONAL_TARGETS else f"baseline_{target}_deg"
        )
        if baseline_col in df.columns:
            aligned = df[[target_col, pred_col, baseline_col]].dropna()
            if not aligned.empty:
                baseline_errors = (
                    _circular_errors(aligned[target_col], aligned[baseline_col])
                    if target in DIRECTIONAL_TARGETS
                    else _scalar_errors(aligned[target_col], aligned[baseline_col])
                )
                if baseline_errors.size:
                    baseline_mse = float(np.mean(np.square(baseline_errors)))
                    row["baseline_mse"] = baseline_mse
                    row["baseline_rmse"] = _rmse_from_errors(baseline_errors)
                    row["baseline_mae"] = float(np.mean(np.abs(baseline_errors)))
                    row["rmse_improvement_vs_baseline"] = row["baseline_rmse"] - row["rmse"]
                    if baseline_mse > 0.0:
                        row["skill"] = 1.0 - (row["mse"] / baseline_mse)

    return row


def _reuse_existing_overall_metrics(context: AnalysisContext) -> pd.DataFrame:
    existing = context.existing_metric_tables.get("metrics_validation.csv")
    if existing is None:
        return pd.DataFrame()
    existing = _normalize_existing_metrics(existing)
    overall_counts, _ = _group_sample_counts(context.predictions, context.available_targets)
    rows: list[dict[str, Any]] = []
    for target in context.available_targets:
        subset = existing.loc[existing["target"] == target].copy()
        if subset.empty:
            continue
        sample_count = overall_counts.get(target, -1)
        subset = subset.loc[pd.to_numeric(subset["sample_count"], errors="coerce") == sample_count]
        if subset.empty:
            continue
        row = subset.iloc[0].to_dict()
        rows.append(
            {
                "target": target,
                "sample_count": int(row.get("sample_count", sample_count)),
                "mse": float(row.get("mse", np.nan)),
                "rmse": float(row.get("rmse", np.nan)),
                "bias": float(row.get("bias", np.nan)),
                "pearson_r": float(row.get("pearson_r", np.nan)),
                "r2": float(row.get("r2", np.nan)),
            }
        )
    if rows:
        context.reuse_log.append(
            "Reused existing overall core metrics from metrics_validation.csv where counts matched the selected split."
        )
    return pd.DataFrame(rows)


def _reuse_existing_site_metrics(context: AnalysisContext) -> pd.DataFrame:
    existing = context.existing_metric_tables.get("metrics_all_sites_combined.csv")
    if existing is None:
        return pd.DataFrame()
    existing = _normalize_existing_metrics(existing)
    _, site_counts = _group_sample_counts(context.predictions, context.available_targets)
    rows: list[dict[str, Any]] = []
    for _, raw in existing.iterrows():
        target = raw.get("target")
        site = str(raw.get("site", ""))
        if target not in context.available_targets or not site or site.lower() == "validation":
            continue
        expected_count = site_counts.get((site, target))
        actual_count = pd.to_numeric(raw.get("sample_count"), errors="coerce")
        if (
            expected_count is None
            or not np.isfinite(actual_count)
            or int(actual_count) != int(expected_count)
        ):
            continue
        rows.append(
            {
                "site": site,
                "target": target,
                "sample_count": int(actual_count),
                "mse": float(raw.get("mse", np.nan)),
                "rmse": float(raw.get("rmse", np.nan)),
                "bias": float(raw.get("bias", np.nan)),
                "pearson_r": float(raw.get("pearson_r", np.nan)),
                "r2": float(raw.get("r2", np.nan)),
            }
        )
    if rows:
        context.reuse_log.append(
            "Reused existing per-site core metrics from metrics_all_sites_combined.csv where site counts matched the selected split."
        )
    return pd.DataFrame(rows)


def compute_metrics_tables(context: AnalysisContext) -> dict[str, pd.DataFrame]:
    baseline_available = context.baseline_predictions is not None and any(
        col.startswith("baseline_") for col in context.predictions.columns
    )
    reused_overall = _reuse_existing_overall_metrics(context)
    reused_site = _reuse_existing_site_metrics(context)

    overall_rows: list[dict[str, Any]] = []
    site_rows: list[dict[str, Any]] = []
    for target in context.available_targets:
        row = _compute_metric_row(
            context.predictions, target, site=None, baseline_available=baseline_available
        )
        if row:
            overall_rows.append(row)
    for site_name, site_df in context.predictions.groupby("site", sort=True):
        for target in context.available_targets:
            row = _compute_metric_row(
                site_df, target, site=str(site_name), baseline_available=baseline_available
            )
            if row:
                site_rows.append(row)

    overall_metrics = pd.DataFrame(overall_rows)
    site_metrics = pd.DataFrame(site_rows)

    if not reused_overall.empty:
        for _, reused_row in reused_overall.iterrows():
            mask = overall_metrics["target"] == reused_row["target"]
            for col in ["mse", "rmse", "bias", "pearson_r", "r2", "sample_count"]:
                overall_metrics.loc[mask, col] = reused_row[col]
    if not reused_site.empty:
        reuse_cols = ["mse", "rmse", "bias", "pearson_r", "r2", "sample_count"]
        site_metrics = site_metrics.merge(
            reused_site[["site", "target"] + reuse_cols],
            on=["site", "target"],
            how="left",
            suffixes=("", "_existing"),
        )
        for col in reuse_cols:
            existing_col = f"{col}_existing"
            if existing_col in site_metrics.columns:
                site_metrics[col] = site_metrics[existing_col].where(
                    site_metrics[existing_col].notna(), site_metrics[col]
                )
        site_metrics = site_metrics[
            [col for col in site_metrics.columns if not col.endswith("_existing")]
        ]

    overall_metrics = overall_metrics.sort_values(
        "target", key=lambda s: s.map({name: idx for idx, name in enumerate(TARGET_ORDER)})
    ).reset_index(drop=True)
    site_metrics = site_metrics.sort_values(["target", "site"]).reset_index(drop=True)

    save_table(overall_metrics, context.tables_dir / "overall_metrics.csv")
    save_table(site_metrics, context.tables_dir / "site_metrics.csv")
    return {"overall_metrics": overall_metrics, "site_metrics": site_metrics}


def compute_site_percent_error_summary(
    site_metrics: pd.DataFrame, output_path: Path
) -> pd.DataFrame:
    if site_metrics.empty:
        summary = pd.DataFrame()
        save_table(summary, output_path)
        return summary
    scalar = site_metrics.loc[
        (~site_metrics["target"].isin(DIRECTIONAL_TARGETS))
        & pd.to_numeric(site_metrics["mean_abs_pct_error"], errors="coerce").notna()
    ].copy()
    if scalar.empty:
        summary = pd.DataFrame()
        save_table(summary, output_path)
        return summary

    rows: list[dict[str, Any]] = []
    for target, group in scalar.groupby("target", sort=False):
        mean_abs = pd.to_numeric(group["mean_abs_pct_error"], errors="coerce").dropna()
        mean_signed = pd.to_numeric(group["mean_pct_error"], errors="coerce").dropna()
        if mean_abs.empty:
            continue
        row: dict[str, Any] = {
            "target": target,
            "target_label": TARGET_LABELS.get(target, str(target)),
            "site_count": int(group["site"].nunique()),
            "site_count_with_pct_error": int(mean_abs.size),
            "median_site_mean_abs_pct_error": float(mean_abs.quantile(0.50)),
            "p75_site_mean_abs_pct_error": float(mean_abs.quantile(0.75)),
            "p90_site_mean_abs_pct_error": float(mean_abs.quantile(0.90)),
            "max_site_mean_abs_pct_error": float(mean_abs.max()),
            "median_site_mean_pct_error": float(mean_signed.quantile(0.50))
            if not mean_signed.empty
            else float("nan"),
            "mean_site_mean_pct_error": float(mean_signed.mean())
            if not mean_signed.empty
            else float("nan"),
        }
        for threshold in PERCENT_ERROR_THRESHOLDS:
            row[f"share_sites_mean_abs_pct_error_le_{int(threshold)}"] = float(
                np.mean(mean_abs <= threshold) * 100.0
            )
        rows.append(row)
    summary = (
        pd.DataFrame(rows).sort_values("target").reset_index(drop=True) if rows else pd.DataFrame()
    )
    save_table(summary, output_path)
    return summary


def compute_error_contributions(
    context: AnalysisContext, site_metrics: pd.DataFrame
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for target in context.available_targets:
        target_col, pred_col = _target_columns(target)
        frame = context.predictions[["site", target_col, pred_col]].dropna().copy()
        if frame.empty:
            continue
        if target in DIRECTIONAL_TARGETS:
            frame["squared_error"] = np.square(_circular_errors(frame[target_col], frame[pred_col]))
        else:
            frame["squared_error"] = np.square(
                pd.to_numeric(frame[pred_col], errors="coerce")
                - pd.to_numeric(frame[target_col], errors="coerce")
            )
        grouped = (
            frame.groupby("site", dropna=False)["squared_error"]
            .sum()
            .reset_index(name="total_squared_error")
        )
        total = float(grouped["total_squared_error"].sum())
        grouped["error_contribution_fraction"] = (
            grouped["total_squared_error"] / total if total > 0.0 else np.nan
        )
        grouped = grouped.sort_values("error_contribution_fraction", ascending=False).reset_index(
            drop=True
        )
        grouped["cumulative_error_contribution"] = grouped["error_contribution_fraction"].cumsum()
        grouped["error_contribution_rank"] = np.arange(1, len(grouped) + 1)
        grouped["target"] = target
        metric_subset = site_metrics.loc[
            site_metrics["target"] == target, ["site", "rmse", "skill"]
        ].copy()
        if not metric_subset.empty:
            metric_subset["rmse_rank"] = metric_subset["rmse"].rank(method="min", ascending=False)
            metric_subset["skill_rank"] = metric_subset["skill"].rank(method="min", ascending=False)
            grouped = grouped.merge(
                metric_subset[["site", "rmse_rank", "skill_rank"]], on="site", how="left"
            )
        rows.append(grouped)
    contributions = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    save_table(contributions, context.tables_dir / "site_error_contributions.csv")
    return contributions


def compute_site_performance_categories(
    site_metrics: pd.DataFrame, contributions: pd.DataFrame, output_path: Path
) -> pd.DataFrame:
    if site_metrics.empty:
        categories = pd.DataFrame()
        save_table(categories, output_path)
        return categories
    merged = site_metrics.merge(
        contributions[["site", "target", "error_contribution_fraction"]]
        if not contributions.empty
        else pd.DataFrame(columns=["site", "target", "error_contribution_fraction"]),
        on=["site", "target"],
        how="left",
    )
    rows: list[pd.DataFrame] = []
    for target, group in merged.groupby("target", sort=False):
        frame = group.copy()
        contribution_p75 = frame["error_contribution_fraction"].quantile(0.75)
        skill_p25 = frame["skill"].quantile(0.25)
        skill_p75 = frame["skill"].quantile(0.75)
        rmse_p25 = frame["rmse"].quantile(0.25)
        var_p25 = frame["target_std"].quantile(0.25)
        sample_p10 = frame["sample_count"].quantile(0.10)
        frame["is_high_contribution"] = frame["error_contribution_fraction"] >= contribution_p75
        frame["is_poor_skill"] = frame["skill"] <= skill_p25
        frame["is_good_skill"] = frame["skill"] >= skill_p75
        frame["is_low_rmse"] = frame["rmse"] <= rmse_p25
        frame["is_low_variance"] = frame["target_std"] <= var_p25
        frame["is_insufficient_samples"] = frame["sample_count"] <= sample_p10
        frame["is_negative_skill"] = frame["skill"] < 0.0
        frame["category_label"] = np.select(
            [
                frame["is_insufficient_samples"],
                frame["is_high_contribution"] & frame["is_poor_skill"],
                frame["is_high_contribution"] & frame["is_good_skill"],
                frame["is_negative_skill"],
                frame["is_low_rmse"] & frame["is_low_variance"],
                frame["is_low_rmse"] & frame["is_good_skill"],
            ],
            [
                "insufficient samples",
                "high contribution + poor skill",
                "high contribution + good skill",
                "negative skill",
                "low RMSE + low variance",
                "low RMSE + high skill",
            ],
            default="mixed",
        )
        rows.append(frame)
    categories = pd.concat(rows, ignore_index=True)
    save_table(categories, output_path)
    return categories


def _derive_regime_label(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "regime_label" in out.columns:
        return out
    for candidate in ["site_regime", "regime", "site_dominant_regime"]:
        if candidate in out.columns:
            out["regime_label"] = out[candidate].astype(str)
            return out
    one_hot = [
        col
        for col in ["site_regime_open", "site_regime_transition", "site_regime_fjord"]
        if col in out.columns
    ]
    if len(one_hot) == 3:
        labels = out[one_hot].idxmax(axis=1).str.replace("site_regime_", "", regex=False)
        labels = labels.where(out[one_hot].sum(axis=1) > 0, other=np.nan)
        out["regime_label"] = labels
    return out


def _normalize_site_metadata_frame(df: pd.DataFrame | None) -> pd.DataFrame | None:
    if df is None or df.empty:
        return None
    out = df.copy()
    if "site" not in out.columns:
        for candidate in ["site_name", "name", "point_name"]:
            if candidate in out.columns:
                out = out.rename(columns={candidate: "site"})
                break
    if "site" not in out.columns:
        return None
    out["site"] = out["site"].astype(str)
    return out


def _load_master_static_features_fallback() -> pd.DataFrame | None:
    path = REPO_ROOT / "data" / "processed" / "master_static_features.csv"
    if not path.exists():
        return None
    try:
        return _normalize_site_metadata_frame(pd.read_csv(path))
    except Exception:
        return None


def _load_sites_yaml_coordinates() -> pd.DataFrame | None:
    sites_path = REPO_ROOT / "configs" / "sites.yaml"
    if not sites_path.exists():
        return None
    try:
        sites_cfg = nh.load_sites_config(sites_path)
    except Exception:
        return None
    rows: list[dict[str, Any]] = []
    for group_name in ["nearshore_sites", "offshore_sites"]:
        for item in sites_cfg.get(group_name, []) or []:
            site_name = item.get("name")
            if not site_name:
                continue
            rows.append(
                {
                    "site": str(site_name),
                    "site_lat": pd.to_numeric(item.get("lat"), errors="coerce"),
                    "site_lon": pd.to_numeric(item.get("lon"), errors="coerce"),
                }
            )
    if not rows:
        return None
    return pd.DataFrame(rows).drop_duplicates(subset=["site"])


def _backfill_site_metadata(
    base_df: pd.DataFrame, metadata_df: pd.DataFrame | None
) -> pd.DataFrame:
    if metadata_df is None or metadata_df.empty or base_df.empty:
        return base_df
    metadata = _normalize_site_metadata_frame(metadata_df)
    if metadata is None or metadata.empty:
        return base_df
    out = base_df.copy()
    meta_cols = [col for col in metadata.columns if col != "site"]
    overlap = [col for col in meta_cols if col in out.columns]
    if not overlap:
        merged = out.merge(metadata[["site"] + meta_cols], on="site", how="left")
        return merged
    merge_cols = ["site"] + overlap + [col for col in meta_cols if col not in overlap]
    merged = out.merge(metadata[merge_cols], on="site", how="left", suffixes=("", "__fill"))
    for col in overlap:
        fill_col = f"{col}__fill"
        if fill_col not in merged.columns:
            continue
        merged[col] = merged[col].where(merged[col].notna(), merged[fill_col])
        merged = merged.drop(columns=[fill_col])
    return merged


def _enrich_site_metadata(context: AnalysisContext, df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    master_static = _load_master_static_features_fallback()
    if context.static_features is not None:
        out = _backfill_site_metadata(out, context.static_features)
    if master_static is not None:
        out = _backfill_site_metadata(out, master_static)
    sites_yaml_coords = _load_sites_yaml_coordinates()
    if sites_yaml_coords is not None:
        out = _backfill_site_metadata(out, sites_yaml_coords)
    return _derive_regime_label(out)


def merge_static_features(context: AnalysisContext, site_metrics: pd.DataFrame) -> pd.DataFrame:
    if site_metrics.empty:
        merged = _derive_regime_label(site_metrics.copy())
        save_table(merged, context.tables_dir / "site_metrics_with_static_features.csv")
        return merged
    merged = site_metrics.copy()
    merged = _enrich_site_metadata(context, merged)
    save_table(merged, context.tables_dir / "site_metrics_with_static_features.csv")
    return merged


def _site_metric_wide(site_metrics_with_features: pd.DataFrame) -> pd.DataFrame:
    if site_metrics_with_features.empty:
        return pd.DataFrame()
    index_cols = ["site"]
    feature_cols = [
        col
        for col in site_metrics_with_features.columns
        if col not in {"target"}
        and not col.startswith("target_")
        and col
        not in {
            "rmse",
            "mae",
            "bias",
            "pearson_r",
            "r2",
            "mse",
            "circular_rmse_deg",
            "circular_mae_deg",
            "pct_within_15deg",
            "pct_within_30deg",
            "pct_error_sample_count",
            "mean_pct_error",
            "mean_abs_pct_error",
            "median_abs_pct_error",
            "p90_abs_pct_error",
            "pct_samples_abs_pct_error_le_5",
            "pct_samples_abs_pct_error_le_10",
            "pct_samples_abs_pct_error_le_20",
            "pct_samples_abs_pct_error_le_50",
            "scatter_index",
            "normalized_rmse",
            "high_energy_rmse",
            "baseline_mse",
            "baseline_rmse",
            "baseline_mae",
            "skill",
            "rmse_improvement_vs_baseline",
        }
    ]
    base = site_metrics_with_features[
        index_cols + [col for col in feature_cols if col not in index_cols]
    ].drop_duplicates(subset=index_cols)
    pivot_cols = [
        "rmse",
        "mae",
        "bias",
        "pearson_r",
        "r2",
        "circular_rmse_deg",
        "circular_mae_deg",
        "mean_pct_error",
        "mean_abs_pct_error",
        "median_abs_pct_error",
        "p90_abs_pct_error",
        "pct_samples_abs_pct_error_le_5",
        "pct_samples_abs_pct_error_le_10",
        "pct_samples_abs_pct_error_le_20",
        "pct_samples_abs_pct_error_le_50",
        "skill",
        "baseline_rmse",
        "sample_count",
    ]
    pivot = site_metrics_with_features.pivot_table(
        index="site",
        columns="target",
        values=[col for col in pivot_cols if col in site_metrics_with_features.columns],
        aggfunc="first",
    )
    pivot.columns = [f"{target}_{metric}" for metric, target in pivot.columns]
    pivot = pivot.reset_index()
    merged = base.merge(pivot, on="site", how="left")
    merged = _derive_regime_label(merged)
    return merged


def compute_site_composite_rankings(
    site_metrics_with_features: pd.DataFrame, output_path: Path
) -> pd.DataFrame:
    wide = _site_metric_wide(site_metrics_with_features)
    if wide.empty:
        save_table(wide, output_path)
        return wide

    rank_columns: list[str] = []
    for target in TARGET_ORDER:
        primary_col = f"{target}_rmse" if target in {"hs", "tp"} else f"{target}_circular_rmse_deg"
        if primary_col in wide.columns:
            rank_col = f"{primary_col}_pct_rank"
            wide[rank_col] = wide[primary_col].rank(pct=True, ascending=True)
            rank_columns.append(rank_col)
        skill_col = f"{target}_skill"
        if skill_col in wide.columns and wide[skill_col].notna().any():
            rank_col = f"{skill_col}_worse_pct_rank"
            wide[rank_col] = wide[skill_col].rank(pct=True, ascending=False)
            rank_columns.append(rank_col)

    if rank_columns:
        wide["composite_worst_score"] = wide[rank_columns].mean(axis=1)
        wide["composite_best_score"] = 1.0 - wide["composite_worst_score"]
        wide["composite_worst_rank"] = (
            wide["composite_worst_score"].rank(method="min", ascending=False).astype("Int64")
        )
        wide["composite_best_rank"] = (
            wide["composite_worst_score"].rank(method="min", ascending=True).astype("Int64")
        )
    else:
        wide["composite_worst_score"] = np.nan
        wide["composite_best_score"] = np.nan
        wide["composite_worst_rank"] = pd.Series([pd.NA] * len(wide), dtype="Int64")
        wide["composite_best_rank"] = pd.Series([pd.NA] * len(wide), dtype="Int64")

    save_table(
        wide.sort_values(["composite_worst_score", "site"], ascending=[False, True]), output_path
    )
    return wide


def _eligible_static_feature_columns(df: pd.DataFrame) -> list[str]:
    eligible: list[str] = []
    for column in df.columns:
        lowered = str(column).lower()
        if any(pattern in lowered for pattern in EXCLUDED_STATIC_PATTERNS):
            continue
        series = df[column]
        if pd.api.types.is_bool_dtype(series):
            continue
        if not pd.api.types.is_numeric_dtype(series):
            continue
        clean = pd.to_numeric(series, errors="coerce").dropna()
        if clean.empty or clean.nunique() <= 1:
            continue
        eligible.append(column)
    return eligible


def compute_feature_error_correlations(
    site_composite: pd.DataFrame, output_path: Path
) -> pd.DataFrame:
    if site_composite.empty:
        correlations = pd.DataFrame()
        save_table(correlations, output_path)
        return correlations
    working = site_composite.loc[:, ~site_composite.columns.duplicated()].copy()

    metric_map = {
        "hs_rmse": "Hs RMSE",
        "hs_skill": "Hs Skill",
        "tp_rmse": "Tp RMSE",
        "dir_circular_rmse_deg": "Dir Circular RMSE",
        "dp_circular_rmse_deg": "Dp Circular RMSE",
        "composite_worst_score": "Composite Normalized Error",
    }
    eligible_metrics = [col for col in metric_map if col in working.columns]
    feature_cols = _eligible_static_feature_columns(working)
    rows: list[dict[str, Any]] = []
    for metric in eligible_metrics:
        for feature in feature_cols:
            pair = pd.DataFrame(
                {
                    "metric_value": pd.to_numeric(working[metric], errors="coerce"),
                    "feature_value": pd.to_numeric(working[feature], errors="coerce"),
                }
            ).dropna()
            if len(pair) < 3:
                continue
            pearson = pair["metric_value"].corr(pair["feature_value"], method="pearson")
            spearman = pair["metric_value"].corr(pair["feature_value"], method="spearman")
            rows.append(
                {
                    "metric": metric,
                    "metric_label": metric_map[metric],
                    "feature": feature,
                    "pearson_r": float(pearson) if pearson is not None else np.nan,
                    "spearman_r": float(spearman) if spearman is not None else np.nan,
                    "abs_pearson_r": abs(float(pearson)) if pearson is not None else np.nan,
                    "abs_spearman_r": abs(float(spearman)) if spearman is not None else np.nan,
                }
            )
    correlations = (
        pd.DataFrame(rows)
        .sort_values(["metric", "abs_spearman_r"], ascending=[True, False])
        .reset_index(drop=True)
        if rows
        else pd.DataFrame()
    )
    save_table(correlations, output_path)
    return correlations


def compute_training_analog_distances(
    context: AnalysisContext,
    site_composite: pd.DataFrame,
    output_path: Path,
) -> pd.DataFrame:
    if site_composite.empty:
        analog = pd.DataFrame()
        save_table(analog, output_path)
        return analog
    feature_cols = _eligible_static_feature_columns(site_composite)
    if not feature_cols:
        analog = pd.DataFrame()
        save_table(analog, output_path)
        return analog

    train_sites = set(context.split_sets.get("train", set()))
    train_df = site_composite.loc[
        site_composite["site"].isin(train_sites), ["site"] + feature_cols
    ].copy()
    eval_sites: set[str] = set()
    for split_name in context.requested_splits:
        eval_sites.update(set(context.split_sets.get(split_name, set())))
    if not eval_sites:
        eval_sites = set(site_composite["site"])
    eval_df = site_composite.loc[
        site_composite["site"].isin(eval_sites), ["site"] + feature_cols
    ].copy()
    if train_df.empty or eval_df.empty:
        analog = pd.DataFrame()
        save_table(analog, output_path)
        return analog

    means = train_df[feature_cols].mean(axis=0)
    stds = train_df[feature_cols].std(axis=0, ddof=0).replace(0.0, np.nan)
    usable_cols = [col for col in feature_cols if np.isfinite(stds.get(col, np.nan))]
    if not usable_cols:
        analog = pd.DataFrame()
        save_table(analog, output_path)
        return analog

    train_matrix = ((train_df[usable_cols] - means[usable_cols]) / stds[usable_cols]).to_numpy(
        dtype=float
    )
    eval_matrix = ((eval_df[usable_cols] - means[usable_cols]) / stds[usable_cols]).to_numpy(
        dtype=float
    )
    rows: list[dict[str, Any]] = []
    k = int(max(1, context.k_nearest_analog))
    for idx, site_name in enumerate(eval_df["site"].astype(str)):
        row_vec = eval_matrix[idx]
        if not np.isfinite(row_vec).all():
            continue
        distances = np.sqrt(np.nansum(np.square(train_matrix - row_vec), axis=1))
        order = np.argsort(distances)
        nearest_idx = int(order[0])
        nearest_distance = float(distances[nearest_idx])
        k_idx = order[: min(k, len(order))]
        rows.append(
            {
                "site": site_name,
                "nearest_training_site": str(train_df.iloc[nearest_idx]["site"]),
                "nearest_training_distance": nearest_distance,
                f"mean_distance_top_{k}": float(np.mean(distances[k_idx])),
                "analog_feature_count": int(len(usable_cols)),
            }
        )
    analog = (
        pd.DataFrame(rows).sort_values(["nearest_training_distance", "site"]).reset_index(drop=True)
        if rows
        else pd.DataFrame()
    )
    save_table(analog, output_path)
    return analog


def compute_regime_summary(
    site_metrics_with_features: pd.DataFrame, output_path: Path
) -> pd.DataFrame:
    if site_metrics_with_features.empty or "regime_label" not in site_metrics_with_features.columns:
        summary = pd.DataFrame()
        save_table(summary, output_path)
        return summary
    rows: list[dict[str, Any]] = []
    for (regime, target), group in site_metrics_with_features.groupby(
        ["regime_label", "target"], dropna=False
    ):
        rows.append(
            {
                "regime_label": regime,
                "target": target,
                "median_rmse": float(group["rmse"].median()),
                "mean_rmse": float(group["rmse"].mean()),
                "median_skill": float(group["skill"].median())
                if group["skill"].notna().any()
                else np.nan,
                "site_count": int(group["site"].nunique()),
                "sample_count": int(group["sample_count"].sum()),
            }
        )
    summary = pd.DataFrame(rows).sort_values(["regime_label", "target"]).reset_index(drop=True)
    save_table(summary, output_path)
    return summary


def compute_case_study_site_selection(
    site_metrics: pd.DataFrame,
    error_contributions: pd.DataFrame,
    site_composite: pd.DataFrame,
    *,
    top_n: int,
    output_path: Path,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    hs_metrics = site_metrics.loc[site_metrics["target"] == "hs"].copy()
    hs_contrib = (
        error_contributions.loc[error_contributions["target"] == "hs"].copy()
        if not error_contributions.empty
        else pd.DataFrame()
    )

    def _append_ranked(
        df: pd.DataFrame, order_col: str, selection_group: str, ascending: bool
    ) -> None:
        if df.empty or order_col not in df.columns:
            return
        ranked = df.sort_values(order_col, ascending=ascending).head(top_n).reset_index(drop=True)
        for idx, row in ranked.iterrows():
            rows.append(
                {
                    "selection_group": selection_group,
                    "rank": idx + 1,
                    "site": str(row["site"]),
                    "value": row.get(order_col),
                }
            )

    _append_ranked(
        hs_contrib, "error_contribution_fraction", "top_hs_error_contribution", ascending=False
    )
    _append_ranked(hs_metrics.dropna(subset=["skill"]), "skill", "worst_hs_skill", ascending=True)
    _append_ranked(hs_metrics.dropna(subset=["skill"]), "skill", "best_hs_skill", ascending=False)
    _append_ranked(
        hs_metrics.dropna(subset=["rmse_improvement_vs_baseline"]),
        "rmse_improvement_vs_baseline",
        "most_improved_vs_baseline",
        ascending=False,
    )
    _append_ranked(
        hs_metrics.dropna(subset=["rmse_improvement_vs_baseline"]),
        "rmse_improvement_vs_baseline",
        "underperforms_baseline",
        ascending=True,
    )
    _append_ranked(
        site_composite.dropna(subset=["composite_worst_score"]),
        "composite_worst_score",
        "composite_worst",
        ascending=False,
    )
    _append_ranked(
        site_composite.dropna(subset=["composite_worst_score"]),
        "composite_worst_score",
        "composite_best",
        ascending=True,
    )

    selection = pd.DataFrame(rows)
    save_table(selection, output_path)
    return selection


def _render_dataframe_table(
    df: pd.DataFrame, title: str, path_stem: Path, *, max_rows: int = 12
) -> None:
    display_df = df.head(max_rows).copy()
    fig_h = max(2.5, 0.45 * (len(display_df) + 2))
    fig, ax = plt.subplots(figsize=(max(8, 1.4 * len(display_df.columns)), fig_h))
    ax.axis("off")
    ax.set_title(title, fontsize=16, loc="left")
    rounded = display_df.copy()
    for column in rounded.columns:
        if pd.api.types.is_numeric_dtype(rounded[column]):
            rounded[column] = rounded[column].map(
                lambda value: f"{value:.3f}" if pd.notna(value) else ""
            )
    table = ax.table(cellText=rounded.values, colLabels=rounded.columns, loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.2)
    save_figure(fig, path_stem)


def _primary_rmse_column(target: str) -> str:
    return "rmse" if target in {"hs", "tp"} else "circular_rmse_deg"


def _site_metric_plot_frame(
    site_metrics: pd.DataFrame, *, value_kind: str = "rmse"
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for target in TARGET_ORDER:
        subset = site_metrics.loc[site_metrics["target"] == target].copy()
        if subset.empty:
            continue
        if value_kind == "rmse":
            col = _primary_rmse_column(target)
        elif value_kind == "skill":
            col = "skill"
        else:
            raise ValueError(f"Unsupported value_kind: {value_kind}")
        if col not in subset.columns:
            continue
        subset["plot_target"] = TARGET_LABELS[target]
        subset["plot_value"] = subset[col]
        if value_kind == "rmse":
            valid = pd.to_numeric(subset["plot_value"], errors="coerce")
            median_value = float(valid.median()) if valid.notna().any() else float("nan")
            if np.isfinite(median_value) and median_value > 0.0:
                subset["plot_value_normalized"] = valid / median_value
            else:
                subset["plot_value_normalized"] = np.nan
            subset["plot_value_normalization_label"] = "Relative to target median RMSE (=1)"
        else:
            subset["plot_value_normalized"] = subset["plot_value"]
            subset["plot_value_normalization_label"] = "Skill vs baseline"
        rows.append(subset)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _map_output_dir(context: AnalysisContext) -> Path:
    path = context.figures_dir / "maps"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _load_local_map_background() -> dict[str, Any] | None:
    candidates = [
        REPO_ROOT / "data" / "processed" / "bathy" / "bathy_field_project_site.npz",
        REPO_ROOT
        / "geometric_builder"
        / "data"
        / "processed"
        / "bathy"
        / "bathy_field_project_site.npz",
        REPO_ROOT / "data" / "processed" / "bathy" / "bathy_field_full.npz",
        REPO_ROOT / "geometric_builder" / "data" / "processed" / "bathy" / "bathy_field_full.npz",
    ]
    for path in candidates:
        if not path.exists():
            continue
        try:
            artifact = np.load(path, allow_pickle=True)
            x = np.asarray(artifact["x"], dtype=float)
            y = np.asarray(artifact["y"], dtype=float)
            z = np.asarray(artifact["z"], dtype=float)
            land_mask = (
                np.asarray(artifact["land_mask"], dtype=float)
                if "land_mask" in artifact.files
                else None
            )
            if z.ndim != 2 or x.ndim != 1 or y.ndim != 1:
                continue
            stride = max(1, int(math.ceil(max(len(x), len(y)) / 1600.0)))
            metadata = artifact["metadata"].item() if "metadata" in artifact.files else {}
            return {
                "path": path.resolve(),
                "x": x[::stride],
                "y": y[::stride],
                "z": z[::stride, ::stride],
                "land_mask": land_mask[::stride, ::stride] if land_mask is not None else None,
                "metadata": metadata if isinstance(metadata, dict) else {},
                "epsg": int((metadata or {}).get("epsg"))
                if isinstance((metadata or {}).get("epsg"), (int, np.integer))
                else None,
                "description": str((metadata or {}).get("product", path.stem)).replace("_", " "),
            }
        except Exception:
            continue
    return None


def _resolve_map_coordinates(
    site_df: pd.DataFrame,
    *,
    preferred_epsg: int | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    info = {
        "coordinate_mode": None,
        "coordinate_columns": [],
        "x_label": "",
        "y_label": "",
    }
    working = site_df.copy()

    projected_pairs = [("site_x", "site_y"), ("x", "y")]
    for x_col, y_col in projected_pairs:
        if {x_col, y_col} <= set(working.columns):
            mask = (
                pd.to_numeric(working[x_col], errors="coerce").notna()
                & pd.to_numeric(working[y_col], errors="coerce").notna()
            )
            if int(mask.sum()) >= 2:
                working["_map_x"] = pd.to_numeric(working[x_col], errors="coerce")
                working["_map_y"] = pd.to_numeric(working[y_col], errors="coerce")
                info.update(
                    {
                        "coordinate_mode": "projected",
                        "coordinate_columns": [x_col, y_col],
                        "x_label": f"Easting ({'EPSG:' + str(preferred_epsg) if preferred_epsg else 'projected'})",
                        "y_label": f"Northing ({'EPSG:' + str(preferred_epsg) if preferred_epsg else 'projected'})",
                    }
                )
                return working, info

    lon_lat_pairs = [("site_lon", "site_lat"), ("lon", "lat")]
    for lon_col, lat_col in lon_lat_pairs:
        if {lon_col, lat_col} <= set(working.columns):
            lon = pd.to_numeric(working[lon_col], errors="coerce")
            lat = pd.to_numeric(working[lat_col], errors="coerce")
            mask = lon.notna() & lat.notna()
            if int(mask.sum()) < 2:
                continue
            working["_map_lon"] = lon
            working["_map_lat"] = lat
            if preferred_epsg is not None and Transformer is not None:
                try:
                    transformer = Transformer.from_crs(
                        "EPSG:4326", f"EPSG:{preferred_epsg}", always_xy=True
                    )
                    map_x, map_y = transformer.transform(
                        lon.to_numpy(dtype=float), lat.to_numpy(dtype=float)
                    )
                    working["_map_x"] = map_x
                    working["_map_y"] = map_y
                    info.update(
                        {
                            "coordinate_mode": "projected_from_lonlat",
                            "coordinate_columns": [lon_col, lat_col],
                            "x_label": f"Easting (EPSG:{preferred_epsg})",
                            "y_label": f"Northing (EPSG:{preferred_epsg})",
                        }
                    )
                    return working, info
                except Exception:
                    pass
            working["_map_x"] = lon
            working["_map_y"] = lat
            info.update(
                {
                    "coordinate_mode": "lonlat",
                    "coordinate_columns": [lon_col, lat_col],
                    "x_label": "Longitude",
                    "y_label": "Latitude",
                }
            )
            return working, info

    return working, info


def _compute_map_extent(
    x_values: pd.Series,
    y_values: pd.Series,
    *,
    projected: bool,
    clip_extent: tuple[float, float, float, float] | None = None,
) -> tuple[float, float, float, float]:
    x = pd.to_numeric(x_values, errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(y_values, errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size == 0 or y.size == 0:
        if clip_extent is not None:
            return clip_extent
        return (0.0, 1.0, 0.0, 1.0)
    x_min = float(np.min(x))
    x_max = float(np.max(x))
    y_min = float(np.min(y))
    y_max = float(np.max(y))
    x_span = max(x_max - x_min, 1.0 if projected else 0.01)
    y_span = max(y_max - y_min, 1.0 if projected else 0.01)
    x_pad = max(0.08 * x_span, 2000.0 if projected else 0.04)
    y_pad = max(0.08 * y_span, 2000.0 if projected else 0.04)
    extent = (x_min - x_pad, x_max + x_pad, y_min - y_pad, y_max + y_pad)
    if clip_extent is None:
        return extent
    return (
        max(extent[0], clip_extent[0]),
        min(extent[1], clip_extent[1]),
        max(extent[2], clip_extent[2]),
        min(extent[3], clip_extent[3]),
    )


def _draw_local_map_background(
    ax: plt.Axes,
    background: dict[str, Any] | None,
) -> tuple[str, tuple[float, float, float, float] | None]:
    if background is None:
        ax.set_facecolor("#f5f7fa")
        return "plain axes", None
    x = background["x"]
    y = background["y"]
    z = background["z"]
    land_mask = background.get("land_mask")
    extent = (float(np.min(x)), float(np.max(x)), float(np.min(y)), float(np.max(y)))
    ax.set_facecolor("#eef2f5")
    water = np.where(np.isfinite(z), z, np.nan)
    water = np.where(land_mask >= 0.5, np.nan, water) if land_mask is not None else water
    ax.imshow(
        water,
        extent=extent,
        origin="lower",
        cmap="Blues",
        alpha=0.6,
        interpolation="nearest",
        zorder=0,
    )
    if land_mask is not None:
        land = np.where(land_mask >= 0.5, 1.0, np.nan)
        ax.imshow(
            land,
            extent=extent,
            origin="lower",
            cmap=mcolors.ListedColormap(["#d8d0c2"]),
            alpha=0.95,
            interpolation="nearest",
            zorder=1,
        )
    return f"local bathymetry / land mask ({background['path'].name})", extent


def _map_metric_specs(site_map_df: pd.DataFrame) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []

    deep_red_cmap = mcolors.LinearSegmentedColormap.from_list(
        "site_error_reds",
        ["#fff5f0", "#fcbba1", "#fb6a4a", "#cb181d", "#67000d"],
    )

    def _add(
        column: str,
        *,
        label: str,
        stem: str,
        cmap: Any,
        higher_is_worse: bool,
        size_metric: str | None = None,
    ) -> None:
        if column not in site_map_df.columns:
            return
        values = pd.to_numeric(site_map_df[column], errors="coerce")
        if not values.notna().any():
            return
        specs.append(
            {
                "column": column,
                "label": label,
                "stem": stem,
                "cmap": cmap,
                "higher_is_worse": higher_is_worse,
                "size_metric": size_metric,
            }
        )

    _add("hs_rmse", label="Hs RMSE (m)", stem="hs_rmse", cmap=deep_red_cmap, higher_is_worse=True)
    _add(
        "hs_skill",
        label="Hs Skill vs baseline",
        stem="hs_skill",
        cmap="coolwarm",
        higher_is_worse=False,
    )
    _add(
        "hs_error_contribution",
        label="Hs Error Contribution Fraction",
        stem="hs_error_contribution",
        cmap=deep_red_cmap,
        higher_is_worse=True,
    )
    _add(
        "composite_normalized_error",
        label="Composite Normalized Error",
        stem="composite_error",
        cmap="cividis",
        higher_is_worse=True,
    )
    _add("tp_rmse", label="Tp RMSE (s)", stem="tp_rmse", cmap=deep_red_cmap, higher_is_worse=True)
    _add(
        "dir_circular_rmse_deg",
        label="Dir Circular RMSE (deg)",
        stem="dir_circular_rmse",
        cmap=deep_red_cmap,
        higher_is_worse=True,
    )
    _add(
        "dp_circular_rmse_deg",
        label="Dp Circular RMSE (deg)",
        stem="dp_circular_rmse",
        cmap=deep_red_cmap,
        higher_is_worse=True,
    )
    return specs


def build_site_map_plotting_table(
    site_composite: pd.DataFrame,
    error_contributions: pd.DataFrame,
    output_path: Path,
) -> pd.DataFrame:
    if site_composite.empty:
        save_table(site_composite, output_path)
        return site_composite

    table = site_composite.copy()
    if (
        "composite_worst_score" in table.columns
        and "composite_normalized_error" not in table.columns
    ):
        table["composite_normalized_error"] = table["composite_worst_score"]
    if "composite_worst_rank" in table.columns and "composite_error_rank" not in table.columns:
        table["composite_error_rank"] = table["composite_worst_rank"]

    if not error_contributions.empty:
        contrib = error_contributions.pivot_table(
            index="site", columns="target", values="error_contribution_fraction", aggfunc="first"
        )
        contrib.columns = [f"{target}_error_contribution" for target in contrib.columns]
        contrib = contrib.reset_index()
        table = table.merge(contrib, on="site", how="left")

    save_table(table, output_path)
    return table


def _label_extreme_sites(
    ax: plt.Axes,
    plot_df: pd.DataFrame,
    *,
    metric_col: str,
    higher_is_worse: bool,
    n_labels: int = 5,
) -> None:
    subset = plot_df.dropna(subset=["_map_x", "_map_y", metric_col]).copy()
    if subset.empty:
        return
    ranked = subset.sort_values(metric_col, ascending=not higher_is_worse).head(n_labels)
    x_span = float(subset["_map_x"].max() - subset["_map_x"].min()) if len(subset) > 1 else 1.0
    y_span = float(subset["_map_y"].max() - subset["_map_y"].min()) if len(subset) > 1 else 1.0
    dx = max(x_span * 0.01, 400.0 if subset["_map_x"].abs().max() > 1000 else 0.01)
    dy = max(y_span * 0.01, 400.0 if subset["_map_y"].abs().max() > 1000 else 0.01)
    for _, row in ranked.iterrows():
        ax.text(
            float(row["_map_x"]) + dx,
            float(row["_map_y"]) + dy,
            str(row["site"]),
            fontsize=8,
            ha="left",
            va="bottom",
            color="#111111",
            bbox={
                "boxstyle": "round,pad=0.18",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.75,
            },
            zorder=6,
        )


def _metric_norm(values: pd.Series, metric_col: str) -> mcolors.Normalize | None:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return None
    if metric_col.endswith("_skill"):
        vmin = float(clean.min())
        vmax = float(clean.max())
        if vmin < 0.0 < vmax:
            return mcolors.TwoSlopeNorm(vcenter=0.0, vmin=vmin, vmax=vmax)
    return mcolors.Normalize(vmin=float(clean.min()), vmax=float(clean.max()))


def _build_point_sizes(plot_df: pd.DataFrame, size_metric: str | None) -> np.ndarray:
    if size_metric is None or size_metric not in plot_df.columns:
        return np.full(len(plot_df), 48.0)
    values = pd.to_numeric(plot_df[size_metric], errors="coerce")
    clean = values.dropna()
    if clean.empty:
        return np.full(len(plot_df), 48.0)
    vmin = float(clean.min())
    vmax = float(clean.max())
    if not np.isfinite(vmin) or not np.isfinite(vmax) or math.isclose(vmin, vmax):
        return np.full(len(plot_df), 48.0)
    scaled = (values - vmin) / (vmax - vmin)
    return np.where(np.isfinite(scaled), 48.0 + 0.0 * scaled, 48.0)


def _plot_discrete_site_map(
    context: AnalysisContext,
    plot_df: pd.DataFrame,
    metric_spec: Mapping[str, Any],
    *,
    background: dict[str, Any] | None,
    extent_hint: tuple[float, float, float, float] | None,
    output_dir: Path,
) -> list[Path]:
    metric_col = str(metric_spec["column"])
    subset = plot_df.dropna(subset=["_map_x", "_map_y", metric_col]).copy()
    if subset.empty:
        return []
    is_contribution_map = metric_col.endswith("_error_contribution")
    is_rmse_map = "rmse" in metric_col
    fig, ax = plt.subplots(figsize=(12.6, 8.9), constrained_layout=True)
    _draw_local_map_background(ax, background if extent_hint is not None else None)
    projected = str(plot_df.attrs.get("coordinate_mode", "")).startswith("projected")
    extent = _compute_map_extent(
        subset["_map_x"], subset["_map_y"], projected=projected, clip_extent=extent_hint
    )
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    norm = _metric_norm(subset[metric_col], metric_col)
    sizes = _build_point_sizes(
        subset, str(metric_spec.get("size_metric")) if metric_spec.get("size_metric") else None
    )
    scatter = ax.scatter(
        subset["_map_x"],
        subset["_map_y"],
        c=pd.to_numeric(subset[metric_col], errors="coerce"),
        s=sizes,
        cmap=metric_spec["cmap"],
        norm=norm,
        edgecolor="black",
        linewidth=0.6,
        alpha=0.92,
        zorder=5,
    )
    if not (is_contribution_map or is_rmse_map):
        _label_extreme_sites(
            ax, subset, metric_col=metric_col, higher_is_worse=bool(metric_spec["higher_is_worse"])
        )
    colorbar = fig.colorbar(scatter, ax=ax, shrink=0.76, fraction=0.034, pad=0.02, aspect=28)
    colorbar.set_label(str(metric_spec["label"]), fontsize=9, labelpad=7)
    colorbar.ax.tick_params(labelsize=8, length=2)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(False)
    ax.tick_params(labelsize=8)
    ax.margins(x=0.01, y=0.01)
    if is_contribution_map or is_rmse_map:
        ax.set_title(f"{metric_spec['label']} by held-out site", loc="left", fontsize=10, pad=6)
    else:
        ax.set_title(
            f"{context.model_name}: {metric_spec['label']} by held-out site",
            loc="left",
            fontsize=10,
            pad=6,
        )
    return save_figure(fig, output_dir / f"map_points_{metric_spec['stem']}")


def _voronoi_finite_polygons_2d(
    vor: Any, radius: float | None = None
) -> tuple[list[list[int]], np.ndarray]:
    if vor.points.shape[1] != 2:
        raise ValueError("Voronoi input must be 2D")
    new_regions: list[list[int]] = []
    new_vertices = vor.vertices.tolist()
    center = vor.points.mean(axis=0)
    if radius is None:
        radius = float(np.ptp(vor.points, axis=0).max() * 2.0)

    all_ridges: dict[int, list[tuple[int, int, int]]] = {}
    for (point_a, point_b), (vertex_a, vertex_b) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(point_a, []).append((point_b, vertex_a, vertex_b))
        all_ridges.setdefault(point_b, []).append((point_a, vertex_a, vertex_b))

    for point_idx, region_idx in enumerate(vor.point_region):
        vertices = vor.regions[region_idx]
        if all(vertex >= 0 for vertex in vertices):
            new_regions.append(vertices)
            continue

        ridges = all_ridges.get(point_idx, [])
        new_region = [vertex for vertex in vertices if vertex >= 0]
        for point_b, vertex_a, vertex_b in ridges:
            if vertex_b < 0:
                vertex_a, vertex_b = vertex_b, vertex_a
            if vertex_a >= 0:
                continue
            tangent = vor.points[point_b] - vor.points[point_idx]
            tangent /= np.linalg.norm(tangent)
            normal = np.array([-tangent[1], tangent[0]])
            midpoint = vor.points[[point_idx, point_b]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, normal)) * normal
            far_point = vor.vertices[vertex_b] + direction * radius
            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        region_vertices = np.asarray([new_vertices[vertex] for vertex in new_region])
        region_center = region_vertices.mean(axis=0)
        angles = np.arctan2(
            region_vertices[:, 1] - region_center[1], region_vertices[:, 0] - region_center[0]
        )
        new_regions.append([vertex for _, vertex in sorted(zip(angles, new_region))])

    return new_regions, np.asarray(new_vertices)


def _build_voronoi_polygons(
    plot_df: pd.DataFrame,
    *,
    extent: tuple[float, float, float, float],
) -> tuple[dict[str, Any], str | None]:
    if Voronoi is None or Polygon is None or box is None:
        return {}, "scipy Voronoi or shapely is unavailable"
    coord_df = plot_df.dropna(subset=["_map_x", "_map_y"]).copy()
    coord_df = coord_df.drop_duplicates(subset=["site"])
    if len(coord_df) < 3:
        return {}, "fewer than 3 sites had usable coordinates"
    points = coord_df[["_map_x", "_map_y"]].to_numpy(dtype=float)
    if np.linalg.matrix_rank(points - points.mean(axis=0, keepdims=True)) < 2:
        return {}, "site coordinates are effectively collinear"
    try:
        vor = Voronoi(points)
        regions, vertices = _voronoi_finite_polygons_2d(vor)
        domain = box(extent[0], extent[2], extent[1], extent[3])
        polygons: dict[str, Any] = {}
        for idx, region in enumerate(regions):
            polygon = Polygon(vertices[region]).buffer(0)
            if polygon.is_empty:
                continue
            clipped = polygon.intersection(domain).buffer(0)
            if clipped.is_empty:
                continue
            polygons[str(coord_df.iloc[idx]["site"])] = clipped
        if len(polygons) < 3:
            return {}, "Voronoi polygons could not be clipped to the plotting domain"
        return polygons, None
    except Exception as exc:
        return {}, f"Voronoi construction failed: {exc}"


def _plot_voronoi_site_map(
    context: AnalysisContext,
    plot_df: pd.DataFrame,
    metric_spec: Mapping[str, Any],
    *,
    polygons: Mapping[str, Any],
    background: dict[str, Any] | None,
    extent_hint: tuple[float, float, float, float] | None,
    output_dir: Path,
) -> list[Path]:
    metric_col = str(metric_spec["column"])
    subset = plot_df.dropna(subset=["_map_x", "_map_y", metric_col]).copy()
    if subset.empty:
        return []
    is_contribution_map = metric_col.endswith("_error_contribution")
    patches: list[MplPolygon] = []
    values: list[float] = []
    for _, row in subset.iterrows():
        geom = polygons.get(str(row["site"]))
        if geom is None or getattr(geom, "is_empty", True):
            continue
        parts = [geom] if geom.geom_type == "Polygon" else list(getattr(geom, "geoms", []))
        for part in parts:
            patches.append(MplPolygon(np.asarray(part.exterior.coords), closed=True))
            values.append(float(row[metric_col]))
    if not patches:
        return []
    fig, ax = plt.subplots(figsize=(12.6, 8.9), constrained_layout=True)
    _draw_local_map_background(ax, background if extent_hint is not None else None)
    projected = str(plot_df.attrs.get("coordinate_mode", "")).startswith("projected")
    extent = _compute_map_extent(
        subset["_map_x"], subset["_map_y"], projected=projected, clip_extent=extent_hint
    )
    collection = PatchCollection(
        patches,
        cmap=metric_spec["cmap"],
        norm=_metric_norm(subset[metric_col], metric_col),
        edgecolor="white",
        linewidth=0.6,
        alpha=0.72,
        zorder=3,
    )
    collection.set_array(np.asarray(values, dtype=float))
    ax.add_collection(collection)
    ax.scatter(
        subset["_map_x"],
        subset["_map_y"],
        c=pd.to_numeric(subset[metric_col], errors="coerce"),
        cmap=metric_spec["cmap"],
        norm=_metric_norm(subset[metric_col], metric_col),
        s=30,
        edgecolor="black",
        linewidth=0.6,
        zorder=5,
    )
    if not is_contribution_map:
        _label_extreme_sites(
            ax, subset, metric_col=metric_col, higher_is_worse=bool(metric_spec["higher_is_worse"])
        )
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(False)
    ax.tick_params(labelsize=8)
    ax.margins(x=0.01, y=0.01)
    if is_contribution_map:
        ax.set_title(
            f"Nearest-site zones for {metric_spec['label']}", loc="left", fontsize=10, pad=6
        )
    else:
        ax.set_title(
            f"{context.model_name}: nearest-site Voronoi zones for {metric_spec['label']}",
            loc="left",
            fontsize=10,
            pad=6,
        )
    colorbar = fig.colorbar(collection, ax=ax, shrink=0.76, fraction=0.034, pad=0.02, aspect=28)
    colorbar.set_label(str(metric_spec["label"]), fontsize=9, labelpad=7)
    colorbar.ax.tick_params(labelsize=8, length=2)
    return save_figure(fig, output_dir / f"map_voronoi_{metric_spec['stem']}")


def _save_voronoi_geojson(
    plot_df: pd.DataFrame, polygons: Mapping[str, Any], output_path: Path
) -> Path | None:
    if not polygons or mapping is None:
        return None
    metric_cols = [
        col
        for col in [
            "hs_rmse",
            "hs_skill",
            "hs_error_contribution",
            "tp_rmse",
            "dir_circular_rmse_deg",
            "dp_circular_rmse_deg",
            "composite_normalized_error",
            "composite_error_rank",
        ]
        if col in plot_df.columns
    ]
    rows = plot_df.set_index("site")
    features: list[dict[str, Any]] = []
    for site_name, geom in polygons.items():
        if site_name not in rows.index or getattr(geom, "is_empty", True):
            continue
        row = rows.loc[site_name]
        properties: dict[str, Any] = {"site": str(site_name)}
        for column in metric_cols + [
            col
            for col in ["regime_label", "site_lat", "site_lon", "site_x", "site_y"]
            if col in rows.columns
        ]:
            value = row.get(column)
            if pd.isna(value):
                properties[column] = None
            elif isinstance(value, (np.floating, float)):
                properties[column] = float(value)
            elif isinstance(value, (np.integer, int)):
                properties[column] = int(value)
            else:
                properties[column] = str(value)
        features.append({"type": "Feature", "geometry": mapping(geom), "properties": properties})
    if not features:
        return None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps({"type": "FeatureCollection", "features": features}), encoding="utf-8"
    )
    return output_path


def create_map_figures(
    context: AnalysisContext,
    *,
    site_composite: pd.DataFrame,
    error_contributions: pd.DataFrame,
) -> tuple[dict[str, list[Path]], dict[str, Any]]:
    outputs: dict[str, list[Path]] = {}
    summary: dict[str, Any] = {
        "background_used": "plain axes",
        "coordinate_columns": [],
        "metrics_mapped": [],
        "saved_files": [],
        "skipped_items": [],
    }
    map_output_dir = _map_output_dir(context)
    plot_table = build_site_map_plotting_table(
        site_composite, error_contributions, context.tables_dir / "site_map_plotting_table.csv"
    )
    outputs["site_map_plotting_table"] = [context.tables_dir / "site_map_plotting_table.csv"]

    background = _load_local_map_background()
    if background is None:
        context.warning_log.append(
            "No local bathymetry or land-mask background was found; site maps will use plain axes."
        )
    else:
        context.reuse_log.append(f"Loaded local map background: {background['path']}")
        summary["background_used"] = f"{background['description']} ({background['path'].name})"

    plot_df, coord_info = _resolve_map_coordinates(
        plot_table, preferred_epsg=background.get("epsg") if background else None
    )
    summary["coordinate_columns"] = list(coord_info.get("coordinate_columns", []))
    plot_df.attrs.update(coord_info)
    if coord_info.get("coordinate_mode") is None:
        searched = ["site_x/site_y", "x/y", "site_lon/site_lat", "lon/lat"]
        message = f"Skipped map generation because no usable coordinate columns were found. Searched: {', '.join(searched)}."
        context.warning_log.append(message)
        summary["skipped_items"].append(message)
        return outputs, summary

    if background is not None and str(coord_info.get("coordinate_mode")).startswith("projected"):
        extent_hint = (
            float(np.min(background["x"])),
            float(np.max(background["x"])),
            float(np.min(background["y"])),
            float(np.max(background["y"])),
        )
    else:
        extent_hint = None
        if background is not None:
            warning = "Map coordinates are not projected into the local bathymetry CRS, so maps will use plain axes without the bathymetry background."
            context.warning_log.append(warning)
            summary["skipped_items"].append(warning)
            summary["background_used"] = "plain axes"

    metric_specs = _map_metric_specs(plot_df)
    summary["metrics_mapped"] = [str(spec["column"]) for spec in metric_specs]
    if not metric_specs:
        message = "No map-ready site metrics were available for plotting."
        context.warning_log.append(message)
        summary["skipped_items"].append(message)
        return outputs, summary

    for spec in metric_specs:
        written = _plot_discrete_site_map(
            context,
            plot_df,
            spec,
            background=background,
            extent_hint=extent_hint,
            output_dir=map_output_dir,
        )
        if written:
            outputs[f"map_points_{spec['stem']}"] = written
            summary["saved_files"].extend([str(path) for path in written])

    voronoi_extent = _compute_map_extent(
        plot_df["_map_x"],
        plot_df["_map_y"],
        projected=str(coord_info.get("coordinate_mode", "")).startswith("projected"),
        clip_extent=extent_hint,
    )
    polygons, voronoi_skip_reason = _build_voronoi_polygons(plot_df, extent=voronoi_extent)
    if polygons:
        geojson_path = _save_voronoi_geojson(
            plot_df, polygons, map_output_dir / "site_voronoi_metrics.geojson"
        )
        if geojson_path is not None:
            outputs["site_voronoi_geojson"] = [geojson_path]
            summary["saved_files"].append(str(geojson_path))
        for spec in metric_specs:
            if spec["column"] not in {
                "hs_rmse",
                "hs_skill",
                "hs_error_contribution",
                "composite_normalized_error",
            }:
                continue
            written = _plot_voronoi_site_map(
                context,
                plot_df,
                spec,
                polygons=polygons,
                background=background,
                extent_hint=extent_hint,
                output_dir=map_output_dir,
            )
            if written:
                outputs[f"map_voronoi_{spec['stem']}"] = written
                summary["saved_files"].extend([str(path) for path in written])
    elif voronoi_skip_reason:
        summary["skipped_items"].append(f"Skipped Voronoi maps: {voronoi_skip_reason}.")
        context.warning_log.append(f"Skipped Voronoi maps: {voronoi_skip_reason}.")

    return outputs, summary


def create_figures(
    context: AnalysisContext,
    *,
    overall_metrics: pd.DataFrame,
    site_metrics: pd.DataFrame,
    error_contributions: pd.DataFrame,
    categories: pd.DataFrame,
    site_metrics_with_features: pd.DataFrame,
    site_composite: pd.DataFrame,
    feature_correlations: pd.DataFrame,
    analog_distances: pd.DataFrame,
    regime_summary: pd.DataFrame,
    case_study_selection: pd.DataFrame,
) -> tuple[dict[str, list[Path]], dict[str, Any]]:
    outputs: dict[str, list[Path]] = {}
    _render_dataframe_table(
        overall_metrics, "Overall Metrics", context.figures_dir / "overall_metrics_table"
    )
    outputs["overall_metrics_table"] = [
        context.figures_dir / "overall_metrics_table.png",
        context.figures_dir / "overall_metrics_table.pdf",
    ]

    rmse_frame = _site_metric_plot_frame(site_metrics, value_kind="rmse")
    if not rmse_frame.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.violinplot(
            data=rmse_frame,
            x="plot_target",
            y="plot_value_normalized",
            inner=None,
            cut=0,
            ax=ax,
            color="#cfe8f3",
        )
        sns.stripplot(
            data=rmse_frame,
            x="plot_target",
            y="plot_value_normalized",
            ax=ax,
            color="#1f4e79",
            alpha=0.55,
            size=4,
        )
        ax.set_xlabel("Target")
        ax.set_ylabel("Per-site primary RMSE / target median RMSE")
        ax.axhline(1.0, color="black", linewidth=1.0, linestyle="--", alpha=0.7)
        ax.set_title("Per-site RMSE distribution by target (normalized)", loc="left")
        outputs["sitewise_rmse_boxplot"] = save_figure(
            fig, context.figures_dir / "sitewise_rmse_boxplot"
        )

    skill_frame = _site_metric_plot_frame(site_metrics.dropna(subset=["skill"]), value_kind="skill")
    if not skill_frame.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(data=skill_frame, x="plot_target", y="plot_value", ax=ax, color="#d9ead3")
        sns.stripplot(
            data=skill_frame,
            x="plot_target",
            y="plot_value",
            ax=ax,
            color="#38761d",
            alpha=0.55,
            size=4,
        )
        ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--")
        ax.set_xlabel("Target")
        ax.set_ylabel("Skill vs baseline")
        ax.set_title("Per-site skill distribution by target", loc="left")
        outputs["sitewise_skill_boxplot"] = save_figure(
            fig, context.figures_dir / "sitewise_skill_boxplot"
        )

    hs_contrib = (
        error_contributions.loc[error_contributions["target"] == "hs"].copy()
        if not error_contributions.empty
        else pd.DataFrame()
    )
    if not hs_contrib.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        top = hs_contrib.head(15).sort_values("error_contribution_fraction", ascending=True)
        ax.barh(top["site"], top["error_contribution_fraction"], color="#b45f06")
        ax.set_xlabel("Fraction of total Hs squared error")
        ax.set_ylabel("Site")
        ax.set_title("Top Hs error-contributing sites", loc="left")
        outputs["top_error_sites_bar_hs"] = save_figure(
            fig, context.figures_dir / "top_error_sites_bar_hs"
        )

    if not error_contributions.empty:
        targets = [
            target for target in TARGET_ORDER if target in error_contributions["target"].unique()
        ]
        for target in targets:
            subset = (
                error_contributions.loc[error_contributions["target"] == target]
                .sort_values("error_contribution_fraction", ascending=False)
                .reset_index(drop=True)
            )
            if subset.empty:
                continue
            fig, ax1 = plt.subplots(figsize=(10, 6))
            ax2 = ax1.twinx()
            ax1.bar(
                np.arange(len(subset)),
                subset["error_contribution_fraction"],
                color="#cc4c02",
                alpha=0.85,
            )
            ax2.plot(
                np.arange(len(subset)),
                subset["cumulative_error_contribution"],
                color="#1f4e79",
                marker="o",
                linewidth=2,
            )
            ax1.set_ylabel("Fraction of total squared error")
            ax2.set_ylabel("Cumulative contribution")
            ax1.set_xlabel(f"Sites ranked by {TARGET_LABELS[target]} error contribution")
            ax1.set_title(f"{TARGET_LABELS[target]} error contribution Pareto", loc="left")
            outputs[f"error_contribution_pareto_{target}"] = save_figure(
                fig, context.figures_dir / f"error_contribution_pareto_{target}"
            )

        fig, axes = plt.subplots(
            len(targets), 1, figsize=(10, max(4, 3.6 * len(targets))), sharex=False
        )
        if not isinstance(axes, np.ndarray):
            axes = np.asarray([axes])
        for ax, target in zip(axes, targets):
            subset = error_contributions.loc[error_contributions["target"] == target].sort_values(
                "error_contribution_fraction", ascending=False
            )
            ax.bar(
                np.arange(len(subset)),
                subset["error_contribution_fraction"],
                color="#9fc5e8",
                alpha=0.85,
            )
            ax.plot(
                np.arange(len(subset)),
                subset["cumulative_error_contribution"],
                color="#073763",
                linewidth=2,
            )
            ax.set_title(f"{TARGET_LABELS[target]} contribution", loc="left")
            ax.set_ylabel("Fraction")
        axes[-1].set_xlabel("Ranked sites")
        fig.suptitle("Error contribution Pareto across targets", x=0.01, ha="left")
        outputs["error_contribution_pareto_all_targets"] = save_figure(
            fig, context.figures_dir / "error_contribution_pareto_all_targets"
        )

    heatmap_cols = [
        col
        for col in [
            "hs_rmse",
            "tp_rmse",
            "dir_circular_rmse_deg",
            "dp_circular_rmse_deg",
            "hs_skill",
            "tp_skill",
            "dir_skill",
            "dp_skill",
        ]
        if col in site_composite.columns
    ]
    if heatmap_cols:
        heat = site_composite[["site"] + heatmap_cols].copy().set_index("site")
        heat = heat.rank(pct=True)
        if "composite_worst_score" in site_composite.columns:
            order = site_composite.sort_values("composite_worst_score", ascending=False)["site"]
            heat = heat.reindex(order)
        fig_h = max(6, min(24, 0.22 * len(heat.index)))
        fig, ax = plt.subplots(figsize=(12, fig_h))
        sns.heatmap(heat, cmap="viridis", ax=ax, cbar_kws={"label": "Percentile rank"})
        ax.set_title("Site metric heatmap sorted by composite rank", loc="left")
        ax.set_xlabel("Metric")
        ax.set_ylabel("Site")
        outputs["site_metric_heatmap"] = save_figure(
            fig, context.figures_dir / "site_metric_heatmap"
        )

    hs_metric = site_metrics.loc[site_metrics["target"] == "hs"].copy()
    if {"baseline_rmse", "rmse"} <= set(hs_metric.columns) and hs_metric[
        "baseline_rmse"
    ].notna().any():
        fig, ax = plt.subplots(figsize=(7, 7))
        scatter = hs_metric.dropna(subset=["baseline_rmse", "rmse"])
        ax.scatter(scatter["baseline_rmse"], scatter["rmse"], color="#1f4e79", alpha=0.7)
        lim_max = (
            float(np.nanmax([scatter["baseline_rmse"].max(), scatter["rmse"].max()]))
            if not scatter.empty
            else 1.0
        )
        ax.plot([0.0, lim_max], [0.0, lim_max], color="black", linestyle="--", linewidth=1.0)
        ax.set_xlabel("Baseline site RMSE (Hs)")
        ax.set_ylabel("Model site RMSE (Hs)")
        ax.set_title("Model vs baseline site Hs RMSE", loc="left")
        outputs["model_vs_baseline_site_scatter_hs"] = save_figure(
            fig, context.figures_dir / "model_vs_baseline_site_scatter_hs"
        )

    if {"site_lat", "site_lon"} <= set(
        site_composite.columns
    ) and "hs_rmse" in site_composite.columns:
        plot_df = site_composite.dropna(subset=["site_lat", "site_lon", "hs_rmse"]).copy()
        if not plot_df.empty:
            fig, ax = plt.subplots(figsize=(8, 10))
            sc = ax.scatter(
                plot_df["site_lon"],
                plot_df["site_lat"],
                c=plot_df["hs_rmse"],
                cmap="magma",
                s=55,
                edgecolor="black",
                linewidth=0.3,
            )
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            ax.set_title("Site-wise Hs RMSE map", loc="left")
            fig.colorbar(sc, ax=ax, label="Hs RMSE")
            outputs["site_rmse_map_hs"] = save_figure(fig, context.figures_dir / "site_rmse_map_hs")
        if "hs_skill" in site_composite.columns and site_composite["hs_skill"].notna().any():
            plot_df = site_composite.dropna(subset=["site_lat", "site_lon", "hs_skill"]).copy()
            if not plot_df.empty:
                fig, ax = plt.subplots(figsize=(8, 10))
                sc = ax.scatter(
                    plot_df["site_lon"],
                    plot_df["site_lat"],
                    c=plot_df["hs_skill"],
                    cmap="coolwarm",
                    s=55,
                    edgecolor="black",
                    linewidth=0.3,
                )
                ax.set_xlabel("Longitude")
                ax.set_ylabel("Latitude")
                ax.set_title("Site-wise Hs skill map", loc="left")
                fig.colorbar(sc, ax=ax, label="Hs skill")
                outputs["site_skill_map_hs"] = save_figure(
                    fig, context.figures_dir / "site_skill_map_hs"
                )

    if {"fjordness_score", "hs_rmse"} <= set(site_composite.columns):
        plot_df = site_composite.dropna(subset=["fjordness_score", "hs_rmse"])
        if not plot_df.empty:
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.regplot(
                data=plot_df,
                x="fjordness_score",
                y="hs_rmse",
                scatter_kws={"alpha": 0.7, "s": 40},
                ax=ax,
                color="#134f5c",
            )
            ax.set_xlabel("Fjordness score")
            ax.set_ylabel("Hs RMSE")
            ax.set_title("Hs RMSE vs fjordness", loc="left")
            outputs["error_vs_fjordness"] = save_figure(
                fig, context.figures_dir / "error_vs_fjordness"
            )

    fetch_feature = None
    for candidate in ["ray_fetch_mean_m", "open_sector_fraction", "ray_fetch_max_m"]:
        if candidate in site_composite.columns:
            fetch_feature = candidate
            break
    if fetch_feature is not None and "hs_rmse" in site_composite.columns:
        plot_df = site_composite.dropna(subset=[fetch_feature, "hs_rmse"])
        if not plot_df.empty:
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.regplot(
                data=plot_df,
                x=fetch_feature,
                y="hs_rmse",
                scatter_kws={"alpha": 0.7, "s": 40},
                ax=ax,
                color="#6a329f",
            )
            ax.set_xlabel(fetch_feature)
            ax.set_ylabel("Hs RMSE")
            ax.set_title("Hs RMSE vs fetch / openness", loc="left")
            outputs["error_vs_fetch"] = save_figure(fig, context.figures_dir / "error_vs_fetch")

    analog_col = f"mean_distance_top_{context.k_nearest_analog}"
    if analog_col in analog_distances.columns and "hs_rmse" in site_composite.columns:
        plot_df = site_composite.merge(
            analog_distances[["site", "nearest_training_distance", analog_col]],
            on="site",
            how="left",
        ).dropna(subset=["nearest_training_distance", "hs_rmse"])
        if not plot_df.empty:
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.regplot(
                data=plot_df,
                x="nearest_training_distance",
                y="hs_rmse",
                scatter_kws={"alpha": 0.7, "s": 40},
                ax=ax,
                color="#0b5394",
            )
            ax.set_xlabel("Nearest training analog distance")
            ax.set_ylabel("Hs RMSE")
            ax.set_title("Hs RMSE vs training analog distance", loc="left")
            outputs["error_vs_training_feature_distance"] = save_figure(
                fig, context.figures_dir / "error_vs_training_feature_distance"
            )

    if (
        "regime_label" in site_metrics_with_features.columns
        and site_metrics_with_features["regime_label"].notna().any()
    ):
        regime_plot = _site_metric_plot_frame(
            site_metrics_with_features.dropna(subset=["regime_label"]), value_kind="rmse"
        )
        if not regime_plot.empty:
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.boxplot(
                data=regime_plot,
                x="regime_label",
                y="plot_value_normalized",
                hue="plot_target",
                ax=ax,
            )
            ax.set_xlabel("Regime")
            ax.set_ylabel("Per-site primary RMSE / target median RMSE")
            ax.axhline(1.0, color="black", linewidth=1.0, linestyle="--", alpha=0.7)
            ax.set_title("Per-site RMSE by regime (normalized)", loc="left")
            ax.legend(title="Target", frameon=False)
            outputs["regimewise_performance_boxplot"] = save_figure(
                fig, context.figures_dir / "regimewise_performance_boxplot"
            )

    map_outputs, map_summary = create_map_figures(
        context,
        site_composite=site_composite,
        error_contributions=error_contributions,
    )
    outputs.update(map_outputs)
    create_case_study_figures(context, site_composite, case_study_selection)
    return outputs, map_summary


def create_case_study_figures(
    context: AnalysisContext, site_composite: pd.DataFrame, case_study_selection: pd.DataFrame
) -> None:
    if case_study_selection.empty:
        return
    selected_sites = case_study_selection["site"].astype(str).drop_duplicates().tolist()
    feature_reference = (
        site_composite.set_index("site") if not site_composite.empty else pd.DataFrame()
    )
    for site_name in selected_sites:
        site_df = context.predictions.loc[
            context.predictions["site"].astype(str) == str(site_name)
        ].copy()
        if site_df.empty:
            continue
        fig, axes = plt.subplots(3, 2, figsize=(16, 14))
        axes = axes.ravel()

        if {"target_hs", "pred_hs"} <= set(site_df.columns):
            axes[0].plot(site_df.index, site_df["target_hs"], label="Observed", color="#1f4e79")
            axes[0].plot(
                site_df.index, site_df["pred_hs"], label="Model", color="#d95f02", alpha=0.9
            )
            if "baseline_hs" in site_df.columns:
                axes[0].plot(
                    site_df.index,
                    site_df["baseline_hs"],
                    label="Baseline",
                    color="#38761d",
                    alpha=0.75,
                )
            axes[0].set_title("Hs time series", loc="left")
            axes[0].set_ylabel("Hs (m)")
            axes[0].legend(frameon=False)

        if {"target_tp", "pred_tp"} <= set(site_df.columns):
            axes[1].plot(site_df.index, site_df["target_tp"], label="Observed", color="#1f4e79")
            axes[1].plot(
                site_df.index, site_df["pred_tp"], label="Model", color="#d95f02", alpha=0.9
            )
            if "baseline_tp" in site_df.columns:
                axes[1].plot(
                    site_df.index,
                    site_df["baseline_tp"],
                    label="Baseline",
                    color="#38761d",
                    alpha=0.75,
                )
            axes[1].set_title("Tp time series", loc="left")
            axes[1].set_ylabel("Tp (s)")

        if {"target_hs", "pred_hs"} <= set(site_df.columns):
            axes[2].scatter(
                site_df["target_hs"],
                site_df["pred_hs"],
                s=14,
                alpha=0.55,
                color="#0b5394",
                label="Model",
            )
            lim = float(np.nanmax([site_df["target_hs"].max(), site_df["pred_hs"].max()]))
            if "baseline_hs" in site_df.columns:
                axes[2].scatter(
                    site_df["target_hs"],
                    site_df["baseline_hs"],
                    s=14,
                    alpha=0.35,
                    color="#6aa84f",
                    label="Baseline",
                )
                lim = float(np.nanmax([lim, site_df["baseline_hs"].max()]))
            axes[2].plot([0.0, lim], [0.0, lim], color="black", linestyle="--", linewidth=1.0)
            axes[2].set_title("Hs observed vs predicted", loc="left")
            axes[2].set_xlabel("Observed Hs (m)")
            axes[2].set_ylabel("Predicted Hs (m)")
            axes[2].legend(frameon=False)

        if {"target_hs", "pred_hs", "target_tp", "pred_tp"} <= set(site_df.columns):
            axes[3].plot(
                site_df.index,
                site_df["pred_hs"] - site_df["target_hs"],
                label="Hs residual",
                color="#d95f02",
            )
            axes[3].plot(
                site_df.index,
                site_df["pred_tp"] - site_df["target_tp"],
                label="Tp residual",
                color="#3c78d8",
            )
            axes[3].axhline(0.0, color="black", linewidth=1.0, linestyle="--")
            axes[3].set_title("Residual time series", loc="left")
            axes[3].legend(frameon=False)

        if {"target_dir_deg", "pred_dir_deg"} <= set(site_df.columns) or {
            "target_dp_deg",
            "pred_dp_deg",
        } <= set(site_df.columns):
            if {"target_dir_deg", "pred_dir_deg"} <= set(site_df.columns):
                axes[4].plot(
                    site_df.index,
                    np.abs(_circular_errors(site_df["target_dir_deg"], site_df["pred_dir_deg"])),
                    label="|Dir error|",
                    color="#674ea7",
                )
            if {"target_dp_deg", "pred_dp_deg"} <= set(site_df.columns):
                axes[4].plot(
                    site_df.index,
                    np.abs(_circular_errors(site_df["target_dp_deg"], site_df["pred_dp_deg"])),
                    label="|Dp error|",
                    color="#cc0000",
                )
            axes[4].set_title("Directional absolute error", loc="left")
            axes[4].set_ylabel("Degrees")
            axes[4].legend(frameon=False)
        else:
            axes[4].axis("off")

        if not feature_reference.empty and site_name in feature_reference.index:
            site_row = feature_reference.loc[site_name]
            bars = []
            for feature in STATIC_BAR_COLUMNS:
                if feature not in feature_reference.columns or not np.isfinite(
                    pd.to_numeric(site_row.get(feature), errors="coerce")
                ):
                    continue
                series = pd.to_numeric(feature_reference[feature], errors="coerce")
                std = float(series.std(ddof=0))
                if std <= 0.0 or not np.isfinite(std):
                    continue
                z_score = (float(site_row[feature]) - float(series.mean())) / std
                bars.append((feature, z_score))
            if bars:
                bar_df = pd.DataFrame(bars, columns=["feature", "z_score"]).sort_values("z_score")
                axes[5].barh(
                    bar_df["feature"],
                    bar_df["z_score"],
                    color=["#1f4e79" if value >= 0 else "#d95f02" for value in bar_df["z_score"]],
                )
                axes[5].axvline(0.0, color="black", linewidth=1.0)
                axes[5].set_title("Static feature z-scores vs split median", loc="left")
                axes[5].set_xlabel("z-score")
            else:
                axes[5].axis("off")
        else:
            axes[5].axis("off")

        fig.suptitle(f"Case study: {site_name}", x=0.01, ha="left")
        save_figure(fig, context.case_study_dir / f"{slugify(site_name)}_case_study")


def build_summary_text(
    context: AnalysisContext,
    *,
    overall_metrics: pd.DataFrame,
    error_contributions: pd.DataFrame,
    site_metrics: pd.DataFrame,
    feature_correlations: pd.DataFrame,
    percent_error_summary: pd.DataFrame | None = None,
) -> str:
    lines: list[str] = []
    if context.reuse_log:
        lines.append("Reused / discovered artifacts:")
        lines.extend([f"- {item}" for item in context.reuse_log])
    if context.warning_log:
        lines.append("")
        lines.append("Warnings:")
        lines.extend([f"- {item}" for item in context.warning_log])

    if not overall_metrics.empty:
        lines.append("")
        lines.append("Overall metrics:")
        for _, row in overall_metrics.iterrows():
            metric_value = (
                row["rmse"] if row["target"] in {"hs", "tp"} else row["circular_rmse_deg"]
            )
            unit = TARGET_UNITS[row["target"]]
            lines.append(
                f"- {TARGET_LABELS[row['target']]} RMSE: {metric_value:.3f} {unit} | bias: {row['bias']:.3f} | n={int(row['sample_count'])}"
            )

    hs_contrib = (
        error_contributions.loc[error_contributions["target"] == "hs"].copy()
        if not error_contributions.empty
        else pd.DataFrame()
    )
    if not hs_contrib.empty:
        top5 = float(hs_contrib.head(5)["error_contribution_fraction"].sum())
        top10 = float(hs_contrib.head(10)["error_contribution_fraction"].sum())
        lines.append("")
        lines.append(
            f"Hs top-5 sites explain {top5:.1%} of total squared error; top-10 explain {top10:.1%}."
        )
        lines.append("Top 10 Hs error-contribution sites:")
        for _, row in hs_contrib.head(10).iterrows():
            lines.append(f"- {row['site']}: {row['error_contribution_fraction']:.2%}")

    hs_metrics = site_metrics.loc[site_metrics["target"] == "hs"].copy()
    if not hs_metrics.empty and hs_metrics["skill"].notna().any():
        best = hs_metrics.sort_values("skill", ascending=False).head(3)
        worst = hs_metrics.sort_values("skill", ascending=True).head(3)
        lines.append("")
        lines.append("Best Hs skill sites:")
        for _, row in best.iterrows():
            lines.append(f"- {row['site']}: skill={row['skill']:.3f}")
        lines.append("Worst Hs skill sites:")
        for _, row in worst.iterrows():
            lines.append(f"- {row['site']}: skill={row['skill']:.3f}")

    if not feature_correlations.empty:
        lines.append("")
        lines.append("Strongest static-feature error correlations:")
        for _, row in (
            feature_correlations.sort_values("abs_spearman_r", ascending=False).head(8).iterrows()
        ):
            lines.append(
                f"- {row['metric_label']} vs {row['feature']}: Spearman={row['spearman_r']:.3f}, Pearson={row['pearson_r']:.3f}"
            )

    if percent_error_summary is not None and not percent_error_summary.empty:
        lines.append("")
        lines.append("Site-level percent error summary:")
        for _, row in percent_error_summary.iterrows():
            label = str(row.get("target_label", row.get("target", "")))
            median_abs = row.get("median_site_mean_abs_pct_error", np.nan)
            p90_abs = row.get("p90_site_mean_abs_pct_error", np.nan)
            share_10 = row.get("share_sites_mean_abs_pct_error_le_10", np.nan)
            share_20 = row.get("share_sites_mean_abs_pct_error_le_20", np.nan)
            lines.append(
                f"- {label}: 50% of sites have mean absolute percent error <= {median_abs:.1f}%; "
                f"90% <= {p90_abs:.1f}%; {share_10:.1f}% of sites are <= 10%; {share_20:.1f}% are <= 20%."
            )

    map_summary = context.map_summary or {}
    if map_summary:
        lines.append("")
        lines.append("Map outputs:")
        lines.append(f"- Background used: {map_summary.get('background_used', 'plain axes')}")
        coord_cols = map_summary.get("coordinate_columns") or []
        lines.append(f"- Coordinate columns: {', '.join(coord_cols) if coord_cols else 'missing'}")
        mapped = map_summary.get("metrics_mapped") or []
        lines.append(f"- Metrics mapped: {', '.join(mapped) if mapped else 'none'}")
        saved_files = map_summary.get("saved_files") or []
        lines.append(f"- Saved map files: {len(saved_files)}")
        for item in (map_summary.get("skipped_items") or [])[:6]:
            lines.append(f"- {item}")

    lines.append("")
    lines.append(f"Tables saved to: {context.tables_dir}")
    lines.append(f"Figures saved to: {context.figures_dir}")
    return "\n".join(lines)


def print_summary(
    context: AnalysisContext,
    *,
    overall_metrics: pd.DataFrame,
    error_contributions: pd.DataFrame,
    site_metrics: pd.DataFrame,
    feature_correlations: pd.DataFrame,
    percent_error_summary: pd.DataFrame | None = None,
) -> str:
    text = build_summary_text(
        context,
        overall_metrics=overall_metrics,
        error_contributions=error_contributions,
        site_metrics=site_metrics,
        feature_correlations=feature_correlations,
        percent_error_summary=percent_error_summary,
    )
    display(maybe_markdown(f"```text\n{text}\n```"))
    return text


def run_sitewise_analysis(
    context: AnalysisContext,
    *,
    case_study_top_n: int = 5,
) -> dict[str, pd.DataFrame]:
    tables = compute_metrics_tables(context)
    site_metrics = tables["site_metrics"]
    overall_metrics = tables["overall_metrics"]
    site_percent_error_summary = compute_site_percent_error_summary(
        site_metrics, context.tables_dir / "site_percent_error_summary.csv"
    )
    error_contributions = compute_error_contributions(context, site_metrics)
    categories = compute_site_performance_categories(
        site_metrics, error_contributions, context.tables_dir / "site_performance_categories.csv"
    )
    site_metrics_with_features = merge_static_features(context, site_metrics)
    site_composite = compute_site_composite_rankings(
        site_metrics_with_features, context.tables_dir / "site_composite_rankings.csv"
    )
    feature_correlations = compute_feature_error_correlations(
        site_composite, context.tables_dir / "site_feature_error_correlations.csv"
    )
    analog_distances = compute_training_analog_distances(
        context, site_composite, context.tables_dir / "site_training_analog_distances.csv"
    )
    if not analog_distances.empty:
        site_composite = site_composite.merge(analog_distances, on="site", how="left")
        save_table(site_composite, context.tables_dir / "site_composite_rankings.csv")
    regime_summary = compute_regime_summary(
        site_metrics_with_features, context.tables_dir / "site_regime_summary.csv"
    )
    case_study_selection = compute_case_study_site_selection(
        site_metrics,
        error_contributions,
        site_composite,
        top_n=int(max(1, case_study_top_n)),
        output_path=context.tables_dir / "case_study_site_selection.csv",
    )
    figure_outputs, map_summary = create_figures(
        context,
        overall_metrics=overall_metrics,
        site_metrics=site_metrics,
        error_contributions=error_contributions,
        categories=categories,
        site_metrics_with_features=site_metrics_with_features,
        site_composite=site_composite,
        feature_correlations=feature_correlations,
        analog_distances=analog_distances,
        regime_summary=regime_summary,
        case_study_selection=case_study_selection,
    )
    context.figure_outputs = figure_outputs
    context.map_summary = map_summary
    return {
        "overall_metrics": overall_metrics,
        "site_metrics": site_metrics,
        "site_percent_error_summary": site_percent_error_summary,
        "site_error_contributions": error_contributions,
        "site_performance_categories": categories,
        "site_metrics_with_static_features": site_metrics_with_features,
        "site_composite_rankings": site_composite,
        "site_feature_error_correlations": feature_correlations,
        "site_training_analog_distances": analog_distances,
        "site_regime_summary": regime_summary,
        "case_study_site_selection": case_study_selection,
        "figure_outputs": figure_outputs,
        "map_summary": map_summary,
    }
