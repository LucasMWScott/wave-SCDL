"""Helpers for the thesis dataset diagnostics notebook."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from matplotlib.lines import Line2D

from notebooks import multisource_notebook_helpers as nh
from src.point_centric_pipeline import apply_nora3_wave_direction_offset, load_site_timeseries

try:
    from IPython.display import Markdown, display
except Exception:  # pragma: no cover - notebook-only convenience
    Markdown = None

    def display(obj: Any) -> None:
        print(obj)


REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_DPI = 300
PLOT_COLORS = {
    "nearshore": "#1f4e79",
    "offshore": "#d95f02",
    "wind": "#2a9d8f",
    "train": "#1f77b4",
    "val": "#ff7f0e",
    "test": "#2ca02c",
}
SEASON_ORDER = ["DJF", "MAM", "JJA", "SON"]


def default_plot_settings() -> dict[str, Any]:
    return {
        "hist_bins": {
            "nearshore_hs": 60,
            "nearshore_tp": 60,
            "offshore_scalar": 60,
            "comparison_hs": 70,
            "comparison_tp": 70,
        },
        "rose_bins": {
            "nearshore": 18,
            "offshore": 18,
            "split": 18,
            "comparison": 18,
        },
        "comparison_scatter_sample_size": 250_000,
        "comparison_hexbin_sample_size": 1_000_000,
        "seasonal_box_sample_size": 250_000,
        "climatology_style": "raw_line",
        "climatology_smoothing_window": 3,
        "direction_point_size": 10,
        "direction_point_alpha": 0.35,
    }


def default_filter_settings(training_config: str | Path | None = None) -> dict[str, Any]:
    settings = {
        "enabled": False,
        "hs_min": 0.0,
        "tp_min": None,
        "match": "all",
        "statistic": "target_timestep",
        "apply_to_splits": ["train", "val", "test"],
    }
    if training_config is None:
        return settings
    cfg = read_yaml(training_config)
    sample_filter = (cfg.get("data", {}) or {}).get("sample_filter", {}) or {}
    settings.update(
        {
            "enabled": bool(sample_filter.get("enabled", settings["enabled"])),
            "hs_min": sample_filter.get("hs_min", settings["hs_min"]),
            "tp_min": sample_filter.get("tp_min", settings["tp_min"]),
            "match": str(sample_filter.get("match", settings["match"])),
            "statistic": str(sample_filter.get("statistic", settings["statistic"])),
            "apply_to_splits": list(
                sample_filter.get("apply_to_splits", settings["apply_to_splits"])
            ),
        }
    )
    return settings


def default_subsetting_settings(training_config: str | Path | None = None) -> dict[str, Any]:
    settings = {
        "enabled": False,
        "fraction": 1.0,
        "seed": 42,
        "mode": "per_site",
        "apply_to_splits": ["train"],
    }
    if training_config is None:
        return settings
    cfg = read_yaml(training_config)
    subsampling = (cfg.get("data", {}) or {}).get("train_sample_subsampling", {}) or {}
    settings.update(
        {
            "enabled": bool(subsampling.get("enabled", settings["enabled"])),
            "fraction": float(subsampling.get("fraction", settings["fraction"])),
            "seed": int(subsampling.get("seed", settings["seed"])),
            "mode": str(subsampling.get("mode", settings["mode"])),
        }
    )
    return settings


def default_site_selection_settings(window_days: int = 21) -> dict[str, Any]:
    return {
        "manual_sites": [],
        "use_auto_if_empty": True,
        "include_auto_val_test": True,
        "window_days": int(window_days),
    }


def resolve_path(path_like: str | Path, anchor: str | Path | None = None) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path.resolve()

    candidates = [(Path.cwd() / path).resolve(), (REPO_ROOT / path).resolve()]
    if anchor is not None:
        anchor_path = Path(anchor).resolve()
        anchor_dir = anchor_path.parent if anchor_path.suffix else anchor_path
        candidates.insert(1, (anchor_dir / path).resolve())

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def read_yaml(path: str | Path) -> dict[str, Any]:
    with resolve_path(path).open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def parse_time_bound(raw_value: str | pd.Timestamp, is_end: bool = False) -> pd.Timestamp:
    ts = pd.Timestamp(raw_value)
    if isinstance(raw_value, str) and "T" not in raw_value and " " not in raw_value and is_end:
        ts = ts + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
    return ts.tz_localize(None) if ts.tzinfo is not None else ts


def season_from_month(month: int) -> str:
    month = int(month)
    if month in (12, 1, 2):
        return "DJF"
    if month in (3, 4, 5):
        return "MAM"
    if month in (6, 7, 8):
        return "JJA"
    return "SON"


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


def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def save_fig(fig: plt.Figure, figure_dir: str | Path, name: str) -> list[Path]:
    out_dir = resolve_path(figure_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = slugify(name)
    written: list[Path] = []
    for suffix in (".png", ".pdf"):
        path = out_dir / f"{stem}{suffix}"
        fig.savefig(path, bbox_inches="tight")
        written.append(path)
    return written


def show_and_close(fig: plt.Figure) -> None:
    if "agg" not in plt.get_backend().lower():
        plt.show()
    else:
        fig.canvas.draw()
    plt.close(fig)


def maybe_markdown(text: str) -> Any:
    if Markdown is not None:
        return Markdown(text)
    return text


def load_sites_context(
    training_config: str | Path,
    sites_config: str | Path,
    point_centric_dir: str | Path,
) -> dict[str, Any]:
    training_config_path = resolve_path(training_config)
    sites_config_path = resolve_path(sites_config, anchor=training_config_path)
    point_centric_dir_path = resolve_path(point_centric_dir, anchor=training_config_path)

    sites_cfg = read_yaml(sites_config_path)
    metadata = nh.load_metadata(point_centric_dir_path)
    source_metadata = nh.load_source_metadata(point_centric_dir_path)
    split_sets = nh.resolve_split_sets(
        metadata, config_path=str(training_config_path), prefer="metadata"
    )

    split_by_site: dict[str, str] = {}
    for split_name, site_names in split_sets.items():
        normalized = "val" if split_name == "val" else split_name
        for site_name in site_names:
            split_by_site[str(site_name)] = normalized

    local_wind_map = {
        str(item.get("norac_point_name")): str(item.get("local_wind_point_name"))
        for item in (metadata.get("local_wind_assignments") or [])
        if item.get("norac_point_name") and item.get("local_wind_point_name")
    }
    site_to_sources = {
        str(site_name): [str(source) for source in source_names]
        for site_name, source_names in zip(
            source_metadata.get("target_sites", []) or [],
            source_metadata.get("source_names", []) or [],
            strict=False,
        )
    }
    metadata_site_to_wave_site = {
        str(key): str(value) for key, value in (metadata.get("site_to_wave_site") or {}).items()
    }
    site_to_wave_site = {
        site_name: str(source_names[0])
        for site_name, source_names in site_to_sources.items()
        if source_names
    }
    for site_name, wave_site in metadata_site_to_wave_site.items():
        site_to_wave_site.setdefault(site_name, wave_site)
    wave_to_wind_site = {
        str(key): str(value) for key, value in (metadata.get("wave_to_wind_site") or {}).items()
    }
    site_to_wave_site_overrides = {
        site_name: {
            "active_wave_site": active_wave_site,
            "metadata_wave_site": metadata_site_to_wave_site.get(site_name),
            "source_ranked_wave_sites": site_to_sources.get(site_name, []),
        }
        for site_name, active_wave_site in site_to_wave_site.items()
        if metadata_site_to_wave_site.get(site_name) not in {None, active_wave_site}
    }

    return {
        "training_config_path": training_config_path,
        "sites_config_path": sites_config_path,
        "point_centric_dir": point_centric_dir_path,
        "sites_cfg": sites_cfg,
        "metadata": metadata,
        "source_metadata": source_metadata,
        "split_sets": split_sets,
        "split_by_site": split_by_site,
        "local_wind_map": local_wind_map,
        "site_to_sources": site_to_sources,
        "site_to_wave_site": site_to_wave_site,
        "metadata_site_to_wave_site": metadata_site_to_wave_site,
        "site_to_wave_site_overrides": site_to_wave_site_overrides,
        "wave_to_wind_site": wave_to_wind_site,
    }


def _prepare_frame_common(
    frame: pd.DataFrame,
    source_name: str,
    source_col: str = "site",
) -> pd.DataFrame:
    out = frame.copy()
    out = out.reset_index().rename(
        columns={"time": "timestamp", out.index.name or "index": "timestamp"}
    )
    if "timestamp" not in out.columns:
        first_col = str(out.columns[0])
        out = out.rename(columns={first_col: "timestamp"})
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    out = out.dropna(subset=["timestamp"]).sort_values("timestamp")
    out[source_col] = str(source_name)
    return out


def _cast_category_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for column in columns:
        if column in out.columns:
            out[column] = out[column].astype("category")
    return out


def load_nearshore_raw(
    sites_cfg: dict[str, Any],
    sites_config_path: str | Path,
    split_by_site: dict[str, str],
    time_start: pd.Timestamp,
    time_end: pd.Timestamp,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    params_dir = resolve_path(sites_cfg["backend"]["norac_params_dir"], anchor=sites_config_path)
    required_cols = ["hs", "tp", "dir", "dp"]
    frames: dict[str, pd.DataFrame] = {}
    table_parts: list[pd.DataFrame] = []

    for site_info in sites_cfg.get("nearshore_sites", []):
        site_name = str(site_info["name"])
        frame = load_site_timeseries(site_name, str(params_dir))
        if frame.empty:
            continue
        frame = frame.loc[(frame.index >= time_start) & (frame.index <= time_end)].copy()
        for column in required_cols:
            if column not in frame.columns:
                frame[column] = np.nan
        frame = frame[required_cols]
        frame.index.name = "timestamp"
        frames[site_name] = frame

        part = _prepare_frame_common(frame, site_name, source_col="site")
        part = part[["site", "timestamp", *required_cols]]
        part["split"] = split_by_site.get(site_name, "train")
        part["year"] = part["timestamp"].dt.year.astype("int16")
        part["month"] = part["timestamp"].dt.month.astype("int8")
        part["month_start"] = part["timestamp"].dt.to_period("M").dt.to_timestamp()
        part["target_row_valid"] = np.isfinite(part[required_cols].to_numpy(dtype=float)).all(
            axis=1
        )
        for column in required_cols:
            part[column] = pd.to_numeric(part[column], errors="coerce").astype("float32")
        table_parts.append(part)

    table = pd.concat(table_parts, ignore_index=True)
    table = _cast_category_columns(table, ["site", "split"])
    return table, frames


def load_offshore_raw(
    sites_cfg: dict[str, Any],
    sites_config_path: str | Path,
    time_start: pd.Timestamp,
    time_end: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    params_dir = resolve_path(sites_cfg["backend"]["nora3_params_dir"], anchor=sites_config_path)
    wave_cols = [
        "hs",
        "tp",
        "tm1",
        "tm2",
        "Pdir",
        "thq",
        "hs_sea",
        "tp_sea",
        "thq_sea",
        "hs_swell",
        "tp_swell",
        "thq_swell",
    ]
    wind_cols = ["wind_speed_10m", "wind_direction_10m"]
    wave_frames: dict[str, pd.DataFrame] = {}
    wind_frames: dict[str, pd.DataFrame] = {}
    wave_parts: list[pd.DataFrame] = []
    wind_parts: list[pd.DataFrame] = []

    for site_info in sites_cfg.get("offshore_sites", []):
        site_name = str(site_info["name"])
        region = str(site_info.get("region", ""))
        frame = load_site_timeseries(site_name, str(params_dir))
        if frame.empty:
            continue
        frame = frame.loc[(frame.index >= time_start) & (frame.index <= time_end)].copy()
        frame.index.name = "timestamp"

        is_wind = "wind" in site_name.lower() or "wind" in region.lower()
        if is_wind:
            for column in wind_cols:
                if column not in frame.columns:
                    frame[column] = np.nan
            frame = frame[wind_cols]
            wind_frames[site_name] = frame
            part = _prepare_frame_common(frame, site_name, source_col="source")
            part = part[["source", "timestamp", *wind_cols]]
            for column in wind_cols:
                part[column] = pd.to_numeric(part[column], errors="coerce").astype("float32")
            wind_parts.append(part)
        else:
            for column in wave_cols:
                if column not in frame.columns:
                    frame[column] = np.nan
            frame = apply_nora3_wave_direction_offset(
                frame, ["Pdir", "thq", "thq_sea", "thq_swell"]
            )
            frame = frame[wave_cols]
            wave_frames[site_name] = frame
            part = _prepare_frame_common(frame, site_name, source_col="source")
            part = part[["source", "timestamp", *wave_cols]]
            for column in wave_cols:
                part[column] = pd.to_numeric(part[column], errors="coerce").astype("float32")
            wave_parts.append(part)

    wave_table = _cast_category_columns(pd.concat(wave_parts, ignore_index=True), ["source"])
    wind_table = _cast_category_columns(pd.concat(wind_parts, ignore_index=True), ["source"])
    return wave_table, wind_table, wave_frames, wind_frames


def load_processed_bundle(point_centric_dir: str | Path) -> dict[str, Any]:
    point_centric_dir = resolve_path(point_centric_dir)
    y_npz = np.load(point_centric_dir / "point_centric_Y_targets.npz", allow_pickle=True)
    site_dynamic_npz = np.load(
        point_centric_dir / "point_centric_X_dynamic_sitewise.npz", allow_pickle=True
    )
    timestamps = pd.to_datetime(y_npz["timestamps"], errors="coerce")
    return {
        "y_npz": y_npz,
        "site_dynamic_npz": site_dynamic_npz,
        "timestamps": timestamps,
        "target_sites": y_npz["target_sites"].astype(str).tolist(),
        "physical_target_names": y_npz["physical_target_names"].astype(str).tolist(),
        "reference_target_names": y_npz["reference_target_names"].astype(str).tolist(),
        "site_dynamic_feature_names": site_dynamic_npz["site_dynamic_feature_names"]
        .astype(str)
        .tolist(),
    }


def load_all_data(
    training_config: str | Path,
    sites_config: str | Path,
    point_centric_dir: str | Path,
    time_start: str,
    time_end: str,
    figure_dir: str | Path,
) -> dict[str, Any]:
    setup_plotting()
    time_start_ts = parse_time_bound(time_start, is_end=False)
    time_end_ts = parse_time_bound(time_end, is_end=True)

    context = load_sites_context(training_config, sites_config, point_centric_dir)
    nearshore_raw, nearshore_frames = load_nearshore_raw(
        context["sites_cfg"],
        context["sites_config_path"],
        context["split_by_site"],
        time_start_ts,
        time_end_ts,
    )
    offshore_wave_raw, offshore_wind_raw, offshore_wave_frames, offshore_wind_frames = (
        load_offshore_raw(
            context["sites_cfg"],
            context["sites_config_path"],
            time_start_ts,
            time_end_ts,
        )
    )
    processed = load_processed_bundle(context["point_centric_dir"])

    context.update(
        {
            "time_start": time_start_ts,
            "time_end": time_end_ts,
            "figure_dir": resolve_path(figure_dir),
            "nearshore_raw": nearshore_raw,
            "nearshore_frames": nearshore_frames,
            "offshore_wave_raw": offshore_wave_raw,
            "offshore_wind_raw": offshore_wind_raw,
            "offshore_wave_frames": offshore_wave_frames,
            "offshore_wind_frames": offshore_wind_frames,
            "processed": processed,
            "cache": {},
        }
    )
    return context


def build_provenance_markdown(context: dict[str, Any]) -> str:
    metadata = context["metadata"]
    multi_source = metadata.get("multi_source", {}) or {}
    transfer_reference = (metadata.get("targets", {}) or {}).get("transfer_reference", "unknown")
    override_count = len(context.get("site_to_wave_site_overrides", {}) or {})
    lines = [
        "## Provenance and Reference Choice",
        "",
        f"- Main analysis window: **{context['time_start'].date()} to {context['time_end'].date()}**.",
        "- Raw NORAC and NORA3 CSV files are used for coverage, missingness, and global raw-unit distributions.",
        "- Processed point-centric artifacts are used for aligned offshore-versus-nearshore comparisons and split-aware analyses.",
        "- The primary offshore reference source for the thesis comparison figures is the per-site nearest raw wave source from the ranked source list in `point_centric_source_metadata.json`.",
        f"- The stored processed transfer reference is **`{transfer_reference}`** in the current dataset metadata.",
        "- The notebook uses the first ranked wave source for each nearshore site as the active offshore comparison source and aligns that raw source directly to the processed timestamps.",
        f"- Multi-source preprocessing is enabled with **k = {multi_source.get('k_nearest', 'unknown')}** nearest offshore wave sources.",
        "- The notebook reports the 3-nearest source list from `point_centric_source_metadata.json`, but the nearest/reference source remains the main comparison target.",
        "- Split-aware plots use the site-heldout train/validation/test site lists stored in processed metadata, not the temporal index arrays from `point_centric_X_dynamic.npz`.",
    ]
    if override_count:
        lines.append(
            f"- The notebook detected **{override_count}** site-to-wave mappings where the older metadata mapping differs from the ranked nearest-source mapping, and it uses the ranked nearest-source mapping for the comparison figures."
        )
    return "\n".join(lines)


def print_dataset_summary(context: dict[str, Any]) -> pd.DataFrame:
    nearshore_raw = context["nearshore_raw"]
    offshore_wave_raw = context["offshore_wave_raw"]
    offshore_wind_raw = context["offshore_wind_raw"]
    summary = pd.DataFrame(
        [
            {
                "item": "Nearshore sites",
                "value": int(nearshore_raw["site"].nunique()),
            },
            {
                "item": "Offshore wave sites",
                "value": int(offshore_wave_raw["source"].nunique()),
            },
            {
                "item": "Offshore wind sites",
                "value": int(offshore_wind_raw["source"].nunique()),
            },
            {
                "item": "Nearshore site-time rows",
                "value": int(len(nearshore_raw)),
            },
            {
                "item": "Offshore wave rows",
                "value": int(len(offshore_wave_raw)),
            },
            {
                "item": "Offshore wind rows",
                "value": int(len(offshore_wind_raw)),
            },
            {
                "item": "Time start",
                "value": str(context["time_start"]),
            },
            {
                "item": "Time end",
                "value": str(context["time_end"]),
            },
        ]
    )
    display(summary)
    return summary


def plot_rose(
    ax: plt.Axes,
    angles_deg: np.ndarray | pd.Series,
    bins: int = 16,
    color: str = "#1f77b4",
    alpha: float = 0.7,
    label: str | None = None,
) -> None:
    values = np.asarray(pd.to_numeric(pd.Series(angles_deg), errors="coerce"), dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        ax.set_title("No finite data")
        return
    radians = np.deg2rad(np.mod(values, 360.0))
    edges = np.linspace(0.0, 2.0 * math.pi, bins + 1)
    counts, _ = np.histogram(radians, bins=edges)
    heights = counts / counts.sum()
    widths = np.diff(edges)
    ax.bar(
        edges[:-1],
        heights,
        width=widths,
        align="edge",
        color=color,
        alpha=alpha,
        edgecolor="white",
        linewidth=0.7,
        label=label,
    )
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_ylim(0, max(heights.max() * 1.15, 0.05))


def overlay_rose(
    ax: plt.Axes,
    series_list: list[tuple[np.ndarray | pd.Series, str, str]],
    bins: int = 16,
) -> None:
    for values, label, color in series_list:
        plot_rose(ax, values, bins=bins, color=color, alpha=0.45, label=label)
    ax.legend(loc="upper right", bbox_to_anchor=(1.15, 1.15), frameon=False)


def compute_missing_fraction_from_frames(
    frames: dict[str, pd.DataFrame],
    variables: list[str],
    family: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for variable in variables:
        total_count = 0
        missing_count = 0
        contributing_sources = 0
        for frame in frames.values():
            if variable not in frame.columns:
                continue
            contributing_sources += 1
            values = pd.to_numeric(frame[variable], errors="coerce").to_numpy(dtype=float)
            total_count += values.size
            missing_count += int((~np.isfinite(values)).sum())
        if total_count == 0:
            continue
        rows.append(
            {
                "variable": variable,
                "family": family,
                "contributing_sources": contributing_sources,
                "total_count": int(total_count),
                "missing_count": int(missing_count),
                "missing_fraction": missing_count / total_count,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["family", "missing_fraction", "variable"], ascending=[True, False, True]
    )


def _target_splits_mask(
    df: pd.DataFrame, splits: list[str] | tuple[str, ...] | set[str], split_col: str = "split"
) -> pd.Series:
    split_values = {str(item) for item in splits}
    return df[split_col].astype(str).isin(split_values)


def apply_target_filter(
    df: pd.DataFrame,
    filter_settings: dict[str, Any] | None,
    *,
    hs_col: str,
    tp_col: str,
    split_col: str = "split",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    settings = filter_settings or {}
    enabled = bool(settings.get("enabled", False))
    base = df.copy()
    summary_rows = [{"stage": "input", "rows": int(len(base))}]
    if not enabled:
        summary_rows.append({"stage": "filtered", "rows": int(len(base))})
        return base, pd.DataFrame(summary_rows)

    target_mask = _target_splits_mask(
        base, settings.get("apply_to_splits", ["train", "val", "test"]), split_col=split_col
    )
    conditions: list[pd.Series] = []
    hs_min = settings.get("hs_min", None)
    tp_min = settings.get("tp_min", None)
    if hs_min is not None:
        conditions.append(pd.to_numeric(base[hs_col], errors="coerce") >= float(hs_min))
    if tp_min is not None:
        conditions.append(pd.to_numeric(base[tp_col], errors="coerce") >= float(tp_min))
    if not conditions:
        summary_rows.append({"stage": "filtered", "rows": int(len(base))})
        return base, pd.DataFrame(summary_rows)

    if str(settings.get("match", "all")).lower() == "any":
        combined = conditions[0].copy()
        for cond in conditions[1:]:
            combined = combined | cond
    else:
        combined = conditions[0].copy()
        for cond in conditions[1:]:
            combined = combined & cond

    keep_mask = (~target_mask) | combined.fillna(False)
    filtered = base.loc[keep_mask].copy()
    summary_rows.extend(
        [
            {"stage": "eligible_for_filter", "rows": int(target_mask.sum())},
            {"stage": "filtered", "rows": int(len(filtered))},
            {"stage": "dropped_by_filter", "rows": int((~keep_mask).sum())},
        ]
    )
    return filtered, pd.DataFrame(summary_rows)


def apply_random_subsetting(
    df: pd.DataFrame,
    subset_settings: dict[str, Any] | None,
    *,
    group_col: str = "site",
    split_col: str = "split",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    settings = subset_settings or {}
    enabled = bool(settings.get("enabled", False))
    base = df.copy()
    summary_rows = [{"stage": "input", "rows": int(len(base))}]
    if not enabled:
        summary_rows.append({"stage": "subset", "rows": int(len(base))})
        return base, pd.DataFrame(summary_rows)

    fraction = float(settings.get("fraction", 1.0))
    fraction = min(max(fraction, 0.0), 1.0)
    seed = int(settings.get("seed", 42))
    mode = str(settings.get("mode", "per_site")).lower()
    target_mask = _target_splits_mask(
        base, settings.get("apply_to_splits", ["train"]), split_col=split_col
    )
    untouched = base.loc[~target_mask].copy()
    target = base.loc[target_mask].copy()

    if fraction >= 1.0 or target.empty:
        summary_rows.extend(
            [
                {"stage": "eligible_for_subset", "rows": int(len(target))},
                {"stage": "subset", "rows": int(len(base))},
                {"stage": "dropped_by_subset", "rows": 0},
            ]
        )
        return base, pd.DataFrame(summary_rows)

    sampled_parts: list[pd.DataFrame] = []
    if mode == "per_site" and group_col in target.columns:
        for idx, (_, group) in enumerate(target.groupby(group_col, observed=True, sort=False)):
            sample_n = int(round(len(group) * fraction))
            sample_n = min(len(group), max(1, sample_n)) if len(group) else 0
            if sample_n > 0:
                sampled_parts.append(group.sample(n=sample_n, random_state=seed + idx))
    else:
        sample_n = int(round(len(target) * fraction))
        sample_n = min(len(target), max(1, sample_n)) if len(target) else 0
        if sample_n > 0:
            sampled_parts.append(target.sample(n=sample_n, random_state=seed))

    sampled_target = (
        pd.concat(sampled_parts, ignore_index=False) if sampled_parts else target.iloc[0:0].copy()
    )
    subset = pd.concat([untouched, sampled_target], ignore_index=False)
    sort_cols = [col for col in ["site", "timestamp"] if col in subset.columns]
    if sort_cols:
        subset = subset.sort_values(sort_cols)
    summary_rows.extend(
        [
            {"stage": "eligible_for_subset", "rows": int(len(target))},
            {"stage": "subset", "rows": int(len(subset))},
            {"stage": "dropped_by_subset", "rows": int(len(target) - len(sampled_target))},
        ]
    )
    return subset, pd.DataFrame(summary_rows)


def _subset_to_keys(df: pd.DataFrame, keys_df: pd.DataFrame) -> pd.DataFrame:
    key_cols = ["site", "timestamp"]
    if not all(col in df.columns for col in key_cols):
        return df.copy()
    keys = keys_df[key_cols].drop_duplicates()
    return df.merge(keys, on=key_cols, how="inner")


def build_analysis_views(
    context: dict[str, Any],
    filter_settings: dict[str, Any] | None = None,
    subset_settings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    nearshore_raw = context["nearshore_raw"].copy()
    aligned_scalar = get_aligned_scalar_df(context).copy()
    aligned_direction = get_aligned_direction_df(context).copy()

    filtered_nearshore, filter_summary = apply_target_filter(
        nearshore_raw,
        filter_settings,
        hs_col="hs",
        tp_col="tp",
    )
    filtered_scalar, scalar_filter_summary = apply_target_filter(
        aligned_scalar,
        filter_settings,
        hs_col="nearshore_hs",
        tp_col="nearshore_tp",
    )
    filtered_direction = _subset_to_keys(aligned_direction, filtered_scalar)

    subset_nearshore, subset_summary = apply_random_subsetting(
        filtered_nearshore,
        subset_settings,
        group_col="site",
    )
    subset_scalar, scalar_subset_summary = apply_random_subsetting(
        filtered_scalar,
        subset_settings,
        group_col="site",
    )
    subset_direction = _subset_to_keys(filtered_direction, subset_scalar)

    return {
        "base_nearshore_raw": nearshore_raw,
        "filtered_nearshore_raw": filtered_nearshore,
        "subset_nearshore_raw": subset_nearshore,
        "base_aligned_scalar": aligned_scalar,
        "filtered_aligned_scalar": filtered_scalar,
        "subset_aligned_scalar": subset_scalar,
        "base_aligned_direction": aligned_direction,
        "filtered_aligned_direction": filtered_direction,
        "subset_aligned_direction": subset_direction,
        "filter_summary": filter_summary,
        "scalar_filter_summary": scalar_filter_summary,
        "subset_summary": subset_summary,
        "scalar_subset_summary": scalar_subset_summary,
    }


def run_coverage_section(
    context: dict[str, Any],
    figure_dir: str | Path | None = None,
    *,
    figure_tag: str = "",
    title_prefix: str = "",
) -> dict[str, pd.DataFrame]:
    figure_dir = figure_dir or context["figure_dir"]
    tag = f"{figure_tag}_" if figure_tag else ""
    prefix = title_prefix
    nearshore_raw = context["nearshore_raw"]

    monthly_counts = (
        nearshore_raw.groupby("month_start", observed=True)["target_row_valid"]
        .sum()
        .rename("valid_target_rows")
        .reset_index()
        .sort_values("month_start")
    )
    yearly_counts = (
        nearshore_raw.groupby("year", observed=True)["target_row_valid"]
        .sum()
        .rename("valid_target_rows")
        .reset_index()
        .sort_values("year")
    )
    target_valid_counts = pd.DataFrame(
        {
            "variable": ["hs", "tp", "dir", "dp"],
            "valid_count": [
                int(np.isfinite(nearshore_raw[col]).sum()) for col in ["hs", "tp", "dir", "dp"]
            ],
            "missing_fraction": [
                float((~np.isfinite(nearshore_raw[col])).mean())
                for col in ["hs", "tp", "dir", "dp"]
            ],
        }
    )
    offshore_missingness = pd.concat(
        [
            compute_missing_fraction_from_frames(
                context["offshore_wave_frames"],
                [
                    "hs",
                    "tp",
                    "Pdir",
                    "thq",
                    "hs_sea",
                    "tp_sea",
                    "thq_sea",
                    "hs_swell",
                    "tp_swell",
                    "thq_swell",
                ],
                family="wave",
            ),
            compute_missing_fraction_from_frames(
                context["offshore_wind_frames"],
                ["wind_speed_10m", "wind_direction_10m"],
                family="wind",
            ),
        ],
        ignore_index=True,
    )

    availability_heatmap = (
        nearshore_raw.groupby(["site", "month_start"], observed=True)["target_row_valid"]
        .mean()
        .unstack(fill_value=np.nan)
        .sort_index()
    )

    fig, axes = plt.subplots(2, 1, figsize=(16, 10), constrained_layout=True)
    axes[0].plot(
        monthly_counts["month_start"],
        monthly_counts["valid_target_rows"],
        color=PLOT_COLORS["nearshore"],
        linewidth=2,
    )
    axes[0].set_title(f"{prefix}Valid Nearshore Target Rows by Month")
    axes[0].set_ylabel("Valid rows")
    axes[1].bar(
        yearly_counts["year"].astype(str),
        yearly_counts["valid_target_rows"],
        color=PLOT_COLORS["nearshore"],
    )
    axes[1].set_title(f"{prefix}Valid Nearshore Target Rows by Year")
    axes[1].set_ylabel("Valid rows")
    save_fig(fig, figure_dir, f"{tag}coverage_valid_target_counts")
    show_and_close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)
    sns.barplot(
        data=target_valid_counts,
        x="variable",
        y="valid_count",
        ax=axes[0],
        color=PLOT_COLORS["nearshore"],
    )
    axes[0].set_title(f"{prefix}Valid Sample Count per Target Variable")
    axes[0].set_ylabel("Valid samples")
    sns.barplot(
        data=target_valid_counts, x="variable", y="missing_fraction", ax=axes[1], color="#c44e52"
    )
    axes[1].set_title(f"{prefix}Missing / Non-finite Fraction per Target Variable")
    axes[1].set_ylabel("Fraction")
    save_fig(fig, figure_dir, f"{tag}coverage_target_variable_quality")
    show_and_close(fig)

    fig, ax = plt.subplots(figsize=(12, 7), constrained_layout=True)
    sns.barplot(
        data=offshore_missingness.sort_values("missing_fraction", ascending=False),
        x="missing_fraction",
        y="variable",
        hue="family",
        dodge=False,
        ax=ax,
        palette={"wave": PLOT_COLORS["offshore"], "wind": PLOT_COLORS["wind"]},
    )
    ax.set_title(f"{prefix}Missing / Non-finite Fraction per Offshore Dynamic Variable")
    ax.set_xlabel("Missing fraction")
    ax.set_ylabel("")
    ax.legend(frameon=False)
    save_fig(fig, figure_dir, f"{tag}coverage_offshore_dynamic_missing_fraction")
    show_and_close(fig)

    fig_height = max(10, availability_heatmap.shape[0] * 0.12)
    fig, ax = plt.subplots(figsize=(18, fig_height), constrained_layout=True)
    sns.heatmap(
        availability_heatmap,
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        cbar_kws={"label": "Valid-row fraction"},
        ax=ax,
    )
    ax.set_title(f"{prefix}Monthly Nearshore Target Availability by Site")
    ax.set_xlabel("Month")
    ax.set_ylabel("Site")
    save_fig(fig, figure_dir, f"{tag}coverage_monthly_site_availability_heatmap")
    show_and_close(fig)

    display(target_valid_counts)
    display(offshore_missingness)
    return {
        "monthly_counts": monthly_counts,
        "yearly_counts": yearly_counts,
        "target_valid_counts": target_valid_counts,
        "offshore_missingness": offshore_missingness,
        "availability_heatmap": availability_heatmap,
    }


def run_nearshore_distribution_section(
    context: dict[str, Any],
    figure_dir: str | Path | None = None,
    *,
    nearshore_df: pd.DataFrame | None = None,
    plot_settings: dict[str, Any] | None = None,
    figure_tag: str = "",
    title_prefix: str = "",
) -> dict[str, Any]:
    figure_dir = figure_dir or context["figure_dir"]
    plot_settings = plot_settings or default_plot_settings()
    hist_bins = plot_settings.get("hist_bins", {})
    rose_bins = plot_settings.get("rose_bins", {})
    df = nearshore_df.copy() if nearshore_df is not None else context["nearshore_raw"].copy()
    finite_hs = df.loc[np.isfinite(df["hs"]), "hs"].to_numpy(dtype=float)
    finite_tp = df.loc[np.isfinite(df["tp"]), "tp"].to_numpy(dtype=float)
    tag = f"{figure_tag}_" if figure_tag else ""
    prefix = title_prefix

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    sns.histplot(
        finite_hs,
        bins=int(hist_bins.get("nearshore_hs", 60)),
        color=PLOT_COLORS["nearshore"],
        ax=ax,
    )
    ax.set_title(f"{prefix}Nearshore Significant Wave Height (Hs)")
    ax.set_xlabel("Hs [m]")
    save_fig(fig, figure_dir, f"{tag}nearshore_hs_histogram")
    show_and_close(fig)

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    sns.histplot(
        finite_hs,
        bins=int(hist_bins.get("nearshore_hs", 60)),
        color=PLOT_COLORS["nearshore"],
        ax=ax,
    )
    ax.set_yscale("log")
    ax.set_title(f"{prefix}Nearshore Significant Wave Height (Hs), Log-scaled Count")
    ax.set_xlabel("Hs [m]")
    ax.set_ylabel("Count (log scale)")
    save_fig(fig, figure_dir, f"{tag}nearshore_hs_histogram_log_y")
    show_and_close(fig)

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    sns.histplot(
        finite_tp,
        bins=int(hist_bins.get("nearshore_tp", 60)),
        color=PLOT_COLORS["nearshore"],
        ax=ax,
    )
    ax.set_title(f"{prefix}Nearshore Peak Period (Tp)")
    ax.set_xlabel("Tp [s]")
    save_fig(fig, figure_dir, f"{tag}nearshore_tp_histogram")
    show_and_close(fig)

    fig, axes = plt.subplots(
        1, 2, figsize=(14, 6), subplot_kw={"projection": "polar"}, constrained_layout=True
    )
    plot_rose(
        axes[0], df["dir"], bins=int(rose_bins.get("nearshore", 18)), color=PLOT_COLORS["nearshore"]
    )
    axes[0].set_title(f"{prefix}Nearshore Mean Wave Direction")
    plot_rose(axes[1], df["dp"], bins=int(rose_bins.get("nearshore", 18)), color="#4c78a8")
    axes[1].set_title(f"{prefix}Nearshore Peak Wave Direction")
    save_fig(fig, figure_dir, f"{tag}nearshore_direction_rose_plots")
    show_and_close(fig)

    hs_sorted = np.sort(finite_hs)[::-1]
    exceedance = np.arange(1, hs_sorted.size + 1, dtype=float) / hs_sorted.size
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    ax.plot(hs_sorted, exceedance, color=PLOT_COLORS["nearshore"], linewidth=2)
    ax.set_yscale("log")
    ax.set_title(f"{prefix}Empirical Exceedance Curve for Nearshore Hs")
    ax.set_xlabel("Hs [m]")
    ax.set_ylabel("Exceedance probability")
    save_fig(fig, figure_dir, f"{tag}nearshore_hs_exceedance_curve")
    show_and_close(fig)

    split_palette = {split: PLOT_COLORS[split] for split in ("train", "val", "test")}
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)
    for split_name, split_df in df.groupby("split", observed=True):
        sns.histplot(
            split_df["hs"].dropna().to_numpy(dtype=float),
            bins=int(hist_bins.get("nearshore_hs", 60)),
            stat="density",
            element="step",
            fill=False,
            common_norm=False,
            ax=axes[0],
            label=str(split_name),
            color=split_palette.get(str(split_name), "#333333"),
        )
        sns.histplot(
            split_df["tp"].dropna().to_numpy(dtype=float),
            bins=int(hist_bins.get("nearshore_tp", 60)),
            stat="density",
            element="step",
            fill=False,
            common_norm=False,
            ax=axes[1],
            label=str(split_name),
            color=split_palette.get(str(split_name), "#333333"),
        )
    axes[0].set_title(f"{prefix}Nearshore Hs by Site-heldout Split")
    axes[0].set_xlabel("Hs [m]")
    axes[1].set_title(f"{prefix}Nearshore Tp by Site-heldout Split")
    axes[1].set_xlabel("Tp [s]")
    for ax in axes:
        ax.legend(frameon=False)
    save_fig(fig, figure_dir, f"{tag}nearshore_split_scalar_distributions")
    show_and_close(fig)

    for direction_col, figure_name, title_text in (
        ("dir", "nearshore_mean_direction_by_split", "Nearshore Mean Wave Direction by Split"),
        ("dp", "nearshore_peak_direction_by_split", "Nearshore Peak Wave Direction by Split"),
    ):
        splits_present = [
            str(item) for item in df["split"].cat.categories if item in {"train", "val", "test"}
        ]
        fig, axes = plt.subplots(
            1,
            len(splits_present),
            figsize=(6 * len(splits_present), 6),
            subplot_kw={"projection": "polar"},
            constrained_layout=False,
        )
        if len(splits_present) == 1:
            axes = [axes]
        for ax, split_name in zip(axes, splits_present, strict=False):
            split_df = df.loc[df["split"] == split_name, direction_col]
            plot_rose(
                ax,
                split_df,
                bins=int(rose_bins.get("split", 18)),
                color=split_palette.get(split_name, "#333333"),
            )
            ax.set_title(f"{split_name.title()} split")
        fig.suptitle(f"{prefix}{title_text}", y=0.98)
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))
        save_fig(fig, figure_dir, f"{tag}{figure_name}")
        show_and_close(fig)

    return {"nearshore_row_count": len(df)}


def run_nearshore_distribution_overlay_section(
    context: dict[str, Any],
    baseline_df: pd.DataFrame,
    compare_df: pd.DataFrame,
    *,
    baseline_label: str = "Pre-filter",
    compare_label: str = "Post-filter",
    figure_dir: str | Path | None = None,
    plot_settings: dict[str, Any] | None = None,
    figure_tag: str = "",
    title_prefix: str = "",
) -> dict[str, Any]:
    figure_dir = figure_dir or context["figure_dir"]
    plot_settings = plot_settings or default_plot_settings()
    hist_bins = plot_settings.get("hist_bins", {})
    rose_bins = plot_settings.get("rose_bins", {})
    tag = f"{figure_tag}_" if figure_tag else ""
    prefix = title_prefix

    base = baseline_df.copy()
    comp = compare_df.copy()

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    sns.histplot(
        base["hs"].dropna().to_numpy(dtype=float),
        bins=int(hist_bins.get("nearshore_hs", 60)),
        stat="density",
        element="step",
        fill=False,
        ax=ax,
        label=baseline_label,
        color=PLOT_COLORS["nearshore"],
    )
    sns.histplot(
        comp["hs"].dropna().to_numpy(dtype=float),
        bins=int(hist_bins.get("nearshore_hs", 60)),
        stat="density",
        element="step",
        fill=False,
        ax=ax,
        label=compare_label,
        color="#c44e52",
    )
    ax.set_title(f"{prefix}Nearshore Hs Overlay")
    ax.set_xlabel("Hs [m]")
    ax.legend(frameon=False)
    save_fig(fig, figure_dir, f"{tag}nearshore_hs_overlay")
    show_and_close(fig)

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    sns.histplot(
        base["hs"].dropna().to_numpy(dtype=float),
        bins=int(hist_bins.get("nearshore_hs", 60)),
        stat="density",
        element="step",
        fill=False,
        ax=ax,
        label=baseline_label,
        color=PLOT_COLORS["nearshore"],
    )
    sns.histplot(
        comp["hs"].dropna().to_numpy(dtype=float),
        bins=int(hist_bins.get("nearshore_hs", 60)),
        stat="density",
        element="step",
        fill=False,
        ax=ax,
        label=compare_label,
        color="#c44e52",
    )
    ax.set_yscale("log")
    ax.set_title(f"{prefix}Nearshore Hs Overlay, Log-scaled Density")
    ax.set_xlabel("Hs [m]")
    ax.legend(frameon=False)
    save_fig(fig, figure_dir, f"{tag}nearshore_hs_overlay_log")
    show_and_close(fig)

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    sns.histplot(
        base["tp"].dropna().to_numpy(dtype=float),
        bins=int(hist_bins.get("nearshore_tp", 60)),
        stat="density",
        element="step",
        fill=False,
        ax=ax,
        label=baseline_label,
        color=PLOT_COLORS["nearshore"],
    )
    sns.histplot(
        comp["tp"].dropna().to_numpy(dtype=float),
        bins=int(hist_bins.get("nearshore_tp", 60)),
        stat="density",
        element="step",
        fill=False,
        ax=ax,
        label=compare_label,
        color="#c44e52",
    )
    ax.set_title(f"{prefix}Nearshore Tp Overlay")
    ax.set_xlabel("Tp [s]")
    ax.legend(frameon=False)
    save_fig(fig, figure_dir, f"{tag}nearshore_tp_overlay")
    show_and_close(fig)

    fig, axes = plt.subplots(
        1, 2, figsize=(14, 6), subplot_kw={"projection": "polar"}, constrained_layout=True
    )
    overlay_rose(
        axes[0],
        [
            (base["dir"], baseline_label, PLOT_COLORS["nearshore"]),
            (comp["dir"], compare_label, "#c44e52"),
        ],
        bins=int(rose_bins.get("comparison", 18)),
    )
    axes[0].set_title(f"{prefix}Nearshore Mean Direction Overlay")
    overlay_rose(
        axes[1],
        [
            (base["dp"], baseline_label, PLOT_COLORS["nearshore"]),
            (comp["dp"], compare_label, "#c44e52"),
        ],
        bins=int(rose_bins.get("comparison", 18)),
    )
    axes[1].set_title(f"{prefix}Nearshore Peak Direction Overlay")
    save_fig(fig, figure_dir, f"{tag}nearshore_direction_overlay")
    show_and_close(fig)

    base_hs = np.sort(base["hs"].dropna().to_numpy(dtype=float))[::-1]
    comp_hs = np.sort(comp["hs"].dropna().to_numpy(dtype=float))[::-1]
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    ax.plot(
        base_hs,
        np.arange(1, base_hs.size + 1) / max(base_hs.size, 1),
        color=PLOT_COLORS["nearshore"],
        linewidth=2,
        label=baseline_label,
    )
    ax.plot(
        comp_hs,
        np.arange(1, comp_hs.size + 1) / max(comp_hs.size, 1),
        color="#c44e52",
        linewidth=2,
        label=compare_label,
    )
    ax.set_yscale("log")
    ax.set_title(f"{prefix}Nearshore Hs Exceedance Overlay")
    ax.set_xlabel("Hs [m]")
    ax.set_ylabel("Exceedance probability")
    ax.legend(frameon=False)
    save_fig(fig, figure_dir, f"{tag}nearshore_hs_exceedance_overlay")
    show_and_close(fig)

    return {"baseline_rows": len(base), "compare_rows": len(comp)}


def _distribution_subplots(n_items: int) -> tuple[plt.Figure, np.ndarray]:
    n_cols = 3
    n_rows = int(math.ceil(n_items / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(6 * n_cols, 4.5 * n_rows), constrained_layout=True
    )
    return fig, np.atleast_1d(axes).ravel()


def plot_source_summary(
    ax: plt.Axes,
    df: pd.DataFrame,
    value_col: str,
    color: str,
    title: str,
) -> pd.DataFrame:
    summary = (
        df.groupby("source", observed=True)[value_col]
        .quantile([0.1, 0.25, 0.5, 0.75, 0.9])
        .unstack()
        .rename(columns={0.1: "p10", 0.25: "p25", 0.5: "median", 0.75: "p75", 0.9: "p90"})
        .sort_values("median")
    )
    y_pos = np.arange(summary.shape[0])
    ax.hlines(y_pos, summary["p10"], summary["p90"], color=color, alpha=0.35, linewidth=2.0)
    ax.hlines(y_pos, summary["p25"], summary["p75"], color=color, alpha=0.8, linewidth=5.0)
    ax.scatter(summary["median"], y_pos, color="black", s=30, zorder=3)
    ax.set_yticks(y_pos, summary.index.astype(str))
    ax.set_title(title)
    ax.invert_yaxis()
    return summary.reset_index()


def run_offshore_distribution_section(
    context: dict[str, Any],
    figure_dir: str | Path | None = None,
    *,
    wave_df: pd.DataFrame | None = None,
    wind_df: pd.DataFrame | None = None,
    plot_settings: dict[str, Any] | None = None,
    figure_tag: str = "",
    title_prefix: str = "",
) -> dict[str, Any]:
    figure_dir = figure_dir or context["figure_dir"]
    plot_settings = plot_settings or default_plot_settings()
    hist_bins = int((plot_settings.get("hist_bins", {}) or {}).get("offshore_scalar", 60))
    rose_bins = int((plot_settings.get("rose_bins", {}) or {}).get("offshore", 18))
    wave_df = wave_df.copy() if wave_df is not None else context["offshore_wave_raw"].copy()
    wind_df = wind_df.copy() if wind_df is not None else context["offshore_wind_raw"].copy()
    tag = f"{figure_tag}_" if figure_tag else ""
    prefix = title_prefix

    def _finite_values(df: pd.DataFrame, column: str) -> np.ndarray:
        if column not in df.columns:
            return np.array([], dtype=float)
        values = pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)
        return values[np.isfinite(values)]

    def _plot_hist_pair(
        ax_linear: plt.Axes,
        ax_log: plt.Axes,
        values: np.ndarray,
        xlabel: str,
        title_stem: str,
        color: str,
    ) -> None:
        if values.size == 0:
            for ax, suffix in ((ax_linear, "Count"), (ax_log, "Log Count")):
                ax.text(
                    0.5, 0.5, "No finite data", ha="center", va="center", transform=ax.transAxes
                )
                ax.set_title(f"{prefix}{title_stem} ({suffix})")
                ax.set_xlabel(xlabel)
            return
        sns.histplot(values, bins=hist_bins, color=color, ax=ax_linear)
        ax_linear.set_title(f"{prefix}{title_stem} (Count)")
        ax_linear.set_xlabel(xlabel)
        sns.histplot(values, bins=hist_bins, color=color, ax=ax_log)
        ax_log.set_yscale("log")
        ax_log.set_title(f"{prefix}{title_stem} (Log Count)")
        ax_log.set_xlabel(xlabel)

    wave_height_specs = [
        ("hs", "Offshore Hs", "Hs [m]"),
        ("hs_sea", "Offshore Sea Hs", "Sea Hs [m]"),
        ("hs_swell", "Offshore Swell Hs", "Swell Hs [m]"),
    ]
    fig, axes = plt.subplots(
        len(wave_height_specs),
        2,
        figsize=(16, 4.8 * len(wave_height_specs)),
        constrained_layout=True,
    )
    axes = np.atleast_2d(axes)
    for row_idx, (column, title_stem, xlabel) in enumerate(wave_height_specs):
        values = _finite_values(wave_df, column)
        _plot_hist_pair(
            axes[row_idx, 0], axes[row_idx, 1], values, xlabel, title_stem, PLOT_COLORS["offshore"]
        )
    save_fig(fig, figure_dir, f"{tag}offshore_wave_height_distributions")
    show_and_close(fig)

    tp_specs = [
        ("tp", "Offshore Tp", "Tp [s]"),
        ("tp_sea", "Offshore Sea Tp", "Sea Tp [s]"),
        ("tp_swell", "Offshore Swell Tp", "Swell Tp [s]"),
    ]
    fig, axes = plt.subplots(
        len(tp_specs), 1, figsize=(10, 4.8 * len(tp_specs)), constrained_layout=True
    )
    axes = np.atleast_1d(axes)
    for ax, (column, title_stem, xlabel) in zip(axes, tp_specs, strict=False):
        values = _finite_values(wave_df, column)
        if values.size == 0:
            ax.text(0.5, 0.5, "No finite data", ha="center", va="center", transform=ax.transAxes)
        else:
            sns.histplot(values, bins=hist_bins, color=PLOT_COLORS["offshore"], ax=ax)
        ax.set_title(f"{prefix}{title_stem}")
        ax.set_xlabel(xlabel)
    save_fig(fig, figure_dir, f"{tag}offshore_tp_distributions")
    show_and_close(fig)

    tm_specs = [
        ("tm1", "Offshore Tm1", "Tm1 [s]"),
        ("tm2", "Offshore Tm2", "Tm2 [s]"),
    ]
    fig, axes = plt.subplots(
        len(tm_specs), 1, figsize=(10, 4.8 * len(tm_specs)), constrained_layout=True
    )
    axes = np.atleast_1d(axes)
    for ax, (column, title_stem, xlabel) in zip(axes, tm_specs, strict=False):
        values = _finite_values(wave_df, column)
        if values.size == 0:
            ax.text(0.5, 0.5, "No finite data", ha="center", va="center", transform=ax.transAxes)
        else:
            sns.histplot(values, bins=hist_bins, color=PLOT_COLORS["offshore"], ax=ax)
        ax.set_title(f"{prefix}{title_stem}")
        ax.set_xlabel(xlabel)
    save_fig(fig, figure_dir, f"{tag}offshore_tm1_tm2_distributions")
    show_and_close(fig)

    fig, axes = plt.subplots(
        1, 2, figsize=(12, 6), subplot_kw={"projection": "polar"}, constrained_layout=True
    )
    plot_rose(axes[0], wave_df["thq"], bins=rose_bins, color=PLOT_COLORS["offshore"])
    axes[0].set_title(f"{prefix}Offshore Mean Wave Direction (thq)")
    plot_rose(axes[1], wave_df["Pdir"], bins=rose_bins, color="#e76f51")
    axes[1].set_title(f"{prefix}Offshore Peak Wave Direction (Pdir)")
    save_fig(fig, figure_dir, f"{tag}offshore_wave_direction_distributions")
    show_and_close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    hs_summary = plot_source_summary(
        axes[0],
        wave_df.dropna(subset=["hs"]),
        "hs",
        PLOT_COLORS["offshore"],
        "Source-by-source Offshore Hs",
    )
    tp_summary = plot_source_summary(
        axes[1],
        wave_df.dropna(subset=["tp"]),
        "tp",
        PLOT_COLORS["offshore"],
        "Source-by-source Offshore Tp",
    )
    axes[0].set_xlabel("Hs [m]")
    axes[1].set_xlabel("Tp [s]")
    save_fig(fig, figure_dir, f"{tag}offshore_wave_source_by_source_comparison")
    show_and_close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True, subplot_kw={})
    wind_values = _finite_values(wind_df, "wind_speed_10m")
    if wind_values.size == 0:
        axes[0].text(
            0.5, 0.5, "No finite data", ha="center", va="center", transform=axes[0].transAxes
        )
    else:
        sns.histplot(wind_values, bins=hist_bins, color=PLOT_COLORS["wind"], ax=axes[0])
    axes[0].set_title(f"{prefix}Offshore Wind Speed")
    axes[0].set_xlabel("Wind speed [m/s]")
    axes[1].remove()
    polar_ax = fig.add_subplot(1, 2, 2, projection="polar")
    plot_rose(polar_ax, wind_df["wind_direction_10m"], bins=rose_bins, color=PLOT_COLORS["wind"])
    polar_ax.set_title(f"{prefix}Offshore Wind Direction")
    save_fig(fig, figure_dir, f"{tag}offshore_wind_distributions")
    show_and_close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(7, 6), constrained_layout=True)
    wind_summary = plot_source_summary(
        ax,
        wind_df.dropna(subset=["wind_speed_10m"]),
        "wind_speed_10m",
        PLOT_COLORS["wind"],
        "Source-by-source Offshore Wind Speed",
    )
    ax.set_xlabel("Wind speed [m/s]")
    save_fig(fig, figure_dir, f"{tag}offshore_wind_source_by_source_comparison")
    show_and_close(fig)

    display(hs_summary.head(10))
    display(tp_summary.head(10))
    display(wind_summary.head(10))
    return {"hs_summary": hs_summary, "tp_summary": tp_summary, "wind_summary": wind_summary}


def run_satellite_site_context_section(
    context: dict[str, Any],
    nearshore_sites: list[str],
    offshore_sites: list[str],
    figure_dir: str | Path | None = None,
    *,
    figure_tag: str = "",
    title_prefix: str = "",
    basemap_zoom: int = 10,
    padding_fraction: float = 0.18,
    padding_x: float | None = None,
    padding_y: float | None = None,
    nearshore_marker_size: float = 90,
    offshore_marker_size: float = 110,
    show_all_offshore_in_view: bool = False,
    background_offshore_marker_size: float = 36,
    connect_pairs: bool = True,
    pairings: list[tuple[str, str]] | None = None,
) -> pd.DataFrame:
    figure_dir = figure_dir or context["figure_dir"]
    tag = f"{figure_tag}_" if figure_tag else ""
    prefix = title_prefix

    nearshore_lookup = {
        str(item.get("name")): item for item in context["sites_cfg"].get("nearshore_sites", [])
    }
    offshore_lookup = {
        str(item.get("name")): item for item in context["sites_cfg"].get("offshore_sites", [])
    }

    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for site_name in nearshore_sites:
        info = nearshore_lookup.get(str(site_name))
        if info is None:
            missing.append(str(site_name))
            continue
        rows.append(
            {
                "site": str(site_name),
                "site_type": "Nearshore (NORAC)",
                "lat": float(info["lat"]),
                "lon": float(info["lon"]),
                "depth_m": float(info.get("depth_m", np.nan)),
            }
        )
    for site_name in offshore_sites:
        info = offshore_lookup.get(str(site_name))
        if info is None:
            missing.append(str(site_name))
            continue
        rows.append(
            {
                "site": str(site_name),
                "site_type": "Offshore (NORA3)",
                "lat": float(info["lat"]),
                "lon": float(info["lon"]),
                "depth_m": float(info.get("depth_m", np.nan)),
            }
        )
    if missing:
        raise KeyError(f"Could not find site definitions for: {missing}")

    site_df = pd.DataFrame(rows)
    try:
        site_df = nh.add_web_mercator_columns(
            site_df, lon_col="lon", lat_col="lat", x_col="x_3857", y_col="y_3857"
        )
        use_projected = True
    except Exception as exc:
        print(f"Satellite projection fallback: {exc}")
        use_projected = False
    has_satellite_basemap = use_projected and getattr(nh, "ctx", None) is not None

    all_offshore_df = pd.DataFrame(
        [
            {
                "site": str(item["name"]),
                "site_type": "Offshore (NORA3)",
                "lat": float(item["lat"]),
                "lon": float(item["lon"]),
                "depth_m": float(item.get("depth_m", np.nan)),
            }
            for item in context["sites_cfg"].get("offshore_sites", [])
        ]
    )
    if use_projected:
        all_offshore_df = nh.add_web_mercator_columns(
            all_offshore_df,
            lon_col="lon",
            lat_col="lat",
            x_col="x_3857",
            y_col="y_3857",
        )

    fig, ax = plt.subplots(figsize=(11.5, 9.0), dpi=180, constrained_layout=True)
    if use_projected:
        x_col = "x_3857"
        y_col = "y_3857"
    else:
        x_col = "lon"
        y_col = "lat"
        ax.set_facecolor("#f2f2f2")

    x_span = float(site_df[x_col].max() - site_df[x_col].min()) if len(site_df) > 1 else 1.0
    y_span = float(site_df[y_col].max() - site_df[y_col].min()) if len(site_df) > 1 else 1.0
    default_x_pad = max(x_span * float(padding_fraction), 12_000.0 if use_projected else 0.05)
    default_y_pad = max(y_span * float(padding_fraction), 12_000.0 if use_projected else 0.05)
    x_pad = float(padding_x) if padding_x is not None else default_x_pad
    y_pad = float(padding_y) if padding_y is not None else default_y_pad

    x_min = float(site_df[x_col].min() - x_pad)
    x_max = float(site_df[x_col].max() + x_pad)
    y_min = float(site_df[y_col].min() - y_pad)
    y_max = float(site_df[y_col].max() + y_pad)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal", adjustable="box")
    if use_projected:
        nh.add_world_imagery_basemap(ax, zoom=int(basemap_zoom))

    if show_all_offshore_in_view and not all_offshore_df.empty:
        background_offshore = all_offshore_df.loc[
            (all_offshore_df[x_col] >= x_min)
            & (all_offshore_df[x_col] <= x_max)
            & (all_offshore_df[y_col] >= y_min)
            & (all_offshore_df[y_col] <= y_max)
            & (~all_offshore_df["site"].isin(site_df["site"]))
        ]
        if not background_offshore.empty:
            ax.scatter(
                background_offshore[x_col],
                background_offshore[y_col],
                s=float(background_offshore_marker_size),
                c=PLOT_COLORS["offshore"],
                marker="^",
                edgecolors="white",
                linewidths=0.8,
                alpha=0.5,
                label="Other NORA3 points",
                zorder=2,
            )

    styles = {
        "Nearshore (NORAC)": {
            "color": PLOT_COLORS["nearshore"],
            "marker": "o",
            "size": float(nearshore_marker_size),
        },
        "Offshore (NORA3)": {
            "color": PLOT_COLORS["offshore"],
            "marker": "^",
            "size": float(offshore_marker_size),
        },
    }
    for site_type, style in styles.items():
        subset = site_df.loc[site_df["site_type"] == site_type]
        if subset.empty:
            continue
        ax.scatter(
            subset[x_col],
            subset[y_col],
            s=style["size"],
            c=style["color"],
            marker=style["marker"],
            edgecolors="white",
            linewidths=1.4,
            alpha=0.95,
            label=site_type,
            zorder=4,
        )

    if connect_pairs:
        resolved_pairings = pairings or list(zip(nearshore_sites, offshore_sites, strict=False))
        for nearshore_name, offshore_name in resolved_pairings:
            pair = site_df.loc[site_df["site"].isin([nearshore_name, offshore_name])]
            if len(pair) != 2:
                continue
            ax.plot(
                pair[x_col].to_numpy(dtype=float),
                pair[y_col].to_numpy(dtype=float),
                linestyle="--",
                linewidth=1.6,
                color="white" if has_satellite_basemap else "0.35",
                alpha=0.85,
                zorder=3,
            )

    for row in site_df.itertuples(index=False):
        ax.annotate(
            str(row.site),
            (getattr(row, x_col), getattr(row, y_col)),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=8,
            color="white" if use_projected else "black",
            ha="left",
            va="bottom",
            bbox=dict(
                boxstyle="round,pad=0.18",
                facecolor="black" if use_projected else "white",
                edgecolor="none",
                alpha=0.6 if use_projected else 0.85,
            ),
            zorder=6,
        )

    ax.legend(
        frameon=True,
        facecolor="white",
        edgecolor="0.8",
        loc="upper right",
        fontsize=8,
        markerscale=0.75,
        borderpad=0.35,
        handletextpad=0.4,
        labelspacing=0.3,
    )
    ax.set_title(
        f"{prefix}Nearshore and Offshore Comparison Sites on Satellite Imagery",
        pad=12,
        fontsize=11,
    )
    if use_projected:
        ax.set_xlabel("Web Mercator x (m)", fontsize=9)
        ax.set_ylabel("Web Mercator y (m)", fontsize=9)
        ax.grid(False)
    else:
        ax.set_xlabel("Longitude", fontsize=9)
        ax.set_ylabel("Latitude", fontsize=9)
    ax.tick_params(axis="both", labelsize=8)

    save_fig(fig, figure_dir, f"{tag}satellite_site_context")
    show_and_close(fig)
    return site_df


def _build_aligned_numeric_df(
    context: dict[str, Any],
    key: str,
    columns: list[str],
    y_prefix: str,
) -> pd.DataFrame:
    cache = context["cache"]
    if key in cache:
        return cache[key]

    processed = context["processed"]
    timestamps = processed["timestamps"]
    rows: list[pd.DataFrame] = []
    y_npz = processed["y_npz"]
    split_by_site = context["split_by_site"]
    for site_name in processed["target_sites"]:
        arr = np.asarray(y_npz[f"{y_prefix}{site_name}"], dtype=np.float32)
        part = pd.DataFrame(arr, columns=columns)
        part.insert(0, "timestamp", timestamps)
        part.insert(0, "site", site_name)
        part.insert(1, "split", split_by_site.get(site_name, "train"))
        rows.append(part)
    df = pd.concat(rows, ignore_index=True)
    df = _cast_category_columns(df, ["site", "split"])
    cache[key] = df
    return df


def get_aligned_scalar_df(context: dict[str, Any]) -> pd.DataFrame:
    scalar_columns = ["nearshore_hs", "nearshore_tp", "offshore_hs", "offshore_tp"]
    cache = context["cache"]
    if "aligned_scalar_df" in cache:
        return cache["aligned_scalar_df"]

    processed = context["processed"]
    timestamps = processed["timestamps"]
    split_by_site = context["split_by_site"]
    rows: list[pd.DataFrame] = []
    for site_name in processed["target_sites"]:
        nearshore_frame = context["nearshore_frames"][site_name].reindex(timestamps)
        wave_source = context["site_to_wave_site"][site_name]
        offshore_frame = context["offshore_wave_frames"][wave_source].reindex(timestamps)
        part = pd.DataFrame(
            {
                "site": site_name,
                "split": split_by_site.get(site_name, "train"),
                "timestamp": timestamps,
                "nearshore_hs": pd.to_numeric(nearshore_frame["hs"], errors="coerce").to_numpy(
                    dtype=np.float32
                ),
                "nearshore_tp": pd.to_numeric(nearshore_frame["tp"], errors="coerce").to_numpy(
                    dtype=np.float32
                ),
                "offshore_hs": pd.to_numeric(offshore_frame["hs"], errors="coerce").to_numpy(
                    dtype=np.float32
                ),
                "offshore_tp": pd.to_numeric(offshore_frame["tp"], errors="coerce").to_numpy(
                    dtype=np.float32
                ),
            }
        )
        rows.append(part)
    df = pd.concat(rows, ignore_index=True)
    df = _cast_category_columns(df, ["site", "split"])
    cache["aligned_scalar_df"] = df
    return df


def get_aligned_direction_df(context: dict[str, Any]) -> pd.DataFrame:
    cache = context["cache"]
    if "aligned_direction_df" in cache:
        return cache["aligned_direction_df"]

    processed = context["processed"]
    timestamps = processed["timestamps"]
    split_by_site = context["split_by_site"]
    rows: list[pd.DataFrame] = []
    for site_name in processed["target_sites"]:
        nearshore_frame = context["nearshore_frames"][site_name].reindex(timestamps)
        wave_source = context["site_to_wave_site"][site_name]
        offshore_frame = context["offshore_wave_frames"][wave_source].reindex(timestamps)
        part = pd.DataFrame(
            {
                "site": site_name,
                "split": split_by_site.get(site_name, "train"),
                "timestamp": timestamps,
                "nearshore_dir": pd.to_numeric(nearshore_frame["dir"], errors="coerce").to_numpy(
                    dtype=np.float32
                ),
                "nearshore_dp": pd.to_numeric(nearshore_frame["dp"], errors="coerce").to_numpy(
                    dtype=np.float32
                ),
                "offshore_dir": pd.to_numeric(offshore_frame["thq"], errors="coerce").to_numpy(
                    dtype=np.float32
                ),
                "offshore_dp": pd.to_numeric(offshore_frame["Pdir"], errors="coerce").to_numpy(
                    dtype=np.float32
                ),
            }
        )
        rows.append(part)
    df = pd.concat(rows, ignore_index=True)
    df = _cast_category_columns(df, ["site", "split"])
    cache["aligned_direction_df"] = df
    return df


def compute_summary_stats(series: pd.Series, label: str) -> dict[str, Any]:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    return {
        "series": label,
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def _sample_rows(df: pd.DataFrame, max_points: int, seed: int = 42) -> pd.DataFrame:
    if len(df) <= max_points:
        return df
    return df.sample(n=max_points, random_state=seed)


def run_offshore_nearshore_comparison_section(
    context: dict[str, Any],
    figure_dir: str | Path | None = None,
    *,
    scalar_df: pd.DataFrame | None = None,
    plot_settings: dict[str, Any] | None = None,
    figure_tag: str = "",
    title_prefix: str = "",
) -> dict[str, Any]:
    figure_dir = figure_dir or context["figure_dir"]
    plot_settings = plot_settings or default_plot_settings()
    hist_bins = plot_settings.get("hist_bins", {})
    scatter_sample_size = int(plot_settings.get("comparison_scatter_sample_size", 250_000))
    hexbin_sample_size = int(plot_settings.get("comparison_hexbin_sample_size", 1_000_000))
    scalar_df = scalar_df.copy() if scalar_df is not None else get_aligned_scalar_df(context).copy()
    scalar_df = scalar_df.dropna(
        subset=["nearshore_hs", "nearshore_tp", "offshore_hs", "offshore_tp"]
    ).copy()
    tag = f"{figure_tag}_" if figure_tag else ""
    prefix = title_prefix

    stats = pd.DataFrame(
        [
            compute_summary_stats(scalar_df["offshore_hs"], "Offshore nearest-source Hs"),
            compute_summary_stats(scalar_df["nearshore_hs"], "Nearshore Hs"),
            compute_summary_stats(scalar_df["offshore_tp"], "Offshore nearest-source Tp"),
            compute_summary_stats(scalar_df["nearshore_tp"], "Nearshore Tp"),
        ]
    )

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)
    sns.histplot(
        scalar_df["offshore_hs"].to_numpy(dtype=float),
        bins=int(hist_bins.get("comparison_hs", 70)),
        stat="density",
        element="step",
        fill=False,
        ax=axes[0],
        label=f"{prefix}Offshore nearest-source Hs" if prefix else "Offshore nearest-source Hs",
        color=PLOT_COLORS["offshore"],
    )
    sns.histplot(
        scalar_df["nearshore_hs"].to_numpy(dtype=float),
        bins=int(hist_bins.get("comparison_hs", 70)),
        stat="density",
        element="step",
        fill=False,
        ax=axes[0],
        label=f"{prefix}Nearshore Hs" if prefix else "Nearshore Hs",
        color=PLOT_COLORS["nearshore"],
    )
    axes[0].set_title(f"{prefix}Offshore Nearest-source vs Nearshore Hs")
    axes[0].set_xlabel("Hs [m]")
    axes[0].legend(frameon=False)
    sns.histplot(
        scalar_df["offshore_tp"].to_numpy(dtype=float),
        bins=int(hist_bins.get("comparison_tp", 70)),
        stat="density",
        element="step",
        fill=False,
        ax=axes[1],
        label=f"{prefix}Offshore nearest-source Tp" if prefix else "Offshore nearest-source Tp",
        color=PLOT_COLORS["offshore"],
    )
    sns.histplot(
        scalar_df["nearshore_tp"].to_numpy(dtype=float),
        bins=int(hist_bins.get("comparison_tp", 70)),
        stat="density",
        element="step",
        fill=False,
        ax=axes[1],
        label=f"{prefix}Nearshore Tp" if prefix else "Nearshore Tp",
        color=PLOT_COLORS["nearshore"],
    )
    axes[1].set_title(f"{prefix}Offshore Nearest-source vs Nearshore Tp")
    axes[1].set_xlabel("Tp [s]")
    axes[1].legend(frameon=False)
    save_fig(fig, figure_dir, f"{tag}offshore_vs_nearshore_distribution_overlays")
    show_and_close(fig)

    scatter_sample = _sample_rows(scalar_df, max_points=scatter_sample_size, seed=42)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)
    axes[0].scatter(
        scatter_sample["offshore_hs"],
        scatter_sample["nearshore_hs"],
        s=4,
        alpha=0.08,
        color=PLOT_COLORS["nearshore"],
    )
    hs_max = float(
        np.nanmax([scatter_sample["offshore_hs"].max(), scatter_sample["nearshore_hs"].max()])
    )
    axes[0].plot([0, hs_max], [0, hs_max], color="black", linestyle="--", linewidth=1.2)
    axes[0].set_title(f"{prefix}Offshore vs Nearshore Hs (sampled n={len(scatter_sample):,})")
    axes[0].set_xlabel("Offshore nearest-source Hs [m]")
    axes[0].set_ylabel("Nearshore Hs [m]")
    axes[1].scatter(
        scatter_sample["offshore_tp"],
        scatter_sample["nearshore_tp"],
        s=4,
        alpha=0.08,
        color=PLOT_COLORS["offshore"],
    )
    tp_max = float(
        np.nanmax([scatter_sample["offshore_tp"].max(), scatter_sample["nearshore_tp"].max()])
    )
    axes[1].plot([0, tp_max], [0, tp_max], color="black", linestyle="--", linewidth=1.2)
    axes[1].set_title(f"{prefix}Offshore vs Nearshore Tp (sampled n={len(scatter_sample):,})")
    axes[1].set_xlabel("Offshore nearest-source Tp [s]")
    axes[1].set_ylabel("Nearshore Tp [s]")
    save_fig(fig, figure_dir, f"{tag}offshore_vs_nearshore_scatter")
    show_and_close(fig)

    hexbin_sample = _sample_rows(scalar_df, max_points=hexbin_sample_size, seed=7)
    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
    hb = ax.hexbin(
        hexbin_sample["offshore_hs"],
        hexbin_sample["nearshore_hs"],
        gridsize=70,
        mincnt=1,
        cmap="viridis",
    )
    hs_hex_max = float(
        np.nanmax([hexbin_sample["offshore_hs"].max(), hexbin_sample["nearshore_hs"].max()])
    )
    ax.plot([0, hs_hex_max], [0, hs_hex_max], color="white", linestyle="--", linewidth=1.2)
    ax.set_title(f"{prefix}Offshore vs Nearshore Hs Hexbin (sampled n={len(hexbin_sample):,})")
    ax.set_xlabel("Offshore nearest-source Hs [m]")
    ax.set_ylabel("Nearshore Hs [m]")
    fig.colorbar(hb, ax=ax, label="Count")
    save_fig(fig, figure_dir, f"{tag}offshore_vs_nearshore_hs_hexbin")
    show_and_close(fig)

    display(stats)
    return {
        "summary_stats": stats,
        "scatter_sample_size": len(scatter_sample),
        "hexbin_sample_size": len(hexbin_sample),
    }


def run_offshore_nearshore_comparison_overlay_section(
    context: dict[str, Any],
    baseline_scalar_df: pd.DataFrame,
    compare_scalar_df: pd.DataFrame,
    *,
    baseline_label: str = "Pre-filter",
    compare_label: str = "Post-filter",
    figure_dir: str | Path | None = None,
    plot_settings: dict[str, Any] | None = None,
    figure_tag: str = "",
    title_prefix: str = "",
) -> dict[str, Any]:
    figure_dir = figure_dir or context["figure_dir"]
    plot_settings = plot_settings or default_plot_settings()
    hist_bins = plot_settings.get("hist_bins", {})
    tag = f"{figure_tag}_" if figure_tag else ""
    prefix = title_prefix
    base = baseline_scalar_df.dropna(
        subset=["nearshore_hs", "nearshore_tp", "offshore_hs", "offshore_tp"]
    ).copy()
    comp = compare_scalar_df.dropna(
        subset=["nearshore_hs", "nearshore_tp", "offshore_hs", "offshore_tp"]
    ).copy()

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)
    for df, label, color in (
        (base, baseline_label, PLOT_COLORS["nearshore"]),
        (comp, compare_label, "#c44e52"),
    ):
        sns.histplot(
            df["nearshore_hs"].to_numpy(dtype=float),
            bins=int(hist_bins.get("comparison_hs", 70)),
            stat="density",
            element="step",
            fill=False,
            ax=axes[0],
            label=f"{label} nearshore",
            color=color,
        )
    axes[0].set_title(f"{prefix}Nearshore Hs Overlay")
    axes[0].set_xlabel("Hs [m]")
    axes[0].legend(frameon=False)
    for df, label, color in (
        (base, baseline_label, PLOT_COLORS["offshore"]),
        (comp, compare_label, "#8c564b"),
    ):
        sns.histplot(
            df["offshore_hs"].to_numpy(dtype=float),
            bins=int(hist_bins.get("comparison_hs", 70)),
            stat="density",
            element="step",
            fill=False,
            ax=axes[1],
            label=f"{label} offshore",
            color=color,
        )
    axes[1].set_title(f"{prefix}Offshore Nearest-source Hs Overlay")
    axes[1].set_xlabel("Hs [m]")
    axes[1].legend(frameon=False)
    save_fig(fig, figure_dir, f"{tag}comparison_hs_overlay")
    show_and_close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)
    for df, label, color in (
        (base, baseline_label, PLOT_COLORS["nearshore"]),
        (comp, compare_label, "#c44e52"),
    ):
        sns.histplot(
            df["nearshore_tp"].to_numpy(dtype=float),
            bins=int(hist_bins.get("comparison_tp", 70)),
            stat="density",
            element="step",
            fill=False,
            ax=axes[0],
            label=f"{label} nearshore",
            color=color,
        )
    axes[0].set_title(f"{prefix}Nearshore Tp Overlay")
    axes[0].set_xlabel("Tp [s]")
    axes[0].legend(frameon=False)
    for df, label, color in (
        (base, baseline_label, PLOT_COLORS["offshore"]),
        (comp, compare_label, "#8c564b"),
    ):
        sns.histplot(
            df["offshore_tp"].to_numpy(dtype=float),
            bins=int(hist_bins.get("comparison_tp", 70)),
            stat="density",
            element="step",
            fill=False,
            ax=axes[1],
            label=f"{label} offshore",
            color=color,
        )
    axes[1].set_title(f"{prefix}Offshore Nearest-source Tp Overlay")
    axes[1].set_xlabel("Tp [s]")
    axes[1].legend(frameon=False)
    save_fig(fig, figure_dir, f"{tag}comparison_tp_overlay")
    show_and_close(fig)

    summary = pd.DataFrame(
        [
            compute_summary_stats(base["nearshore_hs"], f"{baseline_label} nearshore Hs"),
            compute_summary_stats(comp["nearshore_hs"], f"{compare_label} nearshore Hs"),
            compute_summary_stats(base["offshore_hs"], f"{baseline_label} offshore Hs"),
            compute_summary_stats(comp["offshore_hs"], f"{compare_label} offshore Hs"),
            compute_summary_stats(base["nearshore_tp"], f"{baseline_label} nearshore Tp"),
            compute_summary_stats(comp["nearshore_tp"], f"{compare_label} nearshore Tp"),
            compute_summary_stats(base["offshore_tp"], f"{baseline_label} offshore Tp"),
            compute_summary_stats(comp["offshore_tp"], f"{compare_label} offshore Tp"),
        ]
    )
    display(summary)
    return {"summary_stats": summary, "baseline_rows": len(base), "compare_rows": len(comp)}


def compute_site_hs_p95(context: dict[str, Any]) -> pd.DataFrame:
    processed = context["processed"]
    y_npz = processed["y_npz"]
    split_by_site = context["split_by_site"]
    rows: list[dict[str, Any]] = []
    for site_name in processed["target_sites"]:
        hs = np.asarray(y_npz[f"Yphysical__{site_name}"][:, 0], dtype=float)
        hs = hs[np.isfinite(hs)]
        rows.append(
            {
                "site": site_name,
                "split": split_by_site.get(site_name, "train"),
                "hs_p95": float(np.quantile(hs, 0.95)),
                "hs_max": float(np.max(hs)),
            }
        )
    return pd.DataFrame(rows).sort_values("hs_p95").reset_index(drop=True)


def choose_representative_sites(context: dict[str, Any]) -> pd.DataFrame:
    site_stats = compute_site_hs_p95(context)
    low_row = site_stats.iloc[0]
    high_row = site_stats.iloc[-1]
    median_target = float(site_stats["hs_p95"].median())
    median_row = site_stats.iloc[(site_stats["hs_p95"] - median_target).abs().idxmin()]
    selected = [low_row, median_row, high_row]
    selected_sites = {str(item["site"]) for item in selected}
    present_splits = {str(item["split"]) for item in selected}
    for split_name in ("val", "test"):
        if split_name not in present_splits:
            split_rows = site_stats.loc[site_stats["split"] == split_name]
            if not split_rows.empty:
                chosen = split_rows.sort_values("hs_p95", ascending=False).iloc[0]
                if str(chosen["site"]) not in selected_sites:
                    selected.append(chosen)
                    selected_sites.add(str(chosen["site"]))
    return pd.DataFrame(selected).drop_duplicates(subset=["site"]).reset_index(drop=True)


def resolve_selected_sites(
    context: dict[str, Any],
    site_selection_settings: dict[str, Any] | None = None,
) -> pd.DataFrame:
    settings = site_selection_settings or {}
    manual_sites = [str(site) for site in (settings.get("manual_sites") or []) if str(site).strip()]
    site_stats = compute_site_hs_p95(context)
    if manual_sites:
        selected = site_stats.loc[site_stats["site"].isin(manual_sites)].copy()
        missing = [site for site in manual_sites if site not in set(selected["site"].astype(str))]
        if missing:
            raise KeyError(f"Unknown site(s) requested in manual_sites: {missing}")
        selected["site"] = pd.Categorical(selected["site"], categories=manual_sites, ordered=True)
        selected = (
            selected.sort_values("site")
            .assign(site=lambda df: df["site"].astype(str))
            .reset_index(drop=True)
        )
        return selected
    return choose_representative_sites(context)


def find_storm_window(
    site_frame: pd.DataFrame, window_days: int
) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    target_hs = pd.to_numeric(site_frame["hs"], errors="coerce")
    peak_ts = pd.Timestamp(target_hs.idxmax())
    half_window = pd.Timedelta(days=window_days) / 2
    start = max(site_frame.index.min(), peak_ts - half_window)
    end = min(site_frame.index.max(), peak_ts + half_window)
    return start, end, peak_ts


def _smooth_monthly_series(values: pd.Series, window: int) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if window <= 1 or len(arr) == 0:
        return arr
    half = window // 2
    padded = np.concatenate([arr[-half:], arr, arr[:half]])
    kernel = np.ones(window, dtype=float) / float(window)
    smoothed = np.convolve(padded, kernel, mode="valid")
    return smoothed[: len(arr)]


def _window_frame(
    frame: pd.DataFrame | None, start: pd.Timestamp, end: pd.Timestamp
) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    return frame.loc[(frame.index >= start) & (frame.index <= end)].copy()


def _site_dynamic_frame(context: dict[str, Any], site_name: str) -> pd.DataFrame:
    processed = context["processed"]
    npz = processed["site_dynamic_npz"]
    key = f"XdynamicSite__{site_name}"
    arr = np.asarray(npz[key], dtype=np.float32)
    return pd.DataFrame(
        arr, columns=processed["site_dynamic_feature_names"], index=processed["timestamps"]
    )


def _direction_legend(ax: plt.Axes, items: list[tuple[str, str]]) -> None:
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markersize=7,
            markerfacecolor=color,
            markeredgecolor=color,
            color=color,
            label=label,
        )
        for label, color in items
    ]
    ax.legend(
        handles=handles,
        loc="upper left",
        frameon=True,
        facecolor="white",
        framealpha=0.9,
        edgecolor="#cccccc",
        fontsize=10,
    )


def run_timeseries_section(
    context: dict[str, Any],
    window_days: int = 21,
    figure_dir: str | Path | None = None,
    *,
    selected_sites: list[str] | pd.DataFrame | None = None,
    plot_settings: dict[str, Any] | None = None,
    figure_tag: str = "",
    title_prefix: str = "",
) -> pd.DataFrame:
    figure_dir = figure_dir or context["figure_dir"]
    plot_settings = plot_settings or default_plot_settings()
    point_size = float(plot_settings.get("direction_point_size", 10))
    point_alpha = float(plot_settings.get("direction_point_alpha", 0.35))
    tag = f"{figure_tag}_" if figure_tag else ""
    prefix = title_prefix
    if isinstance(selected_sites, pd.DataFrame):
        selected_sites_df = selected_sites.copy()
    elif selected_sites:
        selected_sites_df = resolve_selected_sites(context, {"manual_sites": list(selected_sites)})
    else:
        selected_sites_df = choose_representative_sites(context)
    display(selected_sites_df)

    for row in selected_sites_df.to_dict(orient="records"):
        site_name = str(row["site"])
        nearshore_frame = context["nearshore_frames"][site_name]
        window_start, window_end, peak_ts = find_storm_window(
            nearshore_frame, window_days=window_days
        )
        wave_source = context["site_to_wave_site"].get(site_name)
        offshore_wave_frame = _window_frame(
            context["offshore_wave_frames"].get(wave_source), window_start, window_end
        )
        nearshore_window = _window_frame(nearshore_frame, window_start, window_end)
        offshore_wind_source = context["wave_to_wind_site"].get(wave_source)
        offshore_wind_window = _window_frame(
            context["offshore_wind_frames"].get(offshore_wind_source), window_start, window_end
        )
        local_wind_source = context["local_wind_map"].get(site_name)
        local_wind_window = _window_frame(
            context["offshore_wind_frames"].get(local_wind_source), window_start, window_end
        )

        fig, axes = plt.subplots(6, 1, figsize=(16, 20), sharex=True, constrained_layout=False)
        axes[0].plot(
            offshore_wave_frame.index,
            offshore_wave_frame.get("hs", pd.Series(dtype=float)),
            label=f"{wave_source} Hs",
            color=PLOT_COLORS["offshore"],
        )
        axes[0].plot(
            nearshore_window.index,
            nearshore_window.get("hs", pd.Series(dtype=float)),
            label=f"{site_name} Hs",
            color=PLOT_COLORS["nearshore"],
        )
        axes[0].set_ylabel("Hs [m]")
        axes[0].legend(frameon=False, ncol=2)

        axes[1].plot(
            offshore_wave_frame.index,
            offshore_wave_frame.get("tp", pd.Series(dtype=float)),
            label=f"{wave_source} Tp",
            color=PLOT_COLORS["offshore"],
        )
        axes[1].plot(
            nearshore_window.index,
            nearshore_window.get("tp", pd.Series(dtype=float)),
            label=f"{site_name} Tp",
            color=PLOT_COLORS["nearshore"],
        )
        axes[1].set_ylabel("Tp [s]")
        axes[1].legend(frameon=False, ncol=2)

        axes[2].plot(
            offshore_wind_window.index,
            offshore_wind_window.get("wind_speed_10m", pd.Series(dtype=float)),
            label=f"{offshore_wind_source} offshore wind",
            color=PLOT_COLORS["wind"],
        )
        if not local_wind_window.empty:
            axes[2].plot(
                local_wind_window.index,
                local_wind_window.get("wind_speed_10m", pd.Series(dtype=float)),
                label=f"{local_wind_source} local wind",
                color="#577590",
            )
        axes[2].set_ylabel("Wind [m/s]")
        axes[2].legend(frameon=False, ncol=2)

        axes[3].scatter(
            offshore_wave_frame.index,
            offshore_wave_frame.get("thq", pd.Series(dtype=float)),
            s=point_size,
            alpha=point_alpha,
            color=PLOT_COLORS["offshore"],
        )
        axes[3].scatter(
            nearshore_window.index,
            nearshore_window.get("dir", pd.Series(dtype=float)),
            s=point_size,
            alpha=point_alpha,
            color=PLOT_COLORS["nearshore"],
        )
        axes[3].set_ylabel("Dir [deg]")
        axes[3].set_ylim(0, 360)
        _direction_legend(
            axes[3],
            [
                (f"{wave_source} thq", PLOT_COLORS["offshore"]),
                (f"{site_name} dir", PLOT_COLORS["nearshore"]),
            ],
        )

        axes[4].scatter(
            offshore_wave_frame.index,
            offshore_wave_frame.get("Pdir", pd.Series(dtype=float)),
            s=point_size,
            alpha=point_alpha,
            color=PLOT_COLORS["offshore"],
        )
        axes[4].scatter(
            nearshore_window.index,
            nearshore_window.get("dp", pd.Series(dtype=float)),
            s=point_size,
            alpha=point_alpha,
            color=PLOT_COLORS["nearshore"],
        )
        axes[4].set_ylabel("Dp [deg]")
        axes[4].set_ylim(0, 360)
        _direction_legend(
            axes[4],
            [
                (f"{wave_source} Pdir", PLOT_COLORS["offshore"]),
                (f"{site_name} dp", PLOT_COLORS["nearshore"]),
            ],
        )

        legend_items = [(f"{offshore_wind_source} offshore wind dir", PLOT_COLORS["wind"])]
        axes[5].scatter(
            offshore_wind_window.index,
            offshore_wind_window.get("wind_direction_10m", pd.Series(dtype=float)),
            s=point_size,
            alpha=point_alpha,
            color=PLOT_COLORS["wind"],
        )
        if not local_wind_window.empty:
            axes[5].scatter(
                local_wind_window.index,
                local_wind_window.get("wind_direction_10m", pd.Series(dtype=float)),
                s=point_size,
                alpha=point_alpha,
                color="#577590",
            )
            legend_items.append((f"{local_wind_source} local wind dir", "#577590"))
        axes[5].set_ylabel("Wind dir [deg]")
        axes[5].set_ylim(0, 360)
        _direction_legend(axes[5], legend_items)
        axes[5].set_xlabel("Timestamp")

        fig.suptitle(
            f"{prefix}{site_name} ({row['split']} split) storm-centered window\n"
            f"Peak nearshore Hs at {peak_ts} | nearest wave source: {wave_source} | local wind source: {local_wind_source}",
            y=0.975,
        )
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
        save_fig(fig, figure_dir, f"{tag}timeseries_{site_name}")
        show_and_close(fig)

    return selected_sites_df


def run_seasonal_section(
    context: dict[str, Any],
    figure_dir: str | Path | None = None,
    *,
    nearshore_df: pd.DataFrame | None = None,
    scalar_df: pd.DataFrame | None = None,
    wind_df: pd.DataFrame | None = None,
    plot_settings: dict[str, Any] | None = None,
    figure_tag: str = "",
    title_prefix: str = "",
) -> dict[str, Any]:
    figure_dir = figure_dir or context["figure_dir"]
    plot_settings = plot_settings or default_plot_settings()
    climatology_style = str(plot_settings.get("climatology_style", "raw_line")).lower()
    smoothing_window = int(plot_settings.get("climatology_smoothing_window", 3))
    seasonal_sample_size = int(plot_settings.get("seasonal_box_sample_size", 250_000))
    nearshore_raw = (
        nearshore_df.copy() if nearshore_df is not None else context["nearshore_raw"].copy()
    )
    offshore_wind_raw = (
        wind_df.copy() if wind_df is not None else context["offshore_wind_raw"].copy()
    )
    scalar_df = scalar_df.copy() if scalar_df is not None else get_aligned_scalar_df(context).copy()
    tag = f"{figure_tag}_" if figure_tag else ""
    prefix = title_prefix

    nearshore_raw["month_num"] = nearshore_raw["timestamp"].dt.month.astype("int8")
    offshore_wind_raw["month_num"] = offshore_wind_raw["timestamp"].dt.month.astype("int8")
    scalar_df["month_num"] = scalar_df["timestamp"].dt.month.astype("int8")

    nearshore_monthly = (
        nearshore_raw.groupby("month_num")["hs"]
        .agg(median="median", p95=lambda x: np.nanquantile(x, 0.95))
        .reset_index()
    )
    offshore_monthly = (
        scalar_df.groupby("month_num")["offshore_hs"]
        .agg(median="median", p95=lambda x: np.nanquantile(x, 0.95))
        .reset_index()
    )
    wind_monthly = (
        offshore_wind_raw.groupby("month_num")["wind_speed_10m"]
        .median()
        .rename("median")
        .reset_index()
    )
    annual_counts = (
        nearshore_raw.groupby(nearshore_raw["timestamp"].dt.year)["target_row_valid"]
        .sum()
        .rename("valid_target_rows")
        .reset_index()
        .rename(columns={"timestamp": "year"})
    )

    def _to_percent_change_frame(frame: pd.DataFrame, value_columns: list[str]) -> pd.DataFrame:
        percent_frame = frame.copy()
        for column in value_columns:
            mean_value = float(np.nanmean(percent_frame[column].to_numpy(dtype=float)))
            if not np.isfinite(mean_value) or np.isclose(mean_value, 0.0):
                percent_frame[column] = np.nan
            else:
                percent_frame[column] = (
                    (percent_frame[column].astype(float) / mean_value) - 1.0
                ) * 100.0
        return percent_frame

    def _seasonal_panel_title(text: str) -> str:
        return text

    def plot_monthly_lines(
        ax: plt.Axes,
        frame: pd.DataFrame,
        x_col: str,
        y_cols: list[tuple[str, str, str]],
        title: str,
        ylabel: str,
    ) -> None:
        x = frame[x_col].to_numpy(dtype=float)
        for col, label, color in y_cols:
            y = frame[col].to_numpy(dtype=float)
            if climatology_style == "smooth_line":
                y = _smooth_monthly_series(pd.Series(y), smoothing_window)
            ax.plot(
                x,
                y,
                label=label,
                color=color,
                linewidth=2.0,
                marker="o" if climatology_style == "raw_line" else None,
            )
        title_text = f"{prefix}{_seasonal_panel_title(title)}"
        is_percent_change = "% Change from Mean" in title
        ax.set_title(
            title_text,
            pad=16 if is_percent_change else 12,
            fontsize=12 if is_percent_change else 14,
        )
        ax.set_xlabel("Month")
        ax.set_ylabel(ylabel)
        ax.set_xticks(np.arange(1, 13))
        ax.legend(frameon=False)

    def _save_monthly_line_figure(
        frame: pd.DataFrame,
        y_cols: list[tuple[str, str, str]],
        title: str,
        ylabel: str,
        file_name: str,
        *,
        add_zero_line: bool = False,
        figsize: tuple[float, float] = (8.8, 5.8),
    ) -> None:
        fig, ax = plt.subplots(1, 1, figsize=figsize, constrained_layout=True)
        plot_monthly_lines(ax, frame, "month_num", y_cols, title, ylabel)
        if add_zero_line:
            ax.axhline(0.0, color="0.4", linewidth=1.0, linestyle="--")
        save_fig(fig, figure_dir, f"{tag}{file_name}")
        show_and_close(fig)

    _save_monthly_line_figure(
        nearshore_monthly,
        [("median", "Median", PLOT_COLORS["nearshore"]), ("p95", "P95", "#4c78a8")],
        "Monthly Nearshore Hs Climatology",
        "Hs [m]",
        "seasonal_nearshore_monthly_hs_climatology",
    )
    _save_monthly_line_figure(
        offshore_monthly,
        [("median", "Median", PLOT_COLORS["offshore"]), ("p95", "P95", "#f4a261")],
        "Monthly Offshore Nearest-source Hs Climatology",
        "Hs [m]",
        "seasonal_offshore_monthly_hs_climatology",
    )
    _save_monthly_line_figure(
        wind_monthly,
        [("median", "Median", PLOT_COLORS["wind"])],
        "Monthly Offshore Wind Speed Median",
        "Wind speed [m/s]",
        "seasonal_offshore_monthly_wind_median",
    )

    nearshore_monthly_pct = _to_percent_change_frame(nearshore_monthly, ["median", "p95"])
    offshore_monthly_pct = _to_percent_change_frame(offshore_monthly, ["median", "p95"])
    wind_monthly_pct = _to_percent_change_frame(wind_monthly, ["median"])

    _save_monthly_line_figure(
        nearshore_monthly_pct,
        [("median", "Median", PLOT_COLORS["nearshore"]), ("p95", "P95", "#4c78a8")],
        "Monthly Nearshore Hs Climatology (% Change from Mean)",
        "% change from mean",
        "seasonal_nearshore_monthly_hs_climatology_percent_change",
        add_zero_line=True,
        figsize=(8.8, 6.5),
    )
    _save_monthly_line_figure(
        offshore_monthly_pct,
        [("median", "Median", PLOT_COLORS["offshore"]), ("p95", "P95", "#f4a261")],
        "Monthly Offshore Nearest-source Hs Climatology (% Change from Mean)",
        "% change from mean",
        "seasonal_offshore_monthly_hs_climatology_percent_change",
        add_zero_line=True,
        figsize=(8.8, 6.5),
    )
    _save_monthly_line_figure(
        wind_monthly_pct,
        [("median", "Median", PLOT_COLORS["wind"])],
        "Monthly Offshore Wind Speed Median (% Change from Mean)",
        "% change from mean",
        "seasonal_offshore_monthly_wind_median_percent_change",
        add_zero_line=True,
        figsize=(8.8, 6.5),
    )

    nearshore_seasonal = nearshore_raw.assign(
        season=nearshore_raw["month_num"].map(season_from_month)
    )
    offshore_seasonal = scalar_df.assign(season=scalar_df["month_num"].map(season_from_month))
    nearshore_sample = _sample_rows(
        nearshore_seasonal[["season", "hs", "tp"]].dropna(), seasonal_sample_size, seed=11
    )
    offshore_sample = _sample_rows(
        offshore_seasonal[["season", "offshore_hs", "offshore_tp"]].dropna(),
        seasonal_sample_size,
        seed=12,
    )

    fig, axes = plt.subplots(2, 2, figsize=(17, 11.4), constrained_layout=True)
    sns.boxplot(
        data=nearshore_sample,
        x="season",
        y="hs",
        order=SEASON_ORDER,
        ax=axes[0, 0],
        color=PLOT_COLORS["nearshore"],
        showfliers=False,
    )
    axes[0, 0].set_title(
        f"{prefix}{_seasonal_panel_title('Nearshore Hs by Season')}", pad=12, fontsize=14
    )
    sns.boxplot(
        data=nearshore_sample,
        x="season",
        y="tp",
        order=SEASON_ORDER,
        ax=axes[0, 1],
        color=PLOT_COLORS["nearshore"],
        showfliers=False,
    )
    axes[0, 1].set_title(
        f"{prefix}{_seasonal_panel_title('Nearshore Tp by Season')}", pad=12, fontsize=14
    )
    sns.boxplot(
        data=offshore_sample,
        x="season",
        y="offshore_hs",
        order=SEASON_ORDER,
        ax=axes[1, 0],
        color=PLOT_COLORS["offshore"],
        showfliers=False,
    )
    axes[1, 0].set_title(
        f"{prefix}{_seasonal_panel_title('Offshore Nearest-source Hs by Season')}",
        pad=12,
        fontsize=14,
    )
    sns.boxplot(
        data=offshore_sample,
        x="season",
        y="offshore_tp",
        order=SEASON_ORDER,
        ax=axes[1, 1],
        color=PLOT_COLORS["offshore"],
        showfliers=False,
    )
    axes[1, 1].set_title(
        f"{prefix}{_seasonal_panel_title('Offshore Nearest-source Tp by Season')}",
        pad=12,
        fontsize=14,
    )
    save_fig(fig, figure_dir, f"{tag}seasonal_hs_tp_distributions")
    show_and_close(fig)

    return {
        "nearshore_monthly": nearshore_monthly,
        "offshore_monthly": offshore_monthly,
        "wind_monthly": wind_monthly,
        "nearshore_monthly_percent_change": nearshore_monthly_pct,
        "offshore_monthly_percent_change": offshore_monthly_pct,
        "wind_monthly_percent_change": wind_monthly_pct,
        "annual_counts": annual_counts,
    }


def run_directional_section(
    context: dict[str, Any],
    figure_dir: str | Path | None = None,
    *,
    nearshore_df: pd.DataFrame | None = None,
    wave_df: pd.DataFrame | None = None,
    wind_df: pd.DataFrame | None = None,
    plot_settings: dict[str, Any] | None = None,
    figure_tag: str = "",
    title_prefix: str = "",
) -> dict[str, Any]:
    figure_dir = figure_dir or context["figure_dir"]
    plot_settings = plot_settings or default_plot_settings()
    rose_bins = int((plot_settings.get("rose_bins", {}) or {}).get("comparison", 18))
    nearshore_raw = (
        nearshore_df.copy() if nearshore_df is not None else context["nearshore_raw"].copy()
    )
    wave_df = wave_df.copy() if wave_df is not None else context["offshore_wave_raw"].copy()
    wind_df = wind_df.copy() if wind_df is not None else context["offshore_wind_raw"].copy()
    tag = f"{figure_tag}_" if figure_tag else ""
    prefix = title_prefix

    fig, axes = plt.subplots(
        1, 3, figsize=(18, 6), subplot_kw={"projection": "polar"}, constrained_layout=True
    )
    plot_rose(axes[0], wave_df["thq"], bins=rose_bins, color=PLOT_COLORS["offshore"])
    axes[0].set_title(f"{prefix}Offshore Mean Wave Direction")
    plot_rose(axes[1], wave_df["Pdir"], bins=rose_bins, color="#e76f51")
    axes[1].set_title(f"{prefix}Offshore Peak Wave Direction")
    plot_rose(axes[2], wind_df["wind_direction_10m"], bins=rose_bins, color=PLOT_COLORS["wind"])
    axes[2].set_title(f"{prefix}Offshore Wind Direction")
    save_fig(fig, figure_dir, f"{tag}directional_offshore_diagnostics")
    show_and_close(fig)

    fig, axes = plt.subplots(
        1, 2, figsize=(12, 6), subplot_kw={"projection": "polar"}, constrained_layout=True
    )
    plot_rose(axes[0], nearshore_raw["dir"], bins=rose_bins, color=PLOT_COLORS["nearshore"])
    axes[0].set_title(f"{prefix}Nearshore Mean Wave Direction")
    plot_rose(axes[1], nearshore_raw["dp"], bins=rose_bins, color="#4c78a8")
    axes[1].set_title(f"{prefix}Nearshore Peak Wave Direction")
    save_fig(fig, figure_dir, f"{tag}directional_nearshore_diagnostics")
    show_and_close(fig)

    fig, axes = plt.subplots(
        1, 2, figsize=(12, 6), subplot_kw={"projection": "polar"}, constrained_layout=True
    )
    overlay_rose(
        axes[0],
        [
            (wave_df["thq"], "Offshore thq", PLOT_COLORS["offshore"]),
            (nearshore_raw["dir"], "Nearshore dir", PLOT_COLORS["nearshore"]),
        ],
        bins=rose_bins,
    )
    axes[0].set_title(f"{prefix}Offshore vs Nearshore Mean Direction")
    overlay_rose(
        axes[1],
        [
            (wave_df["Pdir"], "Offshore Pdir", PLOT_COLORS["offshore"]),
            (nearshore_raw["dp"], "Nearshore dp", PLOT_COLORS["nearshore"]),
        ],
        bins=rose_bins,
    )
    axes[1].set_title(f"{prefix}Offshore vs Nearshore Peak Direction")
    save_fig(fig, figure_dir, f"{tag}directional_offshore_vs_nearshore_comparison")
    show_and_close(fig)

    return {
        "offshore_wave_rows": len(wave_df),
        "offshore_wind_rows": len(wind_df),
        "nearshore_rows": len(nearshore_raw),
    }


def validate_reference_alignment(
    context: dict[str, Any], site_name: str | None = None
) -> pd.DataFrame:
    processed = context["processed"]
    y_npz = processed["y_npz"]
    site_name = site_name or processed["target_sites"][0]
    wave_source = context["site_to_wave_site"][site_name]
    raw_wave = context["offshore_wave_frames"][wave_source].copy()
    raw_wave = raw_wave.loc[
        (raw_wave.index >= context["time_start"]) & (raw_wave.index <= context["time_end"])
    ]
    raw_wave = raw_wave.reindex(processed["timestamps"])
    ref = np.asarray(y_npz[f"Yreference__{site_name}"], dtype=np.float32)
    compare = pd.DataFrame(
        {
            "timestamp": processed["timestamps"],
            "ref_hs_npz": ref[:, 0],
            "ref_tp_npz": ref[:, 1],
            "ref_dir_npz": ref[:, 2],
            "ref_dp_npz": ref[:, 3],
            "raw_hs": pd.to_numeric(raw_wave["hs"], errors="coerce").to_numpy(dtype=np.float32),
            "raw_tp": pd.to_numeric(raw_wave["tp"], errors="coerce").to_numpy(dtype=np.float32),
            "raw_dir": pd.to_numeric(raw_wave["thq"], errors="coerce").to_numpy(dtype=np.float32),
            "raw_dp": pd.to_numeric(raw_wave["Pdir"], errors="coerce").to_numpy(dtype=np.float32),
        }
    )
    checks = {
        "hs": np.allclose(compare["ref_hs_npz"], compare["raw_hs"], atol=1e-5, equal_nan=True),
        "tp": np.allclose(compare["ref_tp_npz"], compare["raw_tp"], atol=1e-5, equal_nan=True),
        "dir": np.allclose(compare["ref_dir_npz"], compare["raw_dir"], atol=1e-5, equal_nan=True),
        "dp": np.allclose(compare["ref_dp_npz"], compare["raw_dp"], atol=1e-5, equal_nan=True),
    }
    transfer_reference = (context["metadata"].get("targets", {}) or {}).get(
        "transfer_reference", "unknown"
    )
    expect_match = str(transfer_reference).lower() in {
        "nearest",
        "nearest_source",
        "site_to_wave_site",
    }
    result = pd.DataFrame(
        [
            {
                "site": site_name,
                "wave_source": wave_source,
                "field": key,
                "matches_nearest_raw": value,
                "stored_transfer_reference": transfer_reference,
            }
            for key, value in checks.items()
        ]
    )
    if expect_match and not all(checks.values()):
        raise AssertionError(f"Reference alignment failed for {site_name}: {checks}")
    display(result)
    return result


def validate_split_labels(context: dict[str, Any]) -> pd.DataFrame:
    nearshore_raw = context["nearshore_raw"]
    metadata_sets = context["split_sets"]
    summary = pd.DataFrame(
        [
            {
                "split": "train",
                "expected_sites": len(metadata_sets.get("train", set())),
                "observed_sites": int(
                    nearshore_raw.loc[nearshore_raw["split"] == "train", "site"].nunique()
                ),
            },
            {
                "split": "val",
                "expected_sites": len(metadata_sets.get("val", set())),
                "observed_sites": int(
                    nearshore_raw.loc[nearshore_raw["split"] == "val", "site"].nunique()
                ),
            },
            {
                "split": "test",
                "expected_sites": len(metadata_sets.get("test", set())),
                "observed_sites": int(
                    nearshore_raw.loc[nearshore_raw["split"] == "test", "site"].nunique()
                ),
            },
        ]
    )
    if not np.array_equal(
        summary["expected_sites"].to_numpy(), summary["observed_sites"].to_numpy()
    ):
        raise AssertionError("Observed split site counts do not match metadata split site counts.")
    display(summary)
    return summary
