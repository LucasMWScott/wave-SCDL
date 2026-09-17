"""Helpers for the report-ready correlation analysis notebook."""

from __future__ import annotations

import re
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml

from src.data_pipeline import PointCentricArrays, load_point_centric_arrays


GROUP_ORDER = [
    "Offshore Wave Height & Partitions",
    "Offshore Wave Period",
    "Offshore Wave Direction",
    "Offshore Wind",
    "Wave-Aligned Local Exposure",
    "Local Wind",
    "Coastal Path Geometry",
    "Directional Fetch & Openness",
    "Bathymetry & Seabed Relief",
    "Bottlenecks & Funneling",
    "Porosity & Fjordness",
    "Site Regime & Breaking Limits",
    "Nearshore Targets",
]
GROUP_ORDER_MAP = {name: idx for idx, name in enumerate(GROUP_ORDER)}
COMPASS_ORDER = {
    "N": 0,
    "NNE": 1,
    "NE": 2,
    "ENE": 3,
    "E": 4,
    "ESE": 5,
    "SE": 6,
    "SSE": 7,
    "S": 8,
    "SSW": 9,
    "SW": 10,
    "WSW": 11,
    "W": 12,
    "WNW": 13,
    "NW": 14,
    "NNW": 15,
}

OFFSHORE_FEATURE_SPECS: list[dict[str, Any]] = [
    {
        "input_suffixes": ("_hs",),
        "raw_name": "offshore_hs",
        "display_name": "Offshore Hs",
        "group_name": "Offshore Wave Height & Partitions",
    },
    {
        "input_suffixes": ("_hs_sea",),
        "raw_name": "offshore_hs_sea",
        "display_name": "Offshore Wind-Sea Hs",
        "group_name": "Offshore Wave Height & Partitions",
    },
    {
        "input_suffixes": ("_hs_swell",),
        "raw_name": "offshore_hs_swell",
        "display_name": "Offshore Swell Hs",
        "group_name": "Offshore Wave Height & Partitions",
    },
    {
        "input_suffixes": ("_tp",),
        "raw_name": "offshore_tp",
        "display_name": "Offshore Tp",
        "group_name": "Offshore Wave Period",
    },
    {
        "input_suffixes": ("_tm1",),
        "raw_name": "offshore_tm1",
        "display_name": "Offshore Tm1",
        "group_name": "Offshore Wave Period",
    },
    {
        "input_suffixes": ("_tm2",),
        "raw_name": "offshore_tm2",
        "display_name": "Offshore Tm2",
        "group_name": "Offshore Wave Period",
    },
    {
        "input_suffixes": ("_tmp",),
        "raw_name": "offshore_tmp",
        "display_name": "Offshore Mean Period",
        "group_name": "Offshore Wave Period",
    },
    {
        "input_suffixes": ("_tp_sea",),
        "raw_name": "offshore_tp_sea",
        "display_name": "Offshore Wind-Sea Tp",
        "group_name": "Offshore Wave Period",
    },
    {
        "input_suffixes": ("_tp_swell",),
        "raw_name": "offshore_tp_swell",
        "display_name": "Offshore Swell Tp",
        "group_name": "Offshore Wave Period",
    },
    {
        "input_suffixes": ("_Pdir_sin", "_Pdir"),
        "raw_name": "offshore_peak_direction_sin",
        "display_name": "Offshore Peak Direction (sin)",
        "group_name": "Offshore Wave Direction",
        "transform": "circular_sin",
    },
    {
        "input_suffixes": ("_Pdir_cos", "_Pdir"),
        "raw_name": "offshore_peak_direction_cos",
        "display_name": "Offshore Peak Direction (cos)",
        "group_name": "Offshore Wave Direction",
        "transform": "circular_cos",
    },
    {
        "input_suffixes": ("_thq_sin", "_thq"),
        "raw_name": "offshore_mean_direction_sin",
        "display_name": "Offshore Mean Direction (sin)",
        "group_name": "Offshore Wave Direction",
        "transform": "circular_sin",
    },
    {
        "input_suffixes": ("_thq_cos", "_thq"),
        "raw_name": "offshore_mean_direction_cos",
        "display_name": "Offshore Mean Direction (cos)",
        "group_name": "Offshore Wave Direction",
        "transform": "circular_cos",
    },
    {
        "input_suffixes": ("_thq_sea_sin", "_thq_sea"),
        "raw_name": "offshore_wind_sea_direction_sin",
        "display_name": "Offshore Wind-Sea Direction (sin)",
        "group_name": "Offshore Wave Direction",
        "transform": "circular_sin",
    },
    {
        "input_suffixes": ("_thq_sea_cos", "_thq_sea"),
        "raw_name": "offshore_wind_sea_direction_cos",
        "display_name": "Offshore Wind-Sea Direction (cos)",
        "group_name": "Offshore Wave Direction",
        "transform": "circular_cos",
    },
    {
        "input_suffixes": ("_thq_swell_sin", "_thq_swell"),
        "raw_name": "offshore_swell_direction_sin",
        "display_name": "Offshore Swell Direction (sin)",
        "group_name": "Offshore Wave Direction",
        "transform": "circular_sin",
    },
    {
        "input_suffixes": ("_thq_swell_cos", "_thq_swell"),
        "raw_name": "offshore_swell_direction_cos",
        "display_name": "Offshore Swell Direction (cos)",
        "group_name": "Offshore Wave Direction",
        "transform": "circular_cos",
    },
    {
        "input_suffixes": ("_wind_speed_10m",),
        "raw_name": "offshore_wind_speed_10m",
        "display_name": "Offshore Wind Speed (10 m)",
        "group_name": "Offshore Wind",
    },
    {
        "input_suffixes": ("_wind_direction_10m_sin", "_wind_direction_10m"),
        "raw_name": "offshore_wind_direction_sin",
        "display_name": "Offshore Wind Direction (sin)",
        "group_name": "Offshore Wind",
        "transform": "circular_sin",
    },
    {
        "input_suffixes": ("_wind_direction_10m_cos", "_wind_direction_10m"),
        "raw_name": "offshore_wind_direction_cos",
        "display_name": "Offshore Wind Direction (cos)",
        "group_name": "Offshore Wind",
        "transform": "circular_cos",
    },
]

SITE_DYNAMIC_LABEL_OVERRIDES = {
    "wave_fetch_aligned_m": "Wave-Aligned Fetch (m)",
    "wave_fetch_aligned_ratio": "Wave-Aligned Fetch Ratio",
    "wave_slope_aligned": "Wave-Aligned Slope",
    "wave_laplacian_aligned": "Wave-Aligned Laplacian",
    "wave_min_depth_aligned_m": "Wave-Aligned Minimum Depth (m)",
    "wave_blocked_sector_fraction_pm30": "Wave Blocking Fraction (+/-30 deg)",
    "wave_open_sector_fraction_pm30": "Wave Open Fraction (+/-30 deg)",
    "local_wind_fetch_aligned_m": "Local-Wind-Aligned Fetch (m)",
    "local_wind_fetch_aligned_ratio": "Local-Wind-Aligned Fetch Ratio",
    "local_windsea_proxy_u2_fetch": "Local Windsea Proxy (U^2 x Fetch)",
    "local_windsea_proxy_u2_fetch_ratio": "Local Windsea Proxy (U^2 x Fetch Ratio)",
    "local_windsea_proxy_3h_mean": "Local Windsea Proxy (3 h Mean)",
    "local_windsea_proxy_6h_mean": "Local Windsea Proxy (6 h Mean)",
    "local_windsea_proxy_12h_mean": "Local Windsea Proxy (12 h Mean)",
    "local_wind_speed_10m": "Local Wind Speed (10 m)",
    "local_wind_dir_sin": "Local Wind Direction (sin)",
    "local_wind_dir_cos": "Local Wind Direction (cos)",
}
TARGET_LABEL_OVERRIDES = {
    "target_hs": "Nearshore Hs",
    "target_tp": "Nearshore Tp",
    "target_dir_sin": "Nearshore Direction (sin)",
    "target_dir_cos": "Nearshore Direction (cos)",
    "target_dp_sin": "Nearshore Dp (sin)",
    "target_dp_cos": "Nearshore Dp (cos)",
}
THESIS_PREDICTOR_LABELS = {
    "Offshore Wave Height & Partitions": "Offshore Hs\n& partitions",
    "Offshore Wave Period": "Offshore\nperiod",
    "Offshore Wave Direction": "Offshore\ndirection",
    "Offshore Wind": "Offshore\nwind",
    "Wave-Aligned Local Exposure": "Wave-aligned\nlocal exposure",
    "Local Wind": "Local\nwind",
    "Coastal Path Geometry": "Coastal path\ngeometry",
    "Directional Fetch & Openness": "Directional fetch\n& openness",
    "Bathymetry & Seabed Relief": "Bathymetry\n& relief",
    "Bottlenecks & Funneling": "Bottlenecks\n& funneling",
    "Porosity & Fjordness": "Porosity\n& fjordness",
    "Site Regime & Breaking Limits": "Site regime\n& breaking",
}
THESIS_TARGET_LABELS = {
    "Nearshore Hs": "Nearshore Hs",
    "Nearshore Tp": "Nearshore Tp",
    "Nearshore Direction (sin)": "Nearshore Mean Dir\n(sin)",
    "Nearshore Direction (cos)": "Nearshore Mean Dir\n(cos)",
    "Nearshore Dp (sin)": "Nearshore Peak Dir / Dp\n(sin)",
    "Nearshore Dp (cos)": "Nearshore Peak Dir / Dp\n(cos)",
}
THESIS_SEPARATOR_BOUNDARIES = [4, 6]
THESIS_HEATMAP_CMAP = "RdBu_r"
THESIS_FIGURE_DPI = 300
THESIS_TABLE_FLOAT_FORMAT = "%.4f"


@dataclass
class ThesisOutputManifestEntry:
    kind: str
    name: str
    path: Path


@dataclass
class PreparedCorrelationInputs:
    config_path: Path
    data_dir: Path
    arrays: PointCentricArrays
    training_cfg: dict[str, Any]
    all_indices: np.ndarray
    sites: list[str]
    offshore_matrix: np.ndarray
    offshore_metadata: pd.DataFrame
    site_dynamic_metadata: pd.DataFrame
    static_metadata: pd.DataFrame
    target_metadata: pd.DataFrame
    feature_metadata: pd.DataFrame


class RunningCorrelationAccumulator:
    """Streaming sufficient statistics for correlation matrices."""

    def __init__(self, feature_names: Sequence[str]) -> None:
        self.feature_names = list(feature_names)
        self.count = 0
        self.sum = np.zeros(len(feature_names), dtype=np.float64)
        self.cross = np.zeros((len(feature_names), len(feature_names)), dtype=np.float64)

    def update(self, matrix: np.ndarray) -> int:
        valid = np.isfinite(matrix).all(axis=1)
        clean = np.asarray(matrix[valid], dtype=np.float64)
        if clean.size == 0:
            return 0
        self.count += int(clean.shape[0])
        self.sum += clean.sum(axis=0, dtype=np.float64)
        self.cross += clean.T @ clean
        return int(clean.shape[0])

    def to_correlation_frame(self, labels: Sequence[str]) -> pd.DataFrame:
        if self.count < 2:
            raise ValueError("Need at least two finite rows to compute correlation.")
        means = self.sum / float(self.count)
        covariance = self.cross / float(self.count) - np.outer(means, means)
        variances = np.clip(np.diag(covariance), a_min=0.0, a_max=None)
        std = np.sqrt(variances)
        denom = np.outer(std, std)
        corr = np.divide(
            covariance,
            denom,
            out=np.full_like(covariance, np.nan, dtype=np.float64),
            where=denom > 0.0,
        )
        finite_diag = std > 0.0
        corr[finite_diag, finite_diag] = np.clip(corr[finite_diag, finite_diag], -1.0, 1.0)
        np.fill_diagonal(corr, np.where(finite_diag, 1.0, np.nan))
        return pd.DataFrame(corr, index=list(labels), columns=list(labels))


def resolve_repo_root() -> Path:
    cwd = Path.cwd().resolve()
    for candidate in [cwd, *cwd.parents]:
        if (candidate / "src").exists() and (candidate / "configs").exists():
            return candidate
    return cwd


def resolve_path(path_like: str | Path, repo_root: Path, anchor: str | Path | None = None) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path.resolve()
    candidates = [repo_root / path]
    if anchor is not None:
        anchor_path = Path(anchor).resolve()
        candidates.insert(0, anchor_path.parent / path)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def load_training_config(config_path: str | Path, repo_root: Path) -> tuple[Path, dict[str, Any]]:
    resolved = resolve_path(config_path, repo_root)
    with resolved.open("r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh) or {}
    return resolved, config


def prepare_correlation_inputs(
    config_path: str | Path, repo_root: Path
) -> PreparedCorrelationInputs:
    resolved_config_path, training_cfg = load_training_config(config_path, repo_root)
    data_cfg = training_cfg.get("data", {}) or {}
    data_dir = resolve_path(
        data_cfg.get("point_centric_dir", "data/processed/thesis_set_full_v3"),
        repo_root,
        anchor=resolved_config_path,
    )
    arrays = load_point_centric_arrays(str(data_dir))

    all_indices = np.unique(
        np.concatenate(
            [
                np.asarray(arrays.split_idx["train"], dtype=int),
                np.asarray(arrays.split_idx["val"], dtype=int),
                np.asarray(arrays.split_idx["test"], dtype=int),
            ]
        )
    )
    if all_indices.size == 0:
        raise ValueError("No train/val/test indices were found in the point-centric dataset.")

    offshore_matrix, offshore_metadata = build_offshore_dynamic_matrix(arrays, all_indices)
    site_dynamic_metadata = build_site_dynamic_metadata(arrays.site_dynamic_feature_names)
    static_metadata = build_static_metadata(arrays.static_feature_names)
    target_metadata = build_target_metadata(arrays.target_feature_names)
    feature_metadata = combine_feature_metadata(
        [offshore_metadata, site_dynamic_metadata, static_metadata, target_metadata]
    )

    return PreparedCorrelationInputs(
        config_path=resolved_config_path,
        data_dir=data_dir,
        arrays=arrays,
        training_cfg=training_cfg,
        all_indices=all_indices,
        sites=list(arrays.target_sites),
        offshore_matrix=offshore_matrix,
        offshore_metadata=offshore_metadata,
        site_dynamic_metadata=site_dynamic_metadata,
        static_metadata=static_metadata,
        target_metadata=target_metadata,
        feature_metadata=feature_metadata,
    )


def build_offshore_dynamic_matrix(
    arrays: PointCentricArrays,
    all_indices: np.ndarray,
) -> tuple[np.ndarray, pd.DataFrame]:
    dynamic_matrix = np.asarray(arrays.x_dynamic[all_indices], dtype=np.float32)
    dynamic_names = [str(name) for name in arrays.dynamic_feature_names]

    def cols_for_suffix(suffix: str) -> list[int]:
        return [idx for idx, name in enumerate(dynamic_names) if name.endswith(suffix)]

    collected_vectors: list[np.ndarray] = []
    metadata_rows: list[dict[str, Any]] = []

    for feature_order, spec in enumerate(OFFSHORE_FEATURE_SPECS):
        transform = str(spec.get("transform", "mean"))
        suffixes = tuple(spec["input_suffixes"])
        vector = build_offshore_vector(
            dynamic_matrix,
            dynamic_names,
            suffixes=suffixes,
            transform=transform,
            cols_for_suffix=cols_for_suffix,
        )
        if vector is None:
            continue
        collected_vectors.append(vector)
        metadata_rows.append(
            {
                "raw_name": spec["raw_name"],
                "display_name": spec["display_name"],
                "group_name": spec["group_name"],
                "group_order": GROUP_ORDER_MAP[spec["group_name"]],
                "feature_order": feature_order,
                "source_block": "offshore_dynamic",
            }
        )

    if not collected_vectors:
        raise ValueError(
            "No offshore dynamic features were aggregated from the point-centric dataset."
        )

    matrix = np.column_stack(collected_vectors).astype(np.float32, copy=False)
    metadata = pd.DataFrame(metadata_rows)
    return matrix, metadata


def build_offshore_vector(
    dynamic_matrix: np.ndarray,
    dynamic_names: Sequence[str],
    *,
    suffixes: Sequence[str],
    transform: str,
    cols_for_suffix,
) -> np.ndarray | None:
    first_suffix = suffixes[0]
    cols = cols_for_suffix(first_suffix)
    if transform == "mean":
        if not cols:
            return None
        return dynamic_matrix[:, cols].mean(axis=1)

    raw_suffix = suffixes[1] if len(suffixes) > 1 else None
    if cols:
        if transform == "circular_sin":
            return dynamic_matrix[:, cols].mean(axis=1)
        if transform == "circular_cos":
            return dynamic_matrix[:, cols].mean(axis=1)

    if raw_suffix is None:
        return None
    raw_cols = cols_for_suffix(raw_suffix)
    if not raw_cols:
        return None
    radians = np.deg2rad(dynamic_matrix[:, raw_cols])
    if transform == "circular_sin":
        return np.sin(radians).mean(axis=1)
    if transform == "circular_cos":
        return np.cos(radians).mean(axis=1)
    return None


def build_site_dynamic_metadata(site_dynamic_feature_names: Sequence[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for idx, name in enumerate([str(item) for item in site_dynamic_feature_names]):
        group_name = (
            "Local Wind"
            if name.startswith("local_wind") or name.startswith("local_windsea")
            else "Wave-Aligned Local Exposure"
        )
        rows.append(
            {
                "raw_name": name,
                "display_name": SITE_DYNAMIC_LABEL_OVERRIDES.get(name, prettify_generic_name(name)),
                "group_name": group_name,
                "group_order": GROUP_ORDER_MAP[group_name],
                "feature_order": idx,
                "source_block": "site_dynamic",
            }
        )
    return pd.DataFrame(rows)


def build_static_metadata(static_feature_names: Sequence[str]) -> pd.DataFrame:
    grouped_names: dict[str, list[str]] = {
        group_name: [] for group_name in GROUP_ORDER if group_name != "Nearshore Targets"
    }
    for name in [str(item) for item in static_feature_names]:
        group_name = infer_static_group(name)
        grouped_names[group_name].append(name)

    rows: list[dict[str, Any]] = []
    for group_name in [
        "Coastal Path Geometry",
        "Directional Fetch & Openness",
        "Bathymetry & Seabed Relief",
        "Bottlenecks & Funneling",
        "Porosity & Fjordness",
        "Site Regime & Breaking Limits",
    ]:
        ordered_names = sorted(grouped_names[group_name], key=static_sort_key)
        for feature_order, name in enumerate(ordered_names):
            rows.append(
                {
                    "raw_name": name,
                    "display_name": prettify_static_feature_name(name),
                    "group_name": group_name,
                    "group_order": GROUP_ORDER_MAP[group_name],
                    "feature_order": feature_order,
                    "source_block": "static",
                }
            )
    return pd.DataFrame(rows)


def build_target_metadata(target_feature_names: Sequence[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for idx, name in enumerate([str(item) for item in target_feature_names]):
        raw_name = f"target_{name}"
        rows.append(
            {
                "raw_name": raw_name,
                "display_name": TARGET_LABEL_OVERRIDES.get(
                    raw_name, prettify_generic_name(raw_name)
                ),
                "group_name": "Nearshore Targets",
                "group_order": GROUP_ORDER_MAP["Nearshore Targets"],
                "feature_order": idx,
                "source_block": "target",
            }
        )
    return pd.DataFrame(rows)


def combine_feature_metadata(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    metadata = pd.concat(list(frames), ignore_index=True)
    metadata["display_name"] = deduplicate_names(metadata["display_name"].tolist())
    return metadata


def deduplicate_names(names: Sequence[str]) -> list[str]:
    counts: dict[str, int] = {}
    deduped: list[str] = []
    for name in names:
        seen = counts.get(name, 0)
        counts[name] = seen + 1
        deduped.append(name if seen == 0 else f"{name} [{seen + 1}]")
    return deduped


def infer_static_group(name: str) -> str:
    lowered = name.lower()
    if (
        lowered.startswith("path_")
        or lowered.startswith("static_final_approach")
        or lowered.startswith("static_net_deflection")
        or lowered.startswith("static_signed_curvature")
        or lowered.startswith("static_tortuosity")
    ):
        return "Coastal Path Geometry"
    if (
        lowered.startswith("ray_fetch_")
        or lowered.startswith("fetch_")
        or "open_sector" in lowered
        or "closed_sector" in lowered
        or "dominant_fetch_direction" in lowered
        or "ray_hit_land_fraction" in lowered
        or lowered == "ray_count"
    ):
        return "Directional Fetch & Openness"
    if (
        lowered.startswith("ray_max_slope")
        or lowered.startswith("ray_max_laplacian")
        or lowered.startswith("ray_min_depth")
        or "depth" in lowered
        or "shore" in lowered
        or "coast" in lowered
        or "snap_distance" in lowered
    ):
        return "Bathymetry & Seabed Relief"
    if "bottleneck" in lowered or "funneling" in lowered or "choke" in lowered:
        return "Bottlenecks & Funneling"
    if "porosity" in lowered or "fjordness" in lowered:
        return "Porosity & Fjordness"
    if lowered.startswith("site_regime_") or "breaking" in lowered:
        return "Site Regime & Breaking Limits"
    return "Directional Fetch & Openness"


def static_sort_key(name: str) -> tuple[Any, ...]:
    direction_match = re.search(
        r"_(NNE|ENE|ESE|SSE|SSW|WSW|WNW|NNW|NE|SE|SW|NW|N|E|S|W)(?:_|$)", name
    )
    direction_order = COMPASS_ORDER.get(direction_match.group(1), 99) if direction_match else 99
    return (
        infer_static_group(name),
        re.sub(r"_(NNE|ENE|ESE|SSE|SSW|WSW|WNW|NNW|NE|SE|SW|NW|N|E|S|W)(?:_|$)", "_", name),
        direction_order,
        name,
    )


def prettify_static_feature_name(name: str) -> str:
    direct = {
        "path_length_m": "Path Length (m)",
        "path_bottleneck_m": "Path Bottleneck Width (m)",
        "path_direct_distance_m": "Direct Path Distance (m)",
        "static_local_depth_m": "Static Local Depth (m)",
        "static_nearest_shore_steepness": "Nearest Shore Steepness",
        "funneling_ratio": "Funneling Ratio",
        "choke_out_ratio": "Choke-Out Ratio",
        "fetch_max_over_mean": "Fetch Max / Mean",
        "fetch_std_m": "Fetch Standard Deviation (m)",
        "fetch_cv": "Fetch Coefficient of Variation",
        "fetch_directional_entropy": "Fetch Directional Entropy",
        "fetch_resultant_length": "Fetch Resultant Length",
        "open_sector_fraction": "Open Sector Fraction",
        "closed_sector_fraction": "Closed Sector Fraction",
        "open_sector_width_deg": "Open Sector Width (deg)",
        "fjordness_land_blocking_component": "Fjordness: Land Blocking",
        "fjordness_low_porosity_component": "Fjordness: Low Porosity",
        "fjordness_closed_sector_component": "Fjordness: Closed Sector",
        "fjordness_anisotropy_component": "Fjordness: Anisotropy",
        "fjordness_low_fetch_component": "Fjordness: Low Fetch",
        "fjordness_route_complexity_component": "Fjordness: Route Complexity",
        "fjordness_score": "Fjordness Score",
        "path_tortuosity_ratio": "Path Tortuosity Ratio",
        "bottleneck_to_path_ratio": "Bottleneck / Path Ratio",
        "bottleneck_to_fetch_ratio": "Bottleneck / Fetch Ratio",
        "funneling_log": "Log Funneling",
        "path_point_count": "Path Point Count",
        "snap_distance_m": "Snap Distance (m)",
        "static_tortuosity_sum": "Static Tortuosity Sum",
        "static_dist_to_coast_m": "Distance to Coast (m)",
        "local_depth_m": "Local Depth (m)",
        "local_breaking_hs_cap": "Local Breaking Hs Cap",
        "local_breaking_cap_valid": "Breaking Cap Valid",
        "ray_fetch_min_m": "Ray Fetch Minimum (m)",
        "ray_fetch_mean_m": "Ray Fetch Mean (m)",
        "ray_fetch_max_m": "Ray Fetch Maximum (m)",
        "ray_hit_land_fraction": "Ray Land-Hit Fraction",
        "ray_count": "Ray Count",
    }
    if name in direct:
        return direct[name]

    direction_match = re.match(
        r"^(ray_fetch|ray_max_slope|ray_max_laplacian|ray_min_depth)_([A-Z]+)(?:_m)?$", name
    )
    if direction_match:
        base, direction = direction_match.groups()
        base_label = {
            "ray_fetch": "Ray Fetch",
            "ray_max_slope": "Ray Max Slope",
            "ray_max_laplacian": "Ray Max Laplacian",
            "ray_min_depth": "Ray Minimum Depth",
        }[base]
        unit = " (m)" if name.endswith("_m") else ""
        return f"{base_label} {direction}{unit}"

    site_regime_match = re.match(r"^site_regime_(.+)$", name)
    if site_regime_match:
        return f"Site Regime: {site_regime_match.group(1).replace('_', ' ').title()}"

    if name.endswith("_sin") or name.endswith("_cos"):
        suffix = "sin" if name.endswith("_sin") else "cos"
        base = name[:-4] if suffix == "sin" else name[:-4]
        base = base.rsplit("_", 1)[0]
        return f"{prettify_generic_name(base)} ({suffix})"

    return prettify_generic_name(name)


def prettify_generic_name(name: str) -> str:
    label = str(name)
    label = label.replace("static_", "")
    label = label.replace("target_", "")
    label = label.replace("offshore_", "Offshore ")
    label = label.replace("_u2_", " U^2 ")
    label = label.replace("_pm30", " +/-30 deg")
    label = label.replace("_10m", " 10 m")
    label = label.replace("_1km", " 1 km")
    label = label.replace("_2km", " 2 km")
    label = label.replace("_5km", " 5 km")
    label = label.replace("_500m", " 500 m")
    label = label.replace("_12h", " 12 h")
    label = label.replace("_6h", " 6 h")
    label = label.replace("_3h", " 3 h")
    label = label.replace("_deg", " deg")
    label = label.replace("_m", " (m)")
    label = label.replace("_", " ")
    label = re.sub(r"\s+", " ", label).strip()
    token_overrides = {
        "hs": "Hs",
        "tp": "Tp",
        "tm1": "Tm1",
        "tm2": "Tm2",
        "tmp": "Mean Period",
        "dp": "Dp",
    }
    words = []
    for token in label.split(" "):
        low = token.lower()
        if low in token_overrides:
            words.append(token_overrides[low])
        elif token.upper() in COMPASS_ORDER:
            words.append(token.upper())
        elif low == "u^2":
            words.append("U^2")
        else:
            words.append(token.capitalize())
    return " ".join(words)


def get_selected_site_names(prepared: PreparedCorrelationInputs, site_subset: str) -> list[str]:
    if site_subset == "all":
        return list(prepared.sites)
    if site_subset != "test":
        raise ValueError(f"Unsupported site subset: {site_subset}")
    configured = [
        str(item) for item in (prepared.training_cfg.get("data", {}) or {}).get("test_sites", [])
    ]
    if not configured:
        raise ValueError("No test sites were configured in the selected training config.")
    available = set(prepared.sites)
    selected = [site for site in configured if site in available]
    if not selected:
        raise ValueError(
            "None of the configured test sites were found in the point-centric dataset."
        )
    return selected


def build_site_feature_chunk(
    prepared: PreparedCorrelationInputs,
    site_name: str,
) -> np.ndarray:
    feature_parts = [prepared.offshore_matrix]

    site_dynamic = prepared.arrays.x_dynamic_sitewise.get(site_name)
    if prepared.site_dynamic_metadata.shape[0] > 0:
        if site_dynamic is None:
            raise KeyError(f"Missing site-dynamic feature block for site '{site_name}'.")
        feature_parts.append(np.asarray(site_dynamic[prepared.all_indices], dtype=np.float32))

    static_vector = prepared.arrays.x_static.get(site_name)
    if prepared.static_metadata.shape[0] > 0:
        if static_vector is None:
            raise KeyError(f"Missing static feature block for site '{site_name}'.")
        static_tiled = np.repeat(
            np.asarray(static_vector, dtype=np.float32).reshape(1, -1),
            repeats=prepared.all_indices.size,
            axis=0,
        )
        feature_parts.append(static_tiled)

    targets = prepared.arrays.y_targets.get(site_name)
    if targets is None:
        raise KeyError(f"Missing targets for site '{site_name}'.")
    feature_parts.append(np.asarray(targets[prepared.all_indices], dtype=np.float32))
    return np.concatenate(feature_parts, axis=1)


def compute_detailed_correlation(
    prepared: PreparedCorrelationInputs,
    *,
    site_subset: str,
) -> dict[str, Any]:
    selected_sites = get_selected_site_names(prepared, site_subset)
    feature_labels = prepared.feature_metadata["display_name"].tolist()
    accumulator = RunningCorrelationAccumulator(feature_labels)
    finite_rows = 0

    for site_name in selected_sites:
        finite_rows += accumulator.update(build_site_feature_chunk(prepared, site_name))

    corr_df = accumulator.to_correlation_frame(feature_labels)
    ordered_labels = prepared.feature_metadata.sort_values(
        ["group_order", "feature_order", "display_name"]
    )["display_name"].tolist()
    corr_df = corr_df.loc[ordered_labels, ordered_labels]

    return {
        "corr_df": corr_df,
        "selected_sites": selected_sites,
        "finite_rows": finite_rows,
        "feature_labels": feature_labels,
    }


def compute_family_correlation(
    prepared: PreparedCorrelationInputs,
    *,
    site_subset: str,
) -> dict[str, Any]:
    selected_sites = get_selected_site_names(prepared, site_subset)
    feature_ordered = prepared.feature_metadata.sort_values(
        ["group_order", "feature_order", "display_name"]
    ).reset_index(drop=True)

    means_acc = RunningCorrelationAccumulator(feature_ordered["display_name"].tolist())
    for site_name in selected_sites:
        chunk = build_site_feature_chunk(prepared, site_name)
        ordered_chunk = reorder_chunk(chunk, prepared.feature_metadata, feature_ordered)
        means_acc.update(ordered_chunk)

    means = means_acc.sum / float(means_acc.count)
    covariance = means_acc.cross / float(means_acc.count) - np.outer(means, means)
    std = np.sqrt(np.clip(np.diag(covariance), a_min=0.0, a_max=None))

    family_labels, family_members = build_family_members(feature_ordered)
    family_acc = RunningCorrelationAccumulator(family_labels)
    finite_rows = 0
    for site_name in selected_sites:
        chunk = build_site_feature_chunk(prepared, site_name)
        ordered_chunk = reorder_chunk(chunk, prepared.feature_metadata, feature_ordered)
        family_chunk = build_family_chunk(
            ordered_chunk,
            means=means,
            std=std,
            family_labels=family_labels,
            family_members=family_members,
            feature_names=feature_ordered["display_name"].tolist(),
        )
        finite_rows += family_acc.update(family_chunk)

    corr_df = family_acc.to_correlation_frame(family_labels)
    return {
        "corr_df": corr_df.loc[family_labels, family_labels],
        "selected_sites": selected_sites,
        "finite_rows": finite_rows,
        "family_labels": family_labels,
    }


def reorder_chunk(
    chunk: np.ndarray,
    original_metadata: pd.DataFrame,
    ordered_metadata: pd.DataFrame,
) -> np.ndarray:
    order = [
        int(original_metadata.index[original_metadata["display_name"] == name][0])
        for name in ordered_metadata["display_name"]
    ]
    return chunk[:, order]


def build_family_members(feature_metadata: pd.DataFrame) -> tuple[list[str], dict[str, list[int]]]:
    family_labels: list[str] = []
    family_members: dict[str, list[int]] = {}

    for idx, row in feature_metadata.iterrows():
        if row["source_block"] == "target":
            label = str(row["display_name"])
        else:
            label = str(row["group_name"])
        if label not in family_members:
            family_members[label] = []
            family_labels.append(label)
        family_members[label].append(int(idx))
    return family_labels, family_members


def build_family_chunk(
    ordered_chunk: np.ndarray,
    *,
    means: np.ndarray,
    std: np.ndarray,
    family_labels: Sequence[str],
    family_members: dict[str, list[int]],
    feature_names: Sequence[str],
) -> np.ndarray:
    safe_std = np.where(std > 0.0, std, np.nan)
    z = (ordered_chunk - means.reshape(1, -1)) / safe_std.reshape(1, -1)
    family_vectors: list[np.ndarray] = []
    for label in family_labels:
        idx = family_members[label]
        if len(idx) == 1:
            vector = z[:, idx[0]]
            family_vectors.append(np.where(np.isfinite(vector), vector, 0.0))
            continue
        with np.errstate(invalid="ignore"):
            vector = np.nanmean(z[:, idx], axis=1)
        family_vectors.append(np.where(np.isfinite(vector), vector, 0.0))
    return np.column_stack(family_vectors)


def get_family_correlation_method_description() -> str:
    return (
        "Family-level correlations are computed on pooled rows from all included sites and all "
        "configured train, validation, and test indices. Each raw feature column is standardized "
        "using the pooled mean and pooled standard deviation across those rows. Each multi-feature "
        "predictor family is then represented by the row-wise mean of its standardized member "
        "columns, while each target remains a single standardized column. Pearson correlation is "
        "finally computed across those family-composite scores. Because averaging standardized "
        "members can allow mixed-sign feature relationships within a family to cancel, the notebook "
        "also exports companion member-level strongest-correlation diagnostics rather than silently "
        "replacing the historical aggregation."
    )


def get_ordered_feature_metadata(feature_metadata: pd.DataFrame) -> pd.DataFrame:
    return feature_metadata.sort_values(
        ["group_order", "feature_order", "display_name"]
    ).reset_index(drop=True)


def extract_predictor_family_labels(feature_metadata: pd.DataFrame) -> list[str]:
    ordered = get_ordered_feature_metadata(feature_metadata)
    family_labels, _ = build_family_members(ordered)
    target_labels = set(extract_target_labels(feature_metadata))
    return [label for label in family_labels if label not in target_labels]


def extract_target_labels(feature_metadata: pd.DataFrame) -> list[str]:
    ordered = get_ordered_feature_metadata(feature_metadata)
    return ordered.loc[ordered["source_block"] == "target", "display_name"].tolist()


def build_family_member_name_lookup(feature_metadata: pd.DataFrame) -> dict[str, list[str]]:
    ordered = get_ordered_feature_metadata(feature_metadata)
    family_labels, family_members = build_family_members(ordered)
    ordered_names = ordered["display_name"].tolist()
    return {
        label: [ordered_names[idx] for idx in member_idx]
        for label, member_idx in family_members.items()
        if label in family_labels
    }


def slice_family_correlation_matrices(
    family_corr_df: pd.DataFrame,
    *,
    predictor_labels: Sequence[str],
    target_labels: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    predictor_corr_df = family_corr_df.loc[list(predictor_labels), list(predictor_labels)].copy()
    predictor_target_corr_df = family_corr_df.loc[
        list(predictor_labels), list(target_labels)
    ].copy()
    return predictor_corr_df, predictor_target_corr_df


def get_thesis_family_plot_labels(
    labels: Sequence[str],
    *,
    targets: bool = False,
) -> list[str]:
    label_map = THESIS_TARGET_LABELS if targets else THESIS_PREDICTOR_LABELS
    return [
        label_map.get(label, wrap_axis_label(label, width=16 if targets else 14))
        for label in labels
    ]


def wrap_axis_label(label: str, *, width: int) -> str:
    return textwrap.fill(str(label), width=width, break_long_words=False)


def flatten_wrapped_label(label: str) -> str:
    return str(label).replace("\n", " ")


def set_thesis_plot_style() -> None:
    sns.set_theme(
        style="white",
        context="paper",
        font_scale=1.0,
        rc={
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "font.family": "DejaVu Serif",
            "axes.titlesize": 14,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        },
    )


def draw_group_separators(
    ax: plt.Axes, boundaries: Sequence[int], *, color: str = "#bdbdbd", linewidth: float = 0.9
) -> None:
    for boundary in boundaries:
        ax.axhline(boundary, color=color, linewidth=linewidth)
        ax.axvline(boundary, color=color, linewidth=linewidth)


def plot_predictor_family_heatmap(
    corr_df: pd.DataFrame,
    *,
    title: str = "Correlation Between Predictor Families",
) -> plt.Figure:
    set_thesis_plot_style()
    display_labels = get_thesis_family_plot_labels(corr_df.index.tolist())
    x_display_labels = [flatten_wrapped_label(label) for label in display_labels]
    mask = np.triu(np.ones_like(corr_df, dtype=bool), k=0) | corr_df.isna().to_numpy()
    fig, ax = plt.subplots(figsize=(10.4, 9.2), constrained_layout=True)
    sns.heatmap(
        corr_df,
        mask=mask,
        cmap=THESIS_HEATMAP_CMAP,
        center=0,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 9},
        linewidths=0.5,
        linecolor="#f2f2f2",
        cbar_kws={"label": "Pearson correlation", "shrink": 0.82},
        square=True,
        ax=ax,
    )
    draw_group_separators(ax, THESIS_SEPARATOR_BOUNDARIES)
    ax.set_title(title, pad=18)
    ax.tick_params(axis="x", labeltop=False, labelbottom=True, top=False, bottom=False, pad=8)
    ax.tick_params(axis="y", left=False)
    ax.set_xticklabels(
        x_display_labels, rotation=42, ha="right", rotation_mode="anchor", fontsize=9
    )
    ax.set_yticklabels(display_labels, rotation=0, fontsize=10)
    return fig


def plot_predictor_target_heatmap(
    corr_df: pd.DataFrame,
    *,
    title: str = "Predictor-Family Correlation with Nearshore Targets",
) -> plt.Figure:
    set_thesis_plot_style()
    row_labels = get_thesis_family_plot_labels(corr_df.index.tolist())
    col_labels = get_thesis_family_plot_labels(corr_df.columns.tolist(), targets=True)
    x_col_labels = [flatten_wrapped_label(label) for label in col_labels]
    fig, ax = plt.subplots(figsize=(10.0, 7.2), constrained_layout=True)
    sns.heatmap(
        corr_df,
        mask=corr_df.isna().to_numpy(),
        cmap=THESIS_HEATMAP_CMAP,
        center=0,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 9},
        linewidths=0.5,
        linecolor="#f2f2f2",
        cbar_kws={"label": "Pearson correlation", "shrink": 0.88},
        square=False,
        ax=ax,
    )
    draw_group_separators(ax, THESIS_SEPARATOR_BOUNDARIES, linewidth=0.8)
    ax.set_title(title, pad=14)
    ax.tick_params(axis="x", labeltop=False, labelbottom=True, top=False, bottom=False, pad=8)
    ax.tick_params(axis="y", left=False)
    ax.set_xticklabels(x_col_labels, rotation=38, ha="right", rotation_mode="anchor", fontsize=9)
    ax.set_yticklabels(row_labels, rotation=0, fontsize=10)
    return fig


def plot_appendix_full_family_heatmap(
    corr_df: pd.DataFrame,
    *,
    title: str = "Full Family and Target Correlation Matrix",
) -> plt.Figure:
    set_thesis_plot_style()
    labels = get_thesis_family_plot_labels(corr_df.index.tolist(), targets=False)
    labels = [
        THESIS_TARGET_LABELS.get(
            label, THESIS_PREDICTOR_LABELS.get(label, wrap_axis_label(label, width=15))
        )
        for label in corr_df.index.tolist()
    ]
    x_labels = [flatten_wrapped_label(label) for label in labels]
    fig, ax = plt.subplots(figsize=(13.0, 11.8), constrained_layout=True)
    sns.heatmap(
        corr_df,
        mask=corr_df.isna().to_numpy(),
        cmap=THESIS_HEATMAP_CMAP,
        center=0,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 7},
        linewidths=0.35,
        linecolor="#f2f2f2",
        cbar_kws={"label": "Pearson correlation", "shrink": 0.86},
        square=True,
        ax=ax,
    )
    full_boundaries = list(group_boundaries_from_family_labels(corr_df.index.tolist()))
    draw_group_separators(ax, full_boundaries, linewidth=0.8)
    ax.set_title(title, pad=16)
    ax.tick_params(axis="x", labeltop=False, labelbottom=True, top=False, bottom=False, pad=8)
    ax.tick_params(axis="y", left=False)
    ax.set_xticklabels(x_labels, rotation=50, ha="right", rotation_mode="anchor", fontsize=8)
    ax.set_yticklabels(labels, rotation=0, fontsize=8)
    return fig


def group_boundaries_from_family_labels(labels: Sequence[str]) -> list[int]:
    boundaries = [boundary for boundary in THESIS_SEPARATOR_BOUNDARIES if boundary < len(labels)]
    target_start = next(
        (idx for idx, label in enumerate(labels) if label in THESIS_TARGET_LABELS), None
    )
    if target_start is not None and target_start not in boundaries:
        boundaries.append(target_start)
    return sorted(boundaries)


def summarize_predictor_predictor_correlations(
    corr_df: pd.DataFrame,
    *,
    top_n: int = 10,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    labels = corr_df.index.tolist()
    for row_idx, family_a in enumerate(labels):
        for col_idx in range(row_idx + 1, len(labels)):
            family_b = labels[col_idx]
            pearson_r = float(corr_df.iat[row_idx, col_idx])
            rows.append(
                {
                    "family_a": family_a,
                    "family_b": family_b,
                    "pearson_r": pearson_r,
                    "abs_pearson_r": abs(pearson_r),
                }
            )
    summary_df = (
        pd.DataFrame(rows)
        .sort_values(["abs_pearson_r", "family_a", "family_b"], ascending=[False, True, True])
        .head(top_n)
        .reset_index(drop=True)
    )
    summary_df.insert(0, "rank", np.arange(1, len(summary_df) + 1))
    return summary_df


def summarize_predictor_target_correlations(corr_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for target_label in corr_df.columns.tolist():
        target_series = corr_df[target_label]
        best_family = str(target_series.abs().idxmax())
        pearson_r = float(target_series.loc[best_family])
        rows.append(
            {
                "target": target_label,
                "predictor_family": best_family,
                "pearson_r": pearson_r,
                "abs_pearson_r": abs(pearson_r),
            }
        )
    return pd.DataFrame(rows).sort_values("target").reset_index(drop=True)


def build_combined_summary_table(
    predictor_predictor_summary: pd.DataFrame,
    predictor_target_summary: pd.DataFrame,
) -> pd.DataFrame:
    predictor_predictor_rows = predictor_predictor_summary.assign(
        section="predictor_predictor",
        item_a=predictor_predictor_summary["family_a"],
        item_b=predictor_predictor_summary["family_b"],
    )[["section", "item_a", "item_b", "pearson_r", "abs_pearson_r"]]
    predictor_target_rows = predictor_target_summary.assign(
        section="predictor_target",
        item_a=predictor_target_summary["predictor_family"],
        item_b=predictor_target_summary["target"],
    )[["section", "item_a", "item_b", "pearson_r", "abs_pearson_r"]]
    return pd.concat([predictor_predictor_rows, predictor_target_rows], ignore_index=True)


def strongest_member_correlation_between_families(
    detailed_corr_df: pd.DataFrame,
    member_lookup: dict[str, list[str]],
    family_a: str,
    family_b: str,
) -> dict[str, Any]:
    subset = detailed_corr_df.loc[member_lookup[family_a], member_lookup[family_b]]
    stacked = subset.stack(future_stack=True).reset_index()
    stacked.columns = ["feature_a", "feature_b", "pearson_r"]
    stacked = stacked.dropna(subset=["pearson_r"])
    stacked["abs_pearson_r"] = stacked["pearson_r"].abs()
    best_row = stacked.sort_values(
        ["abs_pearson_r", "feature_a", "feature_b"], ascending=[False, True, True]
    ).iloc[0]
    return {
        "family_a": family_a,
        "family_b": family_b,
        "feature_a": str(best_row["feature_a"]),
        "feature_b": str(best_row["feature_b"]),
        "member_pearson_r": float(best_row["pearson_r"]),
        "member_abs_pearson_r": float(best_row["abs_pearson_r"]),
    }


def strongest_member_correlation_family_to_target(
    detailed_corr_df: pd.DataFrame,
    member_lookup: dict[str, list[str]],
    family_label: str,
    target_label: str,
) -> dict[str, Any]:
    subset = detailed_corr_df.loc[member_lookup[family_label], [target_label]]
    stacked = subset.stack(future_stack=True).reset_index()
    stacked.columns = ["feature", "target", "pearson_r"]
    stacked = stacked.dropna(subset=["pearson_r"])
    stacked["abs_pearson_r"] = stacked["pearson_r"].abs()
    best_row = stacked.sort_values(["abs_pearson_r", "feature"], ascending=[False, True]).iloc[0]
    return {
        "predictor_family": family_label,
        "target": target_label,
        "feature": str(best_row["feature"]),
        "member_pearson_r": float(best_row["pearson_r"]),
        "member_abs_pearson_r": float(best_row["abs_pearson_r"]),
    }


def build_predictor_predictor_member_diagnostics(
    predictor_predictor_summary: pd.DataFrame,
    detailed_corr_df: pd.DataFrame,
    feature_metadata: pd.DataFrame,
) -> pd.DataFrame:
    member_lookup = build_family_member_name_lookup(feature_metadata)
    rows: list[dict[str, Any]] = []
    for _, row in predictor_predictor_summary.iterrows():
        diagnostic_row = strongest_member_correlation_between_families(
            detailed_corr_df,
            member_lookup,
            family_a=str(row["family_a"]),
            family_b=str(row["family_b"]),
        )
        diagnostic_row["composite_family_pearson_r"] = float(row["pearson_r"])
        rows.append(diagnostic_row)
    return pd.DataFrame(rows)


def build_predictor_target_member_diagnostics(
    predictor_target_summary: pd.DataFrame,
    detailed_corr_df: pd.DataFrame,
    feature_metadata: pd.DataFrame,
) -> pd.DataFrame:
    member_lookup = build_family_member_name_lookup(feature_metadata)
    rows: list[dict[str, Any]] = []
    for _, row in predictor_target_summary.iterrows():
        diagnostic_row = strongest_member_correlation_family_to_target(
            detailed_corr_df,
            member_lookup,
            family_label=str(row["predictor_family"]),
            target_label=str(row["target"]),
        )
        diagnostic_row["composite_family_pearson_r"] = float(row["pearson_r"])
        rows.append(diagnostic_row)
    return pd.DataFrame(rows)


def build_validation_checks(
    full_family_corr_df: pd.DataFrame,
    predictor_corr_df: pd.DataFrame,
    predictor_target_corr_df: pd.DataFrame,
    *,
    predictor_labels: Sequence[str],
    target_labels: Sequence[str],
) -> pd.DataFrame:
    checks = [
        {
            "check": "predictor_family_shape",
            "passed": predictor_corr_df.shape == (len(predictor_labels), len(predictor_labels)),
            "detail": str(predictor_corr_df.shape),
        },
        {
            "check": "predictor_target_shape",
            "passed": predictor_target_corr_df.shape == (len(predictor_labels), len(target_labels)),
            "detail": str(predictor_target_corr_df.shape),
        },
        {
            "check": "full_family_shape",
            "passed": full_family_corr_df.shape
            == (
                len(predictor_labels) + len(target_labels),
                len(predictor_labels) + len(target_labels),
            ),
            "detail": str(full_family_corr_df.shape),
        },
        {
            "check": "predictor_slice_matches_primary_family_matrix",
            "passed": predictor_corr_df.equals(
                full_family_corr_df.loc[list(predictor_labels), list(predictor_labels)]
            ),
            "detail": "predictor block matches direct slice of primary family matrix",
        },
        {
            "check": "predictor_target_slice_matches_primary_family_matrix",
            "passed": predictor_target_corr_df.equals(
                full_family_corr_df.loc[list(predictor_labels), list(target_labels)]
            ),
            "detail": "predictor-target block matches direct slice of primary family matrix",
        },
    ]
    return pd.DataFrame(checks)


def export_dataframe(
    df: pd.DataFrame,
    output_dir: Path,
    filename: str,
    *,
    float_format: str | None = THESIS_TABLE_FLOAT_FORMAT,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / filename
    df.to_csv(
        out_path,
        index=True if filename.endswith("_matrix.csv") else False,
        float_format=float_format,
    )
    return out_path


def save_figure_bundle(
    fig: plt.Figure,
    output_dir: Path,
    stem: str,
    *,
    dpi: int = THESIS_FIGURE_DPI,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    return png_path, pdf_path


def build_manifest_frame(entries: Sequence[ThesisOutputManifestEntry]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"kind": entry.kind, "name": entry.name, "path": str(entry.path)} for entry in entries]
    )


def export_correlation_outputs(
    *,
    corr_df: pd.DataFrame,
    output_dir: Path,
    filename: str,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / filename
    corr_df.to_csv(out_path, index=True)
    return out_path


def export_feature_legend(feature_metadata: pd.DataFrame, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    legend = feature_metadata.sort_values(
        ["group_order", "feature_order", "display_name"]
    ).reset_index(drop=True)
    out_path = output_dir / "correlation_feature_legend.csv"
    legend.to_csv(out_path, index=False)
    return out_path


def plot_detailed_correlation(
    corr_df: pd.DataFrame,
    feature_metadata: pd.DataFrame,
    *,
    title: str,
) -> plt.Figure:
    sns.set_theme(style="white")
    n_features = corr_df.shape[0]
    fig_w = max(16, n_features * 0.23)
    fig_h = max(14, n_features * 0.23)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    mask = np.triu(np.ones_like(corr_df, dtype=bool), k=1) | corr_df.isna().to_numpy()
    sns.heatmap(
        corr_df,
        mask=mask,
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        linewidths=0.15,
        linecolor="white",
        cbar_kws={"label": "Pearson correlation", "shrink": 0.8},
        square=False,
        ax=ax,
    )

    for boundary in group_boundaries(feature_metadata):
        ax.axhline(boundary, color="black", linewidth=0.9)
        ax.axvline(boundary, color="black", linewidth=0.9)

    ax.set_title(title)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=60, ha="right", fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)
    plt.tight_layout()
    return fig


def plot_family_correlation(
    corr_df: pd.DataFrame,
    *,
    title: str,
) -> plt.Figure:
    sns.set_theme(style="white")
    fig, ax = plt.subplots(figsize=(12, 9))
    mask = corr_df.isna().to_numpy()
    sns.heatmap(
        corr_df,
        mask=mask,
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 9},
        linewidths=0.2,
        linecolor="white",
        cbar_kws={"label": "Pearson correlation", "shrink": 0.85},
        square=True,
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    plt.tight_layout()
    return fig


def group_boundaries(feature_metadata: pd.DataFrame) -> list[int]:
    ordered_groups = feature_metadata.sort_values(["group_order", "feature_order", "display_name"])[
        "group_name"
    ].tolist()
    boundaries: list[int] = []
    current = ordered_groups[0] if ordered_groups else None
    for idx, group_name in enumerate(ordered_groups):
        if idx == 0:
            continue
        if group_name != current:
            boundaries.append(idx)
            current = group_name
    return boundaries


def run_thesis_correlation_workflow(
    *,
    config_path: str | Path,
    output_dir: str | Path,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    repo_root = resolve_repo_root() if repo_root is None else Path(repo_root).resolve()
    prepared = prepare_correlation_inputs(config_path, repo_root)
    output_root = resolve_path(output_dir, repo_root)
    figures_dir = output_root / "figures"
    tables_dir = output_root / "tables"

    all_detailed = compute_detailed_correlation(prepared, site_subset="all")
    all_family = compute_family_correlation(prepared, site_subset="all")
    ordered_metadata = get_ordered_feature_metadata(prepared.feature_metadata)
    feature_legend = ordered_metadata.copy()

    predictor_labels = extract_predictor_family_labels(ordered_metadata)
    target_labels = extract_target_labels(ordered_metadata)
    predictor_corr_df, predictor_target_corr_df = slice_family_correlation_matrices(
        all_family["corr_df"],
        predictor_labels=predictor_labels,
        target_labels=target_labels,
    )

    predictor_predictor_summary = summarize_predictor_predictor_correlations(
        predictor_corr_df, top_n=10
    )
    predictor_target_summary = summarize_predictor_target_correlations(predictor_target_corr_df)
    combined_summary = build_combined_summary_table(
        predictor_predictor_summary, predictor_target_summary
    )
    predictor_predictor_member_diagnostics = build_predictor_predictor_member_diagnostics(
        predictor_predictor_summary,
        all_detailed["corr_df"],
        ordered_metadata,
    )
    predictor_target_member_diagnostics = build_predictor_target_member_diagnostics(
        predictor_target_summary,
        all_detailed["corr_df"],
        ordered_metadata,
    )
    validation_checks = build_validation_checks(
        all_family["corr_df"],
        predictor_corr_df,
        predictor_target_corr_df,
        predictor_labels=predictor_labels,
        target_labels=target_labels,
    )

    manifest_entries: list[ThesisOutputManifestEntry] = []

    detailed_all_path = export_correlation_outputs(
        corr_df=all_detailed["corr_df"],
        output_dir=tables_dir,
        filename="correlation_detailed_all_sites.csv",
    )
    manifest_entries.append(
        ThesisOutputManifestEntry("table", "detailed_feature_correlation_matrix", detailed_all_path)
    )

    family_all_path = export_correlation_outputs(
        corr_df=all_family["corr_df"],
        output_dir=tables_dir,
        filename="correlation_family_all_sites.csv",
    )
    manifest_entries.append(
        ThesisOutputManifestEntry("table", "full_family_target_correlation_matrix", family_all_path)
    )

    legend_path = export_feature_legend(feature_legend, tables_dir)
    manifest_entries.append(ThesisOutputManifestEntry("table", "feature_legend", legend_path))

    predictor_family_matrix_path = export_correlation_outputs(
        corr_df=predictor_corr_df,
        output_dir=tables_dir,
        filename="predictor_family_correlation_matrix.csv",
    )
    manifest_entries.append(
        ThesisOutputManifestEntry(
            "table", "predictor_family_correlation_matrix", predictor_family_matrix_path
        )
    )

    predictor_target_matrix_path = export_correlation_outputs(
        corr_df=predictor_target_corr_df,
        output_dir=tables_dir,
        filename="predictor_target_correlation_matrix.csv",
    )
    manifest_entries.append(
        ThesisOutputManifestEntry(
            "table", "predictor_target_correlation_matrix", predictor_target_matrix_path
        )
    )

    predictor_predictor_summary_path = export_dataframe(
        predictor_predictor_summary,
        tables_dir,
        "top_predictor_predictor_correlations.csv",
    )
    manifest_entries.append(
        ThesisOutputManifestEntry(
            "table", "top_predictor_predictor_correlations", predictor_predictor_summary_path
        )
    )

    predictor_target_summary_path = export_dataframe(
        predictor_target_summary,
        tables_dir,
        "strongest_predictor_target_correlations.csv",
    )
    manifest_entries.append(
        ThesisOutputManifestEntry(
            "table", "strongest_predictor_target_correlations", predictor_target_summary_path
        )
    )

    combined_summary_path = export_dataframe(
        combined_summary,
        tables_dir,
        "correlation_summary_table_combined.csv",
    )
    manifest_entries.append(
        ThesisOutputManifestEntry(
            "table", "correlation_summary_table_combined", combined_summary_path
        )
    )

    predictor_predictor_member_diag_path = export_dataframe(
        predictor_predictor_member_diagnostics,
        tables_dir,
        "predictor_predictor_member_diagnostics.csv",
    )
    manifest_entries.append(
        ThesisOutputManifestEntry(
            "table", "predictor_predictor_member_diagnostics", predictor_predictor_member_diag_path
        )
    )

    predictor_target_member_diag_path = export_dataframe(
        predictor_target_member_diagnostics,
        tables_dir,
        "predictor_target_member_diagnostics.csv",
    )
    manifest_entries.append(
        ThesisOutputManifestEntry(
            "table", "predictor_target_member_diagnostics", predictor_target_member_diag_path
        )
    )

    validation_checks_path = export_dataframe(
        validation_checks,
        tables_dir,
        "correlation_validation_checks.csv",
        float_format=None,
    )
    manifest_entries.append(
        ThesisOutputManifestEntry("table", "correlation_validation_checks", validation_checks_path)
    )

    predictor_family_fig = plot_predictor_family_heatmap(predictor_corr_df)
    predictor_family_png, predictor_family_pdf = save_figure_bundle(
        predictor_family_fig,
        figures_dir,
        "predictor_family_correlation_heatmap",
    )
    plt.close(predictor_family_fig)
    manifest_entries.append(
        ThesisOutputManifestEntry(
            "figure", "predictor_family_correlation_heatmap_png", predictor_family_png
        )
    )
    manifest_entries.append(
        ThesisOutputManifestEntry(
            "figure", "predictor_family_correlation_heatmap_pdf", predictor_family_pdf
        )
    )

    predictor_target_fig = plot_predictor_target_heatmap(predictor_target_corr_df)
    predictor_target_png, predictor_target_pdf = save_figure_bundle(
        predictor_target_fig,
        figures_dir,
        "predictor_target_correlation_heatmap",
    )
    plt.close(predictor_target_fig)
    manifest_entries.append(
        ThesisOutputManifestEntry(
            "figure", "predictor_target_correlation_heatmap_png", predictor_target_png
        )
    )
    manifest_entries.append(
        ThesisOutputManifestEntry(
            "figure", "predictor_target_correlation_heatmap_pdf", predictor_target_pdf
        )
    )

    appendix_full_fig = plot_appendix_full_family_heatmap(all_family["corr_df"])
    appendix_full_png, appendix_full_pdf = save_figure_bundle(
        appendix_full_fig,
        figures_dir,
        "appendix_full_family_target_correlation_matrix",
    )
    plt.close(appendix_full_fig)
    manifest_entries.append(
        ThesisOutputManifestEntry(
            "figure", "appendix_full_family_target_correlation_matrix_png", appendix_full_png
        )
    )
    manifest_entries.append(
        ThesisOutputManifestEntry(
            "figure", "appendix_full_family_target_correlation_matrix_pdf", appendix_full_pdf
        )
    )

    manifest_df = build_manifest_frame(manifest_entries)
    manifest_path = export_dataframe(
        manifest_df,
        tables_dir,
        "correlation_output_manifest.csv",
        float_format=None,
    )
    manifest_entries.append(
        ThesisOutputManifestEntry("table", "correlation_output_manifest", manifest_path)
    )
    manifest_df = build_manifest_frame(manifest_entries)
    export_dataframe(
        manifest_df,
        tables_dir,
        "correlation_output_manifest.csv",
        float_format=None,
    )

    return {
        "prepared": prepared,
        "feature_legend": feature_legend,
        "method_description": get_family_correlation_method_description(),
        "full_detailed_corr_df": all_detailed["corr_df"],
        "full_family_corr_df": all_family["corr_df"],
        "predictor_family_corr_df": predictor_corr_df,
        "predictor_target_corr_df": predictor_target_corr_df,
        "predictor_predictor_summary": predictor_predictor_summary,
        "predictor_target_summary": predictor_target_summary,
        "combined_summary": combined_summary,
        "predictor_predictor_member_diagnostics": predictor_predictor_member_diagnostics,
        "predictor_target_member_diagnostics": predictor_target_member_diagnostics,
        "validation_checks": validation_checks,
        "predictor_labels": predictor_labels,
        "target_labels": target_labels,
        "output_root": output_root,
        "figures_dir": figures_dir,
        "tables_dir": tables_dir,
        "manifest": manifest_df,
    }


def run_correlation_workflow(
    *,
    config_path: str | Path,
    output_dir: str | Path,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    repo_root = resolve_repo_root() if repo_root is None else Path(repo_root).resolve()
    prepared = prepare_correlation_inputs(config_path, repo_root)
    output_dir = resolve_path(output_dir, repo_root)

    all_detailed = compute_detailed_correlation(prepared, site_subset="all")
    all_family = compute_family_correlation(prepared, site_subset="all")
    test_detailed = compute_detailed_correlation(prepared, site_subset="test")
    test_family = compute_family_correlation(prepared, site_subset="test")

    detailed_all_path = export_correlation_outputs(
        corr_df=all_detailed["corr_df"],
        output_dir=output_dir,
        filename="correlation_detailed_all_sites.csv",
    )
    family_all_path = export_correlation_outputs(
        corr_df=all_family["corr_df"],
        output_dir=output_dir,
        filename="correlation_family_all_sites.csv",
    )
    detailed_test_path = export_correlation_outputs(
        corr_df=test_detailed["corr_df"],
        output_dir=output_dir,
        filename="correlation_detailed_test_sites.csv",
    )
    family_test_path = export_correlation_outputs(
        corr_df=test_family["corr_df"],
        output_dir=output_dir,
        filename="correlation_family_test_sites.csv",
    )
    legend_path = export_feature_legend(prepared.feature_metadata, output_dir)

    return {
        "prepared": prepared,
        "all_detailed": all_detailed,
        "all_family": all_family,
        "test_detailed": test_detailed,
        "test_family": test_family,
        "exports": {
            "detailed_all": detailed_all_path,
            "family_all": family_all_path,
            "detailed_test": detailed_test_path,
            "family_test": family_test_path,
            "legend": legend_path,
        },
    }
