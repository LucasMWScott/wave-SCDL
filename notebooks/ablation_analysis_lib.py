"""Ablation analysis lib."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


METRIC_COLUMNS = {
    "val_loss": "Validation Loss",
    "val_hs_rmse": "Hs RMSE",
    "val_tp_rmse": "Tp RMSE",
    "val_dir_rmse_deg": "Dir RMSE (deg)",
    "val_dp_rmse_deg": "Dp RMSE (deg)",
}

METRIC_UNITS = {
    "val_loss": "",
    "val_hs_rmse": "m",
    "val_tp_rmse": "s",
    "val_dir_rmse_deg": "deg",
    "val_dp_rmse_deg": "deg",
}

METRIC_AXIS_LABELS = {
    "val_loss": "Validation loss delta",
    "val_hs_rmse": "Hs RMSE delta (m)",
    "val_tp_rmse": "Tp RMSE delta (s)",
    "val_dir_rmse_deg": "Dir RMSE delta (deg)",
    "val_dp_rmse_deg": "Dp RMSE delta (deg)",
}

TARGET_PANEL_GROUP_ORDER = [
    "local_shoreline_context",
    "fetch_anisotropy",
    "path_ratios",
    "directional_fetch",
    "fjordness",
    "routing_core",
    "directional_bathymetry",
]

TARGET_METRIC_SPECS = [
    ("val_hs_rmse", "Hs RMSE", "#98c1a9", "paired_target_metric_delta_hs_rmse"),
    ("val_tp_rmse", "Tp RMSE", "#d9c89e", "paired_target_metric_delta_tp_rmse"),
    ("val_dir_rmse_deg", "Dir RMSE", "#d7a5a5", "paired_target_metric_delta_dir_rmse"),
    ("val_dp_rmse_deg", "Dp RMSE", "#b4a7d6", "paired_target_metric_delta_dp_rmse"),
]


def find_repo_root(start: str | Path | None = None) -> Path:
    candidate = Path(start or Path.cwd()).resolve()
    for path in [candidate, *candidate.parents]:
        if (path / "src").exists() and (path / "notebooks").exists():
            return path
    return candidate


REPO_ROOT = find_repo_root()


def resolve_path(path_like: str | Path | None, *, anchor: str | Path | None = None) -> Path | None:
    if path_like in (None, ""):
        return None
    path = Path(path_like).expanduser()
    if path.is_absolute():
        return path.resolve()
    candidates: list[Path] = []
    if anchor is not None:
        anchor_path = Path(anchor).expanduser().resolve()
        anchor_dir = anchor_path if anchor_path.is_dir() else anchor_path.parent
        candidates.append((anchor_dir / path).resolve())
    candidates.append((Path.cwd() / path).resolve())
    candidates.append((REPO_ROOT / path).resolve())
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def extract_run_number(name: str) -> int:
    match = re.match(r"^(\d+)_", str(name))
    return int(match.group(1)) if match else 10**9


def strip_seed_suffix(value: str) -> str:
    return re.sub(r"_seed\d+$", "", str(value))


def parse_seed_from_name(name: str) -> int | None:
    match = re.search(r"_seed(\d+)$", str(name))
    return int(match.group(1)) if match else None


def normalize_descriptor(name: str) -> str:
    base = strip_seed_suffix(str(name))
    base = re.sub(r"^\d+_", "", base)
    base = re.sub(r"^ablation_", "", base)
    if base.startswith("drop_"):
        base = base[len("drop_") :]
    return base


def descriptor_from_name(name: str) -> str:
    return normalize_descriptor(name)


def canonical_run_name(name: str) -> str:
    return strip_seed_suffix(str(name))


def humanize_descriptor(name: str, mode: str) -> str:
    descriptor = descriptor_from_name(name)
    if mode == "sites" and descriptor.startswith("sites_"):
        frac = descriptor.replace("sites_", "")
        frac = frac.replace("pct", "%")
        frac = frac.replace("_samples10pct", " sites / 10% samples")
        return f"{frac} train sites"
    if descriptor == "full_static":
        return "full static"
    return descriptor.replace("_", " ")


def parse_site_fraction_from_name(name: str) -> float | None:
    match = re.search(r"sites_(\d+)pct", str(name))
    if not match:
        return None
    return float(match.group(1)) / 100.0


def relative_to_root(path: Path, root: Path | None) -> str:
    if root is None:
        return str(path)
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except Exception:
        return str(path)


def discover_run_dirs(results_root: str | Path) -> list[Path]:
    root = resolve_path(results_root)
    if root is None or not root.exists():
        return []

    unique: dict[str, Path] = {}
    if root.is_dir() and (root / "train_history.json").exists():
        unique[str(root.resolve())] = root.resolve()

    for history_path in root.rglob("train_history.json"):
        run_dir = history_path.parent.resolve()
        unique[str(run_dir)] = run_dir

    return sorted(
        unique.values(),
        key=lambda path: (
            extract_run_number(path.name),
            descriptor_from_name(path.name),
            relative_to_root(path, root),
        ),
    )


def infer_selection_direction(training_meta: dict) -> str:
    config = (training_meta.get("config", {}) or {}) if isinstance(training_meta, dict) else {}
    selection_cfg = (
        ((config.get("training", {}) or {}).get("selection", {}) or {})
        if isinstance(config, dict)
        else {}
    )
    mode = str(selection_cfg.get("mode", "min")).strip().lower()
    return mode if mode in {"min", "max"} else "min"


def choose_best_history_row(history_rows: list[dict], *, selection_direction: str = "min") -> dict:
    valid_rows = [row for row in history_rows if row.get("validation_ran", True) is not False]
    if not valid_rows:
        valid_rows = list(history_rows)
    if not valid_rows:
        raise ValueError("train_history.json did not contain any rows")

    def row_score(row: dict) -> float:
        raw = row.get("selection_score", row.get("val_loss", float("inf")))
        try:
            value = float(raw)
        except Exception:
            value = float("inf")
        if not math.isfinite(value):
            return float("inf")
        return value

    if selection_direction == "max":
        finite = [row for row in valid_rows if math.isfinite(row_score(row))]
        return max(finite or valid_rows, key=row_score)
    finite = [row for row in valid_rows if math.isfinite(row_score(row))]
    return min(finite or valid_rows, key=row_score)


def first_present(values: list[Any]) -> Any:
    for value in values:
        if value is None:
            continue
        if isinstance(value, float) and math.isnan(value):
            continue
        return value
    return None


def has_repeated_points(df: pd.DataFrame) -> bool:
    if df.empty or "n_runs" not in df.columns:
        return False
    counts = pd.to_numeric(df["n_runs"], errors="coerce").fillna(1)
    return bool((counts > 1).any())


def _infer_dropped_group(case_name: str, mode: str) -> str | None:
    descriptor = descriptor_from_name(case_name)
    if mode == "sites":
        return None
    if descriptor == "full_static":
        return "full_static"
    return descriptor


def _should_attach_physical_group_baselines(
    resolved_root: Path, rows: list[dict[str, Any]]
) -> bool:
    if "features_physical_groups" in str(resolved_root):
        return True
    if not rows:
        return False
    feature_rows = [row for row in rows if str(row.get("ablation_mode", "")).lower() == "features"]
    if not feature_rows:
        return False
    dropped_groups = {str(row.get("dropped_group") or "").strip() for row in feature_rows}
    if "full_static" in dropped_groups:
        return False
    return any(str(row.get("run_name") or "").startswith("drop_") for row in feature_rows)


def _load_reused_full_static_registry_rows(
    mode: str,
    resolved_root: Path,
    rows: list[dict[str, Any]],
    *,
    baseline_registry_path: str | Path | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    if mode != "features":
        return [], []
    if not _should_attach_physical_group_baselines(resolved_root, rows):
        return [], []

    if baseline_registry_path in (None, ""):
        registry_path = (
            REPO_ROOT
            / "configs"
            / "ablations"
            / "features"
            / "physical_groups"
            / "full_static"
            / "baseline_registry.yaml"
        )
    else:
        registry_path = resolve_path(baseline_registry_path, anchor=resolved_root)
        if registry_path is None:
            return [], [f"Could not resolve baseline registry path: {baseline_registry_path}"]
    if not registry_path.exists():
        return [], [f"Missing baseline registry: {registry_path}"]

    registry = read_yaml(registry_path)
    rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    for entry in registry.get("baselines", []) or []:
        result_dir = entry.get("results_dir")
        if not result_dir:
            continue
        try:
            row = load_run_summary(result_dir, mode=mode, root=resolved_root)
            row["baseline_registry_entry"] = True
            rows.append(row)
        except Exception as exc:
            warnings.append(f"Could not load reused full-static baseline {result_dir}: {exc}")
    return rows, warnings


def load_run_summary(
    run_dir: str | Path, mode: str, *, root: str | Path | None = None
) -> dict[str, Any]:
    run_dir = resolve_path(run_dir)
    resolved_root = resolve_path(root) if root is not None else None
    if run_dir is None or not run_dir.exists():
        raise FileNotFoundError(f"Missing run directory: {run_dir}")

    history_path = run_dir / "train_history.json"
    if not history_path.exists():
        raise FileNotFoundError(f"Missing train_history.json under {run_dir}")
    history_payload = read_json(history_path)
    if not isinstance(history_payload, list):
        raise ValueError(f"Expected train_history.json to be a list under {run_dir}")

    metadata_path = run_dir / "training_run_metadata.json"
    training_meta = read_json(metadata_path) if metadata_path.exists() else {}
    best_row = choose_best_history_row(
        history_payload, selection_direction=infer_selection_direction(training_meta)
    )

    config = (training_meta.get("config", {}) or {}) if isinstance(training_meta, dict) else {}
    data_cfg = (config.get("data", {}) or {}) if isinstance(config, dict) else {}
    training_cfg = (config.get("training", {}) or {}) if isinstance(config, dict) else {}
    experiment_cfg = (config.get("experiment", {}) or {}) if isinstance(config, dict) else {}
    splits_cfg = (training_meta.get("splits", {}) or {}) if isinstance(training_meta, dict) else {}
    train_site_subsampling_cfg = (
        (data_cfg.get("train_site_subsampling", {}) or {}) if isinstance(data_cfg, dict) else {}
    )

    site_fraction = None
    if isinstance(train_site_subsampling_cfg, dict) and train_site_subsampling_cfg.get("enabled"):
        try:
            site_fraction = float(train_site_subsampling_cfg.get("fraction"))
        except Exception:
            site_fraction = None
    if site_fraction is None:
        site_fraction = parse_site_fraction_from_name(run_dir.name)

    static_after = best_row.get("static_feature_count_after_ablation")
    static_before = best_row.get("static_feature_count_before_ablation")
    static_drop = best_row.get("static_feature_drop_count")
    try:
        static_after = None if static_after is None else int(static_after)
    except Exception:
        static_after = None
    try:
        static_before = None if static_before is None else int(static_before)
    except Exception:
        static_before = None
    try:
        static_drop = None if static_drop is None else int(static_drop)
    except Exception:
        static_drop = None

    train_sites = (
        list((splits_cfg.get("train_sites", []) or [])) if isinstance(splits_cfg, dict) else []
    )

    training_seed = training_cfg.get("seed")
    if training_seed is None:
        training_seed = parse_seed_from_name(str(experiment_cfg.get("case_name") or run_dir.name))
    try:
        training_seed = None if training_seed is None else int(training_seed)
    except Exception:
        training_seed = None

    case_name = str(experiment_cfg.get("case_name") or run_dir.name)
    base_run_name = canonical_run_name(case_name)
    dropped_group = _infer_dropped_group(base_run_name, mode=mode)

    return {
        "run_dir": str(run_dir),
        "run_relative_dir": relative_to_root(run_dir, resolved_root),
        "run_name": run_dir.name,
        "base_run_name": base_run_name,
        "run_number": extract_run_number(base_run_name),
        "descriptor": descriptor_from_name(base_run_name),
        "label": humanize_descriptor(base_run_name, mode=mode),
        "ablation_mode": mode,
        "training_seed": training_seed,
        "best_epoch": best_row.get("epoch"),
        "selection_mode": best_row.get("selection_mode"),
        "selection_score": best_row.get("selection_score"),
        "val_loss": best_row.get("val_loss"),
        "val_hs_rmse": best_row.get("val_hs_rmse"),
        "val_tp_rmse": best_row.get("val_tp_rmse"),
        "val_dir_rmse_deg": best_row.get("val_dir_rmse_deg"),
        "val_dp_rmse_deg": best_row.get("val_dp_rmse_deg"),
        "static_feature_count_before": static_before,
        "static_feature_count_after": static_after,
        "static_feature_drop_count": static_drop,
        "train_site_fraction": site_fraction,
        "train_site_percent": None if site_fraction is None else 100.0 * float(site_fraction),
        "train_site_count": len(train_sites),
        "is_zero_static_reference": False,
        "is_full_static": bool(dropped_group == "full_static"),
        "dropped_group": dropped_group,
        "baseline_registry_entry": False,
    }


def apply_plot_ordering(df: pd.DataFrame, *, mode: str) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    ordered = df.copy()
    if mode == "features":
        ordered["ordering_key"] = pd.to_numeric(
            ordered["static_feature_count_after"], errors="coerce"
        ).fillna(-1)
        ordered = ordered.sort_values(
            ["ordering_key", "run_number", "label"], ascending=[True, True, True]
        ).reset_index(drop=True)
        ordered["x_value"] = np.arange(len(ordered), dtype=float)
        ordered["x_tick_label"] = [
            (
                "no static"
                if bool(row.get("is_zero_static_reference", False))
                else (
                    f"{int(row['static_feature_count_after'])}\n{row['label']}"
                    if pd.notna(row.get("static_feature_count_after"))
                    else row["label"]
                )
            )
            for _, row in ordered.iterrows()
        ]
    else:
        ordered["ordering_key"] = pd.to_numeric(ordered["train_site_percent"], errors="coerce")
        ordered = ordered.sort_values(
            ["ordering_key", "run_number", "label"], ascending=[True, True, True]
        ).reset_index(drop=True)
        ordered["x_value"] = ordered["train_site_percent"].astype(float)
        ordered["x_tick_label"] = [f"{int(round(v))}%" for v in ordered["x_value"]]
    return ordered


def aggregate_ablation_rows(
    raw_df: pd.DataFrame, *, mode: str, average_repeated_runs: bool
) -> pd.DataFrame:
    if raw_df.empty:
        return raw_df.copy()

    numeric_metrics = list(METRIC_COLUMNS)
    raw = raw_df.copy()
    raw["n_runs"] = 1
    raw["training_seeds"] = raw["training_seed"].apply(
        lambda value: "" if pd.isna(value) else str(int(value))
    )
    raw["source_run_names"] = raw["run_name"]
    raw["source_run_dirs"] = raw["run_relative_dir"]
    for metric in numeric_metrics:
        raw[f"{metric}_std"] = 0.0

    if not average_repeated_runs:
        return apply_plot_ordering(raw, mode=mode)

    grouped_rows: list[dict[str, Any]] = []
    sort_seed = pd.to_numeric(raw["training_seed"], errors="coerce")
    raw = (
        raw.assign(_sort_seed=sort_seed.fillna(-1))
        .sort_values(["run_number", "_sort_seed", "run_name"])
        .reset_index(drop=True)
    )

    for _, group in raw.groupby("base_run_name", sort=False):
        first = group.iloc[0]
        row = {column: first[column] for column in group.columns if column != "_sort_seed"}
        row["run_name"] = first["base_run_name"]
        row["label"] = first["label"]
        row["n_runs"] = int(len(group))
        seed_values = []
        for value in pd.to_numeric(group["training_seed"], errors="coerce").dropna().tolist():
            seed_int = int(value)
            if seed_int not in seed_values:
                seed_values.append(seed_int)
        row["training_seed"] = first_present(seed_values)
        row["training_seeds"] = ", ".join(str(seed) for seed in seed_values)
        row["source_run_names"] = "; ".join(group["run_name"].astype(str).tolist())
        row["source_run_dirs"] = "; ".join(group["run_relative_dir"].astype(str).tolist())
        row["run_relative_dir"] = row["source_run_dirs"]
        row["best_epoch"] = ", ".join(
            str(int(value))
            for value in pd.to_numeric(group["best_epoch"], errors="coerce").dropna().tolist()
        )
        row["selection_score"] = pd.to_numeric(group["selection_score"], errors="coerce").mean()
        row["is_zero_static_reference"] = bool(
            group["is_zero_static_reference"].fillna(False).any()
        )
        row["is_full_static"] = bool(group["is_full_static"].fillna(False).any())

        for column in [
            "run_dir",
            "descriptor",
            "ablation_mode",
            "selection_mode",
            "base_run_name",
            "run_number",
            "static_feature_count_before",
            "static_feature_count_after",
            "static_feature_drop_count",
            "train_site_fraction",
            "train_site_percent",
            "train_site_count",
            "dropped_group",
        ]:
            row[column] = first_present(group[column].tolist())

        for metric in numeric_metrics:
            values = pd.to_numeric(group[metric], errors="coerce")
            row[metric] = float(values.mean()) if values.notna().any() else np.nan
            row[f"{metric}_std"] = float(values.std(ddof=0)) if values.notna().sum() > 1 else 0.0

        grouped_rows.append(row)

    aggregated = pd.DataFrame(grouped_rows)
    return apply_plot_ordering(aggregated, mode=mode)


def build_ablation_summary(
    results_root: str | Path,
    *,
    mode: str,
    zero_static_reference_results_dir: str | Path | None = None,
    average_repeated_runs: bool = True,
    baseline_registry_path: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, Path | None, list[str]]:
    resolved_root = resolve_path(results_root)
    if resolved_root is None or not resolved_root.exists():
        warnings = [f"Missing ablation results root: {resolved_root}"]
        return pd.DataFrame(), pd.DataFrame(), resolved_root, warnings

    rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    seen_run_dirs: set[str] = set()

    for run_dir in discover_run_dirs(resolved_root):
        try:
            row = load_run_summary(run_dir, mode=mode, root=resolved_root)
            rows.append(row)
            seen_run_dirs.add(str(Path(row["run_dir"]).resolve()))
        except Exception as exc:
            warnings.append(f"Skipping {relative_to_root(run_dir, resolved_root)}: {exc}")

    baseline_rows, baseline_warnings = _load_reused_full_static_registry_rows(
        mode,
        resolved_root,
        rows,
        baseline_registry_path=baseline_registry_path,
    )
    warnings.extend(baseline_warnings)
    for row in baseline_rows:
        run_key = str(Path(row["run_dir"]).resolve())
        if run_key not in seen_run_dirs:
            rows.append(row)
            seen_run_dirs.add(run_key)

    if mode == "features" and zero_static_reference_results_dir not in (None, ""):
        try:
            extra = load_run_summary(
                zero_static_reference_results_dir, mode=mode, root=resolved_root
            )
            extra["is_zero_static_reference"] = True
            extra["label"] = "no static"
            extra["descriptor"] = "no_static"
            extra["static_feature_count_after"] = 0
            extra["base_run_name"] = canonical_run_name(extra["base_run_name"])
            rows.append(extra)
        except Exception as exc:
            warnings.append(f"Could not load ZERO_STATIC_REFERENCE_RESULTS_DIR: {exc}")

    raw_df = pd.DataFrame(rows)
    if raw_df.empty:
        return raw_df, raw_df, resolved_root, warnings

    raw_df = apply_plot_ordering(raw_df, mode=mode)
    summary_df = aggregate_ablation_rows(
        raw_df, mode=mode, average_repeated_runs=average_repeated_runs
    )
    return summary_df, raw_df, resolved_root, warnings


def feature_mode_note(df: pd.DataFrame) -> str:
    if df.empty:
        return "No runs discovered."

    if bool(df.get("is_zero_static_reference", pd.Series(dtype=bool)).any()):
        base_note = (
            "Feature ablation curve includes an optional explicit no-static reference run at the left edge, "
            "followed by available ablations ordered by retained static feature count."
        )
    else:
        base_note = (
            "Feature ablation curve orders the available runs from least retained static information to most retained "
            "static information. Full-static baseline appears at the right edge."
        )

    if has_repeated_points(df):
        max_runs = int(pd.to_numeric(df["n_runs"], errors="coerce").fillna(1).max())
        return f"{base_note} Repeated runs are averaged by ablation key (up to n={max_runs})."
    return base_note


def build_feature_seed_paired_tables(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if raw_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    features_df = raw_df.copy()
    features_df = features_df[
        features_df["ablation_mode"].astype(str).str.lower() == "features"
    ].copy()
    if features_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    baselines = features_df[features_df["is_full_static"].fillna(False)].copy()
    ablations = features_df[~features_df["is_full_static"].fillna(False)].copy()
    if baselines.empty or ablations.empty:
        return pd.DataFrame(), pd.DataFrame()

    baseline_by_seed = baselines.sort_values(["training_seed", "run_name"]).drop_duplicates(
        "training_seed", keep="first"
    )
    baseline_lookup = {
        int(seed): row
        for seed, row in baseline_by_seed.set_index("training_seed").iterrows()
        if pd.notna(seed)
    }

    paired_rows: list[dict[str, Any]] = []
    for _, row in ablations.iterrows():
        seed = row.get("training_seed")
        if pd.isna(seed):
            continue
        seed_int = int(seed)
        baseline = baseline_lookup.get(seed_int)
        if baseline is None:
            continue
        paired_row: dict[str, Any] = {
            "training_seed": seed_int,
            "dropped_group": row.get("dropped_group"),
            "ablated_run_name": row.get("run_name"),
            "full_static_run_name": baseline.get("run_name"),
            "ablated_run_dir": row.get("run_relative_dir"),
            "full_static_run_dir": baseline.get("run_relative_dir"),
            "static_feature_count_after": row.get("static_feature_count_after"),
            "static_feature_drop_count": row.get("static_feature_drop_count"),
        }
        for metric in METRIC_COLUMNS:
            baseline_value = pd.to_numeric(pd.Series([baseline.get(metric)]), errors="coerce").iloc[
                0
            ]
            ablated_value = pd.to_numeric(pd.Series([row.get(metric)]), errors="coerce").iloc[0]
            delta = (
                ablated_value - baseline_value
                if pd.notna(ablated_value) and pd.notna(baseline_value)
                else np.nan
            )
            pct = (
                (100.0 * delta / baseline_value)
                if pd.notna(delta) and pd.notna(baseline_value) and baseline_value != 0
                else np.nan
            )
            paired_row[f"{metric}_full_static"] = baseline_value
            paired_row[f"{metric}_ablated"] = ablated_value
            paired_row[f"{metric}_delta"] = delta
            paired_row[f"{metric}_pct_change"] = pct
        paired_rows.append(paired_row)

    paired_df = pd.DataFrame(paired_rows)
    if paired_df.empty:
        return paired_df, pd.DataFrame()

    paired_df = paired_df.sort_values(["dropped_group", "training_seed"]).reset_index(drop=True)

    aggregate_rows: list[dict[str, Any]] = []
    for dropped_group, group in paired_df.groupby("dropped_group", sort=True):
        summary: dict[str, Any] = {
            "dropped_group": dropped_group,
            "n_seeds": int(group["training_seed"].nunique()),
        }
        for metric in METRIC_COLUMNS:
            ablated_col = pd.to_numeric(group[f"{metric}_ablated"], errors="coerce")
            delta_col = pd.to_numeric(group[f"{metric}_delta"], errors="coerce")
            summary[f"{metric}_mean_ablated"] = (
                float(ablated_col.mean()) if ablated_col.notna().any() else np.nan
            )
            summary[f"{metric}_std_ablated"] = (
                float(ablated_col.std(ddof=0)) if ablated_col.notna().sum() > 1 else 0.0
            )
            summary[f"{metric}_mean_paired_change"] = (
                float(delta_col.mean()) if delta_col.notna().any() else np.nan
            )
            summary[f"{metric}_worsened_seed_count"] = int((delta_col > 0).sum())
        aggregate_rows.append(summary)

    aggregate_df = pd.DataFrame(aggregate_rows).sort_values("dropped_group").reset_index(drop=True)
    return paired_df, aggregate_df


def _title_case_group_name(dropped_group: str) -> str:
    text = str(dropped_group).replace("_", " ").strip()
    if not text:
        return text
    return text[0].upper() + text[1:]


def build_feature_delta_analysis(raw_df: pd.DataFrame) -> dict[str, Any]:
    if raw_df.empty:
        return {
            "paired_df": pd.DataFrame(),
            "group_summary_df": pd.DataFrame(),
            "export_table_df": pd.DataFrame(),
            "group_order": [],
            "missing_pairs": [],
            "unmatched_ablation_runs": [],
            "baseline_seeds": [],
        }

    raw_features_df = raw_df.copy()
    raw_features_df = raw_features_df[
        raw_features_df["ablation_mode"].astype(str).str.lower() == "features"
    ].copy()
    baseline_rows_df = raw_features_df[raw_features_df["is_full_static"].fillna(False)].copy()
    baseline_seeds = sorted(
        int(seed)
        for seed in pd.to_numeric(baseline_rows_df["training_seed"], errors="coerce")
        .dropna()
        .unique()
        .tolist()
    )
    baseline_lookup = {
        int(seed): row
        for seed, row in baseline_rows_df.sort_values(["training_seed", "run_name"])
        .drop_duplicates("training_seed", keep="first")
        .set_index("training_seed")
        .iterrows()
        if pd.notna(seed)
    }

    paired_df, _ = build_feature_seed_paired_tables(raw_df)
    if paired_df.empty:
        missing_pairs = []
        for _, row in raw_features_df[~raw_features_df["is_full_static"].fillna(False)].iterrows():
            seed = pd.to_numeric(pd.Series([row.get("training_seed")]), errors="coerce").iloc[0]
            if pd.notna(seed) and int(seed) not in baseline_lookup:
                continue
        return {
            "paired_df": paired_df,
            "group_summary_df": pd.DataFrame(),
            "export_table_df": pd.DataFrame(),
            "group_order": [],
            "missing_pairs": missing_pairs,
            "unmatched_ablation_runs": [],
            "baseline_seeds": baseline_seeds,
        }

    working = paired_df.copy()
    working["training_seed"] = pd.to_numeric(working["training_seed"], errors="coerce").astype(
        "Int64"
    )
    working["static_feature_drop_count"] = pd.to_numeric(
        working["static_feature_drop_count"], errors="coerce"
    ).astype("Int64")
    grouped = working.groupby("dropped_group", sort=False)

    missing_pairs: list[dict[str, Any]] = []
    unmatched_ablation_runs: list[str] = []
    raw_ablation_df = raw_features_df[~raw_features_df["is_full_static"].fillna(False)].copy()
    for _, row in raw_ablation_df.iterrows():
        seed = pd.to_numeric(pd.Series([row.get("training_seed")]), errors="coerce").iloc[0]
        if pd.isna(seed):
            unmatched_ablation_runs.append(str(row.get("run_name")))
            continue
        if int(seed) not in baseline_lookup:
            unmatched_ablation_runs.append(str(row.get("run_name")))
    summary_rows: list[dict[str, Any]] = []
    export_rows: list[dict[str, Any]] = []

    all_groups = sorted(
        {str(group) for group in raw_ablation_df["dropped_group"].dropna().unique().tolist()}
    )
    drop_count_lookup: dict[str, int] = {}
    for group_name in all_groups:
        matches = raw_ablation_df[raw_ablation_df["dropped_group"].astype(str) == group_name]
        if matches.empty or not matches["static_feature_drop_count"].notna().any():
            drop_count_lookup[group_name] = 0
        else:
            drop_count_lookup[group_name] = int(
                pd.to_numeric(matches["static_feature_drop_count"], errors="coerce")
                .dropna()
                .iloc[0]
            )

    for dropped_group in all_groups:
        group = working[working["dropped_group"].astype(str) == dropped_group].copy()
        present_seeds = sorted(
            int(seed) for seed in group["training_seed"].dropna().unique().tolist()
        )
        missing_seeds = [seed for seed in baseline_seeds if seed not in present_seeds]
        drop_count = drop_count_lookup.get(str(dropped_group), 0)
        group_label = f"{_title_case_group_name(str(dropped_group))} ({drop_count})"

        for seed in missing_seeds:
            missing_pairs.append(
                {
                    "dropped_group": dropped_group,
                    "training_seed": seed,
                    "reason": "missing ablated run for baseline seed",
                }
            )

        summary_row: dict[str, Any] = {
            "dropped_group": dropped_group,
            "group_label": group_label,
            "static_feature_drop_count": drop_count,
            "paired_seed_count": int(len(present_seeds)),
            "paired_seeds": ", ".join(str(seed) for seed in present_seeds),
            "missing_seeds": ", ".join(str(seed) for seed in missing_seeds),
        }
        for metric in METRIC_COLUMNS:
            delta_col = pd.to_numeric(group[f"{metric}_delta"], errors="coerce")
            summary_row[f"{metric}_mean_delta"] = (
                float(delta_col.mean()) if delta_col.notna().any() else np.nan
            )
            summary_row[f"{metric}_std_delta"] = (
                float(delta_col.std(ddof=0)) if delta_col.notna().sum() > 1 else 0.0
            )
            summary_row[f"{metric}_positive_count"] = int((delta_col > 0).sum())
            summary_row[f"{metric}_negative_count"] = int((delta_col < 0).sum())
        summary_rows.append(summary_row)

        for metric in METRIC_COLUMNS:
            row: dict[str, Any] = {
                "dropped_group": dropped_group,
                "group_label": group_label,
                "metric": metric,
                "metric_label": METRIC_COLUMNS[metric],
                "unit": METRIC_UNITS[metric],
                "transformed_feature_count_removed": drop_count,
                "paired_seed_count": int(len(present_seeds)),
                "paired_seeds": ", ".join(str(seed) for seed in present_seeds),
                "missing_seeds": ", ".join(str(seed) for seed in missing_seeds),
            }
            metric_deltas = pd.to_numeric(group[f"{metric}_delta"], errors="coerce")
            row["mean_paired_difference"] = (
                float(metric_deltas.mean()) if metric_deltas.notna().any() else np.nan
            )
            row["std_paired_difference"] = (
                float(metric_deltas.std(ddof=0)) if metric_deltas.notna().sum() > 1 else 0.0
            )
            row["positive_difference_count"] = int((metric_deltas > 0).sum())
            row["negative_difference_count"] = int((metric_deltas < 0).sum())
            for seed in baseline_seeds:
                match = group[group["training_seed"] == seed]
                if match.empty:
                    row[f"seed_{seed}_full_static"] = np.nan
                    row[f"seed_{seed}_ablated"] = np.nan
                    row[f"seed_{seed}_delta"] = np.nan
                    continue
                row[f"seed_{seed}_full_static"] = float(
                    pd.to_numeric(match[f"{metric}_full_static"], errors="coerce").iloc[0]
                )
                row[f"seed_{seed}_ablated"] = float(
                    pd.to_numeric(match[f"{metric}_ablated"], errors="coerce").iloc[0]
                )
                row[f"seed_{seed}_delta"] = float(
                    pd.to_numeric(match[f"{metric}_delta"], errors="coerce").iloc[0]
                )
            export_rows.append(row)

    group_summary_df = pd.DataFrame(summary_rows)
    if group_summary_df.empty:
        return {
            "paired_df": working,
            "group_summary_df": group_summary_df,
            "export_table_df": pd.DataFrame(export_rows),
            "group_order": [],
            "missing_pairs": missing_pairs,
            "unmatched_ablation_runs": unmatched_ablation_runs,
            "baseline_seeds": baseline_seeds,
        }

    group_summary_df = group_summary_df.sort_values(
        ["val_loss_mean_delta", "dropped_group"], ascending=[False, True]
    ).reset_index(drop=True)
    group_order = group_summary_df["dropped_group"].astype(str).tolist()
    category = pd.CategoricalDtype(categories=group_order, ordered=True)
    working["dropped_group"] = working["dropped_group"].astype(category)
    working = working.sort_values(["dropped_group", "training_seed"]).reset_index(drop=True)

    export_table_df = pd.DataFrame(export_rows)
    if not export_table_df.empty:
        export_table_df["dropped_group"] = export_table_df["dropped_group"].astype(category)
        export_table_df = export_table_df.sort_values(["dropped_group", "metric"]).reset_index(
            drop=True
        )

    return {
        "paired_df": working,
        "group_summary_df": group_summary_df,
        "export_table_df": export_table_df,
        "group_order": group_order,
        "missing_pairs": missing_pairs,
        "unmatched_ablation_runs": unmatched_ablation_runs,
        "baseline_seeds": baseline_seeds,
    }


def _build_group_y_positions(summary_df: pd.DataFrame) -> np.ndarray:
    return np.arange(len(summary_df), dtype=float)


def build_validation_loss_delta_figure(analysis: dict[str, Any]) -> tuple[plt.Figure, plt.Axes]:
    summary_df = analysis.get("group_summary_df", pd.DataFrame())
    fig, ax = plt.subplots(figsize=(12.2, 6.8))
    if summary_df.empty:
        ax.text(
            0.5,
            0.5,
            "No paired feature-ablation results available.",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        return fig, ax

    y = _build_group_y_positions(summary_df)
    means = pd.to_numeric(summary_df["val_loss_mean_delta"], errors="coerce").to_numpy(dtype=float)
    labels = summary_df["group_label"].astype(str).tolist()

    # Bars show the mean paired change across seeds.
    ax.barh(y, means, color="#8ea6c9", edgecolor="#5f7089", height=0.62, zorder=1)

    ax.axvline(0.0, color="#5b5b5b", linewidth=1.3, linestyle="--", zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=11)
    ax.invert_yaxis()
    ax.set_xlabel("Change in validation loss after removing feature group", fontsize=12)
    ax.set_title(
        "Static-feature ablation: paired validation-loss change vs full static", fontsize=14
    )
    ax.grid(axis="x", alpha=0.22, linewidth=0.8)
    fig.subplots_adjust(left=0.33, right=0.98, top=0.9, bottom=0.12)
    return fig, ax


def build_target_delta_panels_figure(
    analysis: dict[str, Any],
) -> tuple[dict[str, plt.Figure], dict[str, plt.Axes]]:
    summary_df = analysis.get("group_summary_df", pd.DataFrame()).copy()
    figures: dict[str, plt.Figure] = {}
    axes: dict[str, plt.Axes] = {}
    if summary_df.empty:
        for metric, title, _, export_key in TARGET_METRIC_SPECS:
            fig, ax = plt.subplots(figsize=(12.2, 6.8))
            ax.text(
                0.5,
                0.5,
                "No paired feature-ablation results available.",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_axis_off()
            figures[export_key] = fig
            axes[metric] = ax
        return figures, axes

    target_order = [
        group
        for group in TARGET_PANEL_GROUP_ORDER
        if group in set(summary_df["dropped_group"].astype(str))
    ]
    extra_groups = [
        group
        for group in summary_df["dropped_group"].astype(str).tolist()
        if group not in target_order
    ]
    display_order = target_order + extra_groups
    category = pd.CategoricalDtype(categories=display_order, ordered=True)
    summary_df["dropped_group"] = summary_df["dropped_group"].astype(str).astype(category)
    summary_df = summary_df.sort_values("dropped_group").reset_index(drop=True)

    y = _build_group_y_positions(summary_df)
    labels = summary_df["group_label"].astype(str).tolist()
    for metric, title, color, export_key in TARGET_METRIC_SPECS:
        fig, ax = plt.subplots(figsize=(12.2, 6.8))
        means = pd.to_numeric(summary_df[f"{metric}_mean_delta"], errors="coerce").to_numpy(
            dtype=float
        )
        # Bars show the mean paired change across seeds.
        ax.barh(y, means, color=color, edgecolor="#5f5f5f", height=0.62, zorder=1)
        ax.axvline(0.0, color="#5b5b5b", linewidth=1.2, linestyle="--", zorder=2)
        ax.set_title(f"Static-feature ablation: paired {title} change vs full static", fontsize=14)
        ax.set_xlabel(METRIC_AXIS_LABELS[metric], fontsize=11)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=10.5)
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.22, linewidth=0.8)
        fig.subplots_adjust(left=0.33, right=0.98, top=0.9, bottom=0.12)
        figures[export_key] = fig
        axes[metric] = ax
    return figures, axes


def build_feature_delta_heatmap_figure(analysis: dict[str, Any]) -> tuple[plt.Figure, plt.Axes]:
    summary_df = analysis.get("group_summary_df", pd.DataFrame())
    fig, ax = plt.subplots(figsize=(11.5, 6.8))
    if summary_df.empty:
        ax.text(
            0.5,
            0.5,
            "No paired feature-ablation results available.",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        return fig, ax

    metrics = ["val_loss", "val_hs_rmse", "val_tp_rmse", "val_dir_rmse_deg", "val_dp_rmse_deg"]
    value_matrix = np.column_stack(
        [
            pd.to_numeric(summary_df[f"{metric}_mean_delta"], errors="coerce").to_numpy(dtype=float)
            for metric in metrics
        ]
    )
    color_matrix = np.zeros_like(value_matrix, dtype=float)
    for col_idx in range(value_matrix.shape[1]):
        column = value_matrix[:, col_idx]
        finite = column[np.isfinite(column)]
        scale = np.max(np.abs(finite)) if finite.size else 0.0
        if scale > 0:
            color_matrix[:, col_idx] = column / scale

    im = ax.imshow(color_matrix, cmap="RdBu_r", vmin=-1.0, vmax=1.0, aspect="auto")
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_xticklabels(
        [METRIC_COLUMNS[metric] for metric in metrics], rotation=20, ha="right", fontsize=11
    )
    ax.set_yticks(np.arange(len(summary_df)))
    ax.set_yticklabels(summary_df["group_label"].astype(str).tolist(), fontsize=10.5)
    ax.set_title(
        "Static-feature ablation: mean paired change vs full static\nHeatmap colour is normalized within each metric column; annotations show original delta values.",
        fontsize=13,
    )

    for row_idx in range(value_matrix.shape[0]):
        for col_idx in range(value_matrix.shape[1]):
            value = value_matrix[row_idx, col_idx]
            text = "NA" if not np.isfinite(value) else f"{value:+.3f}"
            ax.text(col_idx, row_idx, text, ha="center", va="center", fontsize=9, color="#1f1f1f")

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Per-metric normalized mean paired change", fontsize=10)
    fig.tight_layout()
    return fig, ax


def export_feature_delta_outputs(
    analysis: dict[str, Any],
    *,
    figures: dict[str, plt.Figure | None],
    resolved_root: Path | None,
    output_dir_override: str | Path | None = None,
) -> tuple[Path | None, dict[str, str]]:
    if resolved_root is None:
        return None, {}
    if output_dir_override not in (None, ""):
        output_dir = resolve_path(output_dir_override, anchor=resolved_root)
    else:
        output_dir = (resolved_root / "figures").resolve()
    if output_dir is None:
        return None, {}
    output_dir.mkdir(parents=True, exist_ok=True)

    exported_paths: dict[str, str] = {}
    export_table_df = analysis.get("export_table_df", pd.DataFrame())
    paired_df = analysis.get("paired_df", pd.DataFrame())
    summary_df = analysis.get("group_summary_df", pd.DataFrame())

    if not export_table_df.empty:
        csv_path = output_dir / "features_paired_delta_summary.csv"
        export_table_df.to_csv(csv_path, index=False)
        exported_paths["features_paired_delta_summary_csv"] = str(csv_path)
    if not paired_df.empty:
        paired_path = output_dir / "features_paired_delta_seed_rows.csv"
        paired_df.to_csv(paired_path, index=False)
        exported_paths["features_paired_delta_seed_rows_csv"] = str(paired_path)
    if not summary_df.empty:
        summary_path = output_dir / "features_paired_delta_group_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        exported_paths["features_paired_delta_group_summary_csv"] = str(summary_path)

    for key, fig in figures.items():
        if fig is None:
            continue
        png_path = output_dir / f"features_{key}.png"
        pdf_path = output_dir / f"features_{key}.pdf"
        fig.savefig(png_path, dpi=300, bbox_inches="tight")
        fig.savefig(pdf_path, bbox_inches="tight")
        exported_paths[f"{key}_png"] = str(png_path)
        exported_paths[f"{key}_pdf"] = str(pdf_path)

    if exported_paths:
        print("Feature-delta figure encoding: bars=mean paired change across seeds.")

    return output_dir, exported_paths


__all__ = [
    "build_ablation_summary",
    "build_feature_delta_analysis",
    "build_feature_seed_paired_tables",
    "build_feature_delta_heatmap_figure",
    "TARGET_METRIC_SPECS",
    "build_target_delta_panels_figure",
    "build_validation_loss_delta_figure",
    "export_feature_delta_outputs",
    "feature_mode_note",
]
