#!/usr/bin/env python3
"""Discover parameter CSVs and prepare legacy or point-centric wave datasets."""

from __future__ import annotations

import os
import glob
import logging
from typing import Dict, List

import yaml
from pathlib import Path
import numpy as np

try:
    import pandas as pd

    _HAS_PANDAS = True
except Exception:
    _HAS_PANDAS = False


def read_sites_yaml(path):
    """Read configuration paths relative to their declaring file."""
    from coastal_wave.common.config import read_config

    return read_config(path)


def _safe_glob(params_dir: str, pattern: str) -> List[str]:
    """Recursive glob that excludes Windows Zone.Identifier artifacts."""
    pat = os.path.join(params_dir, "**", pattern)
    files = [
        p for p in glob.glob(pat, recursive=True) if "Zone.Identifier" not in os.path.basename(p)
    ]
    return sorted(files)


def find_param_files(site_name: str, params_dir: str) -> List[str]:
    """Find CSV files that contain `site_name` in `params_dir` (recursive).

    Returns a list of matching file paths (may be empty).
    """
    if not params_dir or not os.path.isdir(params_dir):
        return []
    # look for CSVs that include the site name
    files = _safe_glob(params_dir, f"*{site_name}*.csv")
    if files:
        return files
    # fallback: case-insensitive search
    matches = []
    for root, _, fns in os.walk(params_dir):
        for fn in fns:
            if (
                fn.lower().endswith(".csv")
                and "zone.identifier" not in fn.lower()
                and site_name.lower() in fn.lower()
            ):
                matches.append(os.path.join(root, fn))
    return sorted(matches)


def discover_site_files(sites_yaml_path: str = "configs/sites.yaml") -> Dict[str, dict]:
    """Return a mapping of nearshore site name -> discovered data files.

    The returned dict has entries for each `nearshore_sites` site with the
    following keys:
      - `site_meta`: the site metadata dict from YAML
      - `target_files`: list of matching NORAC param CSV paths
      - `paired_offshore_files`: list of matching NORA3 param CSV paths (for paired_offshore)
      - `nora3_params_dir` / `norac_params_dir`: directories used for discovery
    """
    sites = read_sites_yaml(sites_yaml_path)
    backend = sites.get("backend", {})
    nora3_dir = backend.get("nora3_params_dir") or "data/raw/nora3/params"
    norac_dir = backend.get("norac_params_dir") or "data/raw/norac/params"

    # fallback heuristics if config paths don't exist
    if not os.path.isdir(nora3_dir):
        for cand in ["data/nora3/params", "data/nora3/params/.chunks", "data/nora3"]:
            if os.path.isdir(cand):
                nora3_dir = cand
                break
    if not os.path.isdir(norac_dir):
        for cand in ["data/norac/params", "data/norac/params/.chunks", "data/norac"]:
            if os.path.isdir(cand):
                norac_dir = cand
                break

    result: Dict[str, dict] = {}
    nearshore = sites.get("nearshore_sites") or []
    for s in nearshore:
        name = s.get("name")
        target_files = find_param_files(name, norac_dir)
        paired = s.get("paired_offshore") or []
        paired_paths: List[str] = []
        for p in paired:
            paired_paths.extend(find_param_files(p, nora3_dir))
        result[name] = {
            "site_meta": s,
            "target_files": target_files,
            "paired_offshore_files": paired_paths,
            "nora3_params_dir": nora3_dir,
            "norac_params_dir": norac_dir,
        }
    return result


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Discover site data files defined in configs/sites.yaml"
    )
    parser.add_argument("--sites", default="configs/sites.yaml", help="Path to sites.yaml")
    parser.add_argument("--list", action="store_true", help="List site -> file counts")
    parser.add_argument(
        "--first", action="store_true", help="Print a short sample of the first found target file"
    )
    parser.add_argument(
        "--build-point-centric",
        action="store_true",
        help="Build aligned point-centric arrays for coastal-transformer models",
    )
    parser.add_argument(
        "--training-config", default="configs/training.yaml", help="Path to training.yaml"
    )
    parser.add_argument(
        "--preprocess-config",
        default="configs/preprocess.yaml",
        help="Path to preprocess.yaml (controls processed paths, circular encoding)",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory for preprocessed datasets (overrides training.yaml)",
    )
    parser.add_argument(
        "-n",
        "--name",
        default=None,
        help="Subfolder name under processed dir to create for this run (e.g. test1)",
    )
    args = parser.parse_args()

    mapping = discover_site_files(args.sites)
    print(f"Discovered {len(mapping)} nearshore sites")
    if args.list:
        for k, v in mapping.items():
            print(
                f"{k}: target_files={len(v['target_files'])}, paired_offshore_files={len(v['paired_offshore_files'])}"
            )

    if args.first:
        if not mapping:
            print("No sites discovered")
            return
        first_name, info = next(iter(mapping.items()))
        print("First site:", first_name)
        print("Meta:", info["site_meta"])
        if info["target_files"]:
            fp = info["target_files"][0]
            print("Target file:", fp)
            try:
                import pandas as pd

                df = pd.read_csv(fp, comment="#", nrows=5)
                print(df.head())
            except Exception:
                print("Pandas not available or failed to read; printing raw lines")
                with open(fp, "r") as fh:
                    for _ in range(10):
                        line = fh.readline()
                        if not line:
                            break
                        print(line.rstrip())
        else:
            print("No target file found for this site.")

    out_dir_to_use = None
    if args.build_point_centric:
        # determine output directory: -n name takes precedence and will be created under preprocess.config paths.processed_dir
        if args.name:
            p_cfg = _load_preprocess_cfg(args.preprocess_config)
            base = p_cfg.get("paths", {}).get("processed_dir") if p_cfg else None
            if not base:
                base = "data/processed"
            out_dir_to_use = str(Path(base) / args.name)
        elif args.out_dir:
            out_dir_to_use = args.out_dir

    if args.build_point_centric:
        try:
            from point_centric_pipeline import build_point_centric_dataset
        except Exception:
            from src.point_centric_pipeline import build_point_centric_dataset

        paths = build_point_centric_dataset(
            sites_yaml=args.sites,
            training_config=args.training_config,
            preprocess_config=args.preprocess_config,
            out_dir=out_dir_to_use,
        )
        print("Point-centric outputs:")
        for k, v in paths.items():
            print(f"  {k}: {v}")


def _load_training_cfg(path: str) -> dict:
    """Read configuration with explicit path ownership."""
    from src.config_loader import read_yaml_config

    return read_yaml_config(path)


def _load_preprocess_cfg(path: str) -> dict:
    """Read configuration with explicit path ownership."""
    from src.config_loader import read_yaml_config

    return read_yaml_config(path)


def _encode_circular_features(df, preprocess_cfg: dict, datetime_col: str = "time"):
    """Encode circular variables into sin/cos pairs and optionally add time circular features.

    Modifies `df` in place and returns the list of added column names.
    """
    added = []
    circ_cfg = preprocess_cfg.get("circular", {}) if preprocess_cfg else {}
    # gather column patterns from config (may be base names like 'Pdir' or explicit names)
    patterns = list(circ_cfg.get("columns") or [])
    auto_detect = circ_cfg.get("auto_detect_dir", True)
    if auto_detect:
        for c in df.columns:
            if any(x in c.lower() for x in ("dir", "direction")) and c not in patterns:
                patterns.append(c)

    # Expand patterns against actual df columns (match exact, suffix _{pattern}, prefix, or containing _{pattern}_)
    matched_cols = set()
    df_cols_lower = [c.lower() for c in df.columns]
    for pat in patterns:
        pat_low = pat.lower()
        for c, c_low in zip(df.columns, df_cols_lower):
            if (
                c_low == pat_low
                or c_low.endswith("_" + pat_low)
                or c_low.startswith(pat_low + "_")
                or ("_" + pat_low + "_") in c_low
                or pat_low in c_low
            ):
                matched_cols.add(c)

    # fallback: if no explicit patterns matched, default to auto-detected direction-like columns
    if not matched_cols and auto_detect:
        for c in df.columns:
            if any(x in c.lower() for x in ("dir", "direction")):
                matched_cols.add(c)

    input_degrees = circ_cfg.get("input_degrees", True)
    drop_original = circ_cfg.get("drop_original", True)

    cols = [c for c in matched_cols]
    for c in cols:
        try:
            vals = df[c].astype(float).values
        except Exception:
            # skip non-numeric
            continue
        if input_degrees:
            radians = np.deg2rad(vals)
        else:
            radians = vals
        sin_col = f"{c}_sin"
        cos_col = f"{c}_cos"
        df[sin_col] = np.sin(radians)
        df[cos_col] = np.cos(radians)
        added.extend([sin_col, cos_col])
        if drop_original:
            try:
                df.drop(columns=[c], inplace=True)
            except Exception:
                pass

    # time-based circular features
    time_cfg = circ_cfg.get("time", {})
    add_time = circ_cfg.get("add_time_features", False) or bool(time_cfg)
    if add_time and datetime_col in df.columns:
        if _HAS_PANDAS:
            try:
                df[datetime_col] = pd.to_datetime(
                    df[datetime_col], errors="coerce", infer_datetime_format=True
                )
            except Exception:
                df[datetime_col] = pd.to_datetime(df[datetime_col], errors="coerce")
            # month
            if time_cfg.get("month", True):
                months = df[datetime_col].dt.month.fillna(0).astype(float)
                ang = 2 * np.pi * ((months - 1) / 12.0)
                df["month_sin"] = np.sin(ang)
                df["month_cos"] = np.cos(ang)
                added.extend(["month_sin", "month_cos"])
            # day (day of month)
            if time_cfg.get("day", True):
                days = df[datetime_col].dt.day.fillna(0).astype(float)
                ang = 2 * np.pi * ((days - 1) / 31.0)
                df["day_sin"] = np.sin(ang)
                df["day_cos"] = np.cos(ang)
                added.extend(["day_sin", "day_cos"])
            # hour
            if time_cfg.get("hour", True):
                hours = df[datetime_col].dt.hour.fillna(0).astype(float)
                ang = 2 * np.pi * (hours / 24.0)
                df["hour_sin"] = np.sin(ang)
                df["hour_cos"] = np.cos(ang)
                added.extend(["hour_sin", "hour_cos"])
        else:
            # no pandas: cannot extract time components reliably
            pass

    return added


def _load_normalize_module():
    """Return the installed normalization module."""
    from src.preprocessing import normalize

    return normalize


def _assemble_timeseries(mapping: Dict[str, dict], datetime_col: str = "time"):
    """Read all discovered NORAC CSVs and return a concatenated DataFrame.

    The resulting DataFrame will contain a `norac_site` column set to the nearshore site
    name used in `configs/sites.yaml`.
    """
    if not _HAS_PANDAS:
        raise RuntimeError("pandas is required to assemble timeseries datasets")
    dfs = []
    for site_name, info in mapping.items():
        for fp in info.get("target_files", []):
            try:
                df = pd.read_csv(fp, comment="#")
            except Exception:
                try:
                    df = pd.read_csv(fp)
                except Exception:
                    logging.warning("Failed to read %s", fp)
                    continue
            if datetime_col in df.columns:
                try:
                    df[datetime_col] = pd.to_datetime(
                        df[datetime_col], errors="coerce", infer_datetime_format=True
                    )
                except Exception:
                    df[datetime_col] = pd.to_datetime(df[datetime_col], errors="coerce")
            df = df.reset_index(drop=True)
            df["norac_site"] = site_name
            dfs.append(df)
    if not dfs:
        return pd.DataFrame()
    all_df = pd.concat(dfs, ignore_index=True, sort=False)
    return all_df


def _assemble_site_timeseries(site_names, params_dir: str, datetime_col: str = "time"):
    """Return a dict site_name -> DataFrame for the requested param site names.

    Uses `find_param_files` to locate CSVs for each site and concatenates them.
    """
    if not _HAS_PANDAS:
        raise RuntimeError("pandas is required to assemble timeseries datasets")
    result = {}
    for name in site_names:
        fps = find_param_files(name, params_dir)
        if not fps:
            continue
        parts = []
        for fp in fps:
            try:
                d = pd.read_csv(fp, comment="#")
            except Exception:
                try:
                    d = pd.read_csv(fp)
                except Exception:
                    logging.warning("Failed to read offshore file %s", fp)
                    continue
            if datetime_col in d.columns:
                try:
                    d[datetime_col] = pd.to_datetime(
                        d[datetime_col], errors="coerce", infer_datetime_format=True
                    )
                except Exception:
                    d[datetime_col] = pd.to_datetime(d[datetime_col], errors="coerce")
            d = d.reset_index(drop=True)
            parts.append(d)
        if parts:
            result[name] = pd.concat(parts, ignore_index=True, sort=False)
    return result


def _aggregate_offshore_features(
    mapping: Dict[str, dict],
    df,
    offshore_vars,
    datetime_col: str = "time",
    method: str = "mean",
    direction_vars=None,
):
    """Aggregate paired offshore variables for each nearshore row and add `off_<var>` columns.

    By default aggregates by mean across the paired offshore sites. Rows with no matching
    offshore samples will have NaNs.

    For NORA3 directional variables, apply a +180 degree offset (wrapped to [0, 360))
    except for wind direction variables, which are already aligned.
    """
    if not _HAS_PANDAS:
        raise RuntimeError("pandas is required for offshore aggregation")
    # find a params dir from mapping
    nora3_dir = None
    for info in mapping.values():
        nora3_dir = info.get("nora3_params_dir") or nora3_dir
    if not nora3_dir:
        nora3_dir = "data/raw/nora3/params"

    # collect unique offshore site names referenced by nearshore 'paired_offshore'
    unique_offshore = sorted(
        {nm for info in mapping.values() for nm in (info["site_meta"].get("paired_offshore") or [])}
    )
    offshore_map = _assemble_site_timeseries(unique_offshore, nora3_dir, datetime_col=datetime_col)

    # ensure datetime column in df is datetime
    try:
        df[datetime_col] = pd.to_datetime(
            df[datetime_col], errors="coerce", infer_datetime_format=True
        )
    except Exception:
        df[datetime_col] = pd.to_datetime(df[datetime_col], errors="coerce")

    # create concatenated columns for each offshore site and variable: off_{site}_{var}
    for p in unique_offshore:
        for v in offshore_vars:
            col_name = f"off_{p}_{v}"
            if col_name not in df.columns:
                df[col_name] = np.nan

    direction_set = {str(v).lower() for v in (direction_vars or [])}

    # iterate per nearshore site and fill concatenated offshore variables for its rows
    for site, info in mapping.items():
        paired = info["site_meta"].get("paired_offshore") or []
        if not paired:
            continue
        idx = df[df["norac_site"] == site].index
        if len(idx) == 0:
            continue
        sub_times = df.loc[idx, datetime_col]
        for p in paired:
            off_df = offshore_map.get(p)
            if off_df is None or off_df.empty:
                continue
            if datetime_col not in off_df.columns:
                continue
            off_df = off_df.copy()
            try:
                off_df[datetime_col] = pd.to_datetime(
                    off_df[datetime_col], errors="coerce", infer_datetime_format=True
                )
            except Exception:
                off_df[datetime_col] = pd.to_datetime(off_df[datetime_col], errors="coerce")
            off_df = off_df.set_index(datetime_col)
            for v in offshore_vars:
                col_name = f"off_{p}_{v}"
                if v in off_df.columns:
                    try:
                        vals = off_df.reindex(sub_times.values)[v].values
                    except Exception:
                        vals = off_df.reindex(pd.to_datetime(sub_times.values))[v].values

                    var_name = str(v).lower()
                    is_directional = var_name in direction_set or any(
                        token in var_name
                        for token in ("dir", "direction", "thq", "bearing", "heading")
                    )
                    is_wind_direction = "wind" in var_name and (
                        "dir" in var_name or "direction" in var_name
                    )

                    if is_directional and not is_wind_direction:
                        numeric = pd.to_numeric(vals, errors="coerce").astype(float)
                        vals = np.mod(numeric + 180.0, 360.0)

                    # assign aligned values back to original df rows
                    df.loc[idx, col_name] = vals
                else:
                    # leave NaNs for missing variable
                    continue

    return df


def _assemble_nearshore_targets(
    mapping: Dict[str, dict],
    df,
    nearshore_vars,
    direction_vars,
    datetime_col: str = "time",
    preprocess_cfg: dict = None,
):
    """Create wide target columns by concatenating all nearshore (NORAC) sites.

    Adds columns named `y_<site>_<var>` for scalar vars and
    `y_<site>_<var>_sin`, `y_<site>_<var>_cos` for directional vars.
    Values are aligned by timestamp (nearest exact index - reindexing)
    and may contain NaNs when a site has no value at that timestamp.
    """
    if not _HAS_PANDAS:
        raise RuntimeError("pandas is required to assemble nearshore targets")

    # determine norac params dir from mapping
    norac_dir = None
    for info in mapping.values():
        norac_dir = info.get("norac_params_dir") or norac_dir
    if not norac_dir:
        norac_dir = "data/raw/norac/params"

    unique_nearshore = sorted(list(mapping.keys()))
    # load per-site nearshore timeseries
    near_map = _assemble_site_timeseries(unique_nearshore, norac_dir, datetime_col=datetime_col)

    # ensure datetime col
    try:
        df[datetime_col] = pd.to_datetime(
            df[datetime_col], errors="coerce", infer_datetime_format=True
        )
    except Exception:
        df[datetime_col] = pd.to_datetime(df[datetime_col], errors="coerce")

    # input degrees setting
    input_degrees = True
    if preprocess_cfg:
        input_degrees = preprocess_cfg.get("circular", {}).get("input_degrees", True)

    times = df[datetime_col].values

    # create target columns upfront
    for p in unique_nearshore:
        for v in nearshore_vars:
            if v in (direction_vars or []) or (isinstance(v, str) and v.lower() == "dp"):
                df[f"y_{p}_{v}_sin"] = np.nan
                df[f"y_{p}_{v}_cos"] = np.nan
            else:
                df[f"y_{p}_{v}"] = np.nan

    # fill columns by reindexing each site's series to the global times
    for p in unique_nearshore:
        p_df = near_map.get(p)
        if p_df is None or p_df.empty:
            continue
        if datetime_col not in p_df.columns:
            continue
        p_df = p_df.copy()
        try:
            p_df[datetime_col] = pd.to_datetime(
                p_df[datetime_col], errors="coerce", infer_datetime_format=True
            )
        except Exception:
            p_df[datetime_col] = pd.to_datetime(p_df[datetime_col], errors="coerce")
        p_df = p_df.set_index(datetime_col)

        for v in nearshore_vars:
            if v not in p_df.columns:
                # leave NaNs if variable missing for this site
                continue
            try:
                vals = p_df.reindex(times)[v].values
            except Exception:
                try:
                    vals = p_df.reindex(pd.to_datetime(times))[v].values
                except Exception:
                    vals = np.full(len(times), np.nan)

            if v in (direction_vars or []) or (isinstance(v, str) and v.lower() == "dp"):
                # compute sin/cos for directional variables
                try:
                    numeric = pd.to_numeric(vals, errors="coerce").astype(float)
                except Exception:
                    numeric = np.array([np.nan] * len(vals), dtype=float)
                if input_degrees:
                    radians = np.deg2rad(numeric)
                else:
                    radians = numeric
                sin = np.sin(radians)
                cos = np.cos(radians)
                df[f"y_{p}_{v}_sin"] = sin
                df[f"y_{p}_{v}_cos"] = cos
            else:
                df[f"y_{p}_{v}"] = vals

    return df


def _attach_static_features(mapping: Dict[str, dict], df, preprocess_cfg: dict = None):
    """Attach aggregated static features per nearshore site to `df`.

    The function looks for a static CSV path in `preprocess_cfg` (key
    `static_features.out_path`) and falls back to common locations. It
    returns the augmented DataFrame and a list of the static column names.
    """
    if not _HAS_PANDAS:
        raise RuntimeError("pandas is required to attach static features")

    static_path = None
    if preprocess_cfg:
        static_path = preprocess_cfg.get("static_features", {}).get("out_path")
    candidates = [
        static_path,
        "data/processed/static_features.csv",
        "experiments/static_features.csv",
        "experiments/static_features_wide.csv",
    ]
    csv_path = None
    for c in candidates:
        if not c:
            continue
        if os.path.exists(c):
            csv_path = c
            break

    if csv_path is None:
        logging.info("No static features CSV found; skipping static feature attachment")
        return df, []

    try:
        sdf = pd.read_csv(csv_path)
    except Exception as e:
        logging.warning("Failed to read static features CSV %s: %s", csv_path, e)
        return df, []

    if "norac_name" not in sdf.columns:
        # try common alternate header
        if "norac_name" not in sdf.columns and "norac_name" not in sdf.columns:
            logging.warning("Static CSV has unexpected columns; skipping static features")
            return df, []

    # compute aggregates per nearshore (norac) site
    try:
        grp = sdf.groupby("norac_name")
    except Exception:
        return df, []

    agg = grp.agg(
        {
            "distance_m": "mean",
            "bathy_norac": "mean",
            "bathy_nora3": "mean",
            "bathy_diff": "mean",
        }
    )

    # bearing: compute mean via vector sum on unit circle
    def _mean_bearing(series):
        vals = pd.to_numeric(series, errors="coerce").dropna()
        if vals.size == 0:
            return float("nan"), float("nan")
        r = np.deg2rad(vals.astype(float))
        return float(np.sin(r).mean()), float(np.cos(r).mean())

    bearing_sin = []
    bearing_cos = []
    for name, g in grp:
        s, c = _mean_bearing(g.get("bearing_deg", pd.Series(dtype=float)))
        bearing_sin.append(s)
        bearing_cos.append(c)

    agg["bearing_sin"] = bearing_sin
    agg["bearing_cos"] = bearing_cos
    agg["n_pairs"] = grp.size()

    # prefix static_ to avoid name collisions
    agg.columns = [f"static_{c}" for c in agg.columns]

    # merge into df keyed by norac_site -> norac_name
    df = df.merge(agg, left_on="norac_site", right_index=True, how="left")
    static_cols = list(agg.columns)
    return df, static_cols


def build_and_normalize_datasets(
    sites_yaml: str = "configs/sites.yaml",
    training_config: str = "configs/training.yaml",
    out_dir: str = None,
    save_tensors: bool = False,
    preprocess_config: str = "configs/preprocess.yaml",
):
    """Assemble NORAC timeseries, perform chronological per-site split, normalize features.

    - Reads `sites_yaml` to discover NORAC files.
    - Reads `training_config` for split ratios, normalization method, and site split lists.
    - Performs a per-site chronological split with optional full-site validation/test overrides.
    - Fits normalization on the training subset only and applies to all splits.
    - Saves normalized CSVs and stats to `out_dir` (or training_config.out_dir).
    """
    logging.info("Loading sites from %s", sites_yaml)
    mapping = discover_site_files(sites_yaml)
    cfg = _load_training_cfg(training_config)
    split_cfg = cfg.get("split", {})
    data_cfg = cfg.get("data", {}) or {}
    legacy_keys = []
    if "holdout_sites" in data_cfg:
        legacy_keys.append("data.holdout_sites")
    if "norac_holdout" in split_cfg:
        legacy_keys.append("split.norac_holdout")
    if legacy_keys:
        raise ValueError(
            "Legacy site holdout keys are no longer supported: "
            f"{legacy_keys}. Use data.validation_sites and data.test_sites."
        )
    train_frac = float(split_cfg.get("train", 0.7))
    val_frac = float(split_cfg.get("val", 0.15))
    test_frac = float(split_cfg.get("test", 0.15))
    datetime_col = split_cfg.get("datetime_column", "time")
    validation_sites = set(str(site) for site in (data_cfg.get("validation_sites", []) or []))
    test_sites = set(str(site) for site in (data_cfg.get("test_sites", []) or []))
    # Determine output directory. Prefer explicit out_dir, then preprocess config, then training split.out_dir fallback.
    p_cfg = _load_preprocess_cfg(preprocess_config)
    processed_base = p_cfg.get("paths", {}).get("processed_dir") if p_cfg else None
    if out_dir:
        outdir = Path(out_dir)
    elif processed_base:
        outdir = Path(processed_base)
    else:
        outdir = Path(split_cfg.get("out_dir", "data/processed"))
    outdir.mkdir(parents=True, exist_ok=True)

    logging.info("Assembling timeseries from discovered files")
    df = _assemble_timeseries(mapping, datetime_col=datetime_col)
    if df.empty:
        logging.warning("No timeseries rows found; aborting")
        return {}

    # Ensure datetime column exists
    if datetime_col not in df.columns:
        logging.warning(
            "Datetime column '%s' not found; attempting to infer from index", datetime_col
        )
        try:
            df[datetime_col] = pd.to_datetime(df.iloc[:, 0], errors="coerce")
        except Exception:
            df[datetime_col] = pd.NaT

    # assign split column per-site
    df["split"] = ""
    for site, g in df.groupby("norac_site"):
        idx = g.index
        if site in validation_sites and site in test_sites:
            raise ValueError(
                f"Site '{site}' is configured in both data.validation_sites and data.test_sites"
            )
        if site in validation_sites:
            df.loc[idx, "split"] = "val"
            continue
        if site in test_sites:
            df.loc[idx, "split"] = "test"
            continue
        # chronological split per-site
        sub = g.sort_values(datetime_col)
        n = len(sub)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)
        train_idx = sub.index[:n_train]
        val_idx = sub.index[n_train : n_train + n_val]
        test_idx = sub.index[n_train + n_val :]
        df.loc[train_idx, "split"] = "train"
        df.loc[val_idx, "split"] = "val"
        df.loc[test_idx, "split"] = "test"

    # prepare offshore/nearshore variable lists from training config and aggregate offshore vars
    offshore_vars = data_cfg.get("offshore_vars") or []
    nearshore_vars = data_cfg.get("nearshore_vars") or []
    direction_vars = data_cfg.get("direction_vars") or []

    # aggregate paired offshore variables into `off_<var>` columns (mean across paired sites)
    if offshore_vars:
        try:
            df = _aggregate_offshore_features(
                mapping,
                df,
                offshore_vars,
                datetime_col=datetime_col,
                direction_vars=direction_vars,
            )
        except Exception as e:
            logging.warning("Offshore aggregation failed: %s", e)

    # encode circular variables (sin/cos) before deciding numeric columns or computing stats
    preprocess_cfg = _load_preprocess_cfg(preprocess_config)
    datetime_col = split_cfg.get("datetime_column", "time")
    # ensure direction vars from training config are encoded, include prefixed offshore names
    try:
        circ = preprocess_cfg.get("circular", {}) if preprocess_cfg else {}
        cols = list(circ.get("columns") or [])
        for v in direction_vars:
            if v not in cols:
                cols.append(v)
            offv = f"off_{v}"
            if offv not in cols:
                cols.append(offv)
        circ["columns"] = cols
        preprocess_cfg["circular"] = circ
    except Exception:
        pass

    try:
        added_cols = _encode_circular_features(df, preprocess_cfg, datetime_col=datetime_col)
    except Exception as e:
        logging.warning("Failed to encode circular features: %s", e)

    # assemble wide nearshore targets (Y) and attach static features before normalization
    try:
        df = _assemble_nearshore_targets(
            mapping,
            df,
            nearshore_vars,
            direction_vars,
            datetime_col=datetime_col,
            preprocess_cfg=preprocess_cfg,
        )
    except Exception as e:
        logging.warning("Failed to assemble wide nearshore targets: %s", e)

    try:
        df, static_cols = _attach_static_features(mapping, df, preprocess_cfg)
    except Exception as e:
        logging.warning("Failed to attach static features: %s", e)
        static_cols = []

    # Determine numeric columns, then split into circular/magnitude groups.
    # Circular columns are never scaled.
    norm_cfg = cfg.get("normalization", {})
    feature_columns = norm_cfg.get("numeric_columns")
    if not feature_columns:
        feature_columns = df.select_dtypes(include=[np.number]).columns.tolist()

    # Remove non-feature metadata columns.
    for c in [datetime_col, "norac_site", "split"]:
        if c in feature_columns:
            feature_columns.remove(c)

    circular_features = [c for c in feature_columns if str(c).endswith(("_sin", "_cos"))]
    magnitude_features = [c for c in feature_columns if c not in circular_features]

    method = norm_cfg.get("default_method", "zscore")
    methods_cfg = norm_cfg.get("methods", {})
    method_conf = methods_cfg.get(method, {})

    logging.info("Loading normalization module and computing stats on training data")
    norm_mod = _load_normalize_module()
    train_df = df[df["split"] == "train"]
    if train_df.empty:
        logging.warning("Empty training split; skipping normalization")
        return {}
    if magnitude_features:
        stats = norm_mod.compute_stats(
            train_df[magnitude_features],
            method,
            feature_range=tuple(method_conf.get("feature_range", (0.0, 1.0))),
            quantile_low=(method_conf.get("quantile_range") or [25, 75])[0],
            quantile_high=(method_conf.get("quantile_range") or [25, 75])[1],
        )
        stats["columns"] = list(magnitude_features)
    else:
        logging.warning("No magnitude features found for normalization; skipping scaler fit")
        stats = {"method": method, "columns": []}

    stats["skipped_circular_columns"] = list(circular_features)

    # apply normalization to each split and save X/Y feature files (offshore inputs X, nearshore targets Y)
    out_paths = {}
    for split in ("train", "val", "test"):
        sub = df[df["split"] == split].copy()
        if sub.empty:
            logging.info("No rows for split %s", split)
            continue
        if magnitude_features:
            normed_vals = norm_mod.apply_normalization(sub[magnitude_features], stats, method)
            # if returned numpy array
            if isinstance(normed_vals, np.ndarray):
                sub.loc[:, magnitude_features] = normed_vals
            else:
                # assume DataFrame-like
                sub.loc[:, magnitude_features] = (
                    normed_vals.values if hasattr(normed_vals, "values") else normed_vals
                )
        out_csv = outdir / f"{split}_norm.csv"
        sub.to_csv(str(out_csv), index=False)
        out_paths[split] = str(out_csv)

        # construct dynamic X (offshore), static X, and wide Y (all nearshore sites)
        # dynamic/offshore X columns (per-offshore-site concatenation)
        x_cols = []
        unique_offshore = sorted(
            {
                nm
                for info in mapping.values()
                for nm in (info["site_meta"].get("paired_offshore") or [])
            }
        )
        for p in unique_offshore:
            for v in offshore_vars:
                col_base = f"off_{p}_{v}"
                if v in direction_vars:
                    sin_col = f"{col_base}_sin"
                    cos_col = f"{col_base}_cos"
                    if sin_col in sub.columns and cos_col in sub.columns:
                        x_cols.extend([sin_col, cos_col])
                    elif col_base in sub.columns:
                        x_cols.append(col_base)
                    else:
                        if f"off_{v}_sin" in sub.columns and f"off_{v}_cos" in sub.columns:
                            x_cols.extend([f"off_{v}_sin", f"off_{v}_cos"])
                        elif f"off_{v}" in sub.columns:
                            x_cols.append(f"off_{v}")
                else:
                    if col_base in sub.columns:
                        x_cols.append(col_base)
                    elif f"off_{v}" in sub.columns:
                        x_cols.append(f"off_{v}")

        # static feature columns were created earlier and prefixed with 'static_'
        # ensure we have a variable even if attachment failed
        try:
            static_cols = static_cols or []
        except NameError:
            static_cols = []

        # wide Y: iterate all nearshore sites and nearshore_vars in deterministic order
        y_cols = []
        unique_nearshore = sorted(list(mapping.keys()))
        for p in unique_nearshore:
            for v in nearshore_vars:
                if v in direction_vars or (isinstance(v, str) and v.lower() == "dp"):
                    sin_c = f"y_{p}_{v}_sin"
                    cos_c = f"y_{p}_{v}_cos"
                    if sin_c in sub.columns and cos_c in sub.columns:
                        y_cols.extend([sin_c, cos_c])
                    elif f"y_{p}_{v}" in sub.columns:
                        y_cols.append(f"y_{p}_{v}")
                else:
                    if f"y_{p}_{v}" in sub.columns:
                        y_cols.append(f"y_{p}_{v}")

        # save dynamic X (offshore)
        if x_cols:
            x_csv = outdir / f"{split}_X_dynamic_norm.csv"
            sub[x_cols].to_csv(str(x_csv), index=False)
            out_paths[f"{split}_X_dynamic"] = str(x_csv)
        else:
            logging.info("No offshore X columns found for split %s", split)

        # save static X (if available)
        if static_cols:
            present_static = [c for c in static_cols if c in sub.columns]
            if present_static:
                s_csv = outdir / f"{split}_X_static_norm.csv"
                sub[present_static].to_csv(str(s_csv), index=False)
                out_paths[f"{split}_X_static"] = str(s_csv)
            else:
                logging.info("No static columns present for split %s", split)

        # save wide Y (all nearshore sites concatenated)
        if y_cols:
            y_csv = outdir / f"{split}_Y_norm.csv"
            sub[y_cols].to_csv(str(y_csv), index=False)
            out_paths[f"{split}_Y"] = str(y_csv)
        else:
            logging.info("No nearshore Y columns found for split %s", split)

    # save stats
    stats_path = outdir / "normalization_stats.json"
    norm_mod.save_stats(stats, str(stats_path))
    out_paths["stats"] = str(stats_path)

    # optionally save torch tensors
    if save_tensors:
        try:
            import torch

            for split in ("train", "val", "test"):
                p = outdir / f"{split}_norm.csv"
                if not p.exists():
                    continue
                sub = pd.read_csv(str(p))
                arr = sub[feature_columns].values.astype(float)
                torch.save(torch.tensor(arr), str(outdir / f"{split}_norm.pt"))
            logging.info("Saved tensors to %s", outdir)
        except Exception as e:
            logging.warning("Could not save tensors: %s", e)

    logging.info("Preprocessed datasets saved to %s", outdir)
    return out_paths


if __name__ == "__main__":
    main()
