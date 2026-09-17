"""Site CSV loading and temporal alignment for point-centric preparation."""

from __future__ import annotations
import os
import logging
from pathlib import Path
from typing import List, Dict
import numpy as np
import pandas as pd


def _filename_matches_site_name(filename: str, site_name: str) -> bool:
    stem = Path(filename).stem.lower()
    site = str(site_name).strip().lower()
    if not stem or not site:
        return False
    return (
        stem == site
        or stem.startswith(f"{site}_")
        or stem.endswith(f"_{site}")
        or f"_{site}_" in stem
    )


def find_param_files(site_name: str, params_dir: str) -> List[str]:
    if not params_dir or not os.path.isdir(params_dir):
        return []

    matches: List[str] = []
    for root, _, files in os.walk(params_dir):
        for fn in files:
            low = fn.lower()
            if (
                low.endswith(".csv")
                and "zone.identifier" not in low
                and _filename_matches_site_name(fn, site_name)
            ):
                matches.append(os.path.join(root, fn))
    return sorted(matches)


def _parse_time_index(df: pd.DataFrame, datetime_col: str) -> pd.DataFrame:
    if datetime_col not in df.columns:
        return pd.DataFrame()
    out = df.copy()
    out[datetime_col] = pd.to_datetime(out[datetime_col], errors="coerce")
    out = out.dropna(subset=[datetime_col]).set_index(datetime_col).sort_index()
    out = out[~out.index.duplicated(keep="last")]
    return out


def load_site_timeseries(
    site_name: str, params_dir: str, datetime_col: str = "time"
) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for fp in find_param_files(site_name, params_dir):
        try:
            d = pd.read_csv(fp, comment="#")
        except Exception:
            try:
                d = pd.read_csv(fp)
            except Exception:
                logging.warning("Failed to read %s", fp)
                continue
        d = _parse_time_index(d, datetime_col)
        if not d.empty:
            frames.append(d)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, axis=0, ignore_index=False).sort_index()
    out = out[~out.index.duplicated(keep="last")]
    return out


def _build_split_indices(n_rows: int, train_frac: float, val_frac: float) -> Dict[str, np.ndarray]:
    n_train = int(n_rows * train_frac)
    n_val = int(n_rows * val_frac)
    train_idx = np.arange(0, n_train)
    val_idx = np.arange(n_train, n_train + n_val)
    test_idx = np.arange(n_train + n_val, n_rows)
    return {"train": train_idx, "val": val_idx, "test": test_idx}


def _normalize_timestamp_naive(ts: pd.Timestamp) -> pd.Timestamp:
    """Return a timezone-naive timestamp for reliable timeline comparisons."""
    if ts.tzinfo is not None:
        return ts.tz_convert(None)
    return ts


def _parse_date_range_bound(raw_value, bound_name: str, is_end: bool) -> pd.Timestamp | None:
    """Parse one date-range bound from config.

    Accepts null/empty values and ISO-like strings. For date-only `end`
    strings (`YYYY-MM-DD`), expands to the end of day so filtering remains
    intuitive on sub-daily timelines.
    """
    if raw_value is None:
        return None
    raw_text = str(raw_value).strip()
    if not raw_text:
        return None

    try:
        ts = pd.Timestamp(pd.to_datetime(raw_text, errors="raise"))
    except Exception as exc:
        raise ValueError(
            f"Invalid data.date_range.{bound_name}='{raw_value}'. "
            "Expected ISO date/datetime like '2020-01-01' or '2020-01-01T12:00:00'."
        ) from exc

    ts = _normalize_timestamp_naive(ts)
    if is_end and ("T" not in raw_text) and (" " not in raw_text):
        # Date-only end bounds are interpreted as inclusive end-of-day.
        ts = ts + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
    return ts


def _resolve_and_apply_date_range(
    aligned_index: pd.DatetimeIndex,
    data_cfg: dict,
) -> tuple[pd.DatetimeIndex, dict]:
    """Filter aligned timestamps by optional target-date range config."""
    date_cfg = data_cfg.get("date_range", {}) or {}
    enabled = bool(date_cfg.get("enabled", False))
    start_raw = date_cfg.get("start", None)
    end_raw = date_cfg.get("end", None)
    start_ts = _parse_date_range_bound(start_raw, "start", is_end=False)
    end_ts = _parse_date_range_bound(end_raw, "end", is_end=True)

    if start_ts is not None and end_ts is not None and start_ts > end_ts:
        raise ValueError(
            "Invalid data.date_range bounds: start is after end "
            f"(start={start_ts.isoformat()} end={end_ts.isoformat()})."
        )

    base_index = aligned_index
    if getattr(base_index, "tz", None) is not None:
        base_index = base_index.tz_convert(None)
    before_count = int(len(base_index))

    if enabled:
        mask = np.ones(before_count, dtype=bool)
        if start_ts is not None:
            mask &= np.asarray(base_index >= start_ts, dtype=bool)
        if end_ts is not None:
            mask &= np.asarray(base_index <= end_ts, dtype=bool)
        filtered_index = base_index[mask]
    else:
        filtered_index = base_index

    after_count = int(len(filtered_index))
    dropped_count = int(before_count - after_count)

    if enabled and after_count == 0:
        raise ValueError(
            "Date-range filtering removed all samples. "
            f"Configured range: start={start_raw!r}, end={end_raw!r}. "
            "Expand data.date_range bounds or disable data.date_range.enabled."
        )

    return filtered_index, {
        "enabled": enabled,
        "requested_start": None if start_raw is None else str(start_raw),
        "requested_end": None if end_raw is None else str(end_raw),
        "resolved_start": start_ts.isoformat() if start_ts is not None else None,
        "resolved_end": end_ts.isoformat() if end_ts is not None else None,
        "timestamp_count_before_filter": before_count,
        "timestamp_count_after_filter": after_count,
        "timestamp_count_dropped": dropped_count,
    }
