"""Deterministic source selection shared by geometry and ML."""

from __future__ import annotations
import json
import math
from pathlib import Path
from typing import Sequence, Tuple, List
import numpy as np

EARTH_RADIUS_M = 6_371_000.0


def _try_float(value: object) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    if not np.isfinite(out):
        return None
    return out


def is_wave_source_name(name: object) -> bool:
    return "wind" not in str(name or "").strip().lower()


def great_circle_distance_m(
    lat1_deg: float,
    lon1_deg: float,
    lat2_deg: float,
    lon2_deg: float,
) -> float:
    """Return haversine great-circle distance in meters."""
    lat1 = math.radians(float(lat1_deg))
    lon1 = math.radians(float(lon1_deg))
    lat2 = math.radians(float(lat2_deg))
    lon2 = math.radians(float(lon2_deg))

    d_lat = lat2 - lat1
    d_lon = lon2 - lon1
    a = math.sin(d_lat * 0.5) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(d_lon * 0.5) ** 2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(max(1.0 - a, 0.0)))
    return float(EARTH_RADIUS_M * c)


def initial_bearing_deg(
    lat1_deg: float,
    lon1_deg: float,
    lat2_deg: float,
    lon2_deg: float,
) -> float:
    """Return initial compass bearing in [0, 360)."""
    lat1 = math.radians(float(lat1_deg))
    lon1 = math.radians(float(lon1_deg))
    lat2 = math.radians(float(lat2_deg))
    lon2 = math.radians(float(lon2_deg))

    d_lon = lon2 - lon1
    x = math.sin(d_lon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(d_lon)
    bearing = math.degrees(math.atan2(x, y)) % 360.0
    return float(bearing)


def inverse_distance_weights(
    distances_m: Sequence[float],
    power: float = 1.0,
) -> List[float]:
    distances = np.asarray(distances_m, dtype=np.float64)
    if distances.ndim != 1:
        raise ValueError(f"Expected 1D distance array, got shape {distances.shape}")
    if distances.size == 0:
        return []

    zero_mask = np.isclose(distances, 0.0)
    if np.any(zero_mask):
        weights = np.zeros_like(distances, dtype=np.float64)
        weights[zero_mask] = 1.0 / max(int(np.count_nonzero(zero_mask)), 1)
        return weights.astype(float).tolist()

    safe = np.maximum(distances, 1e-6)
    weights = np.power(1.0 / safe, float(power))
    total = float(np.sum(weights))
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("Failed to compute finite inverse-distance weights")
    return (weights / total).astype(float).tolist()


def resolve_multi_source_config(data_cfg: dict) -> dict:
    raw = (data_cfg.get("multi_source", {}) or {}) if isinstance(data_cfg, dict) else {}
    return {
        "enabled": bool(raw.get("enabled", False)),
        "source_type": str(raw.get("source_type", "nora3_wave")),
        "k_nearest": int(raw.get("k_nearest", 3)),
        "max_distance_km_warn": float(raw.get("max_distance_km_warn", 80.0)),
        "max_distance_km_error": float(raw.get("max_distance_km_error", 150.0)),
        "weight_power": float(raw.get("weight_power", 1.0)),
        "allow_padding": bool(raw.get("allow_padding", False)),
    }


def build_k_nearest_source_metadata(
    nearshore_entries: Sequence[dict],
    offshore_entries: Sequence[dict],
    k_nearest: int,
    source_type: str = "nora3_wave",
    weight_power: float = 1.0,
    max_distance_km_warn: float = 80.0,
    max_distance_km_error: float = 150.0,
    allow_padding: bool = False,
) -> Tuple[dict, List[str]]:
    """Select nearest wave sources for each nearshore site."""
    k_nearest = int(k_nearest)
    if k_nearest < 1:
        raise ValueError(f"k_nearest must be >= 1, got {k_nearest}")

    candidates: List[dict] = []
    for entry in offshore_entries:
        name = str(entry.get("name", "")).strip()
        lat = _try_float(entry.get("lat"))
        lon = _try_float(entry.get("lon"))
        if not name or lat is None or lon is None or not is_wave_source_name(name):
            continue
        candidates.append(
            {
                "name": name,
                "lat": float(lat),
                "lon": float(lon),
            }
        )
    candidates = sorted(candidates, key=lambda item: str(item["name"]))
    if not candidates:
        raise ValueError("No valid wave-source candidates available for multi-source selection")

    warnings: List[str] = []
    payload = {
        "k_nearest": k_nearest,
        "source_type": str(source_type),
        "target_sites": [],
        "source_names": [],
        "source_lons": [],
        "source_lats": [],
        "distances_m": [],
        "bearings_deg": [],
        "weights": [],
    }

    for nearshore in nearshore_entries:
        site_name = str(nearshore.get("name", "")).strip()
        site_lat = _try_float(nearshore.get("lat"))
        site_lon = _try_float(nearshore.get("lon"))
        if not site_name or site_lat is None or site_lon is None:
            raise ValueError(f"Nearshore site is missing valid name/lat/lon: {nearshore}")

        ranked = []
        for candidate in candidates:
            distance_m = great_circle_distance_m(
                site_lat, site_lon, candidate["lat"], candidate["lon"]
            )
            bearing_deg = initial_bearing_deg(
                site_lat, site_lon, candidate["lat"], candidate["lon"]
            )
            ranked.append(
                {
                    "name": candidate["name"],
                    "lat": candidate["lat"],
                    "lon": candidate["lon"],
                    "distance_m": float(distance_m),
                    "bearing_deg": float(bearing_deg % 360.0),
                }
            )
        ranked.sort(key=lambda item: (float(item["distance_m"]), str(item["name"])))
        chosen = ranked[:k_nearest]
        if len({item["name"] for item in chosen}) != len(chosen):
            raise ValueError(f"Duplicate sources selected for site {site_name}: {chosen}")

        if len(chosen) < k_nearest:
            if not allow_padding:
                raise ValueError(
                    f"Site '{site_name}' has only {len(chosen)} valid sources, fewer than k_nearest={k_nearest}"
                )
            while len(chosen) < k_nearest:
                chosen.append(dict(chosen[-1]))

        farthest_m = max(float(item["distance_m"]) for item in chosen)
        if farthest_m > float(max_distance_km_error) * 1000.0 and not allow_padding:
            raise ValueError(
                f"Site '{site_name}' nearest-source mapping exceeds max_distance_km_error: "
                f"{farthest_m / 1000.0:.2f} km > {float(max_distance_km_error):.2f} km"
            )
        if farthest_m > float(max_distance_km_warn) * 1000.0:
            warnings.append(
                f"Site '{site_name}' has source(s) beyond warning distance: {farthest_m / 1000.0:.2f} km"
            )

        distances_m = [float(item["distance_m"]) for item in chosen]
        weights = inverse_distance_weights(distances_m, power=weight_power)

        payload["target_sites"].append(site_name)
        payload["source_names"].append([str(item["name"]) for item in chosen])
        payload["source_lons"].append([float(item["lon"]) for item in chosen])
        payload["source_lats"].append([float(item["lat"]) for item in chosen])
        payload["distances_m"].append(distances_m)
        payload["bearings_deg"].append([float(item["bearing_deg"]) for item in chosen])
        payload["weights"].append([float(weight) for weight in weights])

    return payload, warnings


def save_source_metadata_json(payload: dict, path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        json.dump(payload, fh, indent=2)


def summarize_nearest_distances_km(source_metadata: dict) -> dict:
    distances_m = np.asarray(source_metadata.get("distances_m", []), dtype=np.float64)
    if distances_m.ndim != 2 or distances_m.size == 0:
        return {"min_km": float("nan"), "median_km": float("nan"), "max_km": float("nan")}
    nearest_km = distances_m[:, 0] / 1000.0
    return {
        "min_km": float(np.nanmin(nearest_km)),
        "median_km": float(np.nanmedian(nearest_km)),
        "max_km": float(np.nanmax(nearest_km)),
    }
