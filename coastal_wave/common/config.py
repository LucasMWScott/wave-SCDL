"""Read YAML with explicit, deterministic config-relative paths."""

from pathlib import Path
from typing import Any
import copy
import yaml

PATH_KEYS = {
    "nora3_spectra_dir",
    "norac_spectra_dir",
    "stats_dir",
    "out_dir",
    "storage",
    "checkpoint_path",
    "source_base_config",
    "source_tuning_config",
    "runtime_static_feature_names_source",
    "runtime_static_array_source",
    "results_root",
    "canonical_training_template",
    "canonical_ablation_template",
    "baseline_registry",
    "source_normalization_metadata",
    "sites_config",
    "sites_yaml",
    "static_features_csv",
    "static_ablation_config",
    "point_centric_dir",
    "output_dir",
    "processed_dir",
    "nora3_params_dir",
    "norac_params_dir",
    "full_grid_path",
    "composite_path",
    "raw_dir",
    "master_out_path",
    "out_path",
    "qa_out_path",
    "checkpoint",
    "training_config",
    "preprocess_config",
    "bathymetry_patches",
    "config",
    "base_config",
    "tuning_config",
    "output_root",
    "results_dir",
}


def resolve_paths(value: Any, base: Path, key: str = "") -> Any:
    if isinstance(value, dict):
        return {k: resolve_paths(v, base, k) for k, v in value.items()}
    if isinstance(value, list):
        return [resolve_paths(v, base, key) for v in value]
    if isinstance(value, str) and value and key in PATH_KEYS and "://" not in value:
        path = Path(value).expanduser()
        return str(path.resolve() if path.is_absolute() else (base / path).resolve())
    return value


def _merge(base: dict, override: dict) -> dict:
    result = copy.deepcopy(base)
    for key, value in override.items():
        result[key] = (
            _merge(result[key], value)
            if isinstance(value, dict) and isinstance(result.get(key), dict)
            else copy.deepcopy(value)
        )
    return result


def read_config(path: str | Path, *, _stack: tuple[str, ...] = ()) -> dict:
    path = Path(path).expanduser().resolve()
    if str(path) in _stack:
        raise ValueError(f"Recursive YAML extends cycle: {path.name}")
    if not path.is_file():
        raise FileNotFoundError(f"Missing YAML config: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML mapping in {path}")
    parent = payload.pop("extends", None)
    if payload.get("path_base") == "config":
        payload = resolve_paths(payload, path.parent)
    if parent:
        parent_path = Path(parent)
        if not parent_path.is_absolute():
            parent_path = path.parent / parent_path
        payload = _merge(read_config(parent_path, _stack=(*_stack, str(path))), payload)
    return payload
