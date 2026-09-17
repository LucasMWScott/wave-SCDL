"""Shared YAML config loading with optional recursive inheritance."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping

import yaml

try:
    from runtime_paths import REPO_ROOT
except Exception:
    from src.runtime_paths import REPO_ROOT


def _deep_merge(base: Any, override: Any) -> Any:
    if isinstance(base, Mapping) and isinstance(override, Mapping):
        merged = {str(key): copy.deepcopy(value) for key, value in base.items()}
        for key, value in override.items():
            if key in merged:
                merged[str(key)] = _deep_merge(merged[str(key)], value)
            else:
                merged[str(key)] = copy.deepcopy(value)
        return merged
    return copy.deepcopy(override)


def _resolve_extends_path(path_like: str | Path, *, anchor: str | Path) -> Path:
    raw_path = Path(path_like).expanduser()
    if raw_path.is_absolute():
        return raw_path.resolve()

    anchor_path = Path(anchor).expanduser().resolve()
    anchor_dir = anchor_path.parent if anchor_path.is_file() else anchor_path
    candidates = [
        (anchor_dir / raw_path).resolve(),
        (Path.cwd() / raw_path).resolve(),
        (REPO_ROOT / raw_path).resolve(),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def read_yaml_config(path: str | Path, *, _stack: tuple[str, ...] = ()) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    key = str(config_path)
    if key in _stack:
        cycle = " -> ".join([*_stack, key])
        raise ValueError(f"Recursive YAML extends cycle detected: {cycle}")

    if not config_path.exists():
        raise FileNotFoundError(f"Missing YAML config: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}

    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML mapping at top level in {config_path}")

    if payload.get("path_base") == "config":
        from coastal_wave.common.config import read_config

        return read_config(config_path)

    extends_value = payload.pop("extends", None)
    if extends_value in (None, ""):
        return payload

    if not isinstance(extends_value, (str, Path)):
        raise ValueError(f"'extends' must be a string path in {config_path}")

    parent_path = _resolve_extends_path(extends_value, anchor=config_path)
    parent_payload = read_yaml_config(parent_path, _stack=(*_stack, key))
    return _deep_merge(parent_payload, payload)


__all__ = ["read_yaml_config"]
