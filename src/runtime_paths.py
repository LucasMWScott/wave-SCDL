"""Helpers for resolving runtime config paths across scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parent.parent


def _anchor_dir(anchor: str | Path | None) -> Path | None:
    if anchor is None:
        return None
    anchor_path = Path(anchor).expanduser().resolve()
    if anchor_path.exists():
        return anchor_path.parent if anchor_path.is_file() else anchor_path
    return anchor_path.parent if anchor_path.suffix else anchor_path


def _dedupe_paths(paths: Iterable[Path]) -> list[Path]:
    seen: set[str] = set()
    resolved: list[Path] = []
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        resolved.append(path)
    return resolved


def _get_nested(mapping: dict[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = mapping
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def _set_nested(mapping: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    current = mapping
    for key in path[:-1]:
        next_value = current.get(key)
        if not isinstance(next_value, dict):
            next_value = {}
            current[key] = next_value
        current = next_value
    current[path[-1]] = value


def resolve_input_path(
    path_like: str | Path | None, *, anchor: str | Path | None = None
) -> str | None:
    """Resolve a read-path from config, supporting config-relative and repo-relative styles."""
    if path_like in (None, ""):
        return None

    raw = str(path_like)
    path = Path(raw).expanduser()
    if path.is_absolute():
        return str(path.resolve())

    base_dir = _anchor_dir(anchor)
    candidates = _dedupe_paths(
        candidate
        for candidate in (
            (base_dir / path).resolve() if base_dir is not None else None,
            (Path.cwd() / path).resolve(),
            (REPO_ROOT / path).resolve(),
        )
        if candidate is not None
    )
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    if base_dir is not None:
        return str((base_dir / path).resolve())
    return str((REPO_ROOT / path).resolve())


def resolve_output_path(
    path_like: str | Path | None, *, anchor: str | Path | None = None
) -> str | None:
    """Resolve a write-path from config.

    Plain relative paths stay repo-root-relative to preserve the project's
    historical behavior. Use `./...` or `../...` to opt into config-relative
    output locations.
    """
    if path_like in (None, ""):
        return None

    raw = str(path_like)
    path = Path(raw).expanduser()
    if path.is_absolute():
        return str(path.resolve())

    base_dir = _anchor_dir(anchor)
    if raw.startswith("./") or raw.startswith("../"):
        if base_dir is not None:
            return str((base_dir / path).resolve())
        return str((Path.cwd() / path).resolve())

    cwd_candidate = (Path.cwd() / path).resolve()
    if cwd_candidate.exists():
        return str(cwd_candidate)

    repo_candidate = (REPO_ROOT / path).resolve()
    if repo_candidate.exists():
        return str(repo_candidate)

    return str(repo_candidate)


def normalize_runtime_config_paths(
    config: dict[str, Any], *, config_path: str | Path | None = None
) -> dict[str, Any]:
    """Normalize key runtime paths used by training/evaluation."""
    read_keys = (
        ("data", "sites_config"),
        ("data", "static_features_csv"),
        ("data", "static_ablation_config"),
        ("data", "point_centric_dir"),
    )
    write_keys = (("logging", "output_dir"),)

    for key_path in read_keys:
        current = _get_nested(config, key_path)
        resolved = resolve_input_path(current, anchor=config_path)
        if resolved is not None:
            _set_nested(config, key_path, resolved)

    for key_path in write_keys:
        current = _get_nested(config, key_path)
        resolved = resolve_output_path(current, anchor=config_path)
        if resolved is not None:
            _set_nested(config, key_path, resolved)

    return config


__all__ = [
    "REPO_ROOT",
    "normalize_runtime_config_paths",
    "resolve_input_path",
    "resolve_output_path",
]
