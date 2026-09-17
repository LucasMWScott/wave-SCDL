"""Helpers for sequential batch training plans."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

try:
    from runtime_paths import REPO_ROOT
except Exception:
    from src.runtime_paths import REPO_ROOT


def read_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _candidate_case_paths(case_ref: str, batch_config_path: str | Path) -> list[Path]:
    batch_path = Path(batch_config_path).expanduser().resolve()
    batch_dir = batch_path.parent
    raw_path = Path(case_ref).expanduser()
    variants = [raw_path]
    if raw_path.suffix == "":
        variants.append(raw_path.with_suffix(".yaml"))

    candidates: list[Path] = []
    for variant in variants:
        if variant.is_absolute():
            candidates.append(variant.resolve())
            continue
        candidates.append((batch_dir / variant).resolve())
        candidates.append((REPO_ROOT / variant).resolve())
    return candidates


def resolve_case_config_path(case_ref: str, batch_config_path: str | Path) -> Path:
    for candidate in _candidate_case_paths(case_ref, batch_config_path):
        if candidate.exists():
            return candidate
    candidates = _candidate_case_paths(case_ref, batch_config_path)
    raise FileNotFoundError(
        f"Could not resolve training case '{case_ref}' from batch config '{Path(batch_config_path).resolve()}'. "
        f"Tried: {[str(path) for path in candidates]}"
    )


def load_batch_training_plan(batch_config_path: str | Path) -> tuple[dict[str, Any], list[Path]]:
    payload = read_yaml(batch_config_path)
    raw_cases = payload.get("cases", [])
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("Batch config must contain a non-empty 'cases' list.")

    resolved_cases: list[Path] = []
    for entry in raw_cases:
        enabled = True
        case_ref: str | None = None
        if isinstance(entry, str):
            case_ref = entry
        elif isinstance(entry, dict):
            enabled = bool(entry.get("enabled", True))
            case_ref = entry.get("config") or entry.get("path") or entry.get("case")
        else:
            raise ValueError("Each batch case entry must be a string or a mapping.")

        if not enabled:
            continue
        if not case_ref or not str(case_ref).strip():
            raise ValueError("Enabled batch case entries must include a non-empty config path.")
        resolved_cases.append(resolve_case_config_path(str(case_ref).strip(), batch_config_path))

    if not resolved_cases:
        raise ValueError("Batch config did not resolve any enabled training cases.")

    return payload, resolved_cases


__all__ = [
    "load_batch_training_plan",
    "read_yaml",
    "resolve_case_config_path",
]
