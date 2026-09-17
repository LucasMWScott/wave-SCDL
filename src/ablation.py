"""Helpers for static-feature ablation configuration and matching."""

from __future__ import annotations

from fnmatch import fnmatchcase
from pathlib import Path
from typing import Mapping, Sequence


try:
    from config_loader import read_yaml_config
except Exception:
    from src.config_loader import read_yaml_config


from importlib.resources import files

DEFAULT_ABLATION_CONFIG_PATH = str(
    files("coastal_wave.resources").joinpath("default_ablation.yaml")
)
DEFAULT_VALIDATION_IGNORED_GROUPS = ("site_metadata",)
DEFAULT_ALWAYS_RETAINED_RAW_STATIC_FEATURES = ("ray_count",)


def load_ablation_config(path: str | Path = DEFAULT_ABLATION_CONFIG_PATH) -> dict:
    """Load ablation config with safe defaults when missing."""
    config_path = Path(path)
    payload: dict = {}
    if config_path.exists():
        payload = read_yaml_config(config_path)

    feature_groups = payload.get("feature_groups", {}) or {}
    validation_cfg = payload.get("validation", {}) or {}
    normalized = {
        "path": str(config_path),
        "exists": bool(config_path.exists()),
        "enabled": bool(payload.get("enabled", False)),
        "dropped_static_features": [
            str(item).strip()
            for item in (payload.get("dropped_static_features") or [])
            if str(item).strip()
        ],
        "feature_groups": {
            "drop_groups": [
                str(item).strip()
                for item in (feature_groups.get("drop_groups") or [])
                if str(item).strip()
            ],
        },
        "group_definitions": {
            str(name): [str(item).strip() for item in (patterns or []) if str(item).strip()]
            for name, patterns in (payload.get("group_definitions") or {}).items()
        },
        "validation": {
            "enabled": bool(validation_cfg.get("enabled", False)),
            "ignored_groups": [
                str(item).strip()
                for item in (validation_cfg.get("ignored_groups") or [])
                if str(item).strip()
            ],
            "always_retained_raw_features": [
                str(item).strip()
                for item in (validation_cfg.get("always_retained_raw_features") or [])
                if str(item).strip()
            ],
        },
    }
    return normalized


def _match_patterns(feature_names: Sequence[str], pattern: str) -> list[str]:
    names = [str(name) for name in feature_names]
    if any(token in pattern for token in ("*", "?", "[")):
        return [name for name in names if fnmatchcase(name, pattern)]
    return [name for name in names if name == pattern]


def resolve_raw_static_ablation(
    feature_names: Sequence[str], ablation_cfg: Mapping[str, object]
) -> dict:
    """Resolve raw static feature names to drop from config requests and groups."""
    raw_names = [str(name) for name in feature_names]
    enabled = bool(ablation_cfg.get("enabled", False))
    requested_features = [str(item) for item in (ablation_cfg.get("dropped_static_features") or [])]
    feature_groups = (
        (ablation_cfg.get("feature_groups") or {}) if isinstance(ablation_cfg, Mapping) else {}
    )
    drop_groups = [str(item) for item in (feature_groups.get("drop_groups") or [])]
    group_definitions = {
        str(name): [str(item) for item in (patterns or [])]
        for name, patterns in (
            (ablation_cfg.get("group_definitions") or {})
            if isinstance(ablation_cfg, Mapping)
            else {}
        ).items()
    }

    warnings: list[str] = []
    requested_entries: list[str] = []
    matched_raw: list[str] = []
    seen = set()

    def _append_matches(request_label: str, patterns: Sequence[str]) -> None:
        local_matches: list[str] = []
        for pattern in patterns:
            pattern_matches = _match_patterns(raw_names, str(pattern))
            if not pattern_matches:
                warnings.append(f"requested drop feature not found: {pattern}")
                continue
            local_matches.extend(pattern_matches)
        if not local_matches:
            return
        requested_entries.append(request_label)
        for feature_name in local_matches:
            if feature_name not in seen:
                matched_raw.append(feature_name)
                seen.add(feature_name)

    for feature_name in requested_features:
        _append_matches(feature_name, [feature_name])

    for group_name in drop_groups:
        patterns = group_definitions.get(group_name)
        if not patterns:
            warnings.append(f"requested drop group not found: {group_name}")
            continue
        _append_matches(f"group:{group_name}", patterns)

    return {
        "config_path": str(ablation_cfg.get("path", DEFAULT_ABLATION_CONFIG_PATH)),
        "config_exists": bool(ablation_cfg.get("exists", False)),
        "enabled": enabled,
        "raw_feature_count_before": len(raw_names),
        "requested_entries": requested_entries,
        "requested_raw_features": requested_features,
        "requested_drop_groups": drop_groups,
        "matched_raw_features": matched_raw,
        "warnings": warnings,
    }


def resolve_transformed_static_ablation(
    transformed_feature_names: Sequence[str],
    raw_to_transformed_map: Mapping[str, Sequence[str]] | None,
    ablation_cfg: Mapping[str, object],
) -> dict:
    """Resolve runtime transformed feature columns to drop using raw feature requests."""
    if not raw_to_transformed_map:
        return {
            "config_path": str(ablation_cfg.get("path", DEFAULT_ABLATION_CONFIG_PATH)),
            "config_exists": bool(ablation_cfg.get("exists", False)),
            "enabled": bool(ablation_cfg.get("enabled", False)),
            "raw_feature_count_before": 0,
            "requested_entries": [],
            "requested_raw_features": [],
            "requested_drop_groups": [],
            "matched_raw_features": [],
            "matched_transformed_features": [],
            "warnings": [
                "runtime ablation skipped: raw_to_transformed_feature_map missing from metadata"
            ],
        }

    raw_resolution = resolve_raw_static_ablation(list(raw_to_transformed_map.keys()), ablation_cfg)
    transformed_names = [str(name) for name in transformed_feature_names]
    matched_transformed: list[str] = []
    seen = set()
    for raw_name in raw_resolution["matched_raw_features"]:
        for transformed_name in raw_to_transformed_map.get(str(raw_name), []):
            if transformed_name in transformed_names and transformed_name not in seen:
                matched_transformed.append(str(transformed_name))
                seen.add(str(transformed_name))

    raw_resolution["matched_transformed_features"] = matched_transformed
    return raw_resolution


def build_transformed_static_group_manifest(
    transformed_feature_names: Sequence[str],
    raw_to_transformed_map: Mapping[str, Sequence[str]] | None,
    ablation_cfg: Mapping[str, object],
) -> dict:
    """Resolve group definitions against runtime transformed static features."""
    transformed_names = [str(name) for name in transformed_feature_names]
    validation_cfg = (
        (ablation_cfg.get("validation") or {}) if isinstance(ablation_cfg, Mapping) else {}
    )
    ignored_groups = {
        str(name)
        for name in (validation_cfg.get("ignored_groups") or DEFAULT_VALIDATION_IGNORED_GROUPS)
        if str(name).strip()
    }
    always_retained_raw = [
        str(name).strip()
        for name in (
            validation_cfg.get("always_retained_raw_features")
            or DEFAULT_ALWAYS_RETAINED_RAW_STATIC_FEATURES
        )
        if str(name).strip()
    ]
    raw_map = {
        str(raw_name): [str(col) for col in (mapped or [])]
        for raw_name, mapped in (raw_to_transformed_map or {}).items()
    }
    group_definitions = {
        str(name): [str(item).strip() for item in (patterns or []) if str(item).strip()]
        for name, patterns in (
            (ablation_cfg.get("group_definitions") or {})
            if isinstance(ablation_cfg, Mapping)
            else {}
        ).items()
    }

    groups: dict[str, dict[str, object]] = {}
    transformed_to_groups: dict[str, list[str]] = {}
    unmatched_patterns: list[dict[str, str]] = []
    descendant_leaks: list[dict[str, object]] = []

    for group_name, patterns in group_definitions.items():
        if group_name in ignored_groups:
            continue

        matched_raw: list[str] = []
        matched_raw_seen = set()
        group_unmatched_patterns: list[str] = []

        for pattern in patterns:
            pattern_matches = _match_patterns(list(raw_map.keys()), str(pattern))
            if not pattern_matches:
                group_unmatched_patterns.append(str(pattern))
                unmatched_patterns.append({"group": group_name, "pattern": str(pattern)})
                continue
            for raw_name in pattern_matches:
                if raw_name not in matched_raw_seen:
                    matched_raw.append(raw_name)
                    matched_raw_seen.add(raw_name)

        matched_transformed: list[str] = []
        matched_transformed_seen = set()
        for raw_name in matched_raw:
            descendants = [name for name in raw_map.get(raw_name, []) if name in transformed_names]
            if not descendants:
                group_unmatched_patterns.append(raw_name)
            for transformed_name in descendants:
                if transformed_name not in matched_transformed_seen:
                    matched_transformed.append(transformed_name)
                    matched_transformed_seen.add(transformed_name)
                    transformed_to_groups.setdefault(transformed_name, []).append(group_name)
            if descendants and any(name not in matched_transformed_seen for name in descendants):
                descendant_leaks.append(
                    {
                        "group": group_name,
                        "raw_feature": raw_name,
                        "expected_transformed_features": descendants,
                    }
                )

        groups[group_name] = {
            "patterns": list(patterns),
            "matched_raw_features": matched_raw,
            "matched_transformed_features": matched_transformed,
            "raw_feature_count": len(matched_raw),
            "transformed_feature_count": len(matched_transformed),
            "unmatched_patterns": group_unmatched_patterns,
        }

    overlapping_transformed = {
        name: owners for name, owners in transformed_to_groups.items() if len(set(owners)) > 1
    }

    always_retained_raw_present = [name for name in always_retained_raw if name in raw_map]
    always_retained_transformed: list[str] = []
    always_retained_seen = set()
    for raw_name in always_retained_raw_present:
        for transformed_name in raw_map.get(raw_name, []):
            if (
                transformed_name in transformed_names
                and transformed_name not in always_retained_seen
            ):
                always_retained_transformed.append(transformed_name)
                always_retained_seen.add(transformed_name)

    assigned_transformed = set(transformed_to_groups.keys())
    unassigned_transformed = [
        name
        for name in transformed_names
        if name not in assigned_transformed and name not in always_retained_seen
    ]

    return {
        "config_path": str(ablation_cfg.get("path", DEFAULT_ABLATION_CONFIG_PATH)),
        "transformed_feature_count": len(transformed_names),
        "raw_feature_count": len(raw_map),
        "ignored_groups": sorted(ignored_groups),
        "always_retained_raw_features": always_retained_raw_present,
        "always_retained_transformed_features": always_retained_transformed,
        "groups": groups,
        "group_order": [name for name in group_definitions if name not in ignored_groups],
        "unmatched_patterns": unmatched_patterns,
        "overlapping_transformed_features": overlapping_transformed,
        "unassigned_transformed_features": unassigned_transformed,
        "descendant_leaks": descendant_leaks,
    }


def validate_transformed_static_group_manifest(
    manifest: Mapping[str, object],
    *,
    required_groups: Sequence[str] | None = None,
) -> None:
    """Raise when strict physical-group validation fails."""
    groups = (manifest.get("groups") or {}) if isinstance(manifest, Mapping) else {}
    group_order = [str(name) for name in (required_groups or manifest.get("group_order") or [])]
    errors: list[str] = []

    for group_name in group_order:
        group_payload = (groups.get(group_name) or {}) if isinstance(groups, Mapping) else {}
        transformed_count = int(group_payload.get("transformed_feature_count", 0) or 0)
        if transformed_count <= 0:
            errors.append(f"group '{group_name}' resolves to zero transformed features")
        for pattern in group_payload.get("unmatched_patterns") or []:
            errors.append(f"group '{group_name}' pattern matched no runtime features: {pattern}")

    overlaps = manifest.get("overlapping_transformed_features") or {}
    if overlaps:
        preview = {str(name): list(owners) for name, owners in list(overlaps.items())[:10]}
        errors.append(f"overlapping transformed feature assignments detected: {preview}")

    unassigned = [str(name) for name in (manifest.get("unassigned_transformed_features") or [])]
    if unassigned:
        errors.append(f"unassigned transformed features remain: {unassigned[:20]}")

    descendant_leaks = manifest.get("descendant_leaks") or []
    if descendant_leaks:
        errors.append(
            f"raw angular feature descendants were not fully captured: {descendant_leaks[:10]}"
        )

    if errors:
        raise ValueError("Static ablation group validation failed:\n- " + "\n- ".join(errors))


def format_ablation_summary(summary: Mapping[str, object], max_items: int = 20) -> str:
    """Format a concise ablation summary line for logs."""
    matched = [
        str(item)
        for item in (
            summary.get("matched_transformed_features") or summary.get("matched_raw_features") or []
        )
    ]
    before = int(summary.get("raw_feature_count_before", 0))
    after = before - len(matched)
    preview = matched[:max_items]
    if len(matched) > max_items:
        preview_txt = f"{preview} ... and {len(matched) - max_items} more"
    else:
        preview_txt = str(preview)
    return (
        f"Ablation: enabled={bool(summary.get('enabled', False))} "
        f"config={summary.get('config_path', DEFAULT_ABLATION_CONFIG_PATH)}\n"
        f"Static features: before={before} dropped={len(matched)} after={after}\n"
        f"Dropped preview: {preview_txt}"
    )
