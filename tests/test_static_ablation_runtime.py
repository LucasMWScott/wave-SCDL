"""Test static ablation runtime."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.ablation import (
    load_ablation_config,
    build_transformed_static_group_manifest,
    validate_transformed_static_group_manifest,
)
from src.data_pipeline import (
    PointCentricArrays,
    apply_runtime_static_ablation,
    resolve_static_ablation_config_path,
)
from src.runtime_paths import REPO_ROOT, normalize_runtime_config_paths


class StaticAblationRuntimeTests(unittest.TestCase):
    def _build_arrays(self) -> PointCentricArrays:
        metadata = {
            "normalization": {
                "static_scaler": {
                    "raw_to_transformed_feature_map_before_ablation": {
                        "route_feature": ["route_feature"],
                        "porosity_feature": ["porosity_feature"],
                        "other_feature": ["other_feature"],
                    }
                }
            }
        }
        return PointCentricArrays(
            x_dynamic=np.zeros((2, 1), dtype=np.float32),
            x_dynamic_sources=None,
            x_dynamic_sitewise={},
            source_geometry=None,
            y_targets={"site_a": np.zeros((2, 6), dtype=np.float32)},
            y_physical={"site_a": np.zeros((2, 4), dtype=np.float32)},
            y_transfer={},
            y_reference={},
            x_static={"site_a": np.array([1.0, 2.0, 3.0], dtype=np.float32)},
            x_bathy=None,
            local_depth_m=None,
            local_breaking_hs_cap=None,
            local_breaking_cap_valid=None,
            target_sites=["site_a"],
            target_mode="physical",
            dynamic_feature_names=["dyn_0"],
            source_feature_names=[],
            site_dynamic_feature_names=[],
            source_geometry_feature_names=[],
            target_feature_names=["hs", "tp", "dir_sin", "dir_cos", "dp_sin", "dp_cos"],
            physical_target_names=["hs", "tp", "dir", "dp"],
            transfer_target_names=[],
            reference_target_names=[],
            static_feature_names=["route_feature", "porosity_feature", "other_feature"],
            bathy_channel_names=[],
            bathy_site_to_index={},
            bathy_patch_size=None,
            bathy_resolution_m=None,
            bathy_normalization_metadata={},
            timestamps=np.array(["2020-01-01T00:00:00", "2020-01-01T01:00:00"], dtype=str),
            split_idx={
                "train": np.array([0], dtype=int),
                "val": np.array([1], dtype=int),
                "test": np.array([], dtype=int),
            },
            metadata=metadata,
            ablation_summary=None,
        )

    def test_config_relative_ablation_file_is_resolved_and_full_static_baseline_keeps_all_features(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(dir=REPO_ROOT, prefix="tmp_static_ablation_") as tmpdir:
            base = Path(tmpdir).resolve()
            rel_base = base.relative_to(REPO_ROOT)
            config_dir = base / "configs" / "ablations" / "features"
            config_dir.mkdir(parents=True, exist_ok=True)
            ablation_path = base / "shared" / "ablation_disabled.yaml"
            ablation_path.parent.mkdir(parents=True, exist_ok=True)
            ablation_path.write_text("enabled: false\n", encoding="utf-8")

            config = {
                "data": {
                    "static_ablation_config": str(rel_base / "shared" / "ablation_disabled.yaml"),
                }
            }
            resolved = normalize_runtime_config_paths(
                config,
                config_path=config_dir / "trans_static_cross_full_static.yaml",
            )
            arrays = apply_runtime_static_ablation(
                self._build_arrays(),
                resolve_static_ablation_config_path(resolved),
            )

            self.assertEqual(
                arrays.static_feature_names, ["route_feature", "porosity_feature", "other_feature"]
            )
            self.assertEqual(arrays.ablation_summary["config_path"], str(ablation_path.resolve()))
            self.assertFalse(arrays.ablation_summary["enabled"])

    def test_group_drop_config_removes_matching_transformed_features(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO_ROOT, prefix="tmp_static_ablation_") as tmpdir:
            base = Path(tmpdir).resolve()
            ablation_path = base / "drop_route.yaml"
            ablation_path.write_text(
                "\n".join(
                    [
                        "enabled: true",
                        "feature_groups:",
                        "  drop_groups:",
                        "    - routing_core",
                        "group_definitions:",
                        "  routing_core:",
                        "    - route_feature",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            arrays = apply_runtime_static_ablation(self._build_arrays(), str(ablation_path))

            self.assertEqual(arrays.static_feature_names, ["porosity_feature", "other_feature"])
            self.assertEqual(arrays.x_static["site_a"].tolist(), [2.0, 3.0])
            self.assertEqual(
                arrays.ablation_summary["matched_transformed_features"], ["route_feature"]
            )

    def test_raw_angular_feature_drop_removes_all_transformed_descendants(self) -> None:
        metadata = {
            "normalization": {
                "static_scaler": {
                    "raw_to_transformed_feature_map_before_ablation": {
                        "static_heading_deg": ["static_heading_deg_sin", "static_heading_deg_cos"],
                        "ray_count": ["ray_count"],
                    }
                }
            }
        }
        arrays = PointCentricArrays(
            x_dynamic=np.zeros((2, 1), dtype=np.float32),
            x_dynamic_sources=None,
            x_dynamic_sitewise={},
            source_geometry=None,
            y_targets={"site_a": np.zeros((2, 6), dtype=np.float32)},
            y_physical={"site_a": np.zeros((2, 4), dtype=np.float32)},
            y_transfer={},
            y_reference={},
            x_static={"site_a": np.array([1.0, 2.0, 3.0], dtype=np.float32)},
            x_bathy=None,
            local_depth_m=None,
            local_breaking_hs_cap=None,
            local_breaking_cap_valid=None,
            target_sites=["site_a"],
            target_mode="physical",
            dynamic_feature_names=["dyn_0"],
            source_feature_names=[],
            site_dynamic_feature_names=[],
            source_geometry_feature_names=[],
            target_feature_names=["hs", "tp", "dir_sin", "dir_cos", "dp_sin", "dp_cos"],
            physical_target_names=["hs", "tp", "dir", "dp"],
            transfer_target_names=[],
            reference_target_names=[],
            static_feature_names=["static_heading_deg_sin", "static_heading_deg_cos", "ray_count"],
            bathy_channel_names=[],
            bathy_site_to_index={},
            bathy_patch_size=None,
            bathy_resolution_m=None,
            bathy_normalization_metadata={},
            timestamps=np.array(["2020-01-01T00:00:00", "2020-01-01T01:00:00"], dtype=str),
            split_idx={
                "train": np.array([0], dtype=int),
                "val": np.array([1], dtype=int),
                "test": np.array([], dtype=int),
            },
            metadata=metadata,
            ablation_summary=None,
        )

        with tempfile.TemporaryDirectory(dir=REPO_ROOT, prefix="tmp_static_ablation_") as tmpdir:
            ablation_path = Path(tmpdir).resolve() / "drop_heading.yaml"
            ablation_path.write_text(
                "\n".join(
                    [
                        "enabled: true",
                        "feature_groups:",
                        "  drop_groups:",
                        "    - heading",
                        "group_definitions:",
                        "  heading:",
                        "    - static_heading_deg",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            resolved = apply_runtime_static_ablation(arrays, str(ablation_path))

        self.assertEqual(resolved.static_feature_names, ["ray_count"])
        self.assertEqual(
            resolved.ablation_summary["matched_transformed_features"],
            ["static_heading_deg_sin", "static_heading_deg_cos"],
        )

    def test_strict_group_manifest_validation_allows_ray_count_only_when_allowlisted(self) -> None:
        manifest = build_transformed_static_group_manifest(
            transformed_feature_names=["group_a", "ray_count"],
            raw_to_transformed_map={
                "group_a": ["group_a"],
                "ray_count": ["ray_count"],
            },
            ablation_cfg={
                "path": "inline",
                "group_definitions": {"group_one": ["group_a"], "site_metadata": ["site_lat"]},
                "validation": {
                    "enabled": True,
                    "ignored_groups": ["site_metadata"],
                    "always_retained_raw_features": ["ray_count"],
                },
            },
        )
        validate_transformed_static_group_manifest(manifest)
        self.assertEqual(manifest["always_retained_transformed_features"], ["ray_count"])
        self.assertEqual(manifest["unassigned_transformed_features"], [])

    def test_strict_group_manifest_validation_rejects_overlaps_and_unassigned_features(
        self,
    ) -> None:
        overlap_manifest = build_transformed_static_group_manifest(
            transformed_feature_names=["shared_feature"],
            raw_to_transformed_map={"shared_feature": ["shared_feature"]},
            ablation_cfg={
                "path": "inline",
                "group_definitions": {
                    "group_one": ["shared_feature"],
                    "group_two": ["shared_feature"],
                },
                "validation": {"enabled": True},
            },
        )
        with self.assertRaisesRegex(ValueError, "overlapping transformed feature assignments"):
            validate_transformed_static_group_manifest(overlap_manifest)

        unassigned_manifest = build_transformed_static_group_manifest(
            transformed_feature_names=["assigned_feature", "leftover_feature"],
            raw_to_transformed_map={
                "assigned_feature": ["assigned_feature"],
                "leftover_feature": ["leftover_feature"],
            },
            ablation_cfg={
                "path": "inline",
                "group_definitions": {"group_one": ["assigned_feature"]},
                "validation": {"enabled": True},
            },
        )
        with self.assertRaisesRegex(ValueError, "unassigned transformed features remain"):
            validate_transformed_static_group_manifest(unassigned_manifest)

    def test_inherited_ablation_config_loads_via_extends(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO_ROOT, prefix="tmp_static_ablation_") as tmpdir:
            base = Path(tmpdir).resolve()
            parent_path = base / "full_static.yaml"
            parent_path.write_text(
                "\n".join(
                    [
                        "enabled: false",
                        "feature_groups:",
                        "  drop_groups: []",
                        "validation:",
                        "  enabled: true",
                        "  always_retained_raw_features:",
                        "    - ray_count",
                        "group_definitions:",
                        "  route:",
                        "    - route_feature",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            child_dir = base / "drop_route"
            child_dir.mkdir(parents=True, exist_ok=True)
            child_path = child_dir / "drop_route.yaml"
            child_path.write_text(
                "\n".join(
                    [
                        "extends: ../full_static.yaml",
                        "enabled: true",
                        "feature_groups:",
                        "  drop_groups:",
                        "    - route",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            loaded = load_ablation_config(child_path)

        self.assertTrue(loaded["enabled"])
        self.assertEqual(loaded["feature_groups"]["drop_groups"], ["route"])
        self.assertTrue(loaded["validation"]["enabled"])
        self.assertEqual(loaded["validation"]["always_retained_raw_features"], ["ray_count"])
        self.assertEqual(loaded["group_definitions"]["route"], ["route_feature"])

    def test_strict_group_manifest_validation_rejects_zero_match_group(self) -> None:
        manifest = build_transformed_static_group_manifest(
            transformed_feature_names=["route_feature"],
            raw_to_transformed_map={"route_feature": ["route_feature"]},
            ablation_cfg={
                "path": "inline",
                "group_definitions": {"missing_group": ["no_such_feature"]},
                "validation": {"enabled": True},
            },
        )
        with self.assertRaisesRegex(ValueError, "resolves to zero transformed features"):
            validate_transformed_static_group_manifest(manifest)


if __name__ == "__main__":
    unittest.main()
