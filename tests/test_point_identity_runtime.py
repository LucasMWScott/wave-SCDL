"""Test point identity runtime."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.data_pipeline import load_point_centric_arrays, resolve_site_split_config
from src.point_centric_pipeline import find_param_files


TARGET_FEATURE_ORDER = ["hs", "tp", "dir_sin", "dir_cos", "dp_sin", "dp_cos"]


class SplitConfigRuntimeTests(unittest.TestCase):
    def test_rejects_legacy_site_holdout_keys(self) -> None:
        all_sites = ["site_b", "site_a", "site_c"]

        with self.assertRaisesRegex(ValueError, "data.holdout_sites"):
            resolve_site_split_config(
                all_sites,
                {
                    "data": {
                        "validation_sites": ["site_a"],
                        "test_sites": ["site_c"],
                        "holdout_sites": ["site_c"],
                    }
                },
            )

        with self.assertRaisesRegex(ValueError, "split.norac_holdout"):
            resolve_site_split_config(
                all_sites,
                {
                    "data": {
                        "validation_sites": ["site_a"],
                        "test_sites": ["site_c"],
                    },
                    "split": {"norac_holdout": ["site_c"]},
                },
            )

    def test_preserves_yaml_site_order_in_site_heldout_mode(self) -> None:
        split_info = resolve_site_split_config(
            ["site_b", "site_a", "site_c", "site_d"],
            {
                "data": {
                    "validation_sites": ["site_a"],
                    "test_sites": ["site_d"],
                }
            },
        )

        self.assertEqual(split_info["train_sites"], ["site_b", "site_c"])
        self.assertEqual(split_info["val_sites"], ["site_a"])
        self.assertEqual(split_info["test_sites"], ["site_d"])

    def test_train_site_subsampling_fraction_one_is_noop(self) -> None:
        split_info = resolve_site_split_config(
            ["site_b", "site_a", "site_c", "site_d"],
            {
                "data": {
                    "validation_sites": ["site_a"],
                    "test_sites": ["site_d"],
                    "train_site_subsampling": {"enabled": True, "fraction": 1.0, "seed": 7},
                }
            },
        )

        self.assertEqual(split_info["train_sites"], ["site_b", "site_c"])
        self.assertFalse(split_info["train_site_subsampling"]["applied"])
        self.assertEqual(split_info["train_site_subsampling"]["before_count"], 2)
        self.assertEqual(split_info["train_site_subsampling"]["after_count"], 2)

    def test_train_site_subsampling_is_deterministic_and_preserves_original_order(self) -> None:
        all_sites = ["site_b", "site_a", "site_c", "site_d", "site_e", "site_f"]
        config = {
            "data": {
                "validation_sites": ["site_a"],
                "test_sites": ["site_f"],
                "train_site_subsampling": {"enabled": True, "fraction": 0.5, "seed": 11},
            }
        }

        split_info_1 = resolve_site_split_config(all_sites, config)
        split_info_2 = resolve_site_split_config(all_sites, config)

        self.assertEqual(split_info_1["train_sites"], split_info_2["train_sites"])
        self.assertEqual(split_info_1["val_sites"], ["site_a"])
        self.assertEqual(split_info_1["test_sites"], ["site_f"])
        original_positions = [all_sites.index(site) for site in split_info_1["train_sites"]]
        self.assertEqual(original_positions, sorted(original_positions))
        self.assertTrue(split_info_1["train_site_subsampling"]["applied"])

    def test_train_site_subsampling_respects_explicit_train_sites(self) -> None:
        split_info = resolve_site_split_config(
            ["site_a", "site_b", "site_c", "site_d", "site_e"],
            {
                "data": {
                    "train_sites": ["site_e", "site_c", "site_b"],
                    "validation_sites": ["site_a"],
                    "test_sites": ["site_d"],
                    "train_site_subsampling": {"enabled": True, "fraction": 0.5, "seed": 3},
                }
            },
        )

        self.assertTrue(set(split_info["train_sites"]).issubset({"site_e", "site_c", "site_b"}))
        self.assertEqual(split_info["val_sites"], ["site_a"])
        self.assertEqual(split_info["test_sites"], ["site_d"])
        self.assertEqual(len(split_info["train_sites"]), 1)

    def test_train_site_subsampling_tiny_fraction_keeps_one_site(self) -> None:
        split_info = resolve_site_split_config(
            ["site_a", "site_b", "site_c"],
            {
                "data": {
                    "train_site_subsampling": {"enabled": True, "fraction": 0.01, "seed": 5},
                }
            },
        )

        self.assertEqual(len(split_info["train_sites"]), 1)
        self.assertEqual(split_info["train_site_subsampling"]["after_count"], 1)

    def test_shared_recent_requires_matching_val_and_test_fractions(self) -> None:
        with self.assertRaisesRegex(ValueError, "split.val == split.test"):
            resolve_site_split_config(
                ["site_a", "site_b", "site_c", "site_d"],
                {
                    "data": {
                        "validation_sites": ["site_a"],
                        "test_sites": ["site_d"],
                    },
                    "split": {
                        "train": 0.6,
                        "val": 0.2,
                        "test": 0.3,
                        "site_holdout_temporal_mode": "shared_recent",
                    },
                },
            )

    def test_shared_recent_requires_train_fraction_to_match_remainder(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "requires split.train to equal 1 - recent_fraction"
        ):
            resolve_site_split_config(
                ["site_a", "site_b", "site_c", "site_d"],
                {
                    "data": {
                        "validation_sites": ["site_a"],
                        "test_sites": ["site_d"],
                    },
                    "split": {
                        "train": 0.7,
                        "val": 0.2,
                        "test": 0.2,
                        "site_holdout_temporal_mode": "shared_recent",
                    },
                },
            )

    def test_shared_recent_resolves_temporal_holdout_metadata(self) -> None:
        split_info = resolve_site_split_config(
            ["site_a", "site_b", "site_c", "site_d"],
            {
                "data": {
                    "validation_sites": ["site_a"],
                    "test_sites": ["site_d"],
                },
                "split": {
                    "train": 0.8,
                    "val": 0.2,
                    "test": 0.2,
                    "site_holdout_temporal_mode": "shared_recent",
                },
            },
        )

        self.assertEqual(split_info["site_holdout_temporal_mode"], "shared_recent")
        self.assertTrue(split_info["site_holdout_temporal_active"])
        self.assertAlmostEqual(float(split_info["site_holdout_temporal_recent_fraction"]), 0.2)
        self.assertAlmostEqual(float(split_info["site_holdout_temporal_train_fraction"]), 0.8)


class ParamFileResolutionTests(unittest.TestCase):
    def test_find_param_files_requires_exact_site_name_match(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            (base / "norac_grid_25_NORAC_wave_params.csv").write_text("time,hs\n2020-01-01,1.0\n")
            (base / "norac_grid_250_NORAC_wave_params.csv").write_text("time,hs\n2020-01-01,2.0\n")

            matches = [Path(path).name for path in find_param_files("norac_grid_25", str(base))]

            self.assertEqual(matches, ["norac_grid_25_NORAC_wave_params.csv"])


class PointCentricLoaderOrderingTests(unittest.TestCase):
    def test_loader_reorders_site_indexed_source_artifacts_by_target_sites(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)

            np.savez_compressed(
                base / "point_centric_X_dynamic.npz",
                X_dynamic=np.zeros((3, 2), dtype=np.float32),
                timestamps=np.array(
                    ["2020-01-01T00:00:00", "2020-01-01T01:00:00", "2020-01-01T02:00:00"], dtype=str
                ),
                feature_names=np.array(["off_a_hs", "off_a_tp"], dtype=str),
                offshore_sites=np.array(["off_a"], dtype=str),
                train_idx=np.array([0, 1], dtype=int),
                val_idx=np.array([2], dtype=int),
                test_idx=np.array([], dtype=int),
            )

            y_payload = {
                "target_sites": np.array(["site_b", "site_a"], dtype=str),
                "target_feature_names": np.array(TARGET_FEATURE_ORDER, dtype=str),
                "timestamps": np.array(
                    ["2020-01-01T00:00:00", "2020-01-01T01:00:00", "2020-01-01T02:00:00"], dtype=str
                ),
                "target_mode": np.array(["physical"], dtype=str),
                "physical_target_names": np.array(["hs", "tp", "dir", "dp"], dtype=str),
                "Y__site_b": np.ones((3, 6), dtype=np.float32),
                "Y__site_a": np.full((3, 6), 2.0, dtype=np.float32),
                "Yphysical__site_b": np.ones((3, 4), dtype=np.float32),
                "Yphysical__site_a": np.full((3, 4), 2.0, dtype=np.float32),
            }
            np.savez_compressed(base / "point_centric_Y_targets.npz", **y_payload)

            np.savez_compressed(
                base / "point_centric_X_dynamic_sources.npz",
                X_dynamic_sources=np.array(
                    [
                        np.full((3, 1, 2), 10.0, dtype=np.float32),
                        np.full((3, 1, 2), 20.0, dtype=np.float32),
                    ],
                    dtype=np.float32,
                ),
                target_sites=np.array(["site_a", "site_b"], dtype=str),
                source_feature_names=np.array(["src_hs", "src_tp"], dtype=str),
                timestamps=np.array(
                    ["2020-01-01T00:00:00", "2020-01-01T01:00:00", "2020-01-01T02:00:00"], dtype=str
                ),
            )

            np.savez_compressed(
                base / "point_centric_source_geometry.npz",
                source_geometry=np.array(
                    [
                        np.full((1, 3), 100.0, dtype=np.float32),
                        np.full((1, 3), 200.0, dtype=np.float32),
                    ],
                    dtype=np.float32,
                ),
                target_sites=np.array(["site_a", "site_b"], dtype=str),
                source_geometry_feature_names=np.array(
                    ["distance_norm", "bearing_sin", "bearing_cos"], dtype=str
                ),
            )

            (base / "point_centric_metadata.json").write_text(
                json.dumps({"dynamic_feature_names": ["off_a_hs", "off_a_tp"]})
            )

            arrays = load_point_centric_arrays(str(base))

            self.assertEqual(arrays.target_sites, ["site_b", "site_a"])
            self.assertEqual(float(arrays.x_dynamic_sources[0, 0, 0, 0]), 20.0)
            self.assertEqual(float(arrays.x_dynamic_sources[1, 0, 0, 0]), 10.0)
            self.assertEqual(float(arrays.source_geometry[0, 0, 0]), 200.0)
            self.assertEqual(float(arrays.source_geometry[1, 0, 0]), 100.0)


if __name__ == "__main__":
    unittest.main()
