"""Test train sample subsampling."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.data_pipeline import build_split_dataloader, load_point_centric_arrays


class TrainSampleSubsamplingTests(unittest.TestCase):
    def _write_arrays(self, base: Path) -> None:
        timestamps = np.array([f"2020-01-01T0{i}:00:00" for i in range(8)], dtype=str)
        target_sites = np.array(["train_a", "train_b", "val_site", "test_site"], dtype=str)
        train_idx = np.array([0, 1, 2, 3, 4], dtype=int)
        val_idx = np.array([5, 6], dtype=int)
        test_idx = np.array([7], dtype=int)

        np.savez_compressed(
            base / "point_centric_X_dynamic.npz",
            X_dynamic=np.arange(8 * 3, dtype=np.float32).reshape(8, 3),
            timestamps=timestamps,
            feature_names=np.array(["dyn_0", "dyn_1", "dyn_2"], dtype=str),
            offshore_sites=np.array(["nora3_grid_1"], dtype=str),
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
        )

        encoded_template = np.tile(
            np.array([[0.2, 5.0, 0.0, 1.0, 0.0, 1.0]], dtype=np.float32),
            (8, 1),
        )
        y_payload: dict[str, np.ndarray] = {
            "target_sites": target_sites,
            "target_feature_names": np.array(
                ["hs", "tp", "dir_sin", "dir_cos", "dp_sin", "dp_cos"], dtype=str
            ),
            "timestamps": timestamps,
            "target_mode": np.array(["physical"], dtype=str),
            "physical_target_names": np.array(["hs", "tp", "dir", "dp"], dtype=str),
        }
        site_offsets = {
            "train_a": 0.0,
            "train_b": 1.0,
            "val_site": 2.0,
            "test_site": 3.0,
        }
        for site, offset in site_offsets.items():
            safe = site
            y_payload[f"Y__{safe}"] = encoded_template.copy()
            y_payload[f"Yphysical__{safe}"] = np.array(
                [
                    [1.0 + offset, 5.0, 10.0, 20.0],
                    [1.1 + offset, 5.1, 10.0, 20.0],
                    [1.2 + offset, 5.2, 10.0, 20.0],
                    [1.3 + offset, 5.3, 10.0, 20.0],
                    [1.4 + offset, 5.4, 10.0, 20.0],
                    [1.5 + offset, 5.5, 10.0, 20.0],
                    [1.6 + offset, 5.6, 10.0, 20.0],
                    [1.7 + offset, 5.7, 10.0, 20.0],
                ],
                dtype=np.float32,
            )
        np.savez_compressed(base / "point_centric_Y_targets.npz", **y_payload)

        metadata = {
            "dynamic_feature_names": ["dyn_0", "dyn_1", "dyn_2"],
            "target_feature_names": ["hs", "tp", "dir_sin", "dir_cos", "dp_sin", "dp_cos"],
            "physical_target_names": ["hs", "tp", "dir", "dp"],
            "normalization": {
                "target_scaler": {
                    "feature_names": ["hs", "tp", "dir_sin", "dir_cos", "dp_sin", "dp_cos"],
                    "mean": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    "scale": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                    "fill_values": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                }
            },
        }
        (base / "point_centric_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    def _base_config(self, base: Path) -> dict:
        return {
            "data": {
                "point_centric_dir": str(base),
                "sequence_window": 2,
                "num_workers": 0,
                "pin_memory": False,
                "use_static_features": False,
                "output_columns": ["hs", "tp", "dir_sin", "dir_cos", "dp_sin", "dp_cos"],
                "validation_sites": ["val_site"],
                "test_sites": ["test_site"],
            },
            "training": {
                "batch_size": 2,
                "seed": 42,
                "sampler": {"enabled": False},
                "loss": {"type": "mse"},
            },
            "model": {"coastal_transformer": {}},
        }

    def test_fraction_one_keeps_all_train_samples(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            self._write_arrays(base)
            arrays = load_point_centric_arrays(str(base))
            cfg = self._base_config(base)
            cfg["data"]["train_sample_subsampling"] = {
                "enabled": True,
                "fraction": 1.0,
                "seed": 42,
                "mode": "per_site",
                "validate_sequence_continuity": True,
                "validation_samples": 1000,
            }

            _, baseline_ds = build_split_dataloader(arrays, cfg, split_name="train", shuffle=False)
            _, train_ds = build_split_dataloader(
                arrays,
                cfg,
                split_name="train",
                shuffle=False,
                apply_train_sample_subsampling=True,
            )

            self.assertEqual(train_ds.samples, baseline_ds.samples)
            self.assertEqual(len(train_ds), 14)
            self.assertTrue(train_ds.train_sample_subsampling_summary["enabled"])
            self.assertFalse(train_ds.train_sample_subsampling_summary["applied"])

    def test_fraction_half_per_site_preserves_train_site_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            self._write_arrays(base)
            arrays = load_point_centric_arrays(str(base))
            cfg = self._base_config(base)
            cfg["data"]["train_sample_subsampling"] = {
                "enabled": True,
                "fraction": 0.5,
                "seed": 42,
                "mode": "per_site",
                "validate_sequence_continuity": True,
                "validation_samples": 1000,
            }

            _, train_ds = build_split_dataloader(
                arrays,
                cfg,
                split_name="train",
                shuffle=False,
                apply_train_sample_subsampling=True,
            )

            self.assertEqual(len(train_ds), 6)
            self.assertEqual({site for site, _ in train_ds.samples}, {"train_a", "train_b"})
            summary = train_ds.train_sample_subsampling_summary
            self.assertTrue(summary["applied"])
            self.assertEqual(summary["before_count"], 14)
            self.assertEqual(summary["after_count"], 6)
            self.assertEqual(summary["removed_count"], 8)
            self.assertEqual(summary["train_sites"], 2)
            self.assertEqual(summary["min_before"], 7)
            self.assertEqual(summary["max_before"], 7)
            self.assertEqual(summary["min_after"], 3)
            self.assertEqual(summary["max_after"], 3)

    def test_validation_and_test_counts_are_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            self._write_arrays(base)
            arrays = load_point_centric_arrays(str(base))
            baseline_cfg = self._base_config(base)
            subsampled_cfg = self._base_config(base)
            subsampled_cfg["data"]["train_sample_subsampling"] = {
                "enabled": True,
                "fraction": 0.5,
                "seed": 42,
                "mode": "per_site",
                "validate_sequence_continuity": True,
                "validation_samples": 1000,
            }

            _, baseline_val_ds = build_split_dataloader(
                arrays, baseline_cfg, split_name="val", shuffle=False
            )
            _, baseline_test_ds = build_split_dataloader(
                arrays, baseline_cfg, split_name="test", shuffle=False
            )
            _, val_ds = build_split_dataloader(
                arrays, subsampled_cfg, split_name="val", shuffle=False
            )
            _, test_ds = build_split_dataloader(
                arrays, subsampled_cfg, split_name="test", shuffle=False
            )

            self.assertEqual(len(val_ds), len(baseline_val_ds))
            self.assertEqual(len(test_ds), len(baseline_test_ds))
            self.assertEqual(len(val_ds), 7)
            self.assertEqual(len(test_ds), 7)

    def test_selected_samples_still_produce_continuous_sequence_windows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            self._write_arrays(base)
            arrays = load_point_centric_arrays(str(base))
            cfg = self._base_config(base)
            cfg["data"]["train_sample_subsampling"] = {
                "enabled": True,
                "fraction": 0.5,
                "seed": 42,
                "mode": "per_site",
                "validate_sequence_continuity": True,
                "validation_samples": 1000,
            }

            _, train_ds = build_split_dataloader(
                arrays,
                cfg,
                split_name="train",
                shuffle=False,
                apply_train_sample_subsampling=True,
            )

            timestamps = np.asarray(arrays.timestamps).astype("datetime64[ns]")
            for idx, (site, target_timestep) in enumerate(train_ds.samples):
                sample = train_ds[idx]
                self.assertEqual(int(sample["x_dynamic"].shape[0]), 2)
                self.assertEqual(int(sample["time_index"]), int(target_timestep))
                self.assertEqual(str(sample["site"]), str(site))
                start = int(target_timestep) - 1
                window_timestamps = timestamps[start : int(target_timestep) + 1]
                self.assertEqual(int(window_timestamps.size), 2)
                self.assertTrue(np.all(np.diff(window_timestamps) == np.timedelta64(1, "h")))

    def test_global_mode_uses_total_train_sample_pool(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            self._write_arrays(base)
            arrays = load_point_centric_arrays(str(base))
            cfg = self._base_config(base)
            cfg["data"]["train_sample_subsampling"] = {
                "enabled": True,
                "fraction": 0.5,
                "seed": 42,
                "mode": "global",
                "validate_sequence_continuity": True,
                "validation_samples": 1000,
            }

            _, train_ds = build_split_dataloader(
                arrays,
                cfg,
                split_name="train",
                shuffle=False,
                apply_train_sample_subsampling=True,
            )

            self.assertEqual(len(train_ds), 7)
            self.assertEqual(train_ds.train_sample_subsampling_summary["mode"], "global")
            self.assertEqual(train_ds.train_sample_subsampling_summary["after_count"], 7)

    def test_shared_recent_site_holdout_uses_recent_timestamps_for_val_and_test(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            self._write_arrays(base)
            arrays = load_point_centric_arrays(str(base))
            arrays.split_idx["train"] = np.array([0, 1, 2, 3, 4, 5], dtype=int)
            arrays.split_idx["val"] = np.array([6, 7], dtype=int)
            arrays.split_idx["test"] = np.array([6, 7], dtype=int)

            cfg = self._base_config(base)
            cfg["split"] = {
                "train": 0.75,
                "val": 0.25,
                "test": 0.25,
                "site_holdout_temporal_mode": "shared_recent",
            }

            _, train_ds = build_split_dataloader(arrays, cfg, split_name="train", shuffle=False)
            _, val_ds = build_split_dataloader(arrays, cfg, split_name="val", shuffle=False)
            _, test_ds = build_split_dataloader(arrays, cfg, split_name="test", shuffle=False)

            self.assertEqual({site for site, _ in train_ds.samples}, {"train_a", "train_b"})
            self.assertEqual({site for site, _ in val_ds.samples}, {"val_site"})
            self.assertEqual({site for site, _ in test_ds.samples}, {"test_site"})
            self.assertTrue(
                all(int(target_timestep) <= 5 for _, target_timestep in train_ds.samples)
            )
            self.assertTrue(all(int(target_timestep) == 7 for _, target_timestep in val_ds.samples))
            self.assertTrue(
                all(int(target_timestep) == 7 for _, target_timestep in test_ds.samples)
            )

    def test_shared_recent_train_subsampling_leaves_val_and_test_counts_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            self._write_arrays(base)
            arrays = load_point_centric_arrays(str(base))
            arrays.split_idx["train"] = np.array([0, 1, 2, 3, 4, 5], dtype=int)
            arrays.split_idx["val"] = np.array([6, 7], dtype=int)
            arrays.split_idx["test"] = np.array([6, 7], dtype=int)

            baseline_cfg = self._base_config(base)
            baseline_cfg["split"] = {
                "train": 0.75,
                "val": 0.25,
                "test": 0.25,
                "site_holdout_temporal_mode": "shared_recent",
            }
            subsampled_cfg = self._base_config(base)
            subsampled_cfg["split"] = {
                "train": 0.75,
                "val": 0.25,
                "test": 0.25,
                "site_holdout_temporal_mode": "shared_recent",
            }
            subsampled_cfg["data"]["train_sample_subsampling"] = {
                "enabled": True,
                "fraction": 0.5,
                "seed": 42,
                "mode": "per_site",
                "validate_sequence_continuity": True,
                "validation_samples": 1000,
            }

            _, baseline_val_ds = build_split_dataloader(
                arrays, baseline_cfg, split_name="val", shuffle=False
            )
            _, baseline_test_ds = build_split_dataloader(
                arrays, baseline_cfg, split_name="test", shuffle=False
            )
            _, val_ds = build_split_dataloader(
                arrays, subsampled_cfg, split_name="val", shuffle=False
            )
            _, test_ds = build_split_dataloader(
                arrays, subsampled_cfg, split_name="test", shuffle=False
            )
            _, train_ds = build_split_dataloader(
                arrays,
                subsampled_cfg,
                split_name="train",
                shuffle=False,
                apply_train_sample_subsampling=True,
            )

            self.assertEqual(len(baseline_val_ds), 1)
            self.assertEqual(len(baseline_test_ds), 1)
            self.assertEqual(len(val_ds), len(baseline_val_ds))
            self.assertEqual(len(test_ds), len(baseline_test_ds))
            self.assertLess(len(train_ds), 10)
            self.assertTrue(
                all(int(target_timestep) <= 5 for _, target_timestep in train_ds.samples)
            )


if __name__ == "__main__":
    unittest.main()
