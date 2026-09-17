"""Test runtime sample filter."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.data_pipeline import build_split_dataloader, load_point_centric_arrays


class RuntimeSampleFilterTests(unittest.TestCase):
    def _write_arrays(self, base: Path, physical_rows: np.ndarray) -> None:
        timestamps = np.array([f"2020-01-01T0{i}:00:00" for i in range(8)], dtype=str)
        target_sites = np.array(["site_a"], dtype=str)
        train_idx = np.array([0, 1, 2, 3, 4], dtype=int)
        val_idx = np.array([5, 6], dtype=int)
        test_idx = np.array([7], dtype=int)

        np.savez_compressed(
            base / "point_centric_X_dynamic.npz",
            X_dynamic=np.arange(8 * 4, dtype=np.float32).reshape(8, 4),
            timestamps=timestamps,
            feature_names=np.array(["dyn_0", "dyn_1", "dyn_2", "dyn_3"], dtype=str),
            offshore_sites=np.array(["nora3_grid_1"], dtype=str),
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
        )

        encoded_targets = np.tile(
            np.array([[0.2, 5.0, 0.0, 1.0, 0.0, 1.0]], dtype=np.float32),
            (8, 1),
        )
        np.savez_compressed(
            base / "point_centric_Y_targets.npz",
            target_sites=target_sites,
            target_feature_names=np.array(
                ["hs", "tp", "dir_sin", "dir_cos", "dp_sin", "dp_cos"], dtype=str
            ),
            timestamps=timestamps,
            target_mode=np.array(["physical"], dtype=str),
            physical_target_names=np.array(["hs", "tp", "dir", "dp"], dtype=str),
            Y__site_a=encoded_targets,
            Yphysical__site_a=np.asarray(physical_rows, dtype=np.float32),
        )

        metadata = {
            "dynamic_feature_names": ["dyn_0", "dyn_1", "dyn_2", "dyn_3"],
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
        (base / "point_centric_metadata.json").write_text(json.dumps(metadata))

    def test_runtime_sample_filter_hs_target_timestep(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            physical_rows = np.array(
                [
                    [0.5, 5.0, 10.0, 20.0],
                    [1.0, 5.1, 10.0, 20.0],
                    [2.0, 5.2, 10.0, 20.0],
                    [3.0, 5.3, 10.0, 20.0],
                    [0.7, 5.4, 10.0, 20.0],
                    [0.6, 5.5, 10.0, 20.0],
                    [0.5, 5.6, 10.0, 20.0],
                    [0.4, 5.7, 10.0, 20.0],
                ],
                dtype=np.float32,
            )
            self._write_arrays(base, physical_rows)
            arrays = load_point_centric_arrays(str(base))
            cfg = {
                "data": {
                    "point_centric_dir": str(base),
                    "sequence_window": 1,
                    "num_workers": 0,
                    "pin_memory": False,
                    "output_columns": ["hs", "tp", "dir_sin", "dir_cos", "dp_sin", "dp_cos"],
                    "sample_filter": {
                        "enabled": True,
                        "apply_to_splits": ["train"],
                        "hs_min": 2.5,
                        "match": "any",
                        "statistic": "target_timestep",
                    },
                },
                "training": {"batch_size": 2, "seed": 42},
                "split": {"train": 0.6, "val": 0.2},
            }

            _, train_ds = build_split_dataloader(arrays, cfg, split_name="train", shuffle=False)
            _, val_ds = build_split_dataloader(arrays, cfg, split_name="val", shuffle=False)

            self.assertEqual(train_ds.samples, [("site_a", 3)])
            self.assertEqual(val_ds.samples, [("site_a", 5), ("site_a", 6)])

    def test_runtime_sample_filter_tp_max_over_window(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            physical_rows = np.array(
                [
                    [0.5, 4.0, 10.0, 20.0],
                    [0.6, 8.0, 10.0, 20.0],
                    [0.7, 4.5, 10.0, 20.0],
                    [0.8, 4.6, 10.0, 20.0],
                    [0.9, 4.7, 10.0, 20.0],
                    [1.0, 4.8, 10.0, 20.0],
                    [1.1, 4.9, 10.0, 20.0],
                    [1.2, 5.0, 10.0, 20.0],
                ],
                dtype=np.float32,
            )
            self._write_arrays(base, physical_rows)
            arrays = load_point_centric_arrays(str(base))
            cfg = {
                "data": {
                    "point_centric_dir": str(base),
                    "sequence_window": 3,
                    "num_workers": 0,
                    "pin_memory": False,
                    "output_columns": ["hs", "tp", "dir_sin", "dir_cos", "dp_sin", "dp_cos"],
                    "sample_filter": {
                        "enabled": True,
                        "apply_to_splits": ["train"],
                        "tp_min": 7.5,
                        "match": "any",
                        "statistic": "max_over_window",
                    },
                },
                "training": {"batch_size": 2, "seed": 42},
                "split": {"train": 0.6, "val": 0.2},
            }

            _, train_ds = build_split_dataloader(arrays, cfg, split_name="train", shuffle=False)

            self.assertEqual(train_ds.samples, [("site_a", 2), ("site_a", 3)])


if __name__ == "__main__":
    unittest.main()
