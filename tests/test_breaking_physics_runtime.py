"""Test breaking physics runtime."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from src.data_pipeline import build_split_dataloader, load_point_centric_arrays
from src.evaluate import build_breaking_diagnostics
from src.train import _compute_breaking_penalty


class BreakingPhysicsRuntimeTests(unittest.TestCase):
    def _write_minimal_point_centric(self, base: Path) -> None:
        timestamps = np.array([f"2020-01-01T0{i}:00:00" for i in range(4)], dtype=str)
        target_sites = np.array(["site_a", "site_b"], dtype=str)

        np.savez_compressed(
            base / "point_centric_X_dynamic.npz",
            X_dynamic=np.arange(4 * 3, dtype=np.float32).reshape(4, 3),
            timestamps=timestamps,
            feature_names=np.array(["dyn_0", "dyn_1", "dyn_2"], dtype=str),
            offshore_sites=np.array(["nora3_grid_1"], dtype=str),
            train_idx=np.array([0, 1, 2], dtype=int),
            val_idx=np.array([3], dtype=int),
            test_idx=np.array([], dtype=int),
        )

        np.savez_compressed(
            base / "point_centric_X_static.npz",
            target_sites=target_sites,
            static_feature_names=np.array(["s0", "s1"], dtype=str),
            Xstatic__site_a=np.array([0.1, 0.2], dtype=np.float32),
            Xstatic__site_b=np.array([0.3, 0.4], dtype=np.float32),
        )

        y_payload = {
            "target_sites": target_sites,
            "target_feature_names": np.array(
                ["hs", "tp", "dir_sin", "dir_cos", "dp_sin", "dp_cos"], dtype=str
            ),
            "physical_target_names": np.array(["hs", "tp", "dir", "dp"], dtype=str),
            "target_mode": np.array(["physical"], dtype=str),
            "timestamps": timestamps,
            "Y__site_a": np.tile(
                np.array([[0.2, 5.0, 0.0, 1.0, 0.0, 1.0]], dtype=np.float32), (4, 1)
            ),
            "Y__site_b": np.tile(
                np.array([[0.3, 6.0, 0.0, 1.0, 0.0, 1.0]], dtype=np.float32), (4, 1)
            ),
            "Yphysical__site_a": np.tile(
                np.array([[1.5, 8.0, 10.0, 20.0]], dtype=np.float32), (4, 1)
            ),
            "Yphysical__site_b": np.tile(
                np.array([[2.0, 9.0, 15.0, 25.0]], dtype=np.float32), (4, 1)
            ),
        }
        np.savez_compressed(base / "point_centric_Y_targets.npz", **y_payload)

        # Intentionally reverse site order to validate loader-side realignment.
        np.savez_compressed(
            base / "point_centric_physics.npz",
            target_sites=np.array(["site_b", "site_a"], dtype=str),
            local_depth_m=np.array([6.0, 3.0], dtype=np.float32),
            local_breaking_hs_cap=np.array([4.0, 2.0], dtype=np.float32),
            local_breaking_cap_valid=np.array([0.0, 1.0], dtype=np.float32),
        )

        metadata = {
            "dynamic_feature_names": ["dyn_0", "dyn_1", "dyn_2"],
            "target_feature_names": ["hs", "tp", "dir_sin", "dir_cos", "dp_sin", "dp_cos"],
            "physical_target_names": ["hs", "tp", "dir", "dp"],
            "normalization": {
                "target_scaler": {
                    "method": "standard",
                    "feature_names": ["hs", "tp", "dir_sin", "dir_cos", "dp_sin", "dp_cos"],
                    "fill_values": [0.0] * 6,
                    "mean": [0.0] * 6,
                    "scale": [1.0] * 6,
                    "var": [1.0] * 6,
                }
            },
        }
        (base / "point_centric_metadata.json").write_text(json.dumps(metadata))

    def test_loader_aligns_physics_and_dataset_returns_cap_keys(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            self._write_minimal_point_centric(base)
            arrays = load_point_centric_arrays(str(base))

            cfg = {
                "data": {
                    "point_centric_dir": str(base),
                    "sequence_window": 1,
                    "num_workers": 0,
                    "pin_memory": False,
                    "use_bathymetry": False,
                    "multi_source": {"enabled": False},
                    "targets": {"mode": "physical"},
                    "output_columns": ["hs", "tp", "dir_sin", "dir_cos", "dp_sin", "dp_cos"],
                },
                "training": {"batch_size": 2, "loss": {"type": "mse"}},
                "split": {"train": 0.8, "val": 0.2},
            }
            _, dataset = build_split_dataloader(arrays, cfg, split_name="train", shuffle=False)

            sample_a = dataset[0]
            self.assertEqual(sample_a["site"], "site_a")
            self.assertIn("local_breaking_hs_cap", sample_a)
            self.assertIn("local_breaking_cap_valid", sample_a)
            self.assertAlmostEqual(float(sample_a["local_breaking_hs_cap"].item()), 2.0, places=6)
            self.assertAlmostEqual(
                float(sample_a["local_breaking_cap_valid"].item()), 1.0, places=6
            )

            sample_b = dataset[len(dataset.endpoints)]
            self.assertEqual(sample_b["site"], "site_b")
            self.assertAlmostEqual(float(sample_b["local_breaking_hs_cap"].item()), 4.0, places=6)
            self.assertAlmostEqual(
                float(sample_b["local_breaking_cap_valid"].item()), 0.0, places=6
            )

    def test_breaking_penalty_uses_only_valid_caps(self) -> None:
        pred = torch.tensor(
            [[1.5, 0.0, 0.0, 0.0, 0.0, 0.0], [10.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float32
        )
        target = torch.zeros_like(pred)
        batch = {
            "local_breaking_hs_cap": torch.tensor([1.0, 2.0], dtype=torch.float32),
            "local_breaking_cap_valid": torch.tensor([1.0, 0.0], dtype=torch.float32),
        }
        loss = _compute_breaking_penalty(
            pred=pred,
            target=target,
            batch=batch,
            cap_key="local_breaking_hs_cap",
            valid_key="local_breaking_cap_valid",
            target_scaler_stats=None,
            transfer_scaler_stats=None,
        )
        assert loss is not None
        self.assertAlmostEqual(float(loss.detach().cpu().item()), 0.25, places=6)

    def test_breaking_penalty_transfer_mode_uses_ref_hs(self) -> None:
        pred = {"log_hs_ratio": torch.tensor([0.0], dtype=torch.float32)}
        target = {"ref_hs": torch.tensor([2.0], dtype=torch.float32)}
        batch = {
            "local_breaking_hs_cap": torch.tensor([1.0], dtype=torch.float32),
            "local_breaking_cap_valid": torch.tensor([1.0], dtype=torch.float32),
        }
        loss = _compute_breaking_penalty(
            pred=pred,
            target=target,
            batch=batch,
            cap_key="local_breaking_hs_cap",
            valid_key="local_breaking_cap_valid",
            target_scaler_stats=None,
            transfer_scaler_stats={0: (0.0, 1.0)},
        )
        assert loss is not None
        self.assertAlmostEqual(float(loss.detach().cpu().item()), 1.0, places=6)

    def test_breaking_diagnostics_aggregation(self) -> None:
        diagnostics, exceed_pct = build_breaking_diagnostics(
            site_names=["site_a", "site_a", "site_b"],
            target_hs=np.array([0.5, 1.5, 10.0], dtype=np.float64),
            pred_hs=np.array([0.6, 2.0, 9.0], dtype=np.float64),
            target_sites=["site_a", "site_b"],
            local_depth_m=np.array([3.0, 6.0], dtype=np.float64),
            local_breaking_hs_cap=np.array([1.0, 4.0], dtype=np.float64),
            local_breaking_cap_valid=np.array([1.0, 0.0], dtype=np.float64),
        )

        self.assertEqual(list(diagnostics["site"]), ["site_a", "site_b"])
        row_a = diagnostics.loc[diagnostics["site"] == "site_a"].iloc[0]
        self.assertAlmostEqual(float(row_a["pct_target_exceeds_cap"]), 50.0, places=6)
        self.assertAlmostEqual(float(row_a["pct_pred_exceeds_cap"]), 50.0, places=6)
        self.assertAlmostEqual(float(row_a["mean_pred_excess"]), 0.5, places=6)
        self.assertAlmostEqual(float(row_a["max_pred_excess"]), 1.0, places=6)
        row_b = diagnostics.loc[diagnostics["site"] == "site_b"].iloc[0]
        self.assertEqual(int(row_b["local_breaking_cap_valid"]), 0)
        self.assertTrue(np.isnan(float(row_b["pct_target_exceeds_cap"])))
        self.assertAlmostEqual(float(exceed_pct or 0.0), 50.0, places=6)


if __name__ == "__main__":
    unittest.main()
