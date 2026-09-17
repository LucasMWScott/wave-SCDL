"""Test transfer targets."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from src.data_pipeline import build_split_dataloader, load_point_centric_arrays
from src.config_resolution import resolve_config
from src.losses import build_loss_from_config
from src.models.factory import build_model_from_config
from src.preprocessing.transfer_targets import (
    apply_residual_bounds,
    build_residual_targets,
    build_transfer_targets,
    circular_add_deg,
    circular_difference_deg,
    circular_delta_deg,
    circular_weighted_mean_deg,
    reconstruct_from_residuals,
    reconstruct_physical_from_transfer,
)


class TransferTargetTests(unittest.TestCase):
    def test_circular_delta_deg(self) -> None:
        self.assertAlmostEqual(float(circular_delta_deg(10.0, 350.0)), 20.0, places=6)
        self.assertAlmostEqual(float(circular_delta_deg(350.0, 10.0)), -20.0, places=6)
        self.assertAlmostEqual(float(circular_difference_deg(10.0, 350.0)), 20.0, places=6)
        self.assertAlmostEqual(float(circular_difference_deg(350.0, 10.0)), -20.0, places=6)

    def test_circular_add_deg(self) -> None:
        self.assertAlmostEqual(float(circular_add_deg(350.0, 20.0)), 10.0, places=6)

    def test_transfer_target_shapes(self) -> None:
        y_physical = np.array([[2.0, 8.0, 15.0, 25.0], [1.0, 6.0, 350.0, 5.0]], dtype=np.float32)
        y_reference = np.array([[1.0, 7.5, 5.0, 15.0], [0.5, 5.0, 10.0, 355.0]], dtype=np.float32)
        y_transfer = build_transfer_targets(y_physical, y_reference, eps=1e-3)
        self.assertEqual(tuple(y_transfer.shape), (2, 4))

    def test_weighted_direction_reference(self) -> None:
        directions = np.array([[350.0, 10.0]], dtype=np.float64)
        weights = np.array([[0.5, 0.5]], dtype=np.float64)
        mean = circular_weighted_mean_deg(directions, weights, axis=1)
        wrapped = ((float(mean[0]) + 180.0) % 360.0) - 180.0
        self.assertAlmostEqual(wrapped, 0.0, places=5)

    def test_reconstruct_physical_from_transfer(self) -> None:
        reference = np.array([[2.0, 8.0, 350.0, 15.0]], dtype=np.float32)
        transfer = np.array([[np.log(1.5), 1.0, 20.0, -30.0]], dtype=np.float32)
        physical = reconstruct_physical_from_transfer(transfer, reference)
        self.assertAlmostEqual(float(physical[0, 0]), 3.0, places=5)
        self.assertAlmostEqual(float(physical[0, 1]), 9.0, places=5)
        self.assertAlmostEqual(float(physical[0, 2]), 10.0, places=5)
        self.assertAlmostEqual(float(physical[0, 3]), 345.0, places=5)

    def test_reconstruct_physical_tp_is_clamped(self) -> None:
        reference = np.array([[2.0, 29.5, 350.0, 15.0]], dtype=np.float32)
        transfer = np.array([[0.0, 10.0, 0.0, 0.0]], dtype=np.float32)
        physical = reconstruct_physical_from_transfer(transfer, reference, tp_min=0.5, tp_max=30.0)
        self.assertAlmostEqual(float(physical[0, 1]), 30.0, places=5)

    def test_residual_targets_roundtrip_and_zero_reconstructs_reference(self) -> None:
        truth = np.array([[2.5, 9.0, 350.0, 10.0], [1.2, 6.0, 10.0, 350.0]], dtype=np.float32)
        reference = np.array([[2.0, 8.0, 10.0, 30.0], [1.0, 5.0, 350.0, 20.0]], dtype=np.float32)
        residuals = build_residual_targets(truth, reference, eps_hs=1e-4)
        reconstructed = reconstruct_from_residuals(
            residuals, reference, residual_cfg={"bound_method": "none"}
        )
        np.testing.assert_allclose(reconstructed, truth, atol=1e-5, rtol=1e-5)

        zero_reconstructed = reconstruct_from_residuals(
            np.zeros_like(residuals),
            reference,
            residual_cfg={"bound_method": "none"},
        )
        np.testing.assert_allclose(zero_reconstructed, reference, atol=1e-6, rtol=1e-6)

    def test_apply_residual_bounds(self) -> None:
        residuals = np.array([[2.0, -10.0, 500.0, -500.0]], dtype=np.float32)
        tanh_bounded = apply_residual_bounds(
            residuals,
            {
                "bound_method": "tanh",
                "max_abs_log_hs": 1.25,
                "max_abs_tp": 6.0,
                "max_abs_dir_deg": 120.0,
                "max_abs_dp_deg": 120.0,
            },
        )
        self.assertLessEqual(abs(float(tanh_bounded[0, 0])), 1.25 + 1e-6)
        self.assertLessEqual(abs(float(tanh_bounded[0, 1])), 6.0 + 1e-6)

        clamped = apply_residual_bounds(
            residuals,
            {
                "bound_method": "clamp",
                "max_abs_log_hs": 1.0,
                "max_abs_tp": 5.0,
                "max_abs_dir_deg": 90.0,
                "max_abs_dp_deg": 90.0,
            },
        )
        np.testing.assert_allclose(clamped, np.array([[1.0, -5.0, 90.0, -90.0]], dtype=np.float64))

    def _write_common_arrays(self, base: Path, include_transfer: bool) -> None:
        timestamps = np.array([f"2020-01-01T0{i}:00:00" for i in range(8)], dtype=str)
        target_sites = np.array(["site_a", "site_b"], dtype=str)
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
        np.savez_compressed(
            base / "point_centric_X_static.npz",
            target_sites=target_sites,
            static_feature_names=np.array(["static_0", "static_1", "static_2"], dtype=str),
            Xstatic__site_a=np.array([0.1, 0.2, 0.3], dtype=np.float32),
            Xstatic__site_b=np.array([0.4, 0.5, 0.6], dtype=np.float32),
        )

        legacy_row_a = np.array([[0.2, 5.0, 0.0, 1.0, 0.0, 1.0]], dtype=np.float32)
        legacy_row_b = np.array([[0.3, 6.0, 0.0, 1.0, 0.0, 1.0]], dtype=np.float32)
        y_payload = {
            "target_sites": target_sites,
            "target_feature_names": np.array(
                ["hs", "tp", "dir_sin", "dir_cos", "dp_sin", "dp_cos"], dtype=str
            ),
            "timestamps": timestamps,
            "Y__site_a": np.tile(legacy_row_a, (8, 1)),
            "Y__site_b": np.tile(legacy_row_b, (8, 1)),
            "target_mode": np.array(
                ["physical_and_transfer" if include_transfer else "physical"], dtype=str
            ),
            "physical_target_names": np.array(["hs", "tp", "dir", "dp"], dtype=str),
            "Yphysical__site_a": np.tile(
                np.array([[1.5, 8.0, 10.0, 20.0]], dtype=np.float32), (8, 1)
            ),
            "Yphysical__site_b": np.tile(
                np.array([[2.0, 9.0, 15.0, 25.0]], dtype=np.float32), (8, 1)
            ),
        }
        if include_transfer:
            y_payload["transfer_target_names"] = np.array(
                ["log_hs_ratio", "tp_delta", "dir_delta_deg", "dp_delta_deg"], dtype=str
            )
            y_payload["reference_target_names"] = np.array(
                ["ref_hs", "ref_tp", "ref_dir", "ref_dp"], dtype=str
            )
            y_payload["Yreference__site_a"] = np.tile(
                np.array([[1.0, 7.5, 350.0, 10.0]], dtype=np.float32), (8, 1)
            )
            y_payload["Yreference__site_b"] = np.tile(
                np.array([[1.2, 8.0, 5.0, 15.0]], dtype=np.float32), (8, 1)
            )
            y_payload["Ytransfer__site_a"] = np.tile(
                np.array(
                    [[np.log((1.5 + 1e-3) / (1.0 + 1e-3)), 0.5, 20.0, 10.0]], dtype=np.float32
                ),
                (8, 1),
            )
            y_payload["Ytransfer__site_b"] = np.tile(
                np.array(
                    [[np.log((2.0 + 1e-3) / (1.2 + 1e-3)), 1.0, 10.0, 10.0]], dtype=np.float32
                ),
                (8, 1),
            )
        np.savez_compressed(base / "point_centric_Y_targets.npz", **y_payload)

        metadata = {
            "dynamic_feature_names": ["dyn_0", "dyn_1", "dyn_2", "dyn_3"],
            "target_feature_names": ["hs", "tp", "dir_sin", "dir_cos", "dp_sin", "dp_cos"],
            "physical_target_names": ["hs", "tp", "dir", "dp"],
            "transfer_target_names": ["log_hs_ratio", "tp_delta", "dir_delta_deg", "dp_delta_deg"]
            if include_transfer
            else [],
            "reference_target_names": ["ref_hs", "ref_tp", "ref_dir", "ref_dp"]
            if include_transfer
            else [],
            "targets": {
                "mode": "physical_and_transfer" if include_transfer else "physical",
                "transfer_reference": "nearest_bulk",
                "eps": 0.001,
                "physical_loss_weight": 1.0,
                "transfer_loss_weight": 0.5,
            },
            "normalization": {
                "target_scaler": {
                    "feature_names": ["hs", "tp", "dir_sin", "dir_cos", "dp_sin", "dp_cos"],
                    "mean": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    "scale": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                    "fill_values": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                },
                "transfer_target_scaler": {
                    "feature_names": ["log_hs_ratio", "tp_delta"],
                    "mean": [0.0, 0.0],
                    "scale": [1.0, 1.0],
                    "fill_values": [0.0, 0.0],
                }
                if include_transfer
                else {},
            },
        }
        (base / "point_centric_metadata.json").write_text(json.dumps(metadata))

    def test_target_mode_physical_backward_compatible(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            self._write_common_arrays(base, include_transfer=False)
            arrays = load_point_centric_arrays(str(base))
            self.assertEqual(arrays.target_mode, "physical")
            self.assertTrue(arrays.y_physical)
            self.assertFalse(arrays.y_transfer)

    def test_target_mode_physical_and_transfer(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            self._write_common_arrays(base, include_transfer=True)
            arrays = load_point_centric_arrays(str(base))
            self.assertEqual(arrays.target_mode, "physical_and_transfer")
            self.assertTrue(arrays.y_physical)
            self.assertTrue(arrays.y_transfer)
            self.assertTrue(arrays.y_reference)

            cfg = {
                "data": {
                    "point_centric_dir": str(base),
                    "sequence_window": 3,
                    "num_workers": 0,
                    "pin_memory": False,
                    "use_bathymetry": False,
                    "multi_source": {"enabled": False},
                    "targets": {
                        "mode": "physical_and_transfer",
                        "transfer_reference": "nearest_bulk",
                        "eps": 0.001,
                        "physical_loss_weight": 1.0,
                        "transfer_loss_weight": 0.5,
                    },
                    "output_columns": ["hs", "tp", "dir_sin", "dir_cos", "dp_sin", "dp_cos"],
                },
                "training": {
                    "batch_size": 2,
                    "seed": 42,
                    "loss": {
                        "type": "blueprint_hybrid",
                        "blueprint_hybrid": {},
                    },
                },
                "model": {
                    "architecture": "coastal_transformer",
                    "coastal_transformer": {"model_dim": 32, "num_layers": 1, "num_heads": 4},
                },
                "split": {"train": 0.6, "val": 0.2},
            }

            loader, dataset = build_split_dataloader(arrays, cfg, split_name="train", shuffle=False)
            sample = dataset[0]
            self.assertIn("transfer", sample["y"])
            self.assertIn("reference", sample["y"])
            self.assertIn("physical", sample["y"])
            self.assertEqual(tuple(sample["y"]["transfer"].shape), (4,))
            self.assertEqual(tuple(sample["y"]["reference"].shape), (4,))
            self.assertEqual(tuple(sample["y"]["physical"].shape), (4,))

            model = build_model_from_config(
                config=cfg,
                dynamic_input_dim=int(sample["x_dynamic"].shape[-1]),
                static_input_dim=int(sample["x_static"].shape[-1]),
                output_dim=4,
                dynamic_feature_names=arrays.dynamic_feature_names,
            )
            pred = model(sample["x_dynamic"].unsqueeze(0), sample["x_static"].unsqueeze(0))
            for key in ("log_hs_ratio", "tp_delta", "dir_delta_deg", "dp_delta_deg"):
                self.assertIn(key, pred)
            self.assertIn("raw_tp_delta", pred)
            self.assertLessEqual(float(torch.abs(pred["tp_delta"]).max().detach()), 15.0 + 1e-6)

            loss_fn = build_loss_from_config(cfg)
            batch = next(iter(loader))
            batch_pred = model(batch["x_dynamic"], batch["x_static"])
            loss = loss_fn(batch_pred, batch["y"])
            self.assertTrue(torch.isfinite(loss))

    def test_transfer_mode_requires_transfer_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            self._write_common_arrays(base, include_transfer=False)
            arrays = load_point_centric_arrays(str(base))
            cfg = {
                "data": {
                    "point_centric_dir": str(base),
                    "sequence_window": 3,
                    "num_workers": 0,
                    "pin_memory": False,
                    "use_bathymetry": False,
                    "multi_source": {"enabled": False},
                    "targets": {
                        "mode": "physical_and_transfer",
                        "transfer_reference": "nearest_bulk",
                        "eps": 0.001,
                    },
                    "output_columns": ["hs", "tp", "dir_sin", "dir_cos", "dp_sin", "dp_cos"],
                },
                "training": {
                    "batch_size": 2,
                    "loss": {"type": "blueprint_hybrid", "blueprint_hybrid": {}},
                },
                "model": {"coastal_transformer": {}},
                "split": {"train": 0.6, "val": 0.2},
            }
            with self.assertRaises(ValueError):
                build_split_dataloader(arrays, cfg, split_name="train", shuffle=False)

    def test_transfer_and_hybrid_modes_use_different_loss_objectives(self) -> None:
        pred = {
            "log_hs_ratio": torch.tensor([0.1], dtype=torch.float32),
            "tp_delta": torch.tensor([0.2], dtype=torch.float32),
            "dir_delta_deg": torch.tensor([5.0], dtype=torch.float32),
            "dp_delta_deg": torch.tensor([8.0], dtype=torch.float32),
        }
        target = {
            "log_hs_ratio": torch.tensor([0.0], dtype=torch.float32),
            "tp_delta": torch.tensor([0.0], dtype=torch.float32),
            "dir_delta_deg": torch.tensor([0.0], dtype=torch.float32),
            "dp_delta_deg": torch.tensor([0.0], dtype=torch.float32),
            "reference": torch.tensor([[2.0, 8.0, 350.0, 15.0]], dtype=torch.float32),
            "physical": torch.tensor([[3.5, 9.5, 20.0, 330.0]], dtype=torch.float32),
        }

        base_cfg = {
            "data": {
                "point_centric_dir": "",
                "targets": {
                    "transfer_reference": "nearest_bulk",
                    "eps": 0.001,
                    "physical_loss_weight": 1.0,
                    "transfer_loss_weight": 0.5,
                    "tp_min": 0.5,
                    "tp_max": 30.0,
                },
            },
            "training": {
                "loss": {
                    "type": "blueprint_hybrid",
                    "log_components": True,
                    "blueprint_hybrid": {
                        "huber_delta": 0.5,
                        "weights": {"hs": 1.0, "tp": 1.0, "dir": 1.0, "dp": 1.0},
                    },
                }
            },
        }

        transfer_cfg = json.loads(json.dumps(base_cfg))
        transfer_cfg["data"]["targets"]["mode"] = "transfer"
        hybrid_cfg = json.loads(json.dumps(base_cfg))
        hybrid_cfg["data"]["targets"]["mode"] = "physical_and_transfer"

        transfer_loss_fn = build_loss_from_config(transfer_cfg)
        hybrid_loss_fn = build_loss_from_config(hybrid_cfg)

        transfer_loss = transfer_loss_fn(pred, target)
        hybrid_loss = hybrid_loss_fn(pred, target)

        self.assertTrue(torch.isfinite(transfer_loss))
        self.assertTrue(torch.isfinite(hybrid_loss))
        self.assertGreater(float(hybrid_loss.detach()), float(transfer_loss.detach()))
        self.assertAlmostEqual(
            float(transfer_loss_fn.last_components["physical_loss"]), 0.0, places=6
        )
        self.assertGreater(float(hybrid_loss_fn.last_components["physical_loss"]), 0.0)

    def test_residual_representation_uses_raw_transfer_targets(self) -> None:
        pred = {
            "log_hs_ratio": torch.tensor([0.1], dtype=torch.float32),
            "tp_delta": torch.tensor([0.2], dtype=torch.float32),
            "dir_delta_deg": torch.tensor([5.0], dtype=torch.float32),
            "dp_delta_deg": torch.tensor([8.0], dtype=torch.float32),
        }
        target = {
            "transfer": torch.tensor([[0.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
            "log_hs_ratio": torch.tensor([99.0], dtype=torch.float32),
            "tp_delta": torch.tensor([99.0], dtype=torch.float32),
            "dir_delta_deg": torch.tensor([0.0], dtype=torch.float32),
            "dp_delta_deg": torch.tensor([0.0], dtype=torch.float32),
            "reference": torch.tensor([[2.0, 8.0, 350.0, 15.0]], dtype=torch.float32),
            "physical": torch.tensor([[2.0 * np.exp(0.0), 8.0, 350.0, 15.0]], dtype=torch.float32),
        }
        cfg = resolve_config(
            {
                "data": {
                    "targets": {
                        "mode": "physical_and_transfer",
                        "transfer_reference": "nearest_bulk",
                        "transfer_representation": "residual_correction",
                        "residual_correction": {"enabled": True, "bound_method": "none"},
                    }
                },
                "training": {
                    "loss": {
                        "type": "blueprint_hybrid",
                        "compute_physical_loss_on_reconstruction": True,
                        "blueprint_hybrid": {"huber_delta": 0.5},
                    }
                },
            }
        )
        loss_fn = build_loss_from_config(cfg)
        loss = loss_fn(pred, target)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(loss_fn.last_components["transfer_loss"]), 0.0)


if __name__ == "__main__":
    unittest.main()
