"""Test training extensions runtime."""

from __future__ import annotations

import unittest
from unittest import mock

import numpy as np
import torch

from src.config_resolution import resolve_config
from src.data_pipeline import (
    PointCentricArrays,
    PointCentricWindowDataset,
    _build_training_sampler,
    build_split_dataloader,
)
from src.train import resolve_validation_every_n_epochs, should_run_validation_epoch


class TrainingExtensionsRuntimeTests(unittest.TestCase):
    def _make_arrays(self, static_feature_names: list[str] | None = None) -> PointCentricArrays:
        timestamps = np.array([f"2020-01-01T0{i}:00:00" for i in range(6)], dtype=str)
        feature_names = list(
            ["path_alpha", "static_porosity_500m", "other_feat"]
            if static_feature_names is None
            else static_feature_names
        )
        y_targets = {
            "site_a": np.tile(np.array([[1.0, 5.0, 0.0, 1.0, 0.0, 1.0]], dtype=np.float32), (6, 1)),
            "site_b": np.tile(np.array([[2.0, 6.0, 0.0, 1.0, 0.0, 1.0]], dtype=np.float32), (6, 1)),
        }
        y_physical = {
            "site_a": np.array(
                [
                    [1.0, 5.0, 10.0, 20.0],
                    [1.2, 5.5, 12.0, 22.0],
                    [1.4, 6.0, 14.0, 24.0],
                    [1.6, 6.5, 16.0, 26.0],
                    [1.8, 7.0, 18.0, 28.0],
                    [2.0, 7.5, 20.0, 30.0],
                ],
                dtype=np.float32,
            ),
            "site_b": np.array(
                [
                    [0.8, 4.5, 30.0, 40.0],
                    [1.0, 5.0, 32.0, 42.0],
                    [1.1, 5.5, 34.0, 44.0],
                    [1.2, 6.0, 36.0, 46.0],
                    [1.3, 6.5, 38.0, 48.0],
                    [1.4, 7.0, 40.0, 50.0],
                ],
                dtype=np.float32,
            ),
        }
        x_static = {
            "site_a": np.array([1.0, 2.0, 3.0], dtype=np.float32),
            "site_b": np.array([4.0, 5.0, 6.0], dtype=np.float32),
        }
        return PointCentricArrays(
            x_dynamic=np.arange(12, dtype=np.float32).reshape(6, 2),
            x_dynamic_sources=None,
            x_dynamic_sitewise={},
            source_geometry=None,
            y_targets=y_targets,
            y_physical=y_physical,
            y_transfer={},
            y_reference={},
            x_static=x_static,
            x_bathy=None,
            local_depth_m=None,
            local_breaking_hs_cap=None,
            local_breaking_cap_valid=None,
            target_sites=["site_a", "site_b"],
            target_mode="physical",
            dynamic_feature_names=["dyn_0", "dyn_1"],
            source_feature_names=[],
            site_dynamic_feature_names=[],
            source_geometry_feature_names=[],
            target_feature_names=["hs", "tp", "dir_sin", "dir_cos", "dp_sin", "dp_cos"],
            physical_target_names=["hs", "tp", "dir", "dp"],
            transfer_target_names=[],
            reference_target_names=[],
            static_feature_names=feature_names,
            bathy_channel_names=[],
            bathy_site_to_index={},
            bathy_patch_size=None,
            bathy_resolution_m=None,
            bathy_normalization_metadata={},
            timestamps=timestamps,
            split_idx={
                "train": np.array([0, 1, 2, 3, 4], dtype=int),
                "val": np.array([4, 5], dtype=int),
                "test": np.array([4, 5], dtype=int),
            },
            metadata={"normalization": {"static_scaler": {"method": "column_transformer"}}},
            ablation_summary=None,
        )

    def test_static_group_dropout_applies_only_to_selected_columns(self) -> None:
        arrays = self._make_arrays()
        cfg = {
            "data": {
                "sequence_window": 2,
                "output_columns": ["hs", "tp", "dir_sin", "dir_cos", "dp_sin", "dp_cos"],
                "use_static_features": True,
                "static_regularization": {
                    "enabled": True,
                    "noise": {"enabled": False},
                    "group_dropout": {
                        "enabled": True,
                        "p": 1.0,
                        "mode": "per_batch",
                        "replacement_value": 0.0,
                        "groups": {"route": {"prefixes": ["path_"]}},
                    },
                },
            },
            "training": {"batch_size": 2, "loss": {"type": "mse"}},
            "model": {"coastal_transformer": {}},
        }
        loader, _dataset = build_split_dataloader(arrays, cfg, split_name="train", shuffle=False)
        batch = next(iter(loader))
        self.assertTrue(
            torch.allclose(batch["x_static"][:, 0], torch.zeros_like(batch["x_static"][:, 0]))
        )
        self.assertTrue(torch.allclose(batch["x_static"][:, 1], torch.tensor([2.0, 2.0])))
        self.assertTrue(torch.allclose(batch["x_static"][:, 2], torch.tensor([3.0, 3.0])))
        self.assertTrue(
            torch.allclose(
                batch["x_dynamic_static_concat"][..., -3:],
                batch["x_static"].unsqueeze(1).expand(-1, 2, -1),
            )
        )

    def test_static_noise_is_train_only(self) -> None:
        arrays = self._make_arrays()
        cfg = {
            "data": {
                "sequence_window": 2,
                "output_columns": ["hs", "tp", "dir_sin", "dir_cos", "dp_sin", "dp_cos"],
                "use_static_features": True,
                "static_regularization": {
                    "enabled": True,
                    "noise": {"enabled": True, "std": 0.2, "train_only": True},
                    "group_dropout": {"enabled": False},
                },
            },
            "training": {"batch_size": 1, "loss": {"type": "mse"}},
            "model": {"coastal_transformer": {}},
        }
        torch.manual_seed(0)
        train_loader, _ = build_split_dataloader(arrays, cfg, split_name="train", shuffle=False)
        train_batch = next(iter(train_loader))
        self.assertFalse(torch.allclose(train_batch["x_static"][0], torch.tensor([1.0, 2.0, 3.0])))

        torch.manual_seed(0)
        val_loader, _ = build_split_dataloader(arrays, cfg, split_name="val", shuffle=False)
        val_batch = next(iter(val_loader))
        self.assertTrue(torch.allclose(val_batch["x_static"][0], torch.tensor([1.0, 2.0, 3.0])))

    def test_unit_interval_bathy_depth_noise_is_clipped(self) -> None:
        arrays = self._make_arrays()
        arrays.x_bathy = np.array(
            [
                [[[1.0]], [[1.0]]],
                [[[0.5]], [[1.0]]],
            ],
            dtype=np.float32,
        )
        arrays.bathy_channel_names = ["depth", "land_sea_mask"]
        arrays.bathy_site_to_index = {"site_a": 0, "site_b": 1}
        arrays.bathy_normalization_metadata = {
            "stats": {
                "depth": {
                    "method": "unit_interval_train_max",
                    "train_max": 5.0,
                }
            }
        }

        dataset = PointCentricWindowDataset(
            arrays=arrays,
            split_name="train",
            seq_len=2,
            sites=["site_a"],
            output_indices=[0, 1, 2, 3, 4, 5],
            target_columns=["hs", "tp", "dir_sin", "dir_cos", "dp_sin", "dp_cos"],
            use_static_features=False,
            use_bathymetry=True,
            bathymetry_noise_std=0.5,
        )

        with mock.patch("numpy.random.normal", return_value=np.full((1, 1), 3.0, dtype=np.float32)):
            patch = dataset._get_bathy_patch("site_a")

        self.assertAlmostEqual(float(patch[0, 0, 0]), 1.0, places=6)
        self.assertAlmostEqual(float(patch[1, 0, 0]), 1.0, places=6)

    def test_static_group_dropout_warns_when_feature_names_missing(self) -> None:
        arrays = self._make_arrays(static_feature_names=[])
        cfg = {
            "data": {
                "sequence_window": 2,
                "output_columns": ["hs", "tp", "dir_sin", "dir_cos", "dp_sin", "dp_cos"],
                "use_static_features": True,
                "static_regularization": {
                    "enabled": True,
                    "noise": {"enabled": False},
                    "group_dropout": {
                        "enabled": True,
                        "p": 1.0,
                        "groups": {"route": {"prefixes": ["path_"]}},
                    },
                },
            },
            "training": {"batch_size": 1, "loss": {"type": "mse"}},
            "model": {"coastal_transformer": {}},
        }
        with self.assertLogs(level="WARNING") as cm:
            build_split_dataloader(arrays, cfg, split_name="train", shuffle=False)
        self.assertTrue(
            any("static feature names are unavailable" in message.lower() for message in cm.output)
        )

    def test_site_balanced_sampler_equalizes_site_mass(self) -> None:
        arrays = self._make_arrays()
        dataset = PointCentricWindowDataset(
            arrays=arrays,
            split_name="train",
            seq_len=2,
            sites=["site_a", "site_b"],
            output_indices=[0, 1, 2, 3, 4, 5],
            target_mode="physical",
        )
        dataset.samples = [("site_a", 1), ("site_a", 2), ("site_a", 3), ("site_b", 3)]
        cfg = {
            "training": {
                "sampler": {
                    "enabled": True,
                    "strategy": "site_balanced",
                    "site_balanced": {
                        "replacement": True,
                        "combine_with_hs_weight": False,
                        "normalize_within_site": True,
                    },
                }
            }
        }
        sampler = _build_training_sampler(arrays, dataset, cfg, [], [], "train")
        self.assertIsNotNone(sampler)
        weights = sampler.weights.detach().cpu().numpy()
        self.assertAlmostEqual(float(np.sum(weights[:3])), 0.5, places=6)
        self.assertAlmostEqual(float(np.sum(weights[3:])), 0.5, places=6)

    def test_site_balanced_sampler_with_hs_weight_preserves_equal_site_mass(self) -> None:
        arrays = self._make_arrays()
        dataset = PointCentricWindowDataset(
            arrays=arrays,
            split_name="train",
            seq_len=2,
            sites=["site_a", "site_b", "site_c"],
            output_indices=[0, 1, 2, 3, 4, 5],
            target_mode="physical",
        )
        dataset.samples = [("site_a", 1), ("site_a", 2), ("site_b", 3)]
        dataset.sites = ["site_a", "site_b", "site_c"]
        cfg = {
            "training": {
                "sampler": {
                    "enabled": True,
                    "strategy": "site_balanced",
                    "site_balanced": {
                        "replacement": True,
                        "combine_with_hs_weight": True,
                        "hs_power": 2.0,
                        "normalize_within_site": True,
                    },
                }
            }
        }
        with self.assertLogs(level="WARNING") as cm:
            sampler = _build_training_sampler(arrays, dataset, cfg, [], [], "train")
        self.assertIsNotNone(sampler)
        weights = sampler.weights.detach().cpu().numpy()
        self.assertAlmostEqual(float(np.sum(weights[:2])), 0.5, places=6)
        self.assertAlmostEqual(float(np.sum(weights[2:])), 0.5, places=6)
        self.assertTrue(any("zero valid samples" in message.lower() for message in cm.output))

    def test_config_resolution_prefers_canonical_values(self) -> None:
        cfg = {
            "data": {"targets": {}},
            "training": {
                "learning_rate": 0.01,
                "weight_decay": 0.02,
                "optimizer": {"lr": 0.001, "weight_decay": 0.002},
                "scheduler": {"type": "cosine", "min_lr": 1e-5},
            },
            "model": {
                "coastal_transformer": {
                    "model_dim": 64,
                    "sequence_encoder": {"transformer": {"model_dim": 32}},
                }
            },
        }
        with self.assertWarns(UserWarning):
            resolved = resolve_config(cfg)
        self.assertEqual(float(resolved["training"]["optimizer"]["lr"]), 0.001)
        self.assertEqual(float(resolved["training"]["optimizer"]["weight_decay"]), 0.002)
        self.assertEqual(float(resolved["training"]["scheduler"]["cosine"]["min_lr"]), 1e-5)
        self.assertEqual(
            int(
                resolved["model"]["coastal_transformer"]["sequence_encoder"]["transformer"][
                    "model_dim"
                ]
            ),
            32,
        )

    def test_validation_every_n_epochs_defaults_to_one(self) -> None:
        resolved = resolve_config({"training": {}, "data": {"targets": {}}, "model": {}})
        self.assertEqual(resolve_validation_every_n_epochs(resolved), 1)

    def test_validation_epoch_schedule_forces_final_epoch(self) -> None:
        self.assertFalse(should_run_validation_epoch(1, 5, 2))
        self.assertTrue(should_run_validation_epoch(2, 5, 2))
        self.assertFalse(should_run_validation_epoch(3, 5, 2))
        self.assertTrue(should_run_validation_epoch(5, 5, 2))


if __name__ == "__main__":
    unittest.main()
