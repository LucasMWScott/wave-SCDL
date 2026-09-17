"""Test training run manifests."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from src.evaluate import (
    _extract_model_inputs as extract_eval_model_inputs,
    _validate_checkpoint_runtime_compatibility,
)
from src.train import (
    _build_model_io_manifest,
    _extract_model_inputs,
    _update_training_timing_metadata,
    _write_serialized_artifact,
)


class TrainingRunManifestTests(unittest.TestCase):
    def test_extract_model_inputs_omits_static_tensor_when_disabled(self) -> None:
        batch = {
            "x_dynamic": torch.randn(2, 12, 5),
            "x_static": torch.randn(2, 3),
            "x_dynamic_sources": torch.randn(2, 12, 3, 7),
            "source_geometry": torch.randn(2, 3, 4),
        }

        train_inputs = _extract_model_inputs(
            batch,
            torch.device("cpu"),
            use_static_features=False,
        )
        eval_inputs = extract_eval_model_inputs(
            batch,
            torch.device("cpu"),
            use_static_features=False,
        )

        self.assertIsNone(train_inputs["x_static"])
        self.assertIsNone(eval_inputs["x_static"])
        self.assertIsNotNone(train_inputs["source_geometry"])

        train_inputs_no_geometry = _extract_model_inputs(
            batch,
            torch.device("cpu"),
            use_static_features=False,
            use_source_geometry_features=False,
        )
        eval_inputs_no_geometry = extract_eval_model_inputs(
            batch,
            torch.device("cpu"),
            use_static_features=False,
            use_source_geometry_features=False,
        )

        self.assertIsNone(train_inputs_no_geometry["source_geometry"])
        self.assertIsNone(eval_inputs_no_geometry["source_geometry"])

    def test_build_model_io_manifest_tracks_actual_inputs_and_targets(self) -> None:
        arrays = SimpleNamespace(
            dynamic_feature_names=["hs", "tp"],
            site_dynamic_feature_names=["site_bias"],
            static_feature_names=["static_a", "static_b"],
            source_feature_names=["src_hs"],
            source_geometry_feature_names=["distance_km", "bearing_sin"],
            bathy_channel_names=["depth", "mask"],
            target_feature_names=["hs", "tp", "dir_sin", "dir_cos", "dp_sin", "dp_cos"],
            physical_target_names=["hs", "tp", "dir", "dp"],
            transfer_target_names=["log_hs_ratio", "tp_delta", "dir_delta_deg", "dp_delta_deg"],
            reference_target_names=["ref_hs", "ref_tp", "ref_dir", "ref_dp"],
        )
        cfg = {
            "data": {
                "use_static_features": False,
                "use_bathymetry": False,
                "multi_source": {"enabled": False},
                "targets": {
                    "mode": "transfer",
                    "transfer_reference": "weighted_partitioned",
                    "transfer_representation": "residual_correction",
                    "residual_correction": {"enabled": True, "bound_method": "tanh"},
                },
                "output_columns": ["hs", "tp", "dir_sin", "dir_cos", "dp_sin", "dp_cos"],
            },
            "training": {"optimizer": {"lr": 1e-3, "weight_decay": 1e-4}},
            "model": {"architecture": "coastal_transformer"},
        }
        model = SimpleNamespace(
            decoder_type="dense",
            decoder_config={"type": "dense", "ff_hidden_dim": 64, "ff_dropout": 0.1},
            sequence_encoder_type="lstm",
            use_static_features=False,
            use_bathymetry=False,
            use_multi_source=False,
        )
        batch = {
            "site": ["site_a", "site_b"],
            "timestamp": ["2020-01-01T00:00:00", "2020-01-01T01:00:00"],
            "x_static": torch.randn(2, 2),
            "x_dynamic_static_concat": torch.randn(2, 12, 7),
        }
        model_inputs = {
            "x_dynamic": torch.randn(2, 12, 5),
            "x_static": None,
            "x_dynamic_sources": None,
            "source_geometry": None,
            "x_bathy": None,
        }
        target = {
            "hs": torch.randn(2),
            "tp_soft": torch.randn(2, 32),
            "dir_soft": torch.randn(2, 36),
            "physical": torch.randn(2, 4),
        }

        manifest = _build_model_io_manifest(
            cfg=cfg,
            arrays=arrays,
            batch=batch,
            model_inputs=model_inputs,
            target=target,
            model=model,
            epoch_index=1,
            batch_idx=0,
            split_name="train",
        )

        self.assertFalse(manifest["model_flags"]["use_static_features"])
        self.assertEqual(manifest["model_flags"]["decoder_type"], "dense")
        self.assertEqual(manifest["config_flags"]["model_decoder_type"], "cross_attention")
        self.assertEqual(
            manifest["inputs_passed_to_model"]["x_dynamic"]["feature_names"],
            ["hs", "tp", "site_bias", "time_sin", "time_cos"],
        )
        self.assertTrue(
            manifest["inputs_passed_to_model"]["x_dynamic"]["feature_count_matches_names"]
        )
        self.assertEqual(
            manifest["available_batch_tensors_not_passed_to_model"]["x_dynamic_static_concat"][
                "feature_name_count"
            ],
            7,
        )
        self.assertEqual(
            manifest["available_batch_tensors_not_passed_to_model"]["x_static"]["feature_names"],
            ["static_a", "static_b"],
        )
        self.assertEqual(
            manifest["targets"]["tensors"]["physical"]["feature_names"],
            ["hs", "tp", "dir", "dp"],
        )
        self.assertEqual(
            len(manifest["targets"]["tensors"]["tp_soft"]["feature_names"]),
            32,
        )
        self.assertEqual(manifest["config_flags"]["transfer_representation"], "residual_correction")
        self.assertEqual(manifest["targets_config"]["transfer_reference"], "weighted_partitioned")
        self.assertIn("resolved", manifest["config_resolution"])

    def test_evaluation_runtime_compatibility_rejects_decoder_mismatch(self) -> None:
        checkpoint_meta = {
            "config": {
                "data": {
                    "use_static_features": True,
                    "use_bathymetry": False,
                },
                "model": {
                    "coastal_transformer": {
                        "decoder": {"type": "dense"},
                    }
                },
            }
        }
        runtime_cfg = {
            "data": {
                "use_static_features": True,
                "use_bathymetry": False,
            },
            "model": {
                "coastal_transformer": {
                    "decoder": {"type": "cross_attention"},
                }
            },
        }
        with self.assertRaisesRegex(ValueError, "decoder mismatch"):
            _validate_checkpoint_runtime_compatibility(checkpoint_meta, runtime_cfg)

    def test_write_serialized_artifact_emits_json_and_pickle(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            json_path, pickle_path = _write_serialized_artifact(
                out_dir,
                "artifact",
                {"path": out_dir / "nested", "values": [1, 2, 3]},
            )
            self.assertTrue(json_path.exists())
            self.assertTrue(pickle_path.exists())

    def test_update_training_timing_metadata_sets_runtime_fields(self) -> None:
        payload = {"runtime": {"config_path": "/tmp/config.yaml"}}

        _update_training_timing_metadata(
            payload,
            training_started_at_utc="2026-06-08T10:00:00Z",
        )
        self.assertEqual(payload["runtime"]["training_started_at_utc"], "2026-06-08T10:00:00Z")
        self.assertIsNone(payload["runtime"]["training_finished_at_utc"])
        self.assertIsNone(payload["runtime"]["training_duration_seconds"])

        _update_training_timing_metadata(
            payload,
            training_started_at_utc="2026-06-08T10:00:00Z",
            training_finished_at_utc="2026-06-08T10:05:30Z",
            training_duration_seconds=330.5,
        )
        self.assertEqual(payload["runtime"]["training_finished_at_utc"], "2026-06-08T10:05:30Z")
        self.assertEqual(payload["runtime"]["training_duration_seconds"], 330.5)


if __name__ == "__main__":
    unittest.main()
