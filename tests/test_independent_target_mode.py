"""Test independent target mode."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.diagnostics import load_prediction_frame, load_results_bundle
from src.independent_target_mode import (
    TARGET_NAMES,
    IndependentTargetCompositeModel,
    aggregate_child_histories,
    build_independent_target_child_config,
    get_independent_target_child_entries,
    load_model_from_training_metadata,
    resolve_target_modeling,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


class _DummyTargetModel(nn.Module):
    task_order = TARGET_NAMES

    def __init__(self) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(0.0))
        self.use_static_features = False
        self.use_source_geometry_features = False
        self.use_bathymetry = False
        self.use_multi_source = False
        self.sequence_encoder_type = "transformer"
        self.decoder_type = "cross_attention"
        self.target_mode = "transfer"
        self.transfer_representation = "legacy"

    def forward(
        self,
        x_dynamic=None,
        return_attention: bool = False,
        return_diagnostics: bool = False,
        **kwargs,
    ):
        batch_size = int(x_dynamic.shape[0]) if torch.is_tensor(x_dynamic) else 1
        value = self.bias.reshape(1).repeat(batch_size)
        output = {
            "log_hs_ratio": value + 1.0,
            "raw_log_hs_ratio": value + 10.0,
            "tp_delta": value + 2.0,
            "raw_tp_delta": value + 20.0,
            "dir_delta_deg": value + 3.0,
            "raw_dir_delta_deg": value + 30.0,
            "dp_delta_deg": value + 4.0,
            "raw_dp_delta_deg": value + 40.0,
        }
        if return_attention:
            attn = torch.zeros(batch_size, 2, 4, 3, dtype=torch.float32)
            attn[:, :, :, :] = float(self.bias.detach().cpu().item())
            output["cross_attention_weights"] = attn
        if return_diagnostics:
            task_tokens = torch.zeros(batch_size, 4, 2, dtype=torch.float32)
            task_tokens[:, :, :] = float(self.bias.detach().cpu().item())
            output["diagnostics"] = {
                "task_tokens": task_tokens,
                "context_tokens": torch.ones(batch_size, 3, 2, dtype=torch.float32),
                "dynamic_tokens": torch.ones(batch_size, 2, 2, dtype=torch.float32),
                "context_token_types": ["dynamic", "dynamic", "static"],
            }
        return output


class IndependentTargetModeTests(unittest.TestCase):
    def test_resolve_target_modeling_defaults_and_validates(self) -> None:
        self.assertEqual(resolve_target_modeling({"training": {}}), "joint")
        self.assertEqual(
            resolve_target_modeling({"training": {"target_modeling": "independent_per_target"}}),
            "independent_per_target",
        )
        self.assertEqual(
            resolve_target_modeling(
                {"training": {"target_modeling": "joint_then_independent_finetune"}}
            ),
            "joint_then_independent_finetune",
        )
        with self.assertRaises(ValueError):
            resolve_target_modeling({"training": {"target_modeling": "bogus"}})

    def test_build_independent_target_child_config_preserves_subsampling_and_specializes_weights(
        self,
    ) -> None:
        parent = {
            "data": {
                "train_sample_subsampling": {
                    "enabled": True,
                    "fraction": 0.1,
                    "seed": 42,
                }
            },
            "training": {
                "target_modeling": "independent_per_target",
                "pcgrad": {"enabled": True},
                "selection": {
                    "weights": {
                        "hs_rmse": 8.0,
                        "tp_rmse": 1.0,
                        "direction_rmse_deg": 0.067,
                        "dp_rmse_deg": 0.067,
                    }
                },
                "loss": {
                    "type": "blueprint_hybrid",
                    "blueprint_hybrid": {"weights": {"hs": 1.0, "tp": 1.0, "dir": 1.0, "dp": 1.0}},
                },
            },
        }
        child = build_independent_target_child_config(
            parent,
            target_name="dir",
            child_output_dir="results/composite/child_models/dir",
            child_checkpoint_name="cp_dir.pt",
            init_checkpoint_path="/tmp/joint.pt",
            child_stage="fine_tune_from_joint",
        )
        self.assertEqual(child["training"]["target_modeling"], "joint")
        self.assertFalse(child["training"]["pcgrad"]["enabled"])
        self.assertEqual(
            child["training"]["loss"]["blueprint_hybrid"]["weights"],
            {"hs": 0.0, "tp": 0.0, "dir": 1.0, "dp": 0.0},
        )
        self.assertEqual(child["data"]["train_sample_subsampling"]["fraction"], 0.1)
        self.assertEqual(child["logging"]["checkpoint_name"], "cp_dir.pt")
        self.assertEqual(
            Path(child["training"]["initialization"]["checkpoint_path"]), Path("/tmp/joint.pt")
        )
        self.assertEqual(
            child["training"]["independent_target_child"]["stage"], "fine_tune_from_joint"
        )

    def test_get_independent_target_child_entries_rejects_missing_targets(self) -> None:
        payload = {
            "runtime": {
                "independent_target_composite": {
                    "enabled": True,
                    "targets": {"hs": {"checkpoint_path": "/tmp/a.pt", "run_dir": "/tmp/a"}},
                }
            }
        }
        with self.assertRaises(ValueError):
            get_independent_target_child_entries(payload)

    def test_aggregate_child_histories_builds_parent_epoch_rows(self) -> None:
        histories = {
            "hs": [
                {
                    "epoch": 1,
                    "train_loss": 1.0,
                    "validation_ran": True,
                    "val_loss": 2.0,
                    "selection_score": 2.0,
                }
            ],
            "tp": [
                {
                    "epoch": 1,
                    "train_loss": 3.0,
                    "validation_ran": True,
                    "val_loss": 4.0,
                    "selection_score": 4.0,
                }
            ],
            "dir": [
                {
                    "epoch": 1,
                    "train_loss": 5.0,
                    "validation_ran": False,
                    "val_loss": None,
                    "selection_score": None,
                }
            ],
            "dp": [
                {
                    "epoch": 1,
                    "train_loss": 7.0,
                    "validation_ran": True,
                    "val_loss": 8.0,
                    "selection_score": 8.0,
                }
            ],
        }
        aggregated = aggregate_child_histories(histories)
        self.assertEqual(len(aggregated), 1)
        self.assertAlmostEqual(aggregated[0]["train_loss"], 4.0)
        self.assertAlmostEqual(aggregated[0]["val_loss"], (2.0 + 4.0 + 8.0) / 3.0)
        self.assertTrue(aggregated[0]["validation_ran"])

    def test_composite_model_merges_target_specific_outputs_and_attention(self) -> None:
        children = {}
        for idx, name in enumerate(TARGET_NAMES, start=1):
            model = _DummyTargetModel()
            with torch.no_grad():
                model.bias.fill_(float(idx))
            children[name] = model

        composite = IndependentTargetCompositeModel(children)
        x_dynamic = torch.ones(2, 3, 4)
        output = composite(x_dynamic=x_dynamic, return_attention=True, return_diagnostics=True)
        np.testing.assert_allclose(
            output["log_hs_ratio"].detach().cpu().numpy(), np.array([2.0, 2.0])
        )
        np.testing.assert_allclose(output["tp_delta"].detach().cpu().numpy(), np.array([4.0, 4.0]))
        np.testing.assert_allclose(
            output["dir_delta_deg"].detach().cpu().numpy(), np.array([6.0, 6.0])
        )
        np.testing.assert_allclose(
            output["dp_delta_deg"].detach().cpu().numpy(), np.array([8.0, 8.0])
        )
        self.assertEqual(tuple(output["cross_attention_weights"].shape), (2, 2, 4, 3))
        self.assertEqual(tuple(output["diagnostics"]["task_tokens"].shape), (2, 4, 2))

    def test_load_model_from_training_metadata_builds_composite_wrapper(self) -> None:
        with tempfile.TemporaryDirectory(
            dir=REPO_ROOT, prefix="tmp_independent_target_ckpts_"
        ) as tmpdir:
            tmp = Path(tmpdir).resolve()
            entries = {}
            for idx, name in enumerate(TARGET_NAMES, start=1):
                model = _DummyTargetModel()
                with torch.no_grad():
                    model.bias.fill_(float(idx))
                checkpoint_path = tmp / f"{name}.pt"
                torch.save({"model_state_dict": model.state_dict()}, checkpoint_path)
                entries[name] = {
                    "checkpoint_path": str(checkpoint_path),
                    "run_dir": str(tmp / name),
                }

            payload = {
                "runtime": {
                    "independent_target_composite": {
                        "enabled": True,
                        "targets": entries,
                    }
                }
            }
            loaded = load_model_from_training_metadata(
                training_metadata=payload,
                build_model=_DummyTargetModel,
                checkpoint_loader=lambda path: torch.load(path, map_location="cpu"),
                state_dict_getter=lambda checkpoint: checkpoint["model_state_dict"],
                state_dict_loader=lambda model, state_dict: model.load_state_dict(state_dict),
            )
            self.assertIsInstance(loaded, IndependentTargetCompositeModel)
            output = loaded(x_dynamic=torch.ones(1, 3, 4))
            self.assertAlmostEqual(float(output["dp_delta_deg"].item()), 8.0)

    def test_load_results_bundle_and_prediction_frame_support_composite_run_layout(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO_ROOT, prefix="tmp_composite_results_") as tmpdir:
            run_dir = Path(tmpdir).resolve()
            targets = {
                name: {
                    "checkpoint_path": str(run_dir / "child_models" / name / f"{name}.pt"),
                    "run_dir": str(run_dir / "child_models" / name),
                }
                for name in TARGET_NAMES
            }
            metadata = {
                "config": {
                    "data": {"point_centric_dir": "data/processed/demo"},
                    "logging": {"output_dir": str(run_dir)},
                },
                "runtime": {
                    "config_path": str(run_dir / "config.yaml"),
                    "checkpoint_path": targets["hs"]["checkpoint_path"],
                    "independent_target_composite": {"enabled": True, "targets": targets},
                },
            }
            observed = {
                "inputs_passed_to_model": {},
                "model_flags": {"model_class": "IndependentTargetCompositeModel"},
            }
            frame = pd.DataFrame(
                {
                    "split": ["test"],
                    "site": ["site_a"],
                    "time_index": [0],
                    "timestamp": ["2022-01-01T00:00:00"],
                    "target_hs": [1.0],
                    "pred_hs": [1.1],
                    "target_tp": [5.0],
                    "pred_tp": [5.1],
                    "target_dir_sin": [0.0],
                    "target_dir_cos": [1.0],
                    "pred_dir_sin": [1.0],
                    "pred_dir_cos": [0.0],
                    "target_dp_sin": [0.0],
                    "target_dp_cos": [1.0],
                    "pred_dp_sin": [0.0],
                    "pred_dp_cos": [-1.0],
                }
            )

            (run_dir / "training_run_metadata.json").write_text(
                json.dumps(metadata), encoding="utf-8"
            )
            (run_dir / "observed_model_io.json").write_text(json.dumps(observed), encoding="utf-8")
            frame.to_csv(run_dir / "predictions_test.csv", index=False)

            bundle = load_results_bundle(run_dir)
            self.assertTrue(bundle.results_dir.exists())
            self.assertTrue(
                bundle.training_metadata["runtime"]["independent_target_composite"]["enabled"]
            )
            loaded_frame = load_prediction_frame(run_dir, split="test")
            self.assertIn("pred_dir_deg", loaded_frame.columns)
            self.assertIn("pred_dp_deg", loaded_frame.columns)


if __name__ == "__main__":
    unittest.main()
