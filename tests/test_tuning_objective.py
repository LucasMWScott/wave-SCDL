"""Test tuning objective."""

from __future__ import annotations

import unittest

import torch
from optuna.trial import FixedTrial

from src.models.factory import build_model_from_config
from tuning.objective import build_trial_config


class TuningObjectiveTests(unittest.TestCase):
    def test_build_trial_config_applies_conditional_groups_and_nested_paths(self) -> None:
        base_config = {
            "data": {
                "use_static_features": True,
                "use_bathymetry": True,
                "sequence_window": 24,
                "seq_len": 24,
            },
            "training": {
                "epochs": 30,
                "progress_bar": {"enabled": True, "train": True, "val": True},
            },
            "model": {
                "coastal_transformer": {
                    "sequence_encoder": {"type": "transformer", "transformer": {"model_dim": 128}},
                    "static_branch": {"hidden_dims": [128], "dropout": 0.3},
                    "bathy": {"enabled": True, "num_tokens": 4},
                }
            },
        }
        tuning_config = {
            "training_loop": {
                "epochs_per_trial": 12,
                "max_train_batches": 40,
                "max_val_batches": 10,
                "early_stop_patience": 4,
                "dataloader": {"num_workers": 0, "pin_memory": False},
            },
            "search_groups": {
                "transformer": {
                    "enabled": True,
                    "requires": [
                        {
                            "path": "model.coastal_transformer.sequence_encoder.type",
                            "equals": "transformer",
                        }
                    ],
                },
                "static_branch": {
                    "enabled": True,
                    "requires": [{"path": "data.use_static_features", "equals": True}],
                },
                "bathy_cnn": {
                    "enabled": False,
                    "requires": [{"path": "data.use_bathymetry", "equals": True}],
                },
            },
            "search_space": {
                "sequence_window": {
                    "group": "transformer",
                    "type": "int",
                    "low": 24,
                    "high": 72,
                    "step": 12,
                    "paths": ["data.sequence_window", "data.seq_len"],
                },
                "static_hidden_dims": {
                    "group": "static_branch",
                    "type": "int_list",
                    "choices": [[64], [128, 64]],
                    "path": "model.coastal_transformer.static_branch.hidden_dims",
                },
                "static_dropout": {
                    "group": "static_branch",
                    "type": "float",
                    "low": 0.0,
                    "high": 0.4,
                    "step": 0.05,
                    "path": "model.coastal_transformer.static_branch.dropout",
                },
                "bathy_num_tokens": {
                    "group": "bathy_cnn",
                    "type": "categorical",
                    "choices": [4, 9],
                    "path": "model.coastal_transformer.bathy.num_tokens",
                },
            },
        }
        trial = FixedTrial(
            {
                "sequence_window": 48,
                "static_hidden_dims": "128,64",
                "static_dropout": 0.2,
            }
        )

        config, sampled_params, active_groups = build_trial_config(
            base_config, tuning_config, trial
        )

        self.assertTrue(active_groups["transformer"])
        self.assertTrue(active_groups["static_branch"])
        self.assertFalse(active_groups["bathy_cnn"])
        self.assertEqual(sampled_params["static_hidden_dims"], [128, 64])
        self.assertEqual(config["data"]["sequence_window"], 48)
        self.assertEqual(config["data"]["seq_len"], 48)
        self.assertEqual(
            config["model"]["coastal_transformer"]["static_branch"]["hidden_dims"], [128, 64]
        )
        self.assertEqual(config["model"]["coastal_transformer"]["static_branch"]["dropout"], 0.2)
        self.assertEqual(config["model"]["coastal_transformer"]["bathy"]["num_tokens"], 4)
        self.assertEqual(config["training"]["epochs"], 12)
        self.assertEqual(config["training"]["max_train_batches"], 40)
        self.assertEqual(config["training"]["max_val_batches"], 10)
        self.assertFalse(config["training"]["progress_bar"]["enabled"])
        self.assertEqual(config["data"]["num_workers"], 0)
        self.assertFalse(config["data"]["pin_memory"])

    def test_factory_builds_multi_layer_static_branch(self) -> None:
        config = {
            "data": {
                "use_static_features": True,
                "use_bathymetry": False,
                "targets": {"mode": "physical"},
                "multi_source": {"enabled": False},
            },
            "model": {
                "architecture": "coastal_transformer",
                "coastal_transformer": {
                    "sequence_encoder": {
                        "type": "transformer",
                        "transformer": {
                            "model_dim": 64,
                            "num_layers": 2,
                            "num_heads": 4,
                            "ff_multiplier": 2.0,
                            "attn_dropout": 0.1,
                            "ff_dropout": 0.1,
                        },
                    },
                    "static_branch": {
                        "hidden_dims": [48, 24],
                        "dropout": 0.2,
                    },
                    "task_dropout": 0.1,
                },
            },
        }

        model = build_model_from_config(
            config=config,
            dynamic_input_dim=6,
            static_input_dim=5,
            output_dim=4,
            dynamic_feature_names=["hs", "tp", "tm1", "tm2", "dir", "dp"],
        )

        self.assertEqual(model.static_branch_config["hidden_dims"], [48, 24])
        self.assertAlmostEqual(model.static_branch_config["dropout"], 0.2)
        linear_layers = [
            layer for layer in model.static_encoder if isinstance(layer, torch.nn.Linear)
        ]
        self.assertEqual(len(linear_layers), 3)
        self.assertEqual(linear_layers[0].in_features, 5)
        self.assertEqual(linear_layers[0].out_features, 48)
        self.assertEqual(linear_layers[1].in_features, 48)
        self.assertEqual(linear_layers[1].out_features, 24)
        self.assertEqual(linear_layers[2].in_features, 24)
        self.assertEqual(linear_layers[2].out_features, 64)


if __name__ == "__main__":
    unittest.main()
