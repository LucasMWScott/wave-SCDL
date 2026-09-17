"""Test sequence encoder lstm runtime."""

from __future__ import annotations

import unittest

import torch

from src.models.coastal_transformer import CoastalConditionedTransformer
from src.models.factory import build_model_from_config


class LSTMSequenceEncoderRuntimeTests(unittest.TestCase):
    def test_factory_parses_sequence_encoder_and_static_override(self) -> None:
        cfg = {
            "data": {
                "use_static_features": True,
                "use_bathymetry": False,
                "multi_source": {"enabled": False},
            },
            "model": {
                "architecture": "coastal_transformer",
                "coastal_transformer": {
                    "use_static": False,
                    "sequence_encoder": {
                        "type": "lstm",
                        "transformer": {"model_dim": 64, "num_layers": 2, "num_heads": 4},
                        "lstm": {"hidden_dim": 32, "num_layers": 1, "dropout": 0.2},
                    },
                },
            },
        }
        model = build_model_from_config(
            config=cfg,
            dynamic_input_dim=96,
            static_input_dim=117,
            output_dim=4,
        )
        self.assertEqual(getattr(model, "sequence_encoder_type", ""), "lstm")
        self.assertFalse(bool(getattr(model, "use_static_features", True)))
        self.assertEqual(getattr(model, "decoder_type", ""), "cross_attention")

    def test_factory_accepts_dense_decoder_override(self) -> None:
        cfg = {
            "data": {
                "use_static_features": True,
                "use_bathymetry": False,
                "multi_source": {"enabled": False},
            },
            "model": {
                "architecture": "coastal_transformer",
                "coastal_transformer": {
                    "decoder": {"type": "dense"},
                    "sequence_encoder": {
                        "type": "transformer",
                        "transformer": {"model_dim": 64, "num_layers": 2, "num_heads": 4},
                    },
                },
            },
        }
        model = build_model_from_config(
            config=cfg,
            dynamic_input_dim=96,
            static_input_dim=117,
            output_dim=4,
        )
        self.assertEqual(getattr(model, "decoder_type", ""), "dense")

    def test_lstm_single_source_dynamic_only_forward(self) -> None:
        model = CoastalConditionedTransformer(
            dynamic_input_dim=96,
            static_input_dim=0,
            output_dim=4,
            model_dim=64,
            num_layers=2,
            num_heads=4,
            sequence_encoder_type="lstm",
            lstm_hidden_dim=64,
            lstm_num_layers=2,
            use_static_features=False,
            use_bathymetry=False,
            use_expert_heads=False,
        )
        out = model(x_dynamic=torch.randn(2, 12, 96), x_static=None)
        self.assertEqual(tuple(out["hs"].shape), (2,))

    def test_transformer_dense_single_source_dynamic_only_forward(self) -> None:
        model = CoastalConditionedTransformer(
            dynamic_input_dim=96,
            static_input_dim=0,
            output_dim=4,
            model_dim=64,
            num_layers=2,
            num_heads=4,
            decoder_type="dense",
            sequence_encoder_type="transformer",
            use_static_features=False,
            use_bathymetry=False,
            use_expert_heads=False,
        )
        out = model(x_dynamic=torch.randn(2, 12, 96), x_static=None)
        self.assertEqual(tuple(out["hs"].shape), (2,))

    def test_lstm_single_source_with_static(self) -> None:
        model = CoastalConditionedTransformer(
            dynamic_input_dim=96,
            static_input_dim=117,
            output_dim=4,
            model_dim=64,
            num_layers=2,
            num_heads=4,
            sequence_encoder_type="lstm",
            lstm_hidden_dim=64,
            lstm_num_layers=2,
            use_static_features=True,
            use_bathymetry=False,
        )
        out = model(
            x_dynamic=torch.randn(2, 12, 96),
            x_static=torch.randn(2, 117),
        )
        self.assertEqual(tuple(out["tp_log_probs"].shape), (2, 32))

    def test_lstm_dense_single_source_with_static(self) -> None:
        model = CoastalConditionedTransformer(
            dynamic_input_dim=96,
            static_input_dim=117,
            output_dim=4,
            model_dim=64,
            num_layers=2,
            num_heads=4,
            decoder_type="dense",
            sequence_encoder_type="lstm",
            lstm_hidden_dim=64,
            lstm_num_layers=2,
            use_static_features=True,
            use_bathymetry=False,
        )
        out = model(
            x_dynamic=torch.randn(2, 12, 96),
            x_static=torch.randn(2, 117),
        )
        self.assertEqual(tuple(out["tp_log_probs"].shape), (2, 32))

    def test_lstm_single_source_with_bathy(self) -> None:
        model = CoastalConditionedTransformer(
            dynamic_input_dim=96,
            static_input_dim=0,
            output_dim=4,
            model_dim=64,
            num_layers=2,
            num_heads=4,
            sequence_encoder_type="lstm",
            lstm_hidden_dim=64,
            lstm_num_layers=2,
            use_static_features=False,
            use_bathymetry=True,
            bathy_in_channels=2,
            bathy_conv_channels=[8, 16],
            bathy_num_tokens=4,
        )
        out = model(
            x_dynamic=torch.randn(2, 12, 96),
            x_static=None,
            x_bathy=torch.randn(2, 2, 64, 64),
        )
        self.assertEqual(tuple(out["dp_log_probs"].shape), (2, 36))

    def test_lstm_multi_source(self) -> None:
        model = CoastalConditionedTransformer(
            dynamic_input_dim=0,
            static_input_dim=0,
            output_dim=4,
            source_dynamic_input_dim=31,
            source_geometry_input_dim=7,
            model_dim=64,
            num_layers=2,
            num_heads=4,
            sequence_encoder_type="lstm",
            lstm_hidden_dim=64,
            lstm_num_layers=2,
            use_multi_source=True,
            use_static_features=False,
            use_bathymetry=False,
        )
        out = model(
            x_dynamic=None,
            x_static=None,
            x_dynamic_sources=torch.randn(2, 12, 3, 31),
            source_geometry=torch.randn(2, 3, 7),
        )
        self.assertIn("source_attention_weights", out)
        self.assertEqual(tuple(out["source_attention_weights"].shape), (2, 12, 3))

    def test_lstm_multi_source_with_static_and_bathy(self) -> None:
        model = CoastalConditionedTransformer(
            dynamic_input_dim=0,
            static_input_dim=117,
            output_dim=4,
            source_dynamic_input_dim=31,
            source_geometry_input_dim=7,
            model_dim=64,
            num_layers=2,
            num_heads=4,
            sequence_encoder_type="lstm",
            lstm_hidden_dim=64,
            lstm_num_layers=2,
            use_multi_source=True,
            use_static_features=True,
            use_bathymetry=True,
            bathy_in_channels=2,
            bathy_conv_channels=[8, 16],
            bathy_num_tokens=4,
        )
        out = model(
            x_dynamic=None,
            x_static=torch.randn(2, 117),
            x_bathy=torch.randn(2, 2, 64, 64),
            x_dynamic_sources=torch.randn(2, 12, 3, 31),
            source_geometry=torch.randn(2, 3, 7),
        )
        self.assertEqual(tuple(out["hs"].shape), (2,))

    def test_dense_decoder_multi_source_with_static_and_bathy(self) -> None:
        model = CoastalConditionedTransformer(
            dynamic_input_dim=0,
            static_input_dim=117,
            output_dim=4,
            source_dynamic_input_dim=31,
            source_geometry_input_dim=7,
            model_dim=64,
            num_layers=2,
            num_heads=4,
            decoder_type="dense",
            sequence_encoder_type="lstm",
            lstm_hidden_dim=64,
            lstm_num_layers=2,
            use_multi_source=True,
            use_static_features=True,
            use_bathymetry=True,
            bathy_in_channels=2,
            bathy_conv_channels=[8, 16],
            bathy_num_tokens=4,
        )
        out = model(
            x_dynamic=None,
            x_static=torch.randn(2, 117),
            x_bathy=torch.randn(2, 2, 64, 64),
            x_dynamic_sources=torch.randn(2, 12, 3, 31),
            source_geometry=torch.randn(2, 3, 7),
            return_attention=True,
        )
        self.assertIn("source_attention_weights", out)
        self.assertIn("cross_attention_weights", out)
        self.assertIsNone(out["cross_attention_weights"])

    def test_transformer_and_lstm_output_contract_parity(self) -> None:
        transformer_model = CoastalConditionedTransformer(
            dynamic_input_dim=96,
            static_input_dim=117,
            output_dim=4,
            model_dim=64,
            num_layers=2,
            num_heads=4,
            sequence_encoder_type="transformer",
            use_static_features=True,
            use_bathymetry=False,
        )
        lstm_model = CoastalConditionedTransformer(
            dynamic_input_dim=96,
            static_input_dim=117,
            output_dim=4,
            model_dim=64,
            num_layers=2,
            num_heads=4,
            sequence_encoder_type="lstm",
            lstm_hidden_dim=64,
            lstm_num_layers=2,
            use_static_features=True,
            use_bathymetry=False,
        )
        x_dynamic = torch.randn(2, 12, 96)
        x_static = torch.randn(2, 117)
        transformer_out = transformer_model(x_dynamic=x_dynamic, x_static=x_static)
        lstm_out = lstm_model(x_dynamic=x_dynamic, x_static=x_static)
        self.assertEqual(set(transformer_out.keys()), set(lstm_out.keys()))
        self.assertEqual(
            tuple(transformer_out["tp_log_probs"].shape), tuple(lstm_out["tp_log_probs"].shape)
        )

    def test_cross_attention_and_dense_output_contract_parity(self) -> None:
        cross_attention_model = CoastalConditionedTransformer(
            dynamic_input_dim=96,
            static_input_dim=117,
            output_dim=4,
            model_dim=64,
            num_layers=2,
            num_heads=4,
            decoder_type="cross_attention",
            sequence_encoder_type="transformer",
            use_static_features=True,
            use_bathymetry=False,
        )
        dense_model = CoastalConditionedTransformer(
            dynamic_input_dim=96,
            static_input_dim=117,
            output_dim=4,
            model_dim=64,
            num_layers=2,
            num_heads=4,
            decoder_type="dense",
            sequence_encoder_type="transformer",
            use_static_features=True,
            use_bathymetry=False,
        )
        x_dynamic = torch.randn(2, 12, 96)
        x_static = torch.randn(2, 117)
        cross_attention_out = cross_attention_model(x_dynamic=x_dynamic, x_static=x_static)
        dense_out = dense_model(x_dynamic=x_dynamic, x_static=x_static)
        self.assertEqual(set(cross_attention_out.keys()), set(dense_out.keys()))
        self.assertEqual(
            tuple(cross_attention_out["dp_log_probs"].shape), tuple(dense_out["dp_log_probs"].shape)
        )

    def test_dense_decoder_return_attention_reports_no_cross_attention_map(self) -> None:
        model = CoastalConditionedTransformer(
            dynamic_input_dim=96,
            static_input_dim=117,
            output_dim=4,
            model_dim=64,
            num_layers=2,
            num_heads=4,
            decoder_type="dense",
            sequence_encoder_type="transformer",
            use_static_features=True,
            use_bathymetry=False,
        )
        out = model(
            x_dynamic=torch.randn(2, 12, 96),
            x_static=torch.randn(2, 117),
            return_attention=True,
            return_diagnostics=True,
        )
        self.assertIn("cross_attention_weights", out)
        self.assertIsNone(out["cross_attention_weights"])
        self.assertIn("diagnostics", out)

    def test_lstm_with_expert_heads_physical_mode(self) -> None:
        model = CoastalConditionedTransformer(
            dynamic_input_dim=96,
            static_input_dim=0,
            output_dim=4,
            model_dim=64,
            num_layers=2,
            num_heads=4,
            sequence_encoder_type="lstm",
            lstm_hidden_dim=64,
            lstm_num_layers=2,
            use_static_features=False,
            use_expert_heads=True,
            target_mode="physical",
        )
        out = model(
            x_dynamic=torch.randn(2, 12, 96),
            x_static=None,
        )
        self.assertIn("expert_diagnostics", out)
        self.assertIn("component_energies", out["expert_diagnostics"])
        for key in ("hs", "tp_log_probs", "dir_log_probs", "dp_log_probs"):
            self.assertIn(key, out)


if __name__ == "__main__":
    unittest.main()
