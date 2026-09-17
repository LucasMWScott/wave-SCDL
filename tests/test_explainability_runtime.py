"""Test explainability runtime."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from src.diagnostics import (
    build_dynamic_feature_groups,
    build_expert_regime_feature_groups,
    build_feature_groups,
    circular_error_deg,
    compute_grouped_feature_family_occlusion_per_sample,
    compute_grouped_temporal_integrated_gradients_per_sample,
    compute_proxy_regime_scores_and_labels,
    extract_cross_attention_maps,
    join_prediction_static_and_attribution_summaries,
    load_prediction_frame,
    load_model_for_explainability,
    load_results_bundle,
    recover_physical_predictions,
    resolve_active_branches,
    safe_forward,
    sample_explainability_subset,
    summarize_site_regime_dominance,
)
from src.diagnostics.explainability import (
    _build_grouped_shap_importance_table,
    _fit_monotonic_feature_mapping,
    compute_static_grouped_shap,
)
from src.models.coastal_transformer import CoastalConditionedTransformer


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results" / "prodset_win_tuned_1"
NOTEBOOKS_DIR = REPO_ROOT / "notebooks"


class ExplainabilityRuntimeTests(unittest.TestCase):
    def test_load_results_bundle_supports_current_run_layout(self) -> None:
        bundle = load_results_bundle(RESULTS_DIR)
        self.assertEqual(bundle.results_dir, RESULTS_DIR.resolve())
        self.assertIn("config", bundle.training_metadata)
        self.assertIn("inputs_passed_to_model", bundle.observed_model_io)
        self.assertTrue(bundle.point_centric_dir is not None)

    def test_load_prediction_frame_derives_direction_degrees(self) -> None:
        frame = load_prediction_frame(RESULTS_DIR, split="val")
        self.assertIn("pred_dir_deg", frame.columns)
        self.assertIn("target_dir_deg", frame.columns)
        self.assertIn("pred_dp_deg", frame.columns)
        self.assertIn("target_dp_deg", frame.columns)

    def test_resolve_active_branches_matches_current_manifest(self) -> None:
        branches = resolve_active_branches(load_results_bundle(RESULTS_DIR))
        self.assertTrue(branches["static"])
        self.assertTrue(branches["multi_source"])
        self.assertFalse(branches["source_geometry"])
        self.assertFalse(branches["bathy"])
        self.assertEqual(branches["target_mode"], "transfer")

    def test_group_builders_only_return_present_columns(self) -> None:
        static_groups = build_feature_groups(
            ["local_depth_m", "ray_fetch_mean_m", "path_length_m", "odd_feature"],
        )
        self.assertIn("local_depth_m", static_groups["groups"]["local_site_geometry"])
        self.assertIn("ray_fetch_mean_m", static_groups["groups"]["ray_fetches"])
        self.assertIn("path_length_m", static_groups["groups"]["path_route_geometry"])
        self.assertIn("odd_feature", static_groups["groups"]["misc_static"])
        self.assertIn("odd_feature", static_groups["unmatched"])

        dynamic_groups = build_dynamic_feature_groups(
            ["hs", "tp", "wind_speed_10m", "time_sin", "mystery_dynamic"],
        )
        self.assertIn("hs", dynamic_groups["groups"]["wave_height_energy"])
        self.assertIn("tp", dynamic_groups["groups"]["period_frequency"])
        self.assertIn("wind_speed_10m", dynamic_groups["groups"]["wind"])
        self.assertIn("time_sin", dynamic_groups["groups"]["seasonality"])
        self.assertIn("mystery_dynamic", dynamic_groups["groups"]["misc_dynamic"])

    def test_expert_regime_group_builder_maps_current_source_features(self) -> None:
        groups = build_expert_regime_feature_groups(
            [
                "hs_swell",
                "tp_swell",
                "thq_swell_sin",
                "fetch_at_swell_direction_m",
                "hs_sea",
                "tp_sea",
                "thq_sea_cos",
                "wind_speed_10m",
                "fetch_at_windwave_direction_m",
                "local_wind_speed_10m",
                "local_wind_dir_cos",
                "fetch_at_local_wind_direction_m",
                "hs",
                "thq_cos",
                "time_sin",
                "odd_feature",
            ]
        )
        self.assertIn("hs_swell", groups["groups"]["swell_proxy"])
        self.assertIn("fetch_at_swell_direction_m", groups["groups"]["swell_proxy"])
        self.assertIn("hs_sea", groups["groups"]["windsea_proxy"])
        self.assertIn("wind_speed_10m", groups["groups"]["windsea_proxy"])
        self.assertIn("local_wind_speed_10m", groups["groups"]["local_wind_proxy"])
        self.assertIn("fetch_at_local_wind_direction_m", groups["groups"]["local_wind_proxy"])
        self.assertIn("hs", groups["groups"]["background_wave"])
        self.assertIn("time_sin", groups["groups"]["seasonality"])
        self.assertIn("odd_feature", groups["groups"]["other_dynamic"])

    def test_sampling_is_reproducible(self) -> None:
        frame = pd.DataFrame(
            {
                "site": [f"site_{idx % 4}" for idx in range(40)],
                "target_hs": np.linspace(0.1, 3.0, num=40),
                "target_tp": np.linspace(4.0, 12.0, num=40),
            }
        )
        first = sample_explainability_subset(frame, max_samples=12, random_seed=7)
        second = sample_explainability_subset(frame, max_samples=12, random_seed=7)
        self.assertListEqual(first.index.tolist(), second.index.tolist())

    def test_circular_error_deg_wraps_correctly(self) -> None:
        wrapped = circular_error_deg(np.array([359.0, 10.0]), np.array([1.0, 350.0]))
        np.testing.assert_allclose(wrapped, np.array([2.0, -20.0]))

    def test_grouped_static_shap_sums_signed_values_within_each_group(self) -> None:
        shap_payload = {
            "values": pd.DataFrame(
                {
                    "a": [1.0, -2.0],
                    "b": [0.5, 1.0],
                    "c": [3.0, -1.0],
                }
            )
        }
        feature_groups = {"groups": {"group_one": ["a", "b"], "group_two": ["c"]}}
        grouped = compute_static_grouped_shap(shap_payload, feature_groups)

        summary = grouped["summary"].set_index("group")
        self.assertAlmostEqual(summary.loc["group_one", "mean_abs_value"], 1.25)
        self.assertAlmostEqual(summary.loc["group_one", "mean_signed_value"], 0.25)
        self.assertAlmostEqual(summary.loc["group_two", "mean_abs_value"], 2.0)

        beeswarm = grouped["beeswarm"]
        self.assertCountEqual(
            beeswarm["group"].tolist(), ["group_one", "group_one", "group_two", "group_two"]
        )
        np.testing.assert_allclose(
            beeswarm.loc[beeswarm["group"] == "group_one", "value"].to_numpy(dtype=float),
            np.array([1.5, -1.0]),
        )

    def test_grouped_shap_importance_summary_normalizes_within_target(self) -> None:
        grouped_outputs = {
            "hs": {
                "summary": pd.DataFrame(
                    {
                        "group": ["fetch", "depth"],
                        "mean_abs_value": [4.0, 1.0],
                    }
                )
            },
            "tp": {
                "summary": pd.DataFrame(
                    {
                        "group": ["fetch", "depth"],
                        "mean_abs_value": [2.0, 2.0],
                    }
                )
            },
        }
        table = _build_grouped_shap_importance_table(
            grouped_outputs, group_order=["fetch", "depth"]
        )
        hs_rows = table.loc[table["target"] == "hs"].set_index("feature_group")
        tp_rows = table.loc[table["target"] == "tp"].set_index("feature_group")
        self.assertAlmostEqual(hs_rows.loc["fetch", "normalized_mean_abs_shap"], 0.8)
        self.assertAlmostEqual(hs_rows.loc["depth", "normalized_mean_abs_shap"], 0.2)
        self.assertEqual(int(hs_rows.loc["fetch", "rank_within_target"]), 1)
        self.assertAlmostEqual(tp_rows.loc["fetch", "normalized_mean_abs_shap"], 0.5)
        self.assertAlmostEqual(tp_rows.loc["depth", "normalized_mean_abs_shap"], 0.5)

    def test_monotonic_feature_mapping_recovers_physical_axis_values(self) -> None:
        transformed = np.array([-1.0, 0.0, 1.0, 2.0], dtype=float)
        raw = np.array([10.0, 20.0, 30.0, 40.0], dtype=float)
        mapper = _fit_monotonic_feature_mapping(transformed, raw)
        self.assertIsNotNone(mapper)
        np.testing.assert_allclose(
            mapper(np.array([-0.5, 0.5, 1.5], dtype=float)), np.array([15.0, 25.0, 35.0])
        )

    def test_recover_physical_predictions_supports_transfer_outputs(self) -> None:
        output = {
            "log_hs_ratio": torch.tensor([0.0, np.log(2.0)], dtype=torch.float32),
            "tp_delta": torch.tensor([1.0, -2.0], dtype=torch.float32),
            "dir_delta_deg": torch.tensor([5.0, -20.0], dtype=torch.float32),
            "dp_delta_deg": torch.tensor([-5.0, 15.0], dtype=torch.float32),
        }
        target = {
            "reference": torch.tensor(
                [
                    [1.0, 8.0, 350.0, 5.0],
                    [2.0, 10.0, 10.0, 300.0],
                ],
                dtype=torch.float32,
            )
        }
        recovered = recover_physical_predictions(output, target=target)
        np.testing.assert_allclose(
            recovered["hs"].detach().cpu().numpy(), np.array([1.0, 4.0]), rtol=1e-5
        )
        np.testing.assert_allclose(
            recovered["tp"].detach().cpu().numpy(), np.array([9.0, 8.0]), rtol=1e-5
        )
        np.testing.assert_allclose(
            recovered["dir"].detach().cpu().numpy(), np.array([355.0, 350.0]), rtol=1e-5
        )
        np.testing.assert_allclose(
            recovered["dp"].detach().cpu().numpy(), np.array([0.0, 315.0]), rtol=1e-5
        )

    def test_proxy_regime_labels_cover_swell_local_windsea_and_mixed(self) -> None:
        frame = pd.DataFrame(
            {
                "sample_index": [0, 1, 2],
                "site": ["a", "a", "b"],
                "ig_share_hs_swell_proxy": [0.7, 0.2, 0.4],
                "ig_share_hs_windsea_proxy": [0.1, 0.4, 0.2],
                "ig_share_hs_local_wind_proxy": [0.0, 0.3, 0.2],
                "occ_share_hs_swell_proxy": [0.6, 0.1, 0.35],
                "occ_share_hs_windsea_proxy": [0.1, 0.3, 0.2],
                "occ_share_hs_local_wind_proxy": [0.0, 0.4, 0.2],
            }
        )
        labeled = compute_proxy_regime_scores_and_labels(
            frame, primary_head="hs", dominance_margin=0.10
        )
        self.assertListEqual(
            labeled["dominant_regime"].tolist(), ["swell", "local_windsea", "mixed"]
        )
        site_summary = summarize_site_regime_dominance(labeled)
        self.assertIn("site_dominant_regime", site_summary.columns)
        self.assertIn("swell_fraction", site_summary.columns)

    def test_join_prediction_static_and_attribution_summaries_uses_head_specific_columns(
        self,
    ) -> None:
        bundle = load_results_bundle(RESULTS_DIR)
        sampled = pd.DataFrame(
            {
                "site": ["norac_grid_32", "norac_grid_32"],
                "target_hs": [0.1, 0.2],
            },
            index=[10, 11],
        )
        grouped_ig = pd.DataFrame(
            {
                "sample_index": [10, 10, 11, 11],
                "head": ["hs", "dir", "hs", "dir"],
                "group": ["swell_proxy", "swell_proxy", "windsea_proxy", "windsea_proxy"],
                "share": [0.8, 0.6, 0.3, 0.4],
            }
        )
        grouped_occ = pd.DataFrame(
            {
                "sample_index": [10, 11],
                "head": ["hs", "hs"],
                "group": ["swell_proxy", "windsea_proxy"],
                "share": [0.7, 0.2],
            }
        )
        joined = join_prediction_static_and_attribution_summaries(
            bundle,
            sampled,
            grouped_ig=grouped_ig,
            grouped_occlusion=grouped_occ,
        )
        self.assertIn("ig_share_hs_swell_proxy", joined.columns)
        self.assertIn("ig_share_dir_swell_proxy", joined.columns)
        self.assertIn("occ_share_hs_swell_proxy", joined.columns)
        self.assertEqual(joined["sample_index"].tolist(), [10, 11])

    def test_model_returns_opt_in_diagnostics_without_breaking_attention(self) -> None:
        model = CoastalConditionedTransformer(
            dynamic_input_dim=5,
            static_input_dim=3,
            output_dim=4,
            dynamic_feature_names=["hs", "tp", "wind_speed_10m", "time_sin", "time_cos"],
            source_dynamic_input_dim=7,
            source_geometry_input_dim=4,
            model_dim=32,
            num_layers=1,
            num_heads=4,
            use_multi_source=False,
            use_static_features=True,
            use_bathymetry=False,
        )
        batch_size, seq_len = 2, 6
        output = model(
            torch.randn(batch_size, seq_len, 5),
            torch.randn(batch_size, 3),
            return_attention=True,
            return_diagnostics=True,
        )
        self.assertIn("cross_attention_weights", output)
        self.assertIn("diagnostics", output)
        diagnostics = output["diagnostics"]
        self.assertIn("context_token_types", diagnostics)
        self.assertIn("task_tokens", diagnostics)
        self.assertIn("static_token", diagnostics)

    def test_dense_decoder_cross_attention_extraction_raises_clear_error(self) -> None:
        model = CoastalConditionedTransformer(
            dynamic_input_dim=5,
            static_input_dim=3,
            output_dim=4,
            model_dim=32,
            num_layers=1,
            num_heads=4,
            decoder_type="dense",
            use_multi_source=False,
            use_static_features=True,
            use_bathymetry=False,
        )
        sample = {
            "x_dynamic": torch.randn(6, 5),
            "x_static": torch.randn(3),
        }
        runner = SimpleNamespace(
            model=model,
            dataset=[sample],
            device=torch.device("cpu"),
        )
        with self.assertRaisesRegex(
            RuntimeError, "available only when decoder.type=cross_attention"
        ):
            extract_cross_attention_maps(runner, sample_indices=[0])

    def test_grouped_regime_helpers_run_on_non_expert_checkpoint(self) -> None:
        bundle = load_results_bundle(RESULTS_DIR)
        model_cfg = bundle.config.get("model", {}) or {}
        coastal_cfg = model_cfg.get("coastal_transformer", {}) or {}
        expert_cfg = coastal_cfg.get("experts", {}) or {}
        experts_enabled = bool(expert_cfg.get("enabled", False))
        self.assertFalse(experts_enabled)
        runner = load_model_for_explainability(RESULTS_DIR, split="val", device="cpu")
        groups = build_expert_regime_feature_groups(
            bundle.observed_model_io["inputs_passed_to_model"]["x_dynamic_sources"]["feature_names"]
        )
        ig_df = compute_grouped_temporal_integrated_gradients_per_sample(
            runner,
            head="hs",
            quantity="physical_prediction",
            sample_indices=[0],
            feature_groups=groups,
            steps=2,
        )
        occ_df = compute_grouped_feature_family_occlusion_per_sample(
            runner,
            head="hs",
            quantity="physical_prediction",
            sample_indices=[0],
            feature_groups=groups,
        )
        self.assertFalse(ig_df.empty)
        self.assertFalse(occ_df.empty)
        sample = runner.dataset[0]
        batched_sample = {
            key: value.unsqueeze(0) if torch.is_tensor(value) else value
            for key, value in sample.items()
        }
        self.assertNotIn("expert_diagnostics", safe_forward(runner.model, batched_sample))

    def test_new_notebooks_exist_and_have_config_cells(self) -> None:
        notebook_names = [
            "14_static_explainability.ipynb",
            "15_temporal_explainability.ipynb",
            "16_cross_branch_explainability.ipynb",
            "17_failure_modes.ipynb",
        ]
        for name in notebook_names:
            payload = json.loads((NOTEBOOKS_DIR / name).read_text())
            self.assertEqual(payload["nbformat"], 4)
            joined = "\n".join("".join(cell.get("source", [])) for cell in payload["cells"])
            self.assertIn("RESULTS_DIR", joined)
            self.assertIn("SPLIT", joined)


if __name__ == "__main__":
    unittest.main()
