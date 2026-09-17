"""Test multisource runtime."""

from __future__ import annotations

import json
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import torch
import yaml

from src.data_pipeline import build_split_dataloader, load_point_centric_arrays
from src.models.coastal_transformer import CoastalConditionedTransformer
from src.models.factory import build_model_from_config
from src.multi_source import build_k_nearest_source_metadata
from src.point_centric_pipeline import (
    _prepare_multisource_feature_frame,
    build_point_centric_dataset,
)


class MultiSourceRuntimeTests(unittest.TestCase):
    def test_k_nearest_mapping_contract(self) -> None:
        nearshore = [
            {"name": "site_a", "lat": 63.0, "lon": 7.0},
            {"name": "site_b", "lat": 63.1, "lon": 7.1},
        ]
        offshore = [
            {"name": "nora3_grid_1", "lat": 63.01, "lon": 7.00},
            {"name": "nora3_grid_2", "lat": 63.05, "lon": 7.05},
            {"name": "nora3_grid_3", "lat": 63.10, "lon": 7.10},
            {"name": "nora3_grid_4", "lat": 63.20, "lon": 7.20},
            {"name": "nora3_wind_grid_1", "lat": 63.03, "lon": 7.01},
        ]

        payload, warnings = build_k_nearest_source_metadata(
            nearshore_entries=nearshore,
            offshore_entries=offshore,
            k_nearest=3,
            max_distance_km_warn=5.0,
            max_distance_km_error=30.0,
        )

        self.assertEqual(payload["k_nearest"], 3)
        self.assertEqual(payload["target_sites"], ["site_a", "site_b"])
        self.assertTrue(isinstance(warnings, list))
        for names, distances, bearings, weights in zip(
            payload["source_names"],
            payload["distances_m"],
            payload["bearings_deg"],
            payload["weights"],
        ):
            self.assertEqual(len(names), 3)
            self.assertEqual(len(set(names)), 3)
            self.assertEqual(len(distances), 3)
            self.assertTrue(all(0.0 <= float(value) < 360.0 for value in bearings))
            self.assertAlmostEqual(float(sum(weights)), 1.0, places=6)

    def test_dataset_multisource_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            timestamps = np.array([f"2020-01-01T0{i}:00:00" for i in range(8)], dtype=str)
            train_idx = np.array([0, 1, 2, 3, 4], dtype=int)
            val_idx = np.array([5, 6], dtype=int)
            test_idx = np.array([7], dtype=int)
            target_sites = np.array(["site_a", "site_b"], dtype=str)

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
                base / "point_centric_X_dynamic_sources.npz",
                X_dynamic_sources=np.arange(2 * 8 * 3 * 5, dtype=np.float32).reshape(2, 8, 3, 5),
                target_sites=target_sites,
                timestamps=timestamps,
                source_feature_names=np.array(
                    ["hs", "tp", "thq_swell_sin", "thq_swell_cos", "wind_speed_10m"],
                    dtype=str,
                ),
                k_nearest=np.array([3], dtype=int),
            )
            np.savez_compressed(
                base / "point_centric_source_geometry.npz",
                source_geometry=np.arange(2 * 3 * 7, dtype=np.float32).reshape(2, 3, 7),
                target_sites=target_sites,
                source_geometry_feature_names=np.array(
                    [
                        "distance_m_norm",
                        "bearing_sin",
                        "bearing_cos",
                        "inverse_distance_weight",
                        "source_rank_1",
                        "source_rank_2",
                        "source_rank_3",
                    ],
                    dtype=str,
                ),
                k_nearest=np.array([3], dtype=int),
            )
            np.savez_compressed(
                base / "point_centric_X_static.npz",
                target_sites=target_sites,
                static_feature_names=np.array(["static_0", "static_1", "static_2"], dtype=str),
                Xstatic__site_a=np.array([0.1, 0.2, 0.3], dtype=np.float32),
                Xstatic__site_b=np.array([0.4, 0.5, 0.6], dtype=np.float32),
            )
            y_payload = {
                "target_sites": target_sites,
                "target_feature_names": np.array(
                    ["hs", "tp", "dir_sin", "dir_cos", "dp_sin", "dp_cos"], dtype=str
                ),
                "timestamps": timestamps,
                "Y__site_a": np.tile(
                    np.array([[0.2, 5.0, 0.0, 1.0, 0.0, 1.0]], dtype=np.float32), (8, 1)
                ),
                "Y__site_b": np.tile(
                    np.array([[0.3, 6.0, 0.0, 1.0, 0.0, 1.0]], dtype=np.float32), (8, 1)
                ),
            }
            np.savez_compressed(base / "point_centric_Y_targets.npz", **y_payload)
            np.savez_compressed(
                base / "point_centric_X_bathy.npz",
                X_bathy=np.ones((2, 2, 4, 4), dtype=np.float32),
                target_sites=target_sites,
                channel_names=np.array(["depth", "wet_mask"], dtype=str),
            )
            metadata = {
                "dynamic_feature_names": ["dyn_0", "dyn_1", "dyn_2", "dyn_3"],
                "source_feature_names": [
                    "hs",
                    "tp",
                    "thq_swell_sin",
                    "thq_swell_cos",
                    "wind_speed_10m",
                ],
                "source_geometry_feature_names": [
                    "distance_m_norm",
                    "bearing_sin",
                    "bearing_cos",
                    "inverse_distance_weight",
                    "source_rank_1",
                    "source_rank_2",
                    "source_rank_3",
                ],
                "multi_source": {"enabled": True},
                "normalization": {
                    "target_scaler": {
                        "feature_names": ["hs", "tp", "dir_sin", "dir_cos", "dp_sin", "dp_cos"],
                        "mean": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        "scale": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                    }
                },
            }
            (base / "point_centric_metadata.json").write_text(json.dumps(metadata))

            arrays = load_point_centric_arrays(str(base))
            cfg = {
                "data": {
                    "point_centric_dir": str(base),
                    "sequence_window": 3,
                    "num_workers": 0,
                    "pin_memory": False,
                    "use_bathymetry": True,
                    "multi_source": {"enabled": True},
                    "output_columns": ["hs", "tp", "dir_sin", "dir_cos", "dp_sin", "dp_cos"],
                },
                "training": {
                    "batch_size": 2,
                    "seed": 42,
                    "loss": {
                        "type": "blueprint_hybrid",
                        "blueprint_hybrid": {
                            "num_tp_bins": 8,
                            "num_dp_bins": 12,
                            "tp_bin_range": [0.0, 25.0],
                            "dp_bin_range": [0.0, 360.0],
                        },
                    },
                },
                "model": {"coastal_transformer": {"multi_source": {"use_geometry_features": True}}},
                "split": {"train": 0.6, "val": 0.2},
            }

            loader, dataset = build_split_dataloader(arrays, cfg, split_name="train", shuffle=False)
            self.assertGreater(len(dataset), 0)
            sample = dataset[0]
            self.assertIn("x_dynamic_sources", sample)
            self.assertIn("source_geometry", sample)
            self.assertIn("x_bathy", sample)
            self.assertEqual(tuple(sample["x_dynamic_sources"].shape), (3, 3, 7))
            self.assertEqual(tuple(sample["source_geometry"].shape), (3, 7))
            expected_geometry = arrays.source_geometry[int(sample["site_index"])]
            self.assertTrue(np.allclose(sample["source_geometry"].numpy(), expected_geometry))
            batch = next(iter(loader))
            self.assertIn("x_dynamic_sources", batch)
            self.assertIn("source_geometry", batch)

    def test_model_forward_legacy_and_multisource(self) -> None:
        legacy_model = CoastalConditionedTransformer(
            dynamic_input_dim=6,
            static_input_dim=3,
            output_dim=4,
            dynamic_feature_names=["hs", "tp", "dir_sin", "dir_cos", "dp_sin", "dp_cos"],
            model_dim=32,
            num_layers=1,
            num_heads=4,
            use_bathymetry=False,
        )
        legacy_out = legacy_model(
            torch.randn(2, 5, 6),
            torch.randn(2, 3),
        )
        for key in (
            "hs",
            "tp_log_probs",
            "dir_log_probs",
            "dp_log_probs",
            "tp_pred",
            "dir_pred",
            "dp_pred",
        ):
            self.assertIn(key, legacy_out)

        multi_model = CoastalConditionedTransformer(
            dynamic_input_dim=0,
            static_input_dim=3,
            output_dim=4,
            source_dynamic_input_dim=7,
            source_geometry_input_dim=7,
            model_dim=32,
            num_layers=1,
            num_heads=4,
            use_multi_source=True,
            use_bathymetry=True,
            bathy_in_channels=2,
            bathy_conv_channels=[8, 16],
            bathy_num_tokens=4,
        )
        multi_out = multi_model(
            None,
            torch.randn(2, 3),
            x_bathy=torch.randn(2, 2, 16, 16),
            x_dynamic_sources=torch.randn(2, 5, 3, 7),
            source_geometry=torch.randn(2, 3, 7),
        )
        self.assertIn("source_attention_weights", multi_out)
        self.assertEqual(tuple(multi_out["source_attention_weights"].shape), (2, 5, 3))

        multi_no_geometry = CoastalConditionedTransformer(
            dynamic_input_dim=0,
            static_input_dim=3,
            output_dim=4,
            source_dynamic_input_dim=7,
            source_geometry_input_dim=None,
            model_dim=32,
            num_layers=1,
            num_heads=4,
            use_multi_source=True,
            use_source_geometry_features=False,
        )
        multi_no_geometry_out = multi_no_geometry(
            None,
            torch.randn(2, 3),
            x_dynamic_sources=torch.randn(2, 5, 3, 7),
            source_geometry=None,
        )
        self.assertIn("source_attention_weights", multi_no_geometry_out)
        self.assertEqual(tuple(multi_no_geometry_out["source_attention_weights"].shape), (2, 5, 3))

        expert_model = CoastalConditionedTransformer(
            dynamic_input_dim=0,
            static_input_dim=3,
            output_dim=4,
            source_dynamic_input_dim=7,
            source_geometry_input_dim=7,
            model_dim=32,
            num_layers=1,
            num_heads=4,
            use_multi_source=True,
            use_expert_heads=True,
        )
        expert_out = expert_model(
            None,
            torch.randn(2, 3),
            x_dynamic_sources=torch.randn(2, 5, 3, 7),
            source_geometry=torch.randn(2, 3, 7),
        )
        self.assertIn("expert_diagnostics", expert_out)
        self.assertIn("component_energies", expert_out["expert_diagnostics"])
        for key in ("hs", "tp_log_probs", "dir_log_probs", "dp_log_probs"):
            self.assertIn(key, expert_out)

    def test_prepare_multisource_feature_frame_preserves_directional_feature_values(self) -> None:
        index = pd.to_datetime(["2020-01-01 00:00:00", "2020-01-01 01:00:00"])
        offshore_df = pd.DataFrame(
            {
                "hs": [1.0, 2.0],
                "tp": [5.0, 6.0],
                "thq_swell": [45.0, 50.0],
                "thq_sea": [90.0, 95.0],
                "wind_speed_10m": [10.0, 12.0],
                "wind_direction_10m": [15.0, 20.0],
            },
            index=index,
        )
        local_wind_df = pd.DataFrame(
            {
                "wind_speed_10m": [7.0, 8.0],
                "wind_direction_10m": [180.0, 190.0],
            },
            index=index,
        )
        static_row = pd.Series(
            {
                "ray_fetch_max_m": 1000.0,
                **{
                    f"ray_fetch_{sector}_m": 100.0 + idx
                    for idx, sector in enumerate(
                        [
                            "N",
                            "NNE",
                            "NE",
                            "ENE",
                            "E",
                            "ESE",
                            "SE",
                            "SSE",
                            "S",
                            "SSW",
                            "SW",
                            "WSW",
                            "W",
                            "WNW",
                            "NW",
                            "NNW",
                        ]
                    )
                },
                **{
                    f"ray_max_slope_{sector}": 0.1 + (0.01 * idx)
                    for idx, sector in enumerate(
                        [
                            "N",
                            "NNE",
                            "NE",
                            "ENE",
                            "E",
                            "ESE",
                            "SE",
                            "SSE",
                            "S",
                            "SSW",
                            "SW",
                            "WSW",
                            "W",
                            "WNW",
                            "NW",
                            "NNW",
                        ]
                    )
                },
            }
        )
        frame, feature_names, _ = _prepare_multisource_feature_frame(
            offshore_df=offshore_df,
            static_row=static_row,
            local_wind_df=local_wind_df,
            offshore_vars=[
                "hs",
                "tp",
                "thq_swell",
                "thq_sea",
                "wind_speed_10m",
                "wind_direction_10m",
            ],
            direction_vars={"thq_swell", "thq_sea", "wind_direction_10m"},
            auto_dynamic_direction_vars=set(),
            input_degrees=True,
        )
        self.assertEqual(len(frame), 2)
        for col in (
            "fetch_at_swell_direction_m",
            "blocking_at_swell_direction",
            "slope_at_swell_direction",
            "fetch_at_windwave_direction_m",
            "blocking_at_windwave_direction",
            "slope_at_windwave_direction",
            "fetch_at_local_wind_direction_m",
            "blocking_at_local_wind_direction",
            "slope_at_local_wind_direction",
        ):
            self.assertIn(col, feature_names)
            self.assertTrue(np.isfinite(frame[col].to_numpy(dtype=float)).all(), msg=col)
        self.assertNotIn("fetch_at_wind_direction_m", feature_names)
        self.assertEqual(
            feature_names[-3:],
            ["local_wind_speed_10m", "local_wind_dir_sin", "local_wind_dir_cos"],
        )
        self.assertTrue(
            np.allclose(frame["local_wind_speed_10m"].to_numpy(dtype=float), np.array([7.0, 8.0]))
        )
        self.assertFalse(
            np.allclose(
                frame["fetch_at_local_wind_direction_m"].to_numpy(dtype=float),
                frame["fetch_at_windwave_direction_m"].to_numpy(dtype=float),
            )
        )

    def test_prepare_multisource_feature_frame_can_drop_local_directional_features(self) -> None:
        index = pd.to_datetime(["2020-01-01 00:00:00", "2020-01-01 01:00:00"])
        offshore_df = pd.DataFrame(
            {
                "hs": [1.0, 2.0],
                "tp": [5.0, 6.0],
                "thq_swell": [45.0, 50.0],
                "thq_sea": [90.0, 95.0],
                "wind_speed_10m": [10.0, 12.0],
                "wind_direction_10m": [180.0, 190.0],
            },
            index=index,
        )
        local_wind_df = pd.DataFrame(
            {
                "wind_speed_10m": [7.0, 8.0],
                "wind_direction_10m": [200.0, 210.0],
            },
            index=index,
        )
        frame, feature_names, _ = _prepare_multisource_feature_frame(
            offshore_df=offshore_df,
            static_row=None,
            local_wind_df=local_wind_df,
            offshore_vars=[
                "hs",
                "tp",
                "thq_swell",
                "thq_sea",
                "wind_speed_10m",
                "wind_direction_10m",
            ],
            direction_vars={"thq_swell", "thq_sea", "wind_direction_10m"},
            auto_dynamic_direction_vars=set(),
            input_degrees=True,
            include_local_direction_features=False,
        )
        self.assertEqual(len(frame), 2)
        for col in feature_names:
            self.assertFalse(col.startswith("fetch_at_"))
            self.assertFalse(col.startswith("blocking_at_"))
            self.assertFalse(col.startswith("slope_at_"))
        self.assertIn("thq_swell_sin", feature_names)
        self.assertIn("thq_swell_cos", feature_names)
        self.assertIn("wind_direction_10m_sin", feature_names)
        self.assertIn("wind_direction_10m_cos", feature_names)
        self.assertEqual(
            feature_names[-3:],
            ["local_wind_speed_10m", "local_wind_dir_sin", "local_wind_dir_cos"],
        )

    def test_point_centric_preprocess_records_nearest_local_wind_metadata_and_feature_shapes(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            out_dir = base / "processed"
            sites_path = base / "sites.yaml"
            training_path = base / "training.yaml"
            preprocess_path = base / "preprocess.yaml"

            sites_payload = {
                "backend": {
                    "nora3_params_dir": "unused_nora3",
                    "norac_params_dir": "unused_norac",
                },
                "offshore_sites": [
                    {"name": "nora3_grid_1", "lat": 63.10, "lon": 7.10},
                    {"name": "nora3_wind_grid_1", "lat": 63.00, "lon": 7.00},
                    {"name": "nora3_wind_grid_2", "lat": 63.30, "lon": 7.30},
                ],
                "nearshore_sites": [
                    {"name": "site_a", "lat": 63.01, "lon": 7.01},
                    {"name": "site_b", "lat": 63.29, "lon": 7.29},
                ],
            }
            training_payload = {
                "data": {
                    "sites_config": str(sites_path),
                    "static_features_csv": str(base / "dummy_static.csv"),
                    "use_static_features": True,
                    "use_bathymetry": False,
                    "use_geometry": False,
                    "offshore_vars": [
                        "hs",
                        "tp",
                        "thq",
                        "thq_swell",
                        "thq_sea",
                        "wind_speed_10m",
                        "wind_direction_10m",
                    ],
                    "nearshore_vars": ["hs", "tp", "dir", "dp"],
                    "direction_vars": [
                        "thq",
                        "thq_swell",
                        "thq_sea",
                        "wind_direction_10m",
                        "dir",
                        "dp",
                    ],
                    "multi_source": {
                        "enabled": True,
                        "k_nearest": 1,
                        "max_distance_km_warn": 80.0,
                        "max_distance_km_error": 150.0,
                        "weight_power": 1.0,
                        "allow_padding": False,
                    },
                    "targets": {"mode": "physical"},
                },
                "split": {"train": 0.5, "val": 0.25, "datetime_column": "time"},
                "normalization": {
                    "default_method": "minmax",
                    "methods": {"minmax": {"feature_range": [0.0, 1.0]}},
                },
            }
            preprocess_payload = {
                "paths": {"processed_dir": str(out_dir)},
                "circular": {"input_degrees": True},
            }
            sites_path.write_text(yaml.safe_dump(sites_payload))
            training_path.write_text(yaml.safe_dump(training_payload))
            preprocess_path.write_text(yaml.safe_dump(preprocess_payload))

            index = pd.date_range("2020-01-01 00:00:00", periods=4, freq="1h")

            def _offshore_frame() -> pd.DataFrame:
                return pd.DataFrame(
                    {
                        "hs": [1.0, 1.1, 1.2, 1.3],
                        "tp": [5.0, 5.1, 5.2, 5.3],
                        "thq": [45.0, 50.0, 55.0, 60.0],
                        "thq_swell": [35.0, 40.0, 45.0, 50.0],
                        "thq_sea": [70.0, 75.0, 80.0, 85.0],
                        "wind_speed_10m": [11.0, 11.0, 11.0, 11.0],
                        "wind_direction_10m": [0.0, 0.0, 0.0, 0.0],
                    },
                    index=index,
                )

            def _wind_frame(speed: float, direction: float) -> pd.DataFrame:
                return pd.DataFrame(
                    {
                        "wind_speed_10m": [speed, speed + 1.0, speed + 2.0, speed + 3.0],
                        "wind_direction_10m": [direction, direction, direction, direction],
                    },
                    index=index,
                )

            def _target_frame(offset: float) -> pd.DataFrame:
                return pd.DataFrame(
                    {
                        "hs": [0.5 + offset, 0.6 + offset, 0.7 + offset, 0.8 + offset],
                        "tp": [4.0 + offset, 4.1 + offset, 4.2 + offset, 4.3 + offset],
                        "dir": [100.0, 110.0, 120.0, 130.0],
                        "dp": [140.0, 150.0, 160.0, 170.0],
                    },
                    index=index,
                )

            timeseries_by_name = {
                "nora3_grid_1": _offshore_frame(),
                "nora3_wind_grid_1": _wind_frame(7.0, 180.0),
                "nora3_wind_grid_2": _wind_frame(9.0, 270.0),
                "site_a": _target_frame(0.0),
                "site_b": _target_frame(0.2),
            }

            sectors = [
                "N",
                "NNE",
                "NE",
                "ENE",
                "E",
                "ESE",
                "SE",
                "SSE",
                "S",
                "SSW",
                "SW",
                "WSW",
                "W",
                "WNW",
                "NW",
                "NNW",
            ]

            def _build_master_static_df() -> pd.DataFrame:
                rows = []
                for site_name in ("site_a", "site_b"):
                    row = {"site_name": site_name, "ray_fetch_max_m": 1000.0}
                    for idx, sector in enumerate(sectors):
                        row[f"ray_fetch_{sector}_m"] = 100.0 + idx
                        row[f"ray_max_slope_{sector}"] = 0.1 + (0.01 * idx)
                        row[f"ray_max_laplacian_{sector}"] = 0.2 + (0.01 * idx)
                        row[f"ray_min_depth_{sector}_m"] = 5.0 + idx
                    rows.append(row)
                return pd.DataFrame(rows)

            static_vectors = {
                "site_a": np.array([1.0, 2.0], dtype=float),
                "site_b": np.array([3.0, 4.0], dtype=float),
            }
            master_static_df = _build_master_static_df()

            def _fake_load_site_timeseries(
                site_name: str, params_dir: str, datetime_col: str = "time"
            ) -> pd.DataFrame:
                frame = timeseries_by_name.get(site_name)
                if frame is None:
                    return pd.DataFrame()
                out = frame.copy()
                out.index.name = datetime_col
                return out

            with (
                mock.patch(
                    "src.point_centric_pipeline.load_site_timeseries",
                    side_effect=_fake_load_site_timeseries,
                ),
                mock.patch(
                    "src.point_centric_pipeline._build_static_vectors",
                    return_value=(
                        static_vectors,
                        ["static_feature_1", "static_feature_2"],
                        [],
                        {"method": "minmax", "columns": ["static_feature_1", "static_feature_2"]},
                        {
                            "enabled": False,
                            "matched_raw_features": [],
                            "matched_transformed_features": [],
                            "warnings": [],
                            "static_feature_count_before": 2,
                            "static_feature_count_after": 2,
                        },
                        master_static_df,
                    ),
                ),
            ):
                paths = build_point_centric_dataset(
                    sites_yaml=str(sites_path),
                    training_config=str(training_path),
                    preprocess_config=str(preprocess_path),
                    out_dir=str(out_dir),
                )

            metadata = json.loads(Path(paths["metadata"]).read_text())
            self.assertEqual(len(metadata["local_wind_assignments"]), 2)
            self.assertEqual(metadata["local_wind_diagnostics"]["num_norac_points"], 2)
            self.assertEqual(
                metadata["local_wind_diagnostics"]["num_candidate_local_wind_points"], 2
            )
            assignment_by_site = {
                item["norac_point_name"]: item["local_wind_point_name"]
                for item in metadata["local_wind_assignments"]
            }
            self.assertEqual(assignment_by_site["site_a"], "nora3_wind_grid_1")
            self.assertEqual(assignment_by_site["site_b"], "nora3_wind_grid_2")

            source_names = list(metadata["source_feature_names"])
            self.assertIn("fetch_at_local_wind_direction_m", source_names)
            self.assertNotIn("fetch_at_wind_direction_m", source_names)
            self.assertEqual(
                source_names[-3:],
                ["local_wind_speed_10m", "local_wind_dir_sin", "local_wind_dir_cos"],
            )

            site_dynamic_names = list(metadata["site_dynamic_feature_names"])
            self.assertIn("local_wind_fetch_aligned_m", site_dynamic_names)
            self.assertIn("local_wind_speed_10m", site_dynamic_names)
            self.assertNotIn("wind_fetch_aligned_m", site_dynamic_names)

            x_sources = np.load(paths["x_dynamic_sources"])
            self.assertEqual(len(source_names), int(x_sources["X_dynamic_sources"].shape[-1]))
            x_sitewise = np.load(paths["x_dynamic_sitewise"])
            self.assertEqual(
                len(site_dynamic_names), int(x_sitewise["XdynamicSite__site_a"].shape[1])
            )
            x_static = np.load(paths["x_static"])
            self.assertEqual(
                len(metadata["static_feature_names"]), int(x_static["Xstatic__site_a"].shape[0])
            )

    def test_point_centric_preprocess_honors_separate_wind_toggles(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            out_dir = base / "processed"
            sites_path = base / "sites.yaml"
            training_path = base / "training.yaml"
            preprocess_path = base / "preprocess.yaml"

            sites_payload = {
                "backend": {
                    "nora3_params_dir": "unused_nora3",
                    "norac_params_dir": "unused_norac",
                },
                "offshore_sites": [
                    {"name": "nora3_grid_1", "lat": 63.10, "lon": 7.10},
                    {"name": "nora3_wind_grid_1", "lat": 63.00, "lon": 7.00},
                ],
                "nearshore_sites": [
                    {"name": "site_a", "lat": 63.01, "lon": 7.01},
                ],
            }
            training_payload = {
                "data": {
                    "sites_config": str(sites_path),
                    "static_features_csv": str(base / "dummy_static.csv"),
                    "use_static_features": True,
                    "use_bathymetry": False,
                    "use_geometry": False,
                    "use_offshore_wind_features": False,
                    "use_local_wind_features": False,
                    "offshore_vars": ["hs", "tp", "thq", "wind_speed_10m", "wind_direction_10m"],
                    "nearshore_vars": ["hs", "tp", "dir", "dp"],
                    "direction_vars": ["thq", "wind_direction_10m", "dir", "dp"],
                    "multi_source": {
                        "enabled": True,
                        "k_nearest": 1,
                        "max_distance_km_warn": 80.0,
                        "max_distance_km_error": 150.0,
                        "weight_power": 1.0,
                        "allow_padding": False,
                    },
                    "targets": {"mode": "physical"},
                },
                "split": {"train": 0.5, "val": 0.25, "datetime_column": "time"},
                "normalization": {
                    "default_method": "minmax",
                    "methods": {"minmax": {"feature_range": [0.0, 1.0]}},
                },
            }
            preprocess_payload = {
                "paths": {"processed_dir": str(out_dir)},
                "circular": {"input_degrees": True},
            }
            sites_path.write_text(yaml.safe_dump(sites_payload))
            training_path.write_text(yaml.safe_dump(training_payload))
            preprocess_path.write_text(yaml.safe_dump(preprocess_payload))

            index = pd.date_range("2020-01-01 00:00:00", periods=4, freq="1h")
            offshore_frame = pd.DataFrame(
                {
                    "hs": [1.0, 1.1, 1.2, 1.3],
                    "tp": [5.0, 5.1, 5.2, 5.3],
                    "thq": [45.0, 50.0, 55.0, 60.0],
                    "wind_speed_10m": [11.0, 11.0, 11.0, 11.0],
                    "wind_direction_10m": [0.0, 0.0, 0.0, 0.0],
                },
                index=index,
            )
            wind_frame = pd.DataFrame(
                {
                    "wind_speed_10m": [7.0, 8.0, 9.0, 10.0],
                    "wind_direction_10m": [180.0, 180.0, 180.0, 180.0],
                },
                index=index,
            )
            target_frame = pd.DataFrame(
                {
                    "hs": [0.5, 0.6, 0.7, 0.8],
                    "tp": [4.0, 4.1, 4.2, 4.3],
                    "dir": [100.0, 110.0, 120.0, 130.0],
                    "dp": [140.0, 150.0, 160.0, 170.0],
                },
                index=index,
            )
            timeseries_by_name = {
                "nora3_grid_1": offshore_frame,
                "nora3_wind_grid_1": wind_frame,
                "site_a": target_frame,
            }

            sectors = [
                "N",
                "NNE",
                "NE",
                "ENE",
                "E",
                "ESE",
                "SE",
                "SSE",
                "S",
                "SSW",
                "SW",
                "WSW",
                "W",
                "WNW",
                "NW",
                "NNW",
            ]
            row = {"site_name": "site_a", "ray_fetch_max_m": 1000.0}
            for idx, sector in enumerate(sectors):
                row[f"ray_fetch_{sector}_m"] = 100.0 + idx
                row[f"ray_max_slope_{sector}"] = 0.1 + (0.01 * idx)
                row[f"ray_max_laplacian_{sector}"] = 0.2 + (0.01 * idx)
                row[f"ray_min_depth_{sector}_m"] = 5.0 + idx
            master_static_df = pd.DataFrame([row])

            def _fake_load_site_timeseries(
                site_name: str, params_dir: str, datetime_col: str = "time"
            ) -> pd.DataFrame:
                frame = timeseries_by_name.get(site_name)
                if frame is None:
                    return pd.DataFrame()
                out = frame.copy()
                out.index.name = datetime_col
                return out

            with (
                mock.patch(
                    "src.point_centric_pipeline.load_site_timeseries",
                    side_effect=_fake_load_site_timeseries,
                ),
                mock.patch(
                    "src.point_centric_pipeline._build_static_vectors",
                    return_value=(
                        {"site_a": np.array([1.0, 2.0], dtype=float)},
                        ["static_feature_1", "static_feature_2"],
                        [],
                        {"method": "minmax", "columns": ["static_feature_1", "static_feature_2"]},
                        {
                            "enabled": False,
                            "matched_raw_features": [],
                            "matched_transformed_features": [],
                            "warnings": [],
                            "static_feature_count_before": 2,
                            "static_feature_count_after": 2,
                        },
                        master_static_df,
                    ),
                ),
            ):
                paths = build_point_centric_dataset(
                    sites_yaml=str(sites_path),
                    training_config=str(training_path),
                    preprocess_config=str(preprocess_path),
                    out_dir=str(out_dir),
                )

            metadata = json.loads(Path(paths["metadata"]).read_text())
            self.assertFalse(metadata["wind_feature_toggles"]["use_offshore_wind_features"])
            self.assertFalse(metadata["wind_feature_toggles"]["use_local_wind_features"])
            self.assertEqual(metadata["local_wind_assignments"], [])
            self.assertEqual(metadata["local_wind_source_type"], "disabled")
            self.assertFalse(metadata["local_wind_diagnostics"]["enabled"])
            self.assertEqual(
                metadata["site_dynamic_feature_names"][:2],
                ["wave_fetch_aligned_m", "wave_fetch_aligned_ratio"],
            )
            self.assertFalse(
                any("local_wind" in name for name in metadata["site_dynamic_feature_names"])
            )
            self.assertFalse(
                any("wind_speed_10m" in name for name in metadata["dynamic_feature_names"])
            )
            self.assertFalse(any("local_wind" in name for name in metadata["source_feature_names"]))
            self.assertFalse(
                any("wind_speed_10m" == name for name in metadata["source_feature_names"])
            )

    def test_dataset_multisource_without_geometry_or_static_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            timestamps = np.array([f"2020-01-01T0{i}:00:00" for i in range(8)], dtype=str)
            train_idx = np.array([0, 1, 2, 3, 4], dtype=int)
            val_idx = np.array([5, 6], dtype=int)
            test_idx = np.array([7], dtype=int)
            target_sites = np.array(["site_a", "site_b"], dtype=str)

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
                base / "point_centric_X_dynamic_sources.npz",
                X_dynamic_sources=np.arange(2 * 8 * 3 * 5, dtype=np.float32).reshape(2, 8, 3, 5),
                target_sites=target_sites,
                timestamps=timestamps,
                source_feature_names=np.array(
                    ["hs", "tp", "thq_swell_sin", "thq_swell_cos", "wind_speed_10m"],
                    dtype=str,
                ),
                k_nearest=np.array([3], dtype=int),
            )
            y_payload = {
                "target_sites": target_sites,
                "target_feature_names": np.array(
                    ["hs", "tp", "dir_sin", "dir_cos", "dp_sin", "dp_cos"], dtype=str
                ),
                "timestamps": timestamps,
                "Y__site_a": np.tile(
                    np.array([[0.2, 5.0, 0.0, 1.0, 0.0, 1.0]], dtype=np.float32), (8, 1)
                ),
                "Y__site_b": np.tile(
                    np.array([[0.3, 6.0, 0.0, 1.0, 0.0, 1.0]], dtype=np.float32), (8, 1)
                ),
            }
            np.savez_compressed(base / "point_centric_Y_targets.npz", **y_payload)
            metadata = {
                "dynamic_feature_names": ["dyn_0", "dyn_1", "dyn_2", "dyn_3"],
                "source_feature_names": [
                    "hs",
                    "tp",
                    "thq_swell_sin",
                    "thq_swell_cos",
                    "wind_speed_10m",
                ],
                "static_feature_names": [],
                "source_geometry_feature_names": [],
                "multi_source": {"enabled": True, "route_features_available": False},
                "normalization": {
                    "target_scaler": {
                        "feature_names": ["hs", "tp", "dir_sin", "dir_cos", "dp_sin", "dp_cos"],
                        "mean": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        "scale": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                    }
                },
            }
            (base / "point_centric_metadata.json").write_text(json.dumps(metadata))

            arrays = load_point_centric_arrays(str(base))
            self.assertEqual(arrays.static_feature_names, [])
            self.assertEqual(arrays.x_static, {})
            self.assertIsNone(arrays.source_geometry)

            cfg = {
                "data": {
                    "point_centric_dir": str(base),
                    "sequence_window": 3,
                    "num_workers": 0,
                    "pin_memory": False,
                    "use_static_features": False,
                    "use_geometry": False,
                    "use_bathymetry": False,
                    "multi_source": {"enabled": True},
                    "output_columns": ["hs", "tp", "dir_sin", "dir_cos", "dp_sin", "dp_cos"],
                },
                "training": {
                    "batch_size": 2,
                    "seed": 42,
                    "loss": {
                        "type": "blueprint_hybrid",
                        "blueprint_hybrid": {
                            "num_tp_bins": 8,
                            "num_dp_bins": 12,
                            "tp_bin_range": [0.0, 25.0],
                            "dp_bin_range": [0.0, 360.0],
                        },
                    },
                },
                "model": {
                    "coastal_transformer": {
                        "multi_source": {"enabled": True},
                    }
                },
                "split": {"train": 0.6, "val": 0.2},
            }

            loader, dataset = build_split_dataloader(arrays, cfg, split_name="train", shuffle=False)
            self.assertGreater(len(dataset), 0)
            sample = dataset[0]
            self.assertIn("x_dynamic_sources", sample)
            self.assertNotIn("source_geometry", sample)
            self.assertNotIn("x_static", sample)
            self.assertNotIn("x_dynamic_static_concat", sample)
            self.assertEqual(tuple(sample["x_dynamic_sources"].shape), (3, 3, 7))

            batch = next(iter(loader))
            self.assertIn("x_dynamic_sources", batch)
            self.assertNotIn("source_geometry", batch)
            self.assertNotIn("x_static", batch)

    def test_point_centric_preprocess_shared_recent_uses_train_only_temporal_scaler_fit(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            out_dir = base / "processed"
            sites_path = base / "sites.yaml"
            training_path = base / "training.yaml"
            preprocess_path = base / "preprocess.yaml"

            sites_payload = {
                "backend": {
                    "nora3_params_dir": "unused_nora3",
                    "norac_params_dir": "unused_norac",
                },
                "offshore_sites": [
                    {"name": "nora3_grid_1", "lat": 63.10, "lon": 7.10},
                ],
                "nearshore_sites": [
                    {"name": "site_train_a", "lat": 63.01, "lon": 7.01},
                    {"name": "site_train_b", "lat": 63.02, "lon": 7.02},
                    {"name": "site_val", "lat": 63.03, "lon": 7.03},
                    {"name": "site_test", "lat": 63.04, "lon": 7.04},
                ],
            }
            training_payload = {
                "data": {
                    "sites_config": str(sites_path),
                    "use_static_features": False,
                    "use_bathymetry": False,
                    "use_offshore_wind_features": False,
                    "use_local_wind_features": False,
                    "offshore_vars": ["hs"],
                    "nearshore_vars": ["hs", "tp", "dir", "dp"],
                    "direction_vars": ["dir", "dp"],
                    "validation_sites": ["site_val"],
                    "test_sites": ["site_test"],
                    "targets": {"mode": "physical"},
                },
                "split": {
                    "train": 0.75,
                    "val": 0.25,
                    "test": 0.25,
                    "datetime_column": "time",
                    "site_holdout_temporal_mode": "shared_recent",
                },
                "normalization": {
                    "default_method": "minmax",
                    "methods": {"minmax": {"feature_range": [0.0, 1.0]}},
                },
            }
            preprocess_payload = {
                "paths": {"processed_dir": str(out_dir)},
                "circular": {"input_degrees": True},
            }
            sites_path.write_text(yaml.safe_dump(sites_payload))
            training_path.write_text(yaml.safe_dump(training_payload))
            preprocess_path.write_text(yaml.safe_dump(preprocess_payload))

            index = pd.date_range("2020-01-01 00:00:00", periods=8, freq="1h")
            offshore_frame = pd.DataFrame(
                {"hs": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 100.0, 200.0]}, index=index
            )

            def _target_frame(offset: float) -> pd.DataFrame:
                return pd.DataFrame(
                    {
                        "hs": np.linspace(0.5 + offset, 1.2 + offset, num=8),
                        "tp": np.linspace(4.0, 4.7, num=8),
                        "dir": np.linspace(90.0, 160.0, num=8),
                        "dp": np.linspace(120.0, 190.0, num=8),
                    },
                    index=index,
                )

            timeseries_by_name = {
                "nora3_grid_1": offshore_frame,
                "site_train_a": _target_frame(0.0),
                "site_train_b": _target_frame(0.1),
                "site_val": _target_frame(0.2),
                "site_test": _target_frame(0.3),
            }

            def _fake_load_site_timeseries(
                site_name: str, params_dir: str, datetime_col: str = "time"
            ) -> pd.DataFrame:
                frame = timeseries_by_name.get(site_name)
                if frame is None:
                    return pd.DataFrame()
                out = frame.copy()
                out.index.name = datetime_col
                return out

            with mock.patch(
                "src.point_centric_pipeline.load_site_timeseries",
                side_effect=_fake_load_site_timeseries,
            ):
                paths = build_point_centric_dataset(
                    sites_yaml=str(sites_path),
                    training_config=str(training_path),
                    preprocess_config=str(preprocess_path),
                    out_dir=str(out_dir),
                )

            dynamic_npz = np.load(paths["x_dynamic"])
            metadata = json.loads(Path(paths["metadata"]).read_text())
            np.testing.assert_array_equal(
                dynamic_npz["train_idx"], np.array([0, 1, 2, 3, 4, 5], dtype=int)
            )
            np.testing.assert_array_equal(dynamic_npz["val_idx"], np.array([6, 7], dtype=int))
            np.testing.assert_array_equal(dynamic_npz["test_idx"], np.array([6, 7], dtype=int))
            self.assertEqual(metadata["normalization"]["train_idx_count"], 6)
            self.assertEqual(metadata["splits"]["site_holdout_temporal_mode"], "shared_recent")
            self.assertTrue(metadata["splits"]["site_holdout_temporal_active"])
            self.assertAlmostEqual(
                float(metadata["splits"]["site_holdout_temporal_recent_fraction"]), 0.25
            )
            self.assertGreater(float(dynamic_npz["X_dynamic"][6, 0]), 1.0)
            self.assertGreater(float(dynamic_npz["X_dynamic"][7, 0]), 1.0)

    def test_factory_auto_enables_multisource_when_data_path_requires_it(self) -> None:
        cfg = {
            "data": {
                "multi_source": {"enabled": True},
                "use_bathymetry": False,
            },
            "model": {
                "architecture": "coastal_transformer",
                "coastal_transformer": {
                    "model_dim": 32,
                    "num_layers": 1,
                    "num_heads": 4,
                    "multi_source": {"enabled": False, "aggregation": "attention"},
                },
            },
        }
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = build_model_from_config(
                config=cfg,
                dynamic_input_dim=0,
                static_input_dim=3,
                output_dim=4,
                source_dynamic_input_dim=7,
                source_geometry_input_dim=7,
            )
        self.assertTrue(bool(getattr(model, "use_multi_source", False)))
        self.assertTrue(
            any("Auto-enabling the model multi-source path" in str(w.message) for w in caught)
        )

    def test_factory_disables_runtime_geometry_by_default(self) -> None:
        cfg = {
            "data": {
                "multi_source": {"enabled": True},
                "use_bathymetry": False,
            },
            "model": {
                "architecture": "coastal_transformer",
                "coastal_transformer": {
                    "model_dim": 32,
                    "num_layers": 1,
                    "num_heads": 4,
                    "multi_source": {"enabled": True, "aggregation": "attention"},
                },
            },
        }
        model = build_model_from_config(
            config=cfg,
            dynamic_input_dim=0,
            static_input_dim=3,
            output_dim=4,
            source_dynamic_input_dim=7,
            source_geometry_input_dim=7,
        )
        self.assertTrue(bool(getattr(model, "use_multi_source", False)))
        self.assertFalse(bool(getattr(model, "use_source_geometry_features", True)))


if __name__ == "__main__":
    unittest.main()
