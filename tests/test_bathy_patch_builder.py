"""Test bathy patch builder."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import yaml

from src.preprocess.build_bathy_patches import build_bathymetry_patch_dataset


class _IdentityTransformer:
    def transform(self, lon: float, lat: float) -> tuple[float, float]:
        return float(lon), float(lat)


class BathyPatchBuilderTests(unittest.TestCase):
    def test_v2_depth_channel_uses_training_max_unit_interval(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            bathy_grid_path = base / "bathy_field_full.npz"
            out_path = base / "point_centric_X_bathy.npz"
            sites_path = base / "sites.yaml"

            np.savez_compressed(
                bathy_grid_path,
                x=np.array([0.0, 1.0, 2.0], dtype=np.float32),
                y=np.array([0.0, 1.0, 2.0], dtype=np.float32),
                z=np.array(
                    [
                        [0.0, 0.0, 0.0],
                        [2.0, 5.0, 8.0],
                        [0.0, 0.0, 0.0],
                    ],
                    dtype=np.float32,
                ),
                land_mask=np.array(
                    [
                        [True, True, True],
                        [False, False, False],
                        [True, True, True],
                    ],
                    dtype=bool,
                ),
                metadata=np.array({"vertical_convention": "positive_down"}, dtype=object),
            )

            sites_payload = {
                "nearshore_sites": [
                    {"name": "train_site", "lon": 1.0, "lat": 1.0},
                    {"name": "deeper_eval_site", "lon": 2.0, "lat": 1.0},
                    {"name": "land_site", "lon": 0.0, "lat": 0.0},
                ]
            }
            sites_path.write_text(yaml.safe_dump(sites_payload))

            preprocess_cfg = {
                "bathy": {
                    "full_grid_path": str(bathy_grid_path),
                    "version": "v2",
                    "in_channels": 6,
                    "patch_size": 1,
                    "resolution_m": 50.0,
                    "depth_clip_m": 120.0,
                    "max_snap_cells": 0,
                    "shallow_breaking_depth_m": 5.0,
                    "channels": [
                        "depth",
                        "land_sea_mask",
                        "slope_magnitude",
                        "distance_to_land",
                        "curvature_laplacian",
                        "shallow_breaking_mask",
                    ],
                }
            }

            with mock.patch(
                "src.preprocess.build_bathy_patches._get_transformer",
                return_value=_IdentityTransformer(),
            ):
                info = build_bathymetry_patch_dataset(
                    sites_yaml=str(sites_path),
                    preprocess_cfg=preprocess_cfg,
                    target_sites=["train_site", "deeper_eval_site", "land_site"],
                    train_sites=["train_site"],
                    out_path=str(out_path),
                )

            self.assertEqual(info["shape"], [3, 6, 1, 1])
            self.assertEqual(
                info["normalization_metadata"]["stats"]["depth"]["method"],
                "unit_interval_train_max",
            )
            self.assertAlmostEqual(
                float(info["normalization_metadata"]["stats"]["depth"]["train_max"]),
                5.0,
                places=6,
            )

            with np.load(out_path, allow_pickle=True) as npz:
                x_bathy = np.asarray(npz["X_bathy"], dtype=np.float32)
                normalization_metadata = npz["normalization_metadata"].item()

            self.assertAlmostEqual(float(x_bathy[0, 0, 0, 0]), 1.0, places=6)
            self.assertAlmostEqual(float(x_bathy[1, 0, 0, 0]), 1.0, places=6)
            self.assertAlmostEqual(float(x_bathy[2, 0, 0, 0]), 0.0, places=6)
            self.assertAlmostEqual(float(x_bathy[0, 1, 0, 0]), 1.0, places=6)
            self.assertAlmostEqual(float(x_bathy[2, 1, 0, 0]), 0.0, places=6)
            self.assertEqual(
                normalization_metadata["stats"]["depth"]["method"], "unit_interval_train_max"
            )
            self.assertAlmostEqual(
                float(normalization_metadata["stats"]["depth"]["train_max"]), 5.0, places=6
            )
