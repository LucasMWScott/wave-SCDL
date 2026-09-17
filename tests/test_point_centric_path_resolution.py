"""Test point centric path resolution."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.data_pipeline import load_point_centric_arrays


REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS_DIR = REPO_ROOT / "notebooks"


class PointCentricPathResolutionTests(unittest.TestCase):
    def test_loader_resolves_repo_relative_data_dir_from_notebooks_cwd(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO_ROOT, prefix="tmp_point_centric_") as tmpdir:
            base = Path(tmpdir)
            rel_base = base.relative_to(REPO_ROOT)

            timestamps = np.array(
                [
                    "2020-01-01T00:00:00",
                    "2020-01-01T01:00:00",
                    "2020-01-01T02:00:00",
                    "2020-01-01T03:00:00",
                ],
                dtype=str,
            )
            np.savez_compressed(
                base / "point_centric_X_dynamic.npz",
                X_dynamic=np.arange(8, dtype=np.float32).reshape(4, 2),
                timestamps=timestamps,
                train_idx=np.array([0, 1], dtype=int),
                val_idx=np.array([2], dtype=int),
                test_idx=np.array([3], dtype=int),
            )
            np.savez_compressed(
                base / "point_centric_Y_targets.npz",
                target_sites=np.array(["site_a"], dtype=str),
                target_feature_names=np.array(
                    ["hs", "tp", "dir_sin", "dir_cos", "dp_sin", "dp_cos"],
                    dtype=str,
                ),
                timestamps=timestamps,
                Y__site_a=np.ones((4, 6), dtype=np.float32),
            )
            (base / "point_centric_metadata.json").write_text(
                json.dumps({"dynamic_feature_names": ["dyn_0", "dyn_1"]}),
                encoding="utf-8",
            )

            prev_cwd = Path.cwd()
            try:
                os.chdir(NOTEBOOKS_DIR)
                arrays = load_point_centric_arrays(str(rel_base))
            finally:
                os.chdir(prev_cwd)

            self.assertEqual(arrays.x_dynamic.shape, (4, 2))
            self.assertEqual(arrays.target_sites, ["site_a"])


if __name__ == "__main__":
    unittest.main()
