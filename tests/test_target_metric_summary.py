"""Test target metric summary."""

from __future__ import annotations

import math
import unittest

import numpy as np

from src.metrics import (
    average_target_metric_rows,
    compute_site_target_metric_rows,
    compute_split_target_metric_rows,
)


def _angle_pair(deg: float) -> tuple[float, float]:
    rad = np.deg2rad(float(deg))
    return float(np.sin(rad)), float(np.cos(rad))


class TargetMetricSummaryTests(unittest.TestCase):
    def test_site_rows_include_new_metric_columns(self) -> None:
        dir0 = _angle_pair(0.0)
        dir20 = _angle_pair(20.0)
        dir30 = _angle_pair(30.0)
        dir60 = _angle_pair(60.0)

        y_true = np.asarray(
            [
                [1.0, 5.0, *dir0, *dir0],
                [1.5, 6.0, *dir20, *dir20],
                [2.0, 7.0, *dir30, *dir30],
                [2.5, 8.0, *dir60, *dir60],
            ],
            dtype=float,
        )
        y_pred = np.asarray(
            [
                [1.1, 5.2, *dir20, *dir20],
                [1.4, 5.8, *dir30, *dir30],
                [2.3, 7.5, *dir60, *dir60],
                [2.2, 7.6, *dir30, *dir30],
            ],
            dtype=float,
        )
        site_names = ["site_a", "site_a", "site_b", "site_b"]

        site_rows = compute_site_target_metric_rows(
            y_true=y_true,
            y_pred=y_pred,
            site_names=site_names,
            split="test",
        )

        self.assertEqual(len(site_rows), 8)
        hs_row = next(row for row in site_rows if row["site"] == "site_a" and row["target"] == "hs")
        self.assertIn("mse", hs_row)
        self.assertIn("bias", hs_row)
        self.assertIn("pearson_r", hs_row)
        self.assertIn("r2", hs_row)
        self.assertEqual(hs_row["mse_unit"], "m2")
        self.assertEqual(hs_row["rmse_unit"], "m")
        self.assertEqual(hs_row["bias_unit"], "m")

        dir_row = next(
            row for row in site_rows if row["site"] == "site_b" and row["target"] == "dir"
        )
        self.assertEqual(dir_row["mse_unit"], "rad2")
        self.assertEqual(dir_row["rmse_unit"], "deg")
        self.assertEqual(dir_row["bias_unit"], "deg")
        self.assertTrue(math.isfinite(float(dir_row["mse"])))
        self.assertTrue(math.isfinite(float(dir_row["rmse"])))
        self.assertTrue(math.isfinite(float(dir_row["bias"])))

    def test_split_aggregate_rows_recompute_metrics_from_all_samples(self) -> None:
        perfect = _angle_pair(0.0)

        y_true = np.asarray(
            [
                [0.0, 5.0, *perfect, *perfect],
                [1.0, 6.0, *perfect, *perfect],
                [0.0, 7.0, *perfect, *perfect],
                [10.0, 8.0, *perfect, *perfect],
            ],
            dtype=float,
        )
        y_pred = np.asarray(
            [
                [0.0, 5.0, *perfect, *perfect],
                [1.0, 6.0, *perfect, *perfect],
                [0.0, 7.0, *perfect, *perfect],
                [0.0, 8.0, *perfect, *perfect],
            ],
            dtype=float,
        )
        site_names = ["site_a", "site_a", "site_b", "site_b"]

        site_rows = compute_site_target_metric_rows(
            y_true=y_true,
            y_pred=y_pred,
            site_names=site_names,
            split="val",
        )
        averaged_rows = average_target_metric_rows(
            site_rows,
            split="val",
            aggregation="site_average",
            site="ALL_VALIDATION_SITES",
        )
        pooled_rows = compute_split_target_metric_rows(
            y_true=y_true,
            y_pred=y_pred,
            site_names=site_names,
            split="val",
            aggregation="split_aggregate",
            site="ALL_VALIDATION_SITES",
        )

        avg_hs = next(row for row in averaged_rows if row["target"] == "hs")
        pooled_hs = next(row for row in pooled_rows if row["target"] == "hs")

        self.assertEqual(pooled_hs["aggregation"], "split_aggregate")
        self.assertEqual(pooled_hs["site_count"], 2)
        self.assertEqual(pooled_hs["sample_count"], 4)
        self.assertAlmostEqual(float(avg_hs["r2"]), 0.0)
        self.assertAlmostEqual(float(pooled_hs["r2"]), -0.4134275618374558)
        self.assertNotAlmostEqual(float(pooled_hs["r2"]), float(avg_hs["r2"]))
        self.assertAlmostEqual(float(pooled_hs["mse"]), 25.0)
        self.assertAlmostEqual(float(pooled_hs["rmse"]), 5.0)
        self.assertAlmostEqual(float(pooled_hs["bias"]), -2.5)


if __name__ == "__main__":
    unittest.main()
