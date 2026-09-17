"""Test date range filter runtime."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.point_centric_pipeline import _resolve_and_apply_date_range


class DateRangeFilterRuntimeTests(unittest.TestCase):
    def test_filtered_timestamps_stay_within_bounds(self) -> None:
        aligned_index = pd.date_range("2019-12-31 18:00:00", periods=12, freq="12h")
        cfg = {
            "date_range": {
                "enabled": True,
                "start": "2020-01-01",
                "end": "2020-01-03",
            }
        }

        filtered_index, meta = _resolve_and_apply_date_range(aligned_index, cfg)
        self.assertTrue(bool(meta.get("enabled", False)))
        self.assertGreater(len(filtered_index), 0)

        ts = filtered_index.to_numpy(dtype="datetime64[ns]")
        start = np.datetime64("2020-01-01T00:00:00")
        end = np.datetime64("2020-01-03T23:59:59.999999999")
        self.assertTrue(np.all(ts >= start))
        self.assertTrue(np.all(ts <= end))
        self.assertEqual(int(meta["timestamp_count_after_filter"]), int(len(filtered_index)))

    def test_zero_samples_after_filter_raises_clear_error(self) -> None:
        aligned_index = pd.date_range("2020-01-01 00:00:00", periods=4, freq="1h")
        cfg = {
            "date_range": {
                "enabled": True,
                "start": "2022-01-01",
                "end": "2022-01-02",
            }
        }

        with self.assertRaisesRegex(ValueError, "removed all samples"):
            _resolve_and_apply_date_range(aligned_index, cfg)


if __name__ == "__main__":
    unittest.main()
