import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_external_dataset import (
    STANDARDS,
    WEIGHTS,
    calculate_wqi_interval,
    load_external_data,
)
from run_experiments import WQI_STANDARDS, calculate_wqi


def interval_frame(**overrides):
    values = {
        "station_code": [1], "location": ["Test river"], "state": ["TEST"],
        "year": [2023],
        "ph_min": [7.0], "ph_max": [7.0],
        "do_min": [14.6], "do_max": [14.6],
        "bod_min": [0.0], "bod_max": [0.0],
        "cond_min": [0.0], "cond_max": [0.0],
        "nitrate_min": [0.0], "nitrate_max": [0.0],
        "fc_min": [0.0], "fc_max": [0.0],
        "tc_min": [0.0], "tc_max": [0.0],
    }
    values.update(overrides)
    return pd.DataFrame(values)


class ExternalDatasetTests(unittest.TestCase):
    def test_external_analysis_reuses_primary_project_standards(self):
        self.assertEqual(STANDARDS, WQI_STANDARDS)
        self.assertAlmostEqual(sum(WEIGHTS.values()), 1.0, places=12)

    def test_ideal_values_produce_zero_wqi_envelope(self):
        result = calculate_wqi_interval(interval_frame())
        self.assertAlmostEqual(result.iloc[0]["wqi_lower_bound"], 0.0)
        self.assertAlmostEqual(result.iloc[0]["wqi_upper_bound"], 0.0)
        self.assertEqual(result.iloc[0]["category_lower_bound"], "Good / Excellent")
        self.assertFalse(bool(result.iloc[0]["category_is_ambiguous"]))

    def test_standard_limit_values_match_primary_wqi(self):
        sample = interval_frame(
            ph_min=[8.5], ph_max=[8.5], do_min=[5.0], do_max=[5.0],
            bod_min=[3.0], bod_max=[3.0], cond_min=[2250.0], cond_max=[2250.0],
            nitrate_min=[10.0], nitrate_max=[10.0],
            fc_min=[500.0], fc_max=[500.0], tc_min=[500.0], tc_max=[500.0],
        )
        result = calculate_wqi_interval(sample)
        point = pd.DataFrame({
            "ph": [8.5], "do": [5.0], "bod": [3.0], "cond": [2250.0],
            "nitrate": [10.0], "fc": [500.0], "tc": [500.0],
        })
        expected, _ = calculate_wqi(point)
        self.assertAlmostEqual(result.iloc[0]["wqi_lower_bound"], expected.iloc[0]["wqi"])
        self.assertAlmostEqual(result.iloc[0]["wqi_upper_bound"], expected.iloc[0]["wqi"])
        self.assertEqual(result.iloc[0]["category_lower_bound"], "Very Poor")

    def test_pH_interval_crossing_ideal_has_zero_lower_subindex(self):
        result = calculate_wqi_interval(interval_frame(ph_min=[6.0], ph_max=[8.0]))
        self.assertGreaterEqual(result.iloc[0]["wqi_lower_bound"], 0.0)
        self.assertGreater(result.iloc[0]["wqi_upper_bound"], result.iloc[0]["wqi_lower_bound"])

    def test_missing_parameter_keeps_wqi_missing_not_imputed(self):
        frame = interval_frame(bod_min=[np.nan], bod_max=[np.nan])
        result = calculate_wqi_interval(frame)
        self.assertTrue(np.isnan(result.iloc[0]["wqi_lower_bound"]))
        self.assertTrue(pd.isna(result.iloc[0]["category_lower_bound"]))

    def test_loader_reports_incomplete_and_excludes_invalid_bounds(self):
        raw = pd.DataFrame({
            "Station Code": [1, 2, 3],
            "Monitoring Location": ["A", "B", "C"],
            "State": ["X", "X", "X"],
            "Year": [2022, 2022, 2022],
            "pH Min": [7, 7, 15], "pH Max": [7, 7, 15],
            "Dissolved Oxygen (mg/l) Min": [8, 8, 8],
            "Dissolved Oxygen (mg/l) Max": [9, 9, 9],
            "BOD (mg/l) Min": [1, "-", 1], "BOD (mg/l) Max": [2, "-", 2],
            "Conductivity (µmho/cm) Min": [100, 100, 100],
            "Conductivity (µmho/cm) Max": [200, 200, 200],
            "Nitrate N (mg/l) Min": [1, 1, 1], "Nitrate N (mg/l) Max": [2, 2, 2],
            "Fecal Coliform (MPN/100ml) Min": [10, 10, 10],
            "Fecal Coliform (MPN/100ml) Max": [20, 20, 20],
            "Total Coliform (MPN/100ml) Min": [10, 10, 10],
            "Total Coliform (MPN/100ml) Max": [20, 20, 20],
        })
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.csv"
            raw.to_csv(path, index=False)
            loaded, stats = load_external_data(path)
        self.assertEqual(len(loaded), 2)
        self.assertEqual(stats["source_rows"], 3)
        self.assertEqual(stats["invalid_bound_rows_excluded"], 1)
        self.assertEqual(stats["complete_wqi_rows"], 1)
        self.assertEqual(stats["incomplete_wqi_rows"], 1)

    def test_reversed_bounds_are_rejected(self):
        raw = interval_frame(ph_min=[8.0], ph_max=[7.0])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.csv"
            # Build source-shaped input from the canonical test frame.
            source = pd.DataFrame({
                "station code": [1], "monitoring location": ["A"], "state": ["X"],
                "year": [2023], "pH min": [8.0], "pH max": [7.0],
                "Dissolved Oxygen (mg/l) min": [5], "Dissolved Oxygen (mg/l) max": [8],
                "BOD (mg/l) min": [1], "BOD (mg/l) max": [2],
                "Conductivity (µmho/cm) min": [100], "Conductivity (µmho/cm) max": [200],
                "Nitrate N (mg/l) min": [1], "Nitrate N (mg/l) max": [2],
                "Fecal Coliform (MPN/100ml) min": [10], "Fecal Coliform (MPN/100ml) max": [20],
                "Total Coliform (MPN/100ml) min": [10], "Total Coliform (MPN/100ml) max": [20],
            })
            source.to_csv(path, index=False)
            loaded, stats = load_external_data(path)
        self.assertTrue(loaded.empty)
        self.assertEqual(stats["invalid_bound_rows_excluded"], 1)


if __name__ == "__main__":
    unittest.main()
