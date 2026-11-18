import unittest

import pandas as pd

from analyze_external_dataset import calculate_wqi_interval


class ExternalDatasetTests(unittest.TestCase):
    def test_interval_contains_ideal_sample_wqi(self):
        frame = pd.DataFrame({
            "station_code": [1], "location": ["Test river"], "state": ["TEST"],
            "year": [2023],
            "ph_min": [7.0], "ph_max": [7.0],
            "do_min": [14.6], "do_max": [14.6],
            "bod_min": [0.0], "bod_max": [0.0],
            "cond_min": [0.0], "cond_max": [0.0],
            "nitrate_min": [0.0], "nitrate_max": [0.0],
            "fc_min": [0.0], "fc_max": [0.0],
            "tc_min": [0.0], "tc_max": [0.0],
        })
        result = calculate_wqi_interval(frame)
        self.assertAlmostEqual(result.iloc[0]["wqi_lower_bound"], 0.0)
        self.assertAlmostEqual(result.iloc[0]["wqi_upper_bound"], 0.0)
        self.assertFalse(bool(result.iloc[0]["category_is_ambiguous"]))

    def test_wider_parameter_ranges_do_not_narrow_wqi_interval(self):
        frame = pd.DataFrame({
            "station_code": [1], "location": ["Test river"], "state": ["TEST"],
            "year": [2023],
            "ph_min": [6.0], "ph_max": [8.0],
            "do_min": [5.0], "do_max": [10.0],
            "bod_min": [1.0], "bod_max": [2.0],
            "cond_min": [100.0], "cond_max": [200.0],
            "nitrate_min": [0.2], "nitrate_max": [1.0],
            "fc_min": [2.0], "fc_max": [100.0],
            "tc_min": [10.0], "tc_max": [100.0],
        })
        result = calculate_wqi_interval(frame)
        self.assertGreaterEqual(result.iloc[0]["wqi_upper_bound"],
                                result.iloc[0]["wqi_lower_bound"])
        self.assertGreaterEqual(result.iloc[0]["wqi_interval_width"], 0.0)


if __name__ == "__main__":
    unittest.main()
