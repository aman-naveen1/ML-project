import unittest

import pandas as pd

from run_experiments import (
    WQI_STANDARDS,
    calculate_wqi,
    transform_features,
)


class WQITests(unittest.TestCase):
    def setUp(self):
        self.sample = pd.DataFrame(
            {
                "station_code": [101, 102],
                "year": [2010, 2011],
                "temp": [25.0, 26.0],
                "do": [14.6, 5.0],
                "ph": [7.0, 8.5],
                "cond": [0.0, 2250.0],
                "bod": [0.0, 3.0],
                "nitrate": [0.0, 10.0],
                "fc": [0.0, 500.0],
                "tc": [0.0, 500.0],
            }
        )

    def test_weights_sum_to_one(self):
        _, weights = calculate_wqi(self.sample)
        self.assertAlmostEqual(sum(weights.values()), 1.0, places=10)
        self.assertEqual(set(weights), set(WQI_STANDARDS))

    def test_ideal_values_produce_good_category(self):
        result, _ = calculate_wqi(self.sample.iloc[[0]])
        self.assertAlmostEqual(result.iloc[0]["wqi"], 0.0, places=8)
        self.assertEqual(str(result.iloc[0]["category"]), "Good / Excellent")

    def test_standard_limit_values_produce_very_poor_category(self):
        result, _ = calculate_wqi(self.sample.iloc[[1]])
        self.assertAlmostEqual(result.iloc[0]["wqi"], 100.0, places=8)
        self.assertEqual(str(result.iloc[0]["category"]), "Very Poor")

    def test_log_transform_keeps_feature_columns(self):
        transformed = transform_features(self.sample, ["temp", "cond", "bod"])
        self.assertEqual(list(transformed.columns), ["temp", "cond", "bod"])
        self.assertAlmostEqual(transformed.iloc[0]["cond"], 0.0)
        self.assertAlmostEqual(transformed.iloc[1]["cond"], __import__("numpy").log1p(2250.0))


if __name__ == "__main__":
    unittest.main()
