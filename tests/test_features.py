import unittest

import pandas as pd

from fdic_ml_pipeline.features import engineer_fdic_features, remove_outliers_by_quantile


class FeatureEngineeringTests(unittest.TestCase):
    def test_engineer_features_builds_expected_columns(self) -> None:
        frame = pd.DataFrame(
            {
                "RBCT1J": [100.0, 200.0],
                "ASSET": [1000.0, 2500.0],
                "INTINC": [50.0, 120.0],
                "EINTEXP": [10.0, 20.0],
                "NAASSET": [5.0, 10.0],
                "P9ASSET": [2.0, 3.0],
                "NTLS": [4.0, 6.0],
            },
            index=[1, 2],
        )
        out = engineer_fdic_features(frame)
        self.assertEqual(
            set(out.columns),
            {"Capital_Ratio", "Net_Interest_Margin", "Texas_Ratio", "Charge_Off_Ratio"},
        )
        self.assertAlmostEqual(out.loc[1, "Capital_Ratio"], 0.1)

    def test_remove_outliers_drops_extreme_row(self) -> None:
        features = pd.DataFrame(
            {
                "Capital_Ratio": [0.1, 0.11, 10.0],
                "Net_Interest_Margin": [0.02, 0.025, 7.5],
                "Texas_Ratio": [0.3, 0.28, 20.0],
                "Charge_Off_Ratio": [0.005, 0.004, 9.0],
            },
            index=[1, 2, 3],
        )
        filtered = remove_outliers_by_quantile(features, 0.0, 0.90)
        self.assertNotIn(3, filtered.index)


if __name__ == "__main__":
    unittest.main()
