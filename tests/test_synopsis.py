import unittest

import pandas as pd

from fdic_ml_pipeline.synopsis import build_data_synopsis


class SynopsisTests(unittest.TestCase):
    def test_build_data_synopsis_outputs_expected_artifacts(self) -> None:
        features = pd.DataFrame(
            {
                "Capital_Ratio": [0.05, 0.06, 0.07, 0.2, 0.21, 0.22, 0.08, 0.24],
                "Net_Interest_Margin": [0.01, 0.012, 0.013, 0.04, 0.041, 0.039, 0.015, 0.043],
                "Texas_Ratio": [0.8, 0.75, 0.7, 0.2, 0.18, 0.19, 0.65, 0.17],
                "Charge_Off_Ratio": [0.02, 0.019, 0.018, 0.005, 0.006, 0.004, 0.017, 0.004],
            }
        )

        result = build_data_synopsis(features, random_seed=42)
        self.assertIn("kmeans_k2_silhouette", result.metrics)
        self.assertIn("agglomerative_k2_silhouette", result.metrics)
        self.assertIn("anomaly_share", result.metrics)
        self.assertFalse(result.pca_components_df.empty)
        self.assertFalse(result.anomaly_df.empty)
        self.assertIn("anomaly_flag", result.anomaly_df.columns)
        self.assertIn("anomaly_score", result.anomaly_df.columns)
        self.assertFalse(result.correlation_pairs_df.empty)
        self.assertIn("feature_left", result.correlation_pairs_df.columns)
        self.assertIn("feature_right", result.correlation_pairs_df.columns)


if __name__ == "__main__":
    unittest.main()
