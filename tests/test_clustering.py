import unittest

import pandas as pd

from fdic_ml_pipeline.clustering import train_clustering


class ClusteringTests(unittest.TestCase):
    def test_cluster_training_returns_labels_and_metrics(self) -> None:
        features = pd.DataFrame(
            {
                "Capital_Ratio": [0.05, 0.055, 0.06, 0.20, 0.21, 0.22],
                "Net_Interest_Margin": [0.01, 0.012, 0.011, 0.04, 0.041, 0.039],
                "Texas_Ratio": [0.8, 0.82, 0.79, 0.2, 0.18, 0.19],
                "Charge_Off_Ratio": [0.02, 0.018, 0.019, 0.005, 0.006, 0.004],
            },
            index=[10, 11, 12, 20, 21, 22],
        )
        result = train_clustering(features, max_clusters=4, random_seed=42)
        self.assertIn("Risk_Cluster", result.clustered_df.columns)
        self.assertGreaterEqual(result.best_k, 2)
        self.assertGreater(result.best_score, 0.0)
        self.assertTrue(result.summary_df.shape[0] >= 2)


if __name__ == "__main__":
    unittest.main()
