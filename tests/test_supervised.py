import unittest

import pandas as pd

from fdic_ml_pipeline.supervised import train_supervised


class SupervisedTests(unittest.TestCase):
    def test_supervised_training_selects_model_and_outputs_predictions(self) -> None:
        features = pd.DataFrame(
            {
                "Capital_Ratio": [0.05, 0.06, 0.07, 0.2, 0.21, 0.22, 0.08, 0.24],
                "Net_Interest_Margin": [0.01, 0.012, 0.013, 0.04, 0.041, 0.039, 0.015, 0.043],
                "Texas_Ratio": [0.8, 0.75, 0.7, 0.2, 0.18, 0.19, 0.65, 0.17],
                "Charge_Off_Ratio": [0.02, 0.019, 0.018, 0.005, 0.006, 0.004, 0.017, 0.004],
            }
        )
        labels = pd.Series([1, 1, 1, 0, 0, 0, 1, 0], name="FAILED")
        result = train_supervised(features, labels, random_seed=42)
        self.assertIn(
            result.model_name,
            {"logistic_regression_l2", "logistic_regression_elastic_net", "decision_tree_shallow"},
        )
        self.assertIn("accuracy", result.metrics)
        self.assertFalse(result.predictions_df.empty)
        self.assertFalse(result.explanation_df.empty)
        self.assertIn("feature", result.explanation_df.columns)
        self.assertIn("importance_type", result.explanation_df.columns)


if __name__ == "__main__":
    unittest.main()
