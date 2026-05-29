import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from fdic_ml_pipeline.synopsis_report import generate_synopsis_report


class SynopsisReportTests(unittest.TestCase):
    def test_generate_synopsis_report_from_explicit_run_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "artifacts" / "20260424T010101Z_synopsis"
            reports_dir = root / "reports"
            run_dir.mkdir(parents=True, exist_ok=True)

            metrics = {
                "rows_analyzed": 8,
                "features_analyzed": 4,
                "pca_components": 3,
                "pca_explained_variance_pc1": 0.61,
                "pca_explained_variance_top3": 0.93,
                "kmeans_k2_silhouette": 0.75,
                "agglomerative_k2_silhouette": 0.68,
                "anomaly_count": 1,
                "anomaly_share": 0.125,
            }
            (run_dir / "synopsis_metrics.json").write_text(json.dumps(metrics))
            pd.DataFrame(
                [
                    {"feature_left": "Capital_Ratio", "feature_right": "Texas_Ratio", "correlation": -0.91},
                    {"feature_left": "Capital_Ratio", "feature_right": "Charge_Off_Ratio", "correlation": -0.60},
                ]
            ).to_csv(run_dir / "synopsis_correlations.csv", index=False)
            pd.DataFrame({"PC1": [0.1]}).to_csv(run_dir / "synopsis_pca_components.csv", index=False)
            pd.DataFrame({"anomaly_flag": [1], "anomaly_score": [0.05]}).to_csv(
                run_dir / "synopsis_anomalies.csv", index=False
            )

            report_path = generate_synopsis_report(
                output_dir=root / "artifacts",
                reports_dir=reports_dir,
                run_dir=run_dir,
            )
            self.assertTrue(report_path.exists())
            content = report_path.read_text()
            self.assertIn("Automated Data Synopsis Report", content)
            self.assertIn("KMeans (k=2)", content)
            self.assertIn("12.50%", content)


if __name__ == "__main__":
    unittest.main()
