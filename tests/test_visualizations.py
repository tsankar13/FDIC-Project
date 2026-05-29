import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from fdic_ml_pipeline.visualizations import _latest_artifact_run, generate_visualizations


class VisualizationTests(unittest.TestCase):
    def _seed_artifacts(self, run_dir: Path) -> None:
        clustered = pd.DataFrame(
            {
                "Capital_Ratio": [0.08, 0.09, 0.22, 0.24],
                "Net_Interest_Margin": [0.01, 0.015, 0.03, 0.035],
                "Texas_Ratio": [0.7, 0.6, 0.2, 0.15],
                "Charge_Off_Ratio": [0.02, 0.018, 0.007, 0.005],
                "Risk_Cluster": ["Cluster 0", "Cluster 0", "Cluster 1", "Cluster 1"],
            }
        )
        clustered.to_csv(run_dir / "clustered_features.csv", index=False)

        summary = pd.DataFrame(
            {
                "Capital_Ratio": [0.085, 0.23],
                "Net_Interest_Margin": [0.0125, 0.0325],
                "Texas_Ratio": [0.65, 0.175],
                "Charge_Off_Ratio": [0.019, 0.006],
            },
            index=["Cluster 0", "Cluster 1"],
        )
        summary.to_csv(run_dir / "cluster_summary.csv")

        correlations = pd.DataFrame(
            [
                {
                    "feature_left": "Capital_Ratio",
                    "feature_right": "Texas_Ratio",
                    "correlation": -0.88,
                    "abs_correlation": 0.88,
                },
                {
                    "feature_left": "Capital_Ratio",
                    "feature_right": "Charge_Off_Ratio",
                    "correlation": -0.62,
                    "abs_correlation": 0.62,
                },
            ]
        )
        correlations.to_csv(run_dir / "synopsis_correlations.csv", index=False)

        anomalies = pd.DataFrame({"anomaly_flag": [1, -1, 1, 1], "anomaly_score": [0.1, -0.3, 0.2, 0.15]})
        anomalies.to_csv(run_dir / "synopsis_anomalies.csv", index=False)

        metrics = {
            "rows_analyzed": 4,
            "features_analyzed": 4,
            "pca_components": 3,
            "pca_explained_variance_pc1": 0.63,
            "pca_explained_variance_top3": 0.91,
            "kmeans_k2_silhouette": 0.7,
            "agglomerative_k2_silhouette": 0.66,
            "anomaly_count": 1,
            "anomaly_share": 0.25,
        }
        (run_dir / "synopsis_metrics.json").write_text(json.dumps(metrics))

    def test_latest_artifact_run_selects_newest_with_known_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old = root / "20250101T000000Z_both"
            new = root / "20250201T000000Z_synopsis"
            old.mkdir(parents=True)
            new.mkdir(parents=True)
            (old / "clustered_features.csv").write_text("Risk_Cluster\nCluster 0\n")
            (new / "synopsis_metrics.json").write_text("{}")

            selected = _latest_artifact_run(root)
            self.assertEqual(selected, new)

    def test_generate_visualizations_creates_figures_and_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_dir = root / "artifacts"
            report_dir = root / "reports"
            figures_dir = report_dir / "figures"
            run_dir = output_dir / "20260301T010101Z_both"
            run_dir.mkdir(parents=True)
            self._seed_artifacts(run_dir)

            result = generate_visualizations(
                output_dir=output_dir,
                report_dir=report_dir,
                figures_dir=figures_dir,
                run_dir=run_dir,
            )

            self.assertEqual(result.run_dir, run_dir)
            self.assertTrue(result.figure_paths)
            self.assertEqual(result.skipped_charts, [])
            self.assertTrue(result.report_path.exists())
            for figure in result.figure_paths:
                self.assertTrue(figure.exists())
            content = result.report_path.read_text()
            self.assertIn("Visualization Report", content)
            self.assertIn("Generated Figures", content)
            self.assertIn("Cluster Size", content)

    def test_generate_visualizations_skips_missing_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_dir = root / "artifacts"
            report_dir = root / "reports"
            figures_dir = report_dir / "figures"
            run_dir = output_dir / "20260301T010101Z_clustering"
            run_dir.mkdir(parents=True)
            pd.DataFrame({"Risk_Cluster": ["Cluster 0", "Cluster 1"]}).to_csv(
                run_dir / "clustered_features.csv", index=False
            )

            result = generate_visualizations(
                output_dir=output_dir,
                report_dir=report_dir,
                figures_dir=figures_dir,
                run_dir=run_dir,
            )

            self.assertGreaterEqual(len(result.skipped_charts), 1)
            self.assertTrue(any(item.startswith("top_correlations") for item in result.skipped_charts))
            self.assertTrue(result.report_path.exists())
            content = result.report_path.read_text()
            self.assertIn("Skipped Charts", content)


if __name__ == "__main__":
    unittest.main()
