# Automated Data Synopsis Report

## Overview
This auto-generated synopsis summarizes the latest ML-based structural diagnostics from `backend/artifacts/20260527T174229Z_both`.
The analysis covers `139` institutions and `4` engineered risk features.

## Narrative Synopsis
The feature space is compressible and structured: the top principal component explains `40.17%` of variance, while the first `3` components explain `100.00%`. This indicates that risk variation is driven by a relatively compact set of latent factors.
Cluster-separation diagnostics show the strongest two-group structure under `Agglomerative clustering (k=2)` with silhouette `0.3440` (KMeans: `0.2760`, Agglomerative: `0.3440`). Taken together, these methods suggest meaningful segmentation rather than random dispersion.
Anomaly detection flags `30` institutions (`21.58%`) as structurally unusual under Isolation Forest. This tail should be monitored as potential emerging stress or atypical balance-sheet behavior.

## Strongest Feature Relationships
- `Capital_Ratio` vs `Texas_Ratio`: correlation `-0.1396`
- `Net_Interest_Margin` vs `Texas_Ratio`: correlation `0.0839`
- `Capital_Ratio` vs `Net_Interest_Margin`: correlation `-0.0797`

## Artifacts Used
- `backend/artifacts/20260527T174229Z_both/synopsis_metrics.json`
- `backend/artifacts/20260527T174229Z_both/synopsis_pca_components.csv`
- `backend/artifacts/20260527T174229Z_both/synopsis_anomalies.csv`
- `backend/artifacts/20260527T174229Z_both/synopsis_correlations.csv`

## Next Actions
- Re-run synopsis after each quarterly FDIC update to monitor structural drift.
- Compare anomaly membership period-over-period to detect persistence vs transience.
- Use top correlation pairs to guide feature refinement and hypothesis testing.
