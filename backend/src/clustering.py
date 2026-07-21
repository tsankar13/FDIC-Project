"""Unsupervised risk segmentation via KMeans with silhouette-based k selection."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler


@dataclass
class ClusteringResult:
    """Outputs of a clustering run: labelled rows, chosen k, score, and profiles."""

    clustered_df: pd.DataFrame
    best_k: int
    best_score: float
    summary_df: pd.DataFrame
    quality_metrics: dict[str, float] = field(default_factory=dict)


def _bootstrap_stability(
    x_scaled: np.ndarray, best_k: int, base_labels: np.ndarray, random_seed: int, n_boot: int = 20
) -> tuple[float, float]:
    """Mean/std adjusted Rand index between the base labels and bootstrap re-fits.

    Each bootstrap resamples rows with replacement, re-fits KMeans, and re-predicts on
    the full data; agreement with the original labels (ARI) measures how reproducible
    the clustering is. ~1.0 = very stable, ~0 = noise.
    """
    rng = np.random.default_rng(random_seed)
    n = len(x_scaled)
    scores: list[float] = []
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        model = KMeans(n_clusters=best_k, random_state=random_seed + b, n_init=10)
        model.fit(x_scaled[idx])
        scores.append(float(adjusted_rand_score(base_labels, model.predict(x_scaled))))
    return float(np.mean(scores)), float(np.std(scores))


def train_clustering(features_df: pd.DataFrame, max_clusters: int = 8, random_seed: int = 42) -> ClusteringResult:
    """Standardize features, sweep k in [2, max_clusters], keep the best silhouette.

    The winning KMeans labels are re-mapped so ``Cluster 0`` is the lowest-average
    Capital_Ratio (most stressed) segment, making cluster ids interpretable, and a
    per-cluster mean profile is returned alongside the labelled rows. Beyond the
    selection silhouette, three corroborating quality metrics are reported: Calinski-
    Harabasz, Davies-Bouldin, and a bootstrap stability score (mean/std ARI).
    """
    if len(features_df) < 3:
        raise ValueError("Need at least 3 rows for clustering.")

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(features_df)

    best_k = 2
    best_score = -1.0
    best_model: KMeans | None = None

    upper_k = min(max_clusters, len(features_df) - 1)
    for k in range(2, upper_k + 1):
        model = KMeans(n_clusters=k, random_state=random_seed, n_init=10)
        labels = model.fit_predict(x_scaled)
        score = silhouette_score(x_scaled, labels)
        if score > best_score:
            best_score = score
            best_k = k
            best_model = model

    if best_model is None:
        raise RuntimeError("KMeans model selection failed.")

    base_labels = best_model.labels_
    stability_mean, stability_std = _bootstrap_stability(x_scaled, best_k, base_labels, random_seed)
    quality_metrics = {
        "silhouette": float(best_score),
        "calinski_harabasz": float(calinski_harabasz_score(x_scaled, base_labels)),
        "davies_bouldin": float(davies_bouldin_score(x_scaled, base_labels)),
        "bootstrap_stability_ari_mean": stability_mean,
        "bootstrap_stability_ari_std": stability_std,
    }

    clustered = features_df.copy()
    clustered["Risk_Cluster"] = base_labels

    ordering = clustered.groupby("Risk_Cluster")["Capital_Ratio"].mean().sort_values().index
    mapping = {old_id: f"Cluster {new_id}" for new_id, old_id in enumerate(ordering)}
    clustered["Risk_Cluster"] = clustered["Risk_Cluster"].map(mapping)

    summary = clustered.groupby("Risk_Cluster").mean(numeric_only=True).round(6)
    return ClusteringResult(
        clustered_df=clustered,
        best_k=best_k,
        best_score=best_score,
        summary_df=summary,
        quality_metrics=quality_metrics,
    )
