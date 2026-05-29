from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


@dataclass
class ClusteringResult:
    clustered_df: pd.DataFrame
    best_k: int
    best_score: float
    summary_df: pd.DataFrame


def train_clustering(features_df: pd.DataFrame, max_clusters: int = 8, random_seed: int = 42) -> ClusteringResult:
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

    clustered = features_df.copy()
    clustered["Risk_Cluster"] = best_model.labels_

    ordering = clustered.groupby("Risk_Cluster")["Capital_Ratio"].mean().sort_values().index
    mapping = {old_id: f"Cluster {new_id}" for new_id, old_id in enumerate(ordering)}
    clustered["Risk_Cluster"] = clustered["Risk_Cluster"].map(mapping)

    summary = clustered.groupby("Risk_Cluster").mean(numeric_only=True).round(6)
    return ClusteringResult(clustered_df=clustered, best_k=best_k, best_score=best_score, summary_df=summary)
