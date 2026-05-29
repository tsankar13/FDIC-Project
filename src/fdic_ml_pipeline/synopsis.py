from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


@dataclass
class SynopsisResult:
    metrics: dict[str, float | int]
    pca_components_df: pd.DataFrame
    anomaly_df: pd.DataFrame
    correlation_pairs_df: pd.DataFrame


def _top_correlation_pairs(features_df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    corr = features_df.corr(numeric_only=True)
    rows: list[dict[str, float | str]] = []

    for left, right in combinations(corr.columns, 2):
        value = float(corr.loc[left, right])
        rows.append(
            {
                "feature_left": left,
                "feature_right": right,
                "correlation": value,
                "abs_correlation": abs(value),
            }
        )

    pairs = pd.DataFrame(rows)
    if pairs.empty:
        return pairs
    return pairs.sort_values("abs_correlation", ascending=False).head(top_n)


def build_data_synopsis(features_df: pd.DataFrame, random_seed: int = 42) -> SynopsisResult:
    if len(features_df) < 5:
        raise ValueError("Need at least 5 rows to build a data synopsis.")

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(features_df)

    pca_components = min(3, features_df.shape[1])
    pca = PCA(n_components=pca_components, random_state=random_seed)
    pca_scores = pca.fit_transform(x_scaled)
    pca_columns = [f"PC{i}" for i in range(1, pca_components + 1)]
    pca_components_df = pd.DataFrame(pca_scores, index=features_df.index, columns=pca_columns)

    anomaly_model = IsolationForest(random_state=random_seed, contamination="auto")
    anomaly_flags = anomaly_model.fit_predict(x_scaled)
    anomaly_scores = anomaly_model.decision_function(x_scaled)
    anomaly_df = pd.DataFrame(
        {
            "anomaly_flag": anomaly_flags,
            "anomaly_score": anomaly_scores,
        },
        index=features_df.index,
    )

    kmeans = KMeans(n_clusters=2, random_state=random_seed, n_init=10)
    kmeans_labels = kmeans.fit_predict(x_scaled)
    kmeans_silhouette = float(silhouette_score(x_scaled, kmeans_labels))

    agg = AgglomerativeClustering(n_clusters=2)
    agg_labels = agg.fit_predict(x_scaled)
    agg_silhouette = float(silhouette_score(x_scaled, agg_labels))

    correlation_pairs_df = _top_correlation_pairs(features_df)
    anomaly_count = int((anomaly_flags == -1).sum())

    metrics: dict[str, float | int] = {
        "rows_analyzed": int(len(features_df)),
        "features_analyzed": int(features_df.shape[1]),
        "pca_components": int(pca_components),
        "pca_explained_variance_pc1": float(pca.explained_variance_ratio_[0]),
        "pca_explained_variance_top3": float(pca.explained_variance_ratio_.sum()),
        "kmeans_k2_silhouette": kmeans_silhouette,
        "agglomerative_k2_silhouette": agg_silhouette,
        "anomaly_count": anomaly_count,
        "anomaly_share": float(anomaly_count / len(features_df)),
    }
    return SynopsisResult(
        metrics=metrics,
        pca_components_df=pca_components_df,
        anomaly_df=anomaly_df,
        correlation_pairs_df=correlation_pairs_df,
    )
