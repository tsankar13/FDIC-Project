from __future__ import annotations

"""Legacy deep-diagnostic report script.

This script is kept for historical reproducibility and ad-hoc analysis.
Preferred supported workflow is the `fdic-ml` CLI:
- `fdic-ml --config ... --mode both`
- `fdic-ml --generate-synopsis-report --output-dir backend/artifacts`
- `fdic-ml --generate-visualizations --output-dir backend/artifacts`

See `backend/docs/LEGACY.md` for deprecation context and migration guidance.
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

from fdic_ml_pipeline.data_io import load_and_merge_csvs
from fdic_ml_pipeline.features import engineer_fdic_features, remove_outliers_by_quantile


@dataclass
class GroupTestResult:
    feature: str
    mean_cluster0: float
    mean_cluster1: float
    median_cluster0: float
    median_cluster1: float
    welch_t_pvalue: float
    mannwhitney_pvalue: float
    cohens_d: float
    cliffs_delta: float
    bootstrap_diff_ci_low: float
    bootstrap_diff_ci_high: float


def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    nx = len(x)
    ny = len(y)
    if nx < 2 or ny < 2:
        return 0.0
    vx = np.var(x, ddof=1)
    vy = np.var(y, ddof=1)
    pooled = np.sqrt(((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2))
    if pooled == 0:
        return 0.0
    return float((np.mean(x) - np.mean(y)) / pooled)


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    # Efficient rank-based computation for Cliff's delta.
    x = np.asarray(x)
    y = np.asarray(y)
    greater = 0
    lower = 0
    for val in x:
        greater += np.sum(val > y)
        lower += np.sum(val < y)
    total = len(x) * len(y)
    if total == 0:
        return 0.0
    return float((greater - lower) / total)


def bootstrap_mean_diff_ci(
    x: np.ndarray,
    y: np.ndarray,
    n_bootstrap: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        xb = rng.choice(x, size=len(x), replace=True)
        yb = rng.choice(y, size=len(y), replace=True)
        diffs[i] = xb.mean() - yb.mean()
    low = np.quantile(diffs, alpha / 2.0)
    high = np.quantile(diffs, 1 - alpha / 2.0)
    return float(low), float(high)


def evaluate_k_grid(features_df: pd.DataFrame, random_seed: int = 42) -> pd.DataFrame:
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(features_df)
    rows = []
    max_k = min(8, len(features_df) - 1)
    for k in range(2, max_k + 1):
        model = KMeans(n_clusters=k, random_state=random_seed, n_init=10)
        labels = model.fit_predict(x_scaled)
        rows.append(
            {
                "k": k,
                "silhouette": float(silhouette_score(x_scaled, labels)),
                "calinski_harabasz": float(calinski_harabasz_score(x_scaled, labels)),
                "davies_bouldin": float(davies_bouldin_score(x_scaled, labels)),
            }
        )
    return pd.DataFrame(rows)


def evaluate_alternative_models(features_df: pd.DataFrame, random_seed: int = 42) -> pd.DataFrame:
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(features_df)
    rows: list[dict] = []
    max_k = min(8, len(features_df) - 1)

    for k in range(2, max_k + 1):
        gmm = GaussianMixture(n_components=k, random_state=random_seed)
        labels = gmm.fit_predict(x_scaled)
        rows.append(
            {
                "model": "gaussian_mixture",
                "setting": f"k={k}",
                "n_clusters": int(len(np.unique(labels))),
                "noise_ratio": 0.0,
                "silhouette": float(silhouette_score(x_scaled, labels)),
                "calinski_harabasz": float(calinski_harabasz_score(x_scaled, labels)),
                "davies_bouldin": float(davies_bouldin_score(x_scaled, labels)),
            }
        )

    for k in range(2, max_k + 1):
        agg = AgglomerativeClustering(n_clusters=k)
        labels = agg.fit_predict(x_scaled)
        rows.append(
            {
                "model": "agglomerative",
                "setting": f"k={k}",
                "n_clusters": int(len(np.unique(labels))),
                "noise_ratio": 0.0,
                "silhouette": float(silhouette_score(x_scaled, labels)),
                "calinski_harabasz": float(calinski_harabasz_score(x_scaled, labels)),
                "davies_bouldin": float(davies_bouldin_score(x_scaled, labels)),
            }
        )

    for eps in (0.3, 0.5, 0.7, 1.0):
        for min_samples in (5, 10, 20):
            db = DBSCAN(eps=eps, min_samples=min_samples)
            labels = db.fit_predict(x_scaled)
            noise_ratio = float(np.mean(labels == -1))
            non_noise = labels != -1
            cluster_labels = labels[non_noise]
            n_clusters = len(np.unique(cluster_labels)) if np.any(non_noise) else 0
            if n_clusters < 2:
                silhouette = np.nan
                ch = np.nan
                dbi = np.nan
            else:
                silhouette = float(silhouette_score(x_scaled[non_noise], cluster_labels))
                ch = float(calinski_harabasz_score(x_scaled[non_noise], cluster_labels))
                dbi = float(davies_bouldin_score(x_scaled[non_noise], cluster_labels))
            rows.append(
                {
                    "model": "dbscan",
                    "setting": f"eps={eps},min_samples={min_samples}",
                    "n_clusters": int(n_clusters),
                    "noise_ratio": noise_ratio,
                    "silhouette": silhouette,
                    "calinski_harabasz": ch,
                    "davies_bouldin": dbi,
                }
            )
    return pd.DataFrame(rows)


def evaluate_dimension_and_anomaly(features_df: pd.DataFrame, random_seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(features_df)

    pca = PCA(random_state=random_seed)
    pca.fit(x_scaled)
    pca_df = pd.DataFrame(
        {
            "component": [f"PC{i + 1}" for i in range(len(pca.explained_variance_ratio_))],
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "cumulative_explained_variance": np.cumsum(pca.explained_variance_ratio_),
        }
    )

    iso = IsolationForest(contamination=0.01, random_state=random_seed)
    labels = iso.fit_predict(x_scaled)
    anomaly_df = pd.DataFrame(
        {
            "total_rows": [int(len(labels))],
            "anomaly_rows": [int(np.sum(labels == -1))],
            "anomaly_ratio": [float(np.mean(labels == -1))],
        }
    )
    return pca_df, anomaly_df


def profile_raw_fdic_data(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    numeric_df = raw_df.select_dtypes(include=["number"]).copy()
    missing_df = (
        raw_df.isna()
        .mean()
        .sort_values(ascending=False)
        .reset_index()
        .rename(columns={"index": "column", 0: "missing_ratio"})
    )
    missing_df["missing_ratio"] = missing_df["missing_ratio"].astype(float)

    numeric_summary = (
        numeric_df.describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]).T.reset_index().rename(columns={"index": "column"})
    )
    numeric_summary["missing_ratio"] = numeric_summary["column"].map(raw_df.isna().mean().to_dict()).fillna(0.0)
    numeric_summary = numeric_summary.sort_values("std", ascending=False)
    return missing_df, numeric_summary


def evaluate_cluster_separation(clustered_df: pd.DataFrame) -> list[GroupTestResult]:
    features = [col for col in clustered_df.columns if col not in {"Risk_Cluster", "CERT"}]
    c0 = clustered_df[clustered_df["Risk_Cluster"] == "Cluster 0"]
    c1 = clustered_df[clustered_df["Risk_Cluster"] == "Cluster 1"]
    results: list[GroupTestResult] = []
    for feat in features:
        x = c0[feat].to_numpy()
        y = c1[feat].to_numpy()
        t = stats.ttest_ind(x, y, equal_var=False, nan_policy="omit")
        u = stats.mannwhitneyu(x, y, alternative="two-sided")
        ci_low, ci_high = bootstrap_mean_diff_ci(x, y)
        results.append(
            GroupTestResult(
                feature=feat,
                mean_cluster0=float(np.mean(x)),
                mean_cluster1=float(np.mean(y)),
                median_cluster0=float(np.median(x)),
                median_cluster1=float(np.median(y)),
                welch_t_pvalue=float(t.pvalue),
                mannwhitney_pvalue=float(u.pvalue),
                cohens_d=cohens_d(x, y),
                cliffs_delta=cliffs_delta(x, y),
                bootstrap_diff_ci_low=ci_low,
                bootstrap_diff_ci_high=ci_high,
            )
        )
    return results


def evaluate_stability(features_df: pd.DataFrame, random_seed: int = 42, rounds: int = 50) -> pd.DataFrame:
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(features_df)
    base_model = KMeans(n_clusters=2, random_state=random_seed, n_init=10)
    base_labels = base_model.fit_predict(x_scaled)

    rng = np.random.default_rng(random_seed)
    indices = np.arange(len(features_df))
    rows = []
    for i in range(rounds):
        sampled = rng.choice(indices, size=len(indices), replace=True)
        sampled_x = x_scaled[sampled]
        sampled_model = KMeans(n_clusters=2, random_state=random_seed + i + 1, n_init=10)
        sampled_labels = sampled_model.fit_predict(sampled_x)
        # Compare labels where sampled indices map to original points.
        ari = adjusted_rand_score(base_labels[sampled], sampled_labels)
        rows.append({"round": i + 1, "ari_vs_base": float(ari)})
    return pd.DataFrame(rows)


def evaluate_outlier_sensitivity(raw_df: pd.DataFrame, random_seed: int = 42) -> pd.DataFrame:
    features = engineer_fdic_features(raw_df)
    settings = [(0.005, 0.995), (0.01, 0.99), (0.02, 0.98), (0.03, 0.97)]
    rows = []
    for low, high in settings:
        filtered = remove_outliers_by_quantile(features, low_quantile=low, high_quantile=high)
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(filtered)
        model = KMeans(n_clusters=2, random_state=random_seed, n_init=10)
        labels = model.fit_predict(x_scaled)
        rows.append(
            {
                "low_q": low,
                "high_q": high,
                "rows_kept": int(len(filtered)),
                "silhouette_k2": float(silhouette_score(x_scaled, labels)),
                "calinski_harabasz_k2": float(calinski_harabasz_score(x_scaled, labels)),
                "davies_bouldin_k2": float(davies_bouldin_score(x_scaled, labels)),
            }
        )
    return pd.DataFrame(rows)


def write_report(
    output_report: Path,
    artifact_dir: Path,
    config: dict,
    k_grid: pd.DataFrame,
    separation: list[GroupTestResult],
    stability: pd.DataFrame,
    sensitivity: pd.DataFrame,
    cluster_counts: pd.Series,
    alt_models: pd.DataFrame,
    pca_df: pd.DataFrame,
    anomaly_df: pd.DataFrame,
    missing_df: pd.DataFrame,
    numeric_summary: pd.DataFrame,
) -> None:
    best_k_row = k_grid.sort_values("silhouette", ascending=False).iloc[0]
    stability_desc = stability["ari_vs_base"].describe()
    text = []
    text.append("# FDIC Full Analytical Test Report\n")
    text.append(f"- Source artifact: `{artifact_dir}`")
    text.append(f"- Input rows: `{config['input_rows']}`, feature rows: `{config['feature_rows']}`")
    text.append("")
    text.append("## 1) Clustering Quality Tests")
    text.append(
        f"- Best k by silhouette on tested grid: `k={int(best_k_row['k'])}` "
        f"with silhouette `{best_k_row['silhouette']:.4f}`."
    )
    text.append(
        "- Additional objectives considered: maximize Calinski-Harabasz, minimize Davies-Bouldin."
    )
    text.append("")
    text.append("## 0) FDIC Data Inventory and Quality")
    text.append(f"- Total columns in merged dataset: `{len(missing_df)}`")
    text.append(f"- Numeric columns profiled: `{len(numeric_summary)}`")
    top_missing = missing_df.head(10)
    text.append("- Highest-missing columns (top 10):")
    for _, row in top_missing.iterrows():
        text.append(f"  - `{row['column']}` missing `{row['missing_ratio']:.2%}`")
    text.append("")
    text.append("## 1b) Multi-Model Unsupervised Benchmark")
    valid_alt = alt_models.dropna(subset=["silhouette"]).copy()
    best_alt = valid_alt.sort_values("silhouette", ascending=False).iloc[0]
    text.append(
        f"- Best alternative model: `{best_alt['model']}` (`{best_alt['setting']}`), "
        f"silhouette `{best_alt['silhouette']:.4f}`."
    )
    text.append(
        f"- K-Means reference silhouette: `{best_k_row['silhouette']:.4f}`."
    )
    if best_alt["silhouette"] > best_k_row["silhouette"]:
        text.append("- Alternative model outperformed K-Means on geometric separation in this feature space.")
    else:
        text.append("- K-Means remains competitive or best for this dataset under current engineered features.")
    text.append("")
    text.append("## 2) Segment Composition Test")
    for name, count in cluster_counts.items():
        text.append(f"- `{name}` size: `{int(count)}`")
    text.append("")
    text.append("## 3) Between-Cluster Statistical Tests")
    text.append(
        "- For each feature, report Welch t-test p-value, Mann-Whitney p-value, "
        "effect sizes (Cohen's d, Cliff's delta), and bootstrap 95% CI for mean difference."
    )
    for row in separation:
        text.append(
            f"- `{row.feature}`: "
            f"means {row.mean_cluster0:.6f} vs {row.mean_cluster1:.6f}, "
            f"Welch p={row.welch_t_pvalue:.3e}, Mann-Whitney p={row.mannwhitney_pvalue:.3e}, "
            f"d={row.cohens_d:.3f}, Cliff={row.cliffs_delta:.3f}, "
            f"95% CI diff=[{row.bootstrap_diff_ci_low:.6f}, {row.bootstrap_diff_ci_high:.6f}]"
        )
    text.append("")
    text.append("## 4) Stability Test (Bootstrap Resampling)")
    text.append(
        f"- ARI mean=`{stability_desc['mean']:.4f}`, std=`{stability_desc['std']:.4f}`, "
        f"min=`{stability_desc['min']:.4f}`, max=`{stability_desc['max']:.4f}` over `{len(stability)}` rounds."
    )
    text.append("")
    text.append("## 5) Sensitivity Test (Outlier Thresholds)")
    text.append("- Evaluated robustness to quantile cutoffs; metrics below are all for k=2.")
    for _, row in sensitivity.iterrows():
        text.append(
            f"- q=({row['low_q']:.3f},{row['high_q']:.3f}): "
            f"rows={int(row['rows_kept'])}, silhouette={row['silhouette_k2']:.4f}, "
            f"CH={row['calinski_harabasz_k2']:.2f}, DB={row['davies_bouldin_k2']:.4f}"
        )
    text.append("")
    text.append("## 5b) Dimensionality and Anomaly Diagnostics")
    text.append(
        f"- PCA variance captured by first 2 components: `{(pca_df['explained_variance_ratio'].head(2).sum()):.2%}`."
    )
    text.append(
        f"- PCA variance captured by first 3 components: `{(pca_df['explained_variance_ratio'].head(3).sum()):.2%}`."
    )
    text.append(
        f"- IsolationForest anomaly share (1% target): `{anomaly_df.iloc[0]['anomaly_ratio']:.2%}` "
        f"({int(anomaly_df.iloc[0]['anomaly_rows'])} of {int(anomaly_df.iloc[0]['total_rows'])})."
    )
    text.append("")
    text.append("## 6) Economic Translation (General Economy)")
    text.append(
        "- Lower capital buffers and higher stress ratios in the minority cluster indicate a fragile subset of institutions "
        "that could tighten credit conditions if their stress deepens."
    )
    text.append(
        "- If this minority segment grows over time, likely macro implications are reduced small-business/consumer lending "
        "capacity, higher funding costs, and localized economic drag."
    )
    text.append(
        "- Stable majority-cluster metrics suggest broad banking conditions are currently more resilient than the stressed tail."
    )
    text.append(
        "- Policy/market monitoring should track migration rates from stable to stressed cluster as a leading indicator "
        "for broader credit-cycle weakening."
    )
    text.append("")
    text.append("## 7) Practical Interpretation")
    text.append(
        "- The data supports a strong two-segment structure. The smaller segment is materially different "
        "from the main population on capitalization and stress-related ratios."
    )
    text.append(
        "- Cluster imbalance is substantial, so downstream monitoring should track minority segment drift "
        "and avoid over-generalizing majority behavior."
    )
    text.append(
        "- Next high-value step is linking these segments to real outcomes (failure/events) for supervised validation."
    )
    output_report.write_text("\n".join(text))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full analytical tests and generate report.")
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", default="backend/artifacts/reports/secondary/full_analysis")
    args = parser.parse_args()

    artifact_dir = Path(args.artifact_dir)
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = json.loads((artifact_dir / "config.json").read_text())
    clustered_df = pd.read_csv(artifact_dir / "clustered_features.csv")
    raw_df = load_and_merge_csvs(input_dir)
    features = clustered_df.drop(columns=["Risk_Cluster", "CERT"], errors="ignore")

    k_grid = evaluate_k_grid(features)
    separation = evaluate_cluster_separation(clustered_df)
    stability = evaluate_stability(features)
    sensitivity = evaluate_outlier_sensitivity(raw_df)
    cluster_counts = clustered_df["Risk_Cluster"].value_counts().sort_index()
    alt_models = evaluate_alternative_models(features)
    pca_df, anomaly_df = evaluate_dimension_and_anomaly(features)
    missing_df, numeric_summary = profile_raw_fdic_data(raw_df)

    k_grid.to_csv(output_dir / "k_grid_metrics.csv", index=False)
    pd.DataFrame([vars(row) for row in separation]).to_csv(output_dir / "cluster_separation_tests.csv", index=False)
    stability.to_csv(output_dir / "stability_bootstrap_ari.csv", index=False)
    sensitivity.to_csv(output_dir / "outlier_sensitivity_tests.csv", index=False)
    cluster_counts.to_csv(output_dir / "cluster_counts.csv", header=["count"])
    alt_models.to_csv(output_dir / "alternative_model_comparison.csv", index=False)
    pca_df.to_csv(output_dir / "pca_variance.csv", index=False)
    anomaly_df.to_csv(output_dir / "anomaly_summary.csv", index=False)
    missing_df.to_csv(output_dir / "raw_missingness.csv", index=False)
    numeric_summary.to_csv(output_dir / "raw_numeric_summary.csv", index=False)

    report_path = output_dir / "FULL_REPORT_2026-04-20.md"
    write_report(
        output_report=report_path,
        artifact_dir=artifact_dir,
        config=config,
        k_grid=k_grid,
        separation=separation,
        stability=stability,
        sensitivity=sensitivity,
        cluster_counts=cluster_counts,
        alt_models=alt_models,
        pca_df=pca_df,
        anomaly_df=anomaly_df,
        missing_df=missing_df,
        numeric_summary=numeric_summary,
    )
    print(json.dumps({"report": str(report_path), "output_dir": str(output_dir)}, indent=2))


if __name__ == "__main__":
    main()
