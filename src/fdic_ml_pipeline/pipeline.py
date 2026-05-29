from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from .artifacts import create_run_dir, write_frame, write_json
from .clustering import train_clustering
from .config import PipelineConfig
from .data_io import load_and_merge_csvs, validate_required_columns
from .features import engineer_fdic_features, remove_outliers_by_quantile
from .supervised import train_supervised
from .synopsis import build_data_synopsis


def run_pipeline(config: PipelineConfig) -> dict:
    raw_df = load_and_merge_csvs(config.input_dir)
    validate_required_columns(raw_df, config.feature_columns_required)

    features = engineer_fdic_features(raw_df)
    features = remove_outliers_by_quantile(
        features,
        low_quantile=config.outlier_quantile_low,
        high_quantile=config.outlier_quantile_high,
    )

    run_dir = create_run_dir(config.output_dir, config.mode)
    raw_config = asdict(config) | {"input_rows": int(len(raw_df)), "feature_rows": int(len(features))}
    serializable_config = {
        key: str(value) if isinstance(value, Path) else value for key, value in raw_config.items()
    }
    write_json(serializable_config, run_dir / "config.json")

    output: dict = {"run_dir": str(run_dir)}

    if config.mode in {"clustering", "both"}:
        clustering = train_clustering(features, max_clusters=config.max_clusters, random_seed=config.random_seed)
        write_frame(clustering.clustered_df, run_dir / "clustered_features.csv")
        write_frame(clustering.summary_df, run_dir / "cluster_summary.csv")
        write_json(
            {"best_k": clustering.best_k, "silhouette_score": clustering.best_score},
            run_dir / "clustering_metrics.json",
        )
        output["clustering"] = {"best_k": clustering.best_k, "silhouette_score": clustering.best_score}

    if config.mode in {"synopsis", "both"}:
        synopsis = build_data_synopsis(features, random_seed=config.random_seed)
        write_json(dict(synopsis.metrics), run_dir / "synopsis_metrics.json")
        write_frame(synopsis.pca_components_df, run_dir / "synopsis_pca_components.csv")
        write_frame(synopsis.anomaly_df, run_dir / "synopsis_anomalies.csv")
        write_frame(synopsis.correlation_pairs_df, run_dir / "synopsis_correlations.csv")
        output["synopsis"] = dict(synopsis.metrics)

    if config.mode in {"supervised", "both"}:
        if config.label_column not in raw_df.columns:
            if config.mode == "supervised":
                raise ValueError(
                    f"Label column '{config.label_column}' not found. "
                    "Set --label-column to an existing FDIC target for supervised mode."
                )
            output["supervised"] = {
                "status": "skipped",
                "reason": f"Label column '{config.label_column}' not found in dataset.",
            }
        else:
            labels = raw_df.loc[features.index, config.label_column]
            supervised = train_supervised(features, labels, random_seed=config.random_seed)
            write_frame(supervised.predictions_df, run_dir / "supervised_predictions.csv")
            write_frame(supervised.explanation_df, run_dir / "supervised_feature_explanations.csv")
            write_json(
                {"selected_model": supervised.model_name, "metrics": supervised.metrics},
                run_dir / "supervised_metrics.json",
            )
            output["supervised"] = {"selected_model": supervised.model_name, **supervised.metrics}

    return output
