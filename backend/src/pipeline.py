"""Pipeline orchestrator: ingest, engineer features, and run the selected mode(s)."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from .artifacts import create_run_dir, write_frame, write_json
from .clustering import train_clustering
from .config import PipelineConfig
from .data_io import load_and_merge_csvs, validate_required_columns
from .features import engineer_fdic_features, remove_outliers_by_quantile
from .labels import build_stress_label
from .risk_score import compute_risk_scores
from .supervised import train_supervised
from .synopsis import build_data_synopsis


def run_pipeline(config: PipelineConfig) -> dict:
    """Run the end-to-end pipeline and write per-run artifacts to a timestamped dir.

    Loads and validates the merged CSVs, engineers risk features, trims outliers,
    then executes the ``clustering``, ``synopsis``, and/or ``supervised`` tracks per
    ``config.mode`` (``both`` runs all three). Supervised is skipped with a note when
    the label column is absent (unless mode is explicitly ``supervised``). Returns a
    summary dict mirroring the artifacts written under ``config.output_dir``.
    """
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

    # Composite CAMELS risk score + watchlist (feature-only, so available in every mode).
    risk = compute_risk_scores(features)
    write_frame(risk.scored_df, run_dir / "risk_scores.csv")
    write_frame(risk.watchlist_df, run_dir / "risk_watchlist.csv")
    output["risk_score"] = {
        "scored_features": len(risk.scored_features),
        "watchlist_size": int(len(risk.watchlist_df)),
    }

    if config.mode in {"clustering", "both"}:
        clustering = train_clustering(features, max_clusters=config.max_clusters, random_seed=config.random_seed)
        write_frame(clustering.clustered_df, run_dir / "clustered_features.csv")
        write_frame(clustering.summary_df, run_dir / "cluster_summary.csv")
        write_json(
            {
                "best_k": clustering.best_k,
                "silhouette_score": clustering.best_score,
                **clustering.quality_metrics,
            },
            run_dir / "clustering_metrics.json",
        )
        output["clustering"] = {
            "best_k": clustering.best_k,
            "silhouette_score": clustering.best_score,
            **clustering.quality_metrics,
        }

    if config.mode in {"synopsis", "both"}:
        synopsis = build_data_synopsis(features, random_seed=config.random_seed)
        write_json(dict(synopsis.metrics), run_dir / "synopsis_metrics.json")
        write_frame(synopsis.pca_components_df, run_dir / "synopsis_pca_components.csv")
        write_frame(synopsis.anomaly_df, run_dir / "synopsis_anomalies.csv")
        write_frame(synopsis.correlation_pairs_df, run_dir / "synopsis_correlations.csv")
        output["synopsis"] = dict(synopsis.metrics)

    if config.mode in {"supervised", "both"}:
        supervised_features = features
        if config.label_column in raw_df.columns:
            labels = raw_df.loc[features.index, config.label_column]
            label_source = config.label_column
            defining_feature = None
        elif config.construct_label_if_missing:
            # No shipped label: build a transparent asset-quality stress proxy and drop
            # the feature that defines it, so the model must predict stress from the
            # other CAMELS dimensions rather than trivially echoing the target.
            labels, defining_feature = build_stress_label(features)
            supervised_features = features.drop(columns=[defining_feature])
            label_source = f"constructed:elevated_stress(>=p90 {defining_feature})"
        else:
            labels = None
            label_source = None

        if labels is None:
            if config.mode == "supervised":
                raise ValueError(
                    f"Label column '{config.label_column}' not found and label construction "
                    "is disabled. Set --label-column or enable construct_label_if_missing."
                )
            output["supervised"] = {
                "status": "skipped",
                "reason": f"Label column '{config.label_column}' not found in dataset.",
            }
        else:
            supervised = train_supervised(supervised_features, labels, random_seed=config.random_seed)
            write_frame(supervised.predictions_df, run_dir / "supervised_predictions.csv")
            write_frame(supervised.explanation_df, run_dir / "supervised_feature_explanations.csv")
            write_frame(
                supervised.permutation_importance_df, run_dir / "supervised_permutation_importance.csv"
            )
            write_json(
                {
                    "selected_model": supervised.model_name,
                    "label_source": label_source,
                    "metrics": supervised.metrics,
                    "cv_metrics": supervised.cv_metrics,
                },
                run_dir / "supervised_metrics.json",
            )
            output["supervised"] = {
                "selected_model": supervised.model_name,
                "label_source": label_source,
                **supervised.metrics,
                **supervised.cv_metrics,
            }

    return output
