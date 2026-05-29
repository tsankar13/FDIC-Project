from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


@dataclass
class VisualizationOutput:
    run_dir: Path
    figure_paths: list[Path]
    skipped_charts: list[str]
    report_path: Path


def _latest_artifact_run(output_dir: Path) -> Path:
    candidates = sorted(path for path in output_dir.glob("*_*") if path.is_dir())
    for candidate in reversed(candidates):
        if any(
            (candidate / name).exists()
            for name in (
                "clustered_features.csv",
                "cluster_summary.csv",
                "synopsis_metrics.json",
                "synopsis_correlations.csv",
                "synopsis_anomalies.csv",
            )
        ):
            return candidate
    raise FileNotFoundError(f"No artifact runs with known visualization files found under {output_dir}")


def _save_cluster_size_chart(run_dir: Path, figures_dir: Path) -> Path | None:
    clustered_path = run_dir / "clustered_features.csv"
    if not clustered_path.exists():
        return None

    clustered_df = pd.read_csv(clustered_path)
    if "Risk_Cluster" not in clustered_df.columns:
        return None

    counts = clustered_df["Risk_Cluster"].value_counts().sort_index()
    if counts.empty:
        return None

    fig, ax = plt.subplots(figsize=(8, 5))
    counts.plot(kind="bar", color="#4e79a7", ax=ax)
    ax.set_title("Institution Count by Risk Cluster")
    ax.set_xlabel("Risk Cluster")
    ax.set_ylabel("Institutions")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    out_path = figures_dir / "cluster_size.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def _save_cluster_feature_means_chart(run_dir: Path, figures_dir: Path) -> Path | None:
    summary_path = run_dir / "cluster_summary.csv"
    if not summary_path.exists():
        return None

    summary_df = pd.read_csv(summary_path, index_col=0)
    if summary_df.empty:
        return None

    fig, ax = plt.subplots(figsize=(9, 4.5))
    image = ax.imshow(summary_df.values, aspect="auto", cmap="coolwarm")
    ax.set_title("Average Feature Values by Cluster")
    ax.set_xticks(range(len(summary_df.columns)))
    ax.set_xticklabels(summary_df.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(summary_df.index)))
    ax.set_yticklabels(summary_df.index)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out_path = figures_dir / "cluster_feature_means.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def _save_top_correlations_chart(run_dir: Path, figures_dir: Path) -> Path | None:
    corr_path = run_dir / "synopsis_correlations.csv"
    if not corr_path.exists():
        return None

    corr_df = pd.read_csv(corr_path)
    required_columns = {"feature_left", "feature_right", "correlation"}
    if not required_columns.issubset(corr_df.columns):
        return None
    if corr_df.empty:
        return None

    top = corr_df.head(10).copy()
    top["pair"] = top["feature_left"] + " vs " + top["feature_right"]
    top = top.iloc[::-1]
    colors = top["correlation"].apply(lambda value: "#59a14f" if value >= 0 else "#e15759")

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.barh(top["pair"], top["correlation"], color=list(colors))
    ax.set_title("Top Feature Correlations")
    ax.set_xlabel("Correlation")
    ax.set_ylabel("Feature Pair")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    out_path = figures_dir / "top_correlations.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def _save_pca_variance_chart(run_dir: Path, figures_dir: Path) -> Path | None:
    metrics_path = run_dir / "synopsis_metrics.json"
    if not metrics_path.exists():
        return None

    metrics = json.loads(metrics_path.read_text())
    if "pca_explained_variance_pc1" not in metrics or "pca_explained_variance_top3" not in metrics:
        return None

    pc1 = float(metrics["pca_explained_variance_pc1"])
    top3 = float(metrics["pca_explained_variance_top3"])
    remaining = max(0.0, 1.0 - top3)

    labels = ["PC1", "Top 3 PCs", "Remaining"]
    values = [pc1, top3, remaining]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(labels, values, color=["#4e79a7", "#f28e2b", "#bab0ab"])
    ax.set_title("PCA Explained Variance Overview")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Variance Share")
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.02,
            f"{value * 100:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    fig.tight_layout()
    out_path = figures_dir / "pca_variance_overview.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def _save_anomaly_share_chart(run_dir: Path, figures_dir: Path) -> Path | None:
    metrics_path = run_dir / "synopsis_metrics.json"
    anomalies_path = run_dir / "synopsis_anomalies.csv"
    if not metrics_path.exists() and not anomalies_path.exists():
        return None

    anomalies = 0
    normal = 0
    if anomalies_path.exists():
        anomaly_df = pd.read_csv(anomalies_path)
        if "anomaly_flag" in anomaly_df.columns:
            anomalies = int((anomaly_df["anomaly_flag"] == -1).sum())
            normal = int((anomaly_df["anomaly_flag"] != -1).sum())

    if metrics_path.exists() and (anomalies + normal == 0):
        metrics = json.loads(metrics_path.read_text())
        if "anomaly_count" in metrics and "rows_analyzed" in metrics:
            anomalies = int(metrics["anomaly_count"])
            rows = int(metrics["rows_analyzed"])
            normal = max(0, rows - anomalies)

    total = anomalies + normal
    if total == 0:
        return None

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.bar(["Normal", "Anomalies"], [normal, anomalies], color=["#4e79a7", "#e15759"])
    ax.set_title("Anomaly Detection Breakdown")
    ax.set_ylabel("Institutions")
    ax.grid(axis="y", alpha=0.25)
    ax.text(
        1,
        anomalies + max(total * 0.02, 1),
        f"{(anomalies / total) * 100:.2f}%",
        ha="center",
        va="bottom",
        fontsize=9,
    )
    fig.tight_layout()
    out_path = figures_dir / "anomaly_breakdown.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def _write_visualization_report(
    run_dir: Path,
    report_dir: Path,
    figure_paths: list[Path],
    skipped_charts: list[str],
) -> Path:
    stamp = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
    report_path = report_dir / f"VISUALIZATION_REPORT_{stamp}.md"
    figure_lines = []
    for figure in figure_paths:
        relative_path = figure.relative_to(report_dir)
        title = figure.stem.replace("_", " ").title()
        figure_lines.extend([f"### {title}", f"![{title}]({relative_path.as_posix()})", ""])

    skipped_lines = skipped_charts if skipped_charts else ["None"]

    text = "\n".join(
        [
            "# Visualization Report",
            "",
            "## Run Source",
            f"- `{run_dir}`",
            "",
            "## Generated Figures",
            *(figure_lines if figure_lines else ["No figures were generated.", ""]),
            "## Skipped Charts",
            *[f"- {item}" for item in skipped_lines],
            "",
        ]
    )
    report_path.write_text(text)
    return report_path


def generate_visualizations(
    output_dir: Path = Path("backend/artifacts"),
    report_dir: Path = Path("backend/artifacts/reports/secondary"),
    figures_dir: Path = Path("backend/artifacts/reports/secondary/figures"),
    run_dir: Path | None = None,
) -> VisualizationOutput:
    selected_run_dir = run_dir if run_dir is not None else _latest_artifact_run(output_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    figure_paths: list[Path] = []
    skipped_charts: list[str] = []

    chart_builders = [
        ("cluster_size", _save_cluster_size_chart),
        ("cluster_feature_means", _save_cluster_feature_means_chart),
        ("top_correlations", _save_top_correlations_chart),
        ("pca_variance_overview", _save_pca_variance_chart),
        ("anomaly_breakdown", _save_anomaly_share_chart),
    ]

    for chart_name, builder in chart_builders:
        generated = builder(selected_run_dir, figures_dir)
        if generated is None:
            skipped_charts.append(f"{chart_name}: missing or invalid source artifact")
        else:
            figure_paths.append(generated)

    report_path = _write_visualization_report(
        run_dir=selected_run_dir,
        report_dir=report_dir,
        figure_paths=figure_paths,
        skipped_charts=skipped_charts,
    )
    return VisualizationOutput(
        run_dir=selected_run_dir,
        figure_paths=figure_paths,
        skipped_charts=skipped_charts,
        report_path=report_path,
    )
