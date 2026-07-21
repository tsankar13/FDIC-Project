"""Time-series tracking: pull the same CAMELS ratios across past FDIC filings.

The rest of the pipeline analyzes a single quarter-end snapshot. This module adds a
*temporal* view by querying the public FDIC BankFind Suite API
(https://banks.data.fdic.gov) for several report dates and summarizing how the
industry-wide distribution of each ratio moved over time.

Design notes:

- Network I/O (``fetch_quarter``) is separated from the pure aggregation
  (``summarize_trends``) so the summary logic is unit-testable without the API.
- Each raw quarterly pull is cached to CSV under the output directory, so a second
  run is fully offline and reproducible.
- Only aggregate statistics (median / quartiles across all institutions) are
  reported, matching the descriptive intent of the project.
"""

from __future__ import annotations

import json
import ssl
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

API_URL = "https://banks.data.fdic.gov/api/financials"

# FDIC API field -> readable metric name. These mirror the CAMELS panel used by
# ``features.py`` so the cross-section and the trend speak the same language.
TREND_METRICS: dict[str, str] = {
    "RBCRWAJ": "Total_Capital_Ratio",
    "ROA": "Return_on_Assets",
    "ROE": "Return_on_Equity",
    "NIMY": "Net_Interest_Margin_Pct",
    "EEFFR": "Efficiency_Ratio",
    "NPERFV": "Nonperforming_Assets_Pct",
    "NCLNLSR": "NCO_to_Loans",
}


@dataclass
class TrendOutput:
    """Result of a trend run: source quarters, summary table, CSV, and figure."""

    quarters: list[str]
    summary_df: pd.DataFrame
    summary_path: Path
    figure_path: Path | None
    cached_quarters: list[str] = field(default_factory=list)


def default_quarters(n: int = 8, latest: str = "20250930") -> list[str]:
    """Return the ``n`` most recent quarter-end REPDTE strings ending at ``latest``.

    FDIC report dates are the last day of Mar/Jun/Sep/Dec. The list is returned in
    chronological (oldest-first) order for plotting.
    """
    year = int(latest[:4])
    month = int(latest[4:6])
    ends = {3: "0331", 6: "0630", 9: "0930", 12: "1231"}
    quarters: list[str] = []
    for _ in range(n):
        quarters.append(f"{year}{ends[month]}")
        month -= 3
        if month == 0:
            month = 12
            year -= 1
    return list(reversed(quarters))


def _ssl_context() -> ssl.SSLContext:
    """Build an SSL context, preferring ``certifi`` roots when available."""
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:  # pragma: no cover - fallback when certifi is absent
        return ssl.create_default_context()


def fetch_quarter(
    repdte: str,
    fields: list[str] | None = None,
    cache_dir: Path | None = None,
    page_size: int = 10000,
    timeout: int = 30,
) -> pd.DataFrame:
    """Fetch per-institution ratios for one report date from the FDIC API.

    Results are cached to ``cache_dir/financials_<repdte>.csv``; a cached file is
    reused instead of re-querying. Returns a DataFrame with a ``CERT`` column and
    one numeric column per requested field.
    """
    fields = fields or list(TREND_METRICS.keys())
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"financials_{repdte}.csv"
        if cache_path.exists():
            return pd.read_csv(cache_path)

    field_list = ",".join(["CERT", *fields])
    ctx = _ssl_context()
    records: list[dict] = []
    offset = 0
    while True:
        query = urllib.parse.urlencode(
            {
                "filters": f"REPDTE:{repdte}",
                "fields": field_list,
                "limit": page_size,
                "offset": offset,
            }
        )
        with urllib.request.urlopen(f"{API_URL}?{query}", timeout=timeout, context=ctx) as response:
            payload = json.loads(response.read())
        batch = [row["data"] for row in payload.get("data", [])]
        records.extend(batch)
        total = int(payload.get("meta", {}).get("total", len(records)))
        offset += page_size
        if offset >= total or not batch:
            break

    frame = pd.DataFrame(records)
    if cache_dir is not None:
        frame.to_csv(cache_dir / f"financials_{repdte}.csv", index=False)
    return frame


def summarize_trends(
    frames_by_quarter: dict[str, pd.DataFrame],
    metrics: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Aggregate per-quarter frames into an industry-median trend table (pure).

    For each quarter and each metric, computes the cross-institution median and the
    25th/75th percentiles. Returns a long-format DataFrame indexed by quarter with
    columns ``metric``, ``median``, ``p25``, ``p75`` — ready to pivot or plot.
    """
    metrics = metrics or TREND_METRICS
    rows: list[dict] = []
    for quarter in sorted(frames_by_quarter):
        frame = frames_by_quarter[quarter]
        for source_col, name in metrics.items():
            if source_col not in frame.columns:
                continue
            series = pd.to_numeric(frame[source_col], errors="coerce").dropna()
            if series.empty:
                continue
            rows.append(
                {
                    "quarter": quarter,
                    "metric": name,
                    "median": float(series.median()),
                    "p25": float(series.quantile(0.25)),
                    "p75": float(series.quantile(0.75)),
                }
            )
    return pd.DataFrame(rows)


def _plot_trends(summary_df: pd.DataFrame, figure_path: Path) -> Path | None:
    """Render one small-multiple line panel per metric (median with IQR band)."""
    if summary_df.empty:
        return None

    metrics = list(dict.fromkeys(summary_df["metric"]))
    n = len(metrics)
    cols = 2
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(12, 3.2 * rows), squeeze=False)

    for idx, metric in enumerate(metrics):
        ax = axes[idx // cols][idx % cols]
        sub = summary_df[summary_df["metric"] == metric]
        labels = [f"{q[:4]}Q{(int(q[4:6]) + 2) // 3}" for q in sub["quarter"]]
        ax.plot(labels, sub["median"], marker="o", color="#4e79a7", label="median")
        ax.fill_between(labels, sub["p25"], sub["p75"], alpha=0.18, color="#4e79a7", label="IQR")
        ax.set_title(metric.replace("_", " "))
        ax.grid(alpha=0.25)
        ax.tick_params(axis="x", rotation=45)

    for idx in range(n, rows * cols):
        axes[idx // cols][idx % cols].axis("off")

    fig.suptitle("FDIC Industry Ratio Trends (median, IQR band)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=150)
    plt.close(fig)
    return figure_path


def generate_trend_report(
    output_dir: Path = Path("backend/artifacts"),
    quarters: list[str] | None = None,
    figures_dir: Path = Path("backend/artifacts/reports/secondary/figures"),
) -> TrendOutput:
    """Fetch quarters from the FDIC API, summarize trends, and write CSV + figure.

    Raw pulls are cached under ``output_dir/trends_cache/`` for offline re-runs.
    Returns a :class:`TrendOutput` describing what was produced.
    """
    quarters = quarters or default_quarters()
    cache_dir = output_dir / "trends_cache"

    frames_by_quarter: dict[str, pd.DataFrame] = {}
    cached: list[str] = []
    for quarter in quarters:
        cache_path = cache_dir / f"financials_{quarter}.csv"
        if cache_path.exists():
            cached.append(quarter)
        frames_by_quarter[quarter] = fetch_quarter(quarter, cache_dir=cache_dir)

    summary_df = summarize_trends(frames_by_quarter)

    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
    summary_path = output_dir / f"trend_summary_{stamp}.csv"
    summary_df.to_csv(summary_path, index=False)

    figure_path = _plot_trends(summary_df, figures_dir / "ratio_trends.png")
    return TrendOutput(
        quarters=quarters,
        summary_df=summary_df,
        summary_path=summary_path,
        figure_path=figure_path,
        cached_quarters=cached,
    )
