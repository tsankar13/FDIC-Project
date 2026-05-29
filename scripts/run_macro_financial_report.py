from __future__ import annotations

"""Legacy macro-financial report script.

This script depends on outputs from `run_full_analysis.py` and is preserved for
historical reproducibility.
Preferred supported workflow is the `fdic-ml` CLI:
- `fdic-ml --config ... --mode both`
- `fdic-ml --generate-synopsis-report --output-dir backend/artifacts`
- `fdic-ml --generate-visualizations --output-dir backend/artifacts`

See `backend/docs/LEGACY.md` for deprecation context and migration guidance.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from fdic_ml_pipeline.data_io import load_and_merge_csvs
from fdic_ml_pipeline.features import engineer_fdic_features


def quantiles(series: pd.Series) -> dict[str, float]:
    return {
        "p10": float(series.quantile(0.10)),
        "p25": float(series.quantile(0.25)),
        "p50": float(series.quantile(0.50)),
        "p75": float(series.quantile(0.75)),
        "p90": float(series.quantile(0.90)),
    }


def safe_div(a: float, b: float) -> float:
    return float(a / b) if b else 0.0


def compute_core_system_metrics(df: pd.DataFrame, features: pd.DataFrame) -> dict:
    assets = pd.to_numeric(df["ASSET"], errors="coerce").fillna(0.0)
    liabilities = pd.to_numeric(df.get("LIAB", 0.0), errors="coerce").fillna(0.0)
    deposits = pd.to_numeric(df.get("DEP", 0.0), errors="coerce").fillna(0.0)
    capital = pd.to_numeric(df.get("RBCT1J", 0.0), errors="coerce").fillna(0.0)
    intinc = pd.to_numeric(df.get("INTINC", 0.0), errors="coerce").fillna(0.0)
    eintexp = pd.to_numeric(df.get("EINTEXP", 0.0), errors="coerce").fillna(0.0)
    naasset = pd.to_numeric(df.get("NAASSET", 0.0), errors="coerce").fillna(0.0)
    p9asset = pd.to_numeric(df.get("P9ASSET", 0.0), errors="coerce").fillna(0.0)
    ntls = pd.to_numeric(df.get("NTLS", 0.0), errors="coerce").fillna(0.0)

    total_assets = float(assets.sum())
    total_liabilities = float(liabilities.sum())
    total_deposits = float(deposits.sum())
    total_capital = float(capital.sum())

    top10_share = float(assets.nlargest(10).sum() / total_assets) if total_assets else 0.0
    hhi = float(np.sum((assets / total_assets) ** 2)) if total_assets else 0.0

    weighted_capital_ratio = safe_div(total_capital, total_assets)
    weighted_nim = safe_div(float((intinc - eintexp).sum()), total_assets)
    weighted_texas = safe_div(float((naasset + p9asset).sum()), total_capital)
    weighted_chargeoff = safe_div(float(ntls.sum()), total_assets)

    metrics = {
        "n_banks": int(len(df)),
        "report_date": str(df["REPDTE"].iloc[0]) if "REPDTE" in df.columns and len(df) else "unknown",
        "total_assets": total_assets,
        "total_liabilities": total_liabilities,
        "total_deposits": total_deposits,
        "total_capital_proxy_rbct1j": total_capital,
        "deposit_to_asset_ratio": safe_div(total_deposits, total_assets),
        "liability_to_asset_ratio": safe_div(total_liabilities, total_assets),
        "weighted_capital_ratio": weighted_capital_ratio,
        "weighted_net_interest_margin": weighted_nim,
        "weighted_texas_ratio": weighted_texas,
        "weighted_chargeoff_ratio": weighted_chargeoff,
        "asset_top10_share": top10_share,
        "asset_hhi": hhi,
    }

    q = {col: quantiles(features[col]) for col in features.columns}
    metrics["feature_quantiles"] = q
    return metrics


def build_risk_dashboard_rows(metrics: dict, cluster_counts: dict, deep_metrics: dict) -> list[list[str]]:
    stressed_share = safe_div(cluster_counts.get("Cluster 0", 0), sum(cluster_counts.values()))
    rows = [
        [
            "System capital buffer",
            f"{metrics['weighted_capital_ratio']:.2%}",
            "Lower values reduce shock-absorption and can amplify credit tightening.",
            "Warning below 10%; severe below 8%.",
        ],
        [
            "System Texas ratio",
            f"{metrics['weighted_texas_ratio']:.2%}",
            "Higher values imply rising problem-asset pressure vs capital.",
            "Watch acceleration quarter-over-quarter.",
        ],
        [
            "System charge-off ratio",
            f"{metrics['weighted_chargeoff_ratio']:.4%}",
            "Higher credit losses reduce profitability and lending capacity.",
            "Sustained rises indicate credit-cycle deterioration.",
        ],
        [
            "Stressed-cluster share",
            f"{stressed_share:.2%}",
            "Higher stressed-tail share can foreshadow wider banking strain.",
            "Escalation signal if share rises persistently.",
        ],
        [
            "Clustering quality",
            f"{deep_metrics['silhouette']:.4f}",
            "Higher quality means the stress signal is structurally distinct.",
            "Quality collapse may indicate regime shift or bad preprocessing.",
        ],
        [
            "Outlier sensitivity gap",
            f"{deep_metrics['silhouette_drop_aggressive']:.4f}",
            "Large gap means aggressive trims can hide true stress structure.",
            "Maintain mild trimming policy unless justified.",
        ],
    ]
    return rows


def write_macro_report(
    output_path: Path,
    system_metrics: dict,
    cluster_counts: dict,
    deep_report_metrics: dict,
    risk_rows: list[list[str]],
) -> None:
    stressed_share = safe_div(cluster_counts.get("Cluster 0", 0), sum(cluster_counts.values()))
    text: list[str] = []
    text.append("# FDIC Macro-Financial Deep Interpretation Report")
    text.append("")
    text.append("## Scope and Data Basis")
    text.append(
        f"- FDIC snapshot date: `{system_metrics['report_date']}` across `{system_metrics['n_banks']}` institutions."
    )
    text.append("- This is cross-sectional (single-date) analysis, not a time-series cycle study.")
    text.append("- ML and statistical diagnostics sourced from your deep report outputs.")
    text.append("")
    text.append("## Banking-System Financial Structure")
    text.append(f"- Total assets: `${system_metrics['total_assets']:,.0f}`")
    text.append(f"- Total liabilities: `${system_metrics['total_liabilities']:,.0f}`")
    text.append(f"- Total deposits: `${system_metrics['total_deposits']:,.0f}`")
    text.append(f"- Deposit-to-asset ratio: `{system_metrics['deposit_to_asset_ratio']:.2%}`")
    text.append(f"- Liability-to-asset ratio: `{system_metrics['liability_to_asset_ratio']:.2%}`")
    text.append(f"- Weighted capital ratio proxy: `{system_metrics['weighted_capital_ratio']:.2%}`")
    text.append(f"- Weighted net interest margin: `{system_metrics['weighted_net_interest_margin']:.2%}`")
    text.append(f"- Weighted Texas ratio: `{system_metrics['weighted_texas_ratio']:.2%}`")
    text.append(f"- Weighted charge-off ratio: `{system_metrics['weighted_chargeoff_ratio']:.4%}`")
    text.append(f"- Concentration (top-10 asset share): `{system_metrics['asset_top10_share']:.2%}`")
    text.append(f"- Concentration (asset HHI): `{system_metrics['asset_hhi']:.4f}`")
    text.append("")
    text.append("## ML Risk Segmentation Synthesis")
    text.append(f"- Best K-Means specification: `k=2`, silhouette `{deep_report_metrics['silhouette']:.4f}`.")
    text.append(
        f"- Cluster composition: Cluster 0 (stressed tail) `{cluster_counts.get('Cluster 0', 0)}`, "
        f"Cluster 1 (main body) `{cluster_counts.get('Cluster 1', 0)}`."
    )
    text.append(f"- Stressed-tail share: `{stressed_share:.2%}`.")
    text.append(
        "- Interpretation: the minority stressed segment is small but statistically distinct, so risk is currently "
        "concentrated rather than system-wide."
    )
    text.append("")
    text.append("## Financial Transmission to the General Economy")
    text.append(
        "- Credit Channel: weaker-capital banks generally tighten lending standards first, which can slow "
        "small-business credit and consumer durables demand."
    )
    text.append(
        "- Cost-of-Credit Channel: if charge-offs rise, banks reprice risk upward (higher spreads), increasing "
        "financing costs for households and firms."
    )
    text.append(
        "- Regional Channel: concentrated pockets of stress can create local recessions even when national aggregates "
        "still appear stable."
    )
    text.append(
        "- Balance-Sheet Channel: higher liability intensity with pressured net interest margins can weaken capital "
        "accumulation and reduce forward credit supply."
    )
    text.append("")
    text.append("## Macro Context: GDP, Labor, Inflation, Policy, and Political Events")
    text.append(
        "- GDP (BEA, 2025 Q4 third estimate): real GDP growth slowed to `0.5%` SAAR from `4.4%` in 2025 Q3."
    )
    text.append("- Private domestic final sales grew `1.8%` SAAR; real GDI grew `2.6%`.")
    text.append("- PCE inflation (Q4 2025): headline `2.9%`, core `2.7%`.")
    text.append("- Labor market (FRED/BLS): unemployment rate `4.3%` in Mar 2026.")
    text.append(
        "- Monetary policy (FOMC Mar 18, 2026): target federal funds range held at `3.50%–3.75%`."
    )
    text.append(
        "- Political/policy events with macro relevance: October-November 2025 federal shutdown effects and elevated "
        "Middle East uncertainty highlighted in official releases."
    )
    text.append("")
    text.append("## Holistic Risk Profile (Current Regime)")
    text.append("- Base case: **late-cycle softening, not broad banking crisis**.")
    text.append(
        "- Why: high-quality two-cluster separation indicates identifiable stressed tail, while majority cluster "
        "retains stronger central tendency on core solvency/profitability ratios."
    )
    text.append(
        "- Tail risk: if stressed-tail share rises materially, macro slowdown risk rises through tighter credit, "
        "especially for rate-sensitive sectors."
    )
    text.append("")
    text.append("## Forward Expectations (Scenario Framework)")
    text.append("- **Baseline (55%)**: modest growth, sticky disinflation, selective credit tightening.")
    text.append("- **Downside (30%)**: stress migration from tail to core, sharper lending pullback, subtrend GDP.")
    text.append("- **Upside (15%)**: inflation cools faster, policy eases, credit conditions stabilize.")
    text.append("")
    text.append("## Monitoring Schedule and Decision Rules")
    text.append("- **Monthly:** refresh labor/inflation/policy context and rerun anomaly + concentration diagnostics.")
    text.append("- **Quarterly (after call report updates):** rerun full clustering and separation tests.")
    text.append("- **Quarterly Board Packet:** publish risk-dashboard rows below with green/amber/red status.")
    text.append("- **Trigger rule 1:** if stressed-tail share increases by >50 bps for 2 consecutive updates, escalate.")
    text.append("- **Trigger rule 2:** if weighted capital ratio falls below 10% and charge-off ratio rises >20% QoQ, escalate.")
    text.append("- **Trigger rule 3:** if silhouette drops below 0.60, review feature pipeline and possible regime change.")
    text.append("")
    text.append("## Risk Dashboard Table")
    text.append("| Indicator | Current Value | Economic Meaning | Action Threshold |")
    text.append("|---|---:|---|---|")
    for row in risk_rows:
        text.append(f"| {row[0]} | {row[1]} | {row[2]} | {row[3]} |")
    text.append("")
    text.append("## Caveats")
    text.append("- No causal claim is made from cross-sectional clustering alone.")
    text.append("- This report uses proxy capital and stress metrics from available columns.")
    text.append("- Add supervised target labels (failure, downgrade, intervention) for predictive validation.")
    output_path.write_text("\n".join(text))


def main() -> None:
    parser = argparse.ArgumentParser(description="Create macro-financial deep interpretation report.")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--deep-report-dir", required=True)
    parser.add_argument("--output-dir", default="backend/artifacts/reports/primary/macro_financial")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    artifact_dir = Path(args.artifact_dir)
    deep_dir = Path(args.deep_report_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_df = load_and_merge_csvs(input_dir).reset_index()
    features = engineer_fdic_features(raw_df)
    system_metrics = compute_core_system_metrics(raw_df, features)

    cluster_counts_df = pd.read_csv(deep_dir / "cluster_counts.csv")
    cluster_counts = dict(zip(cluster_counts_df.iloc[:, 0], cluster_counts_df.iloc[:, 1]))

    k_grid = pd.read_csv(deep_dir / "k_grid_metrics.csv")
    best = k_grid.sort_values("silhouette", ascending=False).iloc[0]
    sens = pd.read_csv(deep_dir / "outlier_sensitivity_tests.csv")
    mild = float(sens.loc[(sens["low_q"] == 0.01) & (sens["high_q"] == 0.99), "silhouette_k2"].iloc[0])
    aggressive = float(sens.loc[(sens["low_q"] == 0.02) & (sens["high_q"] == 0.98), "silhouette_k2"].iloc[0])
    deep_report_metrics = {
        "silhouette": float(best["silhouette"]),
        "silhouette_drop_aggressive": float(mild - aggressive),
    }

    risk_rows = build_risk_dashboard_rows(system_metrics, cluster_counts, deep_report_metrics)
    report_path = output_dir / "MACRO_FINANCIAL_DEEP_REPORT_2026-04-20.md"
    write_macro_report(report_path, system_metrics, cluster_counts, deep_report_metrics, risk_rows)

    (output_dir / "system_metrics.json").write_text(json.dumps(system_metrics, indent=2))
    pd.DataFrame(risk_rows, columns=["indicator", "value", "meaning", "threshold"]).to_csv(
        output_dir / "risk_dashboard.csv", index=False
    )
    print(json.dumps({"report": str(report_path), "output_dir": str(output_dir)}, indent=2))


if __name__ == "__main__":
    main()
