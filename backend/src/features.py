"""Feature engineering: derive a CAMELS-style banking-risk panel from FDIC columns.

Two sources are combined into one feature matrix:

- **Computed ratios** derived from raw balance-sheet/income columns (always built).
- **Pre-computed FDIC ratios** passed through from the "Performance & Condition
  Ratios" report when those columns are present (schema-aware, so the 6-file demo
  still runs and the richer panel appears automatically once the ratios file is
  merged in).

Together these span the CAMELS dimensions: Capital, Asset quality, (Management via
Efficiency), Earnings, and Liquidity.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

# Pre-computed FDIC ratios to pass through when present, mapped to readable names.
# Source columns come from the "Performance & Condition Ratios" report and are
# already expressed as percentages/ratios, so no arithmetic is needed.
PASSTHROUGH_RATIOS: dict[str, str] = {
    "RBCRWAJ": "Total_Capital_Ratio",       # Capital
    "RBC1AAJ": "Tier1_Leverage_Ratio",      # Capital
    "IDT1CER": "CET1_Ratio",                # Capital
    "NPERFV": "Nonperforming_Assets_Pct",   # Asset quality
    "NCLNLSR": "NCO_to_Loans",              # Asset quality
    "ELNATRY": "Reserves_to_Loans",         # Asset quality
    "EEFFR": "Efficiency_Ratio",            # Management
    "ROA": "Return_on_Assets",              # Earnings
    "ROE": "Return_on_Equity",              # Earnings
    "NIMY": "Net_Interest_Margin_Pct",      # Earnings
    "LNLSDEPR": "Loans_to_Deposits",        # Liquidity
}


def engineer_fdic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build the risk-feature matrix: computed ratios plus available FDIC ratios.

    Always computes four ratios from raw columns — Capital (RBCT1J/ASSET), Net
    Interest Margin ((INTINC-EINTEXP)/ASSET), Texas ((NAASSET+P9ASSET)/RBCT1J), and
    Charge-Off (NTLS/ASSET). It then appends any :data:`PASSTHROUGH_RATIOS` present
    in ``df`` (the CAMELS panel). Infinities from division-by-zero become NaN and
    all columns are median-imputed so downstream models receive a dense matrix.
    """
    features = pd.DataFrame(index=df.index)
    features["Capital_Ratio"] = df["RBCT1J"] / df["ASSET"]
    features["Net_Interest_Margin"] = (df["INTINC"] - df["EINTEXP"]) / df["ASSET"]
    features["Texas_Ratio"] = (df["NAASSET"] + df["P9ASSET"]) / df["RBCT1J"]
    features["Charge_Off_Ratio"] = df["NTLS"] / df["ASSET"]

    for source_col, feature_name in PASSTHROUGH_RATIOS.items():
        if source_col in df.columns:
            features[feature_name] = pd.to_numeric(df[source_col], errors="coerce")

    features = features.replace([np.inf, -np.inf], np.nan)

    imputer = SimpleImputer(strategy="median")
    cleaned = pd.DataFrame(imputer.fit_transform(features), columns=features.columns, index=features.index)
    return cleaned


def remove_outliers_by_quantile(
    features: pd.DataFrame,
    low_quantile: float = 0.01,
    high_quantile: float = 0.99,
) -> pd.DataFrame:
    """Drop any row with a feature outside the [low, high] quantile band.

    Filtering is row-level: a single out-of-band feature removes the institution,
    so downstream models see a trimmed, more robust core distribution.
    """
    lower = features.quantile(low_quantile)
    upper = features.quantile(high_quantile)
    normal_mask = ~((features < lower) | (features > upper)).any(axis=1)
    return features[normal_mask]
