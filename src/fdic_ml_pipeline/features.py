from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer


def engineer_fdic_features(df: pd.DataFrame) -> pd.DataFrame:
    features = pd.DataFrame(index=df.index)
    features["Capital_Ratio"] = df["RBCT1J"] / df["ASSET"]
    features["Net_Interest_Margin"] = (df["INTINC"] - df["EINTEXP"]) / df["ASSET"]
    features["Texas_Ratio"] = (df["NAASSET"] + df["P9ASSET"]) / df["RBCT1J"]
    features["Charge_Off_Ratio"] = df["NTLS"] / df["ASSET"]
    features = features.replace([np.inf, -np.inf], np.nan)

    imputer = SimpleImputer(strategy="median")
    cleaned = pd.DataFrame(imputer.fit_transform(features), columns=features.columns, index=features.index)
    return cleaned


def remove_outliers_by_quantile(
    features: pd.DataFrame,
    low_quantile: float = 0.01,
    high_quantile: float = 0.99,
) -> pd.DataFrame:
    lower = features.quantile(low_quantile)
    upper = features.quantile(high_quantile)
    normal_mask = ~((features < lower) | (features > upper)).any(axis=1)
    return features[normal_mask]
