"""Composite CAMELS risk score: rank institutions into a single watchlist.

Clustering answers "which group is a bank in?"; this module answers "how risky is
each bank, overall?". Every available feature is standardized (z-score) and oriented
so that larger values mean *more* risk, then averaged into one interpretable score.
Higher score = higher aggregate risk. The top-N banks form a watchlist.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

# Orientation of each feature toward risk:
#   +1 -> higher value means MORE risk (e.g. more nonperforming assets)
#   -1 -> higher value means LESS risk (e.g. more capital, more profit)
# Only features present in the data are used, so this works for the 4-ratio and the
# full 15-ratio CAMELS panel alike.
RISK_DIRECTION: dict[str, int] = {
    "Capital_Ratio": -1,
    "Net_Interest_Margin": -1,
    "Texas_Ratio": +1,
    "Charge_Off_Ratio": +1,
    "Total_Capital_Ratio": -1,
    "Tier1_Leverage_Ratio": -1,
    "CET1_Ratio": -1,
    "Nonperforming_Assets_Pct": +1,
    "NCO_to_Loans": +1,
    "Reserves_to_Loans": +1,
    "Efficiency_Ratio": +1,
    "Return_on_Assets": -1,
    "Return_on_Equity": -1,
    "Net_Interest_Margin_Pct": -1,
    "Loans_to_Deposits": +1,
}


@dataclass
class RiskScoreResult:
    """Full scored table, the top-N watchlist, and the features actually scored."""

    scored_df: pd.DataFrame
    watchlist_df: pd.DataFrame
    scored_features: list[str]


def compute_risk_scores(features_df: pd.DataFrame, top_n: int = 20) -> RiskScoreResult:
    """Score and rank every institution by a composite, risk-oriented z-score.

    Each scored feature is standardized and multiplied by its :data:`RISK_DIRECTION`
    sign, then averaged. Returns a ``scored_df`` (``risk_score`` + integer ``risk_rank``,
    sorted riskiest-first) and the ``top_n`` slice as ``watchlist_df``. Features with
    zero variance are skipped; if none remain the scores are all zero.
    """
    scored_features = [c for c in features_df.columns if c in RISK_DIRECTION]

    oriented = pd.DataFrame(index=features_df.index)
    for col in scored_features:
        series = features_df[col].astype(float)
        std = series.std()
        if std == 0 or pd.isna(std):
            continue
        oriented[col] = ((series - series.mean()) / std) * RISK_DIRECTION[col]

    used = list(oriented.columns)
    scored = pd.DataFrame(index=features_df.index)
    scored["risk_score"] = oriented.mean(axis=1) if used else 0.0
    scored = scored.sort_values("risk_score", ascending=False)
    scored["risk_rank"] = range(1, len(scored) + 1)

    watchlist = scored.head(top_n).copy()
    return RiskScoreResult(scored_df=scored, watchlist_df=watchlist, scored_features=used)
