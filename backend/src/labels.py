"""Construct a transparent supervised target when the dataset ships no label.

FDIC call reports carry no failure/event flag, so the supervised track would otherwise
be unusable. This module builds a **proxy "elevated stress" label** from an asset-quality
ratio (top-decile institutions = stressed). It is deliberately simple and rule-based so
the target is auditable.

Leakage caveat: because the label is derived from the features, a model trained on the
*same* asset-quality feature would be circular. The pipeline therefore drops the
defining feature before training, turning the exercise into a genuine question — *can the
other CAMELS dimensions predict asset-quality stress?* — rather than a tautology.
"""

from __future__ import annotations

import pandas as pd

# Preference order for the ratio used to define stress (first one present wins).
STRESS_SOURCE_FEATURES = ("Nonperforming_Assets_Pct", "Texas_Ratio")


def build_stress_label(
    features_df: pd.DataFrame, quantile: float = 0.90
) -> tuple[pd.Series, str]:
    """Return a binary ``elevated_stress`` label and the feature used to define it.

    The institution is flagged (1) when its defining asset-quality ratio is at or above
    the ``quantile`` threshold, else 0. Raises ``ValueError`` if no known asset-quality
    feature is available.
    """
    source = next((c for c in STRESS_SOURCE_FEATURES if c in features_df.columns), None)
    if source is None:
        raise ValueError(
            "Cannot construct a stress label: none of "
            f"{STRESS_SOURCE_FEATURES} are present in the feature set."
        )
    threshold = features_df[source].quantile(quantile)
    label = (features_df[source] >= threshold).astype(int)
    label.name = "elevated_stress"
    return label, source
