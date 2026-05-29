from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class SupervisedResult:
    model_name: str
    metrics: dict[str, float]
    predictions_df: pd.DataFrame
    explanation_df: pd.DataFrame


def _build_explanation_frame(model: object, features: pd.DataFrame) -> pd.DataFrame:
    if isinstance(model, Pipeline) and "clf" in model.named_steps:
        clf = model.named_steps["clf"]
        if hasattr(clf, "coef_"):
            coef = clf.coef_
            if getattr(coef, "ndim", 1) > 1:
                weights = abs(coef).mean(axis=0)
            else:
                weights = coef
            explanation = pd.DataFrame(
                {
                    "feature": features.columns,
                    "weight": weights,
                    "abs_weight": abs(weights),
                    "importance_type": "coefficient",
                }
            )
            return explanation.sort_values("abs_weight", ascending=False)

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        explanation = pd.DataFrame(
            {
                "feature": features.columns,
                "weight": importances,
                "abs_weight": abs(importances),
                "importance_type": "feature_importance",
            }
        )
        return explanation.sort_values("abs_weight", ascending=False)

    return pd.DataFrame(
        {"feature": features.columns, "weight": 0.0, "abs_weight": 0.0, "importance_type": "unknown"}
    )


def train_supervised(
    features_df: pd.DataFrame,
    labels: pd.Series,
    random_seed: int = 42,
) -> SupervisedResult:
    x_train, x_test, y_train, y_test = train_test_split(
        features_df, labels, test_size=0.25, random_state=random_seed, stratify=labels
    )

    candidates: dict[str, object] = {
        "logistic_regression_l2": Pipeline(
            [("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=2000, random_state=random_seed))]
        ),
        "logistic_regression_elastic_net": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=2000,
                        random_state=random_seed,
                        solver="saga",
                        penalty="elasticnet",
                        l1_ratio=0.5,
                    ),
                ),
            ]
        ),
        "decision_tree_shallow": DecisionTreeClassifier(max_depth=3, min_samples_leaf=2, random_state=random_seed),
    }

    best_name = ""
    best_metrics: dict[str, float] = {}
    best_score = -1.0
    best_predictions = pd.DataFrame(index=x_test.index)
    best_model: object | None = None

    for name, model in candidates.items():
        model.fit(x_train, y_train)
        preds = model.predict(x_test)

        f1 = f1_score(y_test, preds, average="weighted")
        acc = accuracy_score(y_test, preds)
        metrics = {"f1_weighted": float(f1), "accuracy": float(acc)}

        if len(y_test.unique()) == 2 and hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(x_test)[:, 1]
            metrics["roc_auc"] = float(roc_auc_score(y_test, y_proba))
            rank_score = metrics["roc_auc"]
        else:
            rank_score = metrics["f1_weighted"]

        if rank_score > best_score:
            best_score = rank_score
            best_name = name
            best_metrics = metrics
            if len(y_test.unique()) == 2 and hasattr(model, "predict_proba"):
                best_predictions = pd.DataFrame(
                    {"y_true": y_test, "y_pred": preds, "y_score": model.predict_proba(x_test)[:, 1]},
                    index=x_test.index,
                )
            else:
                best_predictions = pd.DataFrame({"y_true": y_test, "y_pred": preds}, index=x_test.index)
            best_model = model

    if best_model is None:
        raise RuntimeError("No supervised model was selected.")

    explanation_df = _build_explanation_frame(best_model, features_df)
    return SupervisedResult(
        model_name=best_name,
        metrics=best_metrics,
        predictions_df=best_predictions,
        explanation_df=explanation_df,
    )
