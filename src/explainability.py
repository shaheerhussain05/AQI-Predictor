"""
Explainability Module for Pearls AQI Predictor.
Computes Global Feature Importance and Local Instance Explanations
using SHAP (SHapley Additive exPlanations) and tree attribution.
"""

import logging
from typing import Dict, Any, Tuple, Optional
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import FEATURE_COLUMNS, PLOTS_DIR

logger = logging.getLogger(__name__)


def compute_feature_importance(
    model: Any,
    X_sample: pd.DataFrame,
    max_features: int = 15,
) -> pd.DataFrame:
    """
    Computes feature importance scores using SHAP values or model attribution.
    Returns a sorted DataFrame with feature names and absolute impact scores.
    """
    try:
        import shap

        # Unpack estimator if wrapped in pipeline/MultiOutputRegressor
        estimator = model
        if hasattr(model, "model"):
            estimator = model.model
        if hasattr(estimator, "estimators_"):
            # MultiOutput regressor: explain the 24h horizon model
            estimator = estimator.estimators_[0]

        # Use TreeExplainer or KernelExplainer
        try:
            explainer = shap.TreeExplainer(estimator)
            shap_values = explainer.shap_values(X_sample.iloc[:100])
        except Exception:
            explainer = shap.Explainer(estimator.predict, X_sample.iloc[:50])
            shap_values = explainer(X_sample.iloc[:50]).values

        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        if hasattr(shap_values, "values"):
            shap_values = shap_values.values

        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
        importance_df = pd.DataFrame({
            "Feature": FEATURE_COLUMNS[:len(mean_abs_shap)],
            "Importance": mean_abs_shap,
        })
    except Exception as e:
        logger.warning(f"SHAP explainer fallback triggered ({e}). Using tree feature importances.")
        # Fallback to ensemble feature importances
        estimator = model
        if hasattr(model, "model"):
            estimator = model.model
        if hasattr(estimator, "estimators_"):
            estimator = estimator.estimators_[0]

        if hasattr(estimator, "feature_importances_"):
            importances = estimator.feature_importances_
        else:
            # Linear model coefficients
            if hasattr(estimator, "coef_"):
                importances = np.abs(estimator.coef_)
            else:
                importances = np.ones(len(FEATURE_COLUMNS))

        importance_df = pd.DataFrame({
            "Feature": FEATURE_COLUMNS[:len(importances)],
            "Importance": importances,
        })

    importance_df = (
        importance_df.sort_values("Importance", ascending=False)
        .head(max_features)
        .reset_index(drop=True)
    )
    return importance_df


def plot_feature_importance(
    importance_df: pd.DataFrame,
    save_path: Optional[str] = None,
) -> str:
    """
    Generates a clean horizontal bar chart of the top feature importance factors.
    """
    if save_path is None:
        save_path = str(PLOTS_DIR / "shap_feature_importance.png")

    plt.figure(figsize=(9, 6), facecolor="#0E1117")
    ax = plt.subplot(111)
    ax.set_facecolor("#1E222D")

    # Invert so highest is on top
    df_sorted = importance_df.sort_values("Importance", ascending=True)

    bars = ax.barh(
        df_sorted["Feature"],
        df_sorted["Importance"],
        color="#3B82F6",
        edgecolor="#60A5FA",
        alpha=0.85,
    )

    ax.set_title("Pearls AQI Predictor — Key Feature Drivers (SHAP Importance)", fontsize=13, color="white", pad=12)
    ax.set_xlabel("Mean |SHAP Value| (Impact on AQI Forecast)", fontsize=11, color="#94A3B8")
    ax.tick_params(colors="white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color("#475569")
    ax.spines["left"].set_color("#475569")

    # Add numeric labels to bars
    for bar in bars:
        width = bar.get_width()
        ax.text(
            width * 1.02,
            bar.get_y() + bar.get_height() / 2,
            f"{width:.2f}",
            va="center",
            ha="left",
            color="#E2E8F0",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved feature importance plot to {save_path}")
    return save_path


def explain_single_prediction(
    model: Any,
    row: pd.Series,
    baseline_val: float = 120.0,
) -> pd.DataFrame:
    """
    Calculates individual feature pushes (waterfall breakdown) for a single live observation.
    """
    estimator = model
    if hasattr(model, "model"):
        estimator = model.model
    if hasattr(estimator, "estimators_"):
        estimator = estimator.estimators_[0]

    feature_vals = row[FEATURE_COLUMNS] if all(c in row for c in FEATURE_COLUMNS) else row

    if hasattr(estimator, "feature_importances_"):
        weights = estimator.feature_importances_
    else:
        weights = np.ones(len(FEATURE_COLUMNS)) / len(FEATURE_COLUMNS)

    contributions = []
    for feat, val, w in zip(FEATURE_COLUMNS, feature_vals, weights):
        # Contribution proxy: deviation relative to feature weight
        contrib = float(w * (float(val) if isinstance(val, (int, float, np.number)) else 0.0))
        contributions.append({
            "Feature": feat,
            "Value": round(float(val), 2) if isinstance(val, (int, float, np.number)) else str(val),
            "Impact": round(contrib, 2),
        })

    contrib_df = pd.DataFrame(contributions).sort_values("Impact", ascending=False).reset_index(drop=True)
    return contrib_df
