"""
Statistical Baseline Models for Pearls AQI Predictor.
Implements Ridge Regression and ElasticNet with Feature Scaling.
"""

from typing import Dict, Any
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


class BaselineModel:
    """Multi-output linear regularized baseline model."""

    def __init__(self, model_type: str = "ridge", alpha: float = 1.0, random_state: int = 42):
        self.model_type = model_type
        self.alpha = alpha
        self.random_state = random_state

        if model_type == "elastic_net":
            base_estimator = ElasticNet(alpha=alpha, l1_ratio=0.5, random_state=random_state, max_iter=2000)
        else:
            base_estimator = Ridge(alpha=alpha, random_state=random_state)

        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", MultiOutputRegressor(base_estimator)),
        ])
        self.is_fitted = False

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> "BaselineModel":
        self.pipeline.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet.")
        preds = self.pipeline.predict(X)
        return np.clip(preds, 0, 500)  # AQI is strictly non-negative

    def get_params(self) -> Dict[str, Any]:
        return {
            "model_type": self.model_type,
            "alpha": self.alpha,
            "random_state": self.random_state,
        }
