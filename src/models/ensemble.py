"""
Ensemble Models for Pearls AQI Predictor.
Implements Random Forest and Gradient Boosting Regressors.
"""

from typing import Dict, Any
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor


class EnsembleModel:
    """Multi-output tree ensemble regressor."""

    def __init__(
        self,
        model_type: str = "random_forest",
        n_estimators: int = 100,
        max_depth: int = 12,
        random_state: int = 42,
    ):
        self.model_type = model_type
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state

        if model_type == "gradient_boosting":
            base_estimator = HistGradientBoostingRegressor(
                max_iter=n_estimators,
                max_depth=max_depth,
                random_state=random_state,
            )
            self.model = MultiOutputRegressor(base_estimator)
        else:
            self.model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                n_jobs=-1,
                random_state=random_state,
            )
        self.is_fitted = False

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> "EnsembleModel":
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet.")
        preds = self.model.predict(X)
        return np.clip(preds, 0, 500)

    def get_params(self) -> Dict[str, Any]:
        return {
            "model_type": self.model_type,
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "random_state": self.random_state,
        }
