"""
Deep Learning Neural Network Model for Pearls AQI Predictor.
Implements a multi-layer perceptron (MLP) regressor architecture with
scaled inputs and early stopping.
"""

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


class NeuralNetModel:
    """Deep Neural Network Regressor for multi-horizon AQI forecasting."""

    def __init__(
        self,
        hidden_layer_sizes: Tuple[int, ...] = (128, 64, 32),
        activation: str = "relu",
        alpha: float = 0.001,
        learning_rate_init: float = 0.001,
        max_iter: int = 300,
        random_state: int = 42,
    ):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.alpha = alpha
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.random_state = random_state

        mlp = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            alpha=alpha,
            learning_rate_init=learning_rate_init,
            max_iter=max_iter,
            early_stopping=True,
            n_iter_no_change=15,
            random_state=random_state,
        )

        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("mlp", mlp),
        ])
        self.is_fitted = False

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> "NeuralNetModel":
        self.pipeline.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet.")
        preds = self.pipeline.predict(X)
        return np.clip(preds, 0, 500)

    def get_params(self) -> Dict[str, Any]:
        return {
            "model_type": "deep_mlp",
            "hidden_layer_sizes": self.hidden_layer_sizes,
            "activation": self.activation,
            "alpha": self.alpha,
            "learning_rate_init": self.learning_rate_init,
            "max_iter": self.max_iter,
            "random_state": self.random_state,
        }
