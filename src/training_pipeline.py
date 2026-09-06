"""
Training Pipeline for Pearls AQI Predictor.
Orchestrates model training, chronological evaluation, performance benchmarking
(RMSE, MAE, R²), and promotion to the Model Registry.
"""

import logging
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.config import TARGET_COLUMNS, TARGET_HORIZONS
from src.feature_store import get_feature_store
from src.feature_pipeline import split_features_and_targets
from src.models import (
    BaselineModel,
    EnsembleModel,
    NeuralNetModel,
    get_model_registry,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def evaluate_predictions(y_true: pd.DataFrame, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute RMSE, MAE, and R² across each target horizon and overall.
    """
    y_true_np = y_true.to_numpy()
    metrics = {}

    horizon_names = list(TARGET_HORIZONS.keys())
    for idx, h in enumerate(horizon_names):
        true_h = y_true_np[:, idx]
        pred_h = y_pred[:, idx]

        rmse = float(np.sqrt(mean_squared_error(true_h, pred_h)))
        mae = float(mean_absolute_error(true_h, pred_h))
        r2 = float(r2_score(true_h, pred_h))

        metrics[f"rmse_{h}"] = round(rmse, 2)
        metrics[f"mae_{h}"] = round(mae, 2)
        metrics[f"r2_{h}"] = round(r2, 4)

    # Aggregate overall metrics
    metrics["overall_rmse"] = round(float(np.sqrt(mean_squared_error(y_true_np, y_pred))), 2)
    metrics["overall_mae"] = round(float(mean_absolute_error(y_true_np, y_pred)), 2)
    metrics["overall_r2"] = round(float(r2_score(y_true_np, y_pred)), 4)
    return metrics


def run_training(
    df: pd.DataFrame = None,
    train_split_ratio: float = 0.8,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Execute end-to-end model training, evaluation, and registry promotion.
    """
    fs = get_feature_store()
    registry = get_model_registry()

    if df is None:
        logger.info("Loading training data from Feature Store...")
        try:
            df = fs.read_feature_group("aqi_features", version=1)
        except Exception as e:
            logger.error(f"Failed to load from feature store: {e}. Please run backfill first.")
            raise

    X, Y, feat_cols, target_cols = split_features_and_targets(df)
    logger.info(f"Dataset shape: {X.shape[0]} rows x {X.shape[1]} features.")

    # Chronological Split (prevents time-series lookahead bias)
    split_idx = int(len(X) * train_split_ratio)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    Y_train, Y_test = Y.iloc[:split_idx], Y.iloc[split_idx:]

    logger.info(f"Train samples: {len(X_train)} | Test samples: {len(X_test)}")

    # Model Candidate Suite
    candidate_models = {
        "Ridge_Baseline": BaselineModel(model_type="ridge", alpha=1.0),
        "Random_Forest": EnsembleModel(model_type="random_forest", n_estimators=100, max_depth=12),
        "Gradient_Boosting": EnsembleModel(model_type="gradient_boosting", n_estimators=120, max_depth=8),
        "Deep_MLP": NeuralNetModel(hidden_layer_sizes=(128, 64, 32), max_iter=250),
    }

    results = []
    trained_artifacts = {}

    for name, model in candidate_models.items():
        logger.info(f"Training {name}...")
        model.fit(X_train, Y_train)
        y_pred = model.predict(X_test)
        metrics = evaluate_predictions(Y_test, y_pred)
        params = model.get_params()

        trained_artifacts[name] = {
            "model": model,
            "metrics": metrics,
            "params": params,
        }

        results.append({
            "Model": name,
            "Overall RMSE": metrics["overall_rmse"],
            "Overall MAE": metrics["overall_mae"],
            "Overall R²": metrics["overall_r2"],
            "RMSE 24h": metrics["rmse_24h"],
            "RMSE 48h": metrics["rmse_48h"],
            "RMSE 72h": metrics["rmse_72h"],
        })
        logger.info(f"-> {name} Results: RMSE={metrics['overall_rmse']} | MAE={metrics['overall_mae']} | R²={metrics['overall_r2']}")

    results_df = pd.DataFrame(results).sort_values("Overall RMSE").reset_index(drop=True)
    best_model_name = results_df.iloc[0]["Model"]
    logger.info(f"Champion Model Selected: {best_model_name}")

    # Register all models and promote champion
    for name, art in trained_artifacts.items():
        is_champ = (name == best_model_name)
        registry.save_model(
            model=art["model"],
            name=name,
            metrics=art["metrics"],
            parameters=art["params"],
            is_champion=is_champ,
        )

    return trained_artifacts[best_model_name], results_df


if __name__ == "__main__":
    champion, leaderboard = run_training()
    print("\n================= MODEL LEADERBOARD =================")
    print(leaderboard.to_string(index=False))
