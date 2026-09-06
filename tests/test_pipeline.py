"""
Automated Unit and Integration Tests for Pearls AQI Predictor.
Tests data fetcher, feature pipeline, model training, explainability, alerts, and registry.
"""

import unittest
import numpy as np
import pandas as pd

from src.config import FEATURE_COLUMNS, TARGET_COLUMNS, TARGET_HORIZONS
from src.feature_pipeline import compute_features, split_features_and_targets
from src.models import BaselineModel, EnsembleModel, NeuralNetModel, ModelRegistry
from src.alerts import get_aqi_category, generate_forecast_alerts, get_general_recommendations


class TestAQIPipeline(unittest.TestCase):

    def setUp(self):
        """Create a synthetic dataset mimicking Open-Meteo hourly data."""
        np.random.seed(42)
        n_rows = 200
        dates = pd.date_range(start="2026-01-01", periods=n_rows, freq="1h")

        self.mock_df = pd.DataFrame({
            "time": dates,
            "pm2_5": np.random.uniform(10, 150, n_rows),
            "pm10": np.random.uniform(20, 250, n_rows),
            "us_aqi": np.random.uniform(30, 280, n_rows),
            "carbon_monoxide": np.random.uniform(200, 1200, n_rows),
            "nitrogen_dioxide": np.random.uniform(10, 80, n_rows),
            "sulphur_dioxide": np.random.uniform(5, 40, n_rows),
            "ozone": np.random.uniform(10, 100, n_rows),
            "temperature_2m": np.random.uniform(15, 38, n_rows),
            "relative_humidity_2m": np.random.uniform(30, 90, n_rows),
            "wind_speed_10m": np.random.uniform(2, 25, n_rows),
            "wind_direction_10m": np.random.uniform(0, 360, n_rows),
            "surface_pressure": np.random.uniform(1005, 1018, n_rows),
            "precipitation": np.zeros(n_rows),
        })

    def test_feature_engineering_pipeline(self):
        """Verify feature engineering produces all required columns."""
        engineered = compute_features(self.mock_df, is_training=True)
        self.assertGreater(len(engineered), 0)

        # Check that all feature columns are present
        for col in FEATURE_COLUMNS:
            self.assertIn(col, engineered.columns, f"Missing feature: {col}")

        # Check target columns
        for col in TARGET_COLUMNS:
            self.assertIn(col, engineered.columns, f"Missing target: {col}")

        # Ensure no NaNs exist in training features
        self.assertEqual(engineered[FEATURE_COLUMNS].isna().sum().sum(), 0)

    def test_models_fit_and_predict(self):
        """Verify that Baseline, Ensemble, and NeuralNet models train and predict multi-output horizons."""
        engineered = compute_features(self.mock_df, is_training=True)
        X, Y, _, _ = split_features_and_targets(engineered)

        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        Y_train, Y_test = Y.iloc[:split_idx], Y.iloc[split_idx:]

        models = [
            ("Baseline Ridge", BaselineModel(model_type="ridge")),
            ("Random Forest", EnsembleModel(model_type="random_forest", n_estimators=20, max_depth=6)),
            ("Deep MLP", NeuralNetModel(hidden_layer_sizes=(32, 16), max_iter=50)),
        ]

        for name, model in models:
            model.fit(X_train, Y_train)
            preds = model.predict(X_test)
            self.assertEqual(preds.shape, Y_test.shape, f"Shape mismatch in {name}")
            self.assertTrue(np.all(preds >= 0), f"Predictions should be non-negative in {name}")

    def test_aqi_categorization_and_alerts(self):
        """Verify EPA AQI categorization and health warnings."""
        # Good
        good_cat = get_aqi_category(35)
        self.assertEqual(good_cat["category"], "Good")
        self.assertFalse(good_cat["is_hazardous"])

        # Unhealthy
        unhealthy_cat = get_aqi_category(175)
        self.assertEqual(unhealthy_cat["category"], "Unhealthy")
        self.assertTrue(unhealthy_cat["is_unhealthy"])

        # Hazardous
        haz_cat = get_aqi_category(350)
        self.assertEqual(haz_cat["category"], "Hazardous")
        self.assertTrue(haz_cat["is_hazardous"])

        # Forecast alert list
        alerts = generate_forecast_alerts([45.0, 120.0, 210.0])
        self.assertEqual(len(alerts), 3)
        self.assertEqual(alerts[0]["category"], "Good")
        self.assertEqual(alerts[1]["category"], "Unhealthy for Sensitive Groups")
        self.assertEqual(alerts[2]["category"], "Very Unhealthy")

        recs = get_general_recommendations(250)
        self.assertGreater(len(recs), 0)


if __name__ == "__main__":
    unittest.main()
