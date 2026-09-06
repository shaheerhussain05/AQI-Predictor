"""
Feature Engineering Pipeline for Pearls AQI Predictor.
Computes temporal, cyclical, meteorological, lag, rolling, and rate-of-change
features, and generates multi-step forecasting targets.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Optional

from src.config import FEATURE_COLUMNS, TARGET_COLUMNS, TARGET_HORIZONS


def compute_features(df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
    """
    Transform raw merged pollutant and weather dataframe into an engineered feature dataframe.
    
    Parameters:
        df: DataFrame containing hourly observations with 'time', pollutant, and weather columns.
        is_training: If True, computes target columns (shifted into future).
    
    Returns:
        Processed DataFrame with all FEATURE_COLUMNS (and TARGET_COLUMNS if is_training).
    """
    df = df.copy()
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])
        df = df.sort_values("time").reset_index(drop=True)

    # 1. Temporal Features
    df["hour"] = df["time"].dt.hour
    df["day"] = df["time"].dt.day
    df["dayofweek"] = df["time"].dt.dayofweek
    df["month"] = df["time"].dt.month

    # 2. Cyclical Encodings (captures diurnal & seasonal continuous loops)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12.0)

    # 3. Wind Vector Components (meteorological standard decomposition)
    if "wind_speed_10m" in df.columns and "wind_direction_10m" in df.columns:
        rad = np.deg2rad(df["wind_direction_10m"])
        df["wind_u"] = -df["wind_speed_10m"] * np.sin(rad)
        df["wind_v"] = -df["wind_speed_10m"] * np.cos(rad)
    else:
        df["wind_u"] = 0.0
        df["wind_v"] = 0.0

    # 4. Particulate Matter Interaction Ratio
    if "pm2_5" in df.columns and "pm10" in df.columns:
        df["pm_ratio"] = df["pm2_5"] / (df["pm10"].replace(0, np.nan) + 1e-4)
        df["pm_ratio"] = df["pm_ratio"].fillna(0.5).clip(0.0, 1.0)
    else:
        df["pm_ratio"] = 0.5

    # 5. Lag Features (AQI and PM2.5 persistence)
    for lag in [1, 3, 6, 12, 24]:
        df[f"us_aqi_lag{lag}"] = df["us_aqi"].shift(lag)

    for lag in [1, 6, 24]:
        df[f"pm2_5_lag{lag}"] = df["pm2_5"].shift(lag)

    # 6. Rolling Statistics
    df["us_aqi_roll_mean_6h"] = df["us_aqi"].rolling(window=6, min_periods=1).mean()
    df["us_aqi_roll_std_6h"] = df["us_aqi"].rolling(window=6, min_periods=1).std().fillna(0)
    df["us_aqi_roll_mean_24h"] = df["us_aqi"].rolling(window=24, min_periods=1).mean()
    df["us_aqi_roll_std_24h"] = df["us_aqi"].rolling(window=24, min_periods=1).std().fillna(0)

    # 7. Derived Rate of Change (AQI velocity)
    df["aqi_change_rate_1h"] = df["us_aqi"] - df["us_aqi_lag1"]
    df["aqi_change_rate_3h"] = (df["us_aqi"] - df["us_aqi_lag3"]) / 3.0

    # 8. Target Horizons (for supervised training)
    if is_training:
        for name, step in TARGET_HORIZONS.items():
            # Shift backwards so row t has target at t + step
            df[f"us_aqi_target_{name}"] = df["us_aqi"].shift(-step)

        # Drop rows where lag features or targets are NaN (due to shifting)
        target_cols = [f"us_aqi_target_{name}" for name in TARGET_HORIZONS.keys()]
        df = df.dropna(subset=FEATURE_COLUMNS + target_cols).reset_index(drop=True)
    else:
        # In inference mode, forward-fill any edge rolling nans
        df = df.bfill().ffill().reset_index(drop=True)

    return df


def split_features_and_targets(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], List[str]]:
    """
    Extract feature matrix X and multi-step target matrix Y from engineered dataset.
    """
    X = df[FEATURE_COLUMNS].copy()
    Y = df[TARGET_COLUMNS].copy()
    return X, Y, FEATURE_COLUMNS, TARGET_COLUMNS
