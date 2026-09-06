"""
Feature Store Abstraction for Pearls AQI Predictor.
Supports both Hopsworks Feature Store (Cloud) and Local Parquet Feature Store (Serverless).
"""

import json
import logging
import os
from pathlib import Path
from typing import Optional
import pandas as pd

from src.config import DATA_DIR, HOPSWORKS_API_KEY, HOPSWORKS_PROJECT

logger = logging.getLogger(__name__)

LOCAL_STORE_DIR = DATA_DIR / "feature_store"
LOCAL_STORE_DIR.mkdir(parents=True, exist_ok=True)


class FeatureStore:
    """Unified interface for saving and retrieving feature groups."""

    def __init__(self, use_hopsworks: bool = False):
        self.use_hopsworks = use_hopsworks and bool(HOPSWORKS_API_KEY)
        self.fs = None

        if self.use_hopsworks:
            try:
                import hopsworks
                logger.info("Connecting to Hopsworks Feature Store...")
                project = hopsworks.login(api_key=HOPSWORKS_API_KEY, project=HOPSWORKS_PROJECT)
                self.fs = project.get_feature_store()
                logger.info("Successfully connected to Hopsworks Feature Store.")
            except Exception as e:
                logger.warning(f"Failed to connect to Hopsworks ({e}). Falling back to Local Feature Store.")
                self.use_hopsworks = False

    def save_feature_group(
        self,
        df: pd.DataFrame,
        name: str = "aqi_features",
        version: int = 1,
        description: str = "Engineered AQI & Weather features",
        primary_key: list = ["time"],
    ) -> str:
        """Save a processed feature group into the store."""
        if self.use_hopsworks and self.fs:
            try:
                fg = self.fs.get_or_create_feature_group(
                    name=name,
                    version=version,
                    primary_key=primary_key,
                    description=description,
                    online_enabled=True,
                )
                fg.insert(df)
                logger.info(f"Inserted {len(df)} rows into Hopsworks Feature Group '{name}' v{version}.")
                return f"hopsworks://{name}_v{version}"
            except Exception as e:
                logger.error(f"Hopsworks insert failed: {e}. Saving to local store fallback.")

        # Local Serverless Parquet Fallback
        file_path = LOCAL_STORE_DIR / f"{name}_v{version}.parquet"
        df.to_parquet(file_path, index=False, engine="pyarrow")

        # Save metadata
        meta_path = LOCAL_STORE_DIR / f"{name}_v{version}_meta.json"
        meta = {
            "name": name,
            "version": version,
            "rows": len(df),
            "columns": list(df.columns),
            "created_at": pd.Timestamp.now().isoformat(),
            "description": description,
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        logger.info(f"Saved {len(df)} rows to Local Feature Store at {file_path}")
        return str(file_path)

    def read_feature_group(self, name: str = "aqi_features", version: int = 1) -> pd.DataFrame:
        """Read a feature group DataFrame from the store."""
        if self.use_hopsworks and self.fs:
            try:
                fg = self.fs.get_feature_group(name=name, version=version)
                return fg.read()
            except Exception as e:
                logger.warning(f"Hopsworks read failed: {e}. Trying local feature store fallback.")

        file_path = LOCAL_STORE_DIR / f"{name}_v{version}.parquet"
        if not file_path.exists():
            raise FileNotFoundError(f"Feature group '{name}' v{version} not found at {file_path}")

        return pd.read_parquet(file_path)

    def get_latest_features(self, name: str = "aqi_features", version: int = 1) -> pd.Series:
        """Fetch the most recent feature vector for inference."""
        df = self.read_feature_group(name=name, version=version)
        if "time" in df.columns:
            df = df.sort_values("time")
        return df.iloc[-1]


def get_feature_store() -> FeatureStore:
    """Factory helper to obtain the active FeatureStore instance."""
    return FeatureStore(use_hopsworks=bool(HOPSWORKS_API_KEY))
