"""
Model Registry for Pearls AQI Predictor.
Manages model versioning, performance metadata, serialization, and promotion.
Supports Local Model Registry and Hopsworks Model Registry.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import joblib

from src.config import MODEL_DIR, HOPSWORKS_API_KEY, HOPSWORKS_PROJECT

logger = logging.getLogger(__name__)

REGISTRY_METADATA_FILE = MODEL_DIR / "model_registry.json"


class ModelRegistry:
    """Manages model artifacts, metadata, and champion promotion."""

    def __init__(self, use_hopsworks: bool = False):
        self.use_hopsworks = use_hopsworks and bool(HOPSWORKS_API_KEY)
        self.mr = None

        if self.use_hopsworks:
            try:
                import hopsworks
                project = hopsworks.login(api_key=HOPSWORKS_API_KEY, project=HOPSWORKS_PROJECT)
                self.mr = project.get_model_registry()
                logger.info("Connected to Hopsworks Model Registry.")
            except Exception as e:
                logger.warning(f"Failed to connect to Hopsworks MR ({e}). Using Local Model Registry.")
                self.use_hopsworks = False

        self._init_local_metadata()

    def _init_local_metadata(self):
        if not REGISTRY_METADATA_FILE.exists():
            with open(REGISTRY_METADATA_FILE, "w", encoding="utf-8") as f:
                json.dump({"models": [], "champion": None}, f, indent=2)

    def _read_metadata(self) -> Dict[str, Any]:
        try:
            with open(REGISTRY_METADATA_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {"models": [], "champion": None}

    def _write_metadata(self, meta: Dict[str, Any]):
        with open(REGISTRY_METADATA_FILE, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    def save_model(
        self,
        model: Any,
        name: str,
        metrics: Dict[str, float],
        parameters: Dict[str, Any],
        is_champion: bool = False,
    ) -> str:
        """Save a model artifact and register its metrics."""
        import datetime
        version = len([m for m in self._read_metadata()["models"] if m["name"] == name]) + 1
        filename = f"{name}_v{version}.joblib"
        artifact_path = MODEL_DIR / filename
        joblib.dump(model, artifact_path)

        record = {
            "name": name,
            "version": version,
            "artifact_path": str(artifact_path),
            "filename": filename,
            "metrics": metrics,
            "parameters": parameters,
            "created_at": datetime.datetime.now().isoformat(),
            "is_champion": is_champion,
        }

        meta = self._read_metadata()
        meta["models"].append(record)
        if is_champion or meta["champion"] is None:
            # Also save a standard champion.joblib pointer for fastest zero-lookup inference
            joblib.dump(model, MODEL_DIR / "champion.joblib")
            meta["champion"] = record
            for m in meta["models"]:
                if m != record:
                    m["is_champion"] = False

        self._write_metadata(meta)
        logger.info(f"Registered model '{name}' v{version} (Champion: {is_champion}) at {artifact_path}")

        # Cloud Hopsworks Registration
        if self.use_hopsworks and self.mr:
            try:
                hw_model = self.mr.python.create_model(
                    name=name,
                    metrics=metrics,
                    description=f"Pearls AQI Predictor - {name}",
                )
                hw_model.save(str(artifact_path))
                logger.info(f"Pushed model '{name}' v{version} to Hopsworks Model Registry.")
            except Exception as e:
                logger.warning(f"Could not push to Hopsworks MR: {e}")

        return str(artifact_path)

    def load_champion_model(self) -> Any:
        """Load the current active champion model."""
        champion_path = MODEL_DIR / "champion.joblib"
        if champion_path.exists():
            return joblib.load(champion_path)

        meta = self._read_metadata()
        if meta.get("champion"):
            champ_file = Path(meta["champion"]["artifact_path"])
            if champ_file.exists():
                return joblib.load(champ_file)

        raise FileNotFoundError("No trained champion model found in registry.")

    def get_champion_metadata(self) -> Optional[Dict[str, Any]]:
        """Retrieve champion model performance metrics and parameters."""
        meta = self._read_metadata()
        return meta.get("champion")

    def list_models(self) -> List[Dict[str, Any]]:
        """List all models registered."""
        return self._read_metadata().get("models", [])


def get_model_registry() -> ModelRegistry:
    return ModelRegistry(use_hopsworks=bool(HOPSWORKS_API_KEY))
