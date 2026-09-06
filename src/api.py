"""
FastAPI Server for Pearls AQI Predictor.
Exposes RESTful endpoints for real-time predictions, latest features, and health checks.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.config import DEFAULT_CITY, CITIES, FEATURE_COLUMNS
from src.data_fetcher import fetch_live_and_forecast_data, get_city_coords
from src.feature_pipeline import compute_features
from src.models.registry import get_model_registry
from src.alerts import generate_forecast_alerts, get_general_recommendations

app = FastAPI(
    title="Pearls AQI Predictor API",
    description="End-to-End Serverless Air Quality Forecasting Service for 24h, 48h, and 72h horizons.",
    version="1.0.0",
)


class PredictionRequest(BaseModel):
    city: Optional[str] = DEFAULT_CITY
    latitude: Optional[float] = None
    longitude: Optional[float] = None


class ForecastResponse(BaseModel):
    city: str
    latitude: float
    longitude: float
    forecast_time: str
    predictions: Dict[str, float]
    alerts: List[Dict[str, Any]]
    recommendations: List[str]


@app.get("/")
def read_root():
    return {
        "service": "Pearls AQI Predictor",
        "status": "online",
        "version": "1.0.0",
        "docs": "/docs",
    }


@app.get("/health")
def health_check():
    registry = get_model_registry()
    champ_meta = registry.get_champion_metadata()
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "champion_model": champ_meta["name"] if champ_meta else "None",
    }


@app.post("/predict", response_model=ForecastResponse)
def predict_aqi(req: PredictionRequest):
    """
    Generate 24h, 48h, and 72h AQI predictions for a requested city or coordinate pair.
    """
    if req.latitude is not None and req.longitude is not None:
        lat, lon = req.latitude, req.longitude
        city_name = req.city or f"Coord({lat:.2f}, {lon:.2f})"
        tz = "auto"
    else:
        city_name = req.city or DEFAULT_CITY
        lat, lon, tz = get_city_coords(city_name)

    # 1. Load Champion Model
    registry = get_model_registry()
    try:
        model = registry.load_champion_model()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load champion model: {e}")

    # 2. Fetch recent + forecast observations
    try:
        aq_df, wx_df = fetch_live_and_forecast_data(lat, lon, timezone=tz, past_days=3, forecast_days=1)
        merged = pd.merge(aq_df, wx_df, on="time", how="inner")
        features_df = compute_features(merged, is_training=False)
        latest_row = features_df.iloc[-1:][FEATURE_COLUMNS]
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to fetch live feature data: {e}")

    # 3. Model Inference
    preds = model.predict(latest_row)[0]
    preds_dict = {
        "aqi_24h": round(float(preds[0]), 1),
        "aqi_48h": round(float(preds[1]), 1),
        "aqi_72h": round(float(preds[2]), 1),
    }

    # 4. Alerts and Advisory
    alerts = generate_forecast_alerts([preds[0], preds[1], preds[2]])
    max_aqi = max(preds)
    recommendations = get_general_recommendations(max_aqi)

    return ForecastResponse(
        city=city_name,
        latitude=lat,
        longitude=lon,
        forecast_time=datetime.now().isoformat(),
        predictions=preds_dict,
        alerts=alerts,
        recommendations=recommendations,
    )


@app.get("/metrics")
def get_model_metrics():
    registry = get_model_registry()
    return {
        "champion": registry.get_champion_metadata(),
        "all_models": registry.list_models(),
    }
