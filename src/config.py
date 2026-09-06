"""
Configuration module for Pearls AQI Predictor.
Centralizes location coordinates, API endpoints, feature definitions,
model hyperparameters, and US EPA AQI breakpoints.
"""

from pathlib import Path
import os

# Project root directory
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
PLOTS_DIR = BASE_DIR / "plots"

# Ensure essential directories exist
for folder in [DATA_DIR, MODEL_DIR, PLOTS_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

# Default City Configuration
DEFAULT_CITY = "Karachi"
CITIES = {
    "Karachi": {"latitude": 24.8607, "longitude": 67.0011, "timezone": "Asia/Karachi", "country": "Pakistan"},
    "Lahore": {"latitude": 31.5204, "longitude": 74.3587, "timezone": "Asia/Karachi", "country": "Pakistan"},
    "Delhi": {"latitude": 28.6139, "longitude": 77.2090, "timezone": "Asia/Kolkata", "country": "India"},
    "Beijing": {"latitude": 39.9042, "longitude": 116.4074, "timezone": "Asia/Shanghai", "country": "China"},
    "London": {"latitude": 51.5074, "longitude": -0.1278, "timezone": "Europe/London", "country": "United Kingdom"},
    "New York": {"latitude": 40.7128, "longitude": -74.0060, "timezone": "America/New_York", "country": "United States"},
    "Barcelona": {"latitude": 41.3851, "longitude": 2.1734, "timezone": "Europe/Madrid", "country": "Spain"},
    "Tokyo": {"latitude": 35.6762, "longitude": 139.6503, "timezone": "Asia/Tokyo", "country": "Japan"},
}

# API Endpoints (Open-Meteo provides free, keyless high-resolution weather & AQI)
AIR_QUALITY_API_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
WEATHER_ARCHIVE_API_URL = "https://archive-api.open-meteo.com/v1/archive"
WEATHER_FORECAST_API_URL = "https://api.open-meteo.com/v1/forecast"

# Raw Pollutant & Meteorological Features
POLLUTANT_VARS = [
    "pm2_5",
    "pm10",
    "us_aqi",
    "carbon_monoxide",
    "nitrogen_dioxide",
    "sulphur_dioxide",
    "ozone",
]

WEATHER_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "wind_direction_10m",
    "surface_pressure",
    "precipitation",
]

# Engineered Features used for Model Training & Inference
FEATURE_COLUMNS = [
    # Meteorological & Pollutants
    "pm2_5",
    "pm10",
    "carbon_monoxide",
    "nitrogen_dioxide",
    "sulphur_dioxide",
    "ozone",
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "surface_pressure",
    "precipitation",
    # Wind components
    "wind_u",
    "wind_v",
    # Pollutant interaction
    "pm_ratio",
    # Temporal & Cyclical
    "hour",
    "dayofweek",
    "month",
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
    # Lags
    "us_aqi_lag1",
    "us_aqi_lag3",
    "us_aqi_lag6",
    "us_aqi_lag12",
    "us_aqi_lag24",
    "pm2_5_lag1",
    "pm2_5_lag6",
    "pm2_5_lag24",
    # Rolling Statistics
    "us_aqi_roll_mean_6h",
    "us_aqi_roll_std_6h",
    "us_aqi_roll_mean_24h",
    "us_aqi_roll_std_24h",
    # Rate of Change
    "aqi_change_rate_1h",
    "aqi_change_rate_3h",
]

# Multi-step Prediction Targets (Forecast horizons)
TARGET_COLUMNS = [
    "us_aqi_target_24h",  # Day 1 prediction (+24 hours)
    "us_aqi_target_48h",  # Day 2 prediction (+48 hours)
    "us_aqi_target_72h",  # Day 3 prediction (+72 hours)
]

TARGET_HORIZONS = {
    "24h": 24,
    "48h": 48,
    "72h": 72,
}

# EPA AQI Breakpoints, Colors, and Health Risk Advisories
AQI_CATEGORIES = [
    {
        "name": "Good",
        "min": 0,
        "max": 50,
        "color": "#10B981",  # Emerald Green
        "icon": "🟢",
        "description": "Air quality is satisfactory, and air pollution poses little or no risk.",
        "advisory": "Great day to be active outside! Windows can stay open.",
    },
    {
        "name": "Moderate",
        "min": 51,
        "max": 100,
        "color": "#FBBF24",  # Amber / Yellow
        "icon": "🟡",
        "description": "Air quality is acceptable; however, sensitive people may experience minor issues.",
        "advisory": "Unusually sensitive individuals should consider reducing prolonged outdoor exertion.",
    },
    {
        "name": "Unhealthy for Sensitive Groups",
        "min": 101,
        "max": 150,
        "color": "#F97316",  # Orange
        "icon": "🟠",
        "description": "Members of sensitive groups may experience health effects. General public not likely affected.",
        "advisory": "People with respiratory or heart conditions, children, and elderly should limit outdoor exertion.",
    },
    {
        "name": "Unhealthy",
        "min": 151,
        "max": 200,
        "color": "#EF4444",  # Red
        "icon": "🔴",
        "description": "Everyone may begin to experience health effects; sensitive groups may experience more serious effects.",
        "advisory": "Wear an N95 mask outdoors, keep windows closed, and run air purifiers indoors.",
    },
    {
        "name": "Very Unhealthy",
        "min": 201,
        "max": 300,
        "color": "#8B5CF6",  # Purple
        "icon": "🟣",
        "description": "Health alert: The risk of health effects is increased for everyone.",
        "advisory": "Avoid strenuous outdoor activities. High-efficiency filtration recommended indoors.",
    },
    {
        "name": "Hazardous",
        "min": 301,
        "max": 1000,
        "color": "#7E22CE",  # Maroon / Deep Purple
        "icon": "☣️",
        "description": "Health warning of emergency conditions: Everyone is more likely to be affected.",
        "advisory": "EMERGENCY: Remain indoors with air filtration. Avoid all physical outdoor exertion.",
    },
]

# Hopsworks Feature Store Configuration
HOPSWORKS_PROJECT = os.getenv("HOPSWORKS_PROJECT_NAME", "pearls_aqi")
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY", "")
