"""
Data Fetcher Module for Pearls AQI Predictor.
Fetches historical archive data and real-time/forecast data for
pollutants and weather conditions from Open-Meteo APIs.
"""

from datetime import date, datetime, timedelta
import logging
from typing import Optional, Tuple
import requests
import pandas as pd

from src.config import (
    AIR_QUALITY_API_URL,
    WEATHER_ARCHIVE_API_URL,
    WEATHER_FORECAST_API_URL,
    POLLUTANT_VARS,
    WEATHER_VARS,
    CITIES,
    DEFAULT_CITY,
)

logger = logging.getLogger(__name__)


def get_city_coords(city_name: str) -> Tuple[float, float, str]:
    """Retrieve latitude, longitude, and timezone for a city name."""
    if city_name in CITIES:
        info = CITIES[city_name]
        return info["latitude"], info["longitude"], info["timezone"]
    default_info = CITIES[DEFAULT_CITY]
    return default_info["latitude"], default_info["longitude"], default_info["timezone"]


def fetch_air_quality_archive(
    latitude: float,
    longitude: float,
    start_date: date,
    end_date: date,
    timezone: str = "auto",
) -> pd.DataFrame:
    """
    Fetch historical hourly pollutant levels (PM2.5, PM10, US AQI, CO, NO2, SO2, O3)
    from Open-Meteo Air Quality API.
    """
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": ",".join(POLLUTANT_VARS),
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "timezone": timezone,
    }
    logger.info(f"Fetching AQ archive: lat={latitude}, lon={longitude} ({start_date} to {end_date})")
    response = requests.get(AIR_QUALITY_API_URL, params=params, timeout=60)
    response.raise_for_status()
    payload = response.json()

    if "hourly" not in payload:
        raise ValueError(f"Open-Meteo AQ API response missing 'hourly' field: {payload}")

    df = pd.DataFrame(payload["hourly"])
    df["time"] = pd.to_datetime(df["time"])
    return df


def fetch_weather_archive(
    latitude: float,
    longitude: float,
    start_date: date,
    end_date: date,
    timezone: str = "auto",
) -> pd.DataFrame:
    """
    Fetch historical hourly weather conditions (temperature, humidity, wind, pressure, rain)
    from Open-Meteo Historical Archive (ERA5).
    """
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": ",".join(WEATHER_VARS),
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "timezone": timezone,
    }
    logger.info(f"Fetching Weather archive: lat={latitude}, lon={longitude} ({start_date} to {end_date})")
    response = requests.get(WEATHER_ARCHIVE_API_URL, params=params, timeout=60)
    response.raise_for_status()
    payload = response.json()

    if "hourly" not in payload:
        raise ValueError(f"Open-Meteo Weather API response missing 'hourly' field: {payload}")

    df = pd.DataFrame(payload["hourly"])
    df["time"] = pd.to_datetime(df["time"])
    return df


def fetch_historical_dataset(
    city_name: str = DEFAULT_CITY,
    days_back: int = 730,
    end_lag_days: int = 3,
) -> pd.DataFrame:
    """
    Fetch and merge combined pollutant and meteorological dataset for the past N days.
    """
    lat, lon, tz = get_city_coords(city_name)
    end_dt = date.today() - timedelta(days=end_lag_days)
    start_dt = end_dt - timedelta(days=days_back)

    aq_df = fetch_air_quality_archive(lat, lon, start_dt, end_dt, tz)
    wx_df = fetch_weather_archive(lat, lon, start_dt, end_dt, tz)

    merged = pd.merge(aq_df, wx_df, on="time", how="inner")
    merged["city"] = city_name
    merged = merged.sort_values("time").reset_index(drop=True)
    return merged


def fetch_live_and_forecast_data(
    latitude: float,
    longitude: float,
    timezone: str = "auto",
    past_days: int = 3,
    forecast_days: int = 4,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fetches past few days of observations and upcoming 72h+ forecasts for live prediction.
    Returns (recent_and_forecast_aq_df, recent_and_forecast_weather_df).
    """
    # Air quality live & forecast
    aq_params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": ",".join(POLLUTANT_VARS),
        "past_days": past_days,
        "forecast_days": forecast_days,
        "timezone": timezone,
    }
    aq_res = requests.get(AIR_QUALITY_API_URL, params=aq_params, timeout=30)
    aq_res.raise_for_status()
    aq_df = pd.DataFrame(aq_res.json()["hourly"])
    aq_df["time"] = pd.to_datetime(aq_df["time"])

    # Weather forecast
    wx_params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": ",".join(WEATHER_VARS),
        "past_days": past_days,
        "forecast_days": forecast_days,
        "timezone": timezone,
    }
    wx_res = requests.get(WEATHER_FORECAST_API_URL, params=wx_params, timeout=30)
    wx_res.raise_for_status()
    wx_df = pd.DataFrame(wx_res.json()["hourly"])
    wx_df["time"] = pd.to_datetime(wx_df["time"])

    return aq_df, wx_df
