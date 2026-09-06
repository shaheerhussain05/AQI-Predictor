"""
Step 1 — Data collection + EDA for Karachi AQI prediction.

What this script does:
  1. Fetches ~2 years of HOURLY air-quality data (PM2.5, PM10, US AQI, gases)
     and weather data (temp, humidity, wind, pressure, rain) for Karachi
     from the free Open-Meteo API (no API key needed).
  2. Merges them, aggregates to DAILY values.
  3. Runs basic EDA (summary stats + 3 plots).
  4. Saves a clean daily dataset to data/karachi_aqi_daily.csv

Run it with:  python 01_data_and_eda.py
"""

import os
from datetime import date, timedelta

import requests
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # save plots to files instead of opening windows
import matplotlib.pyplot as plt

# ---- Karachi location ----
LAT, LON = 24.8607, 67.0011
CITY = "Karachi"

# ---- Date range: last ~2 years, ending 3 days ago (weather archive has a short lag) ----
END = date.today() - timedelta(days=3)
START = END - timedelta(days=730)

# folders for outputs
os.makedirs("data", exist_ok=True)
os.makedirs("plots", exist_ok=True)


def fetch_air_quality(start, end):
    """Hourly pollutants + US AQI from Open-Meteo Air Quality API."""
    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = {
        "latitude": LAT,
        "longitude": LON,
        "hourly": "pm2_5,pm10,us_aqi,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone",
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "timezone": "Asia/Karachi",
    }
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    return pd.DataFrame(r.json()["hourly"])


def fetch_weather(start, end):
    """Hourly weather from Open-Meteo historical archive (ERA5)."""
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": LAT,
        "longitude": LON,
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m,surface_pressure,precipitation",
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "timezone": "Asia/Karachi",
    }
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    return pd.DataFrame(r.json()["hourly"])


def to_daily(aq_df, wx_df):
    """Merge hourly frames on time, then aggregate to daily."""
    df = pd.merge(aq_df, wx_df, on="time", how="inner")
    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time").sort_index()

    # daily means for everything, plus the daily MAX AQI (worst hour of the day)
    daily = df.resample("D").mean()
    daily["us_aqi_max"] = df["us_aqi"].resample("D").max()

    daily = daily.dropna(subset=["us_aqi"])  # drop days with no AQI reading
    return daily


def run_eda(daily):
    """Print summary + save 3 plots to plots/."""
    print("\n=== SHAPE ===")
    print(daily.shape, "rows x cols  (rows = days)")
    print("\n=== FIRST FEW ROWS ===")
    print(daily.head())
    print("\n=== SUMMARY STATS ===")
    print(daily[["us_aqi", "pm2_5", "pm10", "temperature_2m"]].describe().round(1))

    # 1) AQI over time
    plt.figure(figsize=(11, 4))
    plt.plot(daily.index, daily["us_aqi"], linewidth=0.8)
    plt.title(f"{CITY} — Daily US AQI over time")
    plt.ylabel("US AQI")
    plt.tight_layout()
    plt.savefig("plots/aqi_over_time.png", dpi=110)
    plt.close()

    # 2) AQI distribution
    plt.figure(figsize=(7, 4))
    plt.hist(daily["us_aqi"].dropna(), bins=40)
    plt.title(f"{CITY} — Distribution of daily US AQI")
    plt.xlabel("US AQI")
    plt.ylabel("Number of days")
    plt.tight_layout()
    plt.savefig("plots/aqi_distribution.png", dpi=110)
    plt.close()

    # 3) Correlation heatmap (which features move with AQI?)
    cols = ["us_aqi", "pm2_5", "pm10", "carbon_monoxide", "nitrogen_dioxide",
            "ozone", "temperature_2m", "relative_humidity_2m",
            "wind_speed_10m", "surface_pressure"]
    corr = daily[cols].corr()
    plt.figure(figsize=(8, 7))
    im = plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(im, fraction=0.046)
    plt.xticks(range(len(cols)), cols, rotation=90)
    plt.yticks(range(len(cols)), cols)
    plt.title(f"{CITY} — Feature correlation")
    plt.tight_layout()
    plt.savefig("plots/correlation_heatmap.png", dpi=110)
    plt.close()

    print("\nSaved 3 plots to the plots/ folder.")


def main():
    print(f"Fetching {CITY} data from {START} to {END} ...")
    aq = fetch_air_quality(START, END)
    wx = fetch_weather(START, END)
    print(f"  air-quality rows (hourly): {len(aq)}")
    print(f"  weather rows (hourly):     {len(wx)}")

    daily = to_daily(aq, wx)
    run_eda(daily)

    out = "data/karachi_aqi_daily.csv"
    daily.to_csv(out)
    print(f"\nSaved clean daily dataset -> {out}")
    print("Step 1 done. Next: feature engineering + target.")


if __name__ == "__main__":
    main()
