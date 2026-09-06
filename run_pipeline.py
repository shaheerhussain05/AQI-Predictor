"""
Master CLI Pipeline Runner for Pearls AQI Predictor.
Usage:
    python run_pipeline.py --backfill    # Backfill historical features into store
    python run_pipeline.py --train       # Retrain models, benchmark, and promote champion
    python run_pipeline.py --explain     # Compute SHAP feature importances & plots
    python run_pipeline.py --dashboard   # Launch Streamlit web dashboard
    python run_pipeline.py --api         # Start FastAPI inference server
    python run_pipeline.py --all         # Run full end-to-end pipeline
"""

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

# Ensure root in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import DEFAULT_CITY, BASE_DIR
from src.data_fetcher import fetch_historical_dataset
from src.feature_pipeline import compute_features
from src.feature_store import get_feature_store
from src.training_pipeline import run_training
from src.explainability import compute_feature_importance, plot_feature_importance
from src.models.registry import get_model_registry

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_backfill(city: str = DEFAULT_CITY, days_back: int = 730):
    """Fetch 2-year historical data, compute features & targets, save to Feature Store."""
    logger.info(f"=== STEP 1: HISTORICAL DATA BACKFILL ({city.upper()}, {days_back} DAYS) ===")
    raw_df = fetch_historical_dataset(city_name=city, days_back=days_back)
    logger.info(f"Fetched {len(raw_df)} raw hourly records.")

    logger.info("Computing time, cyclical, lag, rolling, and target features...")
    featured_df = compute_features(raw_df, is_training=True)
    logger.info(f"Engineered feature matrix: {featured_df.shape[0]} rows x {featured_df.shape[1]} columns.")

    # Save to local CSV for easy inspection / legacy compatibility
    featured_df.to_csv("data/historical_features.csv", index=False)

    # Save to Feature Store
    fs = get_feature_store()
    uri = fs.save_feature_group(
        df=featured_df,
        name="aqi_features",
        version=1,
        description=f"Hourly AQI & Weather features for {city} ({days_back} days backfill)",
    )
    logger.info(f"Backfill successfully saved to Feature Store: {uri}")
    return featured_df


def run_hourly_features(city: str = DEFAULT_CITY):
    """Fetch recent live observations, engineer features, and update Feature Store."""
    logger.info(f"=== HOURLY FEATURE INGESTION: {city.upper()} ===")
    from src.data_fetcher import get_city_coords, fetch_live_and_forecast_data
    lat, lon, tz = get_city_coords(city)
    aq_df, wx_df = fetch_live_and_forecast_data(lat, lon, timezone=tz, past_days=3, forecast_days=1)
    merged = pd.merge(aq_df, wx_df, on="time", how="inner")
    engineered = compute_features(merged, is_training=False)
    fs = get_feature_store()
    uri = fs.save_feature_group(
        df=engineered,
        name="aqi_live_features",
        version=1,
        description=f"Live hourly updated features for {city}",
    )
    logger.info(f"Updated live feature group at {uri}")
    return engineered


def run_training_step():
    """Train candidate models, benchmark, and update Model Registry."""
    logger.info("=== STEP 2: MODEL TRAINING & BENCHMARKING ===")
    champ_meta, leaderboard = run_training()
    print("\n" + "=" * 55)
    print("           MODEL BENCHMARK LEADERBOARD")
    print("=" * 55)
    print(leaderboard.to_string(index=False))
    print("=" * 55)
    logger.info(f"Champion Model: {champ_meta['model'].__class__.__name__}")
    return champ_meta, leaderboard


def run_explainability_step():
    """Compute SHAP feature importance plot and report."""
    logger.info("=== STEP 3: MODEL EXPLAINABILITY & SHAP ===")
    fs = get_feature_store()
    registry = get_model_registry()

    df = fs.read_feature_group("aqi_features", version=1)
    model = registry.load_champion_model()

    importance_df = compute_feature_importance(model, df.tail(200))
    plot_path = plot_feature_importance(importance_df)

    print("\n" + "=" * 45)
    print("       TOP 10 INFLUENTIAL FEATURES")
    print("=" * 45)
    print(importance_df.head(10).to_string(index=False))
    print("=" * 45)
    logger.info(f"Feature importance visualization saved to {plot_path}")


def launch_dashboard():
    """Start Streamlit dashboard."""
    logger.info("Launching Streamlit Web Dashboard on http://localhost:8501 ...")
    cmd = [sys.executable, "-m", "streamlit", "run", "app.py"]
    subprocess.run(cmd)


def launch_api():
    """Start FastAPI server."""
    logger.info("Launching FastAPI server on http://localhost:8000 ...")
    cmd = [sys.executable, "-m", "uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
    subprocess.run(cmd)


def main():
    parser = argparse.ArgumentParser(description="Pearls AQI Predictor End-to-End Pipeline")
    parser.add_argument("--backfill", action="store_true", help="Run historical data backfill into Feature Store")
    parser.add_argument("--features", action="store_true", help="Run hourly live feature ingestion into Feature Store")
    parser.add_argument("--city", type=str, default=DEFAULT_CITY, help="City name for backfill (default: Karachi)")
    parser.add_argument("--days", type=int, default=730, help="Days of history to backfill (default: 730)")
    parser.add_argument("--train", action="store_true", help="Run training pipeline and evaluate models")
    parser.add_argument("--explain", action="store_true", help="Generate SHAP feature importance analysis")
    parser.add_argument("--dashboard", action="store_true", help="Run Streamlit web dashboard")
    parser.add_argument("--api", action="store_true", help="Run FastAPI inference server")
    parser.add_argument("--all", action="store_true", help="Execute complete pipeline (Backfill -> Train -> Explain)")

    args = parser.parse_args()

    if args.all:
        run_backfill(city=args.city, days_back=args.days)
        run_training_step()
        run_explainability_step()
        logger.info("Complete pipeline finished successfully! Run with --dashboard to view the app.")
    elif args.backfill:
        run_backfill(city=args.city, days_back=args.days)
    elif args.features:
        run_hourly_features(city=args.city)
    elif args.train:
        run_training_step()
    elif args.explain:
        run_explainability_step()
    elif args.dashboard:
        launch_dashboard()
    elif args.api:
        launch_api()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
