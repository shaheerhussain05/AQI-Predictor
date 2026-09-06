# 🌍 Pearls AQI Predictor — Serverless End-to-End ML System

> Predict the Air Quality Index (AQI) in your city across a 3-day (72-hour) horizon using a 100% serverless machine learning architecture.

[![Python](https://img.shields.io/badge/Python-3.11%2B%20%7C%203.14-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.4%2B-F7931E.svg)](https://scikit-learn.org/)
[![Status](https://img.shields.io/badge/Pipeline-Automated-success.svg)]()

---

## 📌 Architecture Overview

```
[External Sensor APIs] (Open-Meteo Air Quality & Weather / ERA5)
          │
          ▼
[Feature Pipeline] (Temporal, Cyclical, Lags, Rolling Stats, AQI Velocity)
          │
          ▼
[Feature Store] (Local Parquet / Hopsworks Cloud Dual-Engine)
          │
     ┌────┴──────────────────────────┐
     ▼                               ▼
[Training Pipeline]           [Inference Service]
├── Ridge Baseline            ├── FastAPI REST API (`/predict`)
├── Random Forest (Champion)  └── Streamlit Dashboard (`app.py`)
├── Gradient Boosting                │
└── Deep Neural Network (MLP)        ▼
     │                        [EPA Health Alerts & SHAP Explanations]
     ▼
[Model Registry] (Metrics, Versioning, Artifact Promotion)
```

---

## 🚀 Quick Start Guide

### 1. Installation

Clone or open the repository, create a virtual environment, and install dependencies:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run Historical Data Backfill

Backfill 1 year of hourly data for Karachi and populate the Feature Store:

```powershell
python run_pipeline.py --backfill --days 365
```

### 3. Train and Benchmark Models

Train Baseline, Random Forest, Gradient Boosting, and Deep MLP, then elect the champion:

```powershell
python run_pipeline.py --train
```

### 4. Run Model Explainability (SHAP)

Compute feature attribution and generate the importance visualization:

```powershell
python run_pipeline.py --explain
```

*(Or run the full pipeline in one command with `python run_pipeline.py --all`)*

---

## 🖥️ User Interfaces

### 1. Interactive Streamlit Dashboard

Launch the web app:

```powershell
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser:
- **Live Atmosphere Metrics**: Real-time $PM_{2.5}$, $PM_{10}$, $NO_2$, $O_3$, $SO_2$, temperature, and humidity.
- **3-Day Multi-Horizon Forecast**: Day 1, Day 2, and Day 3 predictions with EPA risk badges and health advisories.
- **72-Hour Continuous Trajectory Chart**: Interactive Plotly time-series with colored EPA hazard zones.
- **SHAP Drivers & Local Waterfall**: Factor breakdown for current predictions.
- **Diurnal EDA & Correlation Heatmap**: Historical analysis of pollution trends.
- **City Switcher**: Instant forecasts for Karachi, Lahore, Delhi, Beijing, London, New York, Barcelona, Tokyo, or custom GPS coordinates.

### 2. FastAPI REST Microservice

Launch the API server:

```powershell
python run_pipeline.py --api
```

- Interactive OpenAPI Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)
- Health check: `GET http://localhost:8000/health`
- Predict AQI: `POST http://localhost:8000/predict` with JSON body:
  ```json
  {
    "city": "Karachi"
  }
  ```

---

## 📊 Model Leaderboard

Evaluated on chronological test partition (1,738 out-of-time test samples):

| Model | Overall RMSE | Overall MAE | Overall $R^2$ | RMSE 24h | RMSE 48h | RMSE 72h | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | **8.54** | **6.48** | **0.1087** | **6.14** | **8.98** | **10.02** | 🏆 **Champion** |
| Gradient Boosting | 9.03 | 6.94 | 0.0046 | 5.92 | 9.26 | 11.12 | Candidate |
| Ridge Baseline | 9.11 | 6.79 | -0.0127 | 5.53 | 9.67 | 11.17 | Baseline |
| Deep MLP | 12.69 | 9.73 | -0.9635 | 7.98 | 12.10 | 16.52 | Candidate |

---

## 🔄 Automated CI/CD Workflows

Located under `.github/workflows/`:
- **`feature_pipeline.yml`**: Scheduled on cron (`0 * * * *`) to fetch live sensor streams and update the feature store hourly.
- **`training_pipeline.yml`**: Scheduled daily (`0 2 * * *`) to retrain candidate models, evaluate benchmarks, and refresh champion artifacts in the model registry.

---

## 🧪 Automated Testing

Run the unit test suite:

```powershell
python -m unittest discover tests
```

---

## 📑 Detailed Report

For the full theoretical breakdown, data analysis, and interpretability study, refer to [PROJECT_REPORT.md](PROJECT_REPORT.md).
