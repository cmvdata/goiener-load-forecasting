"""
config.py — Single source of truth for the project.

All paths, dates, hyperparameters, and constants live here. Other scripts
import from this file. Do not hardcode dates or paths in pipeline scripts.

Mode of operation is controlled by the GOIENER_DATA_MODE environment variable:
    full   (default) → reads from data/raw/ (real Zenodo data)
    sample           → reads from data/sample/ (synthetic 50 households)
"""

import os
import sys
from pathlib import Path

# Windows consoles default to cp1252, which cannot encode the unicode arrows
# and Greek letters used throughout the pipeline output. Reconfigure stdout/
# stderr to UTF-8 so prints don't crash on Windows. Safe no-op elsewhere.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass


# ====================================================================
# Project paths
# ====================================================================

PROJECT_ROOT = Path(__file__).parent.parent.resolve()

DATA_MODE = os.environ.get("GOIENER_DATA_MODE", "full").lower()

if DATA_MODE not in {"full", "sample"}:
    raise ValueError(
        f"GOIENER_DATA_MODE must be 'full' or 'sample', got '{DATA_MODE}'"
    )

if DATA_MODE == "sample":
    DATA_DIR = PROJECT_ROOT / "data" / "sample"
else:
    DATA_DIR = PROJECT_ROOT / "data" / "raw"

WEATHER_DIR = PROJECT_ROOT / "data" / "weather"
OUTPUT_DIR = PROJECT_ROOT / "output"
CACHE_DIR = OUTPUT_DIR / "_cache"
LOGS_DIR = PROJECT_ROOT / "logs"
MODELS_DIR = PROJECT_ROOT / "models"

# Ensure all dirs exist
for d in (OUTPUT_DIR, CACHE_DIR, LOGS_DIR, MODELS_DIR,
          MODELS_DIR / "champion", MODELS_DIR / "challengers", WEATHER_DIR):
    d.mkdir(parents=True, exist_ok=True)


# ====================================================================
# Subdirectories of the GoiEner archive after decompression
# ====================================================================

# Note: archives unpack into nested folders like imp-pre/goi4_pre/imp_csv/
# Use rglob('*.csv') in pipeline scripts, not glob.

PRE_DIR = DATA_DIR / "imp-pre"
IN_DIR = DATA_DIR / "imp-in"
POST_DIR = DATA_DIR / "imp-post"
METADATA_PATH = DATA_DIR / "metadata.csv"


# ====================================================================
# Key dates (do not change unless GoiEner data changes)
# ====================================================================

# COVID-19 lockdown in Spain
LOCKDOWN_START = "2020-03-14"
LOCKDOWN_END = "2020-06-21"

# Regional curfews
CURFEW_START = "2020-10-25"
CURFEW_END = "2021-05-09"

# Mandatory 2.0TD tariff reform
TARIFF_REFORM = "2021-06-01"

# Sample boundaries for forecasting validation
# We avoid the early lockdown shock by starting validation in mid-2020.
VALIDATION_START = "2020-06-22"
VALIDATION_END = "2022-06-30"

# Random seed (for any stochastic step)
SEED = 20210601


# ====================================================================
# Forecasting hyperparameters
# ====================================================================

# Forecast horizon (hours)
FORECAST_HORIZON_H = 168  # 7 days

# Walk-forward validation
TRAIN_WINDOW_DAYS = 365  # rolling window size
WALK_STEP_DAYS = 7       # step between successive train/predict cycles

# Quantiles for probabilistic forecasting
QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95]

# Coverage rate target intervals. Native to QUANTILES = [0.05, 0.25, 0.5, 0.75,
# 0.95] — no interpolation needed. Adding 80% / 95% intervals would require
# extending QUANTILES with 0.10 / 0.90 and 0.025 / 0.975 respectively (more
# models per fold). The 50% / 90% pair is the standard probabilistic-forecast
# diagnostic and is sufficient for portfolio-level reporting.
NOMINAL_INTERVALS = {
    "50%": (0.25, 0.75),
    "90%": (0.05, 0.95),
}

# Number of segments for k-means clustering. Chosen by k-sweep diagnostic
# (see output/02_ksweep_diagnostic/): silhouette peaks at k=3 (0.168) and
# the inertia elbow inflects at k=3. The cluster structure overall is weak
# (silhouette < 0.20 for all k ∈ {2..8}) — residential load shape is a
# continuum, not a small set of discrete archetypes — so k is an
# operational choice. k=3 is the metric-supported one.
#
# Pipeline 2 (segmentation for tariff/behavioral characterization) uses
# this. Pipeline 1 (forecasting) does NOT — see FORECAST_LEVELS below.
N_SEGMENTS = 3

# Forecasting levels. Pipeline 1 only forecasts the portfolio aggregate.
# Per-segment forecasting was evaluated and dropped because the segment-
# level signal is too noisy in the validation window (small clusters,
# noisy pinball at extreme quantiles) and the recursive single-step
# architecture cannot exploit it. Segments live in Pipeline 2 only.
FORECAST_LEVELS = ["portfolio"]

# LightGBM hyperparameters (fixed; not tuned per quantile by design)
LGB_PARAMS = {
    "objective": "quantile",
    "metric": "quantile",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 50,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.9,
    "bagging_freq": 5,
    "num_iterations": 2500,   # ceiling for early stopping
    "verbose": -1,
    "seed": SEED,
}

# Trailing days reserved for the early-stopping validation set. The split
# is strictly temporal — never random — so the validation set is always
# in the future of the training subset.
LGB_VALID_DAYS = 30

# Patience for lgb.early_stopping(...). Raised from 50 to 100 because
# the pinball loss on extreme quantiles (q=0.05, q=0.95) over a 30-day
# validation window is dominated by a handful of tail observations and
# 50 rounds of patience produced visibly premature stops in the previous
# walk-forward run (227 of 2,550 fits stopped <100 iter, 72% of those at
# extreme quantiles). 100 rounds is the standard recommendation for
# noisy validation signals at lr=0.05.
LGB_EARLY_STOPPING_ROUNDS = 100


# ====================================================================
# Champion/challenger promotion logic
# ====================================================================

# Holdout window for champion vs challenger comparison
COMPARISON_WINDOW_DAYS = 30

# Minimum MAPE improvement required to promote challenger to champion
PROMOTION_THRESHOLD_MAPE_PP = 0.2  # percentage points

# Minimum coverage rate required (calibration must not degrade).
# Names match NOMINAL_INTERVALS so each threshold gates exactly the
# interval evaluated by the model.
MIN_COVERAGE_RATE_50 = 0.45  # 50% nominal interval should cover at least 45%
MIN_COVERAGE_RATE_90 = 0.85  # 90% nominal interval should cover at least 85%


# ====================================================================
# Weather configuration (Open-Meteo)
# ====================================================================

# Three points covering the geographic distribution of GoiEner customers
WEATHER_LOCATIONS = {
    "bilbao":   {"lat": 43.263, "lon": -2.935, "weight": 0.725},  # Basque Country, 72.5%
    "pamplona": {"lat": 42.812, "lon": -1.645, "weight": 0.138},  # Navarre, 13.8%
    "madrid":   {"lat": 40.417, "lon": -3.704, "weight": 0.137},  # Madrid + rest, 13.7%
}

# Open-Meteo API endpoint (no auth required)
OPENMETEO_BASE = "https://archive-api.open-meteo.com/v1/archive"

WEATHER_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "wind_speed_10m",
]


# ====================================================================
# Filtering thresholds
# ====================================================================

# Minimum hours of valid (non-imputed) data per household to include
MIN_VALID_HOURS_PER_HH = 24 * 30 * 12  # ≥1 year of valid data

# Maximum imputation rate per household (skip if too much was imputed)
MAX_IMPUTATION_RATE = 0.20  # 20%


# ====================================================================
# Print config summary on import (helpful for debugging mode confusion)
# ====================================================================

def _print_summary():
    print(f"[config] DATA_MODE   = {DATA_MODE}")
    print(f"[config] DATA_DIR    = {DATA_DIR}")
    print(f"[config] OUTPUT_DIR  = {OUTPUT_DIR}")
    print(f"[config] SEED        = {SEED}")


if __name__ == "__main__":
    _print_summary()
