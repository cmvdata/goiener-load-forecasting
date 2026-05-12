# WORK_PLAN.md — Implementation guide for Claude Code

This document is the implementation roadmap for the project. The user has chosen
an **incremental approach**: rather than receiving all 11 scripts as boilerplate
upfront, Claude Code implements them one at a time, validating each against the
synthetic sample before moving on.

This avoids the bug-fixing storm that comes from delivering 1000 lines of
generated code at once and discovering five subtle issues at runtime.

## Current state of the repository

These files are **already implemented and tested**:

- `code/config.py` — central configuration (paths, dates, hyperparameters)
- `code/01_build_panel.py` — decompresses GoiEner archives and builds the portfolio-hour panel
- `code/generate_sample.py` — generates a synthetic 50-household sample with 4 behavioral segments

These files are **stubs / not yet implemented**:

- `code/00_download_goiener.py`     — fetch the Zenodo archives with resume + retry
- `code/00_download_weather.py`      — fetch hourly weather from Open-Meteo
- `code/02_segment_households.py`    — k-means clustering on daily load profiles
- `code/03_feature_engineering.py`   — lags, calendar, weather, interactions
- `code/04_train_baseline.py`        — persistence + SARIMAX
- `code/05_train_lightgbm.py`        — gradient boosting with quantile regression
- `code/06_walk_forward_validation.py` — rolling-window evaluation
- `code/07_hierarchical_reconcile.py` — bottom-up + MinT
- `code/08_run_daily_batch.py`        — nightly forecast generation
- `code/09_evaluate_challenger.py`    — champion/challenger promotion logic
- `code/10_summarize_results.py`      — final tables and figures

The Streamlit app (`app/streamlit_app.py` and `app/pages/`) is also pending.

## How to work through this plan

For each pending script:

1. Read its specification in this document
2. Implement the script in `code/`
3. Run it in sample mode: `GOIENER_DATA_MODE=sample python code/XX_name.py`
4. Verify the expected outputs are created and look reasonable
5. Move to the next script

Do not implement all scripts at once. Validate each before proceeding.

If you discover a bug or missing dependency in the existing implemented scripts
(`config.py`, `01_build_panel.py`, `generate_sample.py`), fix it. They are
not sacred — they are just the parts that were pre-baked.

If you find a script's specification in this document is unclear or
underspecified, ask the user before guessing. Better to clarify than to
implement the wrong thing.

---

## Script 00a: Download GoiEner archives

**File**: `code/00_download_goiener.py`

**Purpose**: Download the four files from Zenodo with resume capability and retry logic.

**Why it matters**: the Zenodo download of `imp-in.tzst` (530 MB) and others is fragile on residential connections. A naive `requests.get` will fail mid-download. The script must use HTTP Range headers to resume after a connection drop and retry with exponential backoff.

**Files to download** from `https://zenodo.org/records/7362094/files/`:
- `metadata.csv` (~5.6 MB)
- `imp-pre.tzst` (~792 MB)
- `imp-in.tzst` (~530 MB)
- `imp-post.tzst` (~510 MB)

Save to `data/raw/`. Skip files that are already complete (size matches HEAD response exactly).

**Behavior**:
- Use HEAD request to get canonical file size
- If local file exists and matches canonical size exactly, skip
- If local file is shorter, send `Range: bytes=<n>-` request to resume
- Retry up to 6 times with exponential backoff (5s, 10s, 20s, 40s, 80s, 160s)
- Handle HTTP 416 (range not satisfiable) as success (file is already complete)
- Use `tqdm` for progress bar

**No CLI args needed for v1**. Just runs and downloads.

**Output**: files in `data/raw/`. Print summary of sizes when done.

---

## Script 00b: Download weather data

**File**: `code/00_download_weather.py`

**Purpose**: Fetch hourly historical weather (temperature, humidity, precipitation, wind speed) from the Open-Meteo Archive API for three locations representing the GoiEner customer base.

**API endpoint**: `https://archive-api.open-meteo.com/v1/archive`

**Locations** (already in `config.WEATHER_LOCATIONS`):
- Bilbao (Basque Country, 72.5%): lat 43.263, lon -2.935
- Pamplona (Navarre, 13.8%): lat 42.812, lon -1.645
- Madrid (rest, 13.7%): lat 40.417, lon -3.704

**Date range**: from `2018-01-01` to today (or to `2022-06-30` to match GoiEner end).

**Variables** (already in `config.WEATHER_VARIABLES`):
- temperature_2m
- relative_humidity_2m
- precipitation
- wind_speed_10m

**Sample API call**:
```
https://archive-api.open-meteo.com/v1/archive
  ?latitude=43.263
  &longitude=-2.935
  &start_date=2018-01-01
  &end_date=2022-06-30
  &hourly=temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m
  &timezone=Europe%2FMadrid
```

**Output**: one parquet file per location in `data/weather/`:
- `data/weather/bilbao.parquet`
- `data/weather/pamplona.parquet`
- `data/weather/madrid.parquet`

Each parquet has columns: `timestamp`, `temperature_2m`, `relative_humidity_2m`, `precipitation`, `wind_speed_10m`.

**Sample mode**: skip the API call entirely. Generate synthetic weather using the same logic as `generate_sample.py:synthetic_temperature()` for the three locations and write to the same output paths. This ensures `03_feature_engineering.py` works in sample mode.

**Caching**: if all three parquet files already exist with reasonable size (>1 MB each), skip re-downloading.

---

## Script 02: Segment households by load profile

**File**: `code/02_segment_households.py`

**Purpose**: Cluster eligible households into `N_SEGMENTS` (default 4) based on their average daily load shape. The clusters are interpretable archetypes: high-evening, flat-profile, morning-skewed, low-consumption.

**Crucial design rule**: clustering uses **only pre-2020 data** to avoid leakage into validation periods. If a household has no pre-2020 data, exclude it from segmentation.

**Steps**:

1. Read the eligible household list (`output/_cache/eligible_households.csv`)
2. For each household, load its raw hourly series (using `01_build_panel.py:load_household_series`)
3. Filter to pre-2020 timestamps (before `2020-01-01`)
4. Compute the **average daily profile**: 24-element vector of mean kWh per hour-of-day
5. Standardize each profile (subtract mean, divide by std) so clustering is by **shape**, not by **level**
6. Run k-means with `n_clusters=4`, `random_state=SEED`
7. Assign each household its cluster label
8. Save: `output/02_household_segments.csv` with columns `[user_hash, segment, total_pre_kwh]`
9. Save: `output/02_segment_centroids.csv` with the standardized centroid profiles
10. Plot: `output/02_segment_profiles.png` showing the 4 archetypes overlaid

**Sample mode**: same logic. The synthetic data was generated with 4 archetypes already, so clustering should recover them.

**Validation expectation**: in sample mode, the recovered cluster labels should correlate with the `_segment_truth` column in `data/sample/metadata.csv` (it's there for testing). Adjusted Rand Index > 0.5 means we recovered the true segmentation.

---

## Script 03: Feature engineering

**File**: `code/03_feature_engineering.py`

**Purpose**: Build the feature matrix used by all downstream forecasting models. Features must be **strictly causal**: feature value at time t can only use information available before t.

**Inputs**:
- `output/_cache/portfolio_hourly.parquet` (portfolio-level kWh)
- Three weather parquets in `data/weather/`
- `output/02_household_segments.csv`

**Features to construct**:

**Calendar features** (no leakage risk):
- `hour_of_day` (0–23)
- `day_of_week` (0–6)
- `is_weekend` (binary)
- `is_holiday` (binary; use `holidays.Spain()` for national holidays)
- `month` (1–12)
- `day_of_year` (1–366)
- Sine/cosine encodings of `hour_of_day` and `day_of_year` for cyclic features

**Lag features** (causal):
- `kwh_lag_1` (one hour ago)
- `kwh_lag_24` (one day ago at same hour)
- `kwh_lag_168` (one week ago at same hour)
- `kwh_rolling_mean_24` (mean of t-1 to t-24)
- `kwh_rolling_mean_168` (mean of t-1 to t-168)

**Weather features** (use realized weather; document in README that production would use forecasts):
- `temperature_weighted` — weighted average of three locations using `config.WEATHER_LOCATIONS[loc]['weight']`
- `humidity_weighted`
- `precipitation_weighted`
- `wind_weighted`
- `temp_sq` — squared temperature (captures U-shaped heating/cooling response)
- `heating_demand` = max(12 - temperature, 0)
- `cooling_demand` = max(temperature - 24, 0)

**Tariff regime feature**:
- `post_tariff_reform` = 1 if timestamp ≥ `config.TARIFF_REFORM`, else 0

**Output**: `output/_cache/features_portfolio.parquet`

Columns: `timestamp`, `kwh_total`, all features above.

**Same logic at segment level**:
- Also produce `output/_cache/features_segment_<N>.parquet` for each segment N.
- Each segment's `kwh_total` is the sum of kWh of households in that segment.

---

## Script 04: Baseline models

**File**: `code/04_train_baseline.py`

**Purpose**: Train two baseline models against which LightGBM is compared.

**Model A — Persistence**:
- "Forecast for hour t = actual value at hour t-24-h_offset"
- Where h_offset depends on horizon (predict 24h → use t-24; predict 168h → use t-168)
- This is a no-model baseline. If LightGBM doesn't beat persistence, something is broken.
- Implementation: a function that takes the panel and a horizon, returns predictions.
- No training required.

**Model B — SARIMAX**:
- Use `statsmodels.tsa.statespace.sarimax.SARIMAX`
- Order: (1, 0, 1)
- Seasonal order: (1, 0, 1, 24) to capture daily seasonality
- Exogenous: temperature_weighted, is_weekend, is_holiday
- Train on a rolling window (per `config.TRAIN_WINDOW_DAYS`)
- This will be slow; that's expected for SARIMAX. Use it sparingly.

**Output**: a Python module that exposes:
- `persistence_forecast(panel, horizon_h) -> array`
- `sarimax_forecast(panel_train, panel_predict, horizon_h) -> array`

These functions are called by `06_walk_forward_validation.py`.

This script doesn't produce CSV output by itself; it's a library of functions.

---

## Script 05: LightGBM with quantile regression

**File**: `code/05_train_lightgbm.py`

**Purpose**: Train LightGBM models for probabilistic forecasting. Five separate models, one per quantile (5%, 25%, 50%, 75%, 95%).

**Library**: `lightgbm`. Use the `objective='quantile'` and `alpha=q` parameters.

**Hyperparameters**: defined in `config.LGB_PARAMS`. Do not tune per quantile; use the same params for all five.

**Training**:
- Input: feature matrix from `03_feature_engineering.py`
- Target: `kwh_total` at horizon h ahead (so the script must produce a separate model per (segment, horizon))
- For v1, focus on horizons 24h and 168h. Add more later if time permits.

**Output**:
- `models/champion/lgb_segment_<N>_horizon_<H>_q<Q>.txt` for each combination
- A summary CSV `output/05_lgb_training_summary.csv` listing each model trained, training time, and final iteration

This script produces the **trained models**. Evaluation is done by `06_walk_forward_validation.py`.

---

## Script 06: Walk-forward validation

**File**: `code/06_walk_forward_validation.py`

**Purpose**: Evaluate all models (persistence, SARIMAX, LightGBM) using rolling-window walk-forward validation. This is the heart of the validation methodology.

**Algorithm**:

```python
start = VALIDATION_START
end = VALIDATION_END
step = WALK_STEP_DAYS (default 7)

results = []

while start + train_window + horizon < end:
    train_end = start + TRAIN_WINDOW_DAYS
    pred_start = train_end
    pred_end = train_end + FORECAST_HORIZON_H hours

    train_data = panel[panel.timestamp < train_end]
    pred_data = panel[(panel.timestamp >= pred_start) & (panel.timestamp < pred_end)]

    # Persistence baseline
    pers_pred = persistence_forecast(train_data, FORECAST_HORIZON_H)
    pers_error = compute_metrics(pers_pred, pred_data.actual)
    results.append({"model": "persistence", "fold_start": start, ...metrics})

    # SARIMAX
    sarimax_pred = sarimax_forecast(train_data, pred_data.features, FORECAST_HORIZON_H)
    results.append({"model": "sarimax", ...metrics})

    # LightGBM (predicts all 5 quantiles)
    lgb_preds = {q: lgb_models[q].predict(pred_data.features) for q in QUANTILES}
    results.append({"model": "lightgbm", quantile: 0.5, ...metrics})

    start += step  # advance one week
```

**Metrics to compute** per fold and overall:
- MAPE
- RMSE
- sMAPE
- Pinball loss per quantile (LightGBM only)
- Coverage rate at 50%, 80%, 95% intervals (LightGBM only)

**Output**:
- `output/06_walk_forward_results.csv` — long-format with one row per (model, fold, horizon, metric)
- `output/06_metrics_summary.csv` — pivoted table for the README "headline results"
- `output/06_calibration_diagnostic.png` — coverage rate vs nominal interval, by horizon

**Critical**: no leakage. The training window must end strictly before the prediction window starts. Feature normalization (if any) must be fit on training only.

---

## Script 07: Hierarchical reconciliation

**File**: `code/07_hierarchical_reconcile.py`

**Purpose**: Reconcile portfolio-level and segment-level forecasts so they sum coherently.

**Library**: `hierarchicalforecast` from Nixtla.

**Hierarchy**:
- Top: portfolio total
- Middle: 4 segments (from script 02)
- Bottom: each segment's forecast

**Methods to apply**:
- Bottom-up (sum segment forecasts)
- MinT trace minimization with OLS-weighted residuals

**Steps**:
1. Load LightGBM forecasts for each segment from `06_walk_forward_results.csv`
2. Build the hierarchy summing matrix
3. Apply `BottomUp()` and `MinTrace(method='ols')` reconcilers
4. Compute MAPE for each method against the realized portfolio total
5. Save: `output/07_reconciled_forecasts.parquet`
6. Save: `output/07_reconciliation_comparison.csv` with MAPE per method

---

## Script 08: Daily batch forecast generation

**File**: `code/08_run_daily_batch.py`

**Purpose**: Simulate the production pattern. Generate forecasts for the next 168 hours and save them to a timestamped parquet file. This is what would run on cron at 03:00 every night in a real deployment.

**Steps**:
1. Load the most recent panel data (last 365 days for retraining context)
2. Use the current champion models (`models/champion/`) to predict the next 168 hours
3. Apply MinT reconciliation
4. Save to `output/forecasts/forecasts_<YYYY-MM-DD>.parquet` with columns:
   - timestamp
   - segment (or 'portfolio')
   - q05, q25, q50, q75, q95
   - generated_at (timestamp when this forecast was made)

**Logging**: append a structured JSON entry to `logs/batch_runs.jsonl`:
```json
{"event": "batch_complete", "timestamp": "2026-04-30T03:00:00", "horizon_h": 168, "n_segments": 4, "champion_version": "v3"}
```

**This script is meant to be re-run daily**. It is idempotent: re-running it overwrites the file for that date.

---

## Script 09: Champion/challenger evaluation

**File**: `code/09_evaluate_challenger.py`

**Purpose**: Decide whether a newly-trained model should replace the current champion.

**Steps**:
1. Identify the current champion (latest model in `models/champion/`)
2. Train a fresh challenger using the same architecture but on more recent data
3. Evaluate both on the same most-recent 30-day holdout (defined in `config.COMPARISON_WINDOW_DAYS`)
4. Apply promotion criteria:
   - Challenger MAPE must improve by ≥ `config.PROMOTION_THRESHOLD_MAPE_PP` points
   - Challenger coverage at 80% must be ≥ `config.MIN_COVERAGE_RATE_80`
   - Challenger coverage at 95% must be ≥ `config.MIN_COVERAGE_RATE_95`
5. If all criteria pass: move challenger to `models/champion/`, archive previous champion to `models/challengers/<date>_demoted/`
6. If any criterion fails: leave challenger in `models/challengers/<date>_rejected/`
7. Log the decision to `logs/promotions.jsonl` with full reasoning

**Output**: `output/09_promotion_decision.json` summarizing the most recent decision (for the dashboard).

---

## Script 10: Summarize and produce final artifacts

**File**: `code/10_summarize_results.py`

**Purpose**: Generate the final tables and figures used by the README and the Streamlit dashboard.

**Outputs**:
- `output/10_headline_metrics.csv` — for the README's "Headline results" table
- `output/10_feature_importance_grouped.csv` — feature importance aggregated by category (calendar, lag, weather, segment)
- `output/10_calibration_curve.png`
- `output/10_economic_interpretation.md` — auto-generated narrative explaining the top features and their economic meaning
- `output/10_results_narrative.md` — full narrative for the README

This script consolidates results. It does not retrain anything.

---

## Streamlit app

After all 10 scripts work end-to-end, implement the dashboard.

The structure is in `app/streamlit_app.py` (overview/home) and `app/pages/`:
- `1_Forecasts.py` — interactive forecast viewer with intervals
- `2_Validation.py` — walk-forward results, model comparison
- `3_Segments.py` — cluster profiles and per-segment metrics
- `4_Calibration.py` — coverage and pinball loss diagnostics
- `5_Methodology.py` — assumptions, design choices, limitations

The `streamlit_app.py` should mirror the design of the companion repo's app. Each page reads from `output/` files. No models loaded in memory at app start (Streamlit Cloud has 1 GB limit).

A "View source on GitHub" link in the sidebar pointing to:
`https://github.com/cmvdata/goiener-load-forecasting`

---

## Final notes

- **When in doubt, ask the user**. Better to clarify than to guess.
- **Validate sample mode after every script**. If something breaks in sample, don't move on.
- **Don't introduce libraries not in `requirements.txt` without asking**.
- **Don't change `config.py` constants without asking** (especially dates and paths).
- **Don't commit anything yet**. Git operations are coordinated separately.
