# CLAUDE.md — Context for Claude Code

This document gives Claude Code agents the context needed to navigate, run, and extend this project.

## Project goal

Short-term load forecasting (24h to 168h horizon) for a residential electricity portfolio of 16,764 eligible Spanish households (10,531 of which retain a usable pre-2020 daily profile and feed Pipeline 2), using the openly available GoiEner smart meter dataset.

The pipeline is designed to mirror the operational pattern of supply-side forecasting teams in real utilities. It is **portfolio-level**, not production-level: it runs offline on a laptop and is meant for portfolio demonstration, not for serving live forecasts.

The repository ships **two architecturally independent products** built on the same data ingestion stage:

- **Pipeline 1 — Portfolio forecasting** (the headline product): recursive single-step LightGBM with five quantiles, early stopping with refit-on-full, conformal calibration. Drives nightly 168-hour forecasts.
- **Pipeline 2 — Behavioural segmentation** (a complementary commercial toolkit): k-means with k=3 on standardized daily load profiles, three operational archetypes for tariff design and demand-response targeting. **Pipeline 2 does not feed Pipeline 1.**

Headline metric (full-mode walk-forward, 105 weekly folds): MAPE 4.88% at h≤24, +29.6% lift over weekly persistence. Conformal post-hoc calibration passes the gate ([0.45, 0.55] cov_50, [0.85, 0.95] cov_90) on all three pool buckets. Full results, derivations and limitations are in `paper/load_forecasting_paper.pdf`.

## Data

- **Source**: Zenodo (DOI 10.5281/zenodo.7362094), Quesada et al. (2024)
- **Scope**: 25,559 supply points (16,764 residential after CNAE filter), hourly readings, Nov 2014 – Jun 2022. Of those 16,764 residential meters, 10,531 retain a usable pre-2020 daily profile after the data-quality filter and feed Pipeline 2.
- **Pre-segmented archives**: `imp-pre.tzst`, `imp-in.tzst`, `imp-post.tzst` (~2 GB total)
- **Local layout after extraction**: archives unpack into nested folders like `imp-pre/goi4_pre/imp_csv/<hash>.csv`. The pipeline scripts use `rglob` to find CSVs at any depth.

The synthetic sample in `data/sample/` mirrors the real data structure but contains only 50 households over ~3.5 years. It exists only to validate that the pipeline runs end-to-end. Statistical results from the sample are not meaningful.

## Stack

- **Python 3.10–3.12** (3.13+ not supported: the `scikit-learn<1.7` pin lacks wheels for newer Pythons)
- `pandas`, `numpy`, `pyarrow` for data wrangling
- `statsmodels` for SARIMAX
- `lightgbm` for gradient boosting with quantile regression
- `scikit-learn` for k-means clustering and preprocessing
- `streamlit` + `plotly` for the dashboard
- `structlog` for structured logging

Full list pinned in `requirements.txt`. Note: `hierarchicalforecast` and `utilsforecast` were removed as part of the May 2026 reframe; they powered the per-segment MinT-OLS reconciliation that no longer applies.

## Pipeline workflow

The pipeline is divided into numbered scripts in `code/`. Each script reads from `output/` and writes to `output/`, so they can be re-run independently after parameter changes.

1. `00_download_weather.py` — fetches hourly weather from Open-Meteo for Bilbao, Pamplona, and Madrid
2. `01_build_panel.py` — decompresses Zenodo archives, builds household-hour panel, aggregates to portfolio
3. `02_segment_households.py` — Pipeline 2: k-means with k=3 on standardized daily profiles + k-sweep diagnostic + operational characterization (CLI flags: `--ksweep`, `--recharacterize`)
4. `03_feature_engineering.py` — lags (causal `shift(1).rolling()`), calendar, weather, tariff regime indicator
5. `04_train_baseline.py` — persistence + SARIMAX baselines
6. `05_train_lightgbm.py` — Pipeline 1 champion: portfolio-only LightGBM, 5 quantiles, **early stopping with refit-on-full**
7. `06_walk_forward_validation.py` — 105-fold rolling-window evaluation (portfolio only)
8. `06b_conformal_calibration.py` — split conformal post-hoc adjustments
9. `08_run_daily_batch.py` — generates next 168h of portfolio forecasts (the "production" pattern)
10. `09_evaluate_challenger.py` — champion/challenger promotion logic
11. `10_summarize_results.py` — final tables, figures, paper headline metrics, Pipeline 2 narrative

`code/generate_sample.py` creates the synthetic sample for testing the pipeline without the full download.

`code/_verify_no_leakage.py` is an audit helper that checks the temporal-split invariants across all 105 folds. Not part of the production flow; useful when modifying `fit_with_early_stopping`.

`code/_archive/07_hierarchical_reconcile.py` is preserved for audit completeness. MinT-OLS / BottomUp were evaluated and dropped after the May 2026 reframe to portfolio-only forecasting; the `_archive/README.md` records the reasoning.

## Modes of operation

The pipeline reads `GOIENER_DATA_MODE` from the environment:

- **`full`** (default): uses real Zenodo data in `data/raw/`
- **`sample`**: uses synthetic data in `data/sample/`

Always test changes in sample mode first to avoid 30-minute panel rebuilds.

## Key design choices

These are decisions worth understanding before changing things:

- **Walk-forward validation, not random k-fold**: time-series leakage is the main pitfall in this domain. The validation respects time strictly. The temporal-split invariant (`valid_min > train_max`) is asserted at every fit and verified across all 105 folds by `code/_verify_no_leakage.py`.

- **Quantile regression (LightGBM), not point forecasts**: probabilistic outputs are mandatory for any operational use of the forecasts (hedging, imbalance management).

- **Early stopping with refit-on-full Phase 2**: each booster fit runs early stopping on a 30-day temporal hold-out (patience=100, ceiling=2,500 iterations) to determine `best_iteration` $K$, then refits a fresh booster on the entire 365-day window for exactly $K$ rounds. This recovers the validation-window data while preserving the iteration count selected by honest validation. See `fit_with_early_stopping` in `code/05_train_lightgbm.py`.

- **Portfolio-only forecasting, not segment-then-reconcile**: per-segment forecasting + hierarchical reconciliation (MinT-OLS) was implemented and evaluated. The recursive single-step architecture cannot exploit segment heterogeneity at long horizons (segment trajectories drift toward the mean over 168 steps), and validation pinball loss on extreme quantiles for small segments was dominated by tail noise on a 30-day window. The hierarchical reconciliation step yielded marginal improvements at h=25-72 but degraded at h≤24 and h=168, and was therefore dropped. The archived implementation lives in `code/_archive/`.

- **Champion/challenger, not always-deploy**: a new model is promoted only if it beats the current one on the same recent holdout window by a meaningful margin (≥0.2 pp MAPE) and does not degrade calibration (cov_50 ≥ 0.45, cov_90 ≥ 0.85).

- **Realized weather in evaluation**: the project uses realized weather (Open-Meteo historical archive) rather than forecast weather. This caveat is documented prominently and would need to be addressed in any real production setting. Weather features account for ~20.5% of total LightGBM gain in the trained champion suite.

## File structure conventions

- `code/config.py` is the single source of truth. All paths, dates, hyperparameters, and constants live here. Don't hardcode dates or paths in other files. Key constants for Pipeline 1: `LGB_PARAMS["num_iterations"]=2500` (ceiling), `LGB_VALID_DAYS=30` (temporal split), `LGB_EARLY_STOPPING_ROUNDS=100` (patience), `FORECAST_LEVELS=["portfolio"]`.
- `output/_cache/` holds large parquets that take long to generate (e.g., the household-hour panel). It is gitignored. Other things in `output/` are committed because Streamlit Cloud needs them to render results.
- `output/_archive*/` and `models/_archive*/` hold pre-reframe artefacts. Gitignored. Each archive directory has a `README.md` explaining what's there and why.
- `models/champion/` holds the currently deployed model artefacts (post-reframe: 5 portfolio boosters). `models/challengers/` holds evaluated but not promoted models, with a JSON log of why each was rejected.
- `paper/` holds the LaTeX sources for `paper/load_forecasting_paper.pdf`. The PDF is regenerable from sources via `latexmk -pdf`. Figures and tables are sourced from CSVs under `output/`; see `paper/README.md` for the source map.
- `logs/` holds structured JSON logs from the daily batch runs and challenger evaluations. Gitignored.

## What to do when starting fresh

If you are Claude Code and this is the first time you are working on this project, the recommended order is:

1. Read this file, `README.md`, and skim `paper/load_forecasting_paper.pdf`
2. Check `output/_cache/` to see if a panel parquet already exists (saves ~30 min of rebuild)
3. Run `python code/01_build_panel.py` in sample mode first to confirm the pipeline works
4. Run the rest of the numbered scripts in order
5. Verify the Streamlit app launches with `streamlit run app/streamlit_app.py`

## What not to do

- Do not modify `README.md` without explicit user approval. It is the public face of the repository.
- Do not commit anything in `data/raw/`, `data/weather/`, `output/_cache/`, `output/_archive*/`, or `models/_archive*/`. They are large and gitignored.
- Do not introduce deep learning models (LSTM, TFT) without discussion. They are explicitly out of scope.
- Do not change the Python version constraint (3.10–3.12) without explicit user approval.
- Do not re-introduce per-segment forecasting or hierarchical reconciliation without re-evaluating the trade-offs that led to their removal in the May 2026 reframe (see `code/_archive/README.md`).

## Known limitations

- LightGBM beats persistence at h≤72 (lift +12% to +30%) but underperforms at h=72 (-25%) and h=168 (-32%) due to recursive error compounding. Direct multi-horizon architecture would address this; not implemented.
- 8.6% of walk-forward fits stop <100 iterations due to pinball loss noise on extreme quantiles (q=0.05, q=0.95). Patience=100 reduces but does not eliminate this.
- 3.1% of fits hit the 2,500-iteration ceiling. Margin is tight; raising to 3,000-3,500 would clear these at the cost of compute.
- Pipeline 2 silhouette<0.20 across all k tested: archetypes are operational tags for commercial use, not statistically discrete clusters.
- Realized weather features (Open-Meteo historical) used in training. Production deployment would need forecast weather, with corresponding accuracy degradation.

## Common operations

```bash
# Activate venv (Windows)
.venv\Scripts\activate

# Activate venv (macOS/Linux)
source .venv/bin/activate

# Run a script in sample mode
GOIENER_DATA_MODE=sample python code/05_train_lightgbm.py

# Run a script in full mode (default)
python code/05_train_lightgbm.py

# Run Pipeline 2 k-sweep diagnostic (does not overwrite cluster outputs)
python code/02_segment_households.py --ksweep

# Regenerate Pipeline 2 characterization without rebuilding profiles
python code/02_segment_households.py --recharacterize

# Audit the temporal split across all 105 walk-forward folds
python code/_verify_no_leakage.py

# Launch dashboard
streamlit run app/streamlit_app.py

# Generate synthetic sample
python code/generate_sample.py

# Compile the paper (from paper/)
cd paper && latexmk -pdf load_forecasting_paper.tex
```
