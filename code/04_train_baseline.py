"""
04_train_baseline.py — Baseline forecasters (persistence + SARIMAX).

This file is consumed as a *library* by 06_walk_forward_validation.py and
08_run_daily_batch.py. Run as a script only to smoke-test that both
forecasters return sane shapes on a small sample.

Persistence:
  Naive seasonal forecast — predict y(t) using y(t - period). With weekly
  seasonality (period=168) this is the strongest naive baseline for
  residential load and the bar a real model must clear.

SARIMAX:
  (1,0,1) x (1,0,1)_24 with daily seasonality and a small exogenous block
  (temperature_weighted, is_weekend, is_holiday). One-shot fit per fold,
  closed-form forecast over the prediction horizon. Slow — used only at
  the cadence of the walk-forward step.
"""

from __future__ import annotations

import warnings
from typing import Iterable

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

import config as C


# ====================================================================
# Persistence
# ====================================================================

def persistence_forecast(history: pd.DataFrame, predict_timestamps: Iterable,
                         seasonality_h: int = 168) -> pd.Series:
    """Seasonal persistence: y_pred(t) = y_history(t - seasonality_h).

    Args:
      history: DataFrame with columns ['timestamp', 'kwh_total'] strictly
               BEFORE the prediction window.
      predict_timestamps: iterable of timestamps to forecast.
      seasonality_h: lag in hours. 24 = daily, 168 = weekly. Default 168
                     because residential load has a strong weekly pattern.

    Returns:
      pd.Series indexed by predict_timestamps. NaN where the lookback
      timestamp is not present in history.
    """
    h = pd.Series(history["kwh_total"].values,
                  index=pd.to_datetime(history["timestamp"]))
    out_idx = pd.DatetimeIndex(predict_timestamps)
    lookback = out_idx - pd.Timedelta(hours=seasonality_h)
    values = h.reindex(lookback).values
    return pd.Series(values, index=out_idx, name="persistence")


# ====================================================================
# SARIMAX
# ====================================================================

SARIMAX_ORDER = (1, 0, 1)
SARIMAX_SEASONAL = (1, 0, 1, 24)
SARIMAX_EXOG = ["temperature_weighted", "is_weekend", "is_holiday"]


def _prepare_endog_exog(features: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    df = features.dropna(subset=["kwh_total"] + SARIMAX_EXOG).copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").set_index("timestamp")
    df.index.freq = pd.infer_freq(df.index)
    endog = df["kwh_total"].astype(float)
    exog = df[SARIMAX_EXOG].astype(float)
    return endog, exog


def sarimax_forecast(train_features: pd.DataFrame,
                     predict_features: pd.DataFrame) -> pd.Series:
    """Fit SARIMAX on train_features, forecast across predict_features.

    Both inputs must have at minimum: timestamp, kwh_total, and the
    SARIMAX_EXOG columns. predict_features['kwh_total'] is allowed to be
    NaN — only its exog block and timestamps are used.

    Returns a pd.Series indexed by predict_features['timestamp'].
    """
    endog, exog = _prepare_endog_exog(train_features)

    pred = predict_features.copy()
    pred["timestamp"] = pd.to_datetime(pred["timestamp"])
    pred = pred.sort_values("timestamp").set_index("timestamp")
    exog_pred = pred[SARIMAX_EXOG].astype(float)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SARIMAX(
            endog,
            exog=exog,
            order=SARIMAX_ORDER,
            seasonal_order=SARIMAX_SEASONAL,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        fit = model.fit(disp=False, maxiter=50)
        pred_obj = fit.get_forecast(steps=len(exog_pred), exog=exog_pred)

    out = pred_obj.predicted_mean
    out.index = exog_pred.index
    out.name = "sarimax"
    return out


# ====================================================================
# Smoke test (run as script)
# ====================================================================

def main():
    feat_path = C.CACHE_DIR / "features_portfolio.parquet"
    if not feat_path.exists():
        raise FileNotFoundError(
            f"Run code/03_feature_engineering.py first; missing {feat_path}"
        )

    feats = pd.read_parquet(feat_path).dropna(subset=["kwh_total"]).copy()
    feats["timestamp"] = pd.to_datetime(feats["timestamp"])

    # Use a 90-day train window and 24h forecast horizon for the smoke test
    feats = feats.sort_values("timestamp").reset_index(drop=True)
    cutoff_idx = len(feats) - 24
    train = feats.iloc[max(0, cutoff_idx - 24 * 90):cutoff_idx]
    predict = feats.iloc[cutoff_idx:]

    print(f"Smoke test: train rows = {len(train):,}, predict rows = {len(predict):,}")

    pers = persistence_forecast(
        history=train[["timestamp", "kwh_total"]],
        predict_timestamps=predict["timestamp"].tolist(),
        seasonality_h=168,
    )
    print(f"  persistence: shape={pers.shape}, NaN count={pers.isna().sum()}")
    print(f"  persistence head:\n{pers.head(3)}")

    print("\nFitting SARIMAX (slow, ~30s on 90 days of hourly data)…")
    sx = sarimax_forecast(train_features=train, predict_features=predict)
    print(f"  sarimax: shape={sx.shape}, NaN count={sx.isna().sum()}")
    print(f"  sarimax head:\n{sx.head(3)}")

    print("\n[ok] Baseline module sane on portfolio sample.")


if __name__ == "__main__":
    C._print_summary()
    main()
