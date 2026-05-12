"""
03_feature_engineering.py — Build the feature matrix for all forecasting models.

Produces one parquet per level (portfolio + each of N_SEGMENTS), each
with the same schema. Every feature is **strictly causal**: at time t the
row only knows information available before t. The exception is realized
weather, which we use intentionally — the README documents this caveat
(production would consume forecast weather; here we evaluate methodology).

Inputs:
  output/_cache/portfolio_hourly.parquet   (from 01_build_panel.py)
  data/weather/{bilbao,pamplona,madrid}.parquet (from 00_download_weather.py)
  output/02_household_segments.csv          (from 02_segment_households.py)

Outputs:
  output/_cache/features_portfolio.parquet
  output/_cache/features_segment_<N>.parquet  for N in 0..N_SEGMENTS-1

Note on rolling means: we use shift(1).rolling(window) so that the rolling
mean at t aggregates ONLY observations strictly before t. This is the
non-leaky form of the feature.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import holidays
import numpy as np
import pandas as pd
from tqdm import tqdm

import config as C


_BUILD_PATH = Path(__file__).parent / "01_build_panel.py"
_spec = importlib.util.spec_from_file_location("build_panel", _BUILD_PATH)
build_panel = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(build_panel)


# ====================================================================
# Weather aggregation
# ====================================================================

def load_weighted_weather() -> pd.DataFrame:
    """Load the three location parquets and produce a portfolio-weighted hourly frame."""
    pieces = []
    for loc, cfg in C.WEATHER_LOCATIONS.items():
        path = C.WEATHER_DIR / f"{loc}.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing weather parquet for {loc} at {path}. "
                f"Run code/00_download_weather.py first."
            )
        w = pd.read_parquet(path)
        w["location"] = loc
        w["weight"] = cfg["weight"]
        pieces.append(w)

    long = pd.concat(pieces, ignore_index=True)
    long["timestamp"] = pd.to_datetime(long["timestamp"])

    # Weighted average per hour
    weighted = (
        long.assign(
            t_w=long["temperature_2m"] * long["weight"],
            h_w=long["relative_humidity_2m"] * long["weight"],
            p_w=long["precipitation"] * long["weight"],
            ws_w=long["wind_speed_10m"] * long["weight"],
        )
        .groupby("timestamp")
        .agg(
            temperature_weighted=("t_w", "sum"),
            humidity_weighted=("h_w", "sum"),
            precipitation_weighted=("p_w", "sum"),
            wind_weighted=("ws_w", "sum"),
            total_weight=("weight", "sum"),
        )
        .reset_index()
    )

    # Renormalize in case weights don't sum to 1 (they should, but defensively)
    if not np.allclose(weighted["total_weight"], 1.0, atol=1e-3):
        for col in ("temperature_weighted", "humidity_weighted",
                    "precipitation_weighted", "wind_weighted"):
            weighted[col] = weighted[col] / weighted["total_weight"]

    return weighted.drop(columns="total_weight")


# ====================================================================
# Per-segment portfolio aggregation
# ====================================================================

def build_segment_panel(user_hashes: list[str]) -> pd.DataFrame:
    """Aggregate a subset of households to hourly kwh_total.

    Single-segment helper, retained for compatibility. For full-mode runs
    use build_all_segment_panels which scans households once across all
    segments — same disk-I/O bottleneck only paid one time.
    """
    accumulator: dict[pd.Timestamp, list[float]] = {}

    for uh in user_hashes:
        s = build_panel.load_household_series(uh)
        if s is None:
            continue
        for ts, kwh in zip(s["timestamp"], s["kwh"]):
            if ts not in accumulator:
                accumulator[ts] = [0.0, 0]
            accumulator[ts][0] += kwh
            accumulator[ts][1] += 1

    if not accumulator:
        raise RuntimeError("No data loaded for segment aggregation.")

    rows = [
        {"timestamp": ts, "kwh_total": v[0], "n_households": v[1]}
        for ts, v in accumulator.items()
    ]
    return pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)


def sum_segment_panels(seg_panels: dict[int, pd.DataFrame]) -> pd.DataFrame:
    """Combine segment panels into a coherent portfolio panel.

    Portfolio kwh_total at each timestamp = sum over segments of their
    kwh_total. This guarantees sum(segment forecasts) == portfolio forecast
    in the hierarchy, which is what BottomUp / MinT reconciliation expect.
    """
    pieces = []
    for seg, panel in seg_panels.items():
        pieces.append(panel[["timestamp", "kwh_total", "n_households"]])
    if not pieces:
        raise RuntimeError("No segment panels to sum.")
    long = pd.concat(pieces, ignore_index=True)
    return (
        long.groupby("timestamp", as_index=False)
        .agg(kwh_total=("kwh_total", "sum"),
             n_households=("n_households", "sum"))
        .sort_values("timestamp")
        .reset_index(drop=True)
    )


def build_all_segment_panels(segments: pd.DataFrame) -> dict[int, pd.DataFrame]:
    """Build one panel per segment in a SINGLE disk pass.

    Loads each household's series once and routes it to its segment's
    accumulator. On Windows with cold disk cache 01 took 14 hours for the
    same scan — repeating four times (once per segment) would mean ~56h.
    """
    user_to_seg = dict(zip(segments["user_hash"].astype(str),
                           segments["segment"].astype(int)))
    seg_ids = sorted(set(user_to_seg.values()))
    accumulators: dict[int, dict] = {s: {} for s in seg_ids}

    user_hashes = list(user_to_seg.keys())
    for uh in tqdm(user_hashes, desc="HHs (all segments, 1 pass)"):
        s = build_panel.load_household_series(uh)
        if s is None:
            continue
        seg = user_to_seg.get(uh)
        if seg is None:
            continue
        accum = accumulators[seg]
        for ts, kwh in zip(s["timestamp"], s["kwh"]):
            cur = accum.get(ts)
            if cur is None:
                accum[ts] = [kwh, 1]
            else:
                cur[0] += kwh
                cur[1] += 1

    panels: dict[int, pd.DataFrame] = {}
    for seg, accum in accumulators.items():
        if not accum:
            continue
        rows = [{"timestamp": ts, "kwh_total": v[0], "n_households": v[1]}
                for ts, v in accum.items()]
        panels[seg] = (
            pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
        )
    return panels


# ====================================================================
# Feature construction
# ====================================================================

def add_calendar_features(df: pd.DataFrame, holiday_set: set) -> pd.DataFrame:
    ts = df["timestamp"]
    df["hour_of_day"] = ts.dt.hour.astype("int16")
    df["day_of_week"] = ts.dt.dayofweek.astype("int16")
    df["is_weekend"] = (df["day_of_week"] >= 5).astype("int8")
    df["month"] = ts.dt.month.astype("int16")
    df["day_of_year"] = ts.dt.dayofyear.astype("int16")

    df["is_holiday"] = ts.dt.normalize().isin(holiday_set).astype("int8")

    # Cyclical encodings
    df["hour_sin"] = np.sin(2 * np.pi * df["hour_of_day"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour_of_day"] / 24)
    df["doy_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365.25)
    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    # df must be sorted by timestamp on a strict 1h grid for lags to mean what
    # we think they mean; assert at call-site rather than here.
    df["kwh_lag_1"] = df["kwh_total"].shift(1)
    df["kwh_lag_24"] = df["kwh_total"].shift(24)
    df["kwh_lag_168"] = df["kwh_total"].shift(168)

    # Past-only rolling means: shift(1) excludes the current observation,
    # so the value at t aggregates t-1 .. t-window.
    df["kwh_rolling_mean_24"] = df["kwh_total"].shift(1).rolling(window=24, min_periods=24).mean()
    df["kwh_rolling_mean_168"] = df["kwh_total"].shift(1).rolling(window=168, min_periods=168).mean()
    return df


def add_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    t = df["temperature_weighted"]
    df["temp_sq"] = t ** 2
    df["heating_demand"] = (12.0 - t).clip(lower=0)
    df["cooling_demand"] = (t - 24.0).clip(lower=0)
    return df


def add_tariff_feature(df: pd.DataFrame) -> pd.DataFrame:
    cutoff = pd.Timestamp(C.TARIFF_REFORM)
    df["post_tariff_reform"] = (df["timestamp"] >= cutoff).astype("int8")
    return df


def regularize_to_hourly_grid(df: pd.DataFrame) -> pd.DataFrame:
    """Reindex to a strict hourly grid covering the observed range.

    Lag features assume 1h spacing; gaps in the panel would silently produce
    misaligned lags. We reindex with NaN-fill so the forecasting code can see
    and skip rows with missing lags.
    """
    df = df.sort_values("timestamp").reset_index(drop=True)
    if df.empty:
        return df
    full = pd.date_range(df["timestamp"].min(), df["timestamp"].max(), freq="h")
    df = df.set_index("timestamp").reindex(full).rename_axis("timestamp").reset_index()
    return df


def build_features(panel: pd.DataFrame, weather: pd.DataFrame,
                   holiday_set: set) -> pd.DataFrame:
    df = panel.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = regularize_to_hourly_grid(df)

    df = df.merge(weather, on="timestamp", how="left")

    df = add_calendar_features(df, holiday_set)
    df = add_lag_features(df)
    df = add_weather_features(df)
    df = add_tariff_feature(df)

    return df


# ====================================================================
# Main
# ====================================================================

def main():
    portfolio_path = C.CACHE_DIR / "portfolio_hourly.parquet"
    if not portfolio_path.exists():
        raise FileNotFoundError(
            f"Run code/01_build_panel.py first; missing {portfolio_path}"
        )

    seg_path = C.OUTPUT_DIR / "02_household_segments.csv"
    if not seg_path.exists():
        raise FileNotFoundError(
            f"Run code/02_segment_households.py first; missing {seg_path}"
        )

    print("Loading inputs")
    segments = pd.read_csv(seg_path)
    weather = load_weighted_weather()

    # Spanish national holidays for the years 02 covers (need a temporary
    # year set; refined once we have the segment panels)
    portfolio_cached = pd.read_parquet(portfolio_path)
    portfolio_cached["timestamp"] = pd.to_datetime(portfolio_cached["timestamp"])
    years = sorted(set(portfolio_cached["timestamp"].dt.year.unique()))
    es_holidays = holidays.Spain(years=years)
    holiday_dates = set(pd.to_datetime(list(es_holidays.keys())))

    # ---- Per-segment panels (single disk pass) ----------------------------
    print(f"\nBuilding segment panels (single scan over "
          f"{int((segments['segment']>=0).sum()):,} segmented households)")
    seg_panels = build_all_segment_panels(segments)

    # ---- Portfolio = sum of segments (coherent hierarchy) -----------------
    portfolio_coherent = sum_segment_panels(seg_panels)
    print(f"\nBuilding features at portfolio level (segment-coherent: "
          f"sum over {len(seg_panels)} segments, "
          f"{len(portfolio_coherent):,} hourly rows)")
    feats_portfolio = build_features(portfolio_coherent, weather, holiday_dates)
    out_p = C.CACHE_DIR / "features_portfolio.parquet"
    feats_portfolio.to_parquet(out_p, index=False)
    print(f"  → {out_p.name}: {len(feats_portfolio):,} rows, "
          f"{feats_portfolio.shape[1]} cols")

    # ---- Per-segment features ----------------------------------------------
    print("\nBuilding features per segment")
    for seg_id, seg_panel in seg_panels.items():
        feats_seg = build_features(seg_panel, weather, holiday_dates)
        out_s = C.CACHE_DIR / f"features_segment_{seg_id}.parquet"
        feats_seg.to_parquet(out_s, index=False)
        n_members = int((segments["segment"] == seg_id).sum())
        print(f"  segment {seg_id}: {n_members} hh, "
              f"{len(feats_seg):,} rows → {out_s.name}")

    # Sanity check: lag columns should have NaNs only at the head
    head_nans = int(feats_portfolio["kwh_lag_168"].isna().sum())
    print(f"\nSanity: kwh_lag_168 has {head_nans} NaNs (expect 168 leading rows)")
    print("[ok] Feature engineering complete.")


if __name__ == "__main__":
    C._print_summary()
    main()
