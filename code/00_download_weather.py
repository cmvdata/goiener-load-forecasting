"""
00_download_weather.py — Fetch hourly weather from Open-Meteo for three locations.

Full mode hits the Open-Meteo Archive API (no auth, no key) for Bilbao,
Pamplona, and Madrid over the GoiEner observation window.

Sample mode skips the API entirely: it generates synthetic temperature using
the same logic as generate_sample.py and constructs plausible humidity,
precipitation, and wind series so 03_feature_engineering.py has a complete
weather frame to join.

Output: data/weather/{bilbao,pamplona,madrid}.parquet
        Columns: timestamp, temperature_2m, relative_humidity_2m,
                 precipitation, wind_speed_10m
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

import config as C


WEATHER_START = "2018-01-01"
WEATHER_END = "2022-06-30"

REQUEST_TIMEOUT = 60
MAX_RETRIES = 5
BACKOFF_BASE_S = 5
MIN_PARQUET_BYTES = 1_000_000  # >1 MB → assume valid cache


# ====================================================================
# Live API path
# ====================================================================

def fetch_openmeteo(loc_name: str, lat: float, lon: float,
                    start: str, end: str) -> pd.DataFrame:
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": end,
        "hourly": ",".join(C.WEATHER_VARIABLES),
        "timezone": "Europe/Madrid",
    }

    last_exc: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.get(C.OPENMETEO_BASE, params=params, timeout=REQUEST_TIMEOUT)
            r.raise_for_status()
            payload = r.json()
            break
        except Exception as e:
            last_exc = e
            backoff = BACKOFF_BASE_S * (2 ** (attempt - 1))
            print(f"  [retry {attempt}/{MAX_RETRIES}] {loc_name}: {e} "
                  f"— sleeping {backoff}s")
            time.sleep(backoff)
    else:
        raise RuntimeError(
            f"Open-Meteo request failed for {loc_name} after "
            f"{MAX_RETRIES} attempts: {last_exc}"
        )

    hourly = payload.get("hourly")
    if not hourly or "time" not in hourly:
        raise RuntimeError(f"Open-Meteo returned empty hourly block for {loc_name}: {payload}")

    df = pd.DataFrame({
        "timestamp": pd.to_datetime(hourly["time"]),
        **{var: hourly.get(var, []) for var in C.WEATHER_VARIABLES},
    })
    return df


# ====================================================================
# Sample-mode synthesis
# ====================================================================

def synthetic_temperature(start: str, end: str, location_offset: float,
                          rng_seed: int) -> pd.Series:
    """Plausible Spanish hourly temperature, with per-location offset and seed."""
    timestamps = pd.date_range(start=start, end=end, freq="h", tz=None)
    n = len(timestamps)

    day_of_year = timestamps.dayofyear.values
    hour = timestamps.hour.values

    annual = 17.5 + location_offset + 12.5 * np.cos(2 * np.pi * (day_of_year - 200) / 365)
    daily = 5.0 * np.sin(2 * np.pi * (hour - 6) / 24)

    rng = np.random.default_rng(rng_seed)
    noise = rng.normal(0, 1.5, n)

    return pd.Series(annual + daily + noise, index=timestamps)


def synthetic_weather_frame(loc_name: str, start: str, end: str) -> pd.DataFrame:
    """Build a full weather frame matching the Open-Meteo schema for sample mode."""
    # Per-location offsets reflect rough climate differences:
    # Bilbao (oceanic) is mildest, Madrid (continental) is most extreme.
    offsets = {"bilbao": -1.0, "pamplona": -2.0, "madrid": 1.5}
    seeds = {"bilbao": C.SEED + 1, "pamplona": C.SEED + 2, "madrid": C.SEED + 3}

    temp = synthetic_temperature(start, end, offsets[loc_name], seeds[loc_name])
    timestamps = temp.index
    n = len(timestamps)
    rng = np.random.default_rng(seeds[loc_name] + 100)

    # Humidity inversely related to temperature, with noise, clamped to [25, 100]
    humidity = 80.0 - 1.2 * (temp.values - temp.values.mean()) + rng.normal(0, 5, n)
    humidity = np.clip(humidity, 25.0, 100.0)

    # Precipitation: mostly zero, occasional events
    precip = np.where(rng.random(n) < 0.05, rng.exponential(1.0, n), 0.0)

    # Wind speed: positive, mean ~10 km/h
    wind = np.clip(rng.gamma(shape=2.0, scale=5.0, size=n), 0.0, None)

    return pd.DataFrame({
        "timestamp": timestamps,
        "temperature_2m": temp.values,
        "relative_humidity_2m": humidity,
        "precipitation": precip,
        "wind_speed_10m": wind,
    })


# ====================================================================
# Main
# ====================================================================

def cache_is_fresh(out_paths: list[Path]) -> bool:
    if not all(p.exists() for p in out_paths):
        return False
    return all(p.stat().st_size >= MIN_PARQUET_BYTES for p in out_paths)


def main():
    out_paths = [C.WEATHER_DIR / f"{loc}.parquet" for loc in C.WEATHER_LOCATIONS]

    if cache_is_fresh(out_paths):
        print("[ok] All three weather parquets already present (size > 1 MB each).")
        for p in out_paths:
            size_mb = p.stat().st_size / 1024 / 1024
            print(f"  {p.name}: {size_mb:.1f} MB")
        return

    sample_mode = (C.DATA_MODE == "sample")
    if sample_mode:
        print("[sample] Generating synthetic weather (no API call)")
    else:
        print(f"[full] Fetching Open-Meteo {WEATHER_START} → {WEATHER_END}")

    for loc_name, loc_cfg in C.WEATHER_LOCATIONS.items():
        out_path = C.WEATHER_DIR / f"{loc_name}.parquet"

        if sample_mode:
            df = synthetic_weather_frame(loc_name, WEATHER_START, WEATHER_END)
        else:
            df = fetch_openmeteo(
                loc_name, loc_cfg["lat"], loc_cfg["lon"],
                WEATHER_START, WEATHER_END,
            )

        # Sanity-check: every variable should have at least some non-null values.
        for var in C.WEATHER_VARIABLES:
            if df[var].isna().all():
                raise RuntimeError(f"All-NaN column for {loc_name}: {var}")

        df = df.sort_values("timestamp").reset_index(drop=True)
        df.to_parquet(out_path, index=False)
        print(f"  {loc_name}: {len(df):,} hourly rows → {out_path.name}")

    print("\n[ok] Weather download complete.")


if __name__ == "__main__":
    C._print_summary()
    main()
