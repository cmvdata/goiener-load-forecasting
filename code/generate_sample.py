"""
generate_sample.py — Generate a synthetic 50-household sample for pipeline testing.

The sample mirrors the structure of the real GoiEner data but contains injected
patterns (4 behavioral segments, weather sensitivity, weekly cycles) so that
downstream scripts (segmentation, forecasting) produce non-degenerate outputs.

Output: data/sample/metadata.csv
        data/sample/imp-pre/<hash>.csv  (50 files)
        data/sample/imp-in/<hash>.csv   (50 files)
        data/sample/imp-post/<hash>.csv (50 files)

Note: this writes flat (not nested) CSVs in the sample directory for simplicity.
01_build_panel.py uses rglob, so it handles both layouts transparently.
"""

import hashlib
import numpy as np
import pandas as pd
from pathlib import Path

import config as C


SEED = C.SEED
np.random.seed(SEED)

N_HOUSEHOLDS = 50

# Date ranges for each subset
RANGES = {
    "imp-pre":  ("2018-01-01", "2020-02-29"),  # ~2 years pre-COVID
    "imp-in":   ("2020-03-01", "2021-05-30"),  # lockdown + curfew period
    "imp-post": ("2021-05-31", "2022-06-30"),  # post-pandemic, includes 2.0TD reform
}


# ====================================================================
# Behavioral segment archetypes
# ====================================================================

def daily_profile_for_segment(segment: int) -> np.ndarray:
    """Return a 24-hour shape for one of 4 behavioral archetypes."""
    hours = np.arange(24)

    if segment == 0:
        # High-evening users: spike at 19:00–22:00
        shape = 0.3 + 0.4 * np.exp(-((hours - 20) ** 2) / 6)
    elif segment == 1:
        # Flat-profile users: low variance throughout the day
        shape = 0.4 + 0.05 * np.sin(2 * np.pi * (hours - 12) / 24)
    elif segment == 2:
        # Morning-skewed users: spike at 7:00–9:00
        shape = 0.3 + 0.35 * np.exp(-((hours - 8) ** 2) / 4)
    else:
        # Low-consumption users: half magnitude of others
        shape = 0.15 + 0.10 * np.exp(-((hours - 19) ** 2) / 8)

    return shape


def build_household_series(user_hash: str, segment: int,
                           start: str, end: str,
                           temperature_series: pd.Series) -> pd.DataFrame:
    """Generate hourly kWh series for one household over a date range."""
    timestamps = pd.date_range(start=start, end=end, freq="h", tz=None)
    n = len(timestamps)

    base_profile = daily_profile_for_segment(segment)
    hour_idx = timestamps.hour.values
    daily = base_profile[hour_idx]

    # Weekly effect: weekends slightly compressed in segment 2 (morning)
    is_weekend = (timestamps.dayofweek >= 5).astype(float)
    weekly_mult = 1.0 + 0.1 * is_weekend if segment == 2 else 1.0 - 0.05 * is_weekend

    # Temperature sensitivity (heating below 12°C, cooling above 24°C)
    temps = temperature_series.reindex(timestamps).interpolate(method="linear")
    temps = temps.bfill().ffill().values
    heating = np.maximum(12.0 - temps, 0) * 0.025
    cooling = np.maximum(temps - 24.0, 0) * 0.020
    temp_effect = heating + cooling

    # Annual seasonality (lower in summer for non-cooling segments)
    day_of_year = timestamps.dayofyear.values
    annual = 1.0 + 0.10 * np.cos(2 * np.pi * (day_of_year - 15) / 365)

    # Household-specific scale and noise
    rng = np.random.default_rng(int(user_hash[:8], 16))
    scale = rng.uniform(0.7, 1.3)
    noise = rng.normal(0, 0.05, n)

    kwh = scale * (daily * weekly_mult * annual + temp_effect) * (1 + noise)
    kwh = np.maximum(kwh, 0.01)  # no negative consumption

    return pd.DataFrame({
        "timestamp": timestamps,
        "kwh": kwh,
        "imputed": np.zeros(n, dtype=int),
    })


# ====================================================================
# Synthetic temperature
# ====================================================================

def synthetic_temperature(start: str, end: str) -> pd.Series:
    """Generate a plausible Spanish hourly temperature series."""
    timestamps = pd.date_range(start=start, end=end, freq="h", tz=None)
    n = len(timestamps)

    day_of_year = timestamps.dayofyear.values
    hour = timestamps.hour.values

    # Annual cycle: ~5°C in winter, ~30°C in summer
    annual = 17.5 + 12.5 * np.cos(2 * np.pi * (day_of_year - 200) / 365)

    # Daily cycle: cooler night, warmer mid-afternoon
    daily = 5.0 * np.sin(2 * np.pi * (hour - 6) / 24)

    # Noise
    rng = np.random.default_rng(SEED)
    noise = rng.normal(0, 1.5, n)

    return pd.Series(annual + daily + noise, index=timestamps)


# ====================================================================
# Main
# ====================================================================

def main():
    out_dir = C.PROJECT_ROOT / "data" / "sample"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Generate household IDs and assign segments
    rng = np.random.default_rng(SEED)
    user_hashes = [
        hashlib.sha256(f"household-{i}".encode()).hexdigest()
        for i in range(N_HOUSEHOLDS)
    ]
    segments = rng.integers(0, 4, N_HOUSEHOLDS)

    # Metadata
    provinces = ["Bizkaia", "Gipuzkoa", "Araba/Álava", "Navarra", "Madrid"]
    cnae_codes = ["9810", "9820", "9830"]

    md_rows = []
    for h, seg in zip(user_hashes, segments):
        md_rows.append({
            "user": h,
            "start_date": "2018-01-01",
            "end_date": "2022-06-30",
            "length_days": 1641,
            "length_years": 4.5,
            "potential_samples": 1641 * 24,
            "actual_samples": 1641 * 24,
            "missing_samples_abs": 0,
            "missing_samples_pct": 0.0,
            "contract_start_date": "2018-01-01",
            "contract_end_date": "",
            "contracted_tariff": rng.choice(["2.0TD", "2.0DHA"]),
            "self_consumption_type": rng.choice(["00", "00", "00", "41"]),
            "p1": rng.uniform(3.45, 5.75),
            "p2": rng.uniform(3.45, 5.75),
            "p3": 0, "p4": 0, "p5": 0, "p6": 0,
            "province": rng.choice(provinces),
            "municipality": "",
            "zip_code": "",
            "cnae": rng.choice(cnae_codes),
            "_segment_truth": int(seg),  # debug column; not in real metadata
        })

    md = pd.DataFrame(md_rows)
    md.to_csv(out_dir / "metadata.csv", index=False)
    print(f"Wrote metadata.csv with {len(md)} households")

    # Generate temperature series spanning all subsets
    temp_full = synthetic_temperature("2018-01-01", "2022-06-30")

    # Generate per-subset CSVs
    for subset_name, (start, end) in RANGES.items():
        subset_dir = out_dir / subset_name
        subset_dir.mkdir(parents=True, exist_ok=True)

        for h, seg in zip(user_hashes, segments):
            df = build_household_series(h, int(seg), start, end, temp_full)
            df.to_csv(subset_dir / f"{h}.csv", index=False)

        print(f"  {subset_name}: 50 CSVs written")

    print(f"\n[ok] Synthetic sample at {out_dir}")
    print(f"    To use it: export GOIENER_DATA_MODE=sample")


if __name__ == "__main__":
    main()
