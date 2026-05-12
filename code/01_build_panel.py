"""
01_build_panel.py — Build the hourly portfolio panel from GoiEner CSVs.

What it does:
  1. Decompresses the three .tzst archives if they aren't already extracted
  2. Walks all CSVs (using rglob to handle the nested 'goi4_*' subfolders)
  3. Filters to residential households with sufficient valid hours
  4. Aggregates household-hour readings to portfolio-hour
  5. Caches the result as parquet

Output: output/_cache/portfolio_hourly.parquet
        output/_cache/eligible_households.csv

Modes:
  Sample mode (GOIENER_DATA_MODE=sample) reads from data/sample/ and uses
  the synthetic 50-household generator. Useful for testing.

  Full mode (default) reads from data/raw/ and processes all 16,500+
  residential households. First run takes 30–60 minutes; subsequent runs
  are instant via the parquet cache.
"""

from __future__ import annotations

import sys
import tarfile
import time
from pathlib import Path

import pandas as pd
import zstandard as zstd
from tqdm import tqdm

import config as C


# ====================================================================
# Decompression (only if archives not yet extracted)
# ====================================================================

def decompress_if_needed():
    """Decompress .tzst archives into data/raw/ if not already present.

    Sample mode skips this entirely — sample data is already plain CSVs.
    """
    if C.DATA_MODE == "sample":
        return

    archives = [
        ("imp-pre.tzst",  C.PRE_DIR),
        ("imp-in.tzst",   C.IN_DIR),
        ("imp-post.tzst", C.POST_DIR),
    ]

    # Detect already-extracted state with rglob (archives unpack nested)
    needed = [
        (arc, dest) for arc, dest in archives
        if not dest.exists() or len(list(dest.rglob("*.csv"))) == 0
    ]
    if not needed:
        print("[ok] All archives already decompressed.")
        return

    for arc_name, dest_dir in needed:
        arc_path = C.DATA_DIR / arc_name
        if not arc_path.exists():
            raise FileNotFoundError(
                f"Archive not found: {arc_path}\n"
                f"Run `python code/00_download_goiener.py` first, or download"
                f" manually from https://doi.org/10.5281/zenodo.7362094"
            )

        print(f"[decompress] {arc_name} → {dest_dir}")
        dest_dir.mkdir(parents=True, exist_ok=True)

        with open(arc_path, "rb") as fh:
            dctx = zstd.ZstdDecompressor()
            with dctx.stream_reader(fh) as reader:
                with tarfile.open(fileobj=reader, mode="r|") as tar:
                    tar.extractall(dest_dir)

        n_csvs = len(list(dest_dir.rglob("*.csv")))
        print(f"  → extracted {n_csvs} CSVs")


# ====================================================================
# Eligibility filtering
# ====================================================================

def load_metadata() -> pd.DataFrame:
    """Load metadata.csv, filter to residential CNAE codes."""
    if not C.METADATA_PATH.exists():
        raise FileNotFoundError(f"metadata.csv not found at {C.METADATA_PATH}")

    md = pd.read_csv(C.METADATA_PATH, dtype={"cnae": str})

    # Residential = CNAE starting with '98' (households)
    if "cnae" in md.columns:
        md["is_residential"] = md["cnae"].fillna("").str.startswith("98")
    else:
        # Sample mode synthetic data may not have CNAE; treat all as residential
        md["is_residential"] = True

    return md


def find_eligible_households(metadata: pd.DataFrame) -> pd.DataFrame:
    """Identify residential households with adequate data coverage.

    Filters:
      - is_residential
      - actual_samples >= MIN_VALID_HOURS_PER_HH (if column exists)
      - missing_samples_pct <= MAX_IMPUTATION_RATE * 100 (if column exists)
    """
    eligible = metadata[metadata["is_residential"]].copy()

    if "actual_samples" in eligible.columns:
        eligible = eligible[eligible["actual_samples"] >= C.MIN_VALID_HOURS_PER_HH]

    if "missing_samples_pct" in eligible.columns:
        eligible = eligible[
            eligible["missing_samples_pct"] <= C.MAX_IMPUTATION_RATE * 100
        ]

    return eligible


# ====================================================================
# Panel construction
# ====================================================================

# Module-level cache: built lazily on the first find_csvs_for_household
# call. With ~45k CSVs across three nested archives, calling rglob() per
# hash made the original implementation O(N*M) — minutes-per-hash on full
# data. The single tree walk below is paid once per process.
HASH_TO_PATHS: dict[str, list[Path]] | None = None


def _build_index() -> dict[str, list[Path]]:
    """Walk all three archive trees once and index CSVs by their stem (hash)."""
    t0 = time.time()
    index: dict[str, list[Path]] = {}
    for d in (C.PRE_DIR, C.IN_DIR, C.POST_DIR):
        if not d.exists():
            continue
        for p in d.rglob("*.csv"):
            index.setdefault(p.stem, []).append(p)
    elapsed = time.time() - t0
    print(f"[index] built path index for {len(index):,} hashes in {elapsed:.1f}s")
    return index


def find_csvs_for_household(user_hash: str) -> list[Path]:
    """Return all CSV paths for a household across the three archives.

    First call lazily builds a module-global hash→paths index so the per-
    household lookup is O(1) instead of O(total_csvs). Behavior identical
    to the original rglob loop, just without the quadratic cost.
    """
    global HASH_TO_PATHS
    if HASH_TO_PATHS is None:
        HASH_TO_PATHS = _build_index()
    return HASH_TO_PATHS.get(user_hash, [])


def load_household_series(user_hash: str) -> pd.DataFrame | None:
    """Load all hourly readings for one household, concatenated and deduplicated.

    Returns DataFrame with columns: timestamp (datetime), kwh (float),
    or None if no data found.
    """
    paths = find_csvs_for_household(user_hash)
    if not paths:
        return None

    pieces = []
    for p in paths:
        try:
            df = pd.read_csv(p)
        except Exception:
            continue

        # Standardize columns (different subsets may have different headers)
        cols_lower = {c.lower(): c for c in df.columns}
        ts_col = cols_lower.get("timestamp") or cols_lower.get("time") or df.columns[0]
        kwh_col = cols_lower.get("kwh") or df.columns[1]

        df = df[[ts_col, kwh_col]].rename(columns={ts_col: "timestamp", kwh_col: "kwh"})
        pieces.append(df)

    if not pieces:
        return None

    out = pd.concat(pieces, ignore_index=True)
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    out = out.dropna(subset=["timestamp"])
    out["kwh"] = pd.to_numeric(out["kwh"], errors="coerce")
    out = out.dropna(subset=["kwh"])

    # Deduplicate (multiple files may have overlapping timestamps)
    out = out.groupby("timestamp", as_index=False)["kwh"].mean()

    out["user_hash"] = user_hash
    return out


def build_portfolio_panel(eligible_users: list[str]) -> pd.DataFrame:
    """Aggregate all households to portfolio-hour panel.

    Returns DataFrame with columns: timestamp, kwh_total, n_households.
    """
    print(f"\nBuilding portfolio panel from {len(eligible_users):,} households")
    print("(this is the slow step; first run only — subsequent runs hit the cache)")

    accumulator = {}  # timestamp → [sum_kwh, n_hh]

    for user_hash in tqdm(eligible_users, desc="Households"):
        s = load_household_series(user_hash)
        if s is None:
            continue
        for ts, kwh in zip(s["timestamp"], s["kwh"]):
            if ts not in accumulator:
                accumulator[ts] = [0.0, 0]
            accumulator[ts][0] += kwh
            accumulator[ts][1] += 1

    if not accumulator:
        raise RuntimeError("No data loaded — check that decompression succeeded.")

    rows = [
        {"timestamp": ts, "kwh_total": v[0], "n_households": v[1]}
        for ts, v in accumulator.items()
    ]
    df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
    return df


# ====================================================================
# Main
# ====================================================================

def main():
    cache_path = C.CACHE_DIR / "portfolio_hourly.parquet"

    if cache_path.exists():
        size_mb = cache_path.stat().st_size / 1024 / 1024
        print(f"[ok] Panel already cached at {cache_path}")
        print(f"    Size: {size_mb:.1f} MB")
        print(f"    To rebuild: delete the cache file and re-run.")
        df = pd.read_parquet(cache_path)
        print(f"\nPanel summary:")
        print(f"  Rows:        {len(df):,}")
        print(f"  Date range:  {df['timestamp'].min()} → {df['timestamp'].max()}")
        print(f"  Avg n_hh:    {df['n_households'].mean():.0f}")
        print(f"  Total kWh:   {df['kwh_total'].sum() / 1e6:.1f} GWh")
        return

    t0 = time.time()

    # Step 1: Decompress if needed
    decompress_if_needed()

    # Step 2: Load metadata, filter to eligible households
    md = load_metadata()
    eligible = find_eligible_households(md)
    print(f"\nEligible households: {len(eligible):,}")

    eligible_path = C.OUTPUT_DIR / "_cache" / "eligible_households.csv"
    eligible.to_csv(eligible_path, index=False)

    # Step 3: Build the portfolio panel
    user_hashes = eligible["user"].tolist() if "user" in eligible.columns \
        else eligible.iloc[:, 0].astype(str).tolist()

    panel = build_portfolio_panel(user_hashes)

    # Step 4: Cache to parquet
    panel.to_parquet(cache_path, index=False)

    elapsed = time.time() - t0
    print(f"\n[ok] Panel built in {elapsed/60:.1f} min")
    print(f"  Rows:       {len(panel):,}")
    print(f"  Date range: {panel['timestamp'].min()} → {panel['timestamp'].max()}")
    print(f"  Cached to:  {cache_path}")
    print(f"\nDone. Next step: python code/02_segment_households.py")


if __name__ == "__main__":
    C._print_summary()
    main()
