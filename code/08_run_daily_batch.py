"""
08_run_daily_batch.py — Operational batch forecast generator (recursive).

Mirrors the nightly batch pattern of a real supply forecasting team. At
issue time T_now (= the most recent feature row whose lag/rolling
columns are fully populated) we:

  1. Load the recursive single-step champion: 5 quantile boosters at
     the portfolio level.
  2. Roll the booster suite forward 168 steps to produce a full hourly
     quantile trajectory.
  3. Apply post-hoc conformal δ-adjustments from 06b (if present) to the
     non-median quantiles. The median is left untouched.
  4. Write the result to output/forecasts/forecasts_<YYYY-MM-DD>.parquet
     and append a JSONL audit line to logs/batch_runs.jsonl.

The script is idempotent: re-running on the same date overwrites the
parquet and appends a new log line.

Pipeline 1 forecasts the portfolio aggregate only. Per-segment forecasting
and MinT-OLS hierarchical reconciliation were evaluated and dropped — see
config.FORECAST_LEVELS and code/_archive/07_hierarchical_reconcile.py.
"""

from __future__ import annotations

import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

import config as C


_HERE = Path(__file__).parent
_l = importlib.util.spec_from_file_location("train_lgb", _HERE / "05_train_lightgbm.py")
train_lgb = importlib.util.module_from_spec(_l)
_l.loader.exec_module(train_lgb)


N_STEPS = 168

POOL_BUCKETS = [
    ("h<=24", 1, 24),
    ("h=25-72", 25, 72),
    ("h=73-168", 73, 168),
]


def assign_pool_bucket(h: int) -> str:
    for label, lo, hi in POOL_BUCKETS:
        if lo <= h <= hi:
            return label
    raise ValueError(f"horizon_h={h} not covered by POOL_BUCKETS")


def load_conformal_adjustments() -> pd.DataFrame | None:
    """Load split-conformal adjustments produced by 06b. None if missing."""
    path = C.OUTPUT_DIR / "06b_conformal_adjustments.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def apply_conformal(traj: pd.DataFrame, adj: pd.DataFrame, level: str) -> pd.DataFrame:
    """Add calibrated δ to each non-median quantile of a recursive trajectory."""
    if adj is None or adj.empty:
        traj["y_pred_raw"] = traj["y_pred"]
        traj["adjustment"] = 0.0
        return traj
    traj = traj.copy()
    traj["pool_bucket"] = traj["horizon_h"].map(assign_pool_bucket)
    sub = adj[adj["level"] == level][["bucket", "quantile", "adjustment"]].rename(
        columns={"bucket": "pool_bucket"}
    )
    out = traj.merge(sub, on=["pool_bucket", "quantile"], how="left")
    out["adjustment"] = out["adjustment"].fillna(0.0)
    out["y_pred_raw"] = out["y_pred"]
    out["y_pred"] = out["y_pred"] + out["adjustment"]
    return out


def load_champion(level: str) -> dict:
    out = {}
    for q in C.QUANTILES:
        path = train_lgb.model_filename(level, q)
        if not path.exists():
            raise FileNotFoundError(
                f"Missing champion {path.name}; run code/05_train_lightgbm.py"
            )
        out[q] = lgb.Booster(model_file=str(path))
    return out


def load_features(level: str) -> pd.DataFrame:
    if level == "portfolio":
        path = C.CACHE_DIR / "features_portfolio.parquet"
    else:
        path = C.CACHE_DIR / f"features_{level}.parquet"
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def main():
    levels = list(C.FORECAST_LEVELS)
    feats = {lv: load_features(lv) for lv in levels}

    # Issue time selection. For a "true" production batch we'd issue at
    # the most recent moment with full features and project N_STEPS into
    # the future, where calendar + weather forecasts must be available.
    # In this offline demo the features parquet only spans observed data,
    # so we issue N_STEPS hours before the parquet end and produce the
    # forecast over a window that's already in the data — the operational
    # equivalent of a one-shot backtest.
    pf = feats["portfolio"]
    feat_cols = train_lgb.feature_columns(pf)
    valid = pf.dropna(subset=feat_cols)
    if valid.empty:
        raise RuntimeError("No portfolio row with complete features.")
    last_ts = pd.Timestamp(pf["timestamp"].max())
    issue_time = last_ts - pd.Timedelta(hours=N_STEPS)
    if issue_time not in set(valid["timestamp"]):
        candidates = valid["timestamp"][valid["timestamp"] <= issue_time]
        if candidates.empty:
            raise RuntimeError(
                "No valid issue point N_STEPS before the data end."
            )
        issue_time = pd.Timestamp(candidates.max())
    print(f"\nIssue time: {issue_time} (data ends {last_ts})")

    adj = load_conformal_adjustments()
    if adj is None:
        print("  [warn] 06b_conformal_adjustments.csv missing — emitting RAW bands.")
    else:
        print(f"  loaded conformal adjustments: {len(adj)} rows "
              f"({adj['bucket'].nunique()} buckets × "
              f"{adj['quantile'].nunique()} quantiles)")

    # Recursive trajectory (portfolio only — Pipeline 1).
    blocks = []
    for level in levels:
        boosters = load_champion(level)
        traj = train_lgb.recursive_forecast_quantiles(
            boosters_by_q=boosters,
            features=feats[level],
            issue_time=issue_time,
            n_steps=N_STEPS,
        )
        traj["level"] = level
        traj = apply_conformal(traj, adj, level)
        blocks.append(traj)
    raw = pd.concat(blocks, ignore_index=True)
    print(f"  trajectory: {len(raw):,} rows ({len(C.QUANTILES)} quantiles "
          f"× up to {N_STEPS} steps)")

    # Pivot wide on quantile.
    wide = raw.pivot_table(
        index=["level", "timestamp", "horizon_h"],
        columns="quantile",
        values="y_pred",
        aggfunc="first",
    ).reset_index()
    wide.columns.name = None
    rename_map = {q: f"q{int(round(q*1000)):03d}" for q in C.QUANTILES}
    wide = wide.rename(columns=rename_map)

    wide["timestamp"] = pd.to_datetime(wide["timestamp"])
    wide["issue_time"] = issue_time

    generated_at = datetime.now(timezone.utc).isoformat()
    wide["generated_at"] = generated_at

    wide = wide.sort_values(["level", "timestamp"]).reset_index(drop=True)

    fc_dir = C.OUTPUT_DIR / "forecasts"
    fc_dir.mkdir(exist_ok=True)
    today = datetime.now().strftime("%Y-%m-%d")
    out_path = fc_dir / f"forecasts_{today}.parquet"
    wide.to_parquet(out_path, index=False)
    print(f"  → {out_path.relative_to(C.PROJECT_ROOT)}: {len(wide):,} rows")

    log_event = {
        "event": "batch_complete",
        "timestamp": generated_at,
        "issue_time": issue_time.isoformat(),
        "n_steps": N_STEPS,
        "n_levels": len(levels),
        "n_forecast_rows": int(len(wide)),
        "champion_dir": str(C.MODELS_DIR / "champion"),
        "conformal_calibrated": adj is not None,
        "data_mode": C.DATA_MODE,
        "out_path": str(out_path.relative_to(C.PROJECT_ROOT)),
    }
    log_path = C.LOGS_DIR / "batch_runs.jsonl"
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(log_event) + "\n")
    print(f"  → {log_path.relative_to(C.PROJECT_ROOT)}: appended 1 line")

    pf_view = wide[wide["level"] == "portfolio"].head(3)
    print("\nPortfolio forecast preview (first 3 hours):")
    print(pf_view[["timestamp", "horizon_h", "q050", "q500", "q950"]].to_string(index=False))
    print("\n[ok] Daily batch complete.")


if __name__ == "__main__":
    C._print_summary()
    main()
