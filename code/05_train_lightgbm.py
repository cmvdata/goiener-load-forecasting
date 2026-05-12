"""
05_train_lightgbm.py — Train recursive single-step LightGBM quantile models.

For each level (portfolio + 4 segments) and each quantile in
config.QUANTILES we train ONE model with target = y(t+1). At inference
the model is rolled forward 168 steps to produce the operational
1-week trajectory; the rationale is documented in docs/methodology.md.

Outputs:
  models/champion/lgb_<level>_q<Q>.txt for each (level, quantile)
  output/05_lgb_training_summary.csv

This module also exposes the helpers used downstream by 06, 08, and 09:
  - feature_columns(df)
  - make_supervised(features, horizon_h=1)
  - recursive_forecast_quantiles(boosters_by_q, features, issue_time, n_steps)
"""

from __future__ import annotations

import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

import config as C


# Single-step recursive: target is always y(t+1).
SINGLE_STEP_HORIZON = 1

# Columns that are not features. "target" is the dependent variable added by
# make_supervised; excluding it here is critical — otherwise it leaks into X
# and the model learns the identity, producing 0% training error.
NON_FEATURE_COLS = {"timestamp", "kwh_total", "n_households", "target"}

# Lag / rolling fields that have to be re-derived at each recursive step.
# They depend on the (potentially predicted) kwh_total of the working frame.
LAG_HOURS = (1, 24, 168)
ROLLING_WINDOWS = (24, 168)

# Trailing days reserved for early-stopping validation. The split is
# strictly temporal — never random — so the validation set is always
# in the future of the training subset. These knobs live in config.py
# (single source of truth); we re-export the names here for back-compat.
LGB_VALID_DAYS = C.LGB_VALID_DAYS
LGB_EARLY_STOPPING_ROUNDS = C.LGB_EARLY_STOPPING_ROUNDS


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in NON_FEATURE_COLS]


def make_supervised(features: pd.DataFrame, horizon_h: int = SINGLE_STEP_HORIZON) -> pd.DataFrame:
    """Build a row-aligned dataset where target = kwh_total at t+horizon_h.

    Drops rows with NaN in target or in any feature column.
    """
    df = features.sort_values("timestamp").reset_index(drop=True).copy()
    df["target"] = df["kwh_total"].shift(-horizon_h)
    df = df.dropna(subset=["target"] + feature_columns(df))
    return df


def quantile_token(q: float) -> str:
    return f"{int(round(q * 1000)):03d}"


def model_filename(level: str, q: float) -> Path:
    return C.MODELS_DIR / "champion" / f"lgb_{level}_q{quantile_token(q)}.txt"


def fit_with_early_stopping(
    sup: pd.DataFrame,
    q: float,
    iters: int | None = None,
    valid_days: int = LGB_VALID_DAYS,
    verify_dates: bool = False,
    label: str = "",
) -> tuple[lgb.Booster, dict]:
    """Train one quantile booster: TEMPORAL split for early stopping, then
    refit on the FULL window with `num_boost_round = best_iteration`.

    Phase 1 — early stopping (gets the iteration count K honestly).
        The last `valid_days` of `sup` (chronologically) are reserved as
        the validation set. The split is BY DATE — never by random
        sampling — so the validation set is strictly in the future of the
        training subset.

    Phase 2 — refit on full data (gets a model that uses every row).
        K from phase 1 becomes the literal num_boost_round for a fresh
        booster trained on every row of `sup` (no validation set, no
        callbacks). The 30 days that were held out for early stopping
        are recovered for training.

    The final saved booster is the phase-2 one. `info["iterations_used"]`
    is the K that was selected by early stopping, NOT a re-derived
    metric on the full-data refit (the refit has no validation signal,
    so re-deriving "best iteration" there is meaningless).
    """
    sup = sup.sort_values("timestamp")
    valid_cutoff = sup["timestamp"].max() - pd.Timedelta(days=valid_days)
    train_part = sup[sup["timestamp"] <= valid_cutoff]
    valid_part = sup[sup["timestamp"] > valid_cutoff]

    if train_part.empty or valid_part.empty:
        raise RuntimeError(
            f"Temporal split degenerate ({label}): "
            f"train={len(train_part)} valid={len(valid_part)}"
        )

    # Hard invariant: every row in valid_part is strictly after every row
    # in train_part. Aborts if violated — early-stopping must never see
    # future data.
    train_max = train_part["timestamp"].max()
    valid_min = valid_part["timestamp"].min()
    if not (valid_min > train_max):
        raise RuntimeError(
            f"Temporal split violated ({label}): "
            f"valid_min={valid_min} <= train_max={train_max}"
        )

    if verify_dates:
        print(
            f"[temporal-split-verify] {label}\n"
            f"  train: {train_part['timestamp'].min()} -> {train_max} "
            f"(n={len(train_part):,})\n"
            f"  valid: {valid_min} -> {valid_part['timestamp'].max()} "
            f"(n={len(valid_part):,})\n"
            f"  valid_min > train_max: {bool(valid_min > train_max)}\n"
            f"  refit on full sup: n={len(sup):,}"
        )

    feat_cols = feature_columns(sup)
    X_tr, y_tr = train_part[feat_cols], train_part["target"]
    X_va, y_va = valid_part[feat_cols], valid_part["target"]
    X_full, y_full = sup[feat_cols], sup["target"]

    params = dict(C.LGB_PARAMS)
    params["alpha"] = q
    if iters is not None:
        params["num_iterations"] = iters
    iters_cap = params.get("num_iterations", 2500)

    # Phase 1: early stopping on the temporal split.
    train_ds = lgb.Dataset(X_tr, label=y_tr)
    valid_ds = lgb.Dataset(X_va, label=y_va, reference=train_ds)
    es_booster = lgb.train(
        params=params,
        train_set=train_ds,
        num_boost_round=iters_cap,
        valid_sets=[valid_ds],
        valid_names=["valid"],
        callbacks=[
            lgb.early_stopping(LGB_EARLY_STOPPING_ROUNDS, verbose=False),
            lgb.log_evaluation(0),
        ],
    )
    K = int(es_booster.best_iteration)
    if K <= 0:
        # Defensive: best_iteration=0 means early stopping fired at the
        # very first round. Refitting with 0 trees is meaningless — fall
        # back to the early-stopping booster.
        final_booster = es_booster
        refit_full = False
    else:
        # Phase 2: refit on the entire sup (no valid set, no callbacks).
        full_ds = lgb.Dataset(X_full, label=y_full)
        final_booster = lgb.train(
            params=params,
            train_set=full_ds,
            num_boost_round=K,
        )
        refit_full = True

    info = {
        "n_train": len(train_part),
        "n_valid": len(valid_part),
        "n_train_full": len(sup),
        "n_features": len(feat_cols),
        "iterations_used": K,
        "refit_full": refit_full,
        "train_min": train_part["timestamp"].min(),
        "train_max": train_max,
        "valid_min": valid_min,
        "valid_max": valid_part["timestamp"].max(),
    }
    return final_booster, info


def train_one(features: pd.DataFrame, level: str, q: float,
              train_cutoff: pd.Timestamp,
              num_iterations: int | None = None,
              verify_dates: bool = False) -> tuple[Path, dict]:
    sup = make_supervised(features, SINGLE_STEP_HORIZON)
    target_cutoff = train_cutoff - pd.Timedelta(hours=SINGLE_STEP_HORIZON)
    sup = sup[sup["timestamp"] < target_cutoff]
    if sup.empty:
        raise RuntimeError(f"No training rows before {target_cutoff} for {level}")

    t0 = time.time()
    booster, info = fit_with_early_stopping(
        sup=sup,
        q=q,
        iters=num_iterations,
        verify_dates=verify_dates,
        label=f"05/{level} q={q:.2f}",
    )
    elapsed = time.time() - t0

    out_path = model_filename(level, q)
    booster.save_model(str(out_path))

    summary = {
        "level": level,
        "quantile": q,
        "n_train_rows": info["n_train"],
        "n_valid_rows": info["n_valid"],
        "n_train_full_rows": info["n_train_full"],
        "n_features": info["n_features"],
        "iterations_used": info["iterations_used"],
        "refit_full": info["refit_full"],
        "training_seconds": round(elapsed, 2),
        "model_path": out_path.name,
    }
    return out_path, summary


# ====================================================================
# Recursive forecasting helper (used by 06, 08, 09)
# ====================================================================

def recursive_forecast_quantiles(
    boosters_by_q: dict,
    features: pd.DataFrame,
    issue_time: pd.Timestamp,
    n_steps: int = 168,
) -> pd.DataFrame:
    """Roll a quantile suite forward n_steps from issue_time.

    The median (q=0.5) is written back into the working frame's kwh_total
    so that subsequent steps see it as a lag input. Other quantiles are
    recorded but do not influence the recursion path.

    Args:
      boosters_by_q: dict mapping quantile (float) to a fitted lgb.Booster.
      features: the full feature DataFrame (calendar / weather pre-computed
                for every hour; lags will be re-derived per step).
      issue_time: the timestamp at which the forecast is issued. The first
                  predicted step is issue_time + 1h.
      n_steps: number of single-step iterations.

    Returns:
      Long-form DataFrame with columns: timestamp, horizon_h, quantile, y_pred.
    """
    if 0.5 not in boosters_by_q:
        raise ValueError("recursive_forecast_quantiles requires the median (q=0.5) booster")

    feat_cols = feature_columns(features)

    # Build a dict mapping timestamp → row position; access by integer index
    # is much faster than .loc[] in a per-row loop.
    work = (
        features.sort_values("timestamp")
        .reset_index(drop=True)
        .copy()
    )
    work["timestamp"] = pd.to_datetime(work["timestamp"])
    ts_to_idx = pd.Series(work.index.values, index=work["timestamp"].values)

    if issue_time not in ts_to_idx.index:
        raise ValueError(f"issue_time {issue_time} not present in features")

    kwh_arr = work["kwh_total"].to_numpy(dtype=float).copy()
    feat_arr = work[feat_cols].to_numpy(dtype=float).copy()
    feat_pos = {name: i for i, name in enumerate(feat_cols)}

    issue_idx = int(ts_to_idx.loc[issue_time])

    rows: list[dict] = []
    for step in range(1, n_steps + 1):
        target_t = issue_time + pd.Timedelta(hours=step)
        if target_t not in ts_to_idx.index:
            break
        target_idx = int(ts_to_idx.loc[target_t])
        # Model is trained as features(t) → y(t+1). To predict y at target_t
        # we feed it features at target_t - 1h (base_idx).
        base_idx = target_idx - 1

        if base_idx < 0:
            for q in boosters_by_q:
                rows.append(dict(
                    timestamp=target_t, horizon_h=step,
                    quantile=q, y_pred=np.nan,
                ))
            break

        # ---- Recompute lag features at base_idx -------------------------
        for lag in LAG_HOURS:
            col = f"kwh_lag_{lag}"
            if col not in feat_pos:
                continue
            src_idx = base_idx - lag
            feat_arr[base_idx, feat_pos[col]] = (
                kwh_arr[src_idx] if src_idx >= 0 else np.nan
            )

        for window in ROLLING_WINDOWS:
            col = f"kwh_rolling_mean_{window}"
            if col not in feat_pos:
                continue
            lo = max(0, base_idx - window)
            hi = base_idx  # exclusive: window covers [base_idx-window .. base_idx-1]
            slice_vals = kwh_arr[lo:hi]
            if len(slice_vals) < window:
                feat_arr[base_idx, feat_pos[col]] = np.nan
            else:
                feat_arr[base_idx, feat_pos[col]] = float(np.nanmean(slice_vals))

        x = feat_arr[base_idx:base_idx + 1, :]
        if np.isnan(x).any():
            for q in boosters_by_q:
                rows.append(dict(
                    timestamp=target_t, horizon_h=step,
                    quantile=q, y_pred=np.nan,
                ))
            # Without a valid prediction we can't propagate kwh_arr; bail.
            break

        preds: dict[float, float] = {}
        for q, booster in boosters_by_q.items():
            preds[q] = float(booster.predict(x)[0])

        for q, p in preds.items():
            rows.append(dict(
                timestamp=target_t, horizon_h=step,
                quantile=q, y_pred=p,
            ))

        # Store the median prediction at target_idx so it becomes a lag
        # input when computing features for later steps.
        kwh_arr[target_idx] = preds[0.5]

    return pd.DataFrame(rows)


# ====================================================================
# Main: train champion suite
# ====================================================================

def main():
    portfolio_path = C.CACHE_DIR / "features_portfolio.parquet"
    if not portfolio_path.exists():
        raise FileNotFoundError(
            f"Run code/03_feature_engineering.py first; missing {portfolio_path}"
        )

    train_cutoff = pd.Timestamp(C.VALIDATION_START)
    print(f"\nTraining recursive single-step champion with cutoff {train_cutoff}")
    print(f"  quantiles: {C.QUANTILES}")

    # Pipeline 1 forecasts the portfolio aggregate only. Per-segment
    # forecasting was evaluated and dropped after the reframe — see
    # config.FORECAST_LEVELS for the rationale.
    levels: list[tuple[str, Path]] = [
        (lv, C.CACHE_DIR / (
            "features_portfolio.parquet" if lv == "portfolio"
            else f"features_{lv}.parquet"
        ))
        for lv in C.FORECAST_LEVELS
    ]

    summaries: list[dict] = []
    total = len(levels) * len(C.QUANTILES)
    print(f"  total models: {total}")

    n_done = 0
    for level, path in levels:
        if not path.exists():
            print(f"  [skip] {level}: {path.name} not found")
            continue
        feats = pd.read_parquet(path)
        for q in C.QUANTILES:
            verify = (n_done == 0)  # print temporal-split dates once
            out_path, summary = train_one(
                feats, level, q, train_cutoff, verify_dates=verify
            )
            summaries.append(summary)
            n_done += 1
            print(f"  [{n_done:>2}/{total}] {level} q={q:.3f} "
                  f"es_train={summary['n_train_rows']:,} "
                  f"refit_full={summary['n_train_full_rows']:,} "
                  f"iters={summary['iterations_used']} "
                  f"time={summary['training_seconds']}s")

    summary_df = pd.DataFrame(summaries)
    out_csv = C.OUTPUT_DIR / "05_lgb_training_summary.csv"
    summary_df.to_csv(out_csv, index=False)
    print(f"\n  → {out_csv.name}: {len(summary_df)} model rows")
    print("[ok] LightGBM champion training complete.")


if __name__ == "__main__":
    C._print_summary()
    main()
