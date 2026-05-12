"""
06_walk_forward_validation.py — Rolling-window walk-forward evaluation
                                with recursive single-step LightGBM.

For each fold issued at time T_i (advancing every step_days), we:
  - take the prior TRAIN_WINDOW_DAYS as the training window
  - refit each model on that window
  - generate a full 168-hour forecast trajectory from T_i + 1h to T_i + 168h
  - record one row per (fold, model, level, quantile, target_h, target_t)

Models compared at each forecast step:
  - persistence (weekly seasonal: y(t) = y(t - 168h))
  - SARIMAX (exogenous block)
  - LightGBM recursive single-step (all 5 quantiles)

All evaluation is at the portfolio aggregate level. Per-segment forecasting
was evaluated and dropped — see config.FORECAST_LEVELS.

Sample-mode trade-offs (no-op in full mode):
  - WALK_STEP_DAYS overridden to 28 (~27 folds vs ~105 weekly)
  - SARIMAX trained on the last 90 days only
  - LightGBM num_iterations cut to 100

Outputs:
  output/06_walk_forward_results.csv         long format (one row per prediction)
  output/06_metrics_summary.csv              metrics × (model, level, horizon bucket)
  output/06_calibration_diagnostic.png       coverage curves
"""

from __future__ import annotations

import importlib.util
import time
import warnings
from pathlib import Path

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

import config as C


_HERE = Path(__file__).parent

_b = importlib.util.spec_from_file_location("baseline", _HERE / "04_train_baseline.py")
baseline = importlib.util.module_from_spec(_b)
_b.loader.exec_module(baseline)

_l = importlib.util.spec_from_file_location("train_lgb", _HERE / "05_train_lightgbm.py")
train_lgb = importlib.util.module_from_spec(_l)
_l.loader.exec_module(train_lgb)


HORIZON_BUCKETS = [
    # Aggregate buckets
    ("h<=24", 1, 24),
    ("h=25-72", 25, 72),
    ("h=73-168", 73, 168),
    # Single-point horizons used in the README headline
    ("h=1", 1, 1),
    ("h=24", 24, 24),
    ("h=72", 72, 72),
    ("h=168", 168, 168),
]
ALL_HORIZONS = (1, 24, 168)  # for native interval coverage plotting

INTERVAL_PAIRS = {
    "50%": (0.25, 0.75),
    "90%": (0.05, 0.95),
}


# ====================================================================
# Per-mode tunables
# ====================================================================

def get_run_params() -> dict:
    if C.DATA_MODE == "sample":
        return dict(
            step_days=28,
            sarimax_train_days=90,
            lgb_iters=100,
            seg_quantiles=[0.5],
        )
    return dict(
        step_days=C.WALK_STEP_DAYS,
        sarimax_train_days=C.TRAIN_WINDOW_DAYS,
        lgb_iters=C.LGB_PARAMS.get("num_iterations", 300),
        seg_quantiles=list(C.QUANTILES),
    )


# ====================================================================
# Fold generation
# ====================================================================

def generate_folds(start: pd.Timestamp, end: pd.Timestamp,
                   step_days: int, max_horizon_h: int) -> list[pd.Timestamp]:
    issues = []
    t = start
    cutoff = end - pd.Timedelta(hours=max_horizon_h)
    while t <= cutoff:
        issues.append(t)
        t = t + pd.Timedelta(days=step_days)
    return issues


# ====================================================================
# Model fitting per fold
# ====================================================================

# Module-level state for the walk-forward run:
#   _LGB_DATES_VERIFIED — toggled after the first fold prints its temporal split
#   _LGB_ITER_LOG       — appended once per fit; written to CSV at end
_LGB_DATES_VERIFIED = False
_LGB_ITER_LOG: list[dict] = []


def _train_lgb_quantile(features: pd.DataFrame, q: float,
                        train_start: pd.Timestamp, train_end: pd.Timestamp,
                        iters: int, level: str = "?") -> lgb.Booster:
    global _LGB_DATES_VERIFIED
    sup = train_lgb.make_supervised(features, train_lgb.SINGLE_STEP_HORIZON)
    target_cutoff = train_end - pd.Timedelta(hours=train_lgb.SINGLE_STEP_HORIZON)
    sup = sup[(sup["timestamp"] >= train_start) & (sup["timestamp"] < target_cutoff)]
    if sup.empty:
        raise RuntimeError(f"Empty training set for q={q} in [{train_start},{train_end})")

    booster, info = train_lgb.fit_with_early_stopping(
        sup=sup,
        q=q,
        iters=iters,
        verify_dates=not _LGB_DATES_VERIFIED,
        label=f"06/{level} q={q:.2f} fold@{train_end}",
    )
    if not _LGB_DATES_VERIFIED:
        _LGB_DATES_VERIFIED = True

    _LGB_ITER_LOG.append({
        "issue_time": train_end,
        "level": level,
        "quantile": q,
        "iterations_used": info["iterations_used"],
        "n_train": info["n_train"],
        "n_valid": info["n_valid"],
        "n_train_full": info["n_train_full"],
        "refit_full": info["refit_full"],
    })

    return booster


# ====================================================================
# Per-fold evaluation
# ====================================================================

def evaluate_fold(level: str, features: pd.DataFrame,
                  issue_time: pd.Timestamp,
                  train_window_days: int,
                  params: dict,
                  do_full_quantiles: bool) -> list[dict]:
    """All models for one (fold × level). Returns long-format rows."""
    rows: list[dict] = []

    train_end = issue_time
    train_start = issue_time - pd.Timedelta(days=train_window_days)

    feats_sorted = features.sort_values("timestamp").reset_index(drop=True)
    in_window = feats_sorted[
        (feats_sorted["timestamp"] >= train_start)
        & (feats_sorted["timestamp"] <= train_end)
    ]
    history = in_window[["timestamp", "kwh_total"]].dropna()

    quantiles_to_run = list(C.QUANTILES) if do_full_quantiles else params["seg_quantiles"]

    target_window = pd.date_range(
        issue_time + pd.Timedelta(hours=1),
        issue_time + pd.Timedelta(hours=max(ALL_HORIZONS) * 1),
        freq="h",
    )
    # Bound target_window by what's in features (last batch can be short)
    valid_end = features["timestamp"].max()
    target_window = target_window[target_window <= valid_end]
    if len(target_window) == 0:
        return rows

    actual_lookup = (
        features.set_index("timestamp")["kwh_total"]
        .reindex(target_window)
    )

    # ---- Persistence (weekly) ---------------------------------------------
    pers = baseline.persistence_forecast(
        history=history,
        predict_timestamps=target_window,
        seasonality_h=168,
    )
    for t, p in pers.items():
        h = int((t - issue_time) / pd.Timedelta(hours=1))
        rows.append(dict(
            level=level, model="persistence", quantile=np.nan, horizon_h=h,
            issue_time=issue_time, target_time=t,
            y_pred=float(p) if pd.notna(p) else np.nan,
            y_actual=float(actual_lookup.loc[t]) if pd.notna(actual_lookup.loc[t]) else np.nan,
        ))

    # ---- SARIMAX (portfolio only) -----------------------------------------
    if level == "portfolio":
        sarimax_train = in_window.tail(params["sarimax_train_days"] * 24)
        predict_df = (
            features.set_index("timestamp")
            .reindex(target_window)
            .reset_index()
            .rename(columns={"index": "timestamp"})
        )
        try:
            sx = baseline.sarimax_forecast(
                train_features=sarimax_train.dropna(subset=baseline.SARIMAX_EXOG),
                predict_features=predict_df.dropna(subset=baseline.SARIMAX_EXOG),
            )
        except Exception as e:
            tqdm.write(f"  [warn] SARIMAX failed at {issue_time}: {e}")
            sx = pd.Series(dtype=float)

        for t in target_window:
            h = int((t - issue_time) / pd.Timedelta(hours=1))
            pred = float(sx.loc[t]) if t in sx.index and pd.notna(sx.loc[t]) else np.nan
            rows.append(dict(
                level=level, model="sarimax", quantile=np.nan, horizon_h=h,
                issue_time=issue_time, target_time=t,
                y_pred=pred,
                y_actual=float(actual_lookup.loc[t]) if pd.notna(actual_lookup.loc[t]) else np.nan,
            ))

    # ---- LightGBM recursive ------------------------------------------------
    boosters: dict[float, lgb.Booster] = {}
    for q in quantiles_to_run:
        boosters[q] = _train_lgb_quantile(
            features, q, train_start, train_end, params["lgb_iters"], level=level
        )

    # If we're not training median we can't recurse (median feeds the recursion).
    # Add it transiently for prediction even if it's not in the requested set.
    if 0.5 not in boosters:
        boosters[0.5] = _train_lgb_quantile(
            features, 0.5, train_start, train_end, params["lgb_iters"], level=level
        )

    fc = train_lgb.recursive_forecast_quantiles(
        boosters_by_q=boosters,
        features=features,
        issue_time=issue_time,
        n_steps=max(ALL_HORIZONS),
    )

    for _, fr in fc.iterrows():
        if fr["quantile"] not in quantiles_to_run:
            continue  # filter out the median we added only to drive recursion
        t = fr["timestamp"]
        actual = (
            float(actual_lookup.loc[t])
            if t in actual_lookup.index and pd.notna(actual_lookup.loc[t])
            else np.nan
        )
        rows.append(dict(
            level=level, model="lightgbm", quantile=float(fr["quantile"]),
            horizon_h=int(fr["horizon_h"]),
            issue_time=issue_time, target_time=t,
            y_pred=float(fr["y_pred"]) if pd.notna(fr["y_pred"]) else np.nan,
            y_actual=actual,
        ))

    return rows


# ====================================================================
# Metrics
# ====================================================================

def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = (np.abs(y_true) > 1e-6) & np.isfinite(y_pred) & np.isfinite(y_true)
    if not mask.any():
        return np.nan
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = np.isfinite(y_pred) & np.isfinite(y_true)
    if not mask.any():
        return np.nan
    return float(np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2)))


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = np.isfinite(y_pred) & np.isfinite(y_true)
    if not mask.any():
        return np.nan
    denom = (np.abs(y_true[mask]) + np.abs(y_pred[mask])) / 2
    denom = np.where(denom < 1e-6, np.nan, denom)
    return float(np.nanmean(np.abs(y_true[mask] - y_pred[mask]) / denom))


def pinball(y_true: np.ndarray, y_pred: np.ndarray, q: float) -> float:
    mask = np.isfinite(y_pred) & np.isfinite(y_true)
    if not mask.any():
        return np.nan
    diff = y_true[mask] - y_pred[mask]
    return float(np.mean(np.maximum(q * diff, (q - 1) * diff)))


def summarize(long: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for bucket_label, lo, hi in HORIZON_BUCKETS:
        b = long[(long["horizon_h"] >= lo) & (long["horizon_h"] <= hi)]
        if b.empty:
            continue
        for (level, model), grp in b.groupby(["level", "model"]):
            if model == "lightgbm":
                for q, sub in grp.groupby("quantile"):
                    rows.append(dict(
                        bucket=bucket_label, level=level, model=model,
                        quantile=float(q),
                        n=len(sub),
                        mape=mape(sub["y_actual"].values, sub["y_pred"].values),
                        rmse=rmse(sub["y_actual"].values, sub["y_pred"].values),
                        smape=smape(sub["y_actual"].values, sub["y_pred"].values),
                        pinball=pinball(sub["y_actual"].values, sub["y_pred"].values, float(q)),
                    ))
                # Coverage at native intervals
                for label, (lo_q, hi_q) in INTERVAL_PAIRS.items():
                    lo_grp = grp[grp["quantile"] == lo_q].set_index(["issue_time", "target_time"])
                    hi_grp = grp[grp["quantile"] == hi_q].set_index(["issue_time", "target_time"])
                    if lo_grp.empty or hi_grp.empty:
                        continue
                    joined = lo_grp[["y_pred", "y_actual"]].join(
                        hi_grp[["y_pred"]], lsuffix="_lo", rsuffix="_hi", how="inner"
                    ).dropna()
                    if joined.empty:
                        continue
                    covered = (
                        (joined["y_actual"] >= joined["y_pred_lo"])
                        & (joined["y_actual"] <= joined["y_pred_hi"])
                    )
                    rows.append(dict(
                        bucket=bucket_label, level=level,
                        model="lightgbm_coverage", quantile=label,
                        n=len(joined),
                        mape=np.nan, rmse=np.nan, smape=np.nan,
                        pinball=np.nan,
                        coverage_rate=float(covered.mean()),
                    ))
            else:
                rows.append(dict(
                    bucket=bucket_label, level=level, model=model,
                    quantile=np.nan,
                    n=len(grp),
                    mape=mape(grp["y_actual"].values, grp["y_pred"].values),
                    rmse=rmse(grp["y_actual"].values, grp["y_pred"].values),
                    smape=smape(grp["y_actual"].values, grp["y_pred"].values),
                    pinball=np.nan,
                ))
    return pd.DataFrame(rows)


def plot_calibration(summary: pd.DataFrame, out_path: Path) -> None:
    cov = summary[summary["model"] == "lightgbm_coverage"].copy()
    if cov.empty:
        print("[warn] no coverage rows; skipping plot")
        return
    cov["nominal"] = cov["quantile"].map({"50%": 0.50, "90%": 0.90})
    cov = cov.dropna(subset=["nominal", "coverage_rate"])

    headline = cov[cov["bucket"].isin(["h=24", "h=168", "h<=24"])]

    fig, ax = plt.subplots(figsize=(7, 5))
    style = {"h<=24": "-", "h=24": "-", "h=25-72": "--", "h=73-168": ":"}
    color = {"h<=24": "#1f77b4", "h=24": "#1f77b4", "h=25-72": "#ff7f0e", "h=73-168": "#d62728"}

    for bucket, sub in headline.groupby("bucket"):
        # Average across levels for a clean line
        agg = sub.groupby("nominal")["coverage_rate"].mean().reset_index()
        ax.plot(agg["nominal"] * 100, agg["coverage_rate"] * 100,
                marker="o", linewidth=2,
                color=color.get(bucket, "gray"),
                linestyle=style.get(bucket, "-"),
                label=bucket)
    ax.plot([0, 100], [0, 100], "k--", alpha=0.4, label="perfect calibration")
    ax.set_xlim(40, 100)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Nominal interval (%)")
    ax.set_ylabel("Empirical coverage (%)")
    ax.set_title("LightGBM coverage vs nominal — recursive trajectory")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ====================================================================
# Main
# ====================================================================

def load_features(level: str) -> pd.DataFrame:
    if level == "portfolio":
        path = C.CACHE_DIR / "features_portfolio.parquet"
    else:
        path = C.CACHE_DIR / f"features_{level}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}; run 03_feature_engineering.py")
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def main():
    params = get_run_params()
    print(f"\nWalk-forward params: {params}")

    # Pipeline 1 evaluates only the portfolio aggregate. Per-segment
    # walk-forward was evaluated and dropped (see config.FORECAST_LEVELS).
    levels = list(C.FORECAST_LEVELS)
    feats = {lv: load_features(lv) for lv in levels}

    issues = generate_folds(
        start=pd.Timestamp(C.VALIDATION_START),
        end=pd.Timestamp(C.VALIDATION_END),
        step_days=params["step_days"],
        max_horizon_h=max(ALL_HORIZONS),
    )
    print(f"Folds: {len(issues)} (issue {issues[0]} → {issues[-1]})")

    all_rows: list[dict] = []
    t0 = time.time()
    for issue in tqdm(issues, desc="Folds"):
        for level in levels:
            full_q = (level == "portfolio")
            rows = evaluate_fold(
                level=level,
                features=feats[level],
                issue_time=issue,
                train_window_days=C.TRAIN_WINDOW_DAYS,
                params=params,
                do_full_quantiles=full_q,
            )
            all_rows.extend(rows)

    elapsed = time.time() - t0
    print(f"\nWalk-forward complete in {elapsed/60:.1f} min "
          f"({len(all_rows):,} prediction rows)")

    long = pd.DataFrame(all_rows)
    long_path = C.OUTPUT_DIR / "06_walk_forward_results.csv"
    long.to_csv(long_path, index=False)
    print(f"  → {long_path.name}")

    summary = summarize(long)
    summary_path = C.OUTPUT_DIR / "06_metrics_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"  → {summary_path.name}")

    if _LGB_ITER_LOG:
        iter_df = pd.DataFrame(_LGB_ITER_LOG)
        iter_path = C.OUTPUT_DIR / "06_lgb_iterations.csv"
        iter_df.to_csv(iter_path, index=False)
        used = iter_df["iterations_used"]
        print(f"  → {iter_path.name}: {len(iter_df)} fits, "
              f"iters min={int(used.min())} p50={int(used.median())} "
              f"p90={int(used.quantile(0.9))} max={int(used.max())}")

    plot_calibration(summary, C.OUTPUT_DIR / "06_calibration_diagnostic.png")
    print(f"  → 06_calibration_diagnostic.png")

    # At-a-glance: portfolio MAPE per bucket per model
    print("\nPortfolio MAPE by horizon bucket:")
    pf = summary[summary["level"] == "portfolio"]
    for bucket_label, _, _ in HORIZON_BUCKETS:
        for model in ("persistence", "sarimax", "lightgbm"):
            sub = pf[(pf["bucket"] == bucket_label) & (pf["model"] == model)]
            if model == "lightgbm":
                sub = sub[sub["quantile"] == 0.5]
            if sub.empty:
                continue
            r = sub.iloc[0]
            mape_pct = r["mape"] * 100 if pd.notna(r["mape"]) else np.nan
            print(f"  {bucket_label:<10s} {model:12s}  "
                  f"MAPE={mape_pct:6.2f}%  n={int(r['n'])}")

    print("\n[ok] Walk-forward validation complete.")


if __name__ == "__main__":
    C._print_summary()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
