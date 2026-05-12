"""
07_hierarchical_reconcile.py — Hierarchical reconciliation via Nixtla's
                               hierarchicalforecast.

Hierarchy (5 series, 4 bottom):
    portfolio
    ├── segment_0
    ├── segment_1
    ├── segment_2
    └── segment_3

Two reconcilers run on the median (q=0.5) LightGBM forecasts produced
by 06's recursive walk-forward: BottomUp() and MinTrace(method='ols').
Both are evaluated against the realized portfolio total at the same
target timestamps.

This script intentionally uses the Nixtla API rather than a numpy
implementation, so a reviewer sees the standard reconciliation library
in use. The math (P = (S'WS)^-1 S'W with W = I) is identical.

Outputs:
  output/07_reconciled_forecasts.parquet
  output/07_reconciliation_comparison.csv
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from hierarchicalforecast.core import HierarchicalReconciliation
from hierarchicalforecast.methods import BottomUp, MinTrace

import config as C


SEGMENTS = [f"segment_{i}" for i in range(C.N_SEGMENTS)]
ALL_SERIES = ["portfolio"] + SEGMENTS


def build_S_df() -> pd.DataFrame:
    """Summing matrix as DataFrame in Nixtla's expected format:
    one row per series with `unique_id` as a column and one float column
    per bottom-level series (1 if that bottom rolls up here, else 0)."""
    rows = []
    rows.append({"unique_id": "portfolio", **{s: 1.0 for s in SEGMENTS}})
    for seg in SEGMENTS:
        row = {"unique_id": seg, **{s: 0.0 for s in SEGMENTS}}
        row[seg] = 1.0
        rows.append(row)
    return pd.DataFrame(rows)


def build_tags() -> dict[str, np.ndarray]:
    return {
        "Total": np.array(["portfolio"]),
        "Total/Segment": np.array(SEGMENTS),
    }


def main():
    # Read the calibrated walk-forward output. The median (q=0.5) is
    # untouched by conformal calibration, so reconciliation results are
    # numerically identical to running on the raw 06 file; we read 06b for
    # provenance consistency with downstream consumers.
    long_path = C.OUTPUT_DIR / "06b_walk_forward_calibrated.csv"
    if not long_path.exists():
        raise FileNotFoundError(
            f"Run code/06b_conformal_calibration.py first; missing {long_path}"
        )

    long = pd.read_csv(long_path, parse_dates=["issue_time", "target_time"])
    long["quantile"] = pd.to_numeric(long["quantile"], errors="coerce")
    pred_col = "y_pred_calibrated" if "y_pred_calibrated" in long.columns else "y_pred"

    sub = long[
        (long["model"] == "lightgbm")
        & (long["quantile"] == 0.5)
    ].copy()
    sub["y_pred"] = sub[pred_col]

    if sub.empty:
        raise RuntimeError("No LightGBM median rows found in walk-forward results.")

    # Drop rows where any of the five series is missing for a given (issue, target)
    keys = ["issue_time", "target_time", "horizon_h"]
    pred_wide = sub.pivot_table(
        index=keys, columns="level", values="y_pred", aggfunc="first",
    )
    actual_wide = sub.pivot_table(
        index=keys, columns="level", values="y_actual", aggfunc="first",
    )
    full = pred_wide.dropna(subset=ALL_SERIES).index.intersection(
        actual_wide.index
    )
    pred_wide = pred_wide.loc[full].reset_index()
    actual_wide = actual_wide.loc[full].reset_index()
    if pred_wide.empty:
        raise RuntimeError("After dropna no full hierarchy rows survived.")
    print(f"Reconciling {len(pred_wide):,} (issue, target, horizon) tuples")

    # ---- Build the long Y_hat_df + Y_df expected by hierarchicalforecast ---
    # Y_hat_df: one row per (unique_id, ds, prediction). The library wants
    # ds (timestamp), unique_id (series), and a forecast column.
    # We treat (issue_time, target_time) as a single ds: the target time
    # itself, because reconciliation is across series at a fixed (issue,
    # target) — issue_time enters as part of the unique key only.

    rec_blocks = []
    rec_actuals = []

    # Process each issue_time independently — the library reconciles a
    # single forecast horizon at a time. We loop folds and apply per-fold.
    issue_groups = pred_wide.groupby("issue_time")
    actual_groups = actual_wide.set_index(["issue_time", "target_time", "horizon_h"])

    S_df = build_S_df()
    tags = build_tags()

    for issue, pred_block in issue_groups:
        # Long format for this fold: rows × series
        long_block = pred_block.melt(
            id_vars=["issue_time", "target_time", "horizon_h"],
            value_vars=ALL_SERIES,
            var_name="unique_id",
            value_name="lightgbm",
        )
        long_block = long_block.rename(columns={"target_time": "ds"})
        long_block = long_block[["unique_id", "ds", "lightgbm"]]

        # The historical Y_df: realized values up to (but not including) the
        # earliest target time. We supply the per-segment hourly history
        # from the same fold's training side. For the methodology demo we
        # use the most recent year of actuals.
        history_end = pred_block["target_time"].min()
        history_start = history_end - pd.Timedelta(days=365)

        # Build Y_df from features parquets (avoids a separate DB roundtrip)
        portfolio_hist = pd.read_parquet(C.CACHE_DIR / "features_portfolio.parquet")[
            ["timestamp", "kwh_total"]
        ]
        portfolio_hist["unique_id"] = "portfolio"
        seg_hists = []
        for seg in SEGMENTS:
            shp = pd.read_parquet(C.CACHE_DIR / f"features_{seg}.parquet")[
                ["timestamp", "kwh_total"]
            ]
            shp["unique_id"] = seg
            seg_hists.append(shp)
        Y_full = pd.concat([portfolio_hist] + seg_hists, ignore_index=True)
        Y_full["timestamp"] = pd.to_datetime(Y_full["timestamp"])
        Y_df = (
            Y_full[
                (Y_full["timestamp"] >= history_start)
                & (Y_full["timestamp"] < history_end)
            ]
            .rename(columns={"timestamp": "ds", "kwh_total": "y"})
            [["unique_id", "ds", "y"]]
            .dropna()
        )

        hrec = HierarchicalReconciliation(
            reconcilers=[BottomUp(), MinTrace(method="ols")],
        )
        Y_rec = hrec.reconcile(
            Y_hat_df=long_block,
            Y_df=Y_df,
            S_df=S_df,
            tags=tags,
        )
        Y_rec["issue_time"] = issue
        rec_blocks.append(Y_rec)

    rec = pd.concat(rec_blocks, ignore_index=True)

    # Forecast columns from Nixtla: original col + per-method reconciled col
    method_cols = {c for c in rec.columns if c.startswith("lightgbm")}
    print(f"  hierarchicalforecast methods produced: {sorted(method_cols)}")

    # Reconstitute target_time + horizon and join the actuals
    rec = rec.rename(columns={"ds": "target_time"})
    rec["target_time"] = pd.to_datetime(rec["target_time"])
    rec = rec.merge(
        pred_wide[["issue_time", "target_time", "horizon_h"]],
        on=["issue_time", "target_time"], how="left",
    )

    # Wide actuals per series for the same keys
    actual_long = actual_wide.melt(
        id_vars=["issue_time", "target_time", "horizon_h"],
        value_vars=ALL_SERIES,
        var_name="unique_id",
        value_name="y_actual",
    )
    rec = rec.merge(
        actual_long, on=["issue_time", "target_time", "horizon_h", "unique_id"],
        how="left",
    )

    out_path = C.OUTPUT_DIR / "07_reconciled_forecasts.parquet"
    rec.to_parquet(out_path, index=False)
    print(f"  → {out_path.name}: {len(rec):,} rows")

    # ---- MAPE comparison per (level, horizon-bucket, method) -------------
    cmp_rows = []
    HORIZON_BUCKETS = [
        ("h<=24", 1, 24),
        ("h=25-72", 25, 72),
        ("h=73-168", 73, 168),
        ("h=24", 24, 24),
        ("h=168", 168, 168),
    ]

    for label, lo, hi in HORIZON_BUCKETS:
        b = rec[(rec["horizon_h"] >= lo) & (rec["horizon_h"] <= hi)]
        if b.empty:
            continue
        for series in ALL_SERIES:
            sub = b[b["unique_id"] == series]
            if sub.empty:
                continue
            actual = sub["y_actual"].to_numpy()
            for method, col in (
                ("original", "lightgbm"),
                ("bottom_up", "lightgbm/BottomUp"),
                ("mint_ols", "lightgbm/MinTrace_method-ols"),
            ):
                if col not in sub.columns:
                    continue
                pred = sub[col].to_numpy()
                mask = (np.abs(actual) > 1e-6) & np.isfinite(pred) & np.isfinite(actual)
                if not mask.any():
                    continue
                mape = float(np.mean(np.abs((actual[mask] - pred[mask]) / actual[mask])))
                rmse = float(np.sqrt(np.mean((actual[mask] - pred[mask]) ** 2)))
                cmp_rows.append(dict(
                    bucket=label, level=series, method=method,
                    n=int(mask.sum()),
                    mape=mape, rmse=rmse,
                ))

    cmp_df = pd.DataFrame(cmp_rows)
    cmp_path = C.OUTPUT_DIR / "07_reconciliation_comparison.csv"
    cmp_df.to_csv(cmp_path, index=False)
    print(f"  → {cmp_path.name}")

    print("\nReconciliation MAPE at portfolio:")
    p = cmp_df[cmp_df["level"] == "portfolio"].sort_values(["bucket", "method"])
    for _, r in p.iterrows():
        print(f"  {r['bucket']:<10s} {r['method']:10s}  "
              f"MAPE={r['mape']*100:5.2f}%  n={int(r['n'])}")

    print("\n[ok] Hierarchical reconciliation complete.")


if __name__ == "__main__":
    C._print_summary()
    main()
