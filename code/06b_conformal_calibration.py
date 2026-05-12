"""
06b_conformal_calibration.py — Split conformal post-hoc calibration of the
LightGBM quantile forecasts produced by 06_walk_forward_validation.py.

The recursive single-step LightGBM quantile suite undercovers at every
horizon: at h=1 the 50% interval empirically covers ~0.41, at h=24 it
drops to ~0.18, and at h=168 to ~0.15. The cause is structural — the
median is fed back as a lag, smoothing the recursive trajectory relative
to realized variability. Retraining alone cannot fix this; we calibrate
post-hoc.

Two design choices distinguish this implementation from textbook split
conformal:

  1) INTERLEAVED CAL/TEST SPLIT. The validation period straddles the
     June 2021 2.0TD tariff reform, which materially changed household
     consumption patterns. A naive temporal mid-split puts pre-reform
     issues in calibration and post-reform in test, breaking the
     exchangeability assumption that conformal prediction relies on.
     We instead split issue_times by parity of their sorted index — even
     index → cal, odd → test — so both regimes are represented equally
     in each half.

  2) HORIZON-BUCKET POOLING. Per-horizon adjustments computed on ~51
     calibration issues are noisy at the tails (the 0.05 quantile of 51
     samples is essentially the 3rd-smallest value). We pool residuals
     across horizons within three buckets:

         h<=24  ,  h=25-72  ,  h=73-168

     which raises the cal sample size per (level, bucket, quantile) to
     roughly 3,500 / 12,000 / 24,000. The trade-off — losing per-horizon
     resolution — is consistent with how downstream metrics are reported:
     the README headline coverages are bucket-level.

For each (bucket, quantile q ∈ {0.05, 0.25, 0.75, 0.95}) we set

    δ(bucket, q) = quantile_q( y_real - y_pred_q )    on calibration

and apply

    y_pred_q_calibrated = y_pred_q + δ(bucket(h), q)

The median (q=0.5) is left untouched, so median MAPE is preserved by
construction. Under approximate exchangeability between cal and test
residuals (achieved by interleaving), the empirical coverage of
[q_lo + δ_lo, q_hi + δ_hi] on the test set converges to nominal.

The schema still carries a `level` column for backward compatibility, but
Pipeline 1 only ever holds level=portfolio (per-segment forecasting was
dropped — see config.FORECAST_LEVELS).

Outputs:
  output/06b_conformal_adjustments.csv      level, bucket, quantile, adjustment, n_cal
  output/06b_walk_forward_calibrated.csv    long format including y_pred_calibrated
  prints per-bucket coverage (raw vs cal) and gate-check
"""

from __future__ import annotations

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config as C


ADJUST_QUANTILES = (0.05, 0.25, 0.75, 0.95)

# Pool buckets: residuals from every horizon inside a pool share one δ.
POOL_BUCKETS = [
    ("h<=24", 1, 24),
    ("h=25-72", 25, 72),
    ("h=73-168", 73, 168),
]

# Buckets reported at every pass through 06b. The gate is evaluated on the
# three POOL_BUCKETS only — that is the resolution at which calibration is
# computed (one δ per pool-bucket × quantile × level), and applying the
# gate at any finer granularity is incompatible with bucket pooling. The
# single-horizon cells (h=1, h=24, h=72, h=168) are reported for diagnostic
# visibility only; they will exhibit drift inside their pool by design and
# are not part of the acceptance contract.
REPORT_BUCKETS = [
    ("h=1", 1, 1),
    ("h<=24", 1, 24),
    ("h=24", 24, 24),
    ("h=25-72", 25, 72),
    ("h=72", 72, 72),
    ("h=73-168", 73, 168),
    ("h=168", 168, 168),
]
GATE_BUCKET_LABELS = {b for b, _, _ in POOL_BUCKETS}
COV50_BAND = (0.45, 0.55)
COV90_BAND = (0.85, 0.95)


def assign_pool_bucket(h: int) -> str:
    for label, lo, hi in POOL_BUCKETS:
        if lo <= h <= hi:
            return label
    raise ValueError(f"horizon_h={h} not covered by POOL_BUCKETS")


def split_interleaved(issues: list) -> tuple[set, set]:
    """Sort issue_times then alternate even/odd indices. Distributes any
    regime shift evenly between cal and test, restoring approximate
    exchangeability — required because the validation window crosses the
    2021-06-01 tariff reform."""
    s = sorted(pd.Timestamp(t) for t in set(issues))
    return set(s[0::2]), set(s[1::2])


def compute_adjustments(cal: pd.DataFrame) -> pd.DataFrame:
    rows = []
    cal = cal.copy()
    cal["pool_bucket"] = cal["horizon_h"].map(assign_pool_bucket)
    sub = cal[cal["quantile"].isin(ADJUST_QUANTILES)].dropna(
        subset=["y_pred", "y_actual"]
    )
    for (level, bucket, q), grp in sub.groupby(["level", "pool_bucket", "quantile"]):
        residuals = (grp["y_actual"] - grp["y_pred"]).to_numpy()
        delta = float(np.quantile(residuals, q))
        rows.append(dict(
            level=level, bucket=bucket,
            quantile=float(q), adjustment=delta,
            n_cal=int(len(residuals)),
        ))
    return pd.DataFrame(rows)


def apply_adjustments(long_lgb: pd.DataFrame, adj: pd.DataFrame) -> pd.DataFrame:
    df = long_lgb.copy()
    df["pool_bucket"] = df["horizon_h"].map(assign_pool_bucket)
    out = df.merge(
        adj[["level", "bucket", "quantile", "adjustment"]]
            .rename(columns={"bucket": "pool_bucket"}),
        on=["level", "pool_bucket", "quantile"], how="left",
    )
    out["adjustment"] = out["adjustment"].fillna(0.0)
    out["y_pred_calibrated"] = out["y_pred"] + out["adjustment"]
    return out


def coverage_frame(test_long: pd.DataFrame, level: str | None = None) -> pd.DataFrame:
    """Wide per (level, issue_time, target_time, horizon_h) with raw + cal bands."""
    df = test_long if level is None else test_long[test_long["level"] == level]
    keys = ["level", "issue_time", "target_time", "horizon_h"]

    raw = df.pivot_table(index=keys, columns="quantile",
                         values="y_pred", aggfunc="first")
    cal = df.pivot_table(index=keys, columns="quantile",
                         values="y_pred_calibrated", aggfunc="first")
    actuals = df.groupby(keys)["y_actual"].first()

    raw.columns = [f"raw_{q}" for q in raw.columns]
    cal.columns = [f"cal_{q}" for q in cal.columns]

    w = pd.concat([raw, cal, actuals.rename("actual")], axis=1).reset_index()
    needed = [f"raw_{q}" for q in (0.05, 0.25, 0.75, 0.95)] + \
             [f"cal_{q}" for q in (0.05, 0.25, 0.75, 0.95)] + ["actual"]
    w = w.dropna(subset=needed)

    w["raw_in50"] = (w["actual"] >= w["raw_0.25"]) & (w["actual"] <= w["raw_0.75"])
    w["raw_in90"] = (w["actual"] >= w["raw_0.05"]) & (w["actual"] <= w["raw_0.95"])
    w["cal_in50"] = (w["actual"] >= w["cal_0.25"]) & (w["actual"] <= w["cal_0.75"])
    w["cal_in90"] = (w["actual"] >= w["cal_0.05"]) & (w["actual"] <= w["cal_0.95"])
    return w


def bucket_summary(w: pd.DataFrame, level: str) -> pd.DataFrame:
    rows = []
    for label, lo, hi in REPORT_BUCKETS:
        sub = w[(w["horizon_h"] >= lo) & (w["horizon_h"] <= hi)]
        if sub.empty:
            continue
        rows.append(dict(
            level=level, bucket=label, n=int(len(sub)),
            raw_cov50=float(sub["raw_in50"].mean()),
            cal_cov50=float(sub["cal_in50"].mean()),
            raw_cov90=float(sub["raw_in90"].mean()),
            cal_cov90=float(sub["cal_in90"].mean()),
        ))
    return pd.DataFrame(rows)


def main() -> int:
    long_path = C.OUTPUT_DIR / "06_walk_forward_results.csv"
    print(f"\nReading {long_path.name}")
    long = pd.read_csv(long_path)
    long["issue_time"] = pd.to_datetime(long["issue_time"])
    long["target_time"] = pd.to_datetime(long["target_time"])

    lgb_long = long[long["model"] == "lightgbm"].copy()
    print(f"  rows: {len(long):,} total / {len(lgb_long):,} lightgbm")

    issues = sorted(lgb_long["issue_time"].unique())
    cal_issues, test_issues = split_interleaved(issues)
    print(f"  issues: {len(issues)} → cal={len(cal_issues)} / test={len(test_issues)} (interleaved)")
    print(f"  cal range  : {min(cal_issues)} → {max(cal_issues)}")
    print(f"  test range : {min(test_issues)} → {max(test_issues)}")

    cal = lgb_long[lgb_long["issue_time"].isin(cal_issues)]
    test = lgb_long[lgb_long["issue_time"].isin(test_issues)]

    print("\nComputing per-(level, bucket, quantile) adjustments (pooled across horizons)…")
    adj = compute_adjustments(cal)
    print(f"  adjustments: {len(adj):,} rows  "
          f"(levels={adj['level'].nunique()}, "
          f"buckets={adj['bucket'].nunique()}, "
          f"quantiles={adj['quantile'].nunique()})")
    print("  δ by (bucket, quantile) — median across levels:")
    for bucket, _, _ in POOL_BUCKETS:
        for q in ADJUST_QUANTILES:
            sub = adj[(adj["bucket"] == bucket) & (adj["quantile"] == q)]
            n_cal = int(sub["n_cal"].iloc[0]) if len(sub) else 0
            print(f"    {bucket:<10s} q={q:.2f}  median δ = {sub['adjustment'].median():+8.2f}  "
                  f"n_cal/level = {n_cal:,}")

    adj_path = C.OUTPUT_DIR / "06b_conformal_adjustments.csv"
    adj.to_csv(adj_path, index=False)
    print(f"  → {adj_path.name}")

    # Long-format calibrated outputs (full window: cal + test). Non-LGB rows
    # carry through with adjustment=NaN and y_pred_calibrated == y_pred.
    full_lgb_cal = apply_adjustments(lgb_long, adj)
    other = long[long["model"] != "lightgbm"].copy()
    other["adjustment"] = np.nan
    other["y_pred_calibrated"] = other["y_pred"]
    final = pd.concat([full_lgb_cal, other], ignore_index=True)
    out_path = C.OUTPUT_DIR / "06b_walk_forward_calibrated.csv"
    final.to_csv(out_path, index=False)
    print(f"  → {out_path.name}: {len(final):,} rows")

    # Coverage assessment on the test split only ---------------------------
    test_cal = apply_adjustments(test, adj)
    levels = sorted(test_cal["level"].unique())

    print("\n=== Per-bucket coverage on test split ===")
    bucket_rows = []
    for level in levels:
        w = coverage_frame(test_cal, level=level)
        bs = bucket_summary(w, level=level)
        bucket_rows.append(bs)
        print(f"\n  {level}")
        print(bs[["bucket", "n", "raw_cov50", "cal_cov50",
                  "raw_cov90", "cal_cov90"]]
              .to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # The previous "ALL pooled across levels" block is omitted: with the
    # Pipeline 1 reframe to portfolio-only forecasting there is exactly one
    # level, so the pooled row would duplicate the portfolio row verbatim.
    pooled_bs = bucket_summary(coverage_frame(test_cal, level="portfolio"),
                               level="portfolio")

    # Per-horizon table for portfolio (every 24h)
    print("\n=== Per-horizon coverage (portfolio, sample horizons) ===")
    pf = coverage_frame(test_cal, level="portfolio")
    per_h = (pf.groupby("horizon_h")
               .agg(n=("actual", "size"),
                    raw_cov50=("raw_in50", "mean"),
                    cal_cov50=("cal_in50", "mean"),
                    raw_cov90=("raw_in90", "mean"),
                    cal_cov90=("cal_in90", "mean"))
               .reset_index())
    sample_h = per_h[per_h["horizon_h"].isin([1, 6, 12, 24, 48, 72, 96, 120, 144, 168])]
    print(sample_h.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # Median MAPE: should be unchanged by construction
    print("\n=== Median MAPE before/after (must match — q=0.5 untouched) ===")
    median = test_cal[test_cal["quantile"] == 0.5].dropna(
        subset=["y_pred", "y_actual"])
    median = median[median["y_actual"].abs() > 1e-6]
    raw_mape = float((np.abs(
        (median["y_actual"] - median["y_pred"]) / median["y_actual"]).mean())) * 100
    cal_mape = float((np.abs(
        (median["y_actual"] - median["y_pred_calibrated"]) / median["y_actual"]
        ).mean())) * 100
    print(f"  raw MAPE = {raw_mape:.4f}%   cal MAPE = {cal_mape:.4f}%   "
          f"(Δ = {cal_mape - raw_mape:+.4f})")

    # Gate — evaluated only on pool buckets, the resolution at which
    # adjustments are computed.
    summary = pd.concat(bucket_rows, ignore_index=True)
    gate_summary = summary[summary["bucket"].isin(GATE_BUCKET_LABELS)]
    print("\n=== Gate evaluation (pool-bucket level) ===")
    print(f"  cov_50 band: [{COV50_BAND[0]}, {COV50_BAND[1]}]")
    print(f"  cov_90 band: [{COV90_BAND[0]}, {COV90_BAND[1]}]")
    print(f"  cells: {len(gate_summary)} (= {gate_summary['level'].nunique()} levels × {len(GATE_BUCKET_LABELS)} pool buckets)")
    fails = []
    for _, r in gate_summary.iterrows():
        ok50 = COV50_BAND[0] <= r["cal_cov50"] <= COV50_BAND[1]
        ok90 = COV90_BAND[0] <= r["cal_cov90"] <= COV90_BAND[1]
        if not (ok50 and ok90):
            fails.append((r["level"], r["bucket"],
                          r["cal_cov50"], r["cal_cov90"]))
    if fails:
        print(f"\n  [FAIL] {len(fails)} (level, pool-bucket) outside the bands:")
        for lvl, bucket, c50, c90 in fails:
            print(f"    {lvl:<12s} {bucket:<10s}  cov50={c50:.3f}  cov90={c90:.3f}")
        return 1
    print(f"  [OK] all {len(gate_summary)} (level, pool-bucket) cells satisfy both bands.")

    # Persist the gate cells for downstream consumption (Streamlit calibration page).
    gate_path = C.OUTPUT_DIR / "06b_calibration_gate.csv"
    gate_summary.to_csv(gate_path, index=False)
    print(f"  → {gate_path.name}")

    # Full per-(level, bucket) raw/calibrated coverage summary, including
    # the diagnostic single-horizon cells. Consumed by the dashboard and 10.
    full_summary = summary[["level", "bucket", "n",
                            "raw_cov50", "cal_cov50",
                            "raw_cov90", "cal_cov90"]]
    summary_path = C.OUTPUT_DIR / "06b_calibration_summary.csv"
    full_summary.to_csv(summary_path, index=False)
    print(f"  → {summary_path.name}")

    # Calibration curve (raw vs calibrated, pooled across levels)
    plot_path = C.OUTPUT_DIR / "06b_calibration_curve.png"
    pooled_only = pooled_bs[pooled_bs["bucket"].isin(GATE_BUCKET_LABELS)]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot([0, 100], [0, 100], "k--", alpha=0.4, label="perfect calibration")
    palette = {"h<=24": "#1f77b4", "h=25-72": "#ff7f0e", "h=73-168": "#d62728"}
    for _, r in pooled_only.iterrows():
        b = r["bucket"]
        ax.plot([50, 90], [r["raw_cov50"] * 100, r["raw_cov90"] * 100],
                marker="o", linestyle=":", color=palette.get(b, "gray"),
                alpha=0.55, label=f"{b} raw")
        ax.plot([50, 90], [r["cal_cov50"] * 100, r["cal_cov90"] * 100],
                marker="o", linestyle="-", color=palette.get(b, "gray"),
                linewidth=2.2, label=f"{b} calibrated")
    # Gate band overlay
    ax.axhspan(45, 55, xmin=(50 - 40) / 60, xmax=(50 - 40) / 60 + 1e-3, alpha=0)  # no-op anchor
    ax.fill_between([49, 51], 45, 55, alpha=0.18, color="green", label="cov50 gate")
    ax.fill_between([89, 91], 85, 95, alpha=0.18, color="green")
    ax.set_xlim(40, 100)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Nominal interval (%)")
    ax.set_ylabel("Empirical coverage (%)")
    ax.set_title("Conformal calibration — pool buckets, raw vs calibrated")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=120)
    plt.close(fig)
    print(f"  → {plot_path.name}")

    return 0


if __name__ == "__main__":
    C._print_summary()
    sys.exit(main())
