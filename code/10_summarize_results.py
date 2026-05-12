"""
10_summarize_results.py — Final consolidation step.

Reads outputs from scripts 02 (segmentation), 06 / 06b (forecasting) and
the champion artefacts; produces the artefacts the README headline table
and the Streamlit dashboard consume:

  output/10_headline_metrics.csv             headline table input
  output/10_feature_importance_grouped.csv   importance by category
  output/10_calibration_curve.png            coverage vs nominal interval
  output/10_economic_interpretation.md       narrative on the top features
  output/10_results_narrative.md             paragraph-form summary
  output/10_pipeline2_summary.md             segmentation characterization

Pipeline 1 is portfolio-only forecasting; Pipeline 2 is behavioural
segmentation for tariff design (consumed at characterization time, not
forecasted). Hierarchical reconciliation (script 07) was evaluated and
dropped during the reframe; this script no longer reads its outputs.

This script is read-only with respect to models — it consolidates, it
does not retrain.
"""

from __future__ import annotations

import importlib.util
from collections import defaultdict
from pathlib import Path

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config as C


_HERE = Path(__file__).parent
_l = importlib.util.spec_from_file_location("train_lgb", _HERE / "05_train_lightgbm.py")
train_lgb = importlib.util.module_from_spec(_l)
_l.loader.exec_module(train_lgb)


# ====================================================================
# Headline metrics
# ====================================================================

def headline_metrics(metrics_summary: pd.DataFrame,
                     calibration_summary: pd.DataFrame | None) -> pd.DataFrame:
    """Pull the rows needed for the README headline table.

    Persistence and LightGBM median MAPE at each headline horizon, plus
    pinball loss averaged over quantiles, plus coverage at the two native
    intervals. All at the portfolio level. Coverage figures are reported at
    the pool-bucket level (h<=24 / h=25-72 / h=73-168) — that's the
    resolution at which conformal calibration is defined and gated. When
    06b output is available, both raw and calibrated coverage are reported.
    """
    pf = metrics_summary[metrics_summary["level"] == "portfolio"]

    rows = []
    for bucket_label, h_str in [("h=24", "24h"), ("h=168", "168h")]:
        pers = pf[(pf["model"] == "persistence") & (pf["bucket"] == bucket_label)]["mape"]
        sx = pf[(pf["model"] == "sarimax") & (pf["bucket"] == bucket_label)]["mape"]
        lgb_med = pf[
            (pf["model"] == "lightgbm")
            & (pf["bucket"] == bucket_label)
            & (pf["quantile_num"] == 0.5)
        ]["mape"]

        rows.append(dict(metric=f"MAPE {h_str} ahead",
                         lightgbm=float(lgb_med.iloc[0]) if not lgb_med.empty else np.nan,
                         sarimax=float(sx.iloc[0]) if not sx.empty else np.nan,
                         persistence=float(pers.iloc[0]) if not pers.empty else np.nan))

        pin = pf[
            (pf["model"] == "lightgbm")
            & (pf["bucket"] == bucket_label)
        ]["pinball"].dropna()
        rows.append(dict(metric=f"Pinball loss avg ({h_str})",
                         lightgbm=float(pin.mean()) if not pin.empty else np.nan,
                         sarimax=np.nan, persistence=np.nan))

    # Calibrated coverage at the pool-bucket level (portfolio).
    if calibration_summary is not None and not calibration_summary.empty:
        pf_cal = calibration_summary[calibration_summary["level"] == "portfolio"]
        for pool_bucket in ("h<=24", "h=25-72", "h=73-168"):
            row = pf_cal[pf_cal["bucket"] == pool_bucket]
            if row.empty:
                continue
            r = row.iloc[0]
            rows.append(dict(
                metric=f"Coverage @ 50% nominal ({pool_bucket}) — raw / calibrated",
                lightgbm=f"{r['raw_cov50']:.3f} → {r['cal_cov50']:.3f}",
                sarimax=np.nan, persistence=np.nan,
            ))
            rows.append(dict(
                metric=f"Coverage @ 90% nominal ({pool_bucket}) — raw / calibrated",
                lightgbm=f"{r['raw_cov90']:.3f} → {r['cal_cov90']:.3f}",
                sarimax=np.nan, persistence=np.nan,
            ))

    return pd.DataFrame(rows)


# ====================================================================
# Feature importance
# ====================================================================

FEATURE_GROUPS = {
    "calendar": {"hour_of_day", "day_of_week", "is_weekend", "is_holiday",
                 "month", "day_of_year", "hour_sin", "hour_cos",
                 "doy_sin", "doy_cos"},
    "lag": {"kwh_lag_1", "kwh_lag_24", "kwh_lag_168",
            "kwh_rolling_mean_24", "kwh_rolling_mean_168"},
    "weather": {"temperature_weighted", "humidity_weighted",
                "precipitation_weighted", "wind_weighted",
                "temp_sq", "heating_demand", "cooling_demand"},
    "regime": {"post_tariff_reform"},
}


def grouped_importance(level: str = "portfolio") -> pd.DataFrame:
    """Aggregate gain importance across all (quantile) champion models
    for one level, grouped by feature category. With recursive single-step,
    there is one model per (level, quantile) — no horizon dimension."""
    totals: dict[str, float] = defaultdict(float)
    n_models = 0
    for q in C.QUANTILES:
        path = train_lgb.model_filename(level, q)
        if not path.exists():
            continue
        booster = lgb.Booster(model_file=str(path))
        imp = pd.Series(
            booster.feature_importance(importance_type="gain"),
            index=booster.feature_name(),
        )
        n_models += 1
        for name, val in imp.items():
            group = next((g for g, members in FEATURE_GROUPS.items()
                          if name in members), "other")
            totals[group] += float(val)

    if not totals:
        return pd.DataFrame(columns=["group", "total_gain", "share"])

    total = sum(totals.values()) or 1.0
    rows = sorted(
        [{"group": g, "total_gain": v, "share": v / total} for g, v in totals.items()],
        key=lambda r: -r["share"],
    )
    df = pd.DataFrame(rows)
    df["n_models_aggregated"] = n_models
    return df


# ====================================================================
# Calibration plot (final, polished version)
# ====================================================================

def calibration_plot(calibration_summary: pd.DataFrame | None, out_path: Path) -> None:
    """Raw vs conformal-calibrated coverage on the test split, pool buckets."""
    if calibration_summary is None or calibration_summary.empty:
        print("[warn] no calibration summary; skipping plot")
        return
    pool_buckets = ("h<=24", "h=25-72", "h=73-168")
    palette = {"h<=24": "#1f77b4", "h=25-72": "#ff7f0e", "h=73-168": "#d62728"}

    fig, ax = plt.subplots(figsize=(7, 5))
    pf = calibration_summary[
        (calibration_summary["level"] == "portfolio")
        & (calibration_summary["bucket"].isin(pool_buckets))
    ]
    for _, r in pf.iterrows():
        ax.plot([50, 90], [r["raw_cov50"] * 100, r["raw_cov90"] * 100],
                marker="o", linestyle=":", color=palette[r["bucket"]],
                alpha=0.55, label=f"{r['bucket']} raw")
        ax.plot([50, 90], [r["cal_cov50"] * 100, r["cal_cov90"] * 100],
                marker="o", linestyle="-", color=palette[r["bucket"]],
                linewidth=2.2, label=f"{r['bucket']} calibrated")
    ax.plot([0, 100], [0, 100], "k--", alpha=0.5, label="perfect calibration")
    ax.fill_between([49, 51], 45, 55, alpha=0.18, color="green", label="cov50 gate")
    ax.fill_between([89, 91], 85, 95, alpha=0.18, color="green")
    ax.set_xlim(40, 100)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Nominal interval (%)")
    ax.set_ylabel("Empirical coverage (%)")
    ax.set_title("Calibration — raw recursive bands vs conformal-corrected")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ====================================================================
# Narrative
# ====================================================================

def write_economic_interpretation(imp_df: pd.DataFrame, out_path: Path) -> None:
    if imp_df.empty:
        out_path.write_text("# Feature importance unavailable\n", encoding="utf-8")
        return
    top = imp_df.iloc[0]
    second = imp_df.iloc[1] if len(imp_df) > 1 else None

    body = [
        "# Economic interpretation of feature importance",
        "",
        f"Aggregated across {int(imp_df['n_models_aggregated'].iloc[0])} champion models "
        "(per horizon × quantile, portfolio level).",
        "",
        "## Group share of total gain",
        "",
        "| Group | Share | Total gain |",
        "|---|---:|---:|",
    ]
    for _, r in imp_df.iterrows():
        body.append(f"| {r['group']} | {r['share']*100:5.1f}% | {r['total_gain']:.0f} |")

    body += ["", "## Reading the table"]
    body.append(
        f"- **{top['group']}** features explain {top['share']*100:.1f}% of the "
        "total gain. In a residential portfolio this is consistent with the "
        "intra-day and intra-week patterns dominating short-term variability "
        "once the long-term level is stable."
    )
    if second is not None:
        body.append(
            f"- **{second['group']}** is next ({second['share']*100:.1f}%); these "
            "features capture systematic shifts the model needs to track "
            "between average days and unusual ones."
        )
    body.append(
        "- The remaining groups together provide secondary signal that lifts "
        "the model above the persistence baseline. These shares are computed "
        "on synthetic sample data; the full-mode run is expected to reweight "
        "weather upward as real Spanish heating/cooling demand kicks in."
    )

    out_path.write_text("\n".join(body) + "\n", encoding="utf-8")


def write_results_narrative(headline: pd.DataFrame, out_path: Path) -> None:
    lines = ["# Results narrative", ""]
    if headline.empty:
        lines.append("No headline metrics available.")
    else:
        lgb24 = headline[headline["metric"] == "MAPE 24h ahead"]
        lgb168 = headline[headline["metric"] == "MAPE 168h ahead"]
        if not lgb24.empty and not lgb168.empty:
            r24 = lgb24.iloc[0]
            r168 = lgb168.iloc[0]
            lines.append(
                f"At horizon 24h, the LightGBM median forecast achieves "
                f"MAPE {r24['lightgbm']*100:.2f}% versus persistence "
                f"{r24['persistence']*100:.2f}% and SARIMAX "
                f"{r24['sarimax']*100:.2f}%."
            )
            lines.append(
                f"At horizon 168h, the same model achieves MAPE "
                f"{r168['lightgbm']*100:.2f}% versus persistence "
                f"{r168['persistence']*100:.2f}% and SARIMAX "
                f"{r168['sarimax']*100:.2f}%."
            )
            lines.append("")

    lines.append(
        "All metrics come from walk-forward validation: at each fold the "
        "model only sees data strictly prior to the prediction window. This "
        "is the same evaluation protocol that supply-side forecasting teams "
        "use because random k-fold leaks future information into the past "
        "and inflates apparent accuracy."
    )

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_pipeline2_summary(out_path: Path) -> None:
    """Render the Pipeline 2 segmentation characterization as Markdown.

    Reads output/02_segment_characterization.csv (produced by
    code/02_segment_households.py) and renders a short Markdown block
    describing the three behavioural archetypes.
    """
    char_path = C.OUTPUT_DIR / "02_segment_characterization.csv"
    if not char_path.exists():
        out_path.write_text(
            "# Pipeline 2 — segmentation\n\n"
            "Characterization not available (run code/02_segment_households.py).\n",
            encoding="utf-8",
        )
        return

    char = pd.read_csv(char_path)
    body = [
        "# Pipeline 2 — behavioural segmentation",
        "",
        f"k-means on standardized pre-2020 daily load shapes recovers "
        f"**{len(char)} archetypes** across {int(char['n_households'].sum()):,} "
        "households. The cluster structure is weak by design (silhouette "
        "< 0.20 across k ∈ {2..8}; see output/02_ksweep_diagnostic/) — "
        "residential consumption is a continuum, not a small discrete set "
        "of types. k=3 is the metric-supported choice (silhouette peak, "
        "elbow inflection at k=3) and the labels below are operational "
        "tags assigned by amplitude/level heuristics, not statistical "
        "discoveries.",
        "",
        "## Archetypes",
        "",
        "| Segment | Label | Households | % hh | % kWh | mean kWh/hh | peak h | amplitude (σ) |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in char.sort_values("segment").iterrows():
        body.append(
            f"| {int(r['segment'])} | `{r['label']}` | "
            f"{int(r['n_households']):,} | "
            f"{r['pct_households']:.1f}% | "
            f"{r['pct_total_kwh']:.1f}% | "
            f"{r['mean_hh_total_kwh']:,.0f} | "
            f"{int(r['peak_hour']):02d}:00 | "
            f"{r['amplitude_std']:.2f} |"
        )
    body += [
        "",
        "Centroid 24-hour profiles live in `output/02_segment_centroids.csv`; "
        "the overlay plot is at `output/02_segment_profiles.png`.",
    ]
    out_path.write_text("\n".join(body) + "\n", encoding="utf-8")


# ====================================================================
# Main
# ====================================================================

def main():
    summary_path = C.OUTPUT_DIR / "06_metrics_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(
            f"Run code/06_walk_forward_validation.py first; missing {summary_path}"
        )
    metrics_summary = pd.read_csv(summary_path)
    # 06 mixes float quantiles (LightGBM rows) and string interval labels
    # (coverage rows) in the same column, so pandas reads it as object.
    # Coerce numeric where possible; keep strings ("50%", "90%") as-is.
    metrics_summary["quantile_num"] = pd.to_numeric(
        metrics_summary["quantile"], errors="coerce"
    )

    cal_summary_path = C.OUTPUT_DIR / "06b_calibration_summary.csv"
    calibration_summary = (
        pd.read_csv(cal_summary_path) if cal_summary_path.exists() else None
    )

    headline = headline_metrics(metrics_summary, calibration_summary)
    headline_path = C.OUTPUT_DIR / "10_headline_metrics.csv"
    headline.to_csv(headline_path, index=False)
    print(f"  → {headline_path.name}")

    imp_df = grouped_importance("portfolio")
    imp_path = C.OUTPUT_DIR / "10_feature_importance_grouped.csv"
    imp_df.to_csv(imp_path, index=False)
    print(f"  → {imp_path.name}")

    plot_path = C.OUTPUT_DIR / "10_calibration_curve.png"
    calibration_plot(calibration_summary, plot_path)
    print(f"  → {plot_path.name}")

    econ_path = C.OUTPUT_DIR / "10_economic_interpretation.md"
    write_economic_interpretation(imp_df, econ_path)
    print(f"  → {econ_path.name}")

    narr_path = C.OUTPUT_DIR / "10_results_narrative.md"
    write_results_narrative(headline, narr_path)
    print(f"  → {narr_path.name}")

    pipe2_path = C.OUTPUT_DIR / "10_pipeline2_summary.md"
    write_pipeline2_summary(pipe2_path)
    print(f"  → {pipe2_path.name}")

    print("\nHeadline metrics:")
    print(headline.to_string(index=False))
    print("\nGrouped feature importance (portfolio):")
    print(imp_df.to_string(index=False))
    print("\n[ok] Summary complete.")


if __name__ == "__main__":
    C._print_summary()
    main()
