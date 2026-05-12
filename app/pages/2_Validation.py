"""Walk-forward validation results and model comparison."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
OUT = PROJECT_ROOT / "output"

st.set_page_config(page_title="Validation", page_icon="🧪", layout="wide")
st.title("Walk-forward validation")

summary_path = OUT / "06_metrics_summary.csv"
long_path = OUT / "06_walk_forward_results.csv"

if not summary_path.exists():
    st.warning(
        "Run `python code/06_walk_forward_validation.py` to generate validation "
        "results."
    )
    st.stop()

summary = pd.read_csv(summary_path)
summary["quantile_num"] = pd.to_numeric(summary["quantile"], errors="coerce")

st.subheader("MAPE by model, level, horizon bucket")
buckets_available = list(summary["bucket"].unique()) if "bucket" in summary.columns else []
default_buckets = [b for b in ["h=24", "h=168", "h<=24", "h=25-72", "h=73-168"] if b in buckets_available]
chosen_buckets = st.multiselect("Buckets", buckets_available, default=default_buckets or buckets_available)

mape_view = summary[
    (summary["model"].isin(["persistence", "sarimax", "lightgbm"]))
    & (summary["quantile_num"].isna() | (summary["quantile_num"] == 0.5))
    & (summary["bucket"].isin(chosen_buckets) if chosen_buckets else True)
].copy()
mape_view["MAPE %"] = mape_view["mape"] * 100
mape_view["RMSE"] = mape_view["rmse"]
mape_view = mape_view[["level", "bucket", "model", "n", "MAPE %", "RMSE"]]
st.dataframe(mape_view, use_container_width=True, hide_index=True)

if not mape_view.empty:
    fig = px.bar(
        mape_view,
        x="level",
        y="MAPE %",
        color="model",
        barmode="group",
        facet_col="bucket",
        facet_col_spacing=0.05,
        height=420,
    )
    fig.update_layout(margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.subheader("Pinball loss by quantile (LightGBM, portfolio)")
pin = summary[
    (summary["model"] == "lightgbm")
    & (summary["level"] == "portfolio")
    & (summary["quantile_num"].notna())
].copy()
if pin.empty:
    st.info("No pinball-loss rows yet.")
else:
    pin["quantile"] = pin["quantile_num"]
    fig2 = px.line(
        pin,
        x="quantile",
        y="pinball",
        color="bucket",
        markers=True,
        height=350,
    )
    fig2.update_layout(margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(fig2, use_container_width=True)

if long_path.exists():
    with st.expander("Per-fold prediction table (long format)"):
        long = pd.read_csv(long_path)
        st.dataframe(long.head(500), use_container_width=True, hide_index=True)
        st.caption(f"Showing first 500 of {len(long):,} rows")

st.markdown("---")
recon_path = OUT / "07_reconciliation_comparison.csv"
if recon_path.exists():
    st.subheader("Hierarchical reconciliation (BottomUp + MinT-OLS)")
    recon = pd.read_csv(recon_path)
    recon["MAPE %"] = recon["mape"] * 100
    bucket_col = "bucket" if "bucket" in recon.columns else "horizon_h"
    st.dataframe(
        recon[["level", bucket_col, "method", "n", "MAPE %"]],
        use_container_width=True, hide_index=True,
    )
