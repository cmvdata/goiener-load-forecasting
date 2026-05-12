"""Forecast viewer — most recent batch run, with prediction intervals."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
FC_DIR = PROJECT_ROOT / "output" / "forecasts"

st.set_page_config(page_title="Forecasts", page_icon="⚡", layout="wide")
st.title("Forecasts")


def latest_forecast_file() -> Path | None:
    if not FC_DIR.exists():
        return None
    files = sorted(FC_DIR.glob("forecasts_*.parquet"))
    return files[-1] if files else None


fc_path = latest_forecast_file()
if fc_path is None:
    st.warning(
        "No forecast files found in `output/forecasts/`. "
        "Run `python code/08_run_daily_batch.py` after training the champion."
    )
    st.stop()

st.caption(f"Most recent batch: `{fc_path.name}`")

fc = pd.read_parquet(fc_path)
fc["timestamp"] = pd.to_datetime(fc["timestamp"])

levels = sorted(fc["level"].unique())
choice = st.selectbox("Level", levels, index=levels.index("portfolio") if "portfolio" in levels else 0)

sub = fc[fc["level"] == choice].sort_values("timestamp")

st.subheader(f"{choice} — quantile forecasts")

quantile_cols = [c for c in sub.columns if c.startswith("q")]
display = sub[["timestamp", "horizon_h"] + quantile_cols].copy()
display["timestamp"] = display["timestamp"].dt.strftime("%Y-%m-%d %H:%M")
st.dataframe(display, use_container_width=True, hide_index=True)

if {"q050", "q250", "q500", "q750", "q950"}.issubset(sub.columns):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sub["timestamp"], y=sub["q500"],
        mode="lines+markers", name="median (q50)",
        line=dict(color="#1f77b4", width=3),
    ))
    fig.add_trace(go.Scatter(
        x=sub["timestamp"], y=sub["q750"],
        mode="lines", line=dict(width=0), showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=sub["timestamp"], y=sub["q250"],
        mode="lines", line=dict(width=0),
        fill="tonexty", fillcolor="rgba(31,119,180,0.25)",
        name="50% interval",
    ))
    fig.add_trace(go.Scatter(
        x=sub["timestamp"], y=sub["q950"],
        mode="lines", line=dict(width=0), showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=sub["timestamp"], y=sub["q050"],
        mode="lines", line=dict(width=0),
        fill="tonexty", fillcolor="rgba(31,119,180,0.10)",
        name="90% interval",
    ))
    fig.update_layout(
        height=480,
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis_title="Forecast target time",
        yaxis_title="kWh",
        hovermode="x unified",
        legend=dict(orientation="h", y=-0.2),
    )
    st.plotly_chart(fig, use_container_width=True)

with st.expander("Reading the chart"):
    st.markdown(
        "The dark line is the **median** forecast. The darker shaded band is "
        "the **50% prediction interval** (q25–q75); the lighter band is the "
        "**90% interval** (q05–q95). A well-calibrated 90% interval should "
        "contain the realized value 90% of the time across many forecasts.\n\n"
        "All forecasts are **MinT-OLS reconciled** so the segment forecasts "
        "sum to the portfolio forecast at every quantile."
    )
