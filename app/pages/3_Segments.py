"""Behavioral segments: cluster centroids and household counts."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
OUT = PROJECT_ROOT / "output"

st.set_page_config(page_title="Segments", page_icon="🏘️", layout="wide")
st.title("Behavioral segments")

cent_path = OUT / "02_segment_centroids.csv"
seg_path = OUT / "02_household_segments.csv"
img_path = OUT / "02_segment_profiles.png"

if not cent_path.exists():
    st.warning("Run `python code/02_segment_households.py` first.")
    st.stop()

centroids = pd.read_csv(cent_path)

st.subheader("Per-segment household counts")
counts = centroids[["segment", "n_households"]].copy()
st.dataframe(counts, use_container_width=True, hide_index=True)

st.subheader("Standardized 24-hour centroid profiles")
hour_cols = [f"h{h:02d}" for h in range(24)]
fig = go.Figure()
for _, row in centroids.iterrows():
    fig.add_trace(go.Scatter(
        x=list(range(24)),
        y=[row[c] for c in hour_cols],
        mode="lines+markers",
        name=f"segment {int(row['segment'])} (n={int(row['n_households'])})",
    ))
fig.update_layout(
    height=420,
    xaxis_title="Hour of day",
    yaxis_title="Standardized load (z-score within profile)",
    margin=dict(l=20, r=20, t=20, b=20),
    legend=dict(orientation="h", y=-0.2),
)
st.plotly_chart(fig, use_container_width=True)

if img_path.exists():
    st.subheader("Recovered archetypes (raw kWh scale)")
    st.image(str(img_path))

if seg_path.exists():
    with st.expander("Per-household assignment (debug)"):
        seg = pd.read_csv(seg_path)
        st.dataframe(seg, use_container_width=True, hide_index=True)

st.markdown(
    "Clustering is run on **pre-2020 daily profiles only** to keep the "
    "validation period out of the segmentation step. Profiles are "
    "standardized per household so the algorithm clusters by **shape**, "
    "not absolute consumption level."
)
