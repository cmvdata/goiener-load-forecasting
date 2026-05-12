"""Probabilistic forecast calibration — raw bands vs conformal-corrected."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
OUT = PROJECT_ROOT / "output"

st.set_page_config(page_title="Calibration", page_icon="🎯", layout="wide")
st.title("Probabilistic forecast calibration")

st.markdown(
    "The recursive single-step LightGBM produces empirical coverage well "
    "below nominal at every horizon — at h=168 the 50% interval covers only "
    "about 16% and the 90% interval about 46% on the raw walk-forward output. "
    "We correct this post-hoc with **split conformal prediction** "
    "(see Methodology page for the design)."
)

cal_path = OUT / "06b_calibration_summary.csv"
if not cal_path.exists():
    st.warning(
        "Calibration summary missing. Run "
        "`python code/06b_conformal_calibration.py` after `code/06_walk_forward_validation.py`."
    )
    st.stop()

cal = pd.read_csv(cal_path)

POOL_BUCKETS = ["h<=24", "h=25-72", "h=73-168"]
DIAG_BUCKETS = ["h=1", "h=24", "h=72", "h=168"]

st.markdown("---")
st.subheader("Pool-bucket coverage on the test split (raw vs calibrated)")
st.caption(
    "Conformal adjustments are computed at the pool-bucket × level resolution "
    "and the gate is evaluated here. Cells in green satisfy "
    "[0.45, 0.55] for cov50 and [0.85, 0.95] for cov90."
)

pf_pool = cal[cal["bucket"].isin(POOL_BUCKETS)].copy()
levels = ["portfolio"] + sorted(c for c in pf_pool["level"].unique() if c.startswith("segment_"))
chosen_level = st.selectbox("Level", levels, index=0)
display = pf_pool[pf_pool["level"] == chosen_level].set_index("bucket").reindex(POOL_BUCKETS)

cov50_band = (0.45, 0.55)
cov90_band = (0.85, 0.95)


def fmt_cell(value: float, band: tuple[float, float]) -> str:
    in_band = band[0] <= value <= band[1]
    return f"{'✅' if in_band else '⚠️'} {value:.3f}"


nice = pd.DataFrame({
    "bucket": display.index,
    "n (test)": display["n"].astype(int).values,
    "raw cov50": display["raw_cov50"].apply(lambda v: f"{v:.3f}").values,
    "calibrated cov50": [fmt_cell(v, cov50_band) for v in display["cal_cov50"]],
    "raw cov90": display["raw_cov90"].apply(lambda v: f"{v:.3f}").values,
    "calibrated cov90": [fmt_cell(v, cov90_band) for v in display["cal_cov90"]],
})
st.dataframe(nice, use_container_width=True, hide_index=True)


st.markdown("---")
st.subheader("Calibration curve — pool buckets, all levels pooled")
pooled = cal[(cal["level"] == "ALL") & (cal["bucket"].isin(POOL_BUCKETS))]
fig = go.Figure()
fig.add_trace(go.Scatter(x=[0, 100], y=[0, 100], mode="lines",
                         line=dict(dash="dash", color="black"),
                         name="perfect calibration"))
palette = {"h<=24": "#1f77b4", "h=25-72": "#ff7f0e", "h=73-168": "#d62728"}
for _, r in pooled.iterrows():
    color = palette[r["bucket"]]
    fig.add_trace(go.Scatter(
        x=[50, 90], y=[r["raw_cov50"] * 100, r["raw_cov90"] * 100],
        mode="lines+markers", line=dict(color=color, dash="dot"),
        name=f"{r['bucket']} raw", opacity=0.55,
    ))
    fig.add_trace(go.Scatter(
        x=[50, 90], y=[r["cal_cov50"] * 100, r["cal_cov90"] * 100],
        mode="lines+markers", line=dict(color=color, width=3),
        name=f"{r['bucket']} calibrated",
    ))
fig.add_shape(type="rect", x0=49, x1=51, y0=45, y1=55,
              fillcolor="green", opacity=0.15, line=dict(width=0))
fig.add_shape(type="rect", x0=89, x1=91, y0=85, y1=95,
              fillcolor="green", opacity=0.15, line=dict(width=0))
fig.update_layout(
    height=460,
    xaxis_title="Nominal interval (%)",
    yaxis_title="Empirical coverage (%)",
    xaxis=dict(range=[40, 100]),
    yaxis=dict(range=[0, 100]),
    margin=dict(l=20, r=20, t=20, b=20),
)
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.subheader("Per-horizon coverage (diagnostic)")
st.caption(
    "Coverage at single horizons inside the pool buckets. By design, the "
    "calibration uses a single δ per pool, so individual horizons inside the "
    "pool can drift from nominal — h=1 tends to over-cover (raw bands were "
    "already near nominal there) and h=168 to under-cover (the deepest end "
    "of the recursive trajectory). The pool-bucket gate above is what we "
    "promise empirically."
)
diag = cal[cal["bucket"].isin(DIAG_BUCKETS)].copy()
diag["row_key"] = diag["level"] + " — " + diag["bucket"]
selectable = sorted(diag["row_key"].unique())
chosen = st.multiselect(
    "Show diagnostic cells", selectable,
    default=[f"portfolio — {b}" for b in DIAG_BUCKETS],
)
view = diag[diag["row_key"].isin(chosen)][
    ["level", "bucket", "n", "raw_cov50", "cal_cov50", "raw_cov90", "cal_cov90"]
]
st.dataframe(view, use_container_width=True, hide_index=True)

st.markdown("---")
st.subheader("Adjustments table (per level × pool bucket × quantile)")
adj_path = OUT / "06b_conformal_adjustments.csv"
if adj_path.exists():
    adj = pd.read_csv(adj_path)
    st.caption(
        f"δ values added to each non-median quantile. n_cal is the number of "
        "calibration residuals pooled per cell. The median (q=0.5) is "
        "untouched, so MAPE is preserved."
    )
    pivot = adj.pivot_table(
        index=["level", "bucket"],
        columns="quantile", values="adjustment", aggfunc="first",
    ).round(2)
    pivot.columns = [f"δ q={q}" for q in pivot.columns]
    st.dataframe(pivot.reset_index(), use_container_width=True, hide_index=True)
else:
    st.info("Adjustments file not found.")

gate_path = OUT / "06b_calibration_gate.csv"
if gate_path.exists():
    gate = pd.read_csv(gate_path)
    n_pass = (
        (gate["cal_cov50"].between(*cov50_band))
        & (gate["cal_cov90"].between(*cov90_band))
    ).sum()
    st.success(
        f"Gate: {n_pass}/{len(gate)} (level × pool-bucket) cells satisfy "
        f"cov50 ∈ [{cov50_band[0]}, {cov50_band[1]}] and "
        f"cov90 ∈ [{cov90_band[0]}, {cov90_band[1]}]."
    )
