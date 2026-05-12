"""Streamlit entry point for the goiener-load-forecasting dashboard."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
OUTPUT_DIR = PROJECT_ROOT / "output"
LOGS_DIR = PROJECT_ROOT / "logs"

GH_URL = "https://github.com/cmvdata/goiener-load-forecasting"


st.set_page_config(
    page_title="GoiEner load forecasting",
    page_icon="⚡",
    layout="wide",
)


@st.cache_data
def load_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_data
def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def render_sidebar():
    st.sidebar.title("GoiEner load forecasting")
    st.sidebar.markdown(f"[View source on GitHub]({GH_URL})")
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "Pages:\n\n"
        "- **Forecasts** — interactive forecast viewer with intervals\n"
        "- **Validation** — walk-forward results, model comparison\n"
        "- **Segments** — cluster profiles and per-segment metrics\n"
        "- **Calibration** — coverage and pinball loss diagnostics\n"
        "- **Methodology** — assumptions, design choices, limitations"
    )


def render_home():
    st.title("Short-term load forecasting — Spanish residential portfolio")
    st.markdown(
        "Hourly demand forecasting for ~16,500 Spanish households at horizons "
        "from 24 hours to one week ahead, with probabilistic outputs, "
        "hierarchical reconciliation across behavioral segments, and "
        "walk-forward validation that respects time."
    )

    st.markdown("---")

    headline = load_csv(OUTPUT_DIR / "10_headline_metrics.csv")
    if headline is None or headline.empty:
        st.info(
            "Headline metrics not generated yet. Run "
            "`python code/10_summarize_results.py` after the pipeline completes."
        )
    else:
        st.subheader("Headline metrics (portfolio level)")
        def _fmt(v):
            if isinstance(v, str):
                # Conformal coverage rows arrive pre-formatted as "raw → cal".
                return v
            if pd.isna(v):
                return "—"
            return f"{v*100:.2f}%" if v < 1 else f"{v:.3f}"

        nice = headline.copy()
        for col in ("lightgbm", "sarimax", "persistence"):
            nice[col] = nice[col].apply(_fmt)
        st.dataframe(nice, use_container_width=True, hide_index=True)

    promo = load_json(OUTPUT_DIR / "09_promotion_decision.json")
    if promo is not None:
        st.markdown("---")
        st.subheader("Most recent champion / challenger evaluation")
        c1, c2, c3 = st.columns(3)
        c1.metric(
            "Champion MAPE",
            f"{promo['champion']['mape']*100:.2f}%",
        )
        c2.metric(
            "Challenger MAPE",
            f"{promo['challenger']['mape']*100:.2f}%",
            delta=f"{-promo['delta_mape_pp']:.2f} pp",
            delta_color="inverse",
        )
        c3.metric(
            "Promoted?",
            "Yes" if promo["promoted"] else "No",
        )
        with st.expander("Decision details"):
            st.json(promo)

    st.markdown("---")

    st.subheader("What this dashboard shows")
    st.markdown(
        "- **Forecasts** — the most recent batch run, with prediction intervals "
        "for the portfolio and each behavioral segment.\n"
        "- **Validation** — model comparison from walk-forward evaluation. "
        "Each MAPE is shown with its persistence baseline so improvements are "
        "visible in context.\n"
        "- **Segments** — cluster centroids that define the four behavioral "
        "archetypes used in the hierarchy.\n"
        "- **Calibration** — coverage rates and pinball loss for the "
        "probabilistic forecasts.\n"
        "- **Methodology** — assumptions, leakage controls, and the explicit "
        "out-of-scope items (no real-time productionization, realized weather)."
    )


def main():
    render_sidebar()
    render_home()


if __name__ == "__main__":
    main()
