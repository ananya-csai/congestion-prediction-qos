"""Optional - Streamlit dashboard for browsing results (not needed for the paper).

Usage:
    pip install streamlit
    streamlit run dashboard.py
It only READS files in ./outputs produced by the other modules.
"""
import json

import pandas as pd
import streamlit as st

from config import OUT_DIR

st.set_page_config(page_title="Congestion prediction results", layout="wide")
st.title("ML congestion prediction and proactive rerouting")

runs = sorted(p.name for p in OUT_DIR.iterdir() if (p / "dataset_info.json").exists())
if not runs:
    st.warning("No runs found. Run features.py, train_model.py and route_optimizer.py first.")
    st.stop()
run = st.sidebar.selectbox("Run", runs)
folder = OUT_DIR / run

info = json.loads((folder / "dataset_info.json").read_text())
c1, c2, c3 = st.columns(3)
c1.metric("Links", info["links"])
c2.metric("Time steps", info["T"])
c3.metric("Test congestion rate", f"{info['positive_rate']['test']:.2%}")

pred = folder / "metrics_prediction.csv"
if pred.exists():
    st.subheader("Prediction (test period)")
    st.dataframe(pd.read_csv(pred).round(4), use_container_width=True)

imp = folder / "feature_importance.csv"
if imp.exists():
    st.subheader("Random Forest feature importance")
    st.bar_chart(pd.read_csv(imp, index_col=0))

summaries = sorted(folder.glob("routing_summary_*.csv"))
if summaries:
    choice = st.selectbox("Routing configuration", [s.name for s in summaries])
    st.subheader("Routing (test period)")
    st.dataframe(pd.read_csv(folder / choice, index_col=0).round(4), use_container_width=True)
    steps = pd.read_csv(folder / choice.replace("summary", "steps"))
    st.subheader("Maximum link utilisation over time")
    pivot = steps.pivot(index="t", columns="method", values="mlu")
    st.line_chart(pivot)
