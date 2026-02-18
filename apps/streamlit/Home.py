import streamlit as st

st.set_page_config(page_title="Visit Forecast", page_icon="📈", layout="wide")

st.title("📈 Outpatient Visit Forecasting")
st.write(
    "Run Prophet-based forecasts from a local dataset, get backtesting metrics, "
    "and review fiscal-year summaries (FY = Apr → Mar)."
)
st.markdown("Use the pages on the left to run forecasts and validate data.")
