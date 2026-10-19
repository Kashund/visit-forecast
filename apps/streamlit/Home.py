from bootstrap import ensure_project_paths
import streamlit as st

ensure_project_paths()

st.set_page_config(page_title="Visit Forecast", page_icon="📈", layout="wide")

st.title("📈 Outpatient Visit Forecasting")
st.write(
    "Compare forecasting models using local or synthetic data, explore capacity scenarios, get validation metrics, "
    "and review fiscal-year summaries (FY = Apr → Mar)."
)
st.markdown("Use the pages on the left to run forecasts and validate data.")

from components.session import session_sidebar

session_sidebar()
