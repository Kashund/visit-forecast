from bootstrap import ensure_project_paths
import streamlit as st

ensure_project_paths()
from components.forecast_page import render_forecast_page

st.set_page_config(page_title="Department Forecasts", page_icon="🏥", layout="wide")
st.title("🏥 Department Forecasts")
st.caption(
    "Fit each included department independently, then switch the department output to review the same results and diagnostics as Run Forecast."
)
render_forecast_page(department_mode=True)
