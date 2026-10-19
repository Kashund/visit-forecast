from bootstrap import ensure_project_paths
import streamlit as st

ensure_project_paths()
from components.forecast_page import render_forecast_page

st.set_page_config(page_title="Run Forecast", page_icon="🚀", layout="wide")
st.title("🚀 Run Forecast")
from components.session import persistent_page

with persistent_page(""):
    render_forecast_page()
