from __future__ import annotations

from bootstrap import ensure_project_paths
import streamlit as st

ensure_project_paths()
from visit_forecast.io import inspect_data, read_data_file

st.set_page_config(page_title="Data Check", page_icon="✅", layout="wide")
st.title("✅ Data Check")
uploaded = st.file_uploader("Upload Excel (.xlsx) or CSV", type=["xlsx", "csv"])
if uploaded is None:
    st.info(
        "Upload a dataset to validate required columns and monthly department coverage."
    )
    st.stop()
try:
    raw = read_data_file(uploaded)
    report = inspect_data(raw)
    if report.valid:
        st.session_state["shared_dataset"] = raw.copy()
except Exception as exc:
    st.error(f"Data validation failed: {exc}")
    st.stop()
st.dataframe(raw.head(30), width="stretch")
if not report.valid:
    st.error(
        "This dataset is not forecast-ready. Invalid records are never silently discarded."
    )
    st.dataframe(report.issues, width="stretch")
if report.duplicate_rows:
    st.warning(
        f"{report.duplicate_rows} rows share a department/month; forecasting sums these observations."
    )
st.subheader("Monthly coverage by department")
st.dataframe(report.coverage, width="stretch")
missing = not report.coverage.empty and report.coverage.Missing_Months_Count.gt(0).any()
if missing:
    st.warning(
        "Department coverage differs. Correct missing submissions or select a common complete window before aggregating All. Inactive departments require an explicit exclusion, not automatic zero filling."
    )
elif report.valid and not report.normalized.empty:
    st.success(
        "Dataset is forecast-ready. Model eligibility also depends on history length and forecast horizon."
    )
else:
    st.error("No valid observations are available.")
