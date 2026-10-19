from bootstrap import ensure_project_paths
import streamlit as st

ensure_project_paths()
from components.data_sources import (
    select_data_source,
    preview_data,
    selected_training_rows,
    remember_data,
)
from visit_forecast.io import inspect_data

st.set_page_config(page_title="Data Check", page_icon="✅", layout="wide")
st.title("✅ Data Check")
st.caption(
    "Inspect the actual data from your latest forecast or current inputs, or choose an upload, pasted CSV, or sample to check."
)
from components.session import persistent_page

with persistent_page("check_"):
    try:
        source = select_data_source("check_", st, include_current=True)
        if source.mode == "Use synthetic sample":
            left, right = st.columns(2)
            start = left.text_input(
                "Sample start (YYYY-MM-DD)", "2020-04-01", key="check_start"
            )
            end = right.text_input(
                "Sample end (YYYY-MM-DD)", "2025-03-01", key="check_end"
            )
        else:
            start, end = "2020-04-01", "2025-03-01"
        raw = source.resolve(start, end)
    except Exception as exc:
        st.error(f"Could not read data: {exc}")
        st.stop()
    if raw is None:
        st.info("Choose a data source to preview and validate it.")
        st.stop()

    settings = source.settings or {}
    if source.mode != "Current forecast data":
        remember_data(raw, source, settings, "Data Check")
    if settings:
        view = st.radio(
            "Preview rows",
            ["Rows selected for forecast", "Raw input"],
            key="check_preview_rows",
            horizontal=True,
        )
        try:
            preview = (
                selected_training_rows(raw, settings)
                if view == "Rows selected for forecast"
                else raw
            )
        except Exception as exc:
            st.warning(
                f"Could not apply the selected training window: {exc}. Showing raw input."
            )
            preview = raw
        st.caption(
            f"{len(preview):,} selected rows out of {len(raw):,} source rows. Validation below checks the complete source; invalid records are not silently removed."
        )
    else:
        preview = raw
    preview_data(
        preview,
        source.label,
        filtered=bool(settings and view == "Rows selected for forecast"),
    )
    st.download_button(
        "Download preview rows (CSV)",
        preview.to_csv(index=False).encode(),
        "data_preview.csv",
        mime="text/csv",
        key="check_download",
    )
    try:
        report = inspect_data(raw)
    except Exception as exc:
        st.error(f"Data validation failed: {exc}")
        st.stop()
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
    missing = (
        not report.coverage.empty and report.coverage.Missing_Months_Count.gt(0).any()
    )
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
