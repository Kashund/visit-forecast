from __future__ import annotations

import pandas as pd
import streamlit as st
from io import StringIO


@st.cache_data
def _read_uploaded(file) -> pd.DataFrame:
    name = file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(file)
    return pd.read_excel(file)


def _read_pasted_csv(text: str) -> pd.DataFrame:
    return pd.read_csv(StringIO(text))


def sidebar_controls() -> tuple[dict, pd.DataFrame | None]:
    st.sidebar.header("Inputs")

    st.sidebar.markdown("**Data source**")
    source_mode = st.sidebar.radio(
        "Choose how to provide data",
        ["Upload file", "Paste values (CSV)", "Use synthetic sample"],
        index=0,
        key="source_mode",
        help="Upload Excel/CSV, paste raw CSV text, or use built-in synthetic data for testing.",
        label_visibility="collapsed",
    )

    uploaded = None
    pasted_text = None
    df_preview: pd.DataFrame | None = None

    if source_mode == "Upload file":
        uploaded = st.sidebar.file_uploader(
            "Upload Excel or CSV",
            type=["xlsx", "xls", "csv"],
            key="uploader",
            help="File must include columns: Date, Visits, Department.",
        )
        if uploaded is not None:
            df_preview = _read_uploaded(uploaded)

    elif source_mode == "Paste values (CSV)":
        st.sidebar.caption("Paste CSV with headers: Date, Visits, Department")
        pasted_text = st.sidebar.text_area(
            "CSV data",
            height=180,
            placeholder="Date,Visits,Department\n2020-04-01,120,Cardiology\n2020-05-01,130,Cardiology",
            key="pasted_csv",
            help="Paste CSV text exactly as shown (with headers).",
        )
        if pasted_text and pasted_text.strip():
            df_preview = _read_pasted_csv(pasted_text)

    else:
        st.sidebar.caption("Synthetic data includes: Cardiology, Oncology, Neurology (monthly).")

    st.sidebar.divider()

    st.sidebar.markdown("**Time window**")
    timeframe_start = st.sidebar.text_input(
        "Timeframe start (YYYY-MM-DD)",
        "2020-04-01",
        key="start",
        help="Start date for the historical data used to train the model.",
    )
    timeframe_end = st.sidebar.text_input(
        "Timeframe end (YYYY-MM-DD)",
        "2025-03-01",
        key="end",
        help="End date for the historical data used to train the model.",
    )

    forecast_periods = int(
        st.sidebar.slider(
            "Forecast months",
            1,
            36,
            12,
            key="months",
            help="How many future months to predict beyond the last historical date.",
        )
    )

    st.sidebar.divider()
    st.sidebar.markdown("**Capacity adjustment**")

    adjustment_mode_label = st.sidebar.selectbox(
        "Capacity adjustment",
        ["Capacity loss", "Capacity add"],
        index=0,
        key="adj_mode",
        help="Loss reduces forecasted visits; Add increases forecasted visits.",
    )
    adjustment_mode = "loss" if adjustment_mode_label == "Capacity loss" else "add"

    adjustment_percent = float(
        st.sidebar.slider(
            "Adjustment (%)",
            0,
            50,
            0,
            key="adj_pct",
            help="Percent magnitude applied in the selected month range (e.g., 10% loss => multiply by 0.90).",
        )
    )

    default_end = min(12, forecast_periods)
    month_range = st.sidebar.slider(
        "Apply adjustment between forecast months",
        min_value=1,
        max_value=forecast_periods,
        value=(1, default_end),
        step=1,
        key="adj_between",
        help="1 = first forecasted future month. The adjustment applies only within this inclusive range.",
    )
    adjustment_start_month, adjustment_end_month = int(month_range[0]), int(month_range[1])

    st.sidebar.divider()
    st.sidebar.markdown("**Model tuning (advanced)**")

    changepoint_prior_scale = float(
        st.sidebar.slider(
            "Changepoint prior scale",
            0.01,
            0.50,
            0.05,
            key="cp",
            help="Higher values allow the trend to change more aggressively (more flexible, higher overfit risk).",
        )
    )
    interval_width = float(
        st.sidebar.slider(
            "Interval width",
            0.50,
            0.99,
            0.90,
            key="iw",
            help="Uncertainty interval width used by Prophet (e.g., 0.90 = 90% interval).",
        )
    )

    st.sidebar.divider()
    st.sidebar.markdown("**Filtering**")

    dept_options = ["All"]
    exclude_options: list[str] = []

    synthetic_departments = ["Cardiology", "Oncology", "Neurology"]
    if source_mode == "Use synthetic sample":
        dept_options = ["All"] + synthetic_departments
        exclude_options = synthetic_departments
    elif df_preview is not None and "Department" in df_preview.columns:
        depts = sorted(df_preview["Department"].dropna().astype(str).unique().tolist())
        dept_options = ["All"] + depts
        exclude_options = depts

    department = st.sidebar.selectbox(
        "Department",
        dept_options,
        index=0,
        key="dept",
        help="Choose a single department or All to aggregate across included departments.",
    )
    exclude_departments = st.sidebar.multiselect(
        "Exclude departments",
        exclude_options,
        key="exclude",
        help="Remove departments from the dataset before forecasting.",
    )

    run = st.sidebar.button("Run forecast", type="primary", key="run_btn")

    controls = {
        "source_mode": source_mode,
        "uploaded": uploaded,
        "pasted_text": pasted_text,
        "timeframe_start": timeframe_start,
        "timeframe_end": timeframe_end,
        "forecast_periods": forecast_periods,
        "department": department,
        "exclude_departments": exclude_departments,
        "adjustment_mode": adjustment_mode,
        "adjustment_percent": adjustment_percent,
        "adjustment_start_month": adjustment_start_month,
        "adjustment_end_month": adjustment_end_month,
        "changepoint_prior_scale": changepoint_prior_scale,
        "interval_width": interval_width,
        "run": run,
    }

    return controls, df_preview
