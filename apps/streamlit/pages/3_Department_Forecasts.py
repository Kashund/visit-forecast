from __future__ import annotations

from bootstrap import ensure_project_paths
import io as _io

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ensure_project_paths()

from visit_forecast.io import generate_synthetic_data
from visit_forecast.service import forecast_departments

st.set_page_config(
    page_title="Department Forecasts", page_icon="🏥", layout="wide"
)
st.title("🏥 Department Forecasts")
st.caption(
    "Forecast each department independently. With Auto selected, every "
    "department gets its own model comparison, since departments often have "
    "different trend and seasonality profiles."
)

# --- Controls ---
with st.sidebar:
    st.markdown("**Data source**")
    source_mode = st.radio(
        "Source",
        ["Use synthetic sample", "Upload file"],
        key="dept_source_mode",
    )
    uploaded = None
    if source_mode == "Upload file":
        uploaded = st.file_uploader(
            "Upload Excel (.xlsx) or CSV with Date, Visits, Department columns",
            type=["xlsx", "csv"],
            key="dept_upload",
        )

    st.markdown("**Timeframe**")
    timeframe_start = st.text_input(
        "Start (YYYY-MM-DD)", value="2020-04-01", key="dept_start"
    )
    timeframe_end = st.text_input(
        "End (YYYY-MM-DD)", value="2025-03-01", key="dept_end"
    )
    forecast_periods = st.slider(
        "Forecast months", min_value=1, max_value=24, value=12, key="dept_months"
    )

    st.markdown("**Model**")
    model_label_map = {
        "Auto (compare models per department)": "auto",
        "Prophet": "prophet",
        "Holt-Winters (ETS)": "ets",
        "Theta": "theta",
        "SARIMA": "arima",
        "Seasonal naive (baseline)": "seasonal_naive",
    }
    model_choice_label = st.selectbox(
        "Model", list(model_label_map.keys()), index=0, key="dept_model_choice"
    )
    model_choice = model_label_map[model_choice_label]

    include_prophet = True
    if model_choice == "auto":
        include_prophet = st.checkbox(
            "Include Prophet in auto comparison",
            value=False,
            key="dept_include_prophet",
            help=(
                "Prophet is by far the slowest candidate. With several "
                "departments, leaving it out keeps runs fast; the classical "
                "models are usually competitive on monthly data."
            ),
        )

    use_calendar_covariates = st.checkbox(
        "Calendar covariates (business days per month)",
        value=False,
        key="dept_covariates",
    )

    run_clicked = st.button("Run department forecasts", key="dept_run_btn")


def _load_raw_df():
    if source_mode == "Use synthetic sample":
        return generate_synthetic_data(timeframe_start, timeframe_end)
    if uploaded is None:
        st.info("Upload a dataset to run department forecasts.")
        st.stop()
    name = uploaded.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded)
    return pd.read_excel(uploaded)


if run_clicked:
    auto_candidates = None
    if model_choice == "auto" and not include_prophet:
        auto_candidates = ["ets", "theta", "arima", "seasonal_naive"]

    try:
        raw_df = _load_raw_df()
        with st.spinner("Forecasting each department..."):
            bundle = forecast_departments(
                df=raw_df,
                timeframe_start=timeframe_start,
                timeframe_end=timeframe_end,
                forecast_periods=int(forecast_periods),
                model_choice=model_choice,
                auto_candidates=auto_candidates,
                use_calendar_covariates=bool(use_calendar_covariates),
            )
        st.session_state["dept_bundle"] = bundle
    except Exception as run_error:
        st.error(f"Department forecast failed: {run_error}")
        st.stop()

bundle = st.session_state.get("dept_bundle")
if bundle is None:
    st.info("Configure the run in the sidebar, then click Run department forecasts.")
    st.stop()

if bundle.failures:
    st.error(
        "The aggregate line and exports are partial because one or more "
        "department forecasts failed. Failed departments are not included."
    )
    for department_name, failure_message in bundle.failures.items():
        st.warning(f"{department_name}: {failure_message}")

# --- Summary table ---
st.subheader("Per-department model selection and accuracy")
summary_display = bundle.summary_df.copy()
st.dataframe(
    summary_display.style.format(
        {
            "MAPE": "{:.2f}",
            "MAE": "{:.1f}",
            "RMSE": "{:.1f}",
            "forecast_total": "{:,.0f}",
        }
    ),
    width="stretch",
)

# --- Chart ---
st.subheader("Forecast by department")
chart_df = bundle.combined_future_df
fig = go.Figure()
for department_name in sorted(chart_df["Department"].unique()):
    dept_rows = chart_df[chart_df["Department"] == department_name]
    is_total = department_name.startswith(("All (", "Partial total ("))
    fig.add_trace(
        go.Scatter(
            x=dept_rows["ds"],
            y=dept_rows["yhat_adjusted"],
            mode="lines+markers",
            name=department_name,
            line=dict(width=3 if is_total else 1.5, dash="dash" if is_total else "solid"),
        )
    )
fig.update_layout(
    hovermode="x unified", xaxis_title="Date", yaxis_title="Forecasted visits"
)
st.plotly_chart(fig, width="stretch")

# --- Export ---
st.subheader("Export")
export_buffer = _io.BytesIO()
with pd.ExcelWriter(export_buffer, engine="openpyxl") as writer:
    bundle.summary_df.to_excel(writer, sheet_name="Model_Summary", index=False)
    bundle.combined_future_df.to_excel(
        writer, sheet_name="Forecasts_By_Department", index=False
    )
st.download_button(
    "Download department forecasts (Excel)",
    data=export_buffer.getvalue(),
    file_name="department_forecasts.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    key="dept_download_xlsx",
)
st.download_button(
    "Download combined forecasts (CSV)",
    data=bundle.combined_future_df.to_csv(index=False).encode("utf-8"),
    file_name="department_forecasts.csv",
    mime="text/csv",
    key="dept_download_csv",
)
