from __future__ import annotations

from bootstrap import ensure_project_paths
import io as _io

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ensure_project_paths()

from visit_forecast.io import generate_synthetic_data, read_data_file
from visit_forecast.exports import excel_bytes
from visit_forecast.provenance import build_run_manifest, configuration_fingerprint
from state import record_auxiliary_run
from visit_forecast.service import forecast_departments

st.set_page_config(page_title="Department Forecasts", page_icon="🏥", layout="wide")
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
        [
            "Use synthetic sample",
            "Upload file",
            "Paste values (CSV)",
            "Use shared dataset",
        ],
        key="dept_source_mode",
    )
    uploaded = None
    if source_mode == "Upload file":
        uploaded = st.file_uploader(
            "Upload Excel (.xlsx) or CSV with Date, Visits, Department columns",
            type=["xlsx", "csv"],
            key="dept_upload",
        )

    pasted = (
        st.text_area("CSV data", key="dept_pasted")
        if source_mode == "Paste values (CSV)"
        else ""
    )
    if source_mode == "Use shared dataset" and "shared_dataset" not in st.session_state:
        st.info("Run a forecast or upload data on Data Check to share a dataset.")
    st.markdown("**Timeframe**")
    timeframe_start = st.text_input(
        "Start (YYYY-MM-DD)", value="2020-04-01", key="dept_start"
    )
    timeframe_end = st.text_input(
        "End (YYYY-MM-DD)", value="2025-03-01", key="dept_end"
    )
    forecast_periods = st.slider(
        "Forecast months", min_value=1, max_value=36, value=12, key="dept_months"
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

    include_names = st.text_input(
        "Include departments (comma separated; blank = all)", key="dept_include"
    )
    exclude_names = st.text_input(
        "Exclude departments (comma separated)", key="dept_exclude"
    )
    target_coverage = st.slider(
        "Target coverage", 0.50, 0.99, 0.90, key="dept_coverage"
    )
    run_clicked = st.button("Run department forecasts", key="dept_run_btn")


def _load_raw_df():
    if source_mode == "Use synthetic sample":
        return generate_synthetic_data(timeframe_start, timeframe_end)
    if source_mode == "Use shared dataset":
        if "shared_dataset" not in st.session_state:
            st.info("No shared dataset yet.")
            st.stop()
        return st.session_state["shared_dataset"].copy()
    if source_mode == "Paste values (CSV)":
        return pd.read_csv(_io.StringIO(pasted))
    if uploaded is None:
        st.info("Upload a dataset to run department forecasts.")
        st.stop()
    return read_data_file(uploaded)


if run_clicked:
    auto_candidates = None
    if model_choice == "auto" and not include_prophet:
        auto_candidates = ["ets", "theta", "arima", "seasonal_naive"]

    try:
        raw_df = _load_raw_df()
        with st.spinner("Forecasting each department..."):
            progress = st.empty()
            bundle = forecast_departments(
                df=raw_df,
                progress_callback=progress.caption,
                timeframe_start=timeframe_start,
                timeframe_end=timeframe_end,
                forecast_periods=int(forecast_periods),
                model_choice=model_choice,
                auto_candidates=auto_candidates,
                use_calendar_covariates=bool(use_calendar_covariates),
                interval_width=target_coverage,
                departments=[d.strip() for d in include_names.split(",") if d.strip()]
                or None,
                exclude_departments=[
                    d.strip() for d in exclude_names.split(",") if d.strip()
                ],
            )
        st.session_state["shared_dataset"] = raw_df.copy()
        st.session_state["dept_bundle"] = bundle
        manifest = build_run_manifest(
            raw_df,
            dict(
                workflow="departments",
                timeframe_start=timeframe_start,
                timeframe_end=timeframe_end,
                forecast_periods=forecast_periods,
                model_choice=model_choice,
                include_prophet=include_prophet,
                calendar_covariates=use_calendar_covariates,
                include_departments=include_names,
                exclude_departments=exclude_names,
                target_coverage=target_coverage,
            ),
        )
        manifest["department_runs"] = {
            d: r.run_manifest for d, r in bundle.results.items()
        }
        manifest["failures"] = bundle.failures
        st.session_state["dept_manifest"] = manifest
        st.session_state["dept_run_data"] = raw_df.copy()
        st.session_state["dept_shared_signature"] = configuration_fingerprint(
            {}, raw_df
        )
        st.session_state["dept_controls"] = (
            source_mode,
            timeframe_start,
            timeframe_end,
            forecast_periods,
            model_choice,
            include_prophet,
            use_calendar_covariates,
            include_names,
            exclude_names,
            target_coverage,
            pasted,
            uploaded.getvalue() if uploaded else None,
        )
        record_auxiliary_run(
            st.session_state, "departments", manifest, failures=bundle.failures
        )
    except Exception as run_error:
        st.error(f"Department forecast failed: {run_error}")
        st.stop()

bundle = st.session_state.get("dept_bundle")
if bundle is None:
    st.info("Configure the run in the sidebar, then click Run department forecasts.")
    st.stop()

current_controls = (
    source_mode,
    timeframe_start,
    timeframe_end,
    forecast_periods,
    model_choice,
    include_prophet,
    use_calendar_covariates,
    include_names,
    exclude_names,
    target_coverage,
    pasted,
    uploaded.getvalue() if uploaded else None,
)
shared_changed = (
    source_mode == "Use shared dataset"
    and "shared_dataset" in st.session_state
    and configuration_fingerprint({}, st.session_state["shared_dataset"])
    != st.session_state.get("dept_shared_signature")
)
if current_controls != st.session_state.get("dept_controls") or shared_changed:
    st.warning(
        "Controls changed. Displayed results use the previous run; run department forecasts to update."
    )
if st.session_state.get("history_save_error"):
    st.warning(st.session_state["history_save_error"])

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
            line=dict(
                width=3 if is_total else 1.5, dash="dash" if is_total else "solid"
            ),
        )
    )
fig.update_layout(
    hovermode="x unified", xaxis_title="Date", yaxis_title="Forecasted visits"
)
st.plotly_chart(fig, width="stretch")

st.subheader("Department diagnostics")
for department_name, department_result in bundle.results.items():
    with st.expander(department_name):
        st.caption(department_result.tuning_note)
        st.json(department_result.evaluation_summary)
        if department_result.tuning_diagnostics_df is not None:
            st.dataframe(department_result.tuning_diagnostics_df, width="stretch")
        st.dataframe(department_result.future_forecast_df, width="stretch")
        st.dataframe(department_result.fiscal_summary, width="stretch")

# --- Export ---
failures_df = pd.DataFrame(
    [{"Department": d, "Reason": reason} for d, reason in bundle.failures.items()],
    columns=["Department", "Reason"],
)
frames = [
    r.tuning_diagnostics_df.assign(Department=d)
    for d, r in bundle.results.items()
    if r.tuning_diagnostics_df is not None
]
diagnostics = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
export = excel_bytes(
    {
        "Model_Summary": bundle.summary_df,
        "Forecasts_By_Department": bundle.combined_future_df,
        "Failures": failures_df,
        "Candidate_Diagnostics": diagnostics,
    },
    st.session_state.get("dept_manifest"),
)
st.download_button(
    "Download department forecasts (Excel)",
    export,
    "department_forecasts.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    key="dept_download_xlsx",
)
st.download_button(
    "Download combined forecasts (CSV)",
    bundle.combined_future_df.to_csv(index=False).encode(),
    "department_forecasts.csv",
    mime="text/csv",
    key="dept_download_csv",
)
