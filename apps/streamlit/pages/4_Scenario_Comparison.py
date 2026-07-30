from __future__ import annotations

from bootstrap import ensure_project_paths
import io as _io

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ensure_project_paths()

from visit_forecast.io import generate_synthetic_data
from visit_forecast.service import (
    apply_capacity_phases_to_future_df,
    forecast_visits,
)

st.set_page_config(
    page_title="Scenario Comparison", page_icon="⚖️", layout="wide"
)
st.title("⚖️ Scenario Comparison")
st.caption(
    "Fit the forecast once, then compare two capacity phase configurations "
    "side by side against the unadjusted baseline. Both scenarios reuse the "
    "same fitted model, so differences are purely the phase assumptions."
)

MAX_PHASES = 3


def _phase_editor(scenario_key: str, forecast_periods: int) -> list[dict]:
    phases: list[dict] = []
    for phase_number in range(1, MAX_PHASES + 1):
        prefix = f"{scenario_key}_phase_{phase_number}"
        with st.expander(f"Phase {phase_number}", expanded=phase_number == 1):
            enabled = st.checkbox("Enable", key=f"{prefix}_enabled")
            mode = st.selectbox(
                "Mode", ["loss", "add"], key=f"{prefix}_mode", disabled=not enabled
            )
            percent = st.slider(
                "Percent",
                min_value=0,
                max_value=50,
                value=0,
                key=f"{prefix}_percent",
                disabled=not enabled,
            )
            month_range = st.slider(
                "Apply between forecast months",
                min_value=1,
                max_value=forecast_periods,
                value=(1, min(12, forecast_periods)),
                key=f"{prefix}_between",
                disabled=not enabled,
            )
        phases.append(
            {
                "enabled": bool(enabled),
                "mode": str(mode),
                "percent": float(percent),
                "start_month": int(month_range[0]),
                "end_month": int(month_range[1]),
            }
        )
    return phases


def _scenario_summary(phases: list[dict]) -> str:
    parts = [
        f"{p['mode']} {p['percent']:.0f}% (months {p['start_month']}-{p['end_month']})"
        for p in phases
        if p["enabled"] and p["percent"] > 0
    ]
    return "; ".join(parts) if parts else "No adjustments"


# --- Sidebar: baseline forecast controls ---
with st.sidebar:
    st.markdown("**Baseline forecast**")
    source_mode = st.radio(
        "Source", ["Use synthetic sample", "Upload file"], key="cmp_source_mode"
    )
    uploaded = None
    if source_mode == "Upload file":
        uploaded = st.file_uploader(
            "Upload Excel (.xlsx) or CSV with Date, Visits, Department columns",
            type=["xlsx", "csv"],
            key="cmp_upload",
        )
    timeframe_start = st.text_input(
        "Start (YYYY-MM-DD)", value="2020-04-01", key="cmp_start"
    )
    timeframe_end = st.text_input(
        "End (YYYY-MM-DD)", value="2025-03-01", key="cmp_end"
    )
    forecast_periods = st.slider(
        "Forecast months", min_value=1, max_value=24, value=12, key="cmp_months"
    )
    model_label_map = {
        "Auto (compare models)": "auto",
        "Prophet": "prophet",
        "Holt-Winters (ETS)": "ets",
        "Theta": "theta",
        "SARIMA": "arima",
    }
    model_choice_label = st.selectbox(
        "Model", list(model_label_map.keys()), index=0, key="cmp_model_choice"
    )
    model_choice = model_label_map[model_choice_label]
    run_clicked = st.button("Run comparison", key="cmp_run_btn")

# --- Scenario phase editors ---
column_a, column_b = st.columns(2)
with column_a:
    st.subheader("Scenario A")
    scenario_a_phases = _phase_editor("cmp_a", int(forecast_periods))
with column_b:
    st.subheader("Scenario B")
    scenario_b_phases = _phase_editor("cmp_b", int(forecast_periods))

if run_clicked:
    try:
        raw_df = None
        if source_mode == "Upload file":
            if uploaded is None:
                st.info("Upload a dataset to run the comparison.")
                st.stop()
            name = uploaded.name.lower()
            raw_df = (
                pd.read_csv(uploaded)
                if name.endswith(".csv")
                else pd.read_excel(uploaded)
            )
        else:
            raw_df = generate_synthetic_data(timeframe_start, timeframe_end)

        with st.spinner("Fitting baseline forecast..."):
            result = forecast_visits(
                df=raw_df,
                timeframe_start=timeframe_start,
                timeframe_end=timeframe_end,
                forecast_periods=int(forecast_periods),
                department="All",
                model_choice=model_choice,
                uncertainty_method="conformal",
            )
        st.session_state["cmp_result"] = result
        st.session_state["cmp_scenarios"] = {
            "A": scenario_a_phases,
            "B": scenario_b_phases,
        }
    except Exception as run_error:
        st.error(f"Comparison run failed: {run_error}")
        st.stop()

result = st.session_state.get("cmp_result")
scenarios = st.session_state.get("cmp_scenarios")
if result is None or scenarios is None:
    st.info("Configure both scenarios, then click Run comparison in the sidebar.")
    st.stop()

baseline_df = result.future_forecast_df
scenario_a_df = apply_capacity_phases_to_future_df(baseline_df, scenarios["A"])
scenario_b_df = apply_capacity_phases_to_future_df(baseline_df, scenarios["B"])

summary_a = _scenario_summary(scenarios["A"])
summary_b = _scenario_summary(scenarios["B"])
st.caption(f"Scenario A: {summary_a}")
st.caption(f"Scenario B: {summary_b}")

# --- Chart ---
st.subheader("Monthly forecast comparison")
fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=baseline_df["ds"], y=baseline_df["yhat_original"],
        mode="lines", name="Baseline (no adjustments)",
        line=dict(color="gray", dash="dot"),
    )
)
fig.add_trace(
    go.Scatter(
        x=scenario_a_df["ds"], y=scenario_a_df["yhat_adjusted"],
        mode="lines+markers", name="Scenario A",
    )
)
fig.add_trace(
    go.Scatter(
        x=scenario_b_df["ds"], y=scenario_b_df["yhat_adjusted"],
        mode="lines+markers", name="Scenario B",
    )
)
fig.update_layout(hovermode="x unified", xaxis_title="Date", yaxis_title="Visits")
st.plotly_chart(fig, width="stretch")

# --- Comparison tables ---
comparison_df = pd.DataFrame(
    {
        "ds": baseline_df["ds"],
        "Fiscal_Year": baseline_df["Fiscal_Year"],
        "forecast_month": baseline_df["forecast_month"],
        "Baseline": baseline_df["yhat_original"].round(1),
        "Scenario_A": scenario_a_df["yhat_adjusted"].round(1),
        "Scenario_B": scenario_b_df["yhat_adjusted"].round(1),
    }
)
comparison_df["A_minus_Baseline"] = (
    comparison_df["Scenario_A"] - comparison_df["Baseline"]
).round(1)
comparison_df["B_minus_Baseline"] = (
    comparison_df["Scenario_B"] - comparison_df["Baseline"]
).round(1)
comparison_df["A_minus_B"] = (
    comparison_df["Scenario_A"] - comparison_df["Scenario_B"]
).round(1)

st.subheader("Monthly detail")
st.dataframe(comparison_df, width="stretch")

fy_summary = (
    comparison_df.groupby("Fiscal_Year", as_index=False)[
        ["Baseline", "Scenario_A", "Scenario_B"]
    ]
    .sum()
    .round(0)
)
fy_summary["A_minus_Baseline"] = fy_summary["Scenario_A"] - fy_summary["Baseline"]
fy_summary["B_minus_Baseline"] = fy_summary["Scenario_B"] - fy_summary["Baseline"]
fy_summary["A_minus_B"] = fy_summary["Scenario_A"] - fy_summary["Scenario_B"]

st.subheader("Fiscal year totals")
st.dataframe(fy_summary, width="stretch")

# --- Export ---
definitions_df = pd.DataFrame(
    [
        {"Scenario": "A", "Definition": summary_a},
        {"Scenario": "B", "Definition": summary_b},
        {
            "Scenario": "Model",
            "Definition": getattr(result, "selected_model", "prophet"),
        },
    ]
)
export_buffer = _io.BytesIO()
with pd.ExcelWriter(export_buffer, engine="openpyxl") as writer:
    comparison_df.to_excel(writer, sheet_name="Monthly_Comparison", index=False)
    fy_summary.to_excel(writer, sheet_name="FY_Summary", index=False)
    definitions_df.to_excel(writer, sheet_name="Scenario_Definitions", index=False)
st.download_button(
    "Download scenario comparison (Excel)",
    data=export_buffer.getvalue(),
    file_name="scenario_comparison.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    key="cmp_download_xlsx",
)
st.download_button(
    "Download monthly comparison (CSV)",
    data=comparison_df.to_csv(index=False).encode("utf-8"),
    file_name="scenario_comparison.csv",
    mime="text/csv",
    key="cmp_download_csv",
)
