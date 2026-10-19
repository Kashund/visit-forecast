from __future__ import annotations

from bootstrap import ensure_project_paths
import plotly.graph_objects as go
import streamlit as st

ensure_project_paths()

from visit_forecast.service import (
    apply_capacity_phases_to_future_df,
    forecast_visits,
)

st.set_page_config(page_title="Scenario Comparison", page_icon="⚖️", layout="wide")
st.title("⚖️ Scenario Comparison")
st.caption(
    "Fit the forecast once, then compare two capacity phase configurations "
    "side by side against the unadjusted baseline. Both scenarios reuse the "
    "same fitted model, so differences are purely the phase assumptions."
)

from components.phases import phase_editor
from state import configuration_fingerprint, record_auxiliary_run
from visit_forecast.exports import scenario_tables, excel_bytes
from visit_forecast.provenance import build_run_manifest
from components.data_sources import select_data_source, remember_data, preview_data
from components.sidebar import PrefixedState, _sync_timeframe_defaults


def _phase_editor(scenario_key: str, forecast_periods: int) -> list[dict]:
    return phase_editor(f"{scenario_key}_phase", forecast_periods)


def _scenario_summary(phases: list[dict]) -> str:
    parts = [
        f"{p['mode']} {p['percent']:.0f}% (months {p['start_month']}-{p['end_month']})"
        for p in phases
        if p["enabled"] and p["percent"] > 0
    ]
    return "; ".join(parts) if parts else "No adjustments"


from components.session import persistent_page

with persistent_page("cmp_"):
    # --- Sidebar: baseline forecast controls ---
    with st.sidebar:
        st.markdown("**Baseline forecast**")
        try:
            selected_source = select_data_source(
                "cmp_", st.sidebar, default_mode="Use synthetic sample"
            )
            source_mode = selected_source.mode
            _sync_timeframe_defaults(
                PrefixedState(st.session_state, "cmp_"),
                source_mode=source_mode,
                uploaded=selected_source.uploaded,
                pasted_text=selected_source.pasted_text,
                df_preview=selected_source.frame,
            )
        except Exception as exc:
            st.error(f"Could not read data: {exc}")
            st.stop()
        timeframe_start = st.text_input(
            "Start (YYYY-MM-DD)", value="2020-04-01", key="cmp_start"
        )
        timeframe_end = st.text_input(
            "End (YYYY-MM-DD)", value="2025-03-01", key="cmp_end"
        )
        forecast_periods = st.slider(
            "Forecast months", min_value=1, max_value=36, value=12, key="cmp_months"
        )
        model_label_map = {
            "Auto (compare models)": "auto",
            "Prophet": "prophet",
            "Holt-Winters (ETS)": "ets",
            "Theta": "theta",
            "SARIMA": "arima",
            "Seasonal naive (baseline)": "seasonal_naive",
        }
        model_choice_label = st.selectbox(
            "Model", list(model_label_map.keys()), index=0, key="cmp_model_choice"
        )
        model_choice = model_label_map[model_choice_label]
        use_calendar_covariates = st.checkbox(
            "Calendar covariates (business days per month)", key="cmp_covariates"
        )
        department = st.text_input(
            "Department (All or exact name)", value="All", key="cmp_department"
        )
        exclusions = st.text_input(
            "Exclude departments (comma separated)", key="cmp_exclusions"
        )
        run_clicked = st.button("Run comparison", key="cmp_run_btn")

    # --- Scenario phase editors ---
    column_a, column_b = st.columns(2)
    with column_a:
        st.subheader("Scenario A")
        scenario_a_phases = _phase_editor("cmp_a", int(forecast_periods))
    with column_b:
        st.subheader("Scenario B")
        scenario_b_phases = _phase_editor("cmp_b", int(forecast_periods))

    baseline_inputs = dict(
        timeframe_start=timeframe_start,
        timeframe_end=timeframe_end,
        forecast_periods=int(forecast_periods),
        department=department,
        model_choice=model_choice,
        exclude_departments=[d.strip() for d in exclusions.split(",") if d.strip()],
        use_calendar_covariates=use_calendar_covariates,
        uncertainty_method="conformal",
    )
    raw_df = None
    try:
        raw_df = selected_source.resolve(timeframe_start, timeframe_end)
        if raw_df is not None:
            remember_data(
                raw_df, selected_source, baseline_inputs, "Scenario Comparison"
            )
            preview_data(raw_df, selected_source.label)
    except Exception as exc:
        st.error(f"Could not read data: {exc}")

    if run_clicked:
        try:
            if raw_df is None:
                raise ValueError("Choose a dataset before running the comparison.")
            signature = configuration_fingerprint(baseline_inputs, raw_df)
            if signature != st.session_state.get("cmp_baseline_signature"):
                with st.spinner("Fitting baseline forecast..."):
                    result = forecast_visits(df=raw_df, **baseline_inputs)
                st.session_state["cmp_result"] = result
                st.session_state["cmp_baseline_signature"] = signature
            else:
                result = st.session_state["cmp_result"]
            manifest = build_run_manifest(
                raw_df,
                {
                    **baseline_inputs,
                    "scenarios": {"A": scenario_a_phases, "B": scenario_b_phases},
                },
                result.evaluation_summary,
            )
            manifest["baseline_run"] = result.run_manifest
            st.session_state["cmp_manifest"] = manifest
            remember_data(
                raw_df,
                selected_source,
                baseline_inputs,
                "Scenario Comparison",
                ran=True,
            )
            record_auxiliary_run(st.session_state, "scenarios", manifest)
            st.session_state["latest_forecast"] = dict(
                workflow="Scenario Comparison", run_id=manifest["run_id"]
            )
            st.session_state["cmp_scenarios"] = {
                "A": scenario_a_phases,
                "B": scenario_b_phases,
            }
        except Exception as run_error:
            st.error(f"Comparison run failed: {run_error}")

    result = st.session_state.get("cmp_result")
    scenarios = st.session_state.get("cmp_scenarios")
    if result is None or scenarios is None:
        st.info("Configure both scenarios, then click Run comparison in the sidebar.")
        st.stop()

    if scenarios != {"A": scenario_a_phases, "B": scenario_b_phases}:
        st.warning(
            "Scenario controls have changed. Click Run comparison to apply; the baseline is reused when its settings are unchanged."
        )
    if configuration_fingerprint(baseline_inputs, raw_df) != st.session_state.get(
        "cmp_baseline_signature"
    ):
        st.warning(
            "Baseline settings have changed. Displayed results use the previous run; click Run comparison to update."
        )

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
            x=baseline_df["ds"],
            y=baseline_df["yhat_original"],
            mode="lines",
            name="Baseline (no adjustments)",
            line=dict(color="gray", dash="dot"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=scenario_a_df["ds"],
            y=scenario_a_df["yhat_adjusted"],
            mode="lines+markers",
            name="Scenario A",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=scenario_b_df["ds"],
            y=scenario_b_df["yhat_adjusted"],
            mode="lines+markers",
            name="Scenario B",
        )
    )
    fig.update_layout(hovermode="x unified", xaxis_title="Date", yaxis_title="Visits")
    st.plotly_chart(fig, width="stretch")

    # --- Comparison tables ---
    comparison_df, fy_summary, definitions_df = scenario_tables(
        result, scenarios["A"], scenarios["B"]
    )
    st.subheader("Monthly detail")
    st.dataframe(comparison_df, width="stretch")

    st.subheader("Fiscal year totals")
    st.dataframe(fy_summary, width="stretch")

    # --- Export ---
    export = excel_bytes(
        {
            "Monthly_Comparison": comparison_df,
            "FY_Summary": fy_summary,
            "Scenario_Definitions": definitions_df,
        },
        st.session_state.get("cmp_manifest"),
    )
    st.download_button(
        "Download scenario comparison (Excel)",
        export,
        "scenario_comparison.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key="cmp_download_xlsx",
    )
    st.download_button(
        "Download monthly comparison (CSV)",
        comparison_df.to_csv(index=False).encode(),
        "scenario_comparison.csv",
        mime="text/csv",
        key="cmp_download_csv",
    )
    if st.session_state.get("history_save_error"):
        st.warning(st.session_state["history_save_error"])
