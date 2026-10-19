"""Shared forecasting controls and panels, with optional per-department fitting."""

from __future__ import annotations
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from components.sidebar import sidebar_controls
from components.results import render_forecast_result, _build_phase_scenario_summary
from components.data_sources import remember_data, preview_data
from visit_forecast.service import forecast_departments
from visit_forecast.provenance import build_run_manifest, configuration_fingerprint
from visit_forecast.exports import excel_bytes
from state import (
    ensure_history_loaded,
    history_persistence_enabled,
    clear_persisted_history,
    cached_forecast,
    apply_forecast_run_to_session,
    record_auxiliary_run,
)


def forecast_settings(controls):
    names = [
        "timeframe_start",
        "timeframe_end",
        "forecast_periods",
        "department",
        "exclude_departments",
        "capacity_phases",
        "model_choice",
        "use_calendar_covariates",
        "changepoint_prior_scale",
        "target_coverage",
        "uncertainty_method",
        "tuning_mode",
        "joint_auto_tuning_enabled",
        "auto_candidates",
    ]
    return {name: controls[name] for name in names}


def history_sidebar(prefix):
    with st.sidebar.expander("Forecast run history", expanded=False):
        st.caption(
            "Stores the last 20 runs in this browser session. "
            + (
                "Local disk persistence is enabled."
                if history_persistence_enabled()
                else "Disk persistence is disabled."
            )
        )
        if st.button("Clear forecast history", key=f"{prefix}clear_history"):
            st.session_state["forecast_history"] = []
            st.session_state.pop("forecast_cache", None)
            clear_persisted_history()
            st.success(
                "Forecast history cleared. Completed forecasts remain available."
            )


def render_department_overview(bundle, manifest):
    with st.expander("All department outputs", expanded=False):
        st.subheader("Per-department model selection and accuracy")
        st.dataframe(bundle.summary_df, width="stretch")
        fig = go.Figure()
        for name, group in bundle.combined_future_df.groupby("Department", sort=True):
            fig.add_trace(
                go.Scatter(
                    x=group.ds, y=group.yhat_adjusted, mode="lines+markers", name=name
                )
            )
        fig.update_layout(
            hovermode="x unified", xaxis_title="Date", yaxis_title="Forecasted visits"
        )
        st.plotly_chart(fig, width="stretch")
        st.caption(
            "Totals sum department point forecasts. No aggregate interval coverage is implied."
        )
        failures = pd.DataFrame(
            [{"Department": d, "Reason": r} for d, r in bundle.failures.items()],
            columns=["Department", "Reason"],
        )
        diagnostics = [
            r.tuning_diagnostics_df.assign(Department=d)
            for d, r in bundle.results.items()
            if r.tuning_diagnostics_df is not None
        ]
        full_forecasts = pd.concat(
            [
                r.future_forecast_df.assign(Department=d)
                for d, r in bundle.results.items()
            ],
            ignore_index=True,
        )
        fiscal = pd.concat(
            [r.fiscal_summary.assign(Department=d) for d, r in bundle.results.items()],
            ignore_index=True,
        )
        export = excel_bytes(
            {
                "Model_Summary": bundle.summary_df,
                "Forecasts_By_Department": bundle.combined_future_df,
                "Full_Forecasts": full_forecasts,
                "Fiscal_Summaries": fiscal,
                "Failures": failures,
                "Candidate_Diagnostics": pd.concat(diagnostics, ignore_index=True)
                if diagnostics
                else pd.DataFrame(),
            },
            manifest,
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


def render_forecast_page(*, department_mode=False):
    prefix = "dept_" if department_mode else ""
    result_key = "dept_bundle" if department_mode else "forecast_result"
    workflow = "Department Forecasts" if department_mode else "Run Forecast"
    ensure_history_loaded(st.session_state)
    try:
        controls, raw = sidebar_controls(prefix=prefix)
    except Exception as exc:
        st.error(f"Input configuration error: {exc}")
        st.stop()
    history_sidebar(prefix)
    common = forecast_settings(controls)
    if raw is not None:
        remember_data(raw, controls["source"], common, workflow)
        preview_data(raw, controls["source"].label)
    signature = configuration_fingerprint(common, raw)
    if controls["run"] and raw is None:
        st.error(
            "No data provided. Upload a file, paste CSV values, or choose a sample first."
        )
    if controls["run"] and raw is not None:
        try:
            with st.spinner("Running forecast and validation..."):
                progress = st.empty()
                if department_mode:
                    settings = {k: v for k, v in common.items() if k != "department"}
                    settings["departments"] = (
                        None
                        if common["department"] == "All"
                        else [common["department"]]
                    )
                    bundle = forecast_departments(
                        df=raw, **settings, progress_callback=progress.caption
                    )
                    st.session_state["dept_bundle"] = bundle
                    manifest = build_run_manifest(
                        raw, dict(workflow="departments", **common)
                    )
                    manifest["department_runs"] = {
                        d: r.run_manifest for d, r in bundle.results.items()
                    }
                    manifest["failures"] = bundle.failures
                    st.session_state["dept_manifest"] = manifest
                    st.session_state["dept_inputs"] = common
                    record_auxiliary_run(
                        st.session_state,
                        "departments",
                        manifest,
                        failures=bundle.failures,
                    )
                else:
                    result = cached_forecast(
                        st.session_state,
                        raw,
                        common,
                        progress_callback=progress.caption,
                    )
                    apply_forecast_run_to_session(
                        st.session_state,
                        common=common,
                        result=result,
                        source_mode=controls["source_mode"],
                        scenario_summary=_build_phase_scenario_summary(
                            common["capacity_phases"]
                        ),
                    )
                progress.empty()
            remember_data(raw, controls["source"], common, workflow, ran=True)
            st.session_state[f"{prefix}forecast_input_signature"] = signature
            manifest = (
                st.session_state["dept_manifest"]
                if department_mode
                else st.session_state["forecast_result"].run_manifest
            )
            st.session_state["latest_forecast"] = dict(
                workflow=workflow, run_id=manifest["run_id"]
            )
        except Exception as exc:
            st.error(f"Forecast failed: {exc}")
    if result_key in st.session_state and signature != st.session_state.get(
        f"{prefix}forecast_input_signature"
    ):
        st.warning(
            "Inputs changed. Results and exports still describe the previous run. Click Run forecast to update."
        )
    if st.session_state.get("history_save_error"):
        st.warning(st.session_state["history_save_error"])
    if result_key not in st.session_state:
        st.info("Choose inputs in the sidebar and click Run forecast.")
        if st.session_state["forecast_history"]:
            st.subheader("Forecast run history (last 20)")
            st.dataframe(
                pd.DataFrame(st.session_state["forecast_history"]), width="stretch"
            )
        return
    if department_mode:
        bundle = st.session_state["dept_bundle"]
        if bundle.failures:
            st.warning(
                "Some departments failed; combined totals and exports are partial."
            )
            for name, message in bundle.failures.items():
                st.warning(f"{name}: {message}")
        options = list(bundle.results)
        if st.session_state.get("dept_output") not in options:
            st.session_state["dept_output"] = options[0]
        selected = st.selectbox(
            "Department output",
            options,
            key="dept_output",
            help="Switch all result panels to this department without refitting.",
        )
        st.caption(
            "All panels below use the selected department. Switch departments or views without running the models again."
        )
        result = bundle.results[selected]
        saved_inputs = st.session_state.get("dept_inputs")
        if saved_inputs is None:
            # Keep pre-refactor session results readable without borrowing edited controls.
            settings = (result.run_manifest or {}).get("settings", {})
            saved_inputs = dict(
                capacity_phases=settings.get("capacity_phases", []),
                target_coverage=result.target_coverage,
            )
        used_inputs = {**saved_inputs, "department": selected}
        render_forecast_result(
            result, used_inputs, namespace=prefix, output_label=selected
        )
        render_department_overview(bundle, st.session_state["dept_manifest"])
    else:
        render_forecast_result(
            st.session_state["forecast_result"], st.session_state["forecast_inputs"]
        )
