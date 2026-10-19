"""Session controls and durable copies of page widget values."""

from contextlib import contextmanager
import streamlit as st


def _widget_keys(prefix):
    names = {
        "source_mode",
        "pasted_csv",
        "sample_dataset",
        "start",
        "end",
        "months",
        "model_choice",
        "include_prophet",
        "use_calendar_covariates",
        "uncertainty_method",
        "target_coverage",
        "parameter_selection_mode",
        "cp",
        "joint_auto_tuning_enabled",
        "dept",
        "exclude",
        "forecast_results_section",
        "fiscal_summary_granularity",
        "snapshot",
        "preview_rows",
        "output",
        "department",
        "exclusions",
        "covariates",
    }
    keys = {prefix + name for name in names}
    phases = [prefix + "phase"] if prefix != "cmp_" else ["cmp_a_phase", "cmp_b_phase"]
    for phase in phases:
        for number in range(1, 5):
            for field in ("enabled", "mode", "percent", "between"):
                keys.add(f"{phase}_{number}_{field}")
    return keys


@contextmanager
def persistent_page(prefix):
    """Keep widget values outside Streamlit's page-specific widget lifecycle.

    Button and uploader state cannot be restored through session_state. Source
    frames are stored separately; uploader keys are rotated when data is cleared.
    """
    saved = st.session_state.setdefault("page_controls", {}).get(prefix, {})
    for key, value in saved.items():
        if key not in st.session_state:
            st.session_state[key] = value
    refresh = session_sidebar()
    try:
        yield
    finally:
        controls = dict(st.session_state.get("page_controls", {}))
        controls[prefix] = {
            key: st.session_state[key]
            for key in _widget_keys(prefix)
            if key in st.session_state
        }
        st.session_state["page_controls"] = controls
        refresh()


def clear_data():
    """Remove all saved raw sources, including widget and preview copies."""
    for key in (
        "active_dataset",
        "shared_dataset",
        "current_input_data",
        "last_run_data",
        "forecast_data_snapshots",
        "forecast_cache",
        "source_seen",
        "cmp_source_fingerprint",
        "cmp_upload",
        "cmp_pasted",
    ):
        st.session_state.pop(key, None)
    for prefix in ("", "dept_", "check_", "cmp_"):
        for suffix in ("uploader", "pasted_csv", "_timeframe_source_signature"):
            st.session_state.pop(prefix + suffix, None)
        for key in list(st.session_state):
            if key.startswith(prefix + "uploader_"):
                del st.session_state[key]
        st.session_state[prefix + "source_mode"] = (
            "Current forecast data" if prefix == "check_" else "Upload file"
        )
    controls = st.session_state.get("page_controls", {})
    for saved in controls.values():
        for key in list(saved):
            if key.endswith(("source_mode", "pasted_csv", "snapshot")):
                del saved[key]
    st.session_state["page_controls"] = controls
    st.session_state["data_generation"] = st.session_state.get("data_generation", 0) + 1


def session_sidebar():
    with st.sidebar:
        st.subheader("Session data")
        data_status = st.empty()
        st.button("Clear data", key="clear_session_data", on_click=clear_data)
        st.caption(
            "Clears saved source data and previews. Completed forecasts remain available."
        )
        run_status = st.empty()
        st.divider()

    def refresh():
        active = st.session_state.get("active_dataset")
        data_status.caption(
            f"Active dataset: {active['label']} · {len(active['frame']):,} rows"
            if active
            else "No active dataset."
        )
        latest = st.session_state.get("latest_forecast")
        if latest:
            run_status.caption(
                f"Latest completed forecast: {latest['workflow']} · {latest['run_id'][:8]}"
            )

    refresh()
    return refresh
