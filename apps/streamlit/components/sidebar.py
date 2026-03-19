from __future__ import annotations

import hashlib
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


def _infer_timeframe_bounds(df_preview: pd.DataFrame | None) -> tuple[str | None, str | None]:
    if df_preview is None or df_preview.empty or "Date" not in df_preview.columns:
        return None, None

    date_series = pd.to_datetime(df_preview["Date"], errors="coerce").dropna()
    if date_series.empty:
        return None, None

    return (
        date_series.min().strftime("%Y-%m-%d"),
        date_series.max().strftime("%Y-%m-%d"),
    )


def _source_signature(
    source_mode: str,
    uploaded,
    pasted_text: str | None,
    df_preview: pd.DataFrame | None,
) -> str | None:
    detected_start, detected_end = _infer_timeframe_bounds(df_preview)
    if source_mode == "Upload file" and uploaded is not None:
        return (
            f"upload:{getattr(uploaded, 'name', '')}:{detected_start}:{detected_end}:"
            f"{len(df_preview) if df_preview is not None else 0}"
        )

    if source_mode == "Paste values (CSV)" and pasted_text and pasted_text.strip():
        digest = hashlib.sha1(pasted_text.strip().encode("utf-8")).hexdigest()
        return f"paste:{digest}:{detected_start}:{detected_end}"

    return None


def _sync_timeframe_defaults(
    session_state,
    *,
    source_mode: str,
    uploaded,
    pasted_text: str | None,
    df_preview: pd.DataFrame | None,
) -> None:
    detected_start, detected_end = _infer_timeframe_bounds(df_preview)
    signature = _source_signature(source_mode, uploaded, pasted_text, df_preview)
    if signature is None or detected_start is None or detected_end is None:
        return

    if session_state.get("_timeframe_source_signature") == signature:
        return

    session_state["start"] = detected_start
    session_state["end"] = detected_end
    session_state["_timeframe_source_signature"] = signature


def _resolve_joint_auto_tuning_state(
    session_state,
    *,
    tuning_mode: str,
    uncertainty_method: str,
) -> tuple[bool, bool]:
    should_show = tuning_mode == "auto" and uncertainty_method == "auto"
    if not should_show:
        session_state["joint_auto_tuning_enabled"] = False
        return False, False

    if "joint_auto_tuning_enabled" not in session_state:
        session_state["joint_auto_tuning_enabled"] = False

    return True, bool(session_state["joint_auto_tuning_enabled"])


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
        st.sidebar.caption(
            "Synthetic data includes: Cardiology, Oncology, Neurology (monthly)."
        )

    _sync_timeframe_defaults(
        st.session_state,
        source_mode=source_mode,
        uploaded=uploaded,
        pasted_text=pasted_text,
        df_preview=df_preview,
    )

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
    st.sidebar.markdown("**Capacity adjustment phases**")

    capacity_phases = []
    default_end = min(12, forecast_periods)
    default_month_range = (1, default_end)

    for phase_number in range(1, 5):
        with st.sidebar.expander(f"Phase {phase_number}", expanded=False):
            phase_enabled = st.checkbox(
                "Enabled",
                value=False,
                key=f"phase_{phase_number}_enabled",
            )

            phase_mode_label = st.selectbox(
                "Mode",
                ["Capacity loss", "Capacity add"],
                index=0,
                key=f"phase_{phase_number}_mode",
                help="Loss reduces forecasted visits; Add increases forecasted visits.",
            )
            phase_mode = "loss" if phase_mode_label == "Capacity loss" else "add"

            phase_percent = float(
                st.slider(
                    "Percent (%)",
                    0,
                    50,
                    0,
                    key=f"phase_{phase_number}_percent",
                    help="Percent magnitude applied in the selected month range.",
                )
            )

            phase_month_range = st.slider(
                "Apply between forecast months",
                min_value=1,
                max_value=forecast_periods,
                value=default_month_range,
                step=1,
                key=f"phase_{phase_number}_between",
                help="1 = first forecasted future month. Applies within this inclusive range.",
            )

            capacity_phases.append(
                {
                    "enabled": phase_enabled,
                    "mode": phase_mode,
                    "percent": phase_percent,
                    "start_month": int(phase_month_range[0]),
                    "end_month": int(phase_month_range[1]),
                }
            )

    st.sidebar.divider()
    st.sidebar.markdown("**Uncertainty**")

    uncertainty_method_label = st.sidebar.radio(
        "Uncertainty method",
        ["Auto-select", "Prophet interval", "Conformal calibrated"],
        horizontal=True,
        key="uncertainty_method",
        help="Auto-select compares Prophet and conformal interval validation. Prophet interval uses model-based bounds. Conformal calibrated uses rolling forecast residuals to calibrate empirical coverage.",
    )
    if uncertainty_method_label == "Conformal calibrated":
        uncertainty_method = "conformal"
    elif uncertainty_method_label == "Auto-select":
        uncertainty_method = "auto"
    else:
        uncertainty_method = "prophet"

    target_coverage = float(
        st.sidebar.slider(
            "Target coverage",
            0.50,
            0.99,
            0.90,
            key="target_coverage",
            help="Desired interval coverage. In Prophet mode this is the Prophet interval width; in conformal mode it is the empirical target coverage.",
        )
    )

    st.sidebar.divider()
    st.sidebar.markdown("**Model tuning (advanced)**")

    parameter_selection_mode = st.sidebar.radio(
        "Parameter selection",
        ["Auto-tune", "Manual"],
        index=0,
        horizontal=True,
        key="parameter_selection_mode",
        help="Manual uses the sliders below. Auto-tune searches a moderate candidate set for this run.",
    )
    tuning_mode = "auto" if parameter_selection_mode == "Auto-tune" else "manual"

    if tuning_mode == "auto":
        st.sidebar.caption(
            "Auto-tune searches changepoint prior scale "
            "[0.01, 0.03, 0.05, 0.10, 0.20] by balancing MAPE and RMSE. "
            "If Uncertainty method is Auto-select, the app performs a joint search across changepoint and uncertainty method. "
            "If Uncertainty method is set manually, the current interval / target coverage input is preserved for the run."
        )

    show_joint_auto_tuning_toggle, joint_auto_tuning_enabled = (
        _resolve_joint_auto_tuning_state(
            st.session_state,
            tuning_mode=tuning_mode,
            uncertainty_method=uncertainty_method,
        )
    )

    changepoint_prior_scale = float(
        st.sidebar.slider(
            "Changepoint prior scale",
            0.01,
            0.50,
            0.05,
            key="cp",
            help="Higher values allow the trend to change more aggressively (more flexible, higher overfit risk).",
            disabled=tuning_mode == "auto",
        )
    )

    if show_joint_auto_tuning_toggle:
        toggle_fn = getattr(st.sidebar, "toggle", st.sidebar.checkbox)
        joint_auto_tuning_enabled = bool(
            toggle_fn(
                "Joint Auto Tuning Evaluation",
                key="joint_auto_tuning_enabled",
                help=(
                    "On evaluates changepoint prior scale and uncertainty method together "
                    "in one joint search. Off auto-tunes changepoint first, then "
                    "auto-selects the uncertainty method in a separate pass."
                ),
            )
        )
        st.sidebar.caption(
            "Displayed only when both Model tuning and Uncertainty method are set to auto."
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
        "capacity_phases": capacity_phases,
        "changepoint_prior_scale": changepoint_prior_scale,
        "target_coverage": target_coverage,
        "interval_width": target_coverage,
        "uncertainty_method": uncertainty_method,
        "tuning_mode": tuning_mode,
        "joint_auto_tuning_enabled": joint_auto_tuning_enabled,
        "run": run,
    }

    return controls, df_preview
