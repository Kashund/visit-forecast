from __future__ import annotations

from bootstrap import ensure_project_paths
import streamlit as st
import pandas as pd
from datetime import datetime

ensure_project_paths()

from visit_forecast import forecast_visits
from visit_forecast.cv import add_prophet_cv_indicators
from visit_forecast.fiscal import (
    aggregate_summary_by_period,
    append_fiscal_year_totals,
)
from components.sidebar import sidebar_controls
from components.charts import forecast_line_chart, fiscal_bar_chart
from state import apply_forecast_run_to_session

# Optional Prophet diagnostics helpers
try:
    from prophet.diagnostics import cross_validation, performance_metrics  # noqa: F401
except Exception:
    pass


st.set_page_config(page_title="Run Forecast", page_icon="🚀", layout="wide")
st.title("🚀 Run Forecast")


def _timeseries_to_pd(ts) -> pd.DataFrame:
    if hasattr(ts, "pd_dataframe"):
        return ts.pd_dataframe()
    if hasattr(ts, "to_dataframe"):
        return ts.to_dataframe()
    if hasattr(ts, "pandas_dataframe"):
        return ts.pandas_dataframe()
    raise AttributeError(
        "Cannot convert TimeSeries to DataFrame for this Darts version."
    )


def _score_label(metric_name: str, value: float, baseline: float | None = None):
    if metric_name.upper() == "MAPE":
        v = value * 100.0 if value <= 1.0 else value
        if v < 10:
            return ("Optimal", "🟢")
        if v < 20:
            return ("Good", "🟩")
        if v < 50:
            return ("Fair", "🟨")
        return ("Poor", "🟥")

    if baseline is None or baseline <= 0:
        return ("—", "⚪️")

    norm = value / baseline
    if norm < 0.05:
        return ("Optimal", "🟢")
    if norm < 0.10:
        return ("Good", "🟩")
    if norm < 0.20:
        return ("Fair", "🟨")
    return ("Poor", "🟥")


def _coverage_check(value: float, target_coverage: float) -> str:
    if pd.isna(value):
        return "Unavailable"
    gap = abs(float(value) - float(target_coverage))
    if gap <= 0.03:
        return "On target"
    if gap <= 0.08:
        return "Watch"
    return "Off target"


def _build_phase_scenario_summary(capacity_phases: list[dict] | None) -> str:
    if not capacity_phases:
        return "No phases configured"

    enabled_phase_summaries = []
    for phase_number, phase in enumerate(capacity_phases, start=1):
        if not bool(phase.get("enabled", False)):
            continue

        mode_value = str(phase.get("mode", "")).lower()
        mode_label = "Loss" if mode_value == "loss" else "Add"
        phase_percent = float(phase.get("percent", 0.0))
        start_month = int(phase.get("start_month", 1))
        end_month = int(phase.get("end_month", start_month))
        enabled_phase_summaries.append(
            f"P{phase_number} {mode_label} {phase_percent:.1f}% M{start_month}-M{end_month}"
        )

    if not enabled_phase_summaries:
        return "No enabled phases"

    return " | ".join(enabled_phase_summaries)


def _style_fiscal_summary_table(summary_df: pd.DataFrame):
    display_df = summary_df.copy()
    formatted_numeric_columns = [
        column_name
        for column_name in display_df.select_dtypes(include="number").columns
        if column_name not in {"Fiscal_Year", "Fiscal_Quarter"}
    ]

    styler = display_df.style.format(
        {column_name: "{:.1f}" for column_name in formatted_numeric_columns}
    )

    if "Fiscal_Period_Label" not in display_df.columns:
        return styler

    def _row_style(row: pd.Series):
        is_total_row = str(row.get("Fiscal_Period_Label", "")).endswith("Total")
        style = "font-weight: bold" if is_total_row else ""
        return [style] * len(row)

    return styler.apply(_row_style, axis=1)


def _style_prophet_cv_table(cv_display_df: pd.DataFrame):
    display_df = cv_display_df.copy()

    numeric_formatters = {
        column_name: "{:.4f}"
        for column_name in display_df.columns
        if pd.api.types.is_numeric_dtype(display_df[column_name])
        and not pd.api.types.is_timedelta64_dtype(display_df[column_name])
        and column_name != "Horizon_Days"
    }
    if "Horizon_Days" in display_df.columns:
        numeric_formatters["Horizon_Days"] = "{:.0f}"
    timedelta_formatters = {
        column_name: (
            lambda value: ""
            if pd.isna(value)
            else f"{int(pd.to_timedelta(value).total_seconds() // 86400)} days"
        )
        for column_name in display_df.columns
        if pd.api.types.is_timedelta64_dtype(display_df[column_name])
    }

    status_styles = {
        "On target": "background-color: rgba(46, 125, 50, 0.18); font-weight: 600;",
        "Watch": "background-color: rgba(245, 124, 0, 0.18); font-weight: 600;",
        "Off target": "background-color: rgba(198, 40, 40, 0.18); font-weight: 600;",
        "Uniform": "background-color: rgba(46, 125, 50, 0.18); font-weight: 600;",
        "Some spikes": "background-color: rgba(245, 124, 0, 0.18); font-weight: 600;",
        "Big misses": "background-color: rgba(198, 40, 40, 0.18); font-weight: 600;",
        "Aligned": "background-color: rgba(46, 125, 50, 0.18); font-weight: 600;",
        "Some outliers": "background-color: rgba(245, 124, 0, 0.18); font-weight: 600;",
        "MDAPE more reliable": "background-color: rgba(198, 40, 40, 0.18); font-weight: 600;",
        "Baseline": "background-color: rgba(96, 125, 139, 0.16); font-weight: 600;",
        "Improving": "background-color: rgba(46, 125, 50, 0.18); font-weight: 600;",
        "Stable": "background-color: rgba(2, 136, 209, 0.16); font-weight: 600;",
        "Higher error": "background-color: rgba(245, 124, 0, 0.18); font-weight: 600;",
        "Sharp jump": "background-color: rgba(198, 40, 40, 0.18); font-weight: 600;",
    }
    status_columns = [
        column_name
        for column_name in [
            "Horizon_Trend",
            "Coverage_Check",
            "Error_Shape",
            "Outlier_Check",
        ]
        if column_name in display_df.columns
    ]

    styler = display_df.style.format({**numeric_formatters, **timedelta_formatters})
    for column_name in status_columns:
        styler = styler.map(lambda value: status_styles.get(str(value), ""), subset=[column_name])
    return styler


def _style_tuning_diagnostics_table(tuning_df: pd.DataFrame):
    display_df = tuning_df.copy()
    formatters = {
        "primary_score_value": "{:.4f}",
        "tie_breaker_value": "{:.4f}",
    }
    if "candidate_value" in display_df.columns and pd.api.types.is_numeric_dtype(
        display_df["candidate_value"]
    ):
        formatters["candidate_value"] = "{:.2f}"

    styler = display_df.style.format(formatters, na_rep="n/a")

    def _row_style(row: pd.Series):
        selected_style = (
            "font-weight: bold; background-color: rgba(46, 125, 50, 0.14);"
            if bool(row.get("selected", False))
            else ""
        )
        return [selected_style] * len(row)

    return styler.apply(_row_style, axis=1)


def _style_interval_validation_table(interval_df: pd.DataFrame):
    display_df = interval_df.copy()

    numeric_formatters = {
        column_name: "{:.4f}"
        for column_name in display_df.columns
        if pd.api.types.is_numeric_dtype(display_df[column_name])
        and column_name not in {"horizon_step", "Horizon_Days", "n_calibration_forecasts"}
    }
    for integer_column in ["horizon_step", "Horizon_Days", "n_calibration_forecasts"]:
        if integer_column in display_df.columns:
            numeric_formatters[integer_column] = "{:.0f}"

    status_styles = {
        "On target": "background-color: rgba(46, 125, 50, 0.18); font-weight: 600;",
        "Watch": "background-color: rgba(245, 124, 0, 0.18); font-weight: 600;",
        "Off target": "background-color: rgba(198, 40, 40, 0.18); font-weight: 600;",
        "Unavailable": "background-color: rgba(96, 125, 139, 0.16); font-weight: 600;",
    }

    styler = display_df.style.format(numeric_formatters, na_rep="n/a")
    for column_name in ["Coverage_Check"]:
        if column_name in display_df.columns:
            styler = styler.map(
                lambda value: status_styles.get(str(value), ""),
                subset=[column_name],
            )
    return styler


# Initialize history store
if "forecast_history" not in st.session_state:
    st.session_state["forecast_history"] = []  # list[dict]


# --- Sidebar ---
try:
    controls, df_preview = sidebar_controls()
except Exception as e:
    st.error(f"Data source error: {e}")
    st.stop()

# Sidebar: clear history control
with st.sidebar.expander("Forecast run history", expanded=False):
    st.caption("Stores the last 20 runs (this browser session).")
    if st.button("Clear forecast history", key="clear_history"):
        st.session_state["forecast_history"] = []
        # also clear current result
        if "forecast_result" in st.session_state:
            del st.session_state["forecast_result"]
        st.success("Cleared.")
        st.stop()

# Preview
if df_preview is not None:
    st.caption("Preview (first 50 rows)")
    st.dataframe(df_preview.head(50), use_container_width=True)

common = dict(
    timeframe_start=controls["timeframe_start"],
    timeframe_end=controls["timeframe_end"],
    forecast_periods=controls["forecast_periods"],
    department=controls["department"],
    exclude_departments=controls["exclude_departments"] or None,
    capacity_phases=controls["capacity_phases"],
    changepoint_prior_scale=float(controls["changepoint_prior_scale"]),
    target_coverage=float(controls["target_coverage"]),
    uncertainty_method=str(controls["uncertainty_method"]),
    tuning_mode=str(controls["tuning_mode"]),
    joint_auto_tuning_enabled=bool(controls["joint_auto_tuning_enabled"]),
)

# --- Compute only when Run clicked ---
if controls["run"]:
    with st.spinner("Running forecast..."):
        if controls["source_mode"] == "Use synthetic sample":
            result = forecast_visits(use_synthetic=True, **common)
        else:
            if df_preview is None:
                st.error("No data provided. Upload a file or paste CSV values first.")
                st.stop()
            result = forecast_visits(df=df_preview, **common)

    scenario_summary = _build_phase_scenario_summary(controls["capacity_phases"])
    apply_forecast_run_to_session(
        st.session_state,
        common=common,
        result=result,
        source_mode=controls["source_mode"],
        scenario_summary=scenario_summary,
        run_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )

# If no stored result, prompt
if "forecast_result" not in st.session_state:
    st.info("Choose inputs in the sidebar and click **Run forecast**.")
    # Still show history if exists
    if st.session_state["forecast_history"]:
        st.subheader("Forecast run history (last 20)")
        st.dataframe(
            pd.DataFrame(st.session_state["forecast_history"]), use_container_width=True
        )
    st.stop()

result = st.session_state["forecast_result"]

# Baseline for normalized MAE/RMSE interpretation
series_df = _timeseries_to_pd(result.series).reset_index()
val_cols = [
    c for c in series_df.columns if c.lower() not in ("time", "ds", "date", "index")
]
baseline = None
if val_cols:
    baseline = float(
        pd.to_numeric(series_df[val_cols[0]], errors="coerce").dropna().mean()
    )

mape_v = float(result.performance_metrics.get("MAPE", float("nan")))
mae_v = float(result.performance_metrics.get("MAE", float("nan")))
rmse_v = float(result.performance_metrics.get("RMSE", float("nan")))

mape_label, mape_icon = _score_label("MAPE", mape_v, baseline)
mae_label, mae_icon = _score_label("MAE", mae_v, baseline)
rmse_label, rmse_icon = _score_label("RMSE", rmse_v, baseline)

# --- Validation panel ---
st.subheader("Forecast validation (backtesting)")
st.markdown(
    """These metrics are computed using **backtesting** on the model's **monthly** historical forecasts.  
- **MAPE** = average % error for each forecasted month (easier to compare across departments)  
- **MAE** = average absolute error in visit units for each forecasted month  
- **RMSE** = like MAE but penalizes big monthly misses more (in visit units per forecasted month)  
"""
)
st.caption(
    "This app forecasts monthly visit totals. Example: an MAE of 2.14 means the model is off by about 2.14 visits in a typical month, not 2.14 visits per quarter or per year."
)

c1, c2, c3 = st.columns(3)
c1.metric("MAPE", f"{mape_v:.4f}", f"{mape_icon} {mape_label}")
c2.metric("MAE", f"{mae_v:.4f}", f"{mae_icon} {mae_label}")
c3.metric("RMSE", f"{rmse_v:.4f}", f"{rmse_icon} {rmse_label}")

if baseline and baseline > 0:
    st.caption(
        f"Normalization baseline (mean actual visits): {baseline:.2f}  •  "
        f"MAE/mean={mae_v/baseline:.2%}  •  RMSE/mean={rmse_v/baseline:.2%}"
    )

st.caption(
    "MAPE, MAE, and RMSE score the point forecast. Changing uncertainty method or target coverage usually changes the interval bands and interval validation, not these point-error metrics, unless the selected changepoint setting also changes."
)

st.caption(f"Departments included: {result.department_info}")

if result.tuning_mode == "auto":
    t1, t2, t3 = st.columns(3)
    t1.metric(
        "Auto-tuned CP",
        (
            f"{result.selected_changepoint_prior_scale:.2f}"
            if result.selected_changepoint_prior_scale is not None
            else "n/a"
        ),
    )
    t2.metric(
        "Coverage / Interval",
        (
            f"{result.selected_interval_width:.2f}"
            if result.selected_interval_width is not None
            else "n/a"
        ),
    )
    tuning_metric_name = result.tuning_primary_metric or "Score"
    t3.metric(
        f"Auto-tune {tuning_metric_name}",
        (
            f"{result.tuning_primary_score:.4f}"
            if result.tuning_primary_score is not None
            else "n/a"
        ),
    )
    if getattr(result, "requested_uncertainty_method", "") == "auto":
        st.caption(
            "Auto evaluation mode: joint search"
            if getattr(result, "joint_auto_tuning_enabled", False)
            else "Auto evaluation mode: separate passes for changepoint tuning and uncertainty selection"
        )

if result.tuning_note:
    st.caption(result.tuning_note)
if result.tuning_diagnostics_df is not None and not result.tuning_diagnostics_df.empty:
    with st.expander("Auto-tune diagnostics", expanded=False):
        st.dataframe(
            _style_tuning_diagnostics_table(result.tuning_diagnostics_df),
            use_container_width=True,
        )

st.markdown("---")
st.subheader("Interval validation")

interval_summary = getattr(result, "interval_summary_metrics", {}) or {}
interval_diagnostics_df = getattr(result, "interval_diagnostics_df", None)
interval_source = str(interval_summary.get("interval_source_used", result.uncertainty_method))
target_coverage = float(interval_summary.get("target_coverage", result.target_coverage))
empirical_coverage = interval_summary.get("empirical_coverage_overall", float("nan"))
coverage_status = _coverage_check(empirical_coverage, target_coverage)
requested_uncertainty_method = getattr(
    result,
    "requested_uncertainty_method",
    interval_summary.get("requested_uncertainty_method", result.uncertainty_method),
)

i1, i2, i3, i4 = st.columns(4)
i1.metric("Interval source", interval_source.capitalize())
i2.metric("Target coverage", f"{target_coverage:.2%}")
i3.metric(
    "Empirical coverage",
    f"{empirical_coverage:.2%}" if pd.notna(empirical_coverage) else "n/a",
)
i4.metric("Coverage status", coverage_status)

if requested_uncertainty_method != result.uncertainty_method:
    st.caption(
        f"Requested uncertainty method: {requested_uncertainty_method}  •  "
        f"Selected for this run: {result.uncertainty_method}"
    )

avg_interval_width = interval_summary.get("avg_interval_width")
median_interval_width = interval_summary.get("median_interval_width")
interval_width_over_mean_actual = interval_summary.get("interval_width_over_mean_actual")
fallback_used = bool(interval_summary.get("fallback_used", False))
n_calibration_forecasts = int(interval_summary.get("n_calibration_forecasts", 0) or 0)

st.caption(
    f"Calibration rows: {n_calibration_forecasts}  •  "
    f"Average interval width: {avg_interval_width:.2f}" if pd.notna(avg_interval_width) else
    f"Calibration rows: {n_calibration_forecasts}  •  Average interval width: n/a"
)
st.caption(
    f"Median interval width: {median_interval_width:.2f}  •  "
    f"Width / mean actual: {interval_width_over_mean_actual:.2%}"
    if pd.notna(median_interval_width) and pd.notna(interval_width_over_mean_actual)
    else "Median interval width: n/a  •  Width / mean actual: n/a"
)
if interval_source == "conformal":
    st.caption(
        "Conformal intervals are calibrated from rolling forecast residuals. They target empirical coverage under stable historical error behavior."
    )
    if fallback_used:
        st.caption(
            "Fallback used for at least one horizon because there were fewer than 8 calibration residuals for that step."
        )
else:
    st.caption(
        "Prophet interval coverage is summarized from Prophet cross-validation metrics. Detailed Prophet-native diagnostics remain in the CV section below."
    )

if interval_diagnostics_df is not None and not interval_diagnostics_df.empty:
    display_interval_df = interval_diagnostics_df.copy()
    if "empirical_coverage" in display_interval_df.columns:
        display_interval_df["Coverage_Check"] = display_interval_df[
            "empirical_coverage"
        ].map(lambda value: _coverage_check(value, target_coverage))
    preferred_interval_columns = [
        "horizon_step",
        "Horizon_Days",
        "horizon",
        "empirical_coverage",
        "target_coverage",
        "Coverage_Check",
        "avg_interval_width",
        "median_interval_width",
        "n_calibration_forecasts",
        "fallback_used",
    ]
    ordered_interval_columns = [
        column_name
        for column_name in preferred_interval_columns
        if column_name in display_interval_df.columns
    ]
    ordered_interval_columns.extend(
        [
            column_name
            for column_name in display_interval_df.columns
            if column_name not in ordered_interval_columns
        ]
    )
    st.dataframe(
        _style_interval_validation_table(
            display_interval_df[ordered_interval_columns]
        ),
        use_container_width=True,
    )
else:
    st.info("Interval validation details are not available for this run.")

st.markdown("---")
st.subheader("How to interpret these metrics & charts")

with st.expander(
    "Metric glossary + quick interpretation (backtesting & CV)", expanded=False
):
    st.markdown(
        """**Backtesting metrics (computed on historical forecasts):**

- **MAPE (Mean Absolute Percentage Error)**: average absolute % error for each forecasted month.  
  - **Lower is better.** Rough heuristic: **<10% excellent**, **10–20% good**, **20–50% fair**, **>50% poor**.  
  - Watch out when actuals are near zero (MAPE can blow up).

- **MAE (Mean Absolute Error)**: average absolute error in *visit units per forecasted month*.  
  - **Lower is better.** Interpreting MAE depends on your volume scale.  
  - Example: MAE = 2.14 means a typical monthly forecast miss of about 2.14 visits.
  - Helpful rule: compare MAE to the mean monthly volume (we show MAE/mean above).

- **RMSE (Root Mean Squared Error)**: like MAE but penalizes large monthly misses more.  
  - **Lower is better.** RMSE > MAE when you have occasional big errors (spikes).
  - Units are also visits per forecasted month.

**Prophet cross‑validation metrics (performance_metrics):**

- **mse / rmse / mae**: error magnitude for each forecasted month at a given horizon (units = visits per month).  
  - rmse penalizes big misses more than mae.

- **mape / mdape**: percent-based errors (unitless).  
  - These are also evaluated on monthly forecast points.
  - **mdape** is the median version (more robust to outliers).

- **smape**: symmetric MAPE (tries to be more stable than MAPE when values are small).  
  - **Lower is better.**

- **coverage**: fraction of actuals that fall inside Prophet’s uncertainty interval.  
  - If target coverage is 0.90, *ideal* long-run coverage is near **0.90**.  
  - Coverage much **lower** ⇒ intervals too narrow / underestimating uncertainty.  
  - Coverage much **higher** ⇒ intervals too wide / overly conservative.

**Interval types in this app:**

- **Prophet interval**: model-based uncertainty bounds from Prophet.  
  - Best interpreted with the Prophet CV section below.

- **Conformal interval**: residual-calibrated bounds from rolling historical forecasts.  
  - Best interpreted with the **Interval validation** panel above.
  - These intervals aim to match the selected target coverage empirically, assuming future errors behave similarly to recent historical errors.

**What “good” looks like (practical):**
- MAPE: aim for **<20%** for operational planning, **<10%** if you need tight staffing decisions.
- MAE/RMSE: aim for monthly error that is a small slice of typical monthly volume (e.g., **<10% of mean**).
- Coverage: aim close to your target coverage (e.g., **~0.90** for 90% intervals).
"""
    )

with st.expander("Chart guide: what each graph is telling you"):
    st.markdown(
        """- **Forecast (Original vs Adjusted)**: point forecasts for future months.  
  - **Original** = model’s baseline forecast.  
  - **Adjusted** = capacity add/loss applied only in the selected month window.  
  - **Shaded window** highlights months where the adjustment was applied.

- **Forecast interval band**: uncertainty around the adjusted forecast.  
  - In **Prophet interval** mode, the band is model-based.  
  - In **Conformal calibrated** mode, the band is built from rolling forecast residuals to target the selected coverage.  
  - If no explicit interval is available, the chart falls back to an RMSE-based approximation.

- **Historical error bands (±MAE / ±RMSE)**: show typical miss size around the adjusted line.  
  - These bands are based on monthly forecast error, so think of them as visits above/below a typical month-level prediction.
  - Use these as context, not as calibrated coverage intervals.

- **Fiscal Summary**: aggregates monthly totals into fiscal years (Apr–Mar).  
  - Helpful for annual targets and capacity planning at a higher level.
  - Important: the fiscal tables are aggregated views of monthly forecasts; the MAE/RMSE/MAPE validation metrics are still month-based.

- **Prophet CV plots (metric vs horizon)**: how forecast error grows as you predict further out.  
  - You usually expect error to increase with horizon.  
  - A sudden jump can indicate regime changes, strong seasonality shifts, or insufficient data.
"""
    )

result_sections = [
    "📉 Forecast",
    "📅 Future Table",
    "🏛️ Fiscal Summary",
    "🧾 Run History",
    "📊 Prophet CV",
    "🧩 Prophet Plots",
]
if "forecast_results_section" not in st.session_state:
    st.session_state["forecast_results_section"] = result_sections[0]

selected_results_section = st.segmented_control(
    "Results section",
    options=result_sections,
    key="forecast_results_section",
    selection_mode="single",
)

if selected_results_section == "📉 Forecast":
    active_interval_source = (
        (getattr(result, "interval_summary_metrics", {}) or {}).get(
            "interval_source_used", result.uncertainty_method
        )
    )
    fig = forecast_line_chart(
        result.future_forecast_df,
        mae=mae_v,
        rmse=rmse_v,
        target_coverage=result.target_coverage,
        interval_source=str(active_interval_source),
        phase_metadata=st.session_state.get("forecast_inputs", {}).get(
            "capacity_phases"
        ),
    )
    if fig is None:
        st.warning("No future dates found beyond the last historical date.")
    else:
        st.plotly_chart(fig, use_container_width=True)

elif selected_results_section == "📅 Future Table":
    scenario_summary = st.session_state.get(
        "forecast_scenario_summary",
        _build_phase_scenario_summary(
            st.session_state.get("forecast_inputs", {}).get("capacity_phases")
        ),
    )
    future_table_df = result.future_forecast_df.copy()
    future_table_df["scenario_summary"] = scenario_summary
    st.dataframe(future_table_df, use_container_width=True)
    st.download_button(
        "Download future forecast CSV",
        data=future_table_df.to_csv(index=False).encode("utf-8"),
        file_name="future_forecast.csv",
        mime="text/csv",
    )

elif selected_results_section == "🏛️ Fiscal Summary":
    fiscal_summary_mode = st.radio(
        "Fiscal summary granularity",
        ["Fiscal Year", "Fiscal Quarter"],
        horizontal=True,
        key="fiscal_summary_granularity",
    )

    selected_fiscal_summary = result.fiscal_summary
    chart_fiscal_summary = selected_fiscal_summary
    selected_period_column = "Fiscal_Year"
    selected_export_filename = "fiscal_summary_year.csv"

    if fiscal_summary_mode == "Fiscal Quarter":
        series_dataframe = _timeseries_to_pd(result.series).reset_index()
        datetime_column_candidates = [
            column_name
            for column_name in ["ds", "time", "date", "index"]
            if column_name in series_dataframe.columns
        ]
        if not datetime_column_candidates:
            st.error("Could not identify a datetime column for fiscal aggregation.")
            st.stop()

        value_column_candidates = [
            column_name
            for column_name in series_dataframe.columns
            if column_name not in datetime_column_candidates
        ]
        if not value_column_candidates:
            st.error("Could not identify a value column for fiscal aggregation.")
            st.stop()

        historical_dataframe = series_dataframe.rename(
            columns={
                datetime_column_candidates[0]: "ds",
                value_column_candidates[0]: "y",
            }
        )[["ds", "y"]]

        chart_fiscal_summary = aggregate_summary_by_period(
            historical_dataframe,
            result.future_forecast_df,
            period="quarter",
        )
        selected_fiscal_summary = append_fiscal_year_totals(chart_fiscal_summary)
        selected_period_column = "Fiscal_Period_Label"
        selected_export_filename = "fiscal_summary_quarter.csv"

    st.dataframe(
        _style_fiscal_summary_table(selected_fiscal_summary),
        use_container_width=True,
    )
    fig2 = fiscal_bar_chart(
        chart_fiscal_summary, period_column=selected_period_column
    )
    if fig2 is not None:
        st.plotly_chart(fig2, use_container_width=True)

    st.download_button(
        "Download fiscal summary CSV",
        data=selected_fiscal_summary.to_csv(index=False).encode("utf-8"),
        file_name=selected_export_filename,
        mime="text/csv",
    )

elif selected_results_section == "🧾 Run History":
    st.subheader("Forecast run history (last 20)")
    if not st.session_state["forecast_history"]:
        st.info("No runs yet.")
    else:
        hist_df = pd.DataFrame(st.session_state["forecast_history"])
        st.dataframe(hist_df, use_container_width=True)
        st.download_button(
            "Download run history CSV",
            data=hist_df.to_csv(index=False).encode("utf-8"),
            file_name="forecast_run_history.csv",
            mime="text/csv",
        )

elif selected_results_section == "📊 Prophet CV":
    st.subheader("Prophet cross-validation (prophet.diagnostics)")
    st.caption(
        "Cross-validation simulates forecasting from multiple cutoffs to estimate error across different horizons."
    )
    with st.expander("Interpretation tips (CV)", expanded=False):
        st.markdown(
            "- **Horizon** = how far ahead the model is predicting. Errors usually increase with horizon.\n"
            "- **coverage** should be close to your target coverage (e.g., ~0.90 for 90% intervals).\n"
            "- If **rmse** is close to **mae**, errors are fairly uniform. If rmse >> mae, you have big misses.\n"
            "- Use **mdape** when outliers or spikes distort mape.\n"
        )

    cv_metrics = getattr(result, "prophet_cv_metrics_df", None)
    cv_raw = getattr(result, "prophet_cv_raw_df", None)

    if cv_metrics is None or getattr(cv_metrics, "empty", True):
        st.info(
            "Cross-validation metrics not available (often due to insufficient history for the default initial/period/horizon)."
        )
        st.caption("Defaults: initial=730 days, period=180 days, horizon=365 days.")
    else:
        st.caption(
            "Prophet-native cross-validation metrics (performance_metrics output)."
        )
        cv_display_df = add_prophet_cv_indicators(
            cv_metrics.head(50),
            interval_width=st.session_state.get("forecast_inputs", {}).get(
                "target_coverage"
            ),
        )
        st.dataframe(_style_prophet_cv_table(cv_display_df), use_container_width=True)
        st.caption(
            "Indicators: coverage within +/-0.03 of target coverage = On target, "
            "+/-0.08 = Watch; RMSE/MAE <= 1.15 = Uniform, <= 1.40 = Some spikes, "
            "> 1.40 = Big misses; MAPE/MDAPE <= 1.15 = Aligned, <= 1.50 = Some outliers, "
            "> 1.50 = MDAPE more reliable."
        )

        # Plot common metrics vs horizon
        import plotly.express as px

        plot_cols = [
            c
            for c in ["mae", "rmse", "mape", "mdape", "smape", "coverage"]
            if c in cv_metrics.columns
        ]
        if "horizon" in cv_metrics.columns and plot_cols:
            dfp = cv_metrics.copy()
            try:
                dfp["horizon_days"] = pd.to_timedelta(dfp["horizon"]).dt.days
                xcol = "horizon_days"
                xlabel = "Horizon (days)"
            except Exception:
                xcol = "horizon"
                xlabel = "Horizon"

            for c in plot_cols:
                figm = px.line(
                    dfp, x=xcol, y=c, markers=True, title=f"{c.upper()} vs {xlabel}"
                )
                st.plotly_chart(figm, use_container_width=True)

        st.download_button(
            "Download CV metrics CSV",
            data=cv_metrics.to_csv(index=False).encode("utf-8"),
            file_name="prophet_cv_metrics.csv",
            mime="text/csv",
        )

    if cv_raw is not None and not getattr(cv_raw, "empty", True):
        with st.expander("Raw CV rows (df_cv)"):
            st.dataframe(cv_raw.head(200), use_container_width=True)
            st.download_button(
                "Download CV raw CSV",
                data=cv_raw.to_csv(index=False).encode("utf-8"),
                file_name="prophet_cv_raw.csv",
                mime="text/csv",
            )

elif selected_results_section == "🧩 Prophet Plots":
    st.subheader("Prophet plots (forecast + components)")
    prophet_fc = getattr(result, "prophet_forecast_df", None)
    if prophet_fc is None or getattr(prophet_fc, "empty", True):
        st.info("Prophet forecast dataframe not available for plotting.")
    else:
        import plotly.graph_objects as go

        dfp = prophet_fc.copy()
        dfp["ds"] = pd.to_datetime(dfp["ds"])

        fig = go.Figure()

        # Confidence interval from Prophet output (responds to interval_width)
        if "yhat_upper" in dfp.columns and "yhat_lower" in dfp.columns:
            fig.add_trace(
                go.Scatter(
                    x=dfp["ds"],
                    y=dfp["yhat_upper"],
                    mode="lines",
                    name="yhat_upper",
                    line=dict(dash="dot"),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=dfp["ds"],
                    y=dfp["yhat_lower"],
                    mode="lines",
                    name="yhat_lower",
                    line=dict(dash="dot"),
                    fill="tonexty",
                    fillcolor="rgba(0,0,0,0.10)",
                )
            )

        fig.add_trace(go.Scatter(x=dfp["ds"], y=dfp["yhat"], mode="lines", name="yhat"))

        fig.update_layout(
            title="Prophet forecast (native output)",
            hovermode="x unified",
            xaxis_title="Date",
            yaxis_title="Visits",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.caption(
            "Components (trend/seasonality) require the fitted Prophet model object to render exactly like Prophet. If you'd like, we can persist the fitted model in session_state to render full component plots."
        )
