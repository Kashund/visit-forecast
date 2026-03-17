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
    styler = display_df.style.format(
        {
            "candidate_value": "{:.2f}",
            "primary_score_value": "{:.4f}",
            "tie_breaker_value": "{:.4f}",
        },
        na_rep="n/a",
    )

    def _row_style(row: pd.Series):
        selected_style = (
            "font-weight: bold; background-color: rgba(46, 125, 50, 0.14);"
            if bool(row.get("selected", False))
            else ""
        )
        return [selected_style] * len(row)

    return styler.apply(_row_style, axis=1)


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
    interval_width=float(controls["interval_width"]),
    tuning_mode=str(controls["tuning_mode"]),
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

    used_forecast_inputs = {
        **common,
        "changepoint_prior_scale": result.selected_changepoint_prior_scale,
        "interval_width": result.selected_interval_width,
        "tuning_mode": result.tuning_mode,
    }

    st.session_state["forecast_result"] = result
    st.session_state["forecast_inputs"] = used_forecast_inputs
    st.session_state["forecast_source_mode"] = controls["source_mode"]
    scenario_summary = _build_phase_scenario_summary(controls["capacity_phases"])
    st.session_state["forecast_scenario_summary"] = scenario_summary

    # Append to history (keep last 20)
    entry = {
        "run_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source_mode": controls["source_mode"],
        "scenario_summary": scenario_summary,
        **common,
        "tuning_mode": result.tuning_mode,
        "changepoint_prior_scale": result.selected_changepoint_prior_scale,
        "interval_width": result.selected_interval_width,
        "tuning_primary_metric": result.tuning_primary_metric,
        "tuning_primary_score": result.tuning_primary_score,
        "MAPE": float(result.performance_metrics.get("MAPE", float("nan"))),
        "MAE": float(result.performance_metrics.get("MAE", float("nan"))),
        "RMSE": float(result.performance_metrics.get("RMSE", float("nan"))),
        "departments_included": result.department_info,
    }
    hist = st.session_state["forecast_history"]
    hist.insert(0, entry)
    st.session_state["forecast_history"] = hist[:20]
    st.session_state["forecast_results_section"] = "📉 Forecast"

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
        "Auto-tuned Interval",
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
    if result.tuning_note:
        st.caption(result.tuning_note)
    if result.tuning_diagnostics_df is not None and not result.tuning_diagnostics_df.empty:
        with st.expander("Auto-tune diagnostics", expanded=False):
            st.dataframe(
                _style_tuning_diagnostics_table(result.tuning_diagnostics_df),
                use_container_width=True,
            )


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
  - If interval_width=0.90, *ideal* long-run coverage is near **0.90**.  
  - Coverage much **lower** ⇒ intervals too narrow / underestimating uncertainty.  
  - Coverage much **higher** ⇒ intervals too wide / overly conservative.

**What “good” looks like (practical):**
- MAPE: aim for **<20%** for operational planning, **<10%** if you need tight staffing decisions.
- MAE/RMSE: aim for monthly error that is a small slice of typical monthly volume (e.g., **<10% of mean**).
- Coverage: aim close to your interval width (e.g., **~0.90** for 90% intervals).
"""
    )

with st.expander("Chart guide: what each graph is telling you"):
    st.markdown(
        """- **Forecast (Original vs Adjusted)**: point forecasts for future months.  
  - **Original** = model’s baseline forecast.  
  - **Adjusted** = capacity add/loss applied only in the selected month window.  
  - **Shaded window** highlights months where the adjustment was applied.

- **Confidence interval band**: uncertainty around the adjusted forecast.  
  - If Prophet interval bounds exist, the band reflects **interval_width**.  
  - Otherwise, the app approximates a band using RMSE.

- **Error bands (±MAE / ±RMSE)**: show typical miss size around the adjusted line.  
  - These bands are based on monthly forecast error, so think of them as visits above/below a typical month-level prediction.
  - Use this to sanity-check whether the forecast is “tight enough” for your decision.

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
    ci_choice = st.radio(
        "Confidence interval",
        ["95% (≈ ±1.96·RMSE)", "68% (≈ ±1·RMSE)"],
        horizontal=True,
        key="ci_choice",
    )
    ci_mode_val = "rmse_95" if ci_choice.startswith("95%") else "rmse_68"

    fig = forecast_line_chart(
        result.future_forecast_df,
        mae=mae_v,
        rmse=rmse_v,
        ci_mode=ci_mode_val,
        source_interval_width=st.session_state.get("forecast_inputs", {}).get(
            "interval_width"
        ),
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
            "- **coverage** should be close to your interval width (e.g., ~0.90 for 90% intervals).\n"
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
                "interval_width"
            ),
        )
        st.dataframe(_style_prophet_cv_table(cv_display_df), use_container_width=True)
        st.caption(
            "Indicators: coverage within +/-0.03 of interval width = On target, "
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
