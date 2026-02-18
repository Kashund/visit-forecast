from __future__ import annotations

from typing import List, Optional

import pandas as pd

from .fiscal import add_fiscal_year, fiscal_summary
from .io import load_data, prepare_dataframe
from .model_prophet import backtest, fit_and_forecast, prophet_future_intervals, prophet_full_forecast_df, prophet_cross_validation_metrics
from .types import ForecastResult


def _timeseries_to_df(ts) -> pd.DataFrame:
    """Convert a Darts TimeSeries to a pandas DataFrame across Darts versions.

    Darts has used different method names across releases:
      - pd_dataframe()
      - to_dataframe()
      - pandas_dataframe()
    This helper tries the common ones.
    """
    if hasattr(ts, "pd_dataframe"):
        return ts.pd_dataframe()
    if hasattr(ts, "to_dataframe"):
        return ts.to_dataframe()
    if hasattr(ts, "pandas_dataframe"):
        return ts.pandas_dataframe()
    raise AttributeError("Unsupported Darts TimeSeries version: cannot convert to DataFrame.")




def build_future_forecast_df(
    forecast,
    last_hist_date: pd.Timestamp,
    adjustment_mode: str,
    adjustment_percent: float,
    adjustment_start_month: int,
    adjustment_end_month: int,
) -> pd.DataFrame:
    """Build a standardized future forecast DataFrame with *selective* capacity adjustments.

    Parameters
    ----------
    adjustment_mode:
        'loss' or 'add'
    adjustment_percent:
        percent magnitude (e.g., 10 => 10% loss or add)
    adjustment_start_month / adjustment_end_month:
        1-indexed forecast month range (relative to first future month) where adjustment applies, inclusive.

    Output columns
    --------------
    ds, yhat, yhat_original, yhat_adjusted, adj_applied, Fiscal_Year, forecast_month
    """
    fc_df = _timeseries_to_df(forecast).reset_index()

    # Identify time column
    time_col = None
    for cand in ["ds", "time", "index", "Date", "date"]:
        if cand in fc_df.columns:
            time_col = cand
            break
    if time_col is None:
        time_col = fc_df.columns[0]

    fc_df = fc_df.rename(columns={time_col: "ds"})
    fc_df["ds"] = pd.to_datetime(fc_df["ds"], errors="coerce")
    if fc_df["ds"].isna().all():
        raise ValueError("Could not parse forecast time column into datetimes.")

    # Identify value column
    value_cols = [c for c in fc_df.columns if c != "ds"]
    if not value_cols:
        raise ValueError("No forecast value columns found after conversion.")
    if len(value_cols) == 1:
        value_col = value_cols[0]
    else:
        numeric = [c for c in value_cols if pd.api.types.is_numeric_dtype(fc_df[c])]
        value_col = numeric[0] if numeric else value_cols[0]

    fc_df = fc_df.rename(columns={value_col: "yhat"})
    fc_df["yhat_original"] = fc_df["yhat"]

    # Keep only future rows
    fut = fc_df[fc_df["ds"] > last_hist_date].copy()
    fut = fut.sort_values("ds").reset_index(drop=True)

    # 1-indexed forecast month
    fut["forecast_month"] = fut.index + 1

    if fut.empty:
        fut["yhat_adjusted"] = fut["yhat_original"]
        fut["adj_applied"] = False
        fut = add_fiscal_year(fut, "ds")
        return fut

    # Clamp month range
    adjustment_start_month = max(1, int(adjustment_start_month))
    adjustment_end_month = max(adjustment_start_month, int(adjustment_end_month))
    max_m = int(fut["forecast_month"].max())
    adjustment_start_month = min(adjustment_start_month, max_m)
    adjustment_end_month = min(adjustment_end_month, max_m)

    in_range = fut["forecast_month"].between(adjustment_start_month, adjustment_end_month, inclusive="both")
    fut["adj_applied"] = in_range

    p = float(adjustment_percent) / 100.0
    mode = (adjustment_mode or "loss").lower()
    factor = 1.0 + p if mode in ["add", "increase", "gain"] else 1.0 - p

    fut["yhat_adjusted"] = fut["yhat_original"]
    fut.loc[in_range, "yhat_adjusted"] = fut.loc[in_range, "yhat_original"] * factor

    fut = add_fiscal_year(fut, "ds")
    return fut


def forecast_visits(
    *,
    use_synthetic: bool = False,
    filename: Optional[str] = None,
    df: Optional[pd.DataFrame] = None,
    timeframe_start: str = "2017-04-01",
    timeframe_end: str = "2025-03-01",
    forecast_periods: int = 12,
    department: str = "All",
    adjustment_mode: str = "loss",
    adjustment_percent: float = 0.0,
    adjustment_start_month: int = 1,
    adjustment_end_month: int = 12,
    exclude_departments: Optional[List[str]] = None,
    changepoint_prior_scale: float = 0.05,
    interval_width: float = 0.90,
) -> ForecastResult:
    if forecast_periods <= 0:
        raise ValueError("forecast_periods must be > 0")

    df_raw = df if df is not None else load_data(use_synthetic, filename, timeframe_start, timeframe_end)

    df_prepared, dept_info = prepare_dataframe(
        df_raw=df_raw,
        timeframe_start=timeframe_start,
        timeframe_end=timeframe_end,
        department=department,
        exclude_departments=exclude_departments,
    )

    df_prepared = add_fiscal_year(df_prepared, "ds")

    series, forecast, model = fit_and_forecast(
        df_prepared=df_prepared,
        forecast_periods=forecast_periods,
        changepoint_prior_scale=changepoint_prior_scale,
        interval_width=interval_width,
    )

    metrics = backtest(
        model=model,
        series=series,
        forecast_horizon=forecast_periods,
        start=0.7,
        stride=forecast_periods,
    )

    last_hist_date = pd.to_datetime(df_prepared["ds"].max())
    future_df = build_future_forecast_df(
        forecast,
        last_hist_date,
        adjustment_mode=adjustment_mode,
        adjustment_percent=adjustment_percent,
        adjustment_start_month=adjustment_start_month,
        adjustment_end_month=adjustment_end_month,
    )

    # Uncertainty intervals from Prophet (interval_width controls these)
    try:
        intervals = prophet_future_intervals(
            model,
            last_hist_date=last_hist_date,
            forecast_periods=forecast_periods,
        )
        # Expected columns: ds, yhat_lower, yhat_upper
        if intervals is not None and not intervals.empty:
            future_df = future_df.merge(intervals, on="ds", how="left")

            # If capacity adjustment applies, scale bounds to follow the adjusted line
            if {"adj_applied", "yhat_lower", "yhat_upper"}.issubset(future_df.columns):
                f = (future_df["yhat_adjusted"] / future_df["yhat_original"]).fillna(1.0)
                mask = future_df["adj_applied"] == True
                future_df.loc[mask, "yhat_lower"] = future_df.loc[mask, "yhat_lower"] * f.loc[mask]
                future_df.loc[mask, "yhat_upper"] = future_df.loc[mask, "yhat_upper"] * f.loc[mask]
    except Exception:
        # Intervals are optional; UI will fall back to RMSE-based approximation
        pass

    # Prophet-native forecast dataframe (for Prophet plots)
    prophet_fc_df = None
    try:
        prophet_fc_df = prophet_full_forecast_df(model, forecast_periods=forecast_periods, freq="MS")
    except Exception:
        prophet_fc_df = None

    # Prophet cross-validation metrics (optional; may be None if not enough data)
    cv_raw, cv_metrics = prophet_cross_validation_metrics(
        model,
        initial="730 days",
        period="180 days",
        horizon="365 days",
    )

    fy = fiscal_summary(df_prepared, future_df)
    return ForecastResult(
        series=series,
        forecast=forecast,
        future_forecast_df=future_df,
        fiscal_summary=fy,
        performance_metrics=metrics,
        department_info=dept_info,
        prophet_forecast_df=prophet_fc_df,
        prophet_cv_metrics_df=cv_metrics,
        prophet_cv_raw_df=cv_raw,
        )

