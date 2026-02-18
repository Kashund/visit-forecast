from __future__ import annotations

from typing import Dict, Optional, Union, Sequence, List

import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.models import Prophet


def fit_and_forecast(
    df_prepared: pd.DataFrame,
    forecast_periods: int,
    changepoint_prior_scale: float,
    interval_width: float,
) -> tuple[TimeSeries, TimeSeries, Prophet]:
    """Fit Darts Prophet and produce a point forecast."""
    series = TimeSeries.from_dataframe(df_prepared, time_col="ds", value_cols="y")
    model = Prophet(changepoint_prior_scale=changepoint_prior_scale, interval_width=interval_width)
    model.fit(series)
    forecast = model.predict(forecast_periods)
    return series, forecast, model


def prophet_future_intervals(
    model: Prophet,
    last_hist_date: pd.Timestamp,
    forecast_periods: int,
    freq: str = "MS",
) -> pd.DataFrame:
    """Compute Prophet uncertainty intervals for future horizon.

    Uses the underlying Prophet model (model.model) to generate yhat_lower/yhat_upper.
    The interval width is controlled by the model's `interval_width` parameter.
    """
    start = (pd.to_datetime(last_hist_date) + pd.offsets.MonthBegin(1)).normalize()
    future_dates = pd.date_range(start=start, periods=int(forecast_periods), freq=freq)
    future = pd.DataFrame({"ds": future_dates})
    pred = model.model.predict(future)
    out = pred[["ds", "yhat_lower", "yhat_upper"]].copy()
    out["ds"] = pd.to_datetime(out["ds"])
    return out


def _timeseries_to_df(ts) -> pd.DataFrame:
    if hasattr(ts, "pd_dataframe"):
        return ts.pd_dataframe()
    if hasattr(ts, "to_dataframe"):
        return ts.to_dataframe()
    if hasattr(ts, "pandas_dataframe"):
        return ts.pandas_dataframe()
    raise AttributeError("Unsupported Darts TimeSeries version: cannot convert to DataFrame.")


def _collect_forecast_errors(
    actual: TimeSeries,
    forecasts,
) -> np.ndarray:
    """Return array of errors (forecast - actual) for backtest forecasts.

    `historical_forecasts` may return a TimeSeries or a list of TimeSeries depending on stride/settings.
    We align forecast timestamps to actual values.
    """
    actual_df = _timeseries_to_df(actual).reset_index()
    # time col name
    time_col = None
    for cand in ["ds", "time", "index", "Date", "date"]:
        if cand in actual_df.columns:
            time_col = cand
            break
    if time_col is None:
        time_col = actual_df.columns[0]
    actual_df = actual_df.rename(columns={time_col: "ds"})
    actual_df["ds"] = pd.to_datetime(actual_df["ds"])
    # value col
    val_cols = [c for c in actual_df.columns if c != "ds"]
    if not val_cols:
        raise ValueError("Actual series has no value column.")
    if len(val_cols) == 1:
        ycol = val_cols[0]
    else:
        numeric = [c for c in val_cols if pd.api.types.is_numeric_dtype(actual_df[c])]
        ycol = numeric[0] if numeric else val_cols[0]
    actual_map = actual_df.set_index("ds")[ycol]

    # normalize forecasts to list
    fc_list: List[TimeSeries] = forecasts if isinstance(forecasts, list) else [forecasts]
    errs = []
    for fc in fc_list:
        fc_df = _timeseries_to_df(fc).reset_index()
        # time col
        fc_time = None
        for cand in ["ds", "time", "index", "Date", "date"]:
            if cand in fc_df.columns:
                fc_time = cand
                break
        if fc_time is None:
            fc_time = fc_df.columns[0]
        fc_df = fc_df.rename(columns={fc_time: "ds"})
        fc_df["ds"] = pd.to_datetime(fc_df["ds"])
        # value col
        vcols = [c for c in fc_df.columns if c != "ds"]
        if not vcols:
            continue
        if len(vcols) == 1:
            vcol = vcols[0]
        else:
            numeric = [c for c in vcols if pd.api.types.is_numeric_dtype(fc_df[c])]
            vcol = numeric[0] if numeric else vcols[0]
        # align to actual
        for ds, yhat in zip(fc_df["ds"], fc_df[vcol]):
            if ds in actual_map.index and pd.notna(yhat) and pd.notna(actual_map.loc[ds]):
                errs.append(float(yhat) - float(actual_map.loc[ds]))
    return np.array(errs, dtype=float)


def backtest(
    model: Prophet,
    series: TimeSeries,
    forecast_horizon: int,
    start: Union[float, pd.Timestamp] = 0.7,
    stride: Optional[int] = None,
) -> Dict[str, float]:
    """Run historical forecasts and compute MAPE/MAE/RMSE.

    Note: Interval width does not affect point forecasts, only uncertainty bands.
    """
    stride = stride or forecast_horizon

    n = len(series)
    adj_start = start
    if isinstance(start, float):
        latest = max(0.0, 1.0 - (forecast_horizon + 1) / max(n, 1))
        adj_start = min(max(0.0, start), latest)

    try:
        bt = model.historical_forecasts(
            series,
            start=adj_start,
            forecast_horizon=forecast_horizon,
            stride=stride,
            overlap_end=False,
            verbose=False,
        )
    except ValueError:
        bt = model.historical_forecasts(
            series,
            start=adj_start if not isinstance(adj_start, float) else max(0.0, adj_start - 0.05),
            forecast_horizon=forecast_horizon,
            stride=stride,
            overlap_end=True,
            verbose=False,
        )

    errors = _collect_forecast_errors(series, bt)
    if errors.size == 0:
        return {"MAPE": float("nan"), "MAE": float("nan"), "RMSE": float("nan")}

    mae_v = float(np.mean(np.abs(errors)))
    rmse_v = float(np.sqrt(np.mean(errors ** 2)))

    # MAPE: mean(|e|/|y|) * 100
    actual_df = _timeseries_to_df(series).reset_index()
    time_col = next((c for c in ["ds","time","index","Date","date"] if c in actual_df.columns), actual_df.columns[0])
    actual_df = actual_df.rename(columns={time_col:"ds"})
    actual_df["ds"] = pd.to_datetime(actual_df["ds"])
    val_cols = [c for c in actual_df.columns if c!="ds"]
    ycol = val_cols[0]
    actual_map = actual_df.set_index("ds")[ycol]

    # recompute aligned actuals for same timestamps used in errors
    # easiest: reuse collector but also store actuals; for speed, approximate using mean absolute percentage
    # We'll rebuild from bt
    fc_list = bt if isinstance(bt, list) else [bt]
    ape = []
    for fc in fc_list:
        fc_df = _timeseries_to_df(fc).reset_index()
        fc_time = next((c for c in ["ds","time","index","Date","date"] if c in fc_df.columns), fc_df.columns[0])
        fc_df = fc_df.rename(columns={fc_time:"ds"})
        fc_df["ds"] = pd.to_datetime(fc_df["ds"])
        vcols = [c for c in fc_df.columns if c!="ds"]
        if not vcols: 
            continue
        vcol = vcols[0]
        for ds, yhat in zip(fc_df["ds"], fc_df[vcol]):
            if ds in actual_map.index:
                y = float(actual_map.loc[ds])
                if y != 0 and pd.notna(yhat):
                    ape.append(abs(float(yhat) - y) / abs(y))
    mape_v = float(np.mean(ape) * 100.0) if len(ape) else float("nan")

    return {"MAPE": mape_v, "MAE": mae_v, "RMSE": rmse_v}



def prophet_full_forecast_df(
    model: Prophet,
    forecast_periods: int,
    freq: str = "MS",
) -> pd.DataFrame:
    """Return Prophet-native forecast dataframe for history + future.

    This is used for Prophet plots (plot_plotly / plot_components).
    """
    # Prophet's make_future_dataframe includes history by default
    future = model.model.make_future_dataframe(periods=int(forecast_periods), freq=freq, include_history=True)
    forecast_df = model.model.predict(future)
    # Ensure datetime
    forecast_df["ds"] = pd.to_datetime(forecast_df["ds"])
    return forecast_df


def prophet_cross_validation_metrics(
    model: Prophet,
    initial: str = "730 days",
    period: str = "180 days",
    horizon: str = "365 days",
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """Run Prophet cross-validation and return (df_cv, df_p).

    Returns (None, None) if cross-validation can't run due to insufficient history.
    """
    try:
        from prophet.diagnostics import cross_validation, performance_metrics
    except Exception:
        return None, None

    # Prophet CV can fail if initial/horizon don't fit the data length. We'll catch and return None.
    try:
        df_cv = cross_validation(model.model, initial=initial, period=period, horizon=horizon)
        df_p = performance_metrics(df_cv)
        return df_cv, df_p
    except Exception:
        return None, None
