from __future__ import annotations

from typing import Dict, Optional, Union, Sequence, List, Any

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


def _standardize_series_dataframe(ts) -> pd.DataFrame:
    df = _timeseries_to_df(ts).reset_index()

    time_col = None
    for cand in ["ds", "time", "index", "Date", "date"]:
        if cand in df.columns:
            time_col = cand
            break
    if time_col is None:
        time_col = df.columns[0]

    df = df.rename(columns={time_col: "ds"})
    df["ds"] = pd.to_datetime(df["ds"])

    value_cols = [c for c in df.columns if c != "ds"]
    if not value_cols:
        raise ValueError("Series has no value column.")
    preferred_value_cols = [
        column_name
        for column_name in ["y", "yhat", "value", "Visits"]
        if column_name in value_cols
    ]
    if preferred_value_cols:
        value_col = preferred_value_cols[0]
    elif len(value_cols) == 1:
        value_col = value_cols[0]
    else:
        numeric = [c for c in value_cols if pd.api.types.is_numeric_dtype(df[c])]
        numeric = [c for c in numeric if c.lower() != "index"] or numeric
        value_col = numeric[0] if numeric else value_cols[0]

    standardized = df.rename(columns={value_col: "value"})[["ds", "value"]]
    standardized["value"] = pd.to_numeric(
        standardized["value"], errors="coerce"
    ).astype(float)
    return standardized


def _generate_historical_forecasts(
    model: Prophet,
    series: TimeSeries,
    forecast_horizon: int,
    start: Union[float, pd.Timestamp] = 0.7,
    stride: Optional[int] = None,
    last_points_only: bool = True,
):
    stride = stride or forecast_horizon

    n = len(series)
    adj_start = start
    if isinstance(start, float):
        latest = max(0.0, 1.0 - (forecast_horizon + 1) / max(n, 1))
        adj_start = min(max(0.0, start), latest)

    base_kwargs: dict[str, Any] = {
        "start": adj_start,
        "forecast_horizon": forecast_horizon,
        "stride": stride,
        "overlap_end": False,
        "verbose": False,
        "last_points_only": last_points_only,
    }

    try:
        return model.historical_forecasts(series, **base_kwargs)
    except ValueError:
        retry_start = (
            max(0.0, adj_start - 0.05) if isinstance(adj_start, float) else adj_start
        )
        return model.historical_forecasts(
            series,
            **{
                **base_kwargs,
                "start": retry_start,
                "overlap_end": True,
            },
        )


def _normalize_forecast_list(forecasts) -> List[TimeSeries]:
    if isinstance(forecasts, list):
        if forecasts and isinstance(forecasts[0], list):
            flattened: List[TimeSeries] = []
            for nested in forecasts:
                flattened.extend(nested)
            return flattened
        return forecasts
    return [forecasts]


def _collect_forecast_alignment_rows(
    actual: TimeSeries,
    forecasts,
) -> pd.DataFrame:
    """Return aligned forecast rows with residuals for backtest forecasts.

    `historical_forecasts` may return a TimeSeries or a list of TimeSeries depending on stride/settings.
    We align forecast timestamps to actual values.
    """
    actual_df = _standardize_series_dataframe(actual)
    actual_map = actual_df.set_index("ds")["value"]

    fc_list = _normalize_forecast_list(forecasts)
    rows: list[dict[str, Any]] = []
    for fc in fc_list:
        fc_df = _standardize_series_dataframe(fc).sort_values("ds").reset_index(drop=True)
        if fc_df.empty:
            continue

        cutoff = (fc_df["ds"].min() - pd.offsets.MonthBegin(1)).normalize()
        for horizon_step, (ds, yhat) in enumerate(
            zip(fc_df["ds"], fc_df["value"]), start=1
        ):
            if ds not in actual_map.index or pd.isna(yhat):
                continue

            actual_value = actual_map.loc[ds]
            if pd.isna(actual_value):
                continue

            predicted_value = float(yhat)
            actual_float = float(actual_value)
            residual = actual_float - predicted_value
            rows.append(
                {
                    "cutoff": cutoff,
                    "ds": ds,
                    "horizon_step": int(horizon_step),
                    "actual": actual_float,
                    "predicted": predicted_value,
                    "residual": residual,
                    "abs_residual": abs(residual),
                }
            )

    return pd.DataFrame(rows)


def collect_conformal_residuals(
    model: Prophet,
    series: TimeSeries,
    forecast_horizon: int,
    start: Union[float, pd.Timestamp] = 0.7,
    stride: int = 1,
) -> pd.DataFrame:
    forecasts = _generate_historical_forecasts(
        model=model,
        series=series,
        forecast_horizon=forecast_horizon,
        start=start,
        stride=stride,
        last_points_only=False,
    )
    residuals_df = _collect_forecast_alignment_rows(series, forecasts)
    if residuals_df.empty:
        return pd.DataFrame(
            columns=[
                "cutoff",
                "ds",
                "horizon_step",
                "actual",
                "predicted",
                "residual",
                "abs_residual",
            ]
        )
    return residuals_df.sort_values(["cutoff", "horizon_step", "ds"]).reset_index(
        drop=True
    )


def build_conformal_intervals(
    future_df: pd.DataFrame,
    residuals_df: pd.DataFrame,
    target_coverage: float,
    min_residuals_per_horizon: int = 8,
) -> tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    output_df = future_df.copy()
    output_df["yhat_lower_conformal"] = np.nan
    output_df["yhat_upper_conformal"] = np.nan

    if output_df.empty:
        diagnostics_df = pd.DataFrame(
            columns=[
                "horizon_step",
                "n_calibration_forecasts",
                "target_coverage",
                "empirical_coverage",
                "avg_interval_width",
                "median_interval_width",
                "fallback_used",
                "lower_offset",
                "upper_offset",
            ]
        )
        summary = {
            "empirical_coverage_overall": float("nan"),
            "empirical_coverage_by_horizon": {},
            "avg_interval_width": float("nan"),
            "median_interval_width": float("nan"),
            "interval_width_over_mean_actual": float("nan"),
            "n_calibration_forecasts": 0,
            "fallback_used": True,
        }
        return output_df, diagnostics_df, summary

    alpha = 1.0 - float(target_coverage)
    lower_quantile = max(0.0, alpha / 2.0)
    upper_quantile = min(1.0, 1.0 - (alpha / 2.0))

    abs_residuals = (
        pd.to_numeric(residuals_df.get("abs_residual"), errors="coerce")
        .dropna()
        .to_numpy(dtype=float)
    )
    pooled_abs_quantile = (
        float(np.quantile(abs_residuals, float(target_coverage)))
        if abs_residuals.size
        else float("nan")
    )

    diagnostics_rows: list[dict[str, Any]] = []
    horizon_offsets: dict[int, tuple[float, float, bool]] = {}

    max_horizon = int(output_df["forecast_month"].max())
    for horizon_step in range(1, max_horizon + 1):
        horizon_rows = residuals_df[
            pd.to_numeric(residuals_df["horizon_step"], errors="coerce").eq(horizon_step)
        ].copy()
        horizon_residuals = (
            pd.to_numeric(horizon_rows.get("residual"), errors="coerce")
            .dropna()
            .to_numpy(dtype=float)
        )

        fallback_used = False
        if horizon_residuals.size >= int(min_residuals_per_horizon):
            lower_offset = float(np.quantile(horizon_residuals, lower_quantile))
            upper_offset = float(np.quantile(horizon_residuals, upper_quantile))
        elif np.isfinite(pooled_abs_quantile):
            lower_offset = float(-pooled_abs_quantile)
            upper_offset = float(pooled_abs_quantile)
            fallback_used = True
        else:
            lower_offset = float("nan")
            upper_offset = float("nan")
            fallback_used = True

        horizon_offsets[horizon_step] = (lower_offset, upper_offset, fallback_used)

        if horizon_rows.empty or not np.isfinite(lower_offset) or not np.isfinite(upper_offset):
            empirical_coverage = float("nan")
            avg_interval_width = float("nan")
            median_interval_width = float("nan")
        else:
            lower_bound = np.clip(
                horizon_rows["predicted"].to_numpy(dtype=float) + lower_offset,
                a_min=0.0,
                a_max=None,
            )
            upper_bound = horizon_rows["predicted"].to_numpy(dtype=float) + upper_offset
            actual_values = horizon_rows["actual"].to_numpy(dtype=float)
            empirical_coverage = float(
                np.mean((actual_values >= lower_bound) & (actual_values <= upper_bound))
            )
            widths = upper_bound - lower_bound
            avg_interval_width = float(np.mean(widths))
            median_interval_width = float(np.median(widths))

        diagnostics_rows.append(
            {
                "horizon_step": horizon_step,
                "n_calibration_forecasts": int(horizon_rows.shape[0]),
                "target_coverage": float(target_coverage),
                "empirical_coverage": empirical_coverage,
                "avg_interval_width": avg_interval_width,
                "median_interval_width": median_interval_width,
                "fallback_used": bool(fallback_used),
                "lower_offset": lower_offset,
                "upper_offset": upper_offset,
            }
        )

        month_mask = output_df["forecast_month"].eq(horizon_step)
        base_forecast = pd.to_numeric(
            output_df.loc[month_mask, "yhat_original"], errors="coerce"
        )
        if np.isfinite(lower_offset):
            output_df.loc[month_mask, "yhat_lower_conformal"] = np.clip(
                base_forecast + lower_offset,
                a_min=0.0,
                a_max=None,
            )
        if np.isfinite(upper_offset):
            output_df.loc[month_mask, "yhat_upper_conformal"] = base_forecast + upper_offset

    diagnostics_df = pd.DataFrame(diagnostics_rows)

    if residuals_df.empty or diagnostics_df.empty:
        summary = {
            "empirical_coverage_overall": float("nan"),
            "empirical_coverage_by_horizon": {},
            "avg_interval_width": float("nan"),
            "median_interval_width": float("nan"),
            "interval_width_over_mean_actual": float("nan"),
            "n_calibration_forecasts": int(residuals_df.shape[0]),
            "fallback_used": True,
        }
        return output_df, diagnostics_df, summary

    summary_rows: list[dict[str, float]] = []
    for row in residuals_df.itertuples(index=False):
        lower_offset, upper_offset, _ = horizon_offsets.get(
            int(row.horizon_step), (float("nan"), float("nan"), True)
        )
        if not np.isfinite(lower_offset) or not np.isfinite(upper_offset):
            continue

        lower_bound = max(float(row.predicted) + lower_offset, 0.0)
        upper_bound = float(row.predicted) + upper_offset
        summary_rows.append(
            {
                "covered": float(lower_bound <= float(row.actual) <= upper_bound),
                "interval_width": float(upper_bound - lower_bound),
                "actual": float(row.actual),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    if summary_df.empty:
        empirical_coverage_overall = float("nan")
        avg_interval_width = float("nan")
        median_interval_width = float("nan")
        interval_width_over_mean_actual = float("nan")
    else:
        empirical_coverage_overall = float(summary_df["covered"].mean())
        avg_interval_width = float(summary_df["interval_width"].mean())
        median_interval_width = float(summary_df["interval_width"].median())
        mean_actual = float(summary_df["actual"].mean())
        interval_width_over_mean_actual = (
            float(avg_interval_width / mean_actual)
            if mean_actual > 0
            else float("nan")
        )

    coverage_by_horizon = (
        diagnostics_df.set_index("horizon_step")["empirical_coverage"]
        .dropna()
        .to_dict()
    )
    summary = {
        "empirical_coverage_overall": empirical_coverage_overall,
        "empirical_coverage_by_horizon": {
            int(horizon_step): float(coverage)
            for horizon_step, coverage in coverage_by_horizon.items()
        },
        "avg_interval_width": avg_interval_width,
        "median_interval_width": median_interval_width,
        "interval_width_over_mean_actual": interval_width_over_mean_actual,
        "n_calibration_forecasts": int(residuals_df.shape[0]),
        "fallback_used": bool(diagnostics_df["fallback_used"].any()),
    }
    return output_df, diagnostics_df, summary


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
    bt = _generate_historical_forecasts(
        model=model,
        series=series,
        forecast_horizon=forecast_horizon,
        start=start,
        stride=stride,
        last_points_only=True,
    )

    aligned_rows = _collect_forecast_alignment_rows(series, bt)
    if aligned_rows.empty:
        return {"MAPE": float("nan"), "MAE": float("nan"), "RMSE": float("nan")}

    abs_errors = aligned_rows["residual"].abs().to_numpy(dtype=float)
    mae_v = float(np.mean(abs_errors))
    rmse_v = float(np.sqrt(np.mean(np.square(aligned_rows["residual"].to_numpy(dtype=float)))))

    non_zero_actuals = aligned_rows["actual"].replace(0, np.nan)
    ape = (aligned_rows["residual"].abs() / non_zero_actuals.abs()).dropna()
    mape_v = float(ape.mean() * 100.0) if not ape.empty else float("nan")

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
