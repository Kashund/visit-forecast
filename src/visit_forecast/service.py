from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional

import pandas as pd

from .fiscal import add_fiscal_year, fiscal_summary
from .io import load_data, prepare_dataframe
from .model_prophet import (
    backtest,
    fit_and_forecast,
    prophet_future_intervals,
    prophet_full_forecast_df,
    prophet_cross_validation_metrics,
)
from .types import ForecastResult


@dataclass
class CapacityPhase:
    enabled: bool
    mode: str
    percent: float
    start_month: int
    end_month: int


def _normalize_capacity_phases(
    capacity_phases: Optional[List[CapacityPhase | dict[str, Any]]],
    *,
    forecast_periods: int,
) -> List[CapacityPhase]:
    """Normalize and validate capacity phase definitions.

    Overlap precedence: enabled phases are applied sequentially in the provided
    order using multiplicative factors.
    """
    if not capacity_phases:
        return []

    if len(capacity_phases) > 4:
        raise ValueError("A maximum of 4 capacity phases is supported.")

    normalized_phases: List[CapacityPhase] = []
    for index, phase in enumerate(capacity_phases, start=1):
        if isinstance(phase, CapacityPhase):
            normalized_phase = phase
        elif isinstance(phase, dict):
            normalized_phase = CapacityPhase(
                enabled=bool(phase.get("enabled", True)),
                mode=str(phase.get("mode", "loss")),
                percent=float(phase.get("percent", 0.0)),
                start_month=int(phase.get("start_month", 1)),
                end_month=int(phase.get("end_month", forecast_periods)),
            )
        else:
            raise ValueError(f"Phase {index} must be a CapacityPhase or dict.")

        _validate_capacity_phase(
            normalized_phase, forecast_periods=forecast_periods, index=index
        )
        normalized_phases.append(normalized_phase)

    return normalized_phases


def _validate_capacity_phase(
    phase: CapacityPhase, *, forecast_periods: int, index: int
) -> None:
    valid_modes = {"loss", "add"}
    mode_value = str(phase.mode).lower()
    if mode_value not in valid_modes:
        raise ValueError(f"Phase {index}: mode must be one of {sorted(valid_modes)}.")

    if phase.percent < 0:
        raise ValueError(f"Phase {index}: percent must be >= 0.")

    if not 1 <= phase.start_month <= forecast_periods:
        raise ValueError(
            f"Phase {index}: start_month must be within 1 and {forecast_periods}."
        )

    if not 1 <= phase.end_month <= forecast_periods:
        raise ValueError(
            f"Phase {index}: end_month must be within 1 and {forecast_periods}."
        )

    if phase.end_month < phase.start_month:
        raise ValueError(f"Phase {index}: end_month must be >= start_month.")


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
    raise AttributeError(
        "Unsupported Darts TimeSeries version: cannot convert to DataFrame."
    )


def build_future_forecast_df(
    forecast,
    last_hist_date: pd.Timestamp,
    capacity_phases: Optional[List[CapacityPhase | dict[str, Any]]] = None,
) -> pd.DataFrame:
    """Build a standardized future forecast DataFrame with selective capacity adjustments.

    Parameters
    ----------
    capacity_phases:
        List of phase dictionaries or CapacityPhase dataclass instances.
        Supported keys/fields: enabled, mode (loss|add), percent,
        start_month, end_month.
        Overlap precedence is sequential multiplicative application in list order.

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

    fut["yhat_adjusted"] = fut["yhat_original"]
    fut["adj_applied"] = False

    normalized_phases = _normalize_capacity_phases(
        capacity_phases,
        forecast_periods=int(fut["forecast_month"].max()),
    )
    for phase in normalized_phases:
        if not phase.enabled:
            continue

        in_range = fut["forecast_month"].between(
            phase.start_month,
            phase.end_month,
            inclusive="both",
        )
        if not in_range.any():
            continue

        percentage = float(phase.percent) / 100.0
        factor = 1.0 - percentage if phase.mode.lower() == "loss" else 1.0 + percentage
        fut.loc[in_range, "yhat_adjusted"] = fut.loc[in_range, "yhat_adjusted"] * factor
        fut.loc[in_range, "adj_applied"] = True

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
    capacity_phases: Optional[List[CapacityPhase | dict[str, Any]]] = None,
    exclude_departments: Optional[List[str]] = None,
    changepoint_prior_scale: float = 0.05,
    interval_width: float = 0.90,
) -> ForecastResult:
    if forecast_periods <= 0:
        raise ValueError("forecast_periods must be > 0")

    if capacity_phases is None:
        capacity_phases = [
            CapacityPhase(
                enabled=True,
                mode=adjustment_mode,
                percent=adjustment_percent,
                start_month=adjustment_start_month,
                end_month=adjustment_end_month,
            )
        ]

    normalized_capacity_phases = _normalize_capacity_phases(
        capacity_phases,
        forecast_periods=forecast_periods,
    )

    df_raw = (
        df
        if df is not None
        else load_data(use_synthetic, filename, timeframe_start, timeframe_end)
    )

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
        capacity_phases=normalized_capacity_phases,
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
                f = (future_df["yhat_adjusted"] / future_df["yhat_original"]).fillna(
                    1.0
                )
                mask = future_df["adj_applied"] == True
                future_df.loc[mask, "yhat_lower"] = (
                    future_df.loc[mask, "yhat_lower"] * f.loc[mask]
                )
                future_df.loc[mask, "yhat_upper"] = (
                    future_df.loc[mask, "yhat_upper"] * f.loc[mask]
                )
    except Exception:
        # Intervals are optional; UI will fall back to RMSE-based approximation
        pass

    # Prophet-native forecast dataframe (for Prophet plots)
    prophet_fc_df = None
    try:
        prophet_fc_df = prophet_full_forecast_df(
            model, forecast_periods=forecast_periods, freq="MS"
        )
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
