from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional

import numpy as np
import pandas as pd

from .fiscal import add_fiscal_year, fiscal_summary
from .io import load_data, prepare_dataframe
from .model_prophet import (
    backtest,
    build_conformal_intervals,
    collect_conformal_residuals,
    fit_and_forecast,
    prophet_future_intervals,
    prophet_full_forecast_df,
    prophet_cross_validation_metrics,
)
from .tuning import (
    select_joint_forecast_configuration,
    select_prophet_hyperparameters,
    select_uncertainty_configuration,
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
    ds, yhat, yhat_original, yhat_adjusted, adj_applied_any, applied_phase_ids,
    Fiscal_Year, forecast_month
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
        numeric = [c for c in value_cols if pd.api.types.is_numeric_dtype(fc_df[c])]
        numeric = [c for c in numeric if c.lower() != "index"] or numeric
        value_col = numeric[0] if numeric else value_cols[0]

    fc_df = fc_df.rename(columns={value_col: "yhat"})
    fc_df["yhat"] = pd.to_numeric(fc_df["yhat"], errors="coerce").astype(float)
    fc_df["yhat_original"] = fc_df["yhat"].astype(float)

    # Keep only future rows
    fut = fc_df[fc_df["ds"] > last_hist_date].copy()
    fut = fut.sort_values("ds").reset_index(drop=True)

    # 1-indexed forecast month
    fut["forecast_month"] = fut.index + 1

    if fut.empty:
        fut["yhat_adjusted"] = fut["yhat_original"]
        fut["adj_applied_any"] = False
        fut["applied_phase_ids"] = ""
        fut["adj_applied"] = False
        fut = add_fiscal_year(fut, "ds")
        return fut

    fut["yhat_adjusted"] = fut["yhat_original"].astype(float)
    fut["adj_applied_any"] = False
    fut["applied_phase_ids"] = ""
    fut["adj_applied"] = False

    normalized_phases = _normalize_capacity_phases(
        capacity_phases,
        forecast_periods=int(fut["forecast_month"].max()),
    )
    for phase_index, phase in enumerate(normalized_phases, start=1):
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
        fut.loc[in_range, "adj_applied_any"] = True
        fut.loc[in_range, "applied_phase_ids"] = fut.loc[
            in_range, "applied_phase_ids"
        ].map(
            lambda existing_phase_ids: (
                f"{existing_phase_ids},{phase_index}"
                if existing_phase_ids
                else str(phase_index)
            )
        )
        fut.loc[in_range, "adj_applied"] = True

    fut["adj_applied"] = fut["adj_applied_any"]

    fut = add_fiscal_year(fut, "ds")
    return fut


def _initialize_interval_columns(future_df: pd.DataFrame) -> pd.DataFrame:
    initialized = future_df.copy()
    for column_name in [
        "yhat_lower_prophet",
        "yhat_upper_prophet",
        "yhat_lower_conformal",
        "yhat_upper_conformal",
        "yhat_lower",
        "yhat_upper",
    ]:
        if column_name not in initialized.columns:
            initialized[column_name] = np.nan

    initialized["interval_source"] = initialized.get("interval_source", "none")
    return initialized


def _scale_interval_columns_for_adjustments(future_df: pd.DataFrame) -> pd.DataFrame:
    scaled = future_df.copy()
    required_columns = {
        "yhat_adjusted",
        "yhat_original",
    }
    if not required_columns.issubset(scaled.columns):
        return scaled

    adjustment_flag_column = None
    if "adj_applied_any" in scaled.columns:
        adjustment_flag_column = "adj_applied_any"
    elif "adj_applied" in scaled.columns:
        adjustment_flag_column = "adj_applied"

    if adjustment_flag_column is None:
        return scaled

    factor = (
        pd.to_numeric(scaled["yhat_adjusted"], errors="coerce")
        / pd.to_numeric(scaled["yhat_original"], errors="coerce").replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    mask = scaled[adjustment_flag_column].fillna(False)
    if not mask.any():
        return scaled

    for column_name in [
        "yhat_lower_prophet",
        "yhat_upper_prophet",
        "yhat_lower_conformal",
        "yhat_upper_conformal",
    ]:
        if column_name in scaled.columns:
            scaled.loc[mask, column_name] = (
                pd.to_numeric(scaled.loc[mask, column_name], errors="coerce")
                * factor.loc[mask]
            )

    return scaled


def _activate_interval_source(
    future_df: pd.DataFrame,
    uncertainty_method: Literal["prophet", "conformal"],
) -> tuple[pd.DataFrame, str]:
    activated = future_df.copy()
    preferred_order = [uncertainty_method]
    fallback_source = "prophet" if uncertainty_method == "conformal" else "conformal"
    if fallback_source not in preferred_order:
        preferred_order.append(fallback_source)

    for source_name in preferred_order:
        lower_col = f"yhat_lower_{source_name}"
        upper_col = f"yhat_upper_{source_name}"
        if lower_col not in activated.columns or upper_col not in activated.columns:
            continue
        if activated[[lower_col, upper_col]].notna().any().any():
            activated["yhat_lower"] = activated[lower_col]
            activated["yhat_upper"] = activated[upper_col]
            activated["interval_source"] = source_name
            return activated, source_name

    activated["yhat_lower"] = np.nan
    activated["yhat_upper"] = np.nan
    activated["interval_source"] = "none"
    return activated, "none"


def _coverage_status(value: float, target_coverage: float) -> str:
    if pd.isna(value):
        return "Unavailable"

    gap = abs(float(value) - float(target_coverage))
    if gap <= 0.03:
        return "On target"
    if gap <= 0.08:
        return "Watch"
    return "Off target"


def _build_prophet_interval_validation(
    cv_metrics_df: pd.DataFrame | None,
    target_coverage: float,
) -> tuple[pd.DataFrame | None, Dict[str, Any]]:
    if cv_metrics_df is None or cv_metrics_df.empty or "coverage" not in cv_metrics_df.columns:
        return None, {
            "empirical_coverage_overall": float("nan"),
            "empirical_coverage_by_horizon": {},
            "avg_interval_width": float("nan"),
            "median_interval_width": float("nan"),
            "interval_width_over_mean_actual": float("nan"),
            "n_calibration_forecasts": 0,
            "fallback_used": False,
        }

    diagnostics_df = cv_metrics_df.copy()
    diagnostics_df["target_coverage"] = float(target_coverage)
    diagnostics_df["empirical_coverage"] = pd.to_numeric(
        diagnostics_df["coverage"], errors="coerce"
    )
    diagnostics_df["Coverage_Check"] = diagnostics_df["empirical_coverage"].map(
        lambda value: _coverage_status(value, target_coverage)
    )
    if "horizon" in diagnostics_df.columns:
        try:
            diagnostics_df["Horizon_Days"] = pd.to_timedelta(
                diagnostics_df["horizon"]
            ).dt.days
        except Exception:
            diagnostics_df["Horizon_Days"] = np.nan

    coverage_by_horizon = {}
    horizon_key = "Horizon_Days" if "Horizon_Days" in diagnostics_df.columns else None
    if horizon_key:
        coverage_by_horizon = {
            int(horizon_days): float(coverage)
            for horizon_days, coverage in diagnostics_df.dropna(
                subset=[horizon_key, "empirical_coverage"]
            )[[horizon_key, "empirical_coverage"]].itertuples(index=False)
        }

    summary = {
        "empirical_coverage_overall": float(
            diagnostics_df["empirical_coverage"].dropna().mean()
        )
        if diagnostics_df["empirical_coverage"].notna().any()
        else float("nan"),
        "empirical_coverage_by_horizon": coverage_by_horizon,
        "avg_interval_width": float("nan"),
        "median_interval_width": float("nan"),
        "interval_width_over_mean_actual": float("nan"),
        "n_calibration_forecasts": int(diagnostics_df.shape[0]),
        "fallback_used": False,
    }
    return diagnostics_df, summary


def _build_uncertainty_method_selection(
    target_coverage: float,
    prophet_summary: Dict[str, Any] | None,
    conformal_summary: Dict[str, Any] | None,
) -> tuple[str, pd.DataFrame, str | None]:
    candidate_rows: list[dict[str, Any]] = []
    candidate_order = {"prophet": 0, "conformal": 1}

    for method_name, summary in [
        ("prophet", prophet_summary),
        ("conformal", conformal_summary),
    ]:
        summary = summary or {}
        empirical_coverage = summary.get("empirical_coverage_overall", float("nan"))
        primary_score = (
            abs(float(empirical_coverage) - float(target_coverage))
            if pd.notna(empirical_coverage)
            else float("nan")
        )
        tie_breaker = summary.get("interval_width_over_mean_actual", float("nan"))
        if pd.isna(tie_breaker):
            tie_breaker = summary.get("avg_interval_width", float("nan"))

        fallback_used = bool(summary.get("fallback_used", False))
        candidate_rows.append(
            {
                "parameter_family": "uncertainty_method",
                "candidate_value": method_name,
                "primary_score_name": "coverage_gap",
                "primary_score_value": primary_score,
                "tie_breaker_name": "interval_width_over_mean_actual",
                "tie_breaker_value": tie_breaker,
                "empirical_coverage": empirical_coverage,
                "target_coverage": float(target_coverage),
                "fallback_used": fallback_used,
                "selected": False,
                "note": (
                    "Coverage unavailable"
                    if pd.isna(empirical_coverage)
                    else (
                        "Fallback calibration used"
                        if fallback_used
                        else ""
                    )
                ),
                "candidate_order": candidate_order[method_name],
            }
        )

    diagnostics_df = pd.DataFrame(candidate_rows)
    valid_rows = diagnostics_df.dropna(subset=["primary_score_value"]).copy()
    if valid_rows.empty:
        selected_method = "prophet"
        diagnostics_df.loc[
            diagnostics_df["candidate_value"] == selected_method, "selected"
        ] = True
        note = (
            "Auto-selected Prophet interval because interval coverage diagnostics "
            "were unavailable for both methods."
        )
        return selected_method, diagnostics_df.drop(columns=["candidate_order"]), note

    valid_rows["fallback_penalty"] = valid_rows["fallback_used"].astype(int)
    valid_rows["tie_breaker_rank"] = pd.to_numeric(
        valid_rows["tie_breaker_value"], errors="coerce"
    ).fillna(float("inf"))
    selected_row = valid_rows.sort_values(
        [
            "primary_score_value",
            "fallback_penalty",
            "tie_breaker_rank",
            "candidate_order",
        ]
    ).iloc[0]
    selected_method = str(selected_row["candidate_value"])
    diagnostics_df.loc[
        diagnostics_df["candidate_value"] == selected_method, "selected"
    ] = True
    selected_gap = float(selected_row["primary_score_value"])
    note = (
        f"Auto-selected {selected_method} interval using coverage gap "
        f"{selected_gap:.4f} against target coverage {float(target_coverage):.2f}."
    )
    return selected_method, diagnostics_df.drop(columns=["candidate_order"]), note


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
    target_coverage: Optional[float] = None,
    uncertainty_method: Literal["prophet", "conformal", "auto"] = "prophet",
    tuning_mode: str = "manual",
    joint_auto_tuning_enabled: bool = True,
) -> ForecastResult:
    if forecast_periods <= 0:
        raise ValueError("forecast_periods must be > 0")
    if uncertainty_method not in {"prophet", "conformal", "auto"}:
        raise ValueError(
            "uncertainty_method must be either 'prophet', 'conformal', or 'auto'"
        )

    requested_target_coverage = float(
        target_coverage if target_coverage is not None else interval_width
    )

    if capacity_phases is None:
        capacity_phases = [
            CapacityPhase(
                enabled=True,
                mode=adjustment_mode,
                percent=adjustment_percent,
                start_month=adjustment_start_month,
                end_month=min(adjustment_end_month, forecast_periods),
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

    selected_changepoint_prior_scale = float(changepoint_prior_scale)
    selected_interval_width = float(requested_target_coverage)
    effective_target_coverage = float(requested_target_coverage)
    tuning_primary_metric = None
    tuning_primary_score = None
    tuning_note = None
    tuning_diagnostics_df = None
    selected_uncertainty_method_from_tuning = None
    used_joint_auto_tuning = bool(
        tuning_mode == "auto"
        and uncertainty_method == "auto"
        and joint_auto_tuning_enabled
    )
    used_separate_auto_passes = bool(
        tuning_mode == "auto"
        and uncertainty_method == "auto"
        and not joint_auto_tuning_enabled
    )

    if tuning_mode == "auto":
        if used_joint_auto_tuning:
            tuning_selection = select_joint_forecast_configuration(
                df_prepared=df_prepared,
                forecast_periods=forecast_periods,
                target_coverage=requested_target_coverage,
            )
            selected_changepoint_prior_scale = (
                tuning_selection.changepoint_prior_scale
            )
            selected_interval_width = tuning_selection.interval_width
            tuning_primary_metric = tuning_selection.primary_metric
            tuning_primary_score = tuning_selection.primary_score
            tuning_note = tuning_selection.note
            tuning_diagnostics_df = tuning_selection.diagnostics_df
            selected_uncertainty_method_from_tuning = tuning_selection.uncertainty_method
        elif used_separate_auto_passes:
            changepoint_tuning_selection = select_prophet_hyperparameters(
                df_prepared=df_prepared,
                forecast_periods=forecast_periods,
                tune_interval_width=False,
                fixed_interval_width=requested_target_coverage,
            )
            selected_changepoint_prior_scale = (
                changepoint_tuning_selection.changepoint_prior_scale
            )
            uncertainty_tuning_selection = select_uncertainty_configuration(
                df_prepared=df_prepared,
                forecast_periods=forecast_periods,
                changepoint_prior_scale=selected_changepoint_prior_scale,
            )
            selected_interval_width = uncertainty_tuning_selection.interval_width
            effective_target_coverage = uncertainty_tuning_selection.interval_width
            selected_uncertainty_method_from_tuning = (
                uncertainty_tuning_selection.uncertainty_method
            )
            tuning_primary_metric = uncertainty_tuning_selection.primary_metric
            tuning_primary_score = uncertainty_tuning_selection.primary_score
            tuning_note = " ".join(
                note
                for note in [
                    changepoint_tuning_selection.note,
                    uncertainty_tuning_selection.note,
                    (
                        "Joint auto tuning evaluation was off, so changepoint tuning "
                        "and uncertainty tuning were evaluated in separate passes."
                    ),
                ]
                if note
            )
            tuning_diagnostics_df = pd.concat(
                [
                    changepoint_tuning_selection.diagnostics_df,
                    uncertainty_tuning_selection.diagnostics_df,
                ],
                ignore_index=True,
            )
        else:
            tuning_selection = select_prophet_hyperparameters(
                df_prepared=df_prepared,
                forecast_periods=forecast_periods,
                tune_interval_width=False,
                fixed_interval_width=requested_target_coverage,
            )
            selected_changepoint_prior_scale = (
                tuning_selection.changepoint_prior_scale
            )
            selected_interval_width = tuning_selection.interval_width
            tuning_primary_metric = tuning_selection.primary_metric
            tuning_primary_score = tuning_selection.primary_score
            tuning_note = tuning_selection.note
            tuning_diagnostics_df = tuning_selection.diagnostics_df
    elif tuning_mode != "manual":
        raise ValueError("tuning_mode must be either 'manual' or 'auto'")

    series, forecast, model = fit_and_forecast(
        df_prepared=df_prepared,
        forecast_periods=forecast_periods,
        changepoint_prior_scale=selected_changepoint_prior_scale,
        interval_width=selected_interval_width,
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
    future_df = _initialize_interval_columns(future_df)

    # Store Prophet-native uncertainty intervals as a source-specific set.
    try:
        intervals = prophet_future_intervals(
            model,
            last_hist_date=last_hist_date,
            forecast_periods=forecast_periods,
        )
        if intervals is not None and not intervals.empty:
            prophet_interval_df = intervals.rename(
                columns={
                    "yhat_lower": "yhat_lower_prophet",
                    "yhat_upper": "yhat_upper_prophet",
                }
            )
            future_df = future_df.merge(
                prophet_interval_df,
                on="ds",
                how="left",
                suffixes=("", "_incoming"),
            )
            for column_name in ["yhat_lower_prophet", "yhat_upper_prophet"]:
                incoming_column = f"{column_name}_incoming"
                if incoming_column in future_df.columns:
                    future_df[column_name] = future_df[incoming_column]
                    future_df = future_df.drop(columns=[incoming_column])
    except Exception:
        pass

    interval_diagnostics_df = None
    interval_summary_metrics: Dict[str, Any] | None = None
    prophet_interval_diagnostics_df = None
    prophet_interval_summary_metrics: Dict[str, Any] | None = None
    conformal_interval_diagnostics_df = None
    conformal_interval_summary_metrics: Dict[str, Any] | None = None

    if uncertainty_method in {"conformal", "auto"}:
        try:
            conformal_residuals_df = collect_conformal_residuals(
                model=model,
                series=series,
                forecast_horizon=forecast_periods,
                start=0.7,
                stride=1,
            )
            future_df, interval_diagnostics_df, interval_summary_metrics = (
                build_conformal_intervals(
                    future_df=future_df,
                    residuals_df=conformal_residuals_df,
                    target_coverage=effective_target_coverage,
                )
            )
            conformal_interval_diagnostics_df = interval_diagnostics_df
            conformal_interval_summary_metrics = interval_summary_metrics
        except Exception:
            future_df["yhat_lower_conformal"] = np.nan
            future_df["yhat_upper_conformal"] = np.nan
            conformal_interval_diagnostics_df = None
            conformal_interval_summary_metrics = {
                "empirical_coverage_overall": float("nan"),
                "empirical_coverage_by_horizon": {},
                "avg_interval_width": float("nan"),
                "median_interval_width": float("nan"),
                "interval_width_over_mean_actual": float("nan"),
                "n_calibration_forecasts": 0,
                "fallback_used": True,
            }
            if uncertainty_method == "conformal":
                interval_diagnostics_df = conformal_interval_diagnostics_df
                interval_summary_metrics = conformal_interval_summary_metrics

    future_df = _scale_interval_columns_for_adjustments(future_df)

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

    (
        prophet_interval_diagnostics_df,
        prophet_interval_summary_metrics,
    ) = _build_prophet_interval_validation(
        cv_metrics_df=cv_metrics,
        target_coverage=effective_target_coverage,
    )

    selected_uncertainty_method = uncertainty_method
    uncertainty_method_note = None
    uncertainty_method_diagnostics_df = None

    if (
        uncertainty_method == "auto"
        and tuning_mode == "auto"
        and selected_uncertainty_method_from_tuning
    ):
        selected_uncertainty_method = selected_uncertainty_method_from_tuning
    elif uncertainty_method == "auto":
        (
            selected_uncertainty_method,
            uncertainty_method_diagnostics_df,
            uncertainty_method_note,
        ) = _build_uncertainty_method_selection(
            target_coverage=effective_target_coverage,
            prophet_summary=prophet_interval_summary_metrics,
            conformal_summary=conformal_interval_summary_metrics,
        )
    elif uncertainty_method == "prophet":
        selected_uncertainty_method = "prophet"
    elif uncertainty_method == "conformal":
        selected_uncertainty_method = "conformal"

    future_df, interval_source_used = _activate_interval_source(
        future_df,
        uncertainty_method=selected_uncertainty_method,
    )

    if selected_uncertainty_method == "prophet":
        interval_diagnostics_df = prophet_interval_diagnostics_df
        interval_summary_metrics = prophet_interval_summary_metrics
    else:
        interval_diagnostics_df = conformal_interval_diagnostics_df
        interval_summary_metrics = conformal_interval_summary_metrics

    if uncertainty_method_diagnostics_df is not None and not uncertainty_method_diagnostics_df.empty:
        if tuning_diagnostics_df is None or tuning_diagnostics_df.empty:
            tuning_diagnostics_df = uncertainty_method_diagnostics_df.copy()
        else:
            tuning_diagnostics_df = pd.concat(
                [tuning_diagnostics_df, uncertainty_method_diagnostics_df],
                ignore_index=True,
            )
    if uncertainty_method_note:
        tuning_note = (
            f"{tuning_note} {uncertainty_method_note}".strip()
            if tuning_note
            else uncertainty_method_note
        )

    if interval_summary_metrics is None:
        interval_summary_metrics = {
            "empirical_coverage_overall": float("nan"),
            "empirical_coverage_by_horizon": {},
            "avg_interval_width": float("nan"),
            "median_interval_width": float("nan"),
            "interval_width_over_mean_actual": float("nan"),
            "n_calibration_forecasts": 0,
            "fallback_used": True,
        }
    interval_summary_metrics = {
        **interval_summary_metrics,
        "interval_source_used": interval_source_used,
        "requested_uncertainty_method": uncertainty_method,
        "target_coverage": float(effective_target_coverage),
        "uncertainty_method": selected_uncertainty_method,
    }

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
        tuning_mode=tuning_mode,
        selected_changepoint_prior_scale=selected_changepoint_prior_scale,
        selected_interval_width=selected_interval_width,
        requested_uncertainty_method=uncertainty_method,
        uncertainty_method=selected_uncertainty_method,
        joint_auto_tuning_enabled=used_joint_auto_tuning,
        target_coverage=float(effective_target_coverage),
        interval_diagnostics_df=interval_diagnostics_df,
        interval_summary_metrics=interval_summary_metrics,
        tuning_primary_metric=tuning_primary_metric,
        tuning_primary_score=tuning_primary_score,
        tuning_note=tuning_note,
        tuning_diagnostics_df=tuning_diagnostics_df,
    )
