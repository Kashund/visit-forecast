from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from .model_prophet import (
    backtest,
    build_conformal_intervals,
    collect_conformal_residuals,
    fit_and_forecast,
    prophet_cross_validation_metrics,
)

DEFAULT_CHANGEPOINT_CANDIDATES = [0.01, 0.03, 0.05, 0.10, 0.20]
DEFAULT_INTERVAL_WIDTH_CANDIDATES = [0.80, 0.85, 0.90, 0.95]
DEFAULT_CHANGEPOINT_FALLBACK = 0.05
DEFAULT_INTERVAL_WIDTH_FALLBACK = 0.90


@dataclass(frozen=True)
class TuningSelection:
    changepoint_prior_scale: float
    interval_width: float
    primary_metric: str
    primary_score: float | None
    note: str | None
    diagnostics_df: pd.DataFrame
    uncertainty_method: str | None = None


def _cv_parameter_sets(forecast_periods: int) -> list[tuple[str, dict[str, str]]]:
    horizon_days = max(30, int(forecast_periods) * 30)
    desired = {
        "initial": f"{max(730, horizon_days * 2)} days",
        "period": f"{max(30, horizon_days // 2)} days",
        "horizon": f"{horizon_days} days",
    }
    fallback = {
        "initial": "730 days",
        "period": "180 days",
        "horizon": "365 days",
    }
    if desired == fallback:
        return [("current_run_horizon", desired)]
    return [
        ("current_run_horizon", desired),
        ("fallback_default", fallback),
    ]


def _longest_horizon_gap(cv_metrics_df: pd.DataFrame, candidate_interval_width: float) -> float:
    coverage_gap = (pd.to_numeric(cv_metrics_df["coverage"], errors="coerce") - candidate_interval_width).abs()
    if "horizon" not in cv_metrics_df.columns:
        return float(coverage_gap.iloc[-1])

    try:
        horizon_values = pd.to_timedelta(cv_metrics_df["horizon"])
        longest_horizon = horizon_values.max()
        return float(coverage_gap[horizon_values == longest_horizon].mean())
    except Exception:
        return float(coverage_gap.iloc[-1])


def _balanced_error_score(metric_series: pd.Series) -> pd.Series:
    metric_floor = max(float(metric_series.min(skipna=True)), 1e-9)
    return metric_series / metric_floor


def _summarize_prophet_interval_validation(
    cv_metrics_df: pd.DataFrame | None,
    target_coverage: float,
) -> dict[str, float]:
    if cv_metrics_df is None or cv_metrics_df.empty or "coverage" not in cv_metrics_df.columns:
        return {
            "empirical_coverage_overall": float("nan"),
            "coverage_gap": float("nan"),
            "interval_width_over_mean_actual": float("nan"),
            "avg_interval_width": float("nan"),
            "fallback_used": False,
        }

    empirical_coverage = float(
        pd.to_numeric(cv_metrics_df["coverage"], errors="coerce").dropna().mean()
    )
    coverage_gap = (
        abs(empirical_coverage - float(target_coverage))
        if pd.notna(empirical_coverage)
        else float("nan")
    )
    return {
        "empirical_coverage_overall": empirical_coverage,
        "coverage_gap": coverage_gap,
        "interval_width_over_mean_actual": float("nan"),
        "avg_interval_width": float("nan"),
        "fallback_used": False,
    }


def _timeseries_to_forecast_frame(forecast) -> pd.DataFrame:
    if hasattr(forecast, "pd_dataframe"):
        fc_df = forecast.pd_dataframe().reset_index()
    elif hasattr(forecast, "to_dataframe"):
        fc_df = forecast.to_dataframe().reset_index()
    elif hasattr(forecast, "pandas_dataframe"):
        fc_df = forecast.pandas_dataframe().reset_index()
    else:
        raise AttributeError("Unsupported Darts TimeSeries version: cannot convert forecast.")

    time_col = next(
        (c for c in ["ds", "time", "index", "Date", "date"] if c in fc_df.columns),
        fc_df.columns[0],
    )
    fc_df = fc_df.rename(columns={time_col: "ds"})
    fc_df["ds"] = pd.to_datetime(fc_df["ds"], errors="coerce")

    value_cols = [c for c in fc_df.columns if c != "ds"]
    preferred_value_cols = [
        column_name
        for column_name in ["y", "yhat", "value", "Visits"]
        if column_name in value_cols
    ]
    if preferred_value_cols:
        value_col = preferred_value_cols[0]
    else:
        numeric = [c for c in value_cols if pd.api.types.is_numeric_dtype(fc_df[c])]
        numeric = [c for c in numeric if c.lower() != "index"] or numeric
        value_col = numeric[0] if numeric else value_cols[0]

    future_df = fc_df.rename(columns={value_col: "yhat_original"})[["ds", "yhat_original"]]
    future_df["yhat_original"] = pd.to_numeric(
        future_df["yhat_original"], errors="coerce"
    ).astype(float)
    future_df["forecast_month"] = future_df.index + 1
    future_df["yhat_adjusted"] = future_df["yhat_original"]
    return future_df


def _summarize_conformal_interval_validation(
    model,
    series,
    forecast,
    forecast_periods: int,
    target_coverage: float,
) -> dict[str, float]:
    residuals_df = collect_conformal_residuals(
        model=model,
        series=series,
        forecast_horizon=forecast_periods,
        start=0.7,
        stride=1,
    )
    future_df = _timeseries_to_forecast_frame(forecast)
    _, _, summary = build_conformal_intervals(
        future_df=future_df,
        residuals_df=residuals_df,
        target_coverage=target_coverage,
    )
    empirical_coverage = summary.get("empirical_coverage_overall", float("nan"))
    coverage_gap = (
        abs(float(empirical_coverage) - float(target_coverage))
        if pd.notna(empirical_coverage)
        else float("nan")
    )
    return {
        **summary,
        "coverage_gap": coverage_gap,
    }


def _evaluate_changepoint_candidates(
    df_prepared: pd.DataFrame,
    forecast_periods: int,
    changepoint_candidates: Iterable[float],
) -> tuple[float, float | None, pd.DataFrame, str | None]:
    rows = []
    for candidate in changepoint_candidates:
        row = {
            "parameter_family": "changepoint_prior_scale",
            "candidate_value": float(candidate),
            "primary_score_name": "MAPE+RMSE balance",
            "primary_score_value": float("nan"),
            "tie_breaker_name": "RMSE",
            "tie_breaker_value": float("nan"),
            "rmse": float("nan"),
            "mape": float("nan"),
            "mae": float("nan"),
            "selected": False,
            "note": "",
        }
        try:
            series, _, model = fit_and_forecast(
                df_prepared=df_prepared,
                forecast_periods=forecast_periods,
                changepoint_prior_scale=float(candidate),
                interval_width=DEFAULT_INTERVAL_WIDTH_FALLBACK,
            )
            metrics = backtest(
                model=model,
                series=series,
                forecast_horizon=forecast_periods,
                start=0.7,
                stride=forecast_periods,
            )
            row["rmse"] = float(metrics.get("RMSE", float("nan")))
            row["mape"] = float(metrics.get("MAPE", float("nan")))
            row["mae"] = float(metrics.get("MAE", float("nan")))
        except Exception as exc:
            row["note"] = f"{type(exc).__name__}"
        rows.append(row)

    diagnostics_df = pd.DataFrame(rows)
    valid_rows = diagnostics_df.dropna(subset=["rmse", "mape"]).copy()
    if valid_rows.empty:
        diagnostics_df.loc[
            diagnostics_df["candidate_value"] == DEFAULT_CHANGEPOINT_FALLBACK,
            "selected",
        ] = True
        note = (
            "Auto-tune could not score changepoint candidates; "
            f"used fallback changepoint_prior_scale={DEFAULT_CHANGEPOINT_FALLBACK:.2f}."
        )
        return DEFAULT_CHANGEPOINT_FALLBACK, None, diagnostics_df, note

    valid_rows["primary_score_value"] = (
        _balanced_error_score(valid_rows["rmse"])
        + _balanced_error_score(valid_rows["mape"])
    ) / 2.0
    valid_rows["tie_breaker_value"] = valid_rows["rmse"]
    diagnostics_df.loc[valid_rows.index, "primary_score_value"] = valid_rows[
        "primary_score_value"
    ]
    diagnostics_df.loc[valid_rows.index, "tie_breaker_value"] = valid_rows[
        "tie_breaker_value"
    ]

    selected_row = valid_rows.sort_values(
        ["primary_score_value", "tie_breaker_value", "mape", "mae", "candidate_value"]
    ).iloc[0]
    selected_value = float(selected_row["candidate_value"])
    diagnostics_df.loc[
        diagnostics_df["candidate_value"] == selected_value,
        "selected",
    ] = True
    return selected_value, float(selected_row["primary_score_value"]), diagnostics_df, None


def _evaluate_interval_width_candidates(
    df_prepared: pd.DataFrame,
    forecast_periods: int,
    changepoint_prior_scale: float,
    interval_width_candidates: Iterable[float],
) -> tuple[float, pd.DataFrame, str | None]:
    attempted_diagnostics = pd.DataFrame()
    fallback_note = None

    for cv_config_name, cv_params in _cv_parameter_sets(forecast_periods):
        rows = []
        for candidate in interval_width_candidates:
            row = {
                "parameter_family": "interval_width",
                "candidate_value": float(candidate),
                "primary_score_name": "coverage_gap_mean",
                "primary_score_value": float("nan"),
                "tie_breaker_name": "coverage_gap_longest_horizon",
                "tie_breaker_value": float("nan"),
                "selected": False,
                "note": cv_config_name,
            }
            try:
                _, _, model = fit_and_forecast(
                    df_prepared=df_prepared,
                    forecast_periods=forecast_periods,
                    changepoint_prior_scale=changepoint_prior_scale,
                    interval_width=float(candidate),
                )
                _, cv_metrics = prophet_cross_validation_metrics(model, **cv_params)
                if (
                    cv_metrics is not None
                    and not cv_metrics.empty
                    and "coverage" in cv_metrics.columns
                ):
                    coverage_gap = (
                        pd.to_numeric(cv_metrics["coverage"], errors="coerce") - float(candidate)
                    ).abs()
                    row["primary_score_value"] = float(coverage_gap.mean())
                    row["tie_breaker_value"] = _longest_horizon_gap(
                        cv_metrics, float(candidate)
                    )
                else:
                    row["note"] = f"{cv_config_name}:cv_unavailable"
            except Exception as exc:
                row["note"] = f"{cv_config_name}:{type(exc).__name__}"
            rows.append(row)

        diagnostics_df = pd.DataFrame(rows)
        valid_rows = diagnostics_df.dropna(
            subset=["primary_score_value", "tie_breaker_value"]
        )
        if not valid_rows.empty:
            selected_row = valid_rows.sort_values(
                ["primary_score_value", "tie_breaker_value", "candidate_value"]
            ).iloc[0]
            selected_value = float(selected_row["candidate_value"])
            diagnostics_df.loc[
                diagnostics_df["candidate_value"] == selected_value,
                "selected",
            ] = True
            if cv_config_name == "fallback_default":
                fallback_note = (
                    "Used fallback Prophet CV window (730/180/365 days) because "
                    "current-run horizon CV was unavailable."
                )
            return selected_value, diagnostics_df, fallback_note
        attempted_diagnostics = diagnostics_df

    attempted_diagnostics.loc[
        attempted_diagnostics["candidate_value"] == DEFAULT_INTERVAL_WIDTH_FALLBACK,
        "selected",
    ] = True
    fallback_note = (
        "Auto-tune could not score interval width candidates; "
        f"used fallback interval_width={DEFAULT_INTERVAL_WIDTH_FALLBACK:.2f}."
    )
    return DEFAULT_INTERVAL_WIDTH_FALLBACK, attempted_diagnostics, fallback_note


def select_prophet_hyperparameters(
    df_prepared: pd.DataFrame,
    forecast_periods: int,
    changepoint_candidates: Iterable[float] = DEFAULT_CHANGEPOINT_CANDIDATES,
    interval_width_candidates: Iterable[float] = DEFAULT_INTERVAL_WIDTH_CANDIDATES,
    tune_interval_width: bool = True,
    fixed_interval_width: float = DEFAULT_INTERVAL_WIDTH_FALLBACK,
) -> TuningSelection:
    selected_cp, balanced_score, cp_diagnostics_df, cp_note = _evaluate_changepoint_candidates(
        df_prepared=df_prepared,
        forecast_periods=forecast_periods,
        changepoint_candidates=changepoint_candidates,
    )

    if tune_interval_width:
        selected_iw, iw_diagnostics_df, iw_note = _evaluate_interval_width_candidates(
            df_prepared=df_prepared,
            forecast_periods=forecast_periods,
            changepoint_prior_scale=selected_cp,
            interval_width_candidates=interval_width_candidates,
        )
    else:
        selected_iw = float(fixed_interval_width)
        iw_diagnostics_df = pd.DataFrame(
            [
                {
                    "parameter_family": "interval_width",
                    "candidate_value": selected_iw,
                    "primary_score_name": "skipped_for_conformal",
                    "primary_score_value": float("nan"),
                    "tie_breaker_name": "coverage_fixed",
                    "tie_breaker_value": float("nan"),
                    "selected": True,
                    "note": "Interval width tuning skipped; using the fixed interval / target coverage input for this run.",
                }
            ]
        )
        iw_note = (
            "Interval width tuning skipped; using fixed interval / target coverage "
            f"{selected_iw:.2f} from the current uncertainty settings."
        )

    notes = [note for note in [cp_note, iw_note] if note]
    diagnostics_df = pd.concat(
        [cp_diagnostics_df, iw_diagnostics_df], ignore_index=True
    )
    return TuningSelection(
        changepoint_prior_scale=selected_cp,
        interval_width=selected_iw,
        primary_metric="MAPE+RMSE",
        primary_score=balanced_score,
        note=" ".join(notes) if notes else None,
        diagnostics_df=diagnostics_df,
    )


def select_joint_forecast_configuration(
    df_prepared: pd.DataFrame,
    forecast_periods: int,
    target_coverage: float,
    changepoint_candidates: Iterable[float] = DEFAULT_CHANGEPOINT_CANDIDATES,
    interval_width_candidates: Iterable[float] = DEFAULT_INTERVAL_WIDTH_CANDIDATES,
) -> TuningSelection:
    rows: list[dict[str, object]] = []

    for changepoint_candidate in changepoint_candidates:
        for method_name in ["prophet", "conformal"]:
            candidate_interval_widths = (
                interval_width_candidates if method_name == "prophet" else [target_coverage]
            )
            for interval_width in candidate_interval_widths:
                row = {
                    "parameter_family": "joint_model_config",
                    "candidate_value": f"{method_name}|cp={float(changepoint_candidate):.2f}|iw={float(interval_width):.2f}",
                    "uncertainty_method": method_name,
                    "changepoint_prior_scale": float(changepoint_candidate),
                    "interval_width": float(interval_width),
                    "primary_score_name": "joint_model_score",
                    "primary_score_value": float("nan"),
                    "tie_breaker_name": "coverage_gap_then_rmse",
                    "tie_breaker_value": float("nan"),
                    "mape": float("nan"),
                    "mae": float("nan"),
                    "rmse": float("nan"),
                    "empirical_coverage": float("nan"),
                    "coverage_gap": float("nan"),
                    "interval_width_over_mean_actual": float("nan"),
                    "fallback_used": False,
                    "selected": False,
                    "note": "",
                }
                try:
                    series, forecast, model = fit_and_forecast(
                        df_prepared=df_prepared,
                        forecast_periods=forecast_periods,
                        changepoint_prior_scale=float(changepoint_candidate),
                        interval_width=float(interval_width),
                    )
                    point_metrics = backtest(
                        model=model,
                        series=series,
                        forecast_horizon=forecast_periods,
                        start=0.7,
                        stride=forecast_periods,
                    )
                    row["mape"] = float(point_metrics.get("MAPE", float("nan")))
                    row["mae"] = float(point_metrics.get("MAE", float("nan")))
                    row["rmse"] = float(point_metrics.get("RMSE", float("nan")))

                    if method_name == "prophet":
                        _, cv_metrics = prophet_cross_validation_metrics(
                            model,
                            initial="730 days",
                            period="180 days",
                            horizon="365 days",
                        )
                        interval_summary = _summarize_prophet_interval_validation(
                            cv_metrics_df=cv_metrics,
                            target_coverage=float(target_coverage),
                        )
                    else:
                        interval_summary = _summarize_conformal_interval_validation(
                            model=model,
                            series=series,
                            forecast=forecast,
                            forecast_periods=forecast_periods,
                            target_coverage=float(target_coverage),
                        )

                    row["empirical_coverage"] = float(
                        interval_summary.get("empirical_coverage_overall", float("nan"))
                    )
                    row["coverage_gap"] = float(
                        interval_summary.get("coverage_gap", float("nan"))
                    )
                    row["interval_width_over_mean_actual"] = float(
                        interval_summary.get("interval_width_over_mean_actual", float("nan"))
                    )
                    row["fallback_used"] = bool(
                        interval_summary.get("fallback_used", False)
                    )
                    if pd.isna(row["empirical_coverage"]):
                        row["note"] = "Coverage unavailable"
                    elif bool(row["fallback_used"]):
                        row["note"] = "Fallback interval calibration used"
                except Exception as exc:
                    row["note"] = type(exc).__name__

                rows.append(row)

    diagnostics_df = pd.DataFrame(rows)
    valid_rows = diagnostics_df.dropna(subset=["mape", "mae", "rmse", "coverage_gap"]).copy()
    if valid_rows.empty:
        diagnostics_df.loc[
            diagnostics_df["candidate_value"].eq("prophet|cp=0.05|iw=0.90"),
            "selected",
        ] = True
        note = (
            "Joint auto-tune could not score model configurations; "
            "used fallback changepoint_prior_scale=0.05, interval_width=0.90, uncertainty_method=prophet."
        )
        return TuningSelection(
            changepoint_prior_scale=DEFAULT_CHANGEPOINT_FALLBACK,
            interval_width=DEFAULT_INTERVAL_WIDTH_FALLBACK,
            primary_metric="Joint model score",
            primary_score=None,
            note=note,
            diagnostics_df=diagnostics_df,
            uncertainty_method="prophet",
        )

    valid_rows["mape_norm"] = _balanced_error_score(pd.to_numeric(valid_rows["mape"], errors="coerce"))
    valid_rows["mae_norm"] = _balanced_error_score(pd.to_numeric(valid_rows["mae"], errors="coerce"))
    valid_rows["rmse_norm"] = _balanced_error_score(pd.to_numeric(valid_rows["rmse"], errors="coerce"))
    valid_rows["coverage_gap_norm"] = _balanced_error_score(
        pd.to_numeric(valid_rows["coverage_gap"], errors="coerce").fillna(float("nan"))
    )
    valid_rows["primary_score_value"] = (
        valid_rows["mape_norm"]
        + valid_rows["mae_norm"]
        + valid_rows["rmse_norm"]
        + valid_rows["coverage_gap_norm"]
    ) / 4.0
    valid_rows["tie_breaker_value"] = (
        pd.to_numeric(valid_rows["coverage_gap"], errors="coerce") * 1000.0
        + pd.to_numeric(valid_rows["rmse"], errors="coerce")
    )
    diagnostics_df.loc[valid_rows.index, "primary_score_value"] = valid_rows["primary_score_value"]
    diagnostics_df.loc[valid_rows.index, "tie_breaker_value"] = valid_rows["tie_breaker_value"]

    valid_rows["fallback_penalty"] = valid_rows["fallback_used"].astype(int)
    valid_rows["width_penalty"] = pd.to_numeric(
        valid_rows["interval_width_over_mean_actual"], errors="coerce"
    ).fillna(float("inf"))
    selected_row = valid_rows.sort_values(
        [
            "primary_score_value",
            "fallback_penalty",
            "coverage_gap",
            "width_penalty",
            "rmse",
            "uncertainty_method",
            "changepoint_prior_scale",
            "interval_width",
        ]
    ).iloc[0]
    selected_candidate = str(selected_row["candidate_value"])
    diagnostics_df.loc[
        diagnostics_df["candidate_value"] == selected_candidate, "selected"
    ] = True

    note = (
        f"Joint auto-tune selected {selected_row['uncertainty_method']} with "
        f"changepoint_prior_scale={float(selected_row['changepoint_prior_scale']):.2f} "
        f"and interval/coverage={float(selected_row['interval_width']):.2f} by balancing "
        "MAPE, MAE, RMSE, and coverage gap."
    )
    return TuningSelection(
        changepoint_prior_scale=float(selected_row["changepoint_prior_scale"]),
        interval_width=float(selected_row["interval_width"]),
        primary_metric="Joint model score",
        primary_score=float(selected_row["primary_score_value"]),
        note=note,
        diagnostics_df=diagnostics_df,
        uncertainty_method=str(selected_row["uncertainty_method"]),
    )
