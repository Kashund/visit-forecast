from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd

from .model_prophet import backtest, fit_and_forecast, prophet_cross_validation_metrics

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
) -> TuningSelection:
    selected_cp, balanced_score, cp_diagnostics_df, cp_note = _evaluate_changepoint_candidates(
        df_prepared=df_prepared,
        forecast_periods=forecast_periods,
        changepoint_candidates=changepoint_candidates,
    )
    selected_iw, iw_diagnostics_df, iw_note = _evaluate_interval_width_candidates(
        df_prepared=df_prepared,
        forecast_periods=forecast_periods,
        changepoint_prior_scale=selected_cp,
        interval_width_candidates=interval_width_candidates,
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
