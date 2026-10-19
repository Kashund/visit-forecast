from __future__ import annotations

"""Model registry and automatic model selection.

This module generalizes the app beyond Prophet. All candidates are Darts
models, so the existing backtesting and conformal interval machinery
(which only relies on `fit`, `predict`, and `historical_forecasts`)
works unchanged for every candidate.

Candidates
----------
- prophet: Darts Prophet wrapper (original app model)
- ets: Holt-Winters Exponential Smoothing with yearly seasonality
- theta: Theta method (top performer on monthly data in the M3 competition)
- arima: SARIMA (1,1,1)x(1,1,1,12) when history allows, ARIMA(1,1,1) otherwise
- seasonal_naive: repeat-last-season baseline, used as a skill reference

Auto selection backtests every eligible candidate with the same rolling
historical-forecast protocol the app already uses, then picks the best
balanced MAPE + RMSE score. The seasonal naive baseline is always scored
so the diagnostics show whether the winning model actually beats a naive
repeat of last year.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.models import ARIMA, ExponentialSmoothing, NaiveSeasonal, Theta
from darts.models import Prophet
from darts.utils.utils import SeasonalityMode

from .model_prophet import backtest

SEASON_LENGTH = 12
COVARIATE_CAPABLE_MODELS = {"prophet", "arima"}
MODEL_LABELS = {
    "prophet": "Prophet",
    "ets": "Holt-Winters (ETS)",
    "theta": "Theta",
    "arima": "SARIMA",
    "seasonal_naive": "Seasonal naive",
}
SELECTABLE_MODELS = ["prophet", "ets", "theta", "arima", "seasonal_naive"]


@dataclass(frozen=True)
class ModelSelection:
    model_name: str
    primary_metric: str
    primary_score: float | None
    note: str | None
    diagnostics_df: pd.DataFrame


def build_model(
    model_name: str,
    *,
    n_observations: int,
    changepoint_prior_scale: float = 0.05,
    interval_width: float = 0.90,
):
    """Instantiate a Darts model by name, adapting to history length."""
    has_two_seasons = n_observations >= (2 * SEASON_LENGTH) + 2

    if model_name == "prophet":
        return Prophet(
            changepoint_prior_scale=changepoint_prior_scale,
            interval_width=interval_width,
        )
    if model_name == "ets":
        if has_two_seasons:
            return ExponentialSmoothing(seasonal_periods=SEASON_LENGTH)
        return ExponentialSmoothing(seasonal=None)
    if model_name == "theta":
        if has_two_seasons:
            return Theta(
                seasonality_period=SEASON_LENGTH,
                season_mode=SeasonalityMode.ADDITIVE,
            )
        return Theta(season_mode=SeasonalityMode.NONE)
    if model_name == "arima":
        if has_two_seasons:
            return ARIMA(p=1, d=1, q=1, seasonal_order=(1, 1, 1, SEASON_LENGTH))
        return ARIMA(p=1, d=1, q=1)
    if model_name == "seasonal_naive":
        if n_observations >= SEASON_LENGTH + 1:
            return NaiveSeasonal(K=SEASON_LENGTH)
        return NaiveSeasonal(K=1)

    raise ValueError(
        f"Unknown model '{model_name}'. Valid options: {SELECTABLE_MODELS + ['auto']}."
    )


def build_calendar_covariate_series(
    df_prepared: pd.DataFrame, forecast_periods: int
) -> TimeSeries:
    """Business-days-per-month covariate spanning history plus the horizon.

    Months with more business days generate more outpatient visits, so this
    calendar signal helps covariate-capable models (Prophet, SARIMA) explain
    month-to-month wiggle that pure seasonality misses.
    """
    start = pd.to_datetime(df_prepared["ds"].min())
    last_hist = pd.to_datetime(df_prepared["ds"].max())
    end = last_hist + pd.offsets.MonthBegin(forecast_periods)
    month_starts = pd.date_range(start, end, freq="MS")
    business_days = [
        int(
            np.busday_count(
                month_start.date(),
                (month_start + pd.offsets.MonthBegin(1)).date(),
            )
        )
        for month_start in month_starts
    ]
    covariate_df = pd.DataFrame({"ds": month_starts, "business_days": business_days})
    return TimeSeries.from_dataframe(
        covariate_df, time_col="ds", value_cols="business_days"
    )


def _covariate_kwargs(model, future_covariates) -> dict[str, Any]:
    if future_covariates is not None and getattr(
        model, "supports_future_covariates", False
    ):
        return {"future_covariates": future_covariates}
    return {}


def fit_and_forecast_model(
    df_prepared: pd.DataFrame,
    forecast_periods: int,
    model_name: str,
    *,
    changepoint_prior_scale: float = 0.05,
    interval_width: float = 0.90,
    future_covariates: TimeSeries | None = None,
) -> tuple[TimeSeries, TimeSeries, Any]:
    """Fit the named Darts model and produce a point forecast."""
    series = TimeSeries.from_dataframe(df_prepared, time_col="ds", value_cols="y")
    model = build_model(
        model_name,
        n_observations=len(series),
        changepoint_prior_scale=changepoint_prior_scale,
        interval_width=interval_width,
    )
    if len(series) < model.min_train_series_length:
        raise ValueError(
            f"{MODEL_LABELS[model_name]} requires at least {model.min_train_series_length} monthly observations; received {len(series)}. Use a shorter-history model such as seasonal naive or ETS, or provide more history."
        )
    covariate_kwargs = _covariate_kwargs(model, future_covariates)
    model.fit(series, **covariate_kwargs)
    forecast = model.predict(forecast_periods, **covariate_kwargs).map(
        lambda values: np.maximum(values, 0.0)
    )
    return series, forecast, model


def _balanced_scores(diagnostics_df: pd.DataFrame) -> pd.Series:
    """MAPE + RMSE balance: each metric normalized by the best candidate."""
    mape = pd.to_numeric(diagnostics_df["mape"], errors="coerce")
    rmse = pd.to_numeric(diagnostics_df["rmse"], errors="coerce")
    mape_floor = max(float(mape.min(skipna=True)), 1e-9)
    rmse_floor = max(float(rmse.min(skipna=True)), 1e-9)
    return (mape / mape_floor) + (rmse / rmse_floor)


def select_forecast_model(
    df_prepared: pd.DataFrame,
    forecast_periods: int,
    *,
    changepoint_prior_scale: float = 0.05,
    interval_width: float = 0.90,
    candidates: list[str] | None = None,
    future_covariates: TimeSeries | None = None,
    progress_callback=None,
) -> ModelSelection:
    """Backtest candidate models and select the best balanced MAPE+RMSE."""
    candidate_names = (
        list(SELECTABLE_MODELS)
        if candidates is None
        else list(dict.fromkeys(candidates))
    )
    if not candidate_names or set(candidate_names).difference(SELECTABLE_MODELS):
        raise ValueError(f"candidates must be a nonempty subset of {SELECTABLE_MODELS}")
    series = TimeSeries.from_dataframe(df_prepared, time_col="ds", value_cols="y")

    # Pick one feasible window for the candidate set. Darts may otherwise
    # move each model's start independently to satisfy its training minimum.
    candidate_models, ineligible = {}, {}
    for name in candidate_names:
        try:
            model = build_model(
                name,
                n_observations=len(series),
                changepoint_prior_scale=changepoint_prior_scale,
                interval_width=interval_width,
            )
            if model.min_train_series_length + forecast_periods > len(series):
                raise ValueError(
                    f"requires at least {model.min_train_series_length + forecast_periods} observations for a full validation horizon"
                )
            candidate_models[name] = model
        except Exception as exc:
            ineligible[name] = str(exc)
    common_index = min(
        max(
            int((len(series) - 1) * 0.7),
            max(
                (m.min_train_series_length for m in candidate_models.values()),
                default=0,
            ),
        ),
        len(series) - forecast_periods,
    )
    common_start = series.time_index[max(common_index, 0)]
    expected_scored = (
        (len(series) - forecast_periods - common_index) // forecast_periods + 1
    ) * forecast_periods

    rows: list[dict[str, Any]] = []
    for candidate in candidate_names:
        if progress_callback is not None:
            progress_callback(f"Comparing model: {MODEL_LABELS[candidate]}")
        row: dict[str, Any] = {
            "parameter_family": "forecast_model",
            "candidate_value": MODEL_LABELS.get(candidate, candidate),
            "model_key": candidate,
            "primary_score_name": "MAPE+RMSE balance",
            "primary_score_value": float("nan"),
            "tie_breaker_name": "RMSE",
            "tie_breaker_value": float("nan"),
            "mape": float("nan"),
            "mae": float("nan"),
            "rmse": float("nan"),
            "selected": False,
            "note": "",
        }
        try:
            if candidate in ineligible:
                raise ValueError(ineligible[candidate])
            model = candidate_models[candidate]
            covariate_kwargs = _covariate_kwargs(model, future_covariates)
            metrics = backtest(
                model=model,
                series=series,
                forecast_horizon=forecast_periods,
                start=common_start,
                stride=forecast_periods,
                strict_protocol=True,
                future_covariates=covariate_kwargs.get("future_covariates"),
            )
            if metrics.get("n_scored", expected_scored) != expected_scored:
                raise ValueError(
                    "Candidate did not score the common validation window."
                )
            if covariate_kwargs:
                row["note"] = "Used calendar covariates."
            row["mape"] = float(metrics.get("MAPE", float("nan")))
            row["mae"] = float(metrics.get("MAE", float("nan")))
            row["rmse"] = float(metrics.get("RMSE", float("nan")))
            row["tie_breaker_value"] = row["rmse"]
            row["n_scored"] = metrics.get("n_scored")
            row["n_cutoffs"] = metrics.get("n_cutoffs")
            row["effective_variant"] = model_variant(candidate, len(series))
            row["validation_start"] = str(common_start.date())
            row["validation_stride_months"] = forecast_periods
        except Exception as candidate_error:  # noqa: BLE001
            row["note"] = f"Candidate failed: {candidate_error}"
        rows.append(row)

    diagnostics_df = pd.DataFrame(rows)
    # MAPE is undefined for all-zero demand. Compare every candidate using
    # the same zero-safe policy rather than discarding a perfect zero forecast.
    eligible_scores = diagnostics_df[
        np.isfinite(diagnostics_df["mae"]) & np.isfinite(diagnostics_df["rmse"])
    ]
    metric = "mape" if np.isfinite(eligible_scores["mape"]).all() else "mae"
    score_name = (
        "MAPE+RMSE balance"
        if metric == "mape"
        else "MAE+RMSE balance (MAPE unavailable)"
    )
    diagnostics_df["primary_score_name"] = score_name
    scored = diagnostics_df[
        np.isfinite(diagnostics_df[metric]) & np.isfinite(diagnostics_df["rmse"])
    ].copy()
    if scored.empty:
        raise ValueError(
            "No eligible model produced finite validation scores: "
            + "; ".join(
                f"{r.candidate_value}: {r.note or 'insufficient validation history'}"
                for r in diagnostics_df.itertuples()
            )
        )
    first = pd.to_numeric(scored[metric])
    second = pd.to_numeric(scored["rmse"])
    scored["primary_score_value"] = first / max(
        float(first.min()), 1e-9
    ) + second / max(float(second.min()), 1e-9)
    diagnostics_df.loc[scored.index, "primary_score_value"] = scored[
        "primary_score_value"
    ]
    best_row = scored.sort_values(["primary_score_value", "rmse"]).iloc[0]
    selected_name = str(best_row["model_key"])
    diagnostics_df.loc[diagnostics_df["model_key"] == selected_name, "selected"] = True

    note_parts = [
        f"Auto-selected {MODEL_LABELS.get(selected_name, selected_name)} "
        f"(selection {metric.upper()} {float(best_row[metric]):.2f}, "
        f"RMSE {float(best_row['rmse']):.2f})."
    ]
    naive_rows = scored[scored["model_key"] == "seasonal_naive"]
    if not naive_rows.empty and selected_name != "seasonal_naive":
        naive_mape = float(naive_rows.iloc[0]["mape"])
        best_mape = float(best_row["mape"])
        if np.isfinite(naive_mape) and naive_mape > 0:
            improvement = (naive_mape - best_mape) / naive_mape * 100.0
            if improvement >= 0:
                note_parts.append(
                    f"It beats the seasonal naive baseline by {improvement:.1f}% MAPE."
                )
            else:
                note_parts.append(
                    "Warning: it does not beat the seasonal naive baseline; "
                    "treat this forecast with caution."
                )
    elif selected_name == "seasonal_naive":
        note_parts.append(
            "The seasonal naive baseline outperformed all models on this data, "
            "which usually means history is short or highly repetitive."
        )

    return ModelSelection(
        model_name=selected_name,
        primary_metric=score_name,
        primary_score=float(best_row["primary_score_value"]),
        note=" ".join(note_parts),
        diagnostics_df=diagnostics_df.drop(columns=["model_key"]),
    )


def model_variant(name: str, n_observations: int) -> str:
    if name in {"ets", "theta", "arima"}:
        return f"{name}: {'seasonal (12 months)' if n_observations >= 26 else 'non-seasonal (short history)'}"
    if name == "seasonal_naive":
        return (
            f"seasonal_naive: repeat last {12 if n_observations >= 13 else 1} month(s)"
        )
    return name
