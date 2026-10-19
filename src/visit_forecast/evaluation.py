"""Evaluation of a configuration chosen before a reserved final time block."""

from __future__ import annotations

import numpy as np
import pandas as pd
from .models import fit_and_forecast_model
from .model_prophet import (
    collect_conformal_residuals,
    build_conformal_intervals,
    prophet_future_intervals,
)


def evaluate_reserved_block(
    training,
    actual,
    model_name,
    horizon,
    changepoint,
    coverage,
    method,
    covariates=None,
):
    series, forecast, model = fit_and_forecast_model(
        training,
        horizon,
        model_name,
        changepoint_prior_scale=changepoint,
        interval_width=coverage,
        future_covariates=covariates,
    )
    predictions = forecast.to_dataframe().iloc[:, 0].to_numpy()
    observed = actual.y.to_numpy(dtype=float)
    residual = observed - predictions
    nonzero = observed != 0
    metrics = {
        "MAPE": float(np.mean(np.abs(residual[nonzero] / observed[nonzero])) * 100)
        if nonzero.any()
        else float("nan"),
        "MAE": float(np.mean(np.abs(residual))),
        "RMSE": float(np.sqrt(np.mean(residual**2))),
        "n_scored": len(actual),
        "n_zero_actuals": int((~nonzero).sum()),
        "n_cutoffs": 1,
    }
    summary = {
        "mode": "reserved_final_block",
        "training_end": str(training.ds.max().date()),
        "evaluation_start": str(actual.ds.min().date()),
        "evaluation_end": str(actual.ds.max().date()),
        "n_evaluation_forecasts": len(actual),
        "coverage": None,
        "coverage_note": "Baseline, before capacity assumptions; one reserved forecast origin.",
        "interval_source": "unavailable",
        "metrics": metrics,
    }
    try:
        frame = pd.DataFrame(
            {
                "ds": actual.ds.to_numpy(),
                "forecast_month": np.arange(1, horizon + 1),
                "yhat_original": predictions,
                "yhat_adjusted": predictions,
            }
        )
        if method == "conformal":
            residuals = collect_conformal_residuals(
                model, series, horizon, future_covariates=covariates
            )
            # Every outcome here precedes the final evaluation origin, including
            # outcomes used for earlier development diagnostics.
            residuals["conformal_partition"] = "calibration"
            frame, _, _ = build_conformal_intervals(frame, residuals, coverage)
            lower, upper = frame.yhat_lower_conformal, frame.yhat_upper_conformal
        else:
            covframe = (
                covariates.to_dataframe().reset_index()
                if covariates is not None
                else None
            )
            bounds = prophet_future_intervals(
                model, training.ds.max(), horizon, covariates_df=covframe
            )
            lower, upper = (
                bounds.yhat_lower.clip(lower=0),
                bounds.yhat_upper.clip(lower=0),
            )
        if np.isfinite(np.column_stack([lower, upper])).all():
            summary["coverage"] = float(
                np.mean(
                    (observed >= np.asarray(lower)) & (observed <= np.asarray(upper))
                )
            )
            summary["interval_source"] = method
    except Exception as exc:
        summary["coverage_note"] = (
            f"Reserved-block intervals unavailable: {type(exc).__name__}: {exc}"
        )
    return metrics, summary
