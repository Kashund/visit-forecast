from __future__ import annotations

import numpy as np
import pandas as pd

from visit_forecast.model_prophet import (
    _collect_forecast_alignment_rows,
    _partition_conformal_residuals,
    build_conformal_intervals,
)


class MockTimeSeries:
    def __init__(self, dataframe: pd.DataFrame) -> None:
        self._dataframe = dataframe

    def pd_dataframe(self) -> pd.DataFrame:
        return self._dataframe


def test_collect_forecast_alignment_rows_assigns_horizon_steps_per_cutoff() -> None:
    actual = MockTimeSeries(
        pd.DataFrame(
            {
                "ds": pd.date_range("2024-01-01", periods=6, freq="MS"),
                "y": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0],
            }
        )
    )
    forecasts = [
        MockTimeSeries(
            pd.DataFrame(
                {
                    "ds": pd.to_datetime(["2024-03-01", "2024-04-01"]),
                    "y": [100.0, 100.0],
                }
            )
        ),
        MockTimeSeries(
            pd.DataFrame(
                {
                    "ds": pd.to_datetime(["2024-04-01", "2024-05-01"]),
                    "y": [101.0, 101.0],
                }
            )
        ),
    ]

    result = _collect_forecast_alignment_rows(actual, forecasts)

    assert result["horizon_step"].tolist() == [1, 2, 1, 2]
    assert result["cutoff"].dt.strftime("%Y-%m-%d").tolist() == [
        "2024-02-01",
        "2024-02-01",
        "2024-03-01",
        "2024-03-01",
    ]


def test_build_conformal_intervals_uses_asymmetric_quantiles_per_horizon() -> None:
    residual_values = np.array([-10.0, -5.0, -2.0, -1.0, 1.0, 2.0, 5.0, 10.0])
    residuals_df = pd.DataFrame(
        {
            "cutoff": pd.date_range("2023-01-01", periods=8, freq="MS"),
            "ds": pd.date_range("2023-02-01", periods=8, freq="MS"),
            "horizon_step": [1] * 8,
            "actual": 100.0 + residual_values,
            "predicted": [100.0] * 8,
            "residual": residual_values,
            "abs_residual": np.abs(residual_values),
        }
    )
    future_df = pd.DataFrame(
        {
            "ds": pd.to_datetime(["2024-01-01"]),
            "forecast_month": [1],
            "yhat_original": [100.0],
            "yhat_adjusted": [100.0],
        }
    )

    result_df, diagnostics_df, summary = build_conformal_intervals(
        future_df=future_df,
        residuals_df=residuals_df,
        target_coverage=0.90,
    )

    expected_lower = 100.0 + float(np.quantile(residual_values, 0.05))
    expected_upper = 100.0 + float(np.quantile(residual_values, 0.95))
    assert result_df.loc[0, "yhat_lower_conformal"] == expected_lower
    assert result_df.loc[0, "yhat_upper_conformal"] == expected_upper
    assert not bool(diagnostics_df.loc[0, "fallback_used"])
    assert summary["fallback_used"] is False


def test_build_conformal_intervals_uses_pooled_fallback_and_clips_lower_bound() -> None:
    residual_values = np.array([-10.0, 2.0, -1.0, 4.0])
    residuals_df = pd.DataFrame(
        {
            "cutoff": pd.date_range("2023-01-01", periods=4, freq="MS"),
            "ds": pd.date_range("2023-02-01", periods=4, freq="MS"),
            "horizon_step": [1, 1, 2, 2],
            "actual": [0.0, 0.0, 0.0, 0.0],
            "predicted": [3.0, 3.0, 3.0, 3.0],
            "residual": residual_values,
            "abs_residual": np.abs(residual_values),
        }
    )
    future_df = pd.DataFrame(
        {
            "ds": pd.to_datetime(["2024-01-01"]),
            "forecast_month": [1],
            "yhat_original": [3.0],
            "yhat_adjusted": [3.0],
        }
    )

    result_df, diagnostics_df, summary = build_conformal_intervals(
        future_df=future_df,
        residuals_df=residuals_df,
        target_coverage=0.90,
    )

    pooled_abs_quantile = float(np.quantile(np.abs(residual_values), 0.90))
    assert result_df.loc[0, "yhat_lower_conformal"] == 0.0
    assert result_df.loc[0, "yhat_upper_conformal"] == 3.0 + pooled_abs_quantile
    assert bool(diagnostics_df.loc[0, "fallback_used"])
    assert summary["fallback_used"] is True


def test_partition_conformal_residuals_holds_out_newest_cutoffs() -> None:
    residuals_df = pd.DataFrame(
        {
            "cutoff": pd.date_range("2023-01-01", periods=8, freq="MS"),
            "horizon_step": [1] * 8,
        }
    )

    partitioned = _partition_conformal_residuals(residuals_df)

    assert partitioned["conformal_partition"].tolist() == [
        "calibration",
        "calibration",
        "calibration",
        "calibration",
        "calibration",
        "calibration",
        "evaluation",
        "evaluation",
    ]
    calibration_max = partitioned.loc[
        partitioned["conformal_partition"].eq("calibration"), "cutoff"
    ].max()
    evaluation_min = partitioned.loc[
        partitioned["conformal_partition"].eq("evaluation"), "cutoff"
    ].min()
    assert calibration_max < evaluation_min


def test_build_conformal_intervals_reports_held_out_coverage() -> None:
    calibration_residuals = np.array(
        [-10.0, -5.0, -2.0, -1.0, 1.0, 2.0, 5.0, 10.0]
    )
    evaluation_residuals = np.array([0.0, 30.0])
    residual_values = np.concatenate(
        [calibration_residuals, evaluation_residuals]
    )
    residuals_df = pd.DataFrame(
        {
            "cutoff": pd.date_range("2023-01-01", periods=10, freq="MS"),
            "ds": pd.date_range("2023-02-01", periods=10, freq="MS"),
            "horizon_step": [1] * 10,
            "actual": 100.0 + residual_values,
            "predicted": [100.0] * 10,
            "residual": residual_values,
            "abs_residual": np.abs(residual_values),
            "conformal_partition": ["calibration"] * 8 + ["evaluation"] * 2,
        }
    )
    future_df = pd.DataFrame(
        {
            "ds": pd.to_datetime(["2024-01-01"]),
            "forecast_month": [1],
            "yhat_original": [100.0],
            "yhat_adjusted": [100.0],
        }
    )

    _, diagnostics_df, summary = build_conformal_intervals(
        future_df=future_df,
        residuals_df=residuals_df,
        target_coverage=0.90,
    )

    assert diagnostics_df.loc[0, "n_calibration_forecasts"] == 8
    assert diagnostics_df.loc[0, "n_evaluation_forecasts"] == 2
    assert diagnostics_df.loc[0, "empirical_coverage"] == 0.5
    assert summary["empirical_coverage_overall"] == 0.5
    assert summary["validation_mode"] == "held_out_cutoffs"


def test_build_conformal_intervals_does_not_resubstitute_partitioned_residuals() -> None:
    residuals_df = pd.DataFrame(
        {
            "cutoff": pd.date_range("2023-01-01", periods=3, freq="MS"),
            "ds": pd.date_range("2023-02-01", periods=3, freq="MS"),
            "horizon_step": [1] * 3,
            "actual": [98.0, 100.0, 102.0],
            "predicted": [100.0] * 3,
            "residual": [-2.0, 0.0, 2.0],
            "abs_residual": [2.0, 0.0, 2.0],
            "conformal_partition": ["calibration"] * 3,
        }
    )
    future_df = pd.DataFrame(
        {
            "ds": pd.to_datetime(["2024-01-01"]),
            "forecast_month": [1],
            "yhat_original": [100.0],
            "yhat_adjusted": [100.0],
        }
    )

    _, _, summary = build_conformal_intervals(
        future_df=future_df,
        residuals_df=residuals_df,
        target_coverage=0.90,
    )

    assert np.isnan(summary["empirical_coverage_overall"])
    assert summary["n_evaluation_forecasts"] == 0
    assert summary["validation_mode"] == "unavailable"
