import pandas as pd

from visit_forecast import service
from visit_forecast.tuning import (
    TuningSelection,
    select_joint_forecast_configuration,
    select_prophet_hyperparameters,
    select_uncertainty_configuration,
)


class DummyModel:
    def __init__(self, changepoint_prior_scale: float, interval_width: float) -> None:
        self.changepoint_prior_scale = changepoint_prior_scale
        self.interval_width = interval_width


def test_select_prophet_hyperparameters_balances_mape_and_rmse_for_changepoint_selection(
    monkeypatch,
):
    rmse_by_cp = {0.01: 6.0, 0.03: 4.0, 0.05: 4.5, 0.10: 5.0, 0.20: 7.0}
    mape_by_cp = {0.01: 24.0, 0.03: 20.0, 0.05: 10.0, 0.10: 14.0, 0.20: 30.0}
    mae_by_cp = {0.01: 3.0, 0.03: 2.0, 0.05: 2.3, 0.10: 2.8, 0.20: 3.4}
    coverage_by_iw = {
        0.80: [0.73, 0.76],
        0.85: [0.82, 0.84],
        0.90: [0.89, 0.90],
        0.95: [0.90, 0.91],
    }

    def fake_fit_and_forecast(
        df_prepared, forecast_periods, changepoint_prior_scale, interval_width
    ):
        return object(), object(), DummyModel(changepoint_prior_scale, interval_width)

    def fake_backtest(model, series, forecast_horizon, start, stride):
        return {
            "MAPE": mape_by_cp[round(model.changepoint_prior_scale, 2)],
            "MAE": mae_by_cp[round(model.changepoint_prior_scale, 2)],
            "RMSE": rmse_by_cp[round(model.changepoint_prior_scale, 2)],
        }

    def fake_prophet_cv_metrics(model, initial, period, horizon):
        coverages = coverage_by_iw[round(model.interval_width, 2)]
        return (
            pd.DataFrame(),
            pd.DataFrame(
                {
                    "horizon": pd.to_timedelta(["30 days", "60 days"]),
                    "coverage": coverages,
                }
            ),
        )

    monkeypatch.setattr("visit_forecast.tuning.fit_and_forecast", fake_fit_and_forecast)
    monkeypatch.setattr("visit_forecast.tuning.backtest", fake_backtest)
    monkeypatch.setattr(
        "visit_forecast.tuning.prophet_cross_validation_metrics",
        fake_prophet_cv_metrics,
    )

    result = select_prophet_hyperparameters(
        df_prepared=pd.DataFrame({"ds": pd.date_range("2020-01-01", periods=12, freq="MS")}),
        forecast_periods=3,
    )

    assert result.changepoint_prior_scale == 0.05
    assert result.interval_width == 0.90
    assert result.primary_metric == "MAPE+RMSE"
    assert result.primary_score is not None
    assert result.diagnostics_df["selected"].sum() == 2


def test_select_prophet_hyperparameters_breaks_changepoint_ties_by_smaller_value(
    monkeypatch,
):
    def fake_fit_and_forecast(
        df_prepared, forecast_periods, changepoint_prior_scale, interval_width
    ):
        return object(), object(), DummyModel(changepoint_prior_scale, interval_width)

    def fake_backtest(model, series, forecast_horizon, start, stride):
        return {"MAPE": 10.0, "MAE": 2.0, "RMSE": 4.0}

    def fake_prophet_cv_metrics(model, initial, period, horizon):
        return (
            pd.DataFrame(),
            pd.DataFrame(
                {
                    "horizon": pd.to_timedelta(["30 days"]),
                    "coverage": [model.interval_width],
                }
            ),
        )

    monkeypatch.setattr("visit_forecast.tuning.fit_and_forecast", fake_fit_and_forecast)
    monkeypatch.setattr("visit_forecast.tuning.backtest", fake_backtest)
    monkeypatch.setattr(
        "visit_forecast.tuning.prophet_cross_validation_metrics",
        fake_prophet_cv_metrics,
    )

    result = select_prophet_hyperparameters(
        df_prepared=pd.DataFrame({"ds": pd.date_range("2020-01-01", periods=12, freq="MS")}),
        forecast_periods=3,
        changepoint_candidates=[0.03, 0.05],
        interval_width_candidates=[0.90],
    )

    assert result.changepoint_prior_scale == 0.03


def test_select_prophet_hyperparameters_records_interval_width_fallback_when_cv_unavailable(
    monkeypatch,
):
    def fake_fit_and_forecast(
        df_prepared, forecast_periods, changepoint_prior_scale, interval_width
    ):
        return object(), object(), DummyModel(changepoint_prior_scale, interval_width)

    def fake_backtest(model, series, forecast_horizon, start, stride):
        return {"MAPE": 10.0, "MAE": 2.0, "RMSE": 4.0}

    monkeypatch.setattr("visit_forecast.tuning.fit_and_forecast", fake_fit_and_forecast)
    monkeypatch.setattr("visit_forecast.tuning.backtest", fake_backtest)
    monkeypatch.setattr(
        "visit_forecast.tuning.prophet_cross_validation_metrics",
        lambda model, initial, period, horizon: (None, None),
    )

    result = select_prophet_hyperparameters(
        df_prepared=pd.DataFrame({"ds": pd.date_range("2020-01-01", periods=12, freq="MS")}),
        forecast_periods=3,
    )

    assert result.interval_width == 0.90
    assert result.note is not None
    assert "fallback interval_width=0.90" in result.note


def test_select_prophet_hyperparameters_skips_interval_width_tuning_for_conformal_mode(
    monkeypatch,
):
    def fake_fit_and_forecast(
        df_prepared, forecast_periods, changepoint_prior_scale, interval_width
    ):
        return object(), object(), DummyModel(changepoint_prior_scale, interval_width)

    def fake_backtest(model, series, forecast_horizon, start, stride):
        return {"MAPE": 10.0, "MAE": 2.0, "RMSE": 4.0}

    monkeypatch.setattr("visit_forecast.tuning.fit_and_forecast", fake_fit_and_forecast)
    monkeypatch.setattr("visit_forecast.tuning.backtest", fake_backtest)

    result = select_prophet_hyperparameters(
        df_prepared=pd.DataFrame({"ds": pd.date_range("2020-01-01", periods=12, freq="MS")}),
        forecast_periods=3,
        tune_interval_width=False,
        fixed_interval_width=0.92,
    )

    assert result.interval_width == 0.92
    assert result.note is not None
    assert "fixed interval / target coverage" in result.note.lower()
    assert (
        result.diagnostics_df["parameter_family"] == "interval_width"
    ).any()


def test_select_joint_forecast_configuration_balances_model_metrics_and_uncertainty(
    monkeypatch,
):
    prophet_coverage_by_cp = {
        0.03: 0.70,
        0.05: 0.79,
    }
    conformal_summary_by_cp = {
        0.03: {
            "empirical_coverage_overall": 0.87,
            "empirical_coverage_by_horizon": {1: 0.87},
            "avg_interval_width": 14.0,
            "median_interval_width": 14.0,
            "interval_width_over_mean_actual": 0.12,
            "n_calibration_forecasts": 12,
            "fallback_used": False,
        },
        0.05: {
            "empirical_coverage_overall": 0.90,
            "empirical_coverage_by_horizon": {1: 0.90},
            "avg_interval_width": 13.0,
            "median_interval_width": 13.0,
            "interval_width_over_mean_actual": 0.10,
            "n_calibration_forecasts": 12,
            "fallback_used": False,
        },
    }

    class DummySeries:
        pass

    class DummyForecast:
        def pd_dataframe(self):
            return pd.DataFrame(
                {
                    "ds": pd.to_datetime(["2024-04-01"]),
                    "y": [130.0],
                }
            )

    class DummyModel:
        def __init__(self, cp: float, iw: float) -> None:
            self.changepoint_prior_scale = cp
            self.interval_width = iw

    def fake_prophet_cv_metrics(model, initial, period, horizon):
        return (
            pd.DataFrame(),
            pd.DataFrame(
                {
                    "horizon": pd.to_timedelta(["30 days"]),
                    "coverage": [prophet_coverage_by_cp[round(model.changepoint_prior_scale, 2)]],
                }
            ),
        )

    current_cp = {"value": 0.03}

    def fake_fit_and_forecast_with_cp(
        df_prepared, forecast_periods, changepoint_prior_scale, interval_width
    ):
        current_cp["value"] = round(changepoint_prior_scale, 2)
        return DummySeries(), DummyForecast(), DummyModel(changepoint_prior_scale, interval_width)

    def fake_backtest_with_method(model, series, forecast_horizon, start, stride):
        if round(model.interval_width, 2) == 0.85:
            return {
                "MAPE": 10.0 if round(model.changepoint_prior_scale, 2) == 0.03 else 9.5,
                "MAE": 4.0 if round(model.changepoint_prior_scale, 2) == 0.03 else 3.5,
                "RMSE": 5.0 if round(model.changepoint_prior_scale, 2) == 0.03 else 4.6,
            }
        return {
            "MAPE": 9.0 if round(model.changepoint_prior_scale, 2) == 0.03 else 8.8,
            "MAE": 3.0 if round(model.changepoint_prior_scale, 2) == 0.03 else 2.8,
            "RMSE": 4.0 if round(model.changepoint_prior_scale, 2) == 0.03 else 3.8,
        }

    monkeypatch.setattr("visit_forecast.tuning.fit_and_forecast", fake_fit_and_forecast_with_cp)
    monkeypatch.setattr("visit_forecast.tuning.backtest", fake_backtest_with_method)
    monkeypatch.setattr(
        "visit_forecast.tuning.prophet_cross_validation_metrics",
        fake_prophet_cv_metrics,
    )
    monkeypatch.setattr(
        "visit_forecast.tuning.collect_conformal_residuals",
        lambda **kwargs: pd.DataFrame({"residual": [1.0]}),
    )
    monkeypatch.setattr(
        "visit_forecast.tuning.build_conformal_intervals",
        lambda future_df, residuals_df, target_coverage: (
            future_df,
            pd.DataFrame(),
            conformal_summary_by_cp[current_cp["value"]],
        ),
    )

    result = select_joint_forecast_configuration(
        df_prepared=pd.DataFrame({"ds": pd.date_range("2020-01-01", periods=12, freq="MS")}),
        forecast_periods=1,
        target_coverage=0.90,
        changepoint_candidates=[0.03, 0.05],
        interval_width_candidates=[0.85],
    )

    assert result.uncertainty_method == "conformal"
    assert result.changepoint_prior_scale == 0.05
    assert result.interval_width == 0.90
    assert result.primary_metric == "Joint model score"
    assert result.primary_score is not None
    assert result.diagnostics_df["selected"].sum() == 1


def test_select_uncertainty_configuration_can_choose_optimal_target_coverage(
    monkeypatch,
):
    def fake_fit_and_forecast(
        df_prepared, forecast_periods, changepoint_prior_scale, interval_width
    ):
        return object(), object(), DummyModel(changepoint_prior_scale, interval_width)

    def fake_prophet_cv_metrics(model, initial, period, horizon):
        return (
            pd.DataFrame(),
            pd.DataFrame(
                {
                    "horizon": pd.to_timedelta(["30 days"]),
                    "coverage": [
                        0.80 if round(model.interval_width, 2) == 0.80 else 0.84
                    ],
                }
            ),
        )

    def fake_conformal_summary(
        model, series, forecast, forecast_periods, target_coverage
    ):
        if round(float(target_coverage), 2) == 0.80:
            return {
                "empirical_coverage_overall": 0.80,
                "coverage_gap": 0.00,
                "interval_width_over_mean_actual": 0.10,
                "avg_interval_width": 12.0,
                "fallback_used": False,
            }
        return {
            "empirical_coverage_overall": 0.87,
            "coverage_gap": 0.03,
            "interval_width_over_mean_actual": 0.14,
            "avg_interval_width": 16.0,
            "fallback_used": False,
        }

    monkeypatch.setattr("visit_forecast.tuning.fit_and_forecast", fake_fit_and_forecast)
    monkeypatch.setattr(
        "visit_forecast.tuning.prophet_cross_validation_metrics",
        fake_prophet_cv_metrics,
    )
    monkeypatch.setattr(
        "visit_forecast.tuning._summarize_conformal_interval_validation",
        fake_conformal_summary,
    )

    result = select_uncertainty_configuration(
        df_prepared=pd.DataFrame({"ds": pd.date_range("2020-01-01", periods=12, freq="MS")}),
        forecast_periods=3,
        changepoint_prior_scale=0.03,
        target_coverage_candidates=[0.80, 0.90],
    )

    assert result.uncertainty_method == "conformal"
    assert result.interval_width == 0.80
    assert result.primary_metric == "coverage_gap"
    assert result.primary_score == 0.0
    assert result.diagnostics_df["selected"].sum() == 1


def test_forecast_visits_manual_preserves_selected_values(monkeypatch):
    fit_calls = []

    def fake_fit_and_forecast(
        df_prepared, forecast_periods, changepoint_prior_scale, interval_width
    ):
        fit_calls.append((changepoint_prior_scale, interval_width))
        return object(), object(), object()

    monkeypatch.setattr(service, "load_data", lambda *args, **kwargs: pd.DataFrame())
    monkeypatch.setattr(
        service,
        "prepare_dataframe",
        lambda **kwargs: (
            pd.DataFrame(
                {
                    "ds": pd.to_datetime(["2024-01-01", "2024-02-01"]),
                    "y": [100.0, 120.0],
                }
            ),
            "All",
        ),
    )
    monkeypatch.setattr(service, "fit_and_forecast", fake_fit_and_forecast)
    monkeypatch.setattr(
        service,
        "backtest",
        lambda **kwargs: {"MAPE": 1.0, "MAE": 2.0, "RMSE": 3.0},
    )
    monkeypatch.setattr(
        service,
        "build_future_forecast_df",
        lambda forecast, last_hist_date, capacity_phases: pd.DataFrame(
            {
                "ds": pd.to_datetime(["2024-03-01"]),
                "yhat_original": [130.0],
                "yhat_adjusted": [130.0],
            }
        ),
    )
    monkeypatch.setattr(
        service, "prophet_future_intervals", lambda *args, **kwargs: pd.DataFrame()
    )
    monkeypatch.setattr(service, "prophet_full_forecast_df", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        service,
        "prophet_cross_validation_metrics",
        lambda *args, **kwargs: (None, None),
    )

    result = service.forecast_visits(
        use_synthetic=False,
        df=pd.DataFrame(),
        forecast_periods=1,
        department="All",
        changepoint_prior_scale=0.12,
        interval_width=0.88,
        tuning_mode="manual",
    )

    assert fit_calls[-1] == (0.12, 0.88)
    assert result.tuning_mode == "manual"
    assert result.selected_changepoint_prior_scale == 0.12
    assert result.selected_interval_width == 0.88
    assert result.tuning_diagnostics_df is None


def test_forecast_visits_auto_uses_tuned_values(monkeypatch):
    fit_calls = []

    def fake_fit_and_forecast(
        df_prepared, forecast_periods, changepoint_prior_scale, interval_width
    ):
        fit_calls.append((changepoint_prior_scale, interval_width))
        return object(), object(), object()

    monkeypatch.setattr(service, "load_data", lambda *args, **kwargs: pd.DataFrame())
    monkeypatch.setattr(
        service,
        "prepare_dataframe",
        lambda **kwargs: (
            pd.DataFrame(
                {
                    "ds": pd.to_datetime(["2024-01-01", "2024-02-01"]),
                    "y": [100.0, 120.0],
                }
            ),
            "All",
        ),
    )
    monkeypatch.setattr(service, "fit_and_forecast", fake_fit_and_forecast)
    monkeypatch.setattr(
        service,
        "backtest",
        lambda **kwargs: {"MAPE": 1.0, "MAE": 2.0, "RMSE": 3.0},
    )
    monkeypatch.setattr(
        service,
        "build_future_forecast_df",
        lambda forecast, last_hist_date, capacity_phases: pd.DataFrame(
            {
                "ds": pd.to_datetime(["2024-03-01"]),
                "yhat_original": [130.0],
                "yhat_adjusted": [130.0],
            }
        ),
    )
    monkeypatch.setattr(
        service, "prophet_future_intervals", lambda *args, **kwargs: pd.DataFrame()
    )
    monkeypatch.setattr(service, "prophet_full_forecast_df", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        service,
        "prophet_cross_validation_metrics",
        lambda *args, **kwargs: (None, None),
    )
    monkeypatch.setattr(
        service,
        "select_prophet_hyperparameters",
        lambda **kwargs: TuningSelection(
            changepoint_prior_scale=0.03,
            interval_width=0.95,
            primary_metric="MAPE+RMSE",
            primary_score=1.08,
            note="Used fallback Prophet CV window.",
            diagnostics_df=pd.DataFrame(
                {
                    "parameter_family": ["changepoint_prior_scale", "interval_width"],
                    "candidate_value": [0.03, 0.95],
                    "primary_score_name": ["MAPE+RMSE balance", "coverage_gap_mean"],
                    "primary_score_value": [1.08, 0.01],
                    "tie_breaker_name": ["RMSE", "coverage_gap_longest_horizon"],
                    "tie_breaker_value": [2.1, 0.02],
                    "selected": [True, True],
                    "note": ["", "fallback_default"],
                }
            ),
        ),
    )

    result = service.forecast_visits(
        use_synthetic=False,
        df=pd.DataFrame(),
        forecast_periods=1,
        department="All",
        changepoint_prior_scale=0.12,
        interval_width=0.88,
        tuning_mode="auto",
    )

    assert fit_calls[-1] == (0.03, 0.95)
    assert result.tuning_mode == "auto"
    assert result.selected_changepoint_prior_scale == 0.03
    assert result.selected_interval_width == 0.95
    assert result.tuning_primary_metric == "MAPE+RMSE"
    assert result.tuning_primary_score == 1.08
    assert result.tuning_diagnostics_df is not None


def test_forecast_visits_conformal_scales_intervals_with_capacity_adjustments(monkeypatch):
    monkeypatch.setattr(service, "load_data", lambda *args, **kwargs: pd.DataFrame())
    monkeypatch.setattr(
        service,
        "prepare_dataframe",
        lambda **kwargs: (
            pd.DataFrame(
                {
                    "ds": pd.to_datetime(["2024-01-01", "2024-02-01"]),
                    "y": [100.0, 120.0],
                }
            ),
            "All",
        ),
    )
    monkeypatch.setattr(
        service,
        "fit_and_forecast",
        lambda **kwargs: ("series", "forecast", object()),
    )
    monkeypatch.setattr(
        service,
        "backtest",
        lambda **kwargs: {"MAPE": 1.0, "MAE": 2.0, "RMSE": 3.0},
    )
    monkeypatch.setattr(
        service,
        "build_future_forecast_df",
        lambda forecast, last_hist_date, capacity_phases: pd.DataFrame(
            {
                "ds": pd.to_datetime(["2024-03-01"]),
                "forecast_month": [1],
                "yhat_original": [100.0],
                "yhat_adjusted": [80.0],
                "adj_applied_any": [True],
                "adj_applied": [True],
            }
        ),
    )
    monkeypatch.setattr(
        service,
        "prophet_future_intervals",
        lambda *args, **kwargs: pd.DataFrame(
            {
                "ds": pd.to_datetime(["2024-03-01"]),
                "yhat_lower": [90.0],
                "yhat_upper": [110.0],
            }
        ),
    )
    monkeypatch.setattr(
        service,
        "collect_conformal_residuals",
        lambda **kwargs: pd.DataFrame({"residual": [1.0]}),
    )
    monkeypatch.setattr(
        service,
        "build_conformal_intervals",
        lambda future_df, residuals_df, target_coverage: (
            future_df.assign(
                yhat_lower_conformal=[90.0],
                yhat_upper_conformal=[110.0],
            ),
            pd.DataFrame(
                {
                    "horizon_step": [1],
                    "empirical_coverage": [0.91],
                    "target_coverage": [target_coverage],
                    "n_calibration_forecasts": [12],
                    "fallback_used": [False],
                }
            ),
            {
                "empirical_coverage_overall": 0.91,
                "empirical_coverage_by_horizon": {1: 0.91},
                "avg_interval_width": 20.0,
                "median_interval_width": 20.0,
                "interval_width_over_mean_actual": 0.2,
                "n_calibration_forecasts": 12,
                "fallback_used": False,
            },
        ),
    )
    monkeypatch.setattr(service, "prophet_full_forecast_df", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        service,
        "prophet_cross_validation_metrics",
        lambda *args, **kwargs: (None, None),
    )

    result = service.forecast_visits(
        use_synthetic=False,
        df=pd.DataFrame(),
        forecast_periods=1,
        department="All",
        target_coverage=0.90,
        uncertainty_method="conformal",
        tuning_mode="manual",
    )

    future_df = result.future_forecast_df
    assert future_df.loc[0, "yhat_lower_conformal"] == 72.0
    assert future_df.loc[0, "yhat_upper_conformal"] == 88.0
    assert future_df.loc[0, "yhat_lower"] == 72.0
    assert future_df.loc[0, "yhat_upper"] == 88.0
    assert future_df.loc[0, "interval_source"] == "conformal"
    assert result.interval_diagnostics_df is not None
    assert result.interval_summary_metrics["empirical_coverage_overall"] == 0.91


def test_forecast_visits_auto_skips_interval_width_tuning_in_conformal_mode(monkeypatch):
    select_calls = []

    monkeypatch.setattr(service, "load_data", lambda *args, **kwargs: pd.DataFrame())
    monkeypatch.setattr(
        service,
        "prepare_dataframe",
        lambda **kwargs: (
            pd.DataFrame(
                {
                    "ds": pd.to_datetime(["2024-01-01", "2024-02-01"]),
                    "y": [100.0, 120.0],
                }
            ),
            "All",
        ),
    )
    monkeypatch.setattr(
        service,
        "fit_and_forecast",
        lambda **kwargs: ("series", "forecast", object()),
    )
    monkeypatch.setattr(
        service,
        "backtest",
        lambda **kwargs: {"MAPE": 1.0, "MAE": 2.0, "RMSE": 3.0},
    )
    monkeypatch.setattr(
        service,
        "build_future_forecast_df",
        lambda forecast, last_hist_date, capacity_phases: pd.DataFrame(
            {
                "ds": pd.to_datetime(["2024-03-01"]),
                "forecast_month": [1],
                "yhat_original": [130.0],
                "yhat_adjusted": [130.0],
                "adj_applied_any": [False],
                "adj_applied": [False],
            }
        ),
    )
    monkeypatch.setattr(
        service, "prophet_future_intervals", lambda *args, **kwargs: pd.DataFrame()
    )
    monkeypatch.setattr(
        service,
        "collect_conformal_residuals",
        lambda **kwargs: pd.DataFrame(
            {
                "cutoff": [pd.Timestamp("2024-01-01")],
                "ds": [pd.Timestamp("2024-02-01")],
                "horizon_step": [1],
                "actual": [130.0],
                "predicted": [128.0],
                "residual": [2.0],
                "abs_residual": [2.0],
            }
        ),
    )
    monkeypatch.setattr(service, "prophet_full_forecast_df", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        service,
        "prophet_cross_validation_metrics",
        lambda *args, **kwargs: (None, None),
    )

    def fake_select_prophet_hyperparameters(**kwargs):
        select_calls.append(kwargs)
        return TuningSelection(
            changepoint_prior_scale=0.03,
            interval_width=0.92,
            primary_metric="MAPE+RMSE",
            primary_score=1.01,
            note="Skipped interval width tuning for conformal mode.",
            diagnostics_df=pd.DataFrame(),
        )

    monkeypatch.setattr(service, "select_prophet_hyperparameters", fake_select_prophet_hyperparameters)

    result = service.forecast_visits(
        use_synthetic=False,
        df=pd.DataFrame(),
        forecast_periods=1,
        department="All",
        target_coverage=0.92,
        uncertainty_method="conformal",
        tuning_mode="auto",
    )

    assert select_calls
    assert select_calls[0]["tune_interval_width"] is False
    assert select_calls[0]["fixed_interval_width"] == 0.92
    assert result.target_coverage == 0.92


def test_forecast_visits_auto_tune_respects_manual_prophet_interval(monkeypatch):
    select_calls = []
    fit_calls = []

    monkeypatch.setattr(service, "load_data", lambda *args, **kwargs: pd.DataFrame())
    monkeypatch.setattr(
        service,
        "prepare_dataframe",
        lambda **kwargs: (
            pd.DataFrame(
                {
                    "ds": pd.to_datetime(["2024-01-01", "2024-02-01"]),
                    "y": [100.0, 120.0],
                }
            ),
            "All",
        ),
    )

    def fake_fit_and_forecast(
        df_prepared, forecast_periods, changepoint_prior_scale, interval_width
    ):
        fit_calls.append((changepoint_prior_scale, interval_width))
        return "series", "forecast", object()

    monkeypatch.setattr(service, "fit_and_forecast", fake_fit_and_forecast)
    monkeypatch.setattr(
        service,
        "backtest",
        lambda **kwargs: {"MAPE": 1.0, "MAE": 2.0, "RMSE": 3.0},
    )
    monkeypatch.setattr(
        service,
        "build_future_forecast_df",
        lambda forecast, last_hist_date, capacity_phases: pd.DataFrame(
            {
                "ds": pd.to_datetime(["2024-03-01"]),
                "forecast_month": [1],
                "yhat_original": [130.0],
                "yhat_adjusted": [130.0],
                "adj_applied_any": [False],
                "adj_applied": [False],
            }
        ),
    )
    monkeypatch.setattr(
        service,
        "prophet_future_intervals",
        lambda *args, **kwargs: pd.DataFrame(
            {
                "ds": pd.to_datetime(["2024-03-01"]),
                "yhat_lower": [120.0],
                "yhat_upper": [140.0],
            }
        ),
    )
    monkeypatch.setattr(service, "prophet_full_forecast_df", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        service,
        "prophet_cross_validation_metrics",
        lambda *args, **kwargs: (
            pd.DataFrame(),
            pd.DataFrame(
                {
                    "horizon": pd.to_timedelta(["30 days"]),
                    "coverage": [0.88],
                }
            ),
        ),
    )

    def fake_select_prophet_hyperparameters(**kwargs):
        select_calls.append(kwargs)
        return TuningSelection(
            changepoint_prior_scale=0.03,
            interval_width=0.92,
            primary_metric="MAPE+RMSE",
            primary_score=1.01,
            note="Interval width tuning skipped; using fixed interval / target coverage 0.92 from the current uncertainty settings.",
            diagnostics_df=pd.DataFrame(),
        )

    monkeypatch.setattr(service, "select_prophet_hyperparameters", fake_select_prophet_hyperparameters)

    result = service.forecast_visits(
        use_synthetic=False,
        df=pd.DataFrame(),
        forecast_periods=1,
        department="All",
        target_coverage=0.92,
        uncertainty_method="prophet",
        tuning_mode="auto",
    )

    assert select_calls
    assert select_calls[0]["tune_interval_width"] is False
    assert select_calls[0]["fixed_interval_width"] == 0.92
    assert fit_calls[-1] == (0.03, 0.92)
    assert result.uncertainty_method == "prophet"
    assert result.target_coverage == 0.92


def test_forecast_visits_second_manual_conformal_run_updates_target_coverage_even_if_point_metrics_match(
    monkeypatch,
):
    fit_calls = []
    conformal_calls = []

    monkeypatch.setattr(service, "load_data", lambda *args, **kwargs: pd.DataFrame())
    monkeypatch.setattr(
        service,
        "prepare_dataframe",
        lambda **kwargs: (
            pd.DataFrame(
                {
                    "ds": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01"]),
                    "y": [100.0, 120.0, 130.0],
                }
            ),
            "All",
        ),
    )

    def fake_fit_and_forecast(
        df_prepared, forecast_periods, changepoint_prior_scale, interval_width
    ):
        fit_calls.append((changepoint_prior_scale, interval_width))
        return "series", "forecast", DummyModel(changepoint_prior_scale, interval_width)

    monkeypatch.setattr(service, "fit_and_forecast", fake_fit_and_forecast)
    monkeypatch.setattr(
        service,
        "backtest",
        lambda **kwargs: {"MAPE": 1.0, "MAE": 2.0, "RMSE": 3.0},
    )
    monkeypatch.setattr(
        service,
        "build_future_forecast_df",
        lambda forecast, last_hist_date, capacity_phases: pd.DataFrame(
            {
                "ds": pd.to_datetime(["2024-04-01"]),
                "forecast_month": [1],
                "yhat_original": [130.0],
                "yhat_adjusted": [130.0],
                "adj_applied_any": [False],
                "adj_applied": [False],
            }
        ),
    )
    monkeypatch.setattr(
        service,
        "collect_conformal_residuals",
        lambda **kwargs: pd.DataFrame(
            {
                "cutoff": [pd.Timestamp("2024-03-01")],
                "ds": [pd.Timestamp("2024-04-01")],
                "horizon_step": [1],
                "actual": [130.0],
                "predicted": [128.0],
                "residual": [2.0],
                "abs_residual": [2.0],
            }
        ),
    )

    def fake_build_conformal_intervals(future_df, residuals_df, target_coverage):
        conformal_calls.append(float(target_coverage))
        updated = future_df.copy()
        updated["yhat_lower_conformal"] = 120.0
        updated["yhat_upper_conformal"] = 140.0
        return (
            updated,
            pd.DataFrame(
                {
                    "horizon_step": [1],
                    "empirical_coverage": [0.74],
                    "target_coverage": [float(target_coverage)],
                }
            ),
            {
                "empirical_coverage_overall": 0.74,
                "empirical_coverage_by_horizon": {1: 0.74},
                "avg_interval_width": 20.0,
                "median_interval_width": 20.0,
                "interval_width_over_mean_actual": 0.15,
                "n_calibration_forecasts": 1,
                "fallback_used": False,
            },
        )

    monkeypatch.setattr(service, "build_conformal_intervals", fake_build_conformal_intervals)
    monkeypatch.setattr(
        service,
        "prophet_future_intervals",
        lambda *args, **kwargs: pd.DataFrame(
            {
                "ds": pd.to_datetime(["2024-04-01"]),
                "yhat_lower": [121.0],
                "yhat_upper": [139.0],
            }
        ),
    )
    monkeypatch.setattr(service, "prophet_full_forecast_df", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        service,
        "prophet_cross_validation_metrics",
        lambda *args, **kwargs: (
            pd.DataFrame(),
            pd.DataFrame(
                {
                    "horizon": pd.to_timedelta(["30 days"]),
                    "coverage": [0.88],
                }
            ),
        ),
    )

    def fake_select_prophet_hyperparameters(**kwargs):
        return TuningSelection(
            changepoint_prior_scale=0.03,
            interval_width=float(kwargs["fixed_interval_width"]),
            primary_metric="MAPE+RMSE",
            primary_score=1.01,
            note="Interval width tuning skipped for manual uncertainty.",
            diagnostics_df=pd.DataFrame(),
        )

    monkeypatch.setattr(service, "select_prophet_hyperparameters", fake_select_prophet_hyperparameters)

    first_result = service.forecast_visits(
        use_synthetic=False,
        df=pd.DataFrame(),
        forecast_periods=1,
        department="All",
        target_coverage=0.90,
        uncertainty_method="conformal",
        tuning_mode="auto",
    )
    second_result = service.forecast_visits(
        use_synthetic=False,
        df=pd.DataFrame(),
        forecast_periods=1,
        department="All",
        target_coverage=0.75,
        uncertainty_method="conformal",
        tuning_mode="auto",
    )

    assert conformal_calls == [0.90, 0.75]
    assert fit_calls[-2:] == [(0.03, 0.90), (0.03, 0.75)]
    assert first_result.performance_metrics == second_result.performance_metrics
    assert first_result.target_coverage == 0.90
    assert second_result.target_coverage == 0.75
    assert first_result.interval_summary_metrics["target_coverage"] == 0.90
    assert second_result.interval_summary_metrics["target_coverage"] == 0.75
    assert second_result.uncertainty_method == "conformal"
    assert second_result.future_forecast_df.loc[0, "interval_source"] == "conformal"


def test_forecast_visits_auto_selects_best_uncertainty_method(monkeypatch):
    monkeypatch.setattr(service, "load_data", lambda *args, **kwargs: pd.DataFrame())
    monkeypatch.setattr(
        service,
        "prepare_dataframe",
        lambda **kwargs: (
            pd.DataFrame(
                {
                    "ds": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01"]),
                    "y": [100.0, 120.0, 130.0],
                }
            ),
            "All",
        ),
    )
    monkeypatch.setattr(
        service,
        "fit_and_forecast",
        lambda **kwargs: ("series", "forecast", object()),
    )
    monkeypatch.setattr(
        service,
        "backtest",
        lambda **kwargs: {"MAPE": 1.0, "MAE": 2.0, "RMSE": 3.0},
    )
    monkeypatch.setattr(
        service,
        "build_future_forecast_df",
        lambda forecast, last_hist_date, capacity_phases: pd.DataFrame(
            {
                "ds": pd.to_datetime(["2024-04-01"]),
                "forecast_month": [1],
                "yhat_original": [130.0],
                "yhat_adjusted": [130.0],
                "adj_applied_any": [False],
                "adj_applied": [False],
            }
        ),
    )
    monkeypatch.setattr(
        service,
        "prophet_future_intervals",
        lambda *args, **kwargs: pd.DataFrame(
            {
                "ds": pd.to_datetime(["2024-04-01"]),
                "yhat_lower": [120.0],
                "yhat_upper": [140.0],
            }
        ),
    )
    monkeypatch.setattr(
        service,
        "collect_conformal_residuals",
        lambda **kwargs: pd.DataFrame({"residual": [1.0]}),
    )
    monkeypatch.setattr(
        service,
        "build_conformal_intervals",
        lambda future_df, residuals_df, target_coverage: (
            future_df.assign(
                yhat_lower_conformal=[125.0],
                yhat_upper_conformal=[135.0],
            ),
            pd.DataFrame(
                {
                    "horizon_step": [1],
                    "empirical_coverage": [0.89],
                    "target_coverage": [target_coverage],
                    "n_calibration_forecasts": [10],
                    "fallback_used": [False],
                }
            ),
            {
                "empirical_coverage_overall": 0.89,
                "empirical_coverage_by_horizon": {1: 0.89},
                "avg_interval_width": 10.0,
                "median_interval_width": 10.0,
                "interval_width_over_mean_actual": 0.08,
                "n_calibration_forecasts": 10,
                "fallback_used": False,
            },
        ),
    )
    monkeypatch.setattr(service, "prophet_full_forecast_df", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        service,
        "prophet_cross_validation_metrics",
        lambda *args, **kwargs: (
            pd.DataFrame(),
            pd.DataFrame(
                {
                    "horizon": pd.to_timedelta(["30 days"]),
                    "coverage": [0.75],
                }
            ),
        ),
    )

    result = service.forecast_visits(
        use_synthetic=False,
        df=pd.DataFrame(),
        forecast_periods=1,
        department="All",
        target_coverage=0.90,
        uncertainty_method="auto",
        tuning_mode="manual",
    )

    assert result.requested_uncertainty_method == "auto"
    assert result.uncertainty_method == "conformal"
    assert result.future_forecast_df.loc[0, "interval_source"] == "conformal"
    assert result.future_forecast_df.loc[0, "yhat_lower"] == 125.0
    assert result.tuning_note is not None
    assert "auto-selected conformal" in result.tuning_note.lower()
    assert result.tuning_diagnostics_df is not None
    assert (
        result.tuning_diagnostics_df["parameter_family"] == "uncertainty_method"
    ).any()


def test_forecast_visits_joint_auto_uses_joint_selector(monkeypatch):
    select_calls = []

    monkeypatch.setattr(service, "load_data", lambda *args, **kwargs: pd.DataFrame())
    monkeypatch.setattr(
        service,
        "prepare_dataframe",
        lambda **kwargs: (
            pd.DataFrame(
                {
                    "ds": pd.to_datetime(["2024-01-01", "2024-02-01"]),
                    "y": [100.0, 120.0],
                }
            ),
            "All",
        ),
    )
    monkeypatch.setattr(
        service,
        "fit_and_forecast",
        lambda **kwargs: ("series", "forecast", object()),
    )
    monkeypatch.setattr(
        service,
        "backtest",
        lambda **kwargs: {"MAPE": 1.0, "MAE": 2.0, "RMSE": 3.0},
    )
    monkeypatch.setattr(
        service,
        "build_future_forecast_df",
        lambda forecast, last_hist_date, capacity_phases: pd.DataFrame(
            {
                "ds": pd.to_datetime(["2024-03-01"]),
                "forecast_month": [1],
                "yhat_original": [130.0],
                "yhat_adjusted": [130.0],
                "adj_applied_any": [False],
                "adj_applied": [False],
            }
        ),
    )
    monkeypatch.setattr(
        service, "prophet_future_intervals", lambda *args, **kwargs: pd.DataFrame()
    )
    monkeypatch.setattr(
        service,
        "collect_conformal_residuals",
        lambda **kwargs: pd.DataFrame({"residual": [2.0]}),
    )
    monkeypatch.setattr(service, "prophet_full_forecast_df", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        service,
        "prophet_cross_validation_metrics",
        lambda *args, **kwargs: (None, None),
    )

    def fake_select_joint_forecast_configuration(**kwargs):
        select_calls.append(kwargs)
        return TuningSelection(
            changepoint_prior_scale=0.03,
            interval_width=0.90,
            primary_metric="Joint model score",
            primary_score=1.01,
            note="Joint auto-tune selected conformal.",
            diagnostics_df=pd.DataFrame(),
            uncertainty_method="conformal",
        )

    monkeypatch.setattr(
        service,
        "select_joint_forecast_configuration",
        fake_select_joint_forecast_configuration,
    )

    result = service.forecast_visits(
        use_synthetic=False,
        df=pd.DataFrame(),
        forecast_periods=1,
        department="All",
        target_coverage=0.92,
        uncertainty_method="auto",
        tuning_mode="auto",
    )

    assert select_calls
    assert select_calls[0]["target_coverage"] == 0.92
    assert result.uncertainty_method == "conformal"
    assert result.joint_auto_tuning_enabled is True
    assert result.selected_interval_width == 0.90
    assert result.target_coverage == 0.92


def test_forecast_visits_auto_auto_with_joint_toggle_off_uses_separate_passes(
    monkeypatch,
):
    cp_select_calls = []
    uncertainty_selection_calls = []

    monkeypatch.setattr(service, "load_data", lambda *args, **kwargs: pd.DataFrame())
    monkeypatch.setattr(
        service,
        "prepare_dataframe",
        lambda **kwargs: (
            pd.DataFrame(
                {
                    "ds": pd.to_datetime(["2024-01-01", "2024-02-01"]),
                    "y": [100.0, 120.0],
                }
            ),
            "All",
        ),
    )
    monkeypatch.setattr(
        service,
        "fit_and_forecast",
        lambda **kwargs: ("series", "forecast", object()),
    )
    monkeypatch.setattr(
        service,
        "backtest",
        lambda **kwargs: {"MAPE": 1.0, "MAE": 2.0, "RMSE": 3.0},
    )
    monkeypatch.setattr(
        service,
        "build_future_forecast_df",
        lambda forecast, last_hist_date, capacity_phases: pd.DataFrame(
            {
                "ds": pd.to_datetime(["2024-03-01"]),
                "forecast_month": [1],
                "yhat_original": [130.0],
                "yhat_adjusted": [130.0],
                "adj_applied_any": [False],
                "adj_applied": [False],
            }
        ),
    )
    monkeypatch.setattr(
        service,
        "prophet_future_intervals",
        lambda *args, **kwargs: pd.DataFrame(
            {
                "ds": pd.to_datetime(["2024-03-01"]),
                "yhat_lower": [120.0],
                "yhat_upper": [140.0],
            }
        ),
    )
    monkeypatch.setattr(
        service,
        "collect_conformal_residuals",
        lambda **kwargs: pd.DataFrame(
            {
                "cutoff": [pd.Timestamp("2024-02-01")],
                "ds": [pd.Timestamp("2024-03-01")],
                "horizon_step": [1],
                "actual": [120.0],
                "predicted": [118.0],
                "residual": [2.0],
                "abs_residual": [2.0],
            }
        ),
    )
    monkeypatch.setattr(
        service,
        "build_conformal_intervals",
        lambda future_df, residuals_df, target_coverage: (
            future_df.assign(yhat_lower_conformal=119.0, yhat_upper_conformal=141.0),
            pd.DataFrame(
                {
                    "horizon_step": [1],
                    "empirical_coverage": [0.91],
                    "target_coverage": [target_coverage],
                }
            ),
            {
                "empirical_coverage_overall": 0.91,
                "empirical_coverage_by_horizon": {1: 0.91},
                "avg_interval_width": 22.0,
                "median_interval_width": 22.0,
                "interval_width_over_mean_actual": 0.16,
                "n_calibration_forecasts": 1,
                "fallback_used": False,
            },
        ),
    )
    monkeypatch.setattr(service, "prophet_full_forecast_df", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        service,
        "prophet_cross_validation_metrics",
        lambda *args, **kwargs: (
            pd.DataFrame(),
            pd.DataFrame(
                {
                    "horizon": pd.to_timedelta(["30 days"]),
                    "coverage": [0.88],
                }
            ),
        ),
    )
    monkeypatch.setattr(
        service,
        "select_joint_forecast_configuration",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("joint selector should not run")),
    )

    def fake_select_prophet_hyperparameters(**kwargs):
        cp_select_calls.append(kwargs)
        return TuningSelection(
            changepoint_prior_scale=0.03,
            interval_width=0.92,
            primary_metric="MAPE+RMSE",
            primary_score=1.01,
            note="Interval width tuning skipped; using fixed interval / target coverage 0.92 from the current uncertainty settings.",
            diagnostics_df=pd.DataFrame(),
        )

    def fake_select_uncertainty_configuration(**kwargs):
        uncertainty_selection_calls.append(kwargs)
        return TuningSelection(
            changepoint_prior_scale=0.03,
            interval_width=0.80,
            primary_metric="coverage_gap",
            primary_score=0.0,
            note="Separate uncertainty tuning selected conformal with target coverage=0.80.",
            diagnostics_df=pd.DataFrame(
                {
                    "parameter_family": ["uncertainty_configuration"],
                    "candidate_value": ["conformal|coverage=0.80"],
                    "selected": [True],
                }
            ),
            uncertainty_method="conformal",
        )

    monkeypatch.setattr(service, "select_prophet_hyperparameters", fake_select_prophet_hyperparameters)
    monkeypatch.setattr(
        service,
        "select_uncertainty_configuration",
        fake_select_uncertainty_configuration,
    )

    result = service.forecast_visits(
        use_synthetic=False,
        df=pd.DataFrame(),
        forecast_periods=1,
        department="All",
        target_coverage=0.92,
        uncertainty_method="auto",
        tuning_mode="auto",
        joint_auto_tuning_enabled=False,
    )

    assert cp_select_calls
    assert cp_select_calls[0]["tune_interval_width"] is False
    assert cp_select_calls[0]["fixed_interval_width"] == 0.92
    assert uncertainty_selection_calls
    assert result.uncertainty_method == "conformal"
    assert result.joint_auto_tuning_enabled is False
    assert result.selected_interval_width == 0.80
    assert result.target_coverage == 0.80
    assert result.tuning_note is not None
    assert "separate passes" in result.tuning_note.lower()
