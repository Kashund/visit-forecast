import pandas as pd

from visit_forecast import service
from visit_forecast.tuning import TuningSelection, select_prophet_hyperparameters


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
