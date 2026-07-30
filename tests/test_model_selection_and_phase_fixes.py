import pandas as pd
import pytest

from visit_forecast.io import generate_synthetic_data, prepare_dataframe
from visit_forecast.models import (
    SELECTABLE_MODELS,
    build_model,
    fit_and_forecast_model,
    select_forecast_model,
)
from visit_forecast.service import (
    CapacityPhase,
    build_future_forecast_df,
    forecast_visits,
)


class MockForecastSeries:
    def __init__(self, dataframe: pd.DataFrame) -> None:
        self._dataframe = dataframe

    def pd_dataframe(self) -> pd.DataFrame:
        return self._dataframe


def _build_future_frame(capacity_phases, periods: int = 6):
    forecast_dates = pd.date_range("2021-01-01", periods=periods, freq="MS")
    values = pd.DataFrame({"ds": forecast_dates, "y": [100.0] * periods})
    forecast = MockForecastSeries(values)
    return build_future_forecast_df(
        forecast,
        last_hist_date=pd.Timestamp("2020-12-01"),
        capacity_phases=capacity_phases,
    )


def _prepared_df(months: int = 48) -> pd.DataFrame:
    dates = pd.date_range("2020-04-01", periods=months, freq="MS")
    raw = generate_synthetic_data(str(dates.min().date()), str(dates.max().date()))
    prepared, _ = prepare_dataframe(
        raw,
        timeframe_start=str(dates.min().date()),
        timeframe_end=str(dates.max().date()),
        department="All",
        exclude_departments=None,
    )
    return prepared


# ---------------------------------------------------------------------------
# Capacity phase fixes
# ---------------------------------------------------------------------------

def test_disabled_out_of_horizon_phase_does_not_error() -> None:
    result = _build_future_frame(
        [
            CapacityPhase(True, "add", 10, 1, 3),
            CapacityPhase(False, "loss", 15, 8, 12),  # beyond 6-month horizon
        ]
    )
    assert result.loc[0, "yhat_adjusted"] == pytest.approx(110.0)
    assert (result["applied_phase_ids"].isin(["", "1"])).all()


def test_enabled_phase_partially_beyond_horizon_applies_within_horizon() -> None:
    result = _build_future_frame(
        [CapacityPhase(True, "loss", 20, 5, 12)]  # horizon is 6
    )
    assert result.loc[3, "yhat_adjusted"] == pytest.approx(100.0)
    assert result.loc[4, "yhat_adjusted"] == pytest.approx(80.0)
    assert result.loc[5, "yhat_adjusted"] == pytest.approx(80.0)


def test_enabled_phase_fully_beyond_horizon_is_a_no_op() -> None:
    result = _build_future_frame([CapacityPhase(True, "loss", 20, 8, 12)])
    assert (result["yhat_adjusted"] == 100.0).all()
    assert not result["adj_applied_any"].any()


def test_zero_percent_phase_does_not_flag_months_adjusted() -> None:
    result = _build_future_frame(
        [CapacityPhase(True, "add", 0, 1, 6)]
    )
    assert not result["adj_applied_any"].any()
    assert (result["applied_phase_ids"] == "").all()


def test_forecast_visits_without_adjustment_reports_no_adjustment() -> None:
    result = forecast_visits(
        use_synthetic=True,
        timeframe_start="2020-04-01",
        timeframe_end="2021-03-01",
        forecast_periods=3,
        department="All",
    )
    frame = result.future_forecast_df
    assert not frame["adj_applied_any"].any()
    assert (frame["applied_phase_ids"] == "").all()


def test_structural_phase_validation_still_raises() -> None:
    with pytest.raises(ValueError, match="start_month"):
        _build_future_frame([CapacityPhase(True, "add", 5, 0, 2)])
    with pytest.raises(ValueError, match="end_month"):
        _build_future_frame([CapacityPhase(True, "add", 5, 3, 2)])
    with pytest.raises(ValueError, match="percent"):
        _build_future_frame([CapacityPhase(True, "add", -1, 1, 2)])
    with pytest.raises(ValueError, match="mode"):
        _build_future_frame([CapacityPhase(True, "multiply", 5, 1, 2)])


# ---------------------------------------------------------------------------
# Model layer
# ---------------------------------------------------------------------------

def test_build_model_supports_all_candidates() -> None:
    for name in SELECTABLE_MODELS:
        model = build_model(name, n_observations=48)
        assert model is not None


def test_build_model_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="Unknown model"):
        build_model("lstm", n_observations=48)


def test_fit_and_forecast_classical_models() -> None:
    prepared = _prepared_df(48)
    for name in ["ets", "theta", "arima", "seasonal_naive"]:
        series, forecast, _ = fit_and_forecast_model(
            prepared, forecast_periods=6, model_name=name
        )
        assert len(forecast) == 6


def test_select_forecast_model_scores_all_candidates() -> None:
    prepared = _prepared_df(48)
    selection = select_forecast_model(
        prepared,
        forecast_periods=6,
        candidates=["ets", "theta", "seasonal_naive"],
    )
    assert selection.model_name in {"ets", "theta", "seasonal_naive"}
    assert selection.diagnostics_df.shape[0] == 3
    assert selection.diagnostics_df["selected"].sum() == 1
    assert selection.note


def test_forecast_visits_with_classical_model_uses_conformal_intervals() -> None:
    result = forecast_visits(
        use_synthetic=True,
        timeframe_start="2020-04-01",
        timeframe_end="2023-03-01",
        forecast_periods=4,
        department="All",
        model_choice="ets",
        uncertainty_method="prophet",  # should fall back to conformal
    )
    assert result.selected_model == "ets"
    assert result.uncertainty_method == "conformal"
    assert result.prophet_forecast_df is None
    assert result.future_forecast_df["yhat_adjusted"].notna().all()


def test_forecast_visits_rejects_unknown_model() -> None:
    with pytest.raises(ValueError, match="model_choice"):
        forecast_visits(
            use_synthetic=True,
            timeframe_start="2020-04-01",
            timeframe_end="2021-03-01",
            forecast_periods=3,
            model_choice="transformer",
        )
