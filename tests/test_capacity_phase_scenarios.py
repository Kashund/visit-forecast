import pandas as pd
import pytest

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


def _build_future_frame(capacity_phases):
    forecast_dates = pd.date_range("2021-01-01", periods=6, freq="MS")
    values = pd.DataFrame({"ds": forecast_dates, "y": [100.0] * 6})
    forecast = MockForecastSeries(values)
    return build_future_forecast_df(
        forecast,
        last_hist_date=pd.Timestamp("2020-12-01"),
        capacity_phases=capacity_phases,
    )


def test_single_phase_backwards_compatibility_unchanged() -> None:
    legacy_result = forecast_visits(
        use_synthetic=True,
        timeframe_start="2020-04-01",
        timeframe_end="2021-03-01",
        forecast_periods=4,
        department="All",
        adjustment_mode="loss",
        adjustment_percent=10,
        adjustment_start_month=2,
        adjustment_end_month=3,
    )
    phase_result = forecast_visits(
        use_synthetic=True,
        timeframe_start="2020-04-01",
        timeframe_end="2021-03-01",
        forecast_periods=4,
        department="All",
        capacity_phases=[
            {
                "enabled": True,
                "mode": "loss",
                "percent": 10,
                "start_month": 2,
                "end_month": 3,
            }
        ],
    )

    legacy_frame = legacy_result.future_forecast_df[
        ["ds", "yhat_adjusted", "applied_phase_ids"]
    ]
    phase_frame = phase_result.future_forecast_df[
        ["ds", "yhat_adjusted", "applied_phase_ids"]
    ]
    pd.testing.assert_frame_equal(
        legacy_frame.reset_index(drop=True),
        phase_frame.reset_index(drop=True),
        check_exact=False,
        atol=1e-9,
        rtol=1e-9,
    )


def test_multiple_disjoint_phases_apply_correctly() -> None:
    result = _build_future_frame(
        [
            CapacityPhase(True, "add", 10, 1, 2),
            CapacityPhase(True, "loss", 20, 5, 6),
        ]
    )

    assert result.loc[0, "yhat_adjusted"] == pytest.approx(110.0)
    assert result.loc[1, "yhat_adjusted"] == pytest.approx(110.0)
    assert result.loc[2, "yhat_adjusted"] == pytest.approx(100.0)
    assert result.loc[4, "yhat_adjusted"] == pytest.approx(80.0)
    assert result.loc[5, "yhat_adjusted"] == pytest.approx(80.0)


def test_overlapping_phases_follow_precedence_order() -> None:
    result = _build_future_frame(
        [
            CapacityPhase(True, "loss", 10, 2, 4),
            CapacityPhase(True, "add", 20, 3, 5),
        ]
    )

    assert result.loc[1, "applied_phase_ids"] == "1"
    assert result.loc[2, "applied_phase_ids"] == "1,2"
    assert result.loc[3, "applied_phase_ids"] == "1,2"
    assert result.loc[4, "applied_phase_ids"] == "2"
    assert result.loc[2, "yhat_adjusted"] == pytest.approx(108.0)


def test_disabled_phases_do_not_affect_output() -> None:
    result = _build_future_frame(
        [
            CapacityPhase(False, "loss", 50, 1, 6),
            CapacityPhase(True, "add", 5, 2, 2),
        ]
    )

    assert result.loc[0, "yhat_adjusted"] == pytest.approx(100.0)
    assert result.loc[1, "yhat_adjusted"] == pytest.approx(105.0)
    assert result.loc[0, "applied_phase_ids"] == ""
    assert result.loc[1, "applied_phase_ids"] == "2"


def test_max_phase_and_input_validation_errors() -> None:
    too_many_phases = [
        CapacityPhase(True, "loss", 1, 1, 1),
        CapacityPhase(True, "loss", 2, 1, 1),
        CapacityPhase(True, "loss", 3, 1, 1),
        CapacityPhase(True, "loss", 4, 1, 1),
        CapacityPhase(True, "loss", 5, 1, 1),
    ]

    with pytest.raises(ValueError, match="maximum of 4"):
        _build_future_frame(too_many_phases)

    with pytest.raises(ValueError, match="start_month"):
        _build_future_frame([CapacityPhase(True, "add", 5, 0, 2)])

    with pytest.raises(ValueError, match="end_month"):
        _build_future_frame([CapacityPhase(True, "add", 5, 3, 2)])

    with pytest.raises(ValueError, match="percent"):
        _build_future_frame([CapacityPhase(True, "add", -1, 1, 2)])
