import pytest

from visit_forecast.service import (
    CapacityPhase,
    _normalize_capacity_phases,
    forecast_visits,
)


def test_normalize_capacity_phases_rejects_more_than_four() -> None:
    phases = [
        CapacityPhase(enabled=True, mode="loss", percent=1, start_month=1, end_month=1)
        for _ in range(5)
    ]

    with pytest.raises(ValueError, match="maximum of 4"):
        _normalize_capacity_phases(phases, forecast_periods=12)


def test_normalize_capacity_phases_rejects_unknown_mode() -> None:
    phases = [
        CapacityPhase(
            enabled=True,
            mode="multiply",
            percent=5,
            start_month=1,
            end_month=3,
        )
    ]

    with pytest.raises(ValueError, match="mode"):
        _normalize_capacity_phases(phases, forecast_periods=12)


def test_normalize_capacity_phases_rejects_invalid_months_and_percent() -> None:
    with pytest.raises(ValueError, match="start_month"):
        _normalize_capacity_phases(
            [
                CapacityPhase(
                    enabled=True,
                    mode="loss",
                    percent=5,
                    start_month=0,
                    end_month=2,
                )
            ],
            forecast_periods=12,
        )

    with pytest.raises(ValueError, match="end_month must be >= start_month"):
        _normalize_capacity_phases(
            [
                CapacityPhase(
                    enabled=True,
                    mode="add",
                    percent=5,
                    start_month=3,
                    end_month=2,
                )
            ],
            forecast_periods=12,
        )

    with pytest.raises(ValueError, match="percent must be >= 0"):
        _normalize_capacity_phases(
            [
                CapacityPhase(
                    enabled=True,
                    mode="add",
                    percent=-1,
                    start_month=1,
                    end_month=2,
                )
            ],
            forecast_periods=12,
        )


def test_forecast_visits_accepts_capacity_phases() -> None:
    result = forecast_visits(
        use_synthetic=True,
        timeframe_start="2020-04-01",
        timeframe_end="2021-03-01",
        forecast_periods=3,
        department="All",
        adjustment_percent=50,
        capacity_phases=[
            {
                "enabled": True,
                "mode": "loss",
                "percent": 10,
                "start_month": 1,
                "end_month": 2,
            },
            {
                "enabled": True,
                "mode": "add",
                "percent": 10,
                "start_month": 2,
                "end_month": 3,
            },
        ],
    )

    forecast_frame = result.future_forecast_df.sort_values(
        "forecast_month"
    ).reset_index(drop=True)
    assert forecast_frame.loc[0, "adj_applied"]
    assert forecast_frame.loc[1, "adj_applied"]
    assert forecast_frame.loc[2, "adj_applied"]
