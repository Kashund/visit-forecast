import pandas as pd

from visit_forecast.fiscal import (
    aggregate_summary_by_period,
    fiscal_quarter_summary,
    fiscal_summary,
)


def test_fiscal_summary_matches_yearly_aggregation():
    historical = pd.DataFrame(
        {
            "ds": ["2024-04-01", "2025-01-01"],
            "y": [10, 5],
        }
    )
    forecast = pd.DataFrame(
        {
            "ds": ["2024-04-01", "2025-01-01"],
            "yhat_original": [7, 3],
            "yhat_adjusted": [8, 2],
        }
    )

    expected = aggregate_summary_by_period(historical, forecast, period="year")
    result = fiscal_summary(historical, forecast)

    pd.testing.assert_frame_equal(result, expected)


def test_fiscal_quarter_summary_uses_april_start_calendar():
    historical = pd.DataFrame(
        {
            "ds": ["2024-04-01", "2024-07-01", "2025-01-01"],
            "y": [10, 20, 30],
        }
    )
    forecast = pd.DataFrame(
        {
            "ds": ["2024-04-01", "2024-07-01", "2025-01-01"],
            "yhat_original": [1, 2, 3],
            "yhat_adjusted": [4, 5, 6],
        }
    )

    result = fiscal_quarter_summary(historical, forecast)

    assert result["Fiscal_Period_Label"].tolist() == [
        "FY2025 Q1",
        "FY2025 Q2",
        "FY2025 Q4",
    ]
    assert result["Fiscal_Quarter"].tolist() == [1, 2, 4]
    assert result["Historical_Visits"].tolist() == [10, 20, 30]
    assert result["Forecast_Original"].tolist() == [1, 2, 3]
    assert result["Forecast_Adjusted"].tolist() == [4, 5, 6]
