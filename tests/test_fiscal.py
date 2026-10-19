import pandas as pd

from visit_forecast.fiscal import (
    aggregate_summary_by_period,
    append_fiscal_year_totals,
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
        "FY2025 Q3",
        "FY2025 Q4",
    ]
    assert result["Fiscal_Quarter"].tolist() == [1, 2, 3, 4]
    assert result.loc[[0, 1, 3], "Historical_Visits"].tolist() == [10, 20, 30]
    assert result.loc[[0, 1, 3], "Forecast_Original"].tolist() == [1, 2, 3]
    assert result.loc[[0, 1, 3], "Forecast_Adjusted"].tolist() == [4, 5, 6]
    assert (
        result.loc[2, ["Historical_Visits", "Forecast_Original", "Forecast_Adjusted"]]
        .isna()
        .all()
    )
    assert result.loc[2, "Covered_Months"] == 0


def test_append_fiscal_year_totals_adds_total_row_after_each_year():
    quarter_summary = pd.DataFrame(
        {
            "Fiscal_Year": [2025, 2025, 2026],
            "Fiscal_Quarter": [1, 2, 1],
            "Fiscal_Period_Label": ["FY2025 Q1", "FY2025 Q2", "FY2026 Q1"],
            "Historical_Visits": [10, 20, 30],
            "Forecast_Original": [1, 2, 3],
            "Forecast_Adjusted": [4, 5, 6],
        }
    )

    result = append_fiscal_year_totals(quarter_summary)

    assert result["Fiscal_Period_Label"].tolist() == [
        "FY2025 Q1",
        "FY2025 Q2",
        "FY2025 Total",
        "FY2026 Q1",
        "FY2026 Total",
    ]
    assert result["Fiscal_Quarter"].astype("object").tolist() == [1, 2, pd.NA, 1, pd.NA]
    assert result["Historical_Visits"].tolist() == [10, 20, 30, 30, 30]
    assert result["Forecast_Original"].tolist() == [1, 2, 3, 3, 3]
    assert result["Forecast_Adjusted"].tolist() == [4, 5, 9, 6, 6]


def test_fiscal_views_reconcile_across_actual_forecast_boundary():
    historical = pd.DataFrame(
        {"ds": pd.date_range("2024-03-01", periods=11, freq="MS"), "y": [10.25] * 11}
    )
    forecast = pd.DataFrame(
        {
            "ds": pd.date_range("2025-02-01", periods=3, freq="MS"),
            "yhat_original": [20.5] * 3,
            "yhat_adjusted": [30.75] * 3,
        }
    )
    views = {
        p: aggregate_summary_by_period(historical, forecast, p)
        for p in ["month", "quarter", "year"]
    }
    for column in ["Historical_Visits", "Forecast_Original", "Forecast_Adjusted"]:
        assert (
            views["month"][column].sum()
            == views["quarter"][column].sum()
            == views["year"][column].sum()
        )
    monthly = views["month"]
    assert monthly.iloc[0].Fiscal_Period_Label == "FY2024 M12 (Mar 2024)"
    assert monthly.iloc[1].Fiscal_Period_Label == "FY2025 M01 (Apr 2024)"
    assert monthly.iloc[10].Fiscal_Period_Label == "FY2025 M10 (Jan 2025)"
    q4 = views["quarter"].query("Fiscal_Year == 2025 and Fiscal_Quarter == 4").iloc[0]
    assert q4.Historical_Visits == 10.25
    assert q4.Forecast_Adjusted == 61.5
    assert q4.Coverage_Status == "Complete"
    assert views["year"].Coverage_Status.tolist() == ["Partial", "Complete", "Partial"]


def test_fiscal_missing_month_is_not_real_zero():
    hist = pd.DataFrame({"ds": ["2024-04-01", "2024-06-01"], "y": [0.0, 5.0]})
    future = pd.DataFrame(columns=["ds", "yhat_original", "yhat_adjusted"])
    out = aggregate_summary_by_period(hist, future, "month")
    assert out.Historical_Visits.iloc[0] == 0
    assert pd.isna(out.Historical_Visits.iloc[1])
    assert out.Coverage_Status.tolist() == ["Complete", "Partial", "Complete"]
