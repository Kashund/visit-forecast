from visit_forecast import forecast_visits

def test_smoke_synthetic():
    result = forecast_visits(
        use_synthetic=True,
        timeframe_start="2020-04-01",
        timeframe_end="2021-03-01",
        forecast_periods=3,
        department="All",
    )
    assert result.fiscal_summary is not None
    assert "MAPE" in result.performance_metrics
