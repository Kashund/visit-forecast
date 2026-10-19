import pandas as pd
import pytest

from visit_forecast.io import (
    generate_synthetic_data,
    inspect_data,
    load_data,
    prepare_dataframe,
)
from visit_forecast.service import forecast_departments


def test_csv_filename_loader_and_invalid_records():
    assert not load_data(False, "data/sample/synthetic_visits.csv", "", "").empty
    with pytest.raises(ValueError, match="Unsupported file format"):
        load_data(False, "legacy.xls", "", "")
    raw = pd.DataFrame(
        {
            "Date": ["2024-01-01"] * 3,
            "Visits": [10, float("inf"), "bad"],
            "Department": ["A"] * 3,
        }
    )
    report = inspect_data(raw)
    assert len(report.issues) == 2
    with pytest.raises(ValueError, match="finite"):
        prepare_dataframe(raw, "2024-01-01", "2024-01-01", "All", None)


def test_aggregate_does_not_hide_missing_department_month():
    raw = pd.DataFrame(
        {
            "Date": [
                "2024-01-01",
                "2024-02-01",
                "2024-03-01",
                "2024-01-01",
                "2024-03-01",
            ],
            "Visits": [10] * 5,
            "Department": ["A"] * 3 + ["B"] * 2,
        }
    )
    with pytest.raises(ValueError, match="missing department observations"):
        prepare_dataframe(raw, "2024-01-01", "2024-03-01", "All", None)


def test_department_endpoints_are_aligned_and_incomplete_total_is_partial():
    raw = generate_synthetic_data("2020-01-01", "2023-12-01", departments=["A", "B"])
    raw = raw[~(raw.Department.eq("B") & raw.Date.ge("2023-11-01"))]
    bundle = forecast_departments(
        df=raw,
        timeframe_start="2020-01-01",
        timeframe_end="2023-12-01",
        forecast_periods=2,
        model_choice="seasonal_naive",
    )
    assert "B" in bundle.failures
    totals = bundle.combined_future_df[
        bundle.combined_future_df.Department.str.startswith("Partial total")
    ]
    assert totals.ds.tolist() == list(pd.date_range("2024-01-01", periods=2, freq="MS"))
    assert not totals.aggregate_complete.any()
    assert totals.included_department_count.eq(1).all()
    with pytest.raises(ValueError, match="Unknown"):
        forecast_departments(df=raw, departments=["A", "Unknown"])
