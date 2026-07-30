from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from visit_forecast import service
from visit_forecast.io import prepare_dataframe

SAMPLE_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "sample"


@pytest.mark.parametrize(
    "sample_filename",
    ["synthetic_visits_small.csv", "synthetic_visits.csv"],
)
def test_bundled_sample_data_forecasts_end_to_end(
    sample_filename: str,
) -> None:
    raw = pd.read_csv(SAMPLE_DATA_DIR / sample_filename)
    dates = pd.to_datetime(raw["Date"])

    result = service.forecast_visits(
        df=raw,
        timeframe_start=str(dates.min().date()),
        timeframe_end=str(dates.max().date()),
        forecast_periods=3,
        department="All",
        model_choice="seasonal_naive",
        uncertainty_method="conformal",
    )

    future = result.future_forecast_df
    assert result.selected_model == "seasonal_naive"
    assert len(future) == 3
    assert future["ds"].tolist() == list(
        pd.date_range(
            dates.max().to_period("M").to_timestamp() + pd.offsets.MonthBegin(1),
            periods=3,
            freq="MS",
        )
    )
    assert np.isfinite(
        future[
            ["yhat_original", "yhat_adjusted", "yhat_lower", "yhat_upper"]
        ].to_numpy()
    ).all()
    assert future[["yhat_original", "yhat_adjusted"]].ge(0).all().all()
    assert not result.fiscal_summary.empty


def test_prepare_dataframe_aggregates_duplicate_department_months() -> None:
    raw = pd.DataFrame(
        {
            "Date": ["2024-01-03", "2024-01-20", "2024-02-01"],
            "Visits": [10, 15, 20],
            "Department": ["Cardiology", "Cardiology", "Cardiology"],
        }
    )

    prepared, department_info = prepare_dataframe(
        raw,
        timeframe_start="2024-01-01",
        timeframe_end="2024-02-29",
        department="Cardiology",
        exclude_departments=None,
    )

    assert department_info == "Cardiology"
    assert prepared["ds"].tolist() == [
        pd.Timestamp("2024-01-01"),
        pd.Timestamp("2024-02-01"),
    ]
    assert prepared["y"].tolist() == [25, 20]


def test_prepare_dataframe_reports_missing_months_before_model_construction() -> None:
    raw = pd.DataFrame(
        {
            "Date": ["2024-01-01", "2024-03-01"],
            "Visits": [10, 20],
            "Department": ["Cardiology", "Cardiology"],
        }
    )

    with pytest.raises(ValueError, match="2024-02"):
        prepare_dataframe(
            raw,
            timeframe_start="2024-01-01",
            timeframe_end="2024-03-01",
            department="Cardiology",
            exclude_departments=None,
        )


def test_prepare_dataframe_rejects_negative_visits() -> None:
    raw = pd.DataFrame(
        {
            "Date": ["2024-01-01", "2024-02-01"],
            "Visits": [10, -1],
            "Department": ["Cardiology", "Cardiology"],
        }
    )

    with pytest.raises(ValueError, match="non-negative"):
        prepare_dataframe(
            raw,
            timeframe_start="2024-01-01",
            timeframe_end="2024-02-01",
            department="Cardiology",
            exclude_departments=None,
        )


def test_department_failure_produces_explicit_partial_total(
    monkeypatch,
) -> None:
    raw = pd.DataFrame(
        {
            "Date": ["2024-01-01", "2024-01-01"],
            "Visits": [10, 20],
            "Department": ["Valid", "Broken"],
        }
    )

    def fake_forecast_visits(**kwargs):
        if kwargs["department"] == "Broken":
            raise RuntimeError("intentional department failure")
        return SimpleNamespace(
            future_forecast_df=pd.DataFrame(
                {
                    "ds": pd.date_range("2024-02-01", periods=2, freq="MS"),
                    "forecast_month": [1, 2],
                    "yhat_original": [10.0, 11.0],
                    "yhat_adjusted": [10.0, 11.0],
                }
            ),
            performance_metrics={"MAPE": 1.0, "MAE": 1.0, "RMSE": 1.0},
            selected_model="seasonal_naive",
            tuning_note=None,
        )

    monkeypatch.setattr(service, "forecast_visits", fake_forecast_visits)

    bundle = service.forecast_departments(
        df=raw,
        timeframe_start="2024-01-01",
        timeframe_end="2024-01-01",
        forecast_periods=2,
        model_choice="seasonal_naive",
    )

    total_rows = bundle.combined_future_df[
        bundle.combined_future_df["Department"].str.startswith(
            "Partial total"
        )
    ]
    assert not total_rows.empty
    assert total_rows["Department"].iloc[0] == "Partial total (1/2 departments)"
    assert not bool(total_rows["aggregate_complete"].iloc[0])
    assert total_rows["included_department_count"].iloc[0] == 1
    assert total_rows["requested_department_count"].iloc[0] == 2
    assert bundle.failures == {"Broken": "intentional department failure"}
