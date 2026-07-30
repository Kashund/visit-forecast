import json
import sys
from pathlib import Path

import pandas as pd
import pytest

from visit_forecast.io import generate_synthetic_data, prepare_dataframe
from visit_forecast.model_prophet import prophet_future_intervals
from visit_forecast.models import (
    build_calendar_covariate_series,
    fit_and_forecast_model,
)
from visit_forecast.service import forecast_visits

STREAMLIT_APP_DIR = Path(__file__).resolve().parents[1] / "apps" / "streamlit"
if str(STREAMLIT_APP_DIR) not in sys.path:
    sys.path.insert(0, str(STREAMLIT_APP_DIR))

import state  # noqa: E402


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
# Calendar covariates
# ---------------------------------------------------------------------------

def test_calendar_covariates_cover_history_and_horizon() -> None:
    prepared = _prepared_df(36)
    covariates = build_calendar_covariate_series(prepared, forecast_periods=6)
    assert len(covariates) == 36 + 6
    values = covariates.values().flatten()
    assert values.min() >= 19 and values.max() <= 23  # business days per month


def test_prophet_fit_with_covariates() -> None:
    prepared = _prepared_df(48)
    covariates = build_calendar_covariate_series(prepared, forecast_periods=6)
    _, forecast, _ = fit_and_forecast_model(
        prepared, 6, "prophet", future_covariates=covariates
    )
    assert len(forecast) == 6


def test_prophet_future_intervals_include_required_covariates() -> None:
    captured: dict = {}

    class FakeNativeProphet:
        extra_regressors = {"business_days": {}}

        def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
            captured["frame"] = frame.copy()
            return frame.assign(yhat_lower=10.0, yhat_upper=20.0)

    class FakeDartsProphet:
        model = FakeNativeProphet()

    covariates = pd.DataFrame(
        {
            "ds": pd.date_range("2024-01-01", periods=3, freq="MS"),
            "business_days": [23, 21, 21],
        }
    )
    intervals = prophet_future_intervals(
        FakeDartsProphet(),
        last_hist_date=pd.Timestamp("2023-12-01"),
        forecast_periods=3,
        covariates_df=covariates,
    )

    assert captured["frame"]["business_days"].tolist() == [23, 21, 21]
    assert intervals["yhat_lower"].tolist() == [10.0, 10.0, 10.0]


def test_prophet_future_intervals_reject_missing_covariates() -> None:
    class FakeNativeProphet:
        extra_regressors = {"business_days": {}}

        def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
            raise AssertionError("predict should not run without required covariates")

    class FakeDartsProphet:
        model = FakeNativeProphet()

    with pytest.raises(ValueError, match="missing required future covariates"):
        prophet_future_intervals(
            FakeDartsProphet(),
            last_hist_date=pd.Timestamp("2023-12-01"),
            forecast_periods=3,
        )


def test_covariate_incapable_model_ignores_covariates() -> None:
    prepared = _prepared_df(48)
    covariates = build_calendar_covariate_series(prepared, forecast_periods=6)
    _, forecast, _ = fit_and_forecast_model(
        prepared, 6, "ets", future_covariates=covariates
    )
    assert len(forecast) == 6


def test_forecast_visits_with_covariates_end_to_end() -> None:
    result = forecast_visits(
        use_synthetic=True,
        timeframe_start="2020-04-01",
        timeframe_end="2023-03-01",
        forecast_periods=4,
        department="All",
        model_choice="arima",
        use_calendar_covariates=True,
    )
    assert result.used_calendar_covariates is True
    assert "Calendar covariates" in (result.tuning_note or "")
    assert result.future_forecast_df["yhat_adjusted"].notna().all()


def test_prophet_native_intervals_with_covariates_end_to_end() -> None:
    result = forecast_visits(
        use_synthetic=True,
        timeframe_start="2020-04-01",
        timeframe_end="2023-03-01",
        forecast_periods=4,
        department="All",
        model_choice="prophet",
        use_calendar_covariates=True,
        uncertainty_method="prophet",
    )

    assert result.used_calendar_covariates is True
    assert result.uncertainty_method == "prophet"
    assert result.future_forecast_df["yhat_lower"].notna().all()
    assert result.future_forecast_df["yhat_upper"].notna().all()


def test_forecast_visits_covariates_note_for_unsupported_model() -> None:
    result = forecast_visits(
        use_synthetic=True,
        timeframe_start="2020-04-01",
        timeframe_end="2023-03-01",
        forecast_periods=4,
        department="All",
        model_choice="theta",
        use_calendar_covariates=True,
    )
    assert result.used_calendar_covariates is False
    assert "ignored" in (result.tuning_note or "")


# ---------------------------------------------------------------------------
# Persistent run history
# ---------------------------------------------------------------------------

def test_history_round_trip(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("VISIT_FORECAST_HISTORY_DIR", str(tmp_path))
    monkeypatch.setenv("VISIT_FORECAST_PERSIST_HISTORY", "1")
    entries = [{"run_time": "2026-07-07 09:00:00", "MAPE": 3.2, "selected_model": "ets"}]
    state.save_persisted_history(entries)
    assert state.history_file_path().exists()
    loaded = state.load_persisted_history()
    assert loaded == entries

    session: dict = {}
    state.ensure_history_loaded(session)
    assert session["forecast_history"] == entries

    state.clear_persisted_history()
    assert not state.history_file_path().exists()
    assert state.load_persisted_history() == []


def test_history_caps_at_limit(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("VISIT_FORECAST_HISTORY_DIR", str(tmp_path))
    monkeypatch.setenv("VISIT_FORECAST_PERSIST_HISTORY", "1")
    entries = [{"run_time": f"run-{i}"} for i in range(30)]
    state.save_persisted_history(entries)
    assert len(state.load_persisted_history()) == state.HISTORY_LIMIT


def test_history_survives_corrupt_file(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("VISIT_FORECAST_HISTORY_DIR", str(tmp_path))
    monkeypatch.setenv("VISIT_FORECAST_PERSIST_HISTORY", "1")
    path = state.history_file_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not valid json", encoding="utf-8")
    assert state.load_persisted_history() == []


def test_history_entries_are_json_serializable_from_run(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("VISIT_FORECAST_HISTORY_DIR", str(tmp_path))
    monkeypatch.setenv("VISIT_FORECAST_PERSIST_HISTORY", "1")
    result = forecast_visits(
        use_synthetic=True,
        timeframe_start="2020-04-01",
        timeframe_end="2022-03-01",
        forecast_periods=3,
        department="All",
        model_choice="seasonal_naive",
    )
    common = {
        "forecast_periods": 3,
        "capacity_phases": [
            {"enabled": True, "mode": "add", "percent": 5, "start_month": 1, "end_month": 2}
        ],
    }
    entry = state.build_forecast_history_entry(
        common=common,
        result=result,
        source_mode="synthetic",
        scenario_summary="test",
    )
    state.save_persisted_history([entry])
    reloaded = state.load_persisted_history()
    assert reloaded[0]["selected_model"] == "seasonal_naive"
    assert json.dumps(reloaded)  # fully serializable


def test_history_persistence_is_disabled_by_default(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("VISIT_FORECAST_HISTORY_DIR", str(tmp_path))
    monkeypatch.delenv("VISIT_FORECAST_PERSIST_HISTORY", raising=False)

    state.save_persisted_history([{"run_time": "private"}])

    assert state.history_persistence_enabled() is False
    assert state.load_persisted_history() == []
    assert not state.history_file_path().exists()
