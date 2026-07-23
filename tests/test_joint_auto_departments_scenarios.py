import numpy as np
import pandas as pd
import pytest

from visit_forecast import service
from visit_forecast.service import (
    apply_capacity_phases_to_future_df,
    forecast_departments,
    forecast_visits,
)
from visit_forecast.tuning import (
    DEFAULT_CHANGEPOINT_FALLBACK,
    DEFAULT_INTERVAL_WIDTH_FALLBACK,
    TuningSelection,
    select_joint_forecast_configuration,
)


class DummyModel:
    def __init__(self, changepoint_prior_scale: float, interval_width: float) -> None:
        self.changepoint_prior_scale = changepoint_prior_scale
        self.interval_width = interval_width


def _patch_joint_dependencies(
    monkeypatch,
    *,
    metrics_by_cp,
    prophet_coverage=0.90,
    conformal_coverage=0.90,
    fail_all=False,
):
    def fake_fit_and_forecast(
        df_prepared, forecast_periods, changepoint_prior_scale, interval_width
    ):
        if fail_all:
            raise RuntimeError("fit failed")
        return object(), object(), DummyModel(changepoint_prior_scale, interval_width)

    def fake_backtest(model, series, forecast_horizon, start, stride):
        return metrics_by_cp[round(model.changepoint_prior_scale, 2)]

    def fake_prophet_cv_metrics(model, initial, period, horizon):
        return (
            pd.DataFrame(),
            pd.DataFrame(
                {
                    "horizon": pd.to_timedelta(["30 days", "60 days"]),
                    "coverage": [prophet_coverage, prophet_coverage],
                }
            ),
        )

    def fake_conformal_summary(
        model, series, forecast, forecast_periods, target_coverage
    ):
        return {
            "empirical_coverage_overall": conformal_coverage,
            "coverage_gap": abs(conformal_coverage - target_coverage),
            "interval_width_over_mean_actual": 0.3,
            "avg_interval_width": 50.0,
            "fallback_used": False,
        }

    monkeypatch.setattr(
        "visit_forecast.tuning.fit_and_forecast", fake_fit_and_forecast
    )
    monkeypatch.setattr("visit_forecast.tuning.backtest", fake_backtest)
    monkeypatch.setattr(
        "visit_forecast.tuning.prophet_cross_validation_metrics",
        fake_prophet_cv_metrics,
    )
    monkeypatch.setattr(
        "visit_forecast.tuning._summarize_conformal_interval_validation",
        fake_conformal_summary,
    )


# ---------------------------------------------------------------------------
# select_joint_forecast_configuration
# ---------------------------------------------------------------------------

def test_joint_auto_selects_best_balanced_candidate(monkeypatch) -> None:
    # cp=0.03 dominates every error metric, so any winning candidate must
    # carry changepoint_prior_scale=0.03.
    metrics_by_cp = {
        0.01: {"MAPE": 20.0, "MAE": 12.0, "RMSE": 15.0},
        0.03: {"MAPE": 5.0, "MAE": 3.0, "RMSE": 4.0},
        0.05: {"MAPE": 10.0, "MAE": 6.0, "RMSE": 8.0},
        0.10: {"MAPE": 12.0, "MAE": 8.0, "RMSE": 10.0},
        0.20: {"MAPE": 25.0, "MAE": 15.0, "RMSE": 18.0},
    }
    _patch_joint_dependencies(monkeypatch, metrics_by_cp=metrics_by_cp)
    selection = select_joint_forecast_configuration(
        df_prepared=pd.DataFrame({"ds": [], "y": []}),
        forecast_periods=6,
        target_coverage=0.90,
    )
    assert selection.changepoint_prior_scale == pytest.approx(0.03)
    assert selection.primary_metric == "Joint model score"
    assert selection.primary_score is not None
    assert selection.uncertainty_method in {"prophet", "conformal"}
    assert "Joint auto-tune selected" in (selection.note or "")

    diagnostics = selection.diagnostics_df
    assert (diagnostics["parameter_family"] == "joint_model_config").all()
    assert int(diagnostics["selected"].sum()) == 1
    selected_row = diagnostics[diagnostics["selected"]].iloc[0]
    assert selected_row["changepoint_prior_scale"] == pytest.approx(0.03)


def test_joint_auto_prefers_method_with_smaller_coverage_gap(monkeypatch) -> None:
    # Identical error metrics everywhere; conformal hits the 0.90 target
    # exactly while prophet misses badly, so coverage gap must decide.
    metrics_by_cp = {
        cp: {"MAPE": 5.0, "MAE": 3.0, "RMSE": 4.0}
        for cp in [0.01, 0.03, 0.05, 0.10, 0.20]
    }
    _patch_joint_dependencies(
        monkeypatch,
        metrics_by_cp=metrics_by_cp,
        prophet_coverage=0.60,
        conformal_coverage=0.90,
    )
    selection = select_joint_forecast_configuration(
        df_prepared=pd.DataFrame({"ds": [], "y": []}),
        forecast_periods=6,
        target_coverage=0.90,
    )
    assert selection.uncertainty_method == "conformal"


def test_joint_auto_diagnostics_covers_full_candidate_grid(monkeypatch) -> None:
    metrics_by_cp = {
        cp: {"MAPE": 5.0, "MAE": 3.0, "RMSE": 4.0}
        for cp in [0.01, 0.03, 0.05, 0.10, 0.20]
    }
    _patch_joint_dependencies(monkeypatch, metrics_by_cp=metrics_by_cp)
    selection = select_joint_forecast_configuration(
        df_prepared=pd.DataFrame({"ds": [], "y": []}),
        forecast_periods=6,
        target_coverage=0.90,
    )
    diagnostics = selection.diagnostics_df
    # 5 changepoints x (4 prophet interval widths + 1 conformal coverage)
    assert len(diagnostics) == 5 * 5
    assert set(diagnostics["uncertainty_method"].unique()) == {
        "prophet",
        "conformal",
    }


def test_joint_auto_falls_back_when_all_candidates_fail(monkeypatch) -> None:
    _patch_joint_dependencies(monkeypatch, metrics_by_cp={}, fail_all=True)
    selection = select_joint_forecast_configuration(
        df_prepared=pd.DataFrame({"ds": [], "y": []}),
        forecast_periods=6,
        target_coverage=0.90,
    )
    assert selection.changepoint_prior_scale == pytest.approx(
        DEFAULT_CHANGEPOINT_FALLBACK
    )
    assert selection.interval_width == pytest.approx(DEFAULT_INTERVAL_WIDTH_FALLBACK)
    assert selection.uncertainty_method == "prophet"
    assert selection.primary_score is None
    assert "fallback" in (selection.note or "").lower()
    # Every attempted candidate recorded its failure type
    assert (selection.diagnostics_df["note"] == "RuntimeError").all()


def test_forecast_visits_wires_joint_auto_selection(monkeypatch) -> None:
    # Stub the joint selection so service wiring can be asserted exactly.
    stub_selection = TuningSelection(
        changepoint_prior_scale=0.10,
        interval_width=0.85,
        primary_metric="Joint model score",
        primary_score=1.23,
        note="stub joint note",
        diagnostics_df=pd.DataFrame(
            [
                {
                    "parameter_family": "joint_model_config",
                    "candidate_value": "conformal|cp=0.10|iw=0.85",
                    "selected": True,
                }
            ]
        ),
        uncertainty_method="conformal",
    )
    captured_kwargs = {}

    def fake_joint(**kwargs):
        captured_kwargs.update(kwargs)
        return stub_selection

    monkeypatch.setattr(
        "visit_forecast.service.select_joint_forecast_configuration", fake_joint
    )

    result = forecast_visits(
        use_synthetic=True,
        timeframe_start="2020-04-01",
        timeframe_end="2023-03-01",
        forecast_periods=4,
        department="All",
        tuning_mode="auto",
        uncertainty_method="auto",
        joint_auto_tuning_enabled=True,
        target_coverage=0.85,
    )

    assert captured_kwargs["forecast_periods"] == 4
    assert result.selected_changepoint_prior_scale == pytest.approx(0.10)
    assert result.selected_interval_width == pytest.approx(0.85)
    assert result.tuning_primary_metric == "Joint model score"
    assert result.tuning_primary_score == pytest.approx(1.23)
    assert "stub joint note" in (result.tuning_note or "")
    assert result.tuning_diagnostics_df is not None
    assert (
        result.tuning_diagnostics_df["parameter_family"] == "joint_model_config"
    ).any()


def test_joint_auto_skipped_for_non_prophet_models() -> None:
    result = forecast_visits(
        use_synthetic=True,
        timeframe_start="2020-04-01",
        timeframe_end="2023-03-01",
        forecast_periods=4,
        department="All",
        model_choice="ets",
        tuning_mode="auto",
        uncertainty_method="auto",
        joint_auto_tuning_enabled=True,
    )
    assert "skipped" in (result.tuning_note or "")
    assert result.selected_model == "ets"


# ---------------------------------------------------------------------------
# Department-level forecasting
# ---------------------------------------------------------------------------

def test_forecast_departments_selects_model_per_department() -> None:
    bundle = forecast_departments(
        use_synthetic=True,
        timeframe_start="2020-04-01",
        timeframe_end="2023-03-01",
        forecast_periods=4,
        model_choice="auto",
        auto_candidates=["ets", "theta", "seasonal_naive"],
    )
    assert set(bundle.results) == {"Cardiology", "Neurology", "Oncology"}
    assert len(bundle.summary_df) == 3
    assert bundle.summary_df["selected_model"].notna().all()
    assert not bundle.failures

    combined = bundle.combined_future_df
    departments = set(combined["Department"].unique())
    assert "All (sum of departments)" in departments

    totals = combined[combined["Department"] == "All (sum of departments)"]
    parts = combined[combined["Department"] != "All (sum of departments)"]
    summed = parts.groupby("forecast_month")["yhat_adjusted"].sum()
    for _, row in totals.iterrows():
        assert row["yhat_adjusted"] == pytest.approx(
            summed.loc[row["forecast_month"]]
        )


def test_forecast_departments_filters_and_excludes() -> None:
    bundle = forecast_departments(
        use_synthetic=True,
        timeframe_start="2020-04-01",
        timeframe_end="2022-03-01",
        forecast_periods=3,
        model_choice="seasonal_naive",
        departments=["Cardiology", "Oncology"],
        exclude_departments=["Oncology"],
    )
    assert set(bundle.results) == {"Cardiology"}


def test_forecast_departments_raises_when_nothing_left() -> None:
    with pytest.raises(ValueError, match="No departments"):
        forecast_departments(
            use_synthetic=True,
            timeframe_start="2020-04-01",
            timeframe_end="2022-03-01",
            forecast_periods=3,
            departments=["Radiology"],  # not in synthetic data
        )


# ---------------------------------------------------------------------------
# Scenario re-application
# ---------------------------------------------------------------------------

def _baseline_future_df(periods: int = 6) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ds": pd.date_range("2026-01-01", periods=periods, freq="MS"),
            "yhat_original": [100.0] * periods,
            "yhat_adjusted": [100.0] * periods,
            "forecast_month": range(1, periods + 1),
        }
    )


def test_apply_capacity_phases_recomputes_from_original() -> None:
    base = _baseline_future_df()
    scenario_a = apply_capacity_phases_to_future_df(
        base, [{"enabled": True, "mode": "add", "percent": 10, "start_month": 1, "end_month": 3}]
    )
    scenario_b = apply_capacity_phases_to_future_df(
        base, [{"enabled": True, "mode": "loss", "percent": 20, "start_month": 4, "end_month": 6}]
    )
    assert np.allclose(scenario_a["yhat_adjusted"][:3], 110.0)
    assert np.allclose(scenario_a["yhat_adjusted"][3:], 100.0)
    assert np.allclose(scenario_b["yhat_adjusted"][3:], 80.0)
    # base frame untouched
    assert np.allclose(base["yhat_adjusted"], 100.0)


def test_apply_capacity_phases_independent_of_prior_adjustments() -> None:
    base = _baseline_future_df()
    first = apply_capacity_phases_to_future_df(
        base, [{"enabled": True, "mode": "loss", "percent": 50, "start_month": 1, "end_month": 6}]
    )
    # Applying a new scenario to an already-adjusted frame must start from
    # yhat_original, not compound on top of the previous adjustment.
    second = apply_capacity_phases_to_future_df(
        first, [{"enabled": True, "mode": "add", "percent": 10, "start_month": 1, "end_month": 6}]
    )
    assert np.allclose(second["yhat_adjusted"], 110.0)


def test_apply_capacity_phases_requires_original_column() -> None:
    with pytest.raises(ValueError, match="yhat_original"):
        apply_capacity_phases_to_future_df(
            pd.DataFrame({"ds": pd.date_range("2026-01-01", periods=3, freq="MS")}),
            [],
        )
