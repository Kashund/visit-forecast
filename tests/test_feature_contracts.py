import numpy as np
import pandas as pd
import pytest

from visit_forecast import forecast_visits, apply_capacity_phases_to_future_df
from visit_forecast.model_prophet import _partition_conformal_residuals
from visit_forecast.models import select_forecast_model
from visit_forecast.exports import scenario_tables, excel_bytes


def test_partition_purges_unobservable_outcomes():
    rows = [
        {"cutoff": c, "ds": c + pd.offsets.MonthBegin(h), "horizon_step": h}
        for c in pd.date_range("2023-01-01", periods=8, freq="MS")
        for h in range(1, 4)
    ]
    result = _partition_conformal_residuals(pd.DataFrame(rows))
    calibration = result[result.conformal_partition.eq("calibration")]
    evaluation = result[result.conformal_partition.eq("evaluation")]
    assert calibration.ds.max() <= evaluation.cutoff.min()
    assert not set(calibration.ds).intersection(evaluation.ds)
    assert result.conformal_partition.eq("purged").any()


def test_zero_demand_keeps_allowed_model_and_validates_candidates():
    frame = pd.DataFrame(
        {"ds": pd.date_range("2020-01-01", periods=36, freq="MS"), "y": [0.0] * 36}
    )
    choice = select_forecast_model(frame, 3, candidates=["seasonal_naive"])
    assert choice.model_name == "seasonal_naive"
    assert "MAE" in choice.primary_metric
    for candidates in [[], ["invalid"]]:
        with pytest.raises(ValueError, match="nonempty subset"):
            select_forecast_model(frame, 3, candidates=candidates)


def test_classical_auto_uncertainty_and_requested_method_are_truthful():
    result = forecast_visits(
        use_synthetic=True,
        timeframe_start="2020-01-01",
        timeframe_end="2020-12-01",
        forecast_periods=3,
        model_choice="ets",
        uncertainty_method="auto",
        target_coverage=0.85,
    )
    assert result.requested_uncertainty_method == "auto"
    assert result.uncertainty_method == "conformal"
    assert result.target_coverage == 0.85
    assert result.future_forecast_df.interval_source.isin(
        ["conformal", "rmse_approx"]
    ).all()
    assert (
        np.isfinite(result.future_forecast_df[["yhat_lower", "yhat_upper"]]).all().all()
    )
    assert result.run_manifest["settings"]["requested_uncertainty_method"] == "auto"


def test_scenario_points_and_bounds_restart_from_immutable_baseline():
    base = pd.DataFrame(
        {
            "ds": pd.date_range("2025-01-01", periods=2, freq="MS"),
            "forecast_month": [1, 2],
            "yhat_original": [100.0, 0.0],
            "yhat_adjusted": [100.0, 0.0],
            "yhat_lower": [80.0, 0.0],
            "yhat_upper": [120.0, 10.0],
        }
    )
    loss = apply_capacity_phases_to_future_df(base, [{"mode": "loss", "percent": 100}])
    assert loss[["yhat_adjusted", "yhat_lower", "yhat_upper"]].eq(0).all().all()
    add = apply_capacity_phases_to_future_df(loss, [{"mode": "add", "percent": 150}])
    assert add.yhat_adjusted.tolist() == [250.0, 0.0]
    assert add.yhat_upper.tolist() == [300.0, 25.0]
    disabled = apply_capacity_phases_to_future_df(
        base, [{"enabled": False, "percent": "invalid", "start_month": None}]
    )
    assert disabled.yhat_adjusted.tolist() == [100.0, 0.0]


def test_reserved_block_and_export_arithmetic():
    result = forecast_visits(
        use_synthetic=True,
        timeframe_start="2020-01-01",
        timeframe_end="2024-12-01",
        forecast_periods=3,
        model_choice="auto",
        auto_candidates=["seasonal_naive"],
        uncertainty_method="conformal",
    )
    evaluation = result.evaluation_summary
    assert evaluation["mode"] == "reserved_final_block"
    assert evaluation["training_end"] < evaluation["evaluation_start"]
    assert evaluation["metrics"]["n_scored"] == 3
    assert result.selected_model == "seasonal_naive"
    monthly, fy, definitions = scenario_tables(
        result, [{"mode": "add", "percent": 12.34567}], []
    )
    assert monthly.A_minus_Baseline.sum() == pytest.approx(fy.A_minus_Baseline.sum())
    assert fy.Scenario_A.sum() == pytest.approx(monthly.Scenario_A.sum())
    data = excel_bytes(
        {
            "Monthly_Comparison": monthly,
            "FY_Summary": fy,
            "Scenario_Definitions": definitions,
        },
        result.run_manifest,
    )
    import io

    excel = pd.ExcelFile(io.BytesIO(data))
    assert "Run_Manifest" in excel.sheet_names
    exported = pd.read_excel(excel, sheet_name="Monthly_Comparison")
    np.testing.assert_allclose(exported.Scenario_A, monthly.Scenario_A)


def test_cli_workflows_write_machine_readable_exports(monkeypatch, tmp_path):
    import runpy
    import sys
    from pathlib import Path

    script = Path(__file__).resolve().parents[1] / "scripts" / "run_forecast_cli.py"
    for workflow in ["forecast", "departments", "scenarios"]:
        directory = tmp_path / workflow
        monkeypatch.setattr(
            sys,
            "argv",
            [
                str(script),
                "--workflow",
                workflow,
                "--use_synthetic",
                "--model",
                "seasonal_naive",
                "--months",
                "1",
                "--fiscal-view",
                "month",
                "--output-dir",
                str(directory),
            ],
        )
        runpy.run_path(str(script), run_name="__main__")
        assert (directory / "run_manifest.json").is_file()
        assert (directory / "forecast.xlsx").is_file()
        if workflow == "forecast":
            fiscal = pd.read_csv(directory / "fiscal_summary.csv")
            assert "Fiscal_Month" in fiscal


def test_covariates_reach_prophet_tuning(monkeypatch):
    from visit_forecast import tuning

    captured = []
    covariates = object()

    def fit(**kwargs):
        captured.append(kwargs["future_covariates"])
        return object(), object(), object()

    monkeypatch.setattr(tuning, "fit_and_forecast", fit)
    monkeypatch.setattr(
        tuning,
        "backtest",
        lambda **kwargs: {"MAPE": 1.0, "MAE": 1.0, "RMSE": 1.0}
        if kwargs["future_covariates"] is covariates
        else {},
    )
    tuning.select_prophet_hyperparameters(
        pd.DataFrame(),
        3,
        changepoint_candidates=[0.05],
        tune_interval_width=False,
        fixed_interval_width=0.9,
        future_covariates=covariates,
    )
    assert captured == [covariates]


def test_auto_selection_never_sees_reserved_outcomes(monkeypatch):
    from visit_forecast import service

    original = service.select_forecast_model
    endpoints = []

    def select(**kwargs):
        endpoints.append(kwargs["df_prepared"].ds.max())
        return original(**kwargs)

    monkeypatch.setattr(service, "select_forecast_model", select)
    result = forecast_visits(
        use_synthetic=True,
        timeframe_start="2020-01-01",
        timeframe_end="2024-12-01",
        forecast_periods=3,
        model_choice="auto",
        auto_candidates=["seasonal_naive"],
    )
    assert endpoints == [pd.Timestamp("2024-09-01")]
    assert result.evaluation_summary["evaluation_start"] == "2024-10-01"


def test_manifest_json_uses_null_for_unavailable_metrics():
    import json
    from visit_forecast.provenance import manifest_json

    text = manifest_json({"MAPE": float("nan"), "nested": {"coverage": float("inf")}})
    decoded = json.loads(
        text, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value))
    )
    assert decoded == {"MAPE": None, "nested": {"coverage": None}}


def test_failed_candidate_does_not_change_metric_policy(monkeypatch):
    from visit_forecast import models

    original = models.build_model

    def build(name, **kwargs):
        if name == "ets":
            raise ValueError("candidate deliberately ineligible")
        return original(name, **kwargs)

    monkeypatch.setattr(models, "build_model", build)
    frame = pd.DataFrame(
        {"ds": pd.date_range("2020-01-01", periods=36, freq="MS"), "y": [100.0] * 36}
    )
    choice = models.select_forecast_model(
        frame, 3, candidates=["ets", "seasonal_naive"]
    )
    assert choice.primary_metric == "MAPE+RMSE balance"
    assert choice.model_name == "seasonal_naive"


def test_auto_candidates_use_identical_validation_dates():
    from visit_forecast.io import generate_synthetic_data, prepare_dataframe
    from visit_forecast.models import select_forecast_model

    raw = generate_synthetic_data("2020-01-01", "2022-12-01")
    prepared, _ = prepare_dataframe(raw, "2020-01-01", "2022-12-01", "All", None)
    choice = select_forecast_model(
        prepared, 6, candidates=["ets", "arima", "seasonal_naive"]
    )
    scored = choice.diagnostics_df.dropna(subset=["rmse"])
    assert len(scored) == 3
    assert scored.n_scored.nunique() == 1
    assert scored.validation_start.nunique() == 1
    assert scored.n_scored.iloc[0] == 6


def test_short_arima_request_has_actionable_eligibility_error():
    from visit_forecast.models import fit_and_forecast_model

    frame = pd.DataFrame(
        {"ds": pd.date_range("2020-01-01", periods=12, freq="MS"), "y": [100.0] * 12}
    )
    with pytest.raises(ValueError, match="requires at least 30 monthly observations"):
        fit_and_forecast_model(frame, 3, "arima")
