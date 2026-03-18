from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace


state_module_path = (
    Path(__file__).resolve().parents[1]
    / "apps"
    / "streamlit"
    / "state.py"
)
state_module_spec = importlib.util.spec_from_file_location(
    "streamlit_state", state_module_path
)
state_module = importlib.util.module_from_spec(state_module_spec)
assert state_module_spec is not None and state_module_spec.loader is not None
state_module_spec.loader.exec_module(state_module)

apply_forecast_run_to_session = state_module.apply_forecast_run_to_session


def _make_result(
    *,
    requested_uncertainty_method: str,
    selected_uncertainty_method: str,
    target_coverage: float,
    changepoint_prior_scale: float,
    interval_width: float,
    tuning_mode: str,
) -> SimpleNamespace:
    return SimpleNamespace(
        selected_changepoint_prior_scale=changepoint_prior_scale,
        target_coverage=target_coverage,
        selected_interval_width=interval_width,
        requested_uncertainty_method=requested_uncertainty_method,
        uncertainty_method=selected_uncertainty_method,
        tuning_mode=tuning_mode,
        tuning_primary_metric="Joint model score" if tuning_mode == "auto" else None,
        tuning_primary_score=1.01 if tuning_mode == "auto" else None,
        performance_metrics={"MAPE": 8.0, "MAE": 3.0, "RMSE": 4.0},
        department_info="All",
    )


def test_apply_forecast_run_to_session_preserves_requested_vs_selected_uncertainty_across_toggle_sequence():
    session_state: dict[str, object] = {"forecast_history": []}

    common_auto_first = {
        "timeframe_start": "2020-04-01",
        "timeframe_end": "2025-03-01",
        "forecast_periods": 12,
        "department": "All",
        "exclude_departments": None,
        "capacity_phases": [],
        "changepoint_prior_scale": 0.05,
        "target_coverage": 0.90,
        "uncertainty_method": "auto",
        "tuning_mode": "auto",
    }
    apply_forecast_run_to_session(
        session_state,
        common=common_auto_first,
        result=_make_result(
            requested_uncertainty_method="auto",
            selected_uncertainty_method="conformal",
            target_coverage=0.90,
            changepoint_prior_scale=0.03,
            interval_width=0.90,
            tuning_mode="auto",
        ),
        source_mode="Use synthetic sample",
        scenario_summary="No phases configured",
        run_time="2026-03-17 10:00:00",
    )

    assert session_state["forecast_inputs"]["requested_uncertainty_method"] == "auto"
    assert session_state["forecast_inputs"]["selected_uncertainty_method"] == "conformal"

    common_manual_second = {
        **common_auto_first,
        "target_coverage": 0.95,
        "uncertainty_method": "prophet",
        "tuning_mode": "auto",
    }
    apply_forecast_run_to_session(
        session_state,
        common=common_manual_second,
        result=_make_result(
            requested_uncertainty_method="prophet",
            selected_uncertainty_method="prophet",
            target_coverage=0.95,
            changepoint_prior_scale=0.10,
            interval_width=0.95,
            tuning_mode="auto",
        ),
        source_mode="Use synthetic sample",
        scenario_summary="No phases configured",
        run_time="2026-03-17 10:05:00",
    )

    assert session_state["forecast_inputs"]["requested_uncertainty_method"] == "prophet"
    assert session_state["forecast_inputs"]["selected_uncertainty_method"] == "prophet"

    common_auto_third = {
        **common_auto_first,
        "target_coverage": 0.92,
        "uncertainty_method": "auto",
        "tuning_mode": "auto",
    }
    apply_forecast_run_to_session(
        session_state,
        common=common_auto_third,
        result=_make_result(
            requested_uncertainty_method="auto",
            selected_uncertainty_method="prophet",
            target_coverage=0.92,
            changepoint_prior_scale=0.05,
            interval_width=0.92,
            tuning_mode="auto",
        ),
        source_mode="Use synthetic sample",
        scenario_summary="No phases configured",
        run_time="2026-03-17 10:10:00",
    )

    assert session_state["forecast_inputs"]["requested_uncertainty_method"] == "auto"
    assert session_state["forecast_inputs"]["selected_uncertainty_method"] == "prophet"

    history = session_state["forecast_history"]
    assert len(history) == 3
    assert history[0]["requested_uncertainty_method"] == "auto"
    assert history[0]["selected_uncertainty_method"] == "prophet"
    assert history[1]["requested_uncertainty_method"] == "prophet"
    assert history[1]["selected_uncertainty_method"] == "prophet"
    assert history[2]["requested_uncertainty_method"] == "auto"
    assert history[2]["selected_uncertainty_method"] == "conformal"


def test_apply_forecast_run_to_session_keeps_manual_uncertainty_inputs_during_auto_tuning():
    session_state: dict[str, object] = {"forecast_history": []}
    common = {
        "timeframe_start": "2020-04-01",
        "timeframe_end": "2025-03-01",
        "forecast_periods": 12,
        "department": "All",
        "exclude_departments": None,
        "capacity_phases": [],
        "changepoint_prior_scale": 0.05,
        "target_coverage": 0.95,
        "uncertainty_method": "prophet",
        "tuning_mode": "auto",
    }

    apply_forecast_run_to_session(
        session_state,
        common=common,
        result=_make_result(
            requested_uncertainty_method="prophet",
            selected_uncertainty_method="prophet",
            target_coverage=0.95,
            changepoint_prior_scale=0.03,
            interval_width=0.95,
            tuning_mode="auto",
        ),
        source_mode="Use synthetic sample",
        scenario_summary="No phases configured",
        run_time="2026-03-17 10:15:00",
    )

    assert session_state["forecast_inputs"]["uncertainty_method"] == "prophet"
    assert session_state["forecast_inputs"]["requested_uncertainty_method"] == "prophet"
    assert session_state["forecast_inputs"]["target_coverage"] == 0.95
    assert session_state["forecast_inputs"]["interval_width"] == 0.95
