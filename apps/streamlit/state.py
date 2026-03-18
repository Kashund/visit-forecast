from __future__ import annotations

from datetime import datetime
from typing import Any, Mapping, MutableMapping


def build_used_forecast_inputs(
    common: Mapping[str, Any],
    result: Any,
) -> dict[str, Any]:
    return {
        **common,
        "changepoint_prior_scale": result.selected_changepoint_prior_scale,
        "target_coverage": result.target_coverage,
        "interval_width": result.selected_interval_width,
        "requested_uncertainty_method": result.requested_uncertainty_method,
        "selected_uncertainty_method": result.uncertainty_method,
        "tuning_mode": result.tuning_mode,
    }


def build_forecast_history_entry(
    *,
    common: Mapping[str, Any],
    result: Any,
    source_mode: str,
    scenario_summary: str,
    run_time: str | None = None,
) -> dict[str, Any]:
    entry = {
        "run_time": run_time or datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source_mode": source_mode,
        "scenario_summary": scenario_summary,
        **common,
        "tuning_mode": result.tuning_mode,
        "changepoint_prior_scale": result.selected_changepoint_prior_scale,
        "target_coverage": result.target_coverage,
        "interval_width": result.selected_interval_width,
        "requested_uncertainty_method": result.requested_uncertainty_method,
        "selected_uncertainty_method": result.uncertainty_method,
        "tuning_primary_metric": result.tuning_primary_metric,
        "tuning_primary_score": result.tuning_primary_score,
        "MAPE": float(result.performance_metrics.get("MAPE", float("nan"))),
        "MAE": float(result.performance_metrics.get("MAE", float("nan"))),
        "RMSE": float(result.performance_metrics.get("RMSE", float("nan"))),
        "departments_included": result.department_info,
    }
    return entry


def apply_forecast_run_to_session(
    session_state: MutableMapping[str, Any],
    *,
    common: Mapping[str, Any],
    result: Any,
    source_mode: str,
    scenario_summary: str,
    run_time: str | None = None,
) -> None:
    used_forecast_inputs = build_used_forecast_inputs(common, result)
    history_entry = build_forecast_history_entry(
        common=common,
        result=result,
        source_mode=source_mode,
        scenario_summary=scenario_summary,
        run_time=run_time,
    )

    session_state["forecast_result"] = result
    session_state["forecast_inputs"] = used_forecast_inputs
    session_state["forecast_source_mode"] = source_mode
    session_state["forecast_scenario_summary"] = scenario_summary

    history = list(session_state.get("forecast_history", []))
    history.insert(0, history_entry)
    session_state["forecast_history"] = history[:20]
    session_state["forecast_results_section"] = "📉 Forecast"
