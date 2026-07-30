from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, MutableMapping

HISTORY_LIMIT = 20


def history_persistence_enabled() -> bool:
    """Return whether cross-session history is explicitly enabled.

    Disk persistence is opt-in because a shared server process would otherwise
    expose one user's forecast metadata to other users.
    """
    return os.environ.get("VISIT_FORECAST_PERSIST_HISTORY", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def history_file_path() -> Path:
    """Location of the persisted run history JSON.

    Override with the VISIT_FORECAST_HISTORY_DIR environment variable.
    """
    base_dir = os.environ.get("VISIT_FORECAST_HISTORY_DIR")
    directory = Path(base_dir) if base_dir else Path.home() / ".visit_forecast"
    return directory / "run_history.json"


def load_persisted_history() -> list[dict[str, Any]]:
    """Load run history from disk. Returns [] on any problem."""
    if not history_persistence_enabled():
        return []
    try:
        path = history_file_path()
        if not path.exists():
            return []
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(loaded, list):
            return [entry for entry in loaded if isinstance(entry, dict)][
                :HISTORY_LIMIT
            ]
    except Exception:
        pass
    return []


def save_persisted_history(history: list[dict[str, Any]]) -> None:
    """Persist run history to disk. Failures are silent by design; the
    in-session history still works even if the disk write fails."""
    if not history_persistence_enabled():
        return
    try:
        path = history_file_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(history[:HISTORY_LIMIT], default=str, indent=2),
            encoding="utf-8",
        )
    except Exception:
        pass


def clear_persisted_history() -> None:
    if not history_persistence_enabled():
        return
    try:
        path = history_file_path()
        if path.exists():
            path.unlink()
    except Exception:
        pass


def ensure_history_loaded(session_state: MutableMapping[str, Any]) -> None:
    """Initialize session history, hydrating from disk only when opted in."""
    if "forecast_history" not in session_state:
        session_state["forecast_history"] = load_persisted_history()


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
        "requested_model": getattr(result, "requested_model", "prophet"),
        "selected_model": getattr(result, "selected_model", "prophet"),
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
        "requested_model": getattr(result, "requested_model", "prophet"),
        "selected_model": getattr(result, "selected_model", "prophet"),
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
    session_state["forecast_history"] = history[:HISTORY_LIMIT]
    save_persisted_history(session_state["forecast_history"])
    session_state["forecast_results_section"] = "📉 Forecast"
