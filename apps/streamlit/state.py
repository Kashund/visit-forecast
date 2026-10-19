from __future__ import annotations

import json
import os
import tempfile
import warnings
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, MutableMapping

from visit_forecast.provenance import configuration_fingerprint, json_safe

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


@contextmanager
def _history_lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(".lock").open("a") as lock:
        try:
            import fcntl
        except ImportError:
            fcntl = None
        if fcntl:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            if fcntl:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def save_persisted_history(history: list[dict[str, Any]]) -> str | None:
    if not history_persistence_enabled():
        return None
    try:
        path = history_file_path()
        with _history_lock(path):
            existing = load_persisted_history()
            seen, merged = set(), []
            for entry in history + existing:
                key = entry.get("run_id") or configuration_fingerprint(entry)
                if key not in seen:
                    seen.add(key)
                    merged.append(entry)
            merged.sort(key=lambda x: str(x.get("run_time", "")), reverse=True)
            temp_path = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="w", encoding="utf-8", dir=path.parent, delete=False
                ) as temp:
                    temp_path = Path(temp.name)
                    temp.write(
                        json.dumps(
                            json_safe(merged[:HISTORY_LIMIT]),
                            default=str,
                            indent=2,
                            allow_nan=False,
                        )
                    )
                    temp.flush()
                    os.fsync(temp.fileno())
                temp_path.replace(path)
            finally:
                if temp_path and temp_path.exists():
                    temp_path.unlink()
    except Exception as exc:
        message = f"History remains in this session but could not be saved: {exc}"
        warnings.warn(message, RuntimeWarning)
        return message
    return None


def clear_persisted_history() -> None:
    if not history_persistence_enabled():
        return
    try:
        path = history_file_path()
        with _history_lock(path):
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
    manifest = getattr(result, "run_manifest", None) or {}
    entry = {
        "run_id": manifest.get("run_id"),
        "workflow": "forecast",
        "schema_version": 1,
        "data_fingerprint": manifest.get("data_fingerprint"),
        "run_time": run_time or datetime.now(timezone.utc).isoformat(),
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
    session_state["history_save_error"] = save_persisted_history(
        session_state["forecast_history"]
    )
    session_state["forecast_results_section"] = "📉 Forecast"


def record_auxiliary_run(session_state, workflow, manifest, **details):
    ensure_history_loaded(session_state)
    entry = dict(
        run_id=manifest["run_id"],
        run_time=manifest["created_at"],
        schema_version=1,
        workflow=workflow,
        data_fingerprint=manifest["data_fingerprint"],
        **details,
    )
    session_state["forecast_history"] = [entry, *session_state["forecast_history"]][
        :HISTORY_LIMIT
    ]
    session_state["history_save_error"] = save_persisted_history(
        session_state["forecast_history"]
    )


def cached_forecast(session_state, raw, settings, progress_callback=None):
    """Bounded, session-private reuse keyed by data, settings and engine source."""
    import copy
    import hashlib
    from dataclasses import replace
    from datetime import datetime, timezone
    from uuid import uuid4
    import visit_forecast
    from visit_forecast import forecast_visits

    engine = hashlib.sha256()
    for path in sorted(Path(visit_forecast.__file__).parent.glob("*.py")):
        engine.update(path.name.encode())
        engine.update(path.read_bytes())
    signature = configuration_fingerprint(
        {**settings, "engine": engine.hexdigest()}, raw
    )
    cache = session_state.setdefault("forecast_cache", {})
    if signature in cache:
        result = copy.deepcopy(cache[signature])
        manifest = {
            **result.run_manifest,
            "cached_from_run": result.run_manifest["run_id"],
            "run_id": str(uuid4()),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        if progress_callback:
            progress_callback("Reusing matching validated run from this session")
        return replace(result, run_manifest=manifest)
    result = forecast_visits(df=raw, **settings, progress_callback=progress_callback)
    if len(cache) >= 3:
        del cache[next(iter(cache))]
    cache[signature] = copy.deepcopy(result)
    return result
