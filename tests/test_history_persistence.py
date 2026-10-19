from pathlib import Path
import importlib.util
import json
from concurrent.futures import ThreadPoolExecutor
import pytest

path = Path(__file__).resolve().parents[1] / "apps" / "streamlit" / "state.py"
spec = importlib.util.spec_from_file_location("history_persistence_state", path)
state = importlib.util.module_from_spec(spec)
spec.loader.exec_module(state)


def test_concurrent_history_writes_preserve_distinct_runs(monkeypatch, tmp_path):
    monkeypatch.setenv("VISIT_FORECAST_PERSIST_HISTORY", "1")
    monkeypatch.setenv("VISIT_FORECAST_HISTORY_DIR", str(tmp_path))
    entries = [
        dict(run_id=f"run-{i}", run_time=f"2026-10-19T10:00:0{i}+00:00")
        for i in range(4)
    ]
    with ThreadPoolExecutor(max_workers=4) as pool:
        errors = list(
            pool.map(lambda entry: state.save_persisted_history([entry]), entries)
        )
    assert errors == [None] * 4
    history = json.loads((tmp_path / "run_history.json").read_text())
    assert {entry["run_id"] for entry in history} == {
        entry["run_id"] for entry in entries
    }
    assert not list(tmp_path.glob("tmp*"))


def test_failed_history_save_is_reported(monkeypatch, tmp_path):
    monkeypatch.setenv("VISIT_FORECAST_PERSIST_HISTORY", "1")
    monkeypatch.setenv("VISIT_FORECAST_HISTORY_DIR", str(tmp_path))
    (tmp_path / "run_history.json").mkdir()
    with pytest.warns(RuntimeWarning, match="could not be saved"):
        error = state.save_persisted_history([{"run_id": "run"}])
    assert error and "session" in error
