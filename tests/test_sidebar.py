from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pandas as pd


sidebar_module_path = (
    Path(__file__).resolve().parents[1]
    / "apps"
    / "streamlit"
    / "components"
    / "sidebar.py"
)
sidebar_module_spec = importlib.util.spec_from_file_location(
    "streamlit_sidebar", sidebar_module_path
)
sidebar_module = importlib.util.module_from_spec(sidebar_module_spec)
assert sidebar_module_spec is not None and sidebar_module_spec.loader is not None
sidebar_module_spec.loader.exec_module(sidebar_module)

_infer_timeframe_bounds = sidebar_module._infer_timeframe_bounds
_sync_timeframe_defaults = sidebar_module._sync_timeframe_defaults


def test_infer_timeframe_bounds_uses_min_and_max_valid_dates():
    df_preview = pd.DataFrame(
        {
            "Date": ["2024-03-01", "2024-01-01", "bad-date", "2024-02-01"],
            "Visits": [1, 2, 3, 4],
        }
    )

    result = _infer_timeframe_bounds(df_preview)

    assert result == ("2024-01-01", "2024-03-01")


def test_sync_timeframe_defaults_sets_session_values_for_uploaded_dataset():
    session_state: dict[str, object] = {}
    df_preview = pd.DataFrame(
        {
            "Date": ["2024-01-01", "2024-04-01", "2024-02-01"],
            "Visits": [10, 20, 30],
            "Department": ["A", "A", "A"],
        }
    )
    uploaded = SimpleNamespace(name="visits.csv")

    _sync_timeframe_defaults(
        session_state,
        source_mode="Upload file",
        uploaded=uploaded,
        pasted_text=None,
        df_preview=df_preview,
    )

    assert session_state["start"] == "2024-01-01"
    assert session_state["end"] == "2024-04-01"
    assert str(session_state["_timeframe_source_signature"]).startswith("upload:")


def test_sync_timeframe_defaults_does_not_overwrite_manual_values_for_same_dataset():
    df_preview = pd.DataFrame(
        {
            "Date": ["2024-01-01", "2024-04-01"],
            "Visits": [10, 20],
            "Department": ["A", "A"],
        }
    )
    session_state: dict[str, object] = {}
    uploaded = SimpleNamespace(name="visits.csv")

    _sync_timeframe_defaults(
        session_state,
        source_mode="Upload file",
        uploaded=uploaded,
        pasted_text=None,
        df_preview=df_preview,
    )
    session_state["start"] = "2024-02-01"
    session_state["end"] = "2024-03-01"

    _sync_timeframe_defaults(
        session_state,
        source_mode="Upload file",
        uploaded=uploaded,
        pasted_text=None,
        df_preview=df_preview,
    )

    assert session_state["start"] == "2024-02-01"
    assert session_state["end"] == "2024-03-01"


def test_sync_timeframe_defaults_updates_when_pasted_dataset_changes():
    session_state: dict[str, object] = {}
    first_df = pd.DataFrame(
        {
            "Date": ["2024-01-01", "2024-03-01"],
            "Visits": [10, 20],
            "Department": ["A", "A"],
        }
    )
    second_df = pd.DataFrame(
        {
            "Date": ["2025-02-01", "2025-05-01"],
            "Visits": [15, 25],
            "Department": ["B", "B"],
        }
    )

    _sync_timeframe_defaults(
        session_state,
        source_mode="Paste values (CSV)",
        uploaded=None,
        pasted_text="Date,Visits,Department\n2024-01-01,10,A\n2024-03-01,20,A",
        df_preview=first_df,
    )
    _sync_timeframe_defaults(
        session_state,
        source_mode="Paste values (CSV)",
        uploaded=None,
        pasted_text="Date,Visits,Department\n2025-02-01,15,B\n2025-05-01,25,B",
        df_preview=second_df,
    )

    assert session_state["start"] == "2025-02-01"
    assert session_state["end"] == "2025-05-01"
