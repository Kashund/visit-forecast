"""Data selection shared by forecasting and Data Check."""

from __future__ import annotations
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
import pandas as pd
import streamlit as st
from visit_forecast.io import generate_synthetic_data, read_data_file
from visit_forecast.provenance import configuration_fingerprint

SOURCE_MODES = [
    "Upload file",
    "Paste values (CSV)",
    "Use synthetic sample",
    "Use bundled sample",
    "Use shared dataset",
]
SAMPLE_DIR = Path(__file__).resolve().parents[3] / "data" / "sample"


@dataclass
class DataSource:
    mode: str
    label: str
    frame: pd.DataFrame | None = None
    uploaded: object = None
    pasted_text: str | None = None
    settings: dict | None = None

    def resolve(self, start, end):
        if self.mode == "Use synthetic sample":
            return generate_synthetic_data(start, end)
        return self.frame.copy() if self.frame is not None else None


def select_data_source(
    prefix="", container=None, *, include_current=False, default_mode=None
):
    ui = container if container is not None else st.sidebar
    choices = (["Current forecast data"] if include_current else []) + SOURCE_MODES
    active = st.session_state.get("active_dataset")
    seen = dict(st.session_state.get("source_seen", {}))
    active_id = active["fingerprint"] if active else None
    mode_key = f"{prefix}source_mode"
    # Another page may have selected a new dataset since this page was visited.
    # Prefer it automatically over an old uploader or pasted text widget.
    if active and seen.get(prefix) != active_id and not include_current:
        st.session_state[mode_key] = "Use shared dataset"
    elif active and seen.get(prefix) != active_id and include_current:
        st.session_state[mode_key] = "Current forecast data"
        st.session_state.pop(f"{prefix}snapshot", None)
    elif mode_key not in st.session_state and default_mode:
        st.session_state[mode_key] = (
            "Upload file"
            if st.session_state.get("data_generation", 0)
            else default_mode
        )
    seen[prefix] = active_id
    st.session_state["source_seen"] = seen
    mode = ui.radio("Choose how to provide data", choices, key=f"{prefix}source_mode")
    selection = DataSource(mode, mode)
    if mode == "Upload file":
        selection.uploaded = ui.file_uploader(
            "Upload Excel (.xlsx) or CSV",
            type=["xlsx", "csv"],
            key=f"{prefix}uploader_{st.session_state.get('data_generation', 0)}",
        )
        if selection.uploaded is not None:
            selection.label = selection.uploaded.name
            selection.frame = read_data_file(selection.uploaded)
        elif active and active["source_mode"] == mode:
            selection.label = active["label"]
            selection.frame = active["frame"].copy(deep=True)
            ui.caption(
                f"Saved upload: {selection.label}. Upload another file to replace it."
            )
    elif mode == "Paste values (CSV)":
        selection.pasted_text = ui.text_area(
            "CSV data",
            height=180,
            key=f"{prefix}pasted_csv",
            placeholder="Date,Visits,Department\n2024-01-01,120,Cardiology",
        )
        selection.label = "Pasted CSV"
        if selection.pasted_text.strip():
            selection.frame = pd.read_csv(StringIO(selection.pasted_text))
    elif mode == "Use bundled sample":
        filename = ui.selectbox(
            "Sample dataset",
            ["synthetic_visits.csv", "synthetic_visits_small.csv"],
            key=f"{prefix}sample_dataset",
        )
        selection.label = filename
        selection.frame = read_data_file(SAMPLE_DIR / filename)
    elif mode == "Use shared dataset":
        frame = active["frame"] if active else st.session_state.get("shared_dataset")
        if frame is not None:
            selection.frame = frame.copy()
            selection.label = active["label"] if active else "Shared dataset"
        else:
            ui.info(
                "No shared dataset yet. Upload or paste data, select a sample, or run a forecast first."
            )
    elif mode == "Current forecast data":
        snapshots = st.session_state.get("forecast_data_snapshots", {})
        available = {}
        if active:
            available["Active dataset"] = {
                **active,
                "settings": {},
                "stage": "Active dataset",
            }
        if "last_run_data" in st.session_state:
            available["Last run"] = st.session_state["last_run_data"]
        if "current_input_data" in st.session_state:
            available["Current inputs"] = st.session_state["current_input_data"]
        for name, snapshot in snapshots.items():
            available[f"Last run: {name}"] = snapshot
        if not available and st.session_state.get("shared_dataset") is not None:
            available["Shared dataset"] = dict(
                frame=st.session_state["shared_dataset"],
                workflow="Shared data",
                label="Previously selected dataset",
                source_mode="Use shared dataset",
                stage="Available input",
                settings={},
            )
        if available:
            choice = ui.selectbox(
                "Data to inspect", list(available), key=f"{prefix}snapshot"
            )
            snapshot = available[choice]
            selection.frame = snapshot["frame"].copy()
            selection.label = (
                f"{snapshot['workflow']} · {snapshot['label']} ({snapshot['stage']})"
            )
            ui.caption(f"Source: {snapshot['source_mode']}")
            settings = snapshot.get("settings", {})
            selection.settings = dict(settings)
            if settings:
                ui.caption(
                    f"Training window: {settings.get('timeframe_start')} to {settings.get('timeframe_end')}"
                )
        else:
            ui.info(
                "No forecast inputs selected yet. Choose an upload, pasted CSV, or a sample below to inspect it."
            )
    else:
        selection.label = "Generated synthetic sample"
    return selection


def remember_data(frame, source, settings, workflow, *, ran=False):
    """Copy raw inputs before validation so invalid data can also be inspected."""
    if frame is None:
        return
    snapshot = dict(
        frame=frame.copy(deep=True),
        source_mode=source.mode,
        label=source.label,
        settings=dict(settings),
        workflow=workflow,
        stage="Last run" if ran else "Current inputs",
        fingerprint=configuration_fingerprint({}, frame),
    )
    if source.mode != "Current forecast data":
        active = st.session_state.get("active_dataset")
        if active is None or active["fingerprint"] != snapshot["fingerprint"]:
            st.session_state["active_dataset"] = dict(snapshot)
        seen = dict(st.session_state.get("source_seen", {}))
        prefixes = {
            "Run Forecast": "",
            "Department Forecasts": "dept_",
            "Scenario Comparison": "cmp_",
            "Data Check": "check_",
        }
        seen[prefixes[workflow]] = snapshot["fingerprint"]
        st.session_state["source_seen"] = seen
    st.session_state["current_input_data"] = snapshot
    if source.mode != "Current forecast data":
        st.session_state["shared_dataset"] = frame.copy(deep=True)
    if ran:
        st.session_state["last_run_data"] = snapshot
        snapshots = dict(st.session_state.get("forecast_data_snapshots", {}))
        snapshots[workflow] = snapshot
        st.session_state["forecast_data_snapshots"] = snapshots


def preview_data(frame, label, *, filtered=False):
    st.subheader("Data preview")
    detail = (
        "Rows selected by the run filters, before monthly aggregation."
        if filtered
        else "Raw observations before filtering or aggregation."
    )
    st.caption(f"{label} · {len(frame):,} rows · showing the first 50 rows. {detail}")
    st.dataframe(frame.head(50), width="stretch")


def selected_training_rows(frame, settings):
    """Preview the source observations selected by a run, without cleaning them."""
    selected = frame.copy()
    if (
        "Date" in selected
        and settings.get("timeframe_start")
        and settings.get("timeframe_end")
    ):
        months = pd.to_datetime(
            selected.Date, errors="coerce", format="mixed"
        ).dt.to_period("M")
        start = pd.Timestamp(settings["timeframe_start"]).to_period("M")
        end = pd.Timestamp(settings["timeframe_end"]).to_period("M")
        selected = selected[months.between(start, end)]
    if "Department" in selected:
        names = selected.Department.astype("string").str.strip()
        department = settings.get("department", "All")
        if department != "All":
            selected = selected[names.eq(str(department).strip())]
            names = selected.Department.astype("string").str.strip()
        excluded = {
            str(d).strip() for d in settings.get("exclude_departments", []) or []
        }
        selected = selected[~names.isin(excluded)]
    return selected
