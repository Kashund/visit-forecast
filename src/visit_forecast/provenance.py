"""Portable metadata for interpreting and reproducing exported results."""

from __future__ import annotations
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from importlib.metadata import version, PackageNotFoundError
from uuid import uuid4
import math


def configuration_fingerprint(settings, frame=None):
    digest = hashlib.sha256(json.dumps(settings, sort_keys=True, default=str).encode())
    if frame is not None:
        digest.update(
            frame.to_csv(index=False, date_format="%Y-%m-%dT%H:%M:%S").encode()
        )
    return digest.hexdigest()


def build_run_manifest(frame, settings, evaluation=None):
    packages = {}
    for name in ["visit-forecast", "darts", "prophet", "pandas", "numpy", "streamlit"]:
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = "unavailable"
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL, text=True
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    return dict(
        schema_version=1,
        run_id=str(uuid4()),
        created_at=datetime.now(timezone.utc).isoformat(),
        data_fingerprint=configuration_fingerprint({}, frame),
        code_commit=commit,
        working_tree_modified=dirty,
        dependencies=packages,
        settings=settings,
        evaluation=evaluation or {},
        assumptions=[
            "Monthly counts; negative point predictions clipped to zero.",
            "Capacity adjustments are assumptions, not estimated causal effects.",
            "Empirical intervals assume sufficiently stable errors; nominal coverage is not guaranteed.",
        ],
    )


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def manifest_json(manifest):
    return json.dumps(json_safe(manifest), indent=2, default=str, allow_nan=False)
