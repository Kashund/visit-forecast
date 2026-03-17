#!/usr/bin/env python3
"""Launch the Streamlit app and open it in a browser."""

from __future__ import annotations

import importlib.util
import shutil
import socket
import subprocess
import sys
import time
import webbrowser
from pathlib import Path
from typing import Optional, Sequence


def wait_for_port(host: str, port: int, timeout_seconds: int) -> bool:
    """Return True when the TCP port is reachable before timeout."""
    end_time = time.time() + timeout_seconds
    while time.time() < end_time:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
            client_socket.settimeout(0.5)
            if client_socket.connect_ex((host, port)) == 0:
                return True
        time.sleep(0.2)
    return False


def has_streamlit_module(python_executable: str) -> bool:
    """Return True when Streamlit is importable in this interpreter."""
    if python_executable != sys.executable:
        return False
    return importlib.util.find_spec("streamlit") is not None


def build_streamlit_launch_command(
    app_file_path: Path, host: str, port: int
) -> Optional[Sequence[str]]:
    """Build the best available launch command for Streamlit."""
    if has_streamlit_module(sys.executable):
        return [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(app_file_path),
            "--server.address",
            host,
            "--server.port",
            str(port),
        ]

    streamlit_executable = shutil.which("streamlit")
    if streamlit_executable:
        return [
            streamlit_executable,
            "run",
            str(app_file_path),
            "--server.address",
            host,
            "--server.port",
            str(port),
        ]

    return None


def main() -> int:
    repository_root = Path(__file__).resolve().parents[1]
    app_file_path = repository_root / "apps" / "streamlit" / "Home.py"
    host = "127.0.0.1"
    port = 8501
    application_url = f"http://{host}:{port}"

    launch_command = build_streamlit_launch_command(app_file_path, host, port)
    if launch_command is None:
        print(
            "Could not find Streamlit in this Python interpreter or on PATH.\n"
            "Activate your project environment, then install dependencies and rerun:\n"
            "  pip install -e .\n"
            "Then run:\n"
            "  python scripts/launch_streamlit_app.py",
            file=sys.stderr,
        )
        return 1

    streamlit_process = subprocess.Popen(launch_command, cwd=repository_root)

    if wait_for_port(host=host, port=port, timeout_seconds=30):
        webbrowser.open(application_url)
        print(f"Opened {application_url}")
    else:
        print(
            "Streamlit did not become available within 30 seconds. "
            f"Open {application_url} manually.",
            file=sys.stderr,
        )

    try:
        return streamlit_process.wait()
    except KeyboardInterrupt:
        streamlit_process.terminate()
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
