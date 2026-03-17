from __future__ import annotations

import sys
from pathlib import Path


def ensure_project_paths() -> None:
    """Prefer this checkout's source tree over any stale editable installs."""
    app_root = Path(__file__).resolve().parent
    repository_root = app_root.parents[1]
    source_root = repository_root / "src"

    for path in (app_root, source_root):
        path_str = str(path)
        if path_str in sys.path:
            sys.path.remove(path_str)
        sys.path.insert(0, path_str)
