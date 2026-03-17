from __future__ import annotations

import sys
from pathlib import Path


def _ensure_local_src_on_path() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    source_root = repository_root / "src"
    source_root_str = str(source_root)
    if source_root_str in sys.path:
        sys.path.remove(source_root_str)
    sys.path.insert(0, source_root_str)


_ensure_local_src_on_path()
