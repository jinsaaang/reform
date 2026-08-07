from __future__ import annotations

import os
from pathlib import Path


def _find_repository_root() -> Path:
    candidates = []
    configured = os.environ.get("HGF_ROOT")
    if configured:
        candidates.append(Path(configured))
    candidates.extend((Path.cwd(), Path(__file__).resolve().parents[2]))
    for candidate in candidates:
        resolved = candidate.resolve()
        if (
            resolved / "configs" / "reproduction.json"
        ).is_file():
            return resolved
    return Path(__file__).resolve().parents[2]


PACKAGE_ROOT = _find_repository_root()
