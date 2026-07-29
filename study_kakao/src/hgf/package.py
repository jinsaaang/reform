from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


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
PROTOCOL_PATH = PACKAGE_ROOT / "configs" / "reproduction.json"


def load_protocol() -> dict[str, Any]:
    return json.loads(
        PROTOCOL_PATH.read_text(encoding="utf-8")
    )
