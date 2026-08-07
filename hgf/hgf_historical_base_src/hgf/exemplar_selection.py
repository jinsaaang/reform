"""Deterministic per-question selection from the fixed exemplar bank."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_fixed_exemplar_bank(
    paths: list[Path],
) -> dict[str, dict[str, Any]]:
    """Load and deduplicate worked exemplars by memory-question ID."""
    bank: dict[str, dict[str, Any]] = {}
    canonical: dict[str, str] = {}
    source: dict[str, Path] = {}
    for root in paths:
        if not root.exists():
            continue
        for path in root.rglob("*.json"):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            worked = payload.get("worked_exemplar")
            memory_id = (
                payload.get("retrieved_memory_question_id")
                or payload.get("source_question_id")
                or payload.get("memory_question_id")
            )
            if not isinstance(worked, dict) or not memory_id:
                continue
            key = str(memory_id)
            rendered = json.dumps(
                worked,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            if key in canonical and canonical[key] != rendered:
                raise ValueError(
                    f"conflicting fixed exemplars for {key}: "
                    f"{source[key]} vs {path}"
                )
            canonical[key] = rendered
            source[key] = path
            bank[key] = worked
    return bank
