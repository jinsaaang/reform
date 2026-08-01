"""Shared exact-family historical retrieval manifests for controlled comparisons."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from hgf.contracts import _is_temporally_eligible
from hgf.question_io import family_metadata


def load_retrieval_manifest(
    path: Path,
    *,
    expected_model: str,
    required_question_ids: list[str],
) -> tuple[dict[str, Any], dict[str, list[str]]]:
    payload = json.loads(path.resolve().read_text(encoding="utf-8"))
    if payload.get("schema_version") != "shared_retrieval_manifest_v1":
        raise ValueError("unsupported retrieval manifest schema")
    if str(payload.get("model") or "") != expected_model:
        raise ValueError("retrieval manifest belongs to another model")
    rows = {
        str(row.get("question_id") or ""): [
            str(value) for value in row.get("retrieved_memory_question_ids") or []
        ]
        for row in payload.get("results") or []
        if isinstance(row, dict)
    }
    missing = sorted(set(required_question_ids) - set(rows))
    if missing:
        raise ValueError(f"retrieval manifest misses questions: {missing}")
    return payload, rows


def validate_retrieval_ids(
    ids: list[str],
    *,
    target_question: Any,
    cutoff: Any,
    memory_questions: dict[str, Any],
    blueprints_by_id: dict[str, dict[str, Any]],
    maximum: int,
) -> list[dict[str, Any]]:
    if not ids or len(ids) > maximum or len(ids) != len(set(ids)):
        raise ValueError("retrieval IDs must be unique and within the registered limit")
    target = family_metadata(target_question)
    selected: list[dict[str, Any]] = []
    for question_id in ids:
        memory_question = memory_questions.get(question_id)
        blueprint = blueprints_by_id.get(question_id)
        if memory_question is None or blueprint is None:
            raise ValueError(f"retrieval references missing memory {question_id}")
        if not _is_temporally_eligible(memory_question, cutoff):
            raise ValueError(f"retrieval is not cutoff eligible: {question_id}")
        memory = family_metadata(memory_question)
        if (
            str(memory.get("family_id") or "")
            != str(target.get("family_id") or "")
            or str(memory.get("target_metric") or "")
            != str(target.get("target_metric") or "")
        ):
            raise ValueError(f"retrieval is not exact-family compatible: {question_id}")
        if (blueprint.get("graph_diagnosis") or {}).get("usable") is False:
            raise ValueError(f"retrieval uses an unusable blueprint: {question_id}")
        selected.append(blueprint)
    return selected
