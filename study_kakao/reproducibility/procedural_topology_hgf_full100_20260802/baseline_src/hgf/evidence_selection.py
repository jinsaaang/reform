"""Model-specific selection from the frozen cutoff-safe current-evidence pool."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any

from openai import OpenAI

from hgf.exemplar import _call_with_repair
from hgf.forecast_core import _seed


_ROLES = [
    "target_baseline",
    "current_driver",
    "target_period_driver",
    "counterevidence",
    "target_period_magnitude",
    "timing",
    "source_verification",
]


def evidence_selection_schema(
    candidate_ids: list[str],
    *,
    minimum: int,
    maximum: int,
) -> dict[str, Any]:
    return {
        "name": "model_specific_current_evidence_selection",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "query_intents": {
                    "type": "array",
                    "minItems": 2,
                    "maxItems": 6,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "query": {"type": "string", "maxLength": 260},
                            "purpose": {"type": "string", "maxLength": 300},
                            "preferred_source": {
                                "type": "string",
                                "maxLength": 120,
                            },
                        },
                        "required": ["query", "purpose", "preferred_source"],
                    },
                },
                "selected_evidence": {
                    "type": "array",
                    "minItems": minimum,
                    "maxItems": maximum,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "evidence_id": {
                                "type": "string",
                                "enum": candidate_ids,
                            },
                            "roles": {
                                "type": "array",
                                "minItems": 1,
                                "items": {"type": "string", "enum": _ROLES},
                            },
                            "rationale": {"type": "string", "maxLength": 400},
                        },
                        "required": ["evidence_id", "roles", "rationale"],
                    },
                },
                "coverage_gaps": {
                    "type": "array",
                    "maxItems": 6,
                    "items": {"type": "string", "maxLength": 300},
                },
                "selection_summary": {"type": "string", "maxLength": 600},
            },
            "required": [
                "query_intents",
                "selected_evidence",
                "coverage_gaps",
                "selection_summary",
            ],
        },
    }


def validate_evidence_selection(
    payload: dict[str, Any],
    *,
    candidate_ids: set[str],
    minimum: int,
    maximum: int,
) -> tuple[dict[str, float], list[str]]:
    """Reject invalid selection without adding, dropping, or reordering IDs."""
    errors: list[str] = []
    rows = payload.get("selected_evidence")
    if not isinstance(rows, list):
        return {}, ["selected_evidence must be a list"]
    if not minimum <= len(rows) <= maximum:
        errors.append(
            f"selected_evidence count must be {minimum}-{maximum}, got {len(rows)}"
        )
    selected: list[str] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            errors.append(f"selected_evidence[{index}] is not an object")
            continue
        evidence_id = str(row.get("evidence_id") or "")
        selected.append(evidence_id)
        if evidence_id not in candidate_ids:
            errors.append(f"selected_evidence[{index}] has unknown evidence ID")
        roles = row.get("roles")
        if not isinstance(roles, list) or not roles:
            errors.append(f"selected_evidence[{index}] has no role")
        elif any(str(role) not in _ROLES for role in roles):
            errors.append(f"selected_evidence[{index}] has an unknown role")
        if not str(row.get("rationale") or "").strip():
            errors.append(f"selected_evidence[{index}] rationale is empty")
    if len(selected) != len(set(selected)):
        errors.append("selected_evidence contains duplicate IDs")
    intents = payload.get("query_intents")
    if not isinstance(intents, list) or len(intents) < 2:
        errors.append("at least two query intents are required")
    if not str(payload.get("selection_summary") or "").strip():
        errors.append("selection_summary is empty")
    forbidden = {"prediction", "option_probabilities", "forecast", "answer"}
    if forbidden & set(payload):
        errors.append("evidence selection contains a forecast field")
    return {}, errors


def call_model_evidence_selection(
    *,
    client: OpenAI,
    model: str,
    question_id: str,
    search_target: dict[str, Any],
    cutoff: str,
    candidates: list[dict[str, Any]],
    maximum: int = 20,
    max_tokens: int = 4000,
) -> tuple[dict[str, Any], dict[str, int], float, bool]:
    """Let each forecaster model independently select its current evidence."""
    candidate_ids = [str(item["id"]) for item in candidates]
    if not candidate_ids:
        raise ValueError("current evidence candidate pool is empty")
    maximum = min(maximum, len(candidate_ids))
    minimum = min(8, maximum)
    prompt = (
        "Plan current-evidence retrieval for the unresolved financial target and "
        "select the most useful items from the frozen cutoff-safe candidate pool. "
        "This stage is shared by every forecasting method using this model. Do "
        "not forecast, infer an answer, assign a direction, or use historical DAG "
        "memory. Cover the target baseline, target-period driver evidence, genuine "
        "counterevidence, timing, magnitude, and source verification when available. "
        "Prefer dated primary or authoritative sources. Record missing coverage "
        "instead of inventing it. Select each evidence ID at most once.\n\n"
        f"SEARCH TARGET:\n{json.dumps(search_target, ensure_ascii=False)}\n\n"
        f"FORECAST CUTOFF:\n{cutoff}\n\n"
        "FROZEN CUTOFF-SAFE CANDIDATES:\n"
        f"{json.dumps(candidates, ensure_ascii=False)}"
    )
    payload, _, usage, seconds, repaired = _call_with_repair(
        client,
        model=model,
        system=(
            "You are a cutoff-safe financial evidence selector. You never "
            "produce a forecast or probability. Return only valid JSON."
        ),
        prompt=prompt,
        schema=evidence_selection_schema(
            candidate_ids,
            minimum=minimum,
            maximum=maximum,
        ),
        seed=_seed(question_id, "model-specific-evidence-selection-v1"),
        max_tokens=max_tokens,
        validator=lambda value: validate_evidence_selection(
            value,
            candidate_ids=set(candidate_ids),
            minimum=minimum,
            maximum=maximum,
        ),
    )
    return payload, usage, seconds, repaired


def selected_evidence_ids(payload: dict[str, Any]) -> list[str]:
    return [
        str(item["evidence_id"])
        for item in payload.get("selected_evidence") or []
    ]


def load_evidence_selection_manifest(
    path: Path,
    *,
    expected_model: str,
    required_question_ids: list[str],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Load a complete model-specific manifest without cross-model reuse."""
    resolved = path.resolve()
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "model_specific_evidence_manifest_v1":
        raise ValueError("unsupported evidence-selection manifest schema")
    if str(payload.get("model") or "") != expected_model:
        raise ValueError("evidence-selection manifest belongs to another model")
    rows = {
        str(row.get("question_id") or ""): row
        for row in payload.get("results") or []
        if isinstance(row, dict) and row.get("status") == "success"
    }
    missing = sorted(set(required_question_ids) - set(rows))
    if missing:
        raise ValueError(f"evidence-selection manifest misses questions: {missing}")
    return payload, rows


def apply_evidence_selection(
    row: dict[str, Any],
    *,
    db_path: Path,
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Materialize the recorded model order and reject input drift."""
    db_sha256 = hashlib.sha256(db_path.read_bytes()).hexdigest()
    if str(row.get("evidence_db_sha256") or "") != db_sha256:
        raise ValueError("frozen E1 database hash differs from evidence manifest")
    candidate_ids = [str(item.get("id") or "") for item in candidates]
    if row.get("candidate_evidence_ids") != candidate_ids:
        raise ValueError("frozen E1 candidate order differs from evidence manifest")
    selected_ids = [str(value) for value in row.get("selected_evidence_ids") or []]
    if not selected_ids or len(selected_ids) != len(set(selected_ids)):
        raise ValueError("evidence manifest has an empty or duplicate selection")
    by_id = {str(item.get("id") or ""): item for item in candidates}
    unknown = [value for value in selected_ids if value not in by_id]
    if unknown:
        raise ValueError(f"evidence manifest selects unknown IDs: {unknown}")
    return [by_id[value] for value in selected_ids]
