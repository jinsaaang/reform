#!/usr/bin/env python3
"""Principle Memory baseline primitives on frozen cutoff-safe evidence.

One resolved historical event is distilled into answer-free forecasting
principles in ordinary prose. The distiller may read the resolved outcome to
decide what mattered, but the stored memory must not reveal the answer label,
resolved direction, or resolved value, and must not mention a graph, nodes,
edges, paths, checkpoints, or DAGs.

The schema and seed strings below keep the names the registered runs used, so
a replay reproduces the same requests and seeds.
"""

from __future__ import annotations

import json
import re
import threading
from pathlib import Path
from typing import Any

from openai import OpenAI

from hgf.contracts import _target_contract
from hgf.exemplar import (
    _call_with_repair,
    _exemplar_article_ids,
)
from hgf.forecast_core import _atomic_write, _seed
from hgf.question_io import resolve_forecast_cutoff

_WRITE_LOCK = threading.Lock()


def _principle_memory_schema() -> dict[str, Any]:
    return {
        "name": "plain_text_hindsight_memory",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "historical_case_summary": {"type": "string"},
                "evidence_selection_lesson": {"type": "string"},
                "plain_text_reasoning_lesson": {"type": "string"},
                "counterevidence_lesson": {"type": "string"},
                "calibration_lesson": {"type": "string"},
                "outcome_redacted": {"type": "boolean"},
            },
            "required": [
                "historical_case_summary",
                "evidence_selection_lesson",
                "plain_text_reasoning_lesson",
                "counterevidence_lesson",
                "calibration_lesson",
                "outcome_redacted",
            ],
        },
    }


def _validate_principle_memory(
    payload: dict[str, Any],
    *,
    ground_truth: str,
) -> tuple[dict[str, float], list[str]]:
    errors: list[str] = []
    answer = ground_truth.strip()
    if len(answer) > 3:
        pattern = re.compile(re.escape(answer), flags=re.IGNORECASE)

        def redact(value: Any) -> Any:
            if isinstance(value, str):
                return pattern.sub("[resolved option redacted]", value)
            if isinstance(value, list):
                return [redact(item) for item in value]
            if isinstance(value, dict):
                return {key: redact(item) for key, item in value.items()}
            return value

        redacted = redact(payload)
        payload.clear()
        payload.update(redacted)
    if payload.get("outcome_redacted") is not True:
        errors.append("outcome_redacted must be true")
    for field in (
        "historical_case_summary",
        "evidence_selection_lesson",
        "plain_text_reasoning_lesson",
        "counterevidence_lesson",
        "calibration_lesson",
    ):
        if not str(payload.get(field) or "").strip():
            errors.append(f"{field} is empty")
    rendered = json.dumps(payload, ensure_ascii=False).lower()
    if re.search(r"\b(?:dag|directed acyclic graph)\b", rendered):
        errors.append("plain text memory mentions a DAG")
    return {}, errors


def _distill_principle_memory(
    *,
    client: OpenAI,
    model: str,
    memory_question: Any,
    memory_graph: dict[str, Any],
    cache_dir: Path,
    max_tokens: int,
) -> dict[str, Any]:
    cache_path = cache_dir / f"{memory_question.id}.json"
    if cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        _, errors = _validate_principle_memory(
            cached,
            ground_truth=str(memory_question.ground_truth),
        )
        if not errors:
            return {
                "status": "success",
                "question_id": str(memory_question.id),
                "memory": cached,
                "cached": True,
                "usage": {},
                "seconds": 0.0,
            }

    historical_cutoff, _ = resolve_forecast_cutoff(memory_question)
    allowed_ids = _exemplar_article_ids(memory_graph, historical_cutoff)
    articles = [
        article
        for article in memory_graph.get("evidence", {}).get("articles", [])
        if str(article.get("id")) in allowed_ids
    ][:20]
    resolved_case = {
        "question": memory_question.question_text,
        "context": memory_question.context,
        "target_contract": _target_contract(memory_question),
        "historical_cutoff": historical_cutoff.isoformat(),
        "resolved_answer_for_distillation_only": memory_question.ground_truth,
        "resolution_reasoning_for_distillation_only": (
            memory_question.resolution_reasoning
        ),
    }
    prompt = (
        "Create one plain-text hindsight memory from this resolved historical "
        "forecasting case. This is the non-graph memory baseline. You may use the "
        "resolved outcome internally to identify what mattered, but the output "
        "must not reveal the answer label, resolved direction, resolved value, "
        "or exact post-resolution fact. Do not use or mention a graph, nodes, "
        "edges, causal paths, checkpoints, or DAGs. Do not invent article facts. "
        "Explain in ordinary prose what evidence should be prioritized, what "
        "reasoning pattern is transferable, what counterevidence matters, and "
        "how uncertainty should be calibrated. The memory may describe the "
        "historical target type, but it must remain safe to use as an analogy "
        "for a new unresolved case.\n\n"
        f"RESOLVED CASE (DISTILLER ONLY):\n"
        f"{json.dumps(resolved_case, ensure_ascii=False)}\n\n"
        "ARTICLES AVAILABLE AT THE HISTORICAL FORECAST CUTOFF:\n"
        f"{json.dumps(articles, ensure_ascii=False)}"
    )
    def validator(payload: dict[str, Any]) -> tuple[dict[str, float], list[str]]:
        return _validate_principle_memory(
            payload,
            ground_truth=str(memory_question.ground_truth),
        )
    memory, _, usage, seconds, repaired = _call_with_repair(
        client,
        model=model,
        system=(
            "You distill outcome-redacted historical forecasting lessons into "
            "ordinary text. Return only schema-conforming JSON."
        ),
        prompt=prompt,
        schema=_principle_memory_schema(),
        seed=_seed(str(memory_question.id), "plain-text-hindsight-memory"),
        max_tokens=max_tokens,
        validator=validator,
    )
    with _WRITE_LOCK:
        _atomic_write(cache_path, memory)
    return {
        "status": "success",
        "question_id": str(memory_question.id),
        "memory": memory,
        "cached": False,
        "usage": usage,
        "seconds": seconds,
        "repaired": repaired,
    }
