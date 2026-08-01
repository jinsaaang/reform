"""Generate the cutoff-safe worked exemplars used by canonical HGF.

The generator uses only the historical question's cutoff-safe evidence and
the outcome-redacted canonical HGF Blueprint.
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
    _add_usage,
    _call_json,
    _exemplar_article_ids,
    _transferable_dag_structure,
)
from hgf.forecast_core import _atomic_write, _seed
from hgf.question_io import resolve_forecast_cutoff


_EXEMPLAR_LOCK = threading.Lock()


def _exemplar_schema() -> dict[str, Any]:
    return {
        "name": "dag_derived_forecast_exemplar",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "task_signature": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "category": {"type": "string"},
                        "target_operation": {"type": "string"},
                        "option_geometry": {"type": "string"},
                    },
                    "required": [
                        "category",
                        "target_operation",
                        "option_geometry",
                    ],
                },
                "historical_question": {"type": "string"},
                "historical_cutoff": {"type": "string"},
                "target_semantics": {"type": "string"},
                "forecast_time_evidence": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "article_id": {"type": "string"},
                            "takeaway": {"type": "string"},
                            "why_predictive": {"type": "string"},
                        },
                        "required": [
                            "article_id",
                            "takeaway",
                            "why_predictive",
                        ],
                    },
                },
                "expert_reasoning": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "counterevidence": {"type": "string"},
                "prospective_target_estimate": {"type": "string"},
                "option_mapping": {"type": "string"},
                "uncertainty": {"type": "string"},
                "dag_derived_lesson": {"type": "string"},
            },
            "required": [
                "task_signature",
                "historical_question",
                "historical_cutoff",
                "target_semantics",
                "forecast_time_evidence",
                "expert_reasoning",
                "counterevidence",
                "prospective_target_estimate",
                "option_mapping",
                "uncertainty",
                "dag_derived_lesson",
            ],
        },
    }


def _validate_exemplar(
    exemplar: dict[str, Any],
    *,
    allowed_article_ids: set[str],
) -> list[str]:
    errors: list[str] = []
    used = {
        str(item.get("article_id"))
        for item in exemplar.get("forecast_time_evidence", [])
    }
    unknown = used - allowed_article_ids
    if unknown:
        errors.append(
            f"exemplar uses post-cutoff/unknown evidence {sorted(unknown)}"
        )
    if len(exemplar.get("expert_reasoning", [])) < 3:
        errors.append("expert_reasoning must have at least 3 steps")
    if not str(exemplar.get("target_semantics") or "").strip():
        errors.append("target_semantics is empty")
    rendered = json.dumps(exemplar, ensure_ascii=False)
    cited = set(re.findall(r"\bart_[a-zA-Z0-9_]+\b", rendered))
    unknown_citations = cited - allowed_article_ids
    if unknown_citations:
        errors.append(
            "exemplar text cites post-cutoff/unknown evidence "
            f"{sorted(unknown_citations)}"
        )
    for field in ("counterevidence", "uncertainty", "dag_derived_lesson"):
        if not str(exemplar.get(field) or "").strip():
            errors.append(f"{field} is empty")
    return errors


def _normalize_required_fields(
    exemplar: dict[str, Any],
    *,
    historical_case: dict[str, Any],
) -> dict[str, Any]:
    """Fill only missing procedural fields without adding historical facts."""
    payload = dict(exemplar)
    payload.setdefault(
        "task_signature",
        {
            "category": "general_forecasting",
            "target_operation": "apply the stated resolution criteria",
            "option_geometry": "binary option space",
        },
    )
    payload.setdefault("historical_question", historical_case["question"])
    payload.setdefault("historical_cutoff", historical_case["cutoff"])
    if not str(payload.get("target_semantics") or "").strip():
        payload["target_semantics"] = (
            "Apply the question's stated target, horizon, and resolution "
            "criteria using only cutoff-safe information."
        )
    if not isinstance(payload.get("forecast_time_evidence"), list):
        payload["forecast_time_evidence"] = []
    reasoning = [
        str(value).strip()
        for value in payload.get("expert_reasoning", [])
        if str(value).strip()
    ]
    procedural_steps = [
        "Define the exact target, forecast horizon, and option mapping.",
        "Evaluate only cutoff-safe evidence for supported drivers and counterevidence.",
        "Map the prospective estimate to options while preserving uncertainty.",
    ]
    for step in procedural_steps:
        if len(reasoning) >= 3:
            break
        if step not in reasoning:
            reasoning.append(step)
    payload["expert_reasoning"] = reasoning
    if not str(payload.get("counterevidence") or "").strip():
        payload["counterevidence"] = (
            "The supplied cutoff-safe evidence does not establish a specific "
            "countervailing mechanism; preserve uncertainty."
        )
    if not str(payload.get("prospective_target_estimate") or "").strip():
        payload["prospective_target_estimate"] = (
            "The evidence is insufficient for a narrow point estimate."
        )
    if not str(payload.get("option_mapping") or "").strip():
        payload["option_mapping"] = (
            "Map the prospective assessment to the stated option definitions."
        )
    if not str(payload.get("uncertainty") or "").strip():
        payload["uncertainty"] = (
            "Evidence is incomplete, so retain broad forecast uncertainty."
        )
    if not str(payload.get("dag_derived_lesson") or "").strip():
        payload["dag_derived_lesson"] = (
            "Verify each causal link with cutoff-safe evidence and preserve "
            "uncertainty when support is absent."
        )
    return payload


def _distill_exemplar(
    *,
    client: OpenAI,
    model: str,
    memory_question: Any,
    graph: dict[str, Any],
    blueprint: dict[str, Any],
    cache_dir: Path,
    max_tokens: int,
) -> tuple[dict[str, Any], dict[str, int], float, bool]:
    """Distill one historical DAG into a cutoff-safe worked exemplar."""
    cache_path = cache_dir / f"{memory_question.id}.json"
    historical_cutoff, _ = resolve_forecast_cutoff(memory_question)
    articles = graph.get("evidence", {}).get("articles", [])
    allowed_ids = _exemplar_article_ids(graph, historical_cutoff)
    with _EXEMPLAR_LOCK:
        if cache_path.exists():
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            if not _validate_exemplar(
                payload,
                allowed_article_ids=allowed_ids,
            ):
                return payload, {}, 0.0, True
    forecast_time_articles = [
        item for item in articles if str(item.get("id")) in allowed_ids
    ]
    historical_case = {
        "question": memory_question.question_text,
        "context": memory_question.context,
        "options": [str(option) for option in memory_question.options or []],
        "target_contract": _target_contract(memory_question),
        "cutoff": historical_cutoff.isoformat(),
    }
    transferable_structure = _transferable_dag_structure(blueprint)
    prompt = (
        "Distill this resolved WorldReasoner hindsight DAG into one short worked "
        "example of expert reasoning AS IT COULD HAVE BEEN PERFORMED AT THAT "
        "HISTORICAL CUTOFF. The DAG has already been reduced to transferable "
        "causal roles; no outcome facts or post-cutoff events are provided. Use "
        "the DAG-derived structure to decide which causal connections to test, "
        "but make every historical factual claim from the supplied forecast-time "
        "articles only. Do not reconstruct or guess the resolved outcome. "
        "forecast_time_evidence and all prose may cite ONLY allowed article IDs. "
        "Write a concise sequence: target semantics, baseline, supported drivers, "
        "evidence-to-target mechanism, genuine counterevidence, prospective "
        "estimate, option mapping, and uncertainty. dag_derived_lesson must be an "
        "abstract reasoning lesson without entity names, dates, answer labels, or "
        "historical outcomes. If evidence is weak, preserve uncertainty rather "
        "than inventing support. expert_reasoning must contain at least three "
        "nonempty reasoning steps. counterevidence, uncertainty, and "
        "dag_derived_lesson must each be nonempty. Do not mention the future "
        "current case.\n\n"
        "HISTORICAL CASE WITHOUT OUTCOME:\n"
        f"{json.dumps(historical_case, ensure_ascii=False)}\n\n"
        f"ALLOWED FORECAST-TIME ARTICLE IDS:\n{json.dumps(sorted(allowed_ids))}\n\n"
        "FORECAST-TIME ARTICLES ONLY (all strictly before the historical cutoff):\n"
        f"{json.dumps(forecast_time_articles, ensure_ascii=False)}\n\n"
        "OUTCOME-REDACTED STRUCTURE EXTRACTED FROM THE VALIDATED HINDSIGHT DAG:\n"
        f"{json.dumps(transferable_structure, ensure_ascii=False)}"
    )
    system = (
        "You are an expert forecast-example distiller. Keep forecast-time "
        "evidence and hindsight audit strictly separated."
    )
    seed = _seed(str(memory_question.id), "dag-exemplar")
    try:
        exemplar, usage, seconds = _call_json(
            client,
            model=model,
            system=system,
            prompt=prompt,
            schema=_exemplar_schema(),
            seed=seed,
            max_tokens=max_tokens,
        )
    except ValueError as exc:
        if "invalid/truncated JSON" not in str(exc):
            raise
        exemplar, usage, seconds = _call_json(
            client,
            model=model,
            system=system + " Return one complete valid JSON object.",
            prompt=prompt,
            schema=_exemplar_schema(),
            seed=seed + 17,
            max_tokens=max_tokens,
        )
    exemplar = _normalize_required_fields(
        exemplar,
        historical_case=historical_case,
    )
    errors = _validate_exemplar(
        exemplar,
        allowed_article_ids=allowed_ids,
    )
    if errors:
        repair_prompt = (
            f"{prompt}\n\nRepair the following exemplar once. Fix every "
            "validation error without adding facts, article IDs, outcome "
            "labels, or post-cutoff information.\n\n"
            f"ERRORS:\n{json.dumps(errors)}\n\n"
            f"INVALID EXEMPLAR:\n{json.dumps(exemplar, ensure_ascii=False)}"
        )
        repaired, repair_usage, repair_seconds = _call_json(
            client,
            model=model,
            system=system + " Repair the structure only.",
            prompt=repair_prompt,
            schema=_exemplar_schema(),
            seed=seed + 1,
            max_tokens=max_tokens,
        )
        repaired = _normalize_required_fields(
            repaired,
            historical_case=historical_case,
        )
        repair_errors = _validate_exemplar(
            repaired,
            allowed_article_ids=allowed_ids,
        )
        if repair_errors:
            raise ValueError(
                "invalid DAG exemplar after repair: "
                + "; ".join(repair_errors)
            )
        exemplar = repaired
        usage = _add_usage(usage, repair_usage)
        seconds += repair_seconds
    with _EXEMPLAR_LOCK:
        _atomic_write(cache_path, exemplar)
    return exemplar, usage, seconds, False
