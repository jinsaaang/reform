"""Evidence reranking, JSON repair, and fixed-exemplar forecast contracts."""

from __future__ import annotations

import json
import math
import re
import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from openai import OpenAI

from hgf.forecast_core import (
    _probabilities,
)
from hgf.generation import completion_parameters
from hgf.question_io import family_metadata
from hgf.repair_resilience import (
    conservative_repair_merge,
    serialize_neutral_probabilities,
)

_EVIDENCE_STOPWORDS = {
    "about",
    "after",
    "before",
    "change",
    "ending",
    "financial",
    "fiscal",
    "forecast",
    "from",
    "growth",
    "monthly",
    "quarterly",
    "range",
    "recent",
    "target",
    "that",
    "the",
    "this",
    "which",
    "will",
    "with",
    "year",
}


def _evidence_tokens(value: Any) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]{3,}", str(value or "").lower())
        if token not in _EVIDENCE_STOPWORDS
        and not re.fullmatch(r"(?:19|20)\d{2}", token)
    }


def _rerank_current_evidence(
    question: Any,
    evidence: list[dict[str, Any]],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    """Apply one target-only evidence ranking shared by both forecast arms."""
    metadata = family_metadata(question)
    target_text = " ".join(
        str(metadata.get(field) or "")
        for field in ("entity", "target_metric", "subdomain", "family_id")
    )
    target_tokens = _evidence_tokens(
        f"{question.question_text} {question.context or ''} {target_text}"
    )
    entity_tokens = _evidence_tokens(metadata.get("entity"))
    official_markers = {
        "bls",
        "bureau of labor statistics",
        "eia",
        "energy information administration",
        "federal reserve",
        "fred",
        "sec",
        "investor relations",
    }
    scored: list[tuple[float, str, dict[str, Any]]] = []
    for recency_rank, article in enumerate(evidence):
        title_tokens = _evidence_tokens(article.get("title"))
        excerpt_tokens = _evidence_tokens(article.get("excerpt"))
        overlap = target_tokens & (title_tokens | excerpt_tokens)
        entity_overlap = entity_tokens & (title_tokens | excerpt_tokens)
        source_text = (
            f"{article.get('source') or ''} {article.get('title') or ''}"
        ).lower()
        official = any(marker in source_text for marker in official_markers)
        score = (
            7 * len(target_tokens & title_tokens)
            + 2 * len(overlap)
            + 8 * len(entity_overlap)
            + (4 if official else 0)
            + max(0.0, 2.0 - 0.04 * recency_rank)
        )
        scored.append(
            (score, str(article.get("published_date") or ""), article)
        )
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [item[2] for item in scored[:limit]]


def _usage(response: Any) -> dict[str, int]:
    raw = response.usage.model_dump() if response.usage else {}
    return {
        "prompt_tokens": int(raw.get("prompt_tokens") or 0),
        "completion_tokens": int(raw.get("completion_tokens") or 0),
        "total_tokens": int(raw.get("total_tokens") or 0),
        "call_count": 1,
    }


def _add_usage(*items: dict[str, int]) -> dict[str, int]:
    return {
        key: sum(int(item.get(key, 0)) for item in items)
        for key in (
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "call_count",
        )
    }


def _normalize_probability_rows(
    payload: dict[str, Any],
    options: list[str],
) -> None:
    """Repair only malformed probability serialization and modal labeling."""
    raw_rows = payload.get("option_probabilities") or []
    if not isinstance(raw_rows, list):
        raw_rows = []
    rows = [row for row in raw_rows if isinstance(row, dict)]
    rows = serialize_neutral_probabilities(rows=rows, options=options)
    payload["option_probabilities"] = rows
    by_option: dict[str, float] = {}
    for row in rows:
        option = str(row.get("option"))
        if option not in options:
            continue
        try:
            raw_value = row.get("probability")
            if isinstance(raw_value, str) and raw_value.strip().endswith("%"):
                by_option[option] = float(raw_value.strip()[:-1]) / 100.0
            else:
                by_option[option] = float(raw_value)
        except (TypeError, ValueError):
            return
    if set(by_option) != set(options):
        raise AssertionError("probability serializer did not cover all options")

    values = [by_option[option] for option in options]
    needs_projection = (
        len(rows) != len(options)
        or any(not math.isfinite(value) for value in values)
        or any(value < 0.01 or value > 0.99 for value in values)
        or abs(sum(values) - 1.0) > 0.011
    )
    if needs_projection:
        floor = 0.01
        usable = [
            max(0.0, min(0.99, value) - floor)
            if math.isfinite(value)
            else 0.0
            for value in values
        ]
        usable_total = sum(usable)
        remaining = 1.0 - floor * len(options)
        if usable_total:
            values = [
                floor + remaining * value / usable_total
                for value in usable
            ]
        else:
            values = [1.0 / len(options)] * len(options)
        by_option = dict(zip(options, values, strict=True))

    payload["option_probabilities"] = [
        {"option": option, "probability": by_option[option]}
        for option in options
    ]
    payload["prediction"] = max(
        options,
        key=lambda option: by_option[option],
    )


def _ensure_baseline_reasoning_step(
    payload: dict[str, Any],
    *,
    source_checkpoint_id: str | None = None,
) -> None:
    """Restore a required neutral baseline step without inventing new facts."""
    raw_steps = payload.get("reasoning_steps") or []
    if not isinstance(raw_steps, list):
        raw_steps = [raw_steps]
    steps = [
        step
        if isinstance(step, dict)
        else {
            "step_type": "driver",
            "statement": str(step),
            "evidence_ids": [],
            "effect_on_target": "uncertain",
        }
        for step in raw_steps
        if isinstance(step, dict) or str(step).strip()
    ]
    payload["reasoning_steps"] = steps
    if any(step.get("step_type") == "baseline" for step in steps):
        return
    source = next(
        (
            step
            for step in steps
            if step.get("step_type") in {"driver", "mechanism"}
        ),
        None,
    )
    baseline = {
        "step_type": "baseline",
        "statement": (
            "Baseline reference for the exact target: "
            f"{payload.get('target_semantics', '')}"
        ).strip(),
        "evidence_ids": list(source.get("evidence_ids", [])) if source else [],
        "effect_on_target": "neutral",
    }
    if source_checkpoint_id is not None:
        baseline["source_checkpoint_id"] = source_checkpoint_id
    steps.insert(0, baseline)
    if len(steps) > 7:
        removable = next(
            (
                index
                for index in range(len(steps) - 1, 0, -1)
                if steps[index].get("step_type") == "option_mapping"
            ),
            None,
        )
        if removable is None:
            removable = next(
                (
                    index
                    for index in range(len(steps) - 1, 0, -1)
                    if steps[index].get("step_type") != "target_bridge"
                ),
                len(steps) - 1,
            )
        steps.pop(removable)
    payload["reasoning_steps"] = steps


def _call_json(
    client: OpenAI,
    *,
    model: str,
    system: str,
    prompt: str,
    schema: dict[str, Any],
    seed: int,
    max_tokens: int,
    reasoning_effort_override: str | None = None,
) -> tuple[dict[str, Any], dict[str, int], float]:
    started = time.monotonic()
    response_format = (
        {"type": "json_object"}
        if model.startswith("google/gemini-")
        else {"type": "json_schema", "json_schema": schema}
    )
    user_prompt = prompt
    if model.startswith("google/gemini-"):
        user_prompt = (
            f"{prompt}\n\nReturn exactly one JSON object matching this output "
            "schema. Do not add fields outside the schema:\n"
            f"{json.dumps(schema['schema'], ensure_ascii=False)}"
        )
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_prompt},
        ],
        response_format=response_format,
        seed=seed,
        **completion_parameters(
            model=model,
            stage_max_tokens=max_tokens,
            reasoning_effort_override=reasoning_effort_override,
        ),
    )
    choice = response.choices[0]
    raw = choice.message.content
    if choice.finish_reason == "length" or not raw:
        raise ValueError(
            "invalid/truncated JSON "
            f"finish_reason={choice.finish_reason!r} "
            f"content_present={bool(raw)}"
        )
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"invalid/truncated JSON characters={len(raw)} "
            f"line={exc.lineno} column={exc.colno}"
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError(
            "invalid/truncated JSON top-level object required, got "
            f"{type(payload).__name__}"
        )
    return payload, _usage(response), time.monotonic() - started


def _call_with_repair(
    client: OpenAI,
    *,
    model: str,
    system: str,
    prompt: str,
    schema: dict[str, Any],
    seed: int,
    max_tokens: int,
    validator: Callable[[dict[str, Any]], tuple[dict[str, float], list[str]]],
    fallback_factory: (
        Callable[[dict[str, Any], list[str]], dict[str, Any]] | None
    ) = None,
) -> tuple[dict[str, Any], dict[str, float], dict[str, int], float, bool]:
    parse_retried = False
    try:
        payload, usage, seconds = _call_json(
            client,
            model=model,
            system=system,
            prompt=prompt,
            schema=schema,
            seed=seed,
            max_tokens=max_tokens,
        )
    except ValueError as exc:
        if "invalid/truncated JSON" not in str(exc):
            raise
        parse_retried = True
        try:
            payload, usage, seconds = _call_json(
                client,
                model=model,
                system=system + " Return one complete valid JSON object.",
                prompt=prompt,
                schema=schema,
                seed=seed + 17,
                max_tokens=max_tokens,
                reasoning_effort_override=None,
            )
        except ValueError as retry_exc:
            if (
                fallback_factory is None
                or "invalid/truncated JSON" not in str(retry_exc)
            ):
                raise
            payload = fallback_factory({}, [str(retry_exc)])
            probabilities, errors = validator(payload)
            if errors:
                raise ValueError(
                    "invalid fallback after parse retry: "
                    + "; ".join(errors)
                ) from retry_exc
            return payload, probabilities, {}, 0.0, True
    probabilities, errors = validator(payload)
    if not errors:
        return payload, probabilities, usage, seconds, parse_retried
    current = payload
    total_usage = usage
    total_seconds = seconds
    for attempt in range(4):
        repaired_prompt = (
            f"{prompt}\n\nRepair the following output only to satisfy validation. "
            "Do not add facts or evidence IDs. Preserve substantive reasoning, "
            "prediction, and probabilities unless an error explicitly requires "
            "a consistency change.\nERRORS:\n"
            f"{json.dumps(errors)}\nOUTPUT:\n"
            f"{json.dumps(current, ensure_ascii=False)}"
        )
        try:
            repaired, repair_usage, repair_seconds = _call_json(
                client,
                model=model,
                system=system,
                prompt=repaired_prompt,
                schema=schema,
                seed=seed + attempt + 1,
                max_tokens=max_tokens,
                reasoning_effort_override=None,
            )
        except ValueError as exc:
            if "invalid/truncated JSON" in str(exc):
                errors = [str(exc)]
                continue
            raise
        repaired = conservative_repair_merge(
            original=current,
            repaired=repaired,
        )
        total_usage = _add_usage(total_usage, repair_usage)
        total_seconds += repair_seconds
        probabilities, errors = validator(repaired)
        if not errors:
            return (
                repaired,
                probabilities,
                total_usage,
                total_seconds,
                True,
            )
        current = repaired
    if fallback_factory is not None:
        fallback = fallback_factory(current, errors)
        probabilities, fallback_errors = validator(fallback)
        if not fallback_errors:
            return (
                fallback,
                probabilities,
                total_usage,
                total_seconds,
                True,
            )
        errors = fallback_errors
    raise ValueError("invalid after repair: " + "; ".join(errors))


def _parse_date(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _exemplar_article_ids(graph: dict[str, Any], cutoff: datetime) -> set[str]:
    return {
        str(item.get("id"))
        for item in graph.get("evidence", {}).get("articles", [])
        if (published := _parse_date(item.get("published_date")))
        and published < cutoff
    }


def _forecast_schema_exemplar(
    options: list[str],
    transfer_policy: str,
) -> dict[str, Any]:
    payload = {
        "name": "worked_exemplar_forecast",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "target_semantics": {"type": "string", "minLength": 1},
                "selected_evidence_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "evidence_fit": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "metric_match": {
                            "type": "string",
                            "enum": ["direct", "partial", "weak"],
                        },
                        "horizon_match": {
                            "type": "string",
                            "enum": ["direct", "partial", "weak"],
                        },
                        "magnitude_support": {
                            "type": "string",
                            "enum": ["supported", "partial", "unsupported"],
                        },
                        "assessment": {"type": "string", "minLength": 1},
                    },
                    "required": [
                        "metric_match",
                        "horizon_match",
                        "magnitude_support",
                        "assessment",
                    ],
                },
                "current_evidence_only_estimate": {"type": "string"},
                "current_evidence_only_prediction": {
                    "type": "string",
                    "enum": options,
                },
                "transfer_check": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "transformation_supported": {"type": "boolean"},
                        "horizon_supported": {"type": "boolean"},
                        "bridge_supported": {"type": "boolean"},
                        "revision_evidence_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "assessment": {"type": "string"},
                    },
                    "required": [
                        "transformation_supported",
                        "horizon_supported",
                        "bridge_supported",
                        "revision_evidence_ids",
                        "assessment",
                    ],
                },
                "transfer_verdict": {
                    "type": "string",
                    "enum": ["CONFIRM", "REVISE", "AUDIT_ONLY"],
                },
                "revision_reason": {"type": "string"},
                "reasoning_steps": {
                    "type": "array",
                    "minItems": 3,
                    "maxItems": 7,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "step_type": {
                                "type": "string",
                                "enum": [
                                    "baseline",
                                    "driver",
                                    "mechanism",
                                    "counterevidence",
                                    "target_bridge",
                                    "option_mapping",
                                ],
                            },
                            "statement": {"type": "string", "minLength": 1},
                            "evidence_ids": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "effect_on_target": {
                                "type": "string",
                                "enum": [
                                    "up",
                                    "down",
                                    "neutral",
                                    "mixed",
                                    "uncertain",
                                ],
                            },
                        },
                        "required": [
                            "step_type",
                            "statement",
                            "evidence_ids",
                            "effect_on_target",
                        ],
                    },
                },
                "counterevidence": {"type": "string", "minLength": 1},
                "target_estimate": {"type": "string", "minLength": 1},
                "option_mapping": {"type": "string", "minLength": 1},
                "prediction": {"type": "string", "enum": options},
                "option_probabilities": {
                    "type": "array",
                    "minItems": len(options),
                    "maxItems": len(options),
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "option": {"type": "string", "enum": options},
                            "probability": {"type": "number"},
                        },
                        "required": ["option", "probability"],
                    },
                },
                "uncertainty": {"type": "string", "minLength": 1},
            },
            "required": [
                "target_semantics",
                "selected_evidence_ids",
                "evidence_fit",
                "current_evidence_only_estimate",
                "current_evidence_only_prediction",
                "transfer_check",
                "transfer_verdict",
                "revision_reason",
                "reasoning_steps",
                "counterevidence",
                "target_estimate",
                "option_mapping",
                "prediction",
                "option_probabilities",
                "uncertainty",
            ],
        },
    }
    if transfer_policy == "none":
        gated_fields = {
            "current_evidence_only_estimate",
            "current_evidence_only_prediction",
            "transfer_check",
            "transfer_verdict",
            "revision_reason",
        }
        properties = payload["schema"]["properties"]
        for field in gated_fields:
            properties.pop(field, None)
        payload["schema"]["required"] = [
            field
            for field in payload["schema"]["required"]
            if field not in gated_fields
        ]
    return payload


def _validate_exemplar_forecast(
    payload: dict[str, Any],
    *,
    options: list[str],
    evidence_ids: set[str],
    transfer_policy: str,
) -> tuple[dict[str, float], list[str]]:
    probabilities, errors = _probabilities(payload, options)
    raw_selected = payload.get("selected_evidence_ids") or []
    if not isinstance(raw_selected, (list, tuple, set)):
        raw_selected = []
    used = set(raw_selected)
    if not used:
        errors.append("selected_evidence_ids is empty")
    unknown = used - evidence_ids
    if unknown:
        errors.append(f"unknown current evidence IDs {sorted(unknown)}")
    if not str(payload.get("target_estimate") or "").strip():
        errors.append("target_estimate is empty")
    evidence_fit = payload.get("evidence_fit", {})
    if not isinstance(evidence_fit, dict):
        evidence_fit = {"assessment": str(evidence_fit)}
        payload["evidence_fit"] = evidence_fit
    if not str(evidence_fit.get("assessment") or "").strip():
        errors.append("evidence_fit assessment is empty")
    if transfer_policy != "none":
        transfer = payload.get("transfer_check", {})
        revision_ids = {
            str(article_id)
            for article_id in transfer.get("revision_evidence_ids", [])
        }
        unknown_revision_ids = revision_ids - evidence_ids
        if unknown_revision_ids:
            errors.append(
                "transfer check uses unknown evidence IDs "
                f"{sorted(unknown_revision_ids)}"
            )
        verdict = str(payload.get("transfer_verdict") or "")
        initial_prediction = str(
            payload.get("current_evidence_only_prediction") or ""
        )
        final_prediction = str(payload.get("prediction") or "")
        if verdict in {"CONFIRM", "AUDIT_ONLY"}:
            if final_prediction != initial_prediction:
                errors.append(
                    f"{verdict} must preserve current_evidence_only_prediction"
                )
        elif verdict == "REVISE":
            transformation = bool(
                transfer.get("transformation_supported")
            )
            horizon = bool(transfer.get("horizon_supported"))
            bridge = bool(transfer.get("bridge_supported"))
            if transfer_policy == "strict":
                qualified = (
                    transformation
                    and horizon
                    and bridge
                    and len(revision_ids) >= 2
                )
            else:
                qualified = (
                    bridge
                    and (transformation or horizon)
                    and bool(revision_ids)
                )
            if not qualified:
                errors.append(
                    f"REVISE does not satisfy {transfer_policy} transfer gate"
                )
    steps = payload.get("reasoning_steps", [])
    step_ids = {
        str(article_id)
        for step in steps
        for article_id in step.get("evidence_ids", [])
    }
    unknown_step_ids = step_ids - evidence_ids
    if unknown_step_ids:
        errors.append(
            f"reasoning steps use unknown evidence IDs {sorted(unknown_step_ids)}"
        )
    step_types = {str(step.get("step_type")) for step in steps}
    if transfer_policy == "none":
        for required in ("baseline", "target_bridge"):
            if required not in step_types:
                errors.append(f"reasoning_steps missing {required}")
    for field in (
        "target_semantics",
        "counterevidence",
        "option_mapping",
        "uncertainty",
    ):
        if not str(payload.get(field) or "").strip():
            errors.append(f"{field} is empty")
    prediction = str(payload.get("prediction") or "")
    if probabilities:
        max_probability = max(probabilities.values())
        if (
            prediction not in probabilities
            or probabilities[prediction] < max_probability - 1e-9
        ):
            errors.append(
                "prediction must equal the highest-probability option"
            )
    return probabilities, errors
