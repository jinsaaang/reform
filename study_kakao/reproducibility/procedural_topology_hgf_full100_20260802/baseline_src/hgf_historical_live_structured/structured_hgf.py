"""Evidence-first instantiation of hindsight-derived reasoning structures."""

from __future__ import annotations

import copy
import json
import re
from typing import Any

from openai import OpenAI

from .exemplar import _call_with_repair
from .forecast_core import _seed


_ROUTING_STOPWORDS = {
    "about",
    "after",
    "before",
    "change",
    "conditions",
    "current",
    "evidence",
    "factor",
    "financial",
    "forecast",
    "from",
    "growth",
    "historical",
    "monthly",
    "recent",
    "target",
    "that",
    "the",
    "this",
    "with",
}


def _tokens(value: Any) -> set[str]:
    return {
        token
        for token in re.findall(
            r"[a-z0-9]{3,}",
            json.dumps(value, ensure_ascii=False).lower(),
        )
        if token not in _ROUTING_STOPWORDS
        and not re.fullmatch(r"(?:19|20)\d{2}", token)
    }


def attach_worked_reasoning_demonstration(
    memory: dict[str, Any],
    worked_exemplar: dict[str, Any],
) -> dict[str, Any]:
    """Attach the pre-cutoff procedure while excluding its proposed answer."""
    result = copy.deepcopy(memory)
    result["worked_reasoning_demonstration"] = {
        "task_signature": copy.deepcopy(
            worked_exemplar.get("task_signature", {})
        ),
        "reasoning_sequence": [
            str(value)
            for value in worked_exemplar.get("expert_reasoning", [])[:7]
        ],
        "counterevidence": str(
            worked_exemplar.get("counterevidence") or ""
        ),
        "uncertainty": str(worked_exemplar.get("uncertainty") or ""),
        "structural_lesson": str(
            worked_exemplar.get("dag_derived_lesson") or ""
        ),
    }
    result["transfer_rule"] = (
        "The worked case demonstrates how an expert organized evidence before "
        "that event resolved. Its proposed estimate, option mapping, final "
        "answer, probabilities, and realized outcome are unavailable. Reuse "
        "only its reasoning order. Instantiate every DAG checkpoint with "
        "current cutoff-safe evidence before activating a conditional path."
    )
    return result


def _evidence_ledger_schema() -> dict[str, Any]:
    return {
        "name": "current_financial_evidence_ledger",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "target_baseline": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "status": {
                            "type": "string",
                            "enum": ["observed", "partial", "unavailable"],
                        },
                        "observation": {
                            "type": "string",
                            "maxLength": 600,
                        },
                        "evidence_ids": {
                            "type": "array",
                            "maxItems": 6,
                            "items": {"type": "string"},
                        },
                        "assessment": {
                            "type": "string",
                            "maxLength": 600,
                        },
                    },
                    "required": [
                        "status",
                        "observation",
                        "evidence_ids",
                        "assessment",
                    ],
                },
                "current_signals": {
                    "type": "array",
                    "minItems": 2,
                    "maxItems": 8,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "signal_id": {
                                "type": "string",
                                "maxLength": 40,
                            },
                            "factor": {
                                "type": "string",
                                "maxLength": 180,
                            },
                            "state": {
                                "type": "string",
                                "enum": [
                                    "strengthening",
                                    "weakening",
                                    "elevated",
                                    "depressed",
                                    "stable",
                                    "mixed",
                                    "unknown",
                                ],
                            },
                            "temporal_role": {
                                "type": "string",
                                "enum": [
                                    "target_period",
                                    "leading",
                                    "lagging",
                                    "structural",
                                ],
                            },
                            "observation": {
                                "type": "string",
                                "maxLength": 600,
                            },
                            "evidence_ids": {
                                "type": "array",
                                "minItems": 1,
                                "maxItems": 6,
                                "items": {"type": "string"},
                            },
                        },
                        "required": [
                            "signal_id",
                            "factor",
                            "state",
                            "temporal_role",
                            "observation",
                            "evidence_ids",
                        ],
                    },
                },
                "data_gaps": {
                    "type": "array",
                    "maxItems": 4,
                    "items": {"type": "string", "maxLength": 300},
                },
                "ledger_summary": {
                    "type": "string",
                    "maxLength": 700,
                },
            },
            "required": [
                "target_baseline",
                "current_signals",
                "data_gaps",
                "ledger_summary",
            ],
        },
    }


def _validate_evidence_ledger(
    payload: dict[str, Any],
    evidence_ids: set[str],
) -> tuple[dict[str, float], list[str]]:
    errors: list[str] = []
    baseline = payload.get("target_baseline", {})
    baseline_ids = {
        str(value)
        for value in baseline.get("evidence_ids", [])
        if str(value) in evidence_ids
    }
    baseline["evidence_ids"] = sorted(baseline_ids)
    if baseline.get("status") == "observed" and not baseline_ids:
        baseline["status"] = "unavailable"
    signals = payload.get("current_signals", [])
    kept_signals = []
    seen_signal_ids: set[str] = set()
    for item in signals:
        signal_id = str(item.get("signal_id") or "")
        if not signal_id or signal_id in seen_signal_ids:
            continue
        used = {
            str(value)
            for value in item.get("evidence_ids", [])
            if str(value) in evidence_ids
        }
        if not used:
            continue
        item["evidence_ids"] = sorted(used)
        kept_signals.append(item)
        seen_signal_ids.add(signal_id)
    payload["current_signals"] = kept_signals
    if not kept_signals:
        errors.append("current evidence ledger has no grounded signal")
    forbidden = {
        "prediction",
        "probabilities",
        "option_probabilities",
        "forecast",
    }
    if forbidden & set(payload):
        errors.append("evidence ledger contains a forecast output")
    return {}, errors


def _deterministic_evidence_ledger(
    evidence: list[dict[str, Any]],
) -> dict[str, Any]:
    """Preserve raw current evidence when provider ledger JSON is unusable."""
    signals = []
    for index, item in enumerate(evidence[:8], start=1):
        evidence_id = str(item.get("id") or "")
        if not evidence_id:
            continue
        factor = str(
            item.get("title")
            or item.get("source")
            or item.get("name")
            or f"current evidence {index}"
        )[:180]
        observation = str(
            item.get("snippet")
            or item.get("summary")
            or item.get("text")
            or item.get("content")
            or factor
        )[:600]
        signals.append(
            {
                "signal_id": f"fallback_signal_{index}",
                "factor": factor,
                "state": "unknown",
                "temporal_role": "structural",
                "observation": observation,
                "evidence_ids": [evidence_id],
            }
        )
    if not signals:
        raise ValueError("deterministic evidence ledger has no evidence")
    return {
        "target_baseline": {
            "status": "unavailable",
            "observation": "No target baseline was inferred.",
            "evidence_ids": [],
            "assessment": (
                "Provider ledger generation was unusable, so no baseline "
                "claim is made."
            ),
        },
        "current_signals": signals,
        "data_gaps": [
            "Signal states and temporal roles were not inferred in fallback."
        ],
        "ledger_summary": (
            "Raw cutoff-safe evidence is preserved without inferred direction, "
            "magnitude, answer, or probability."
        ),
    }


def call_current_evidence_ledger(
    *,
    client: OpenAI,
    model: str,
    question_id: str,
    public_case: dict[str, Any],
    target_operator: dict[str, Any],
    evidence: list[dict[str, Any]],
    max_tokens: int,
    allow_deterministic_fallback: bool = True,
) -> tuple[dict[str, Any], dict[str, int], float, bool]:
    evidence_ids = {str(item["id"]) for item in evidence}
    prompt = (
        "Read the current cutoff-safe evidence before any historical memory is "
        "shown. Build a compact factual ledger for the exact financial target. "
        "Record the latest usable target baseline, current drivers, their "
        "temporal relationship to the target period, and missing information. "
        "Do not forecast, choose an option, assign probabilities, or infer what "
        "a past event implies. Distinguish a level from a change, a general "
        "outlook from the target period, and a leading signal from a direct "
        "target observation. Use only supplied evidence IDs. Keep each field "
        "under 80 words.\n\n"
        f"CURRENT CASE:\n{json.dumps(public_case, ensure_ascii=False)}\n\n"
        "DETERMINISTIC TARGET OPERATOR:\n"
        f"{json.dumps(target_operator, ensure_ascii=False)}\n\n"
        "CURRENT CUTOFF-SAFE EVIDENCE:\n"
        f"{json.dumps(evidence, ensure_ascii=False)}"
    )
    call_args = {
        "model": model,
        "system": (
            "You extract a cutoff-safe financial evidence ledger. You never "
            "produce a forecast, answer, or probability. Return JSON."
        ),
        "prompt": prompt,
        "schema": _evidence_ledger_schema(),
        "seed": _seed(question_id, "structured-hgf-evidence-ledger"),
        "max_tokens": min(max_tokens, 6000),
        "validator": lambda payload: _validate_evidence_ledger(
            payload,
            evidence_ids,
        ),
    }
    try:
        ledger, _, usage, seconds, repaired = _call_with_repair(
            client,
            **call_args,
        )
    except ValueError as exc:
        if "invalid/truncated JSON" not in str(exc):
            raise
        if not allow_deterministic_fallback:
            raise
        fallback_args = dict(call_args)
        fallback_args["seed"] = int(call_args["seed"]) + 7919
        fallback_args["max_tokens"] = min(
            int(call_args["max_tokens"]),
            4000,
        )
        try:
            ledger, _, usage, seconds, repaired = _call_with_repair(
                client,
                **fallback_args,
                reasoning_effort="low",
            )
            repaired = True
        except ValueError as fallback_exc:
            if "invalid/truncated JSON" not in str(fallback_exc):
                raise
            ledger = _deterministic_evidence_ledger(evidence)
            usage = {}
            seconds = 0.0
            repaired = True
    return ledger, usage, seconds, repaired


def route_structured_memory(
    memory: dict[str, Any],
    evidence_ledger: dict[str, Any],
    *,
    max_paths: int = 4,
    max_factors: int = 9,
) -> dict[str, Any]:
    """Route a bounded set of complete paths after seeing current signals."""
    result = copy.deepcopy(memory)
    factors = {
        str(item.get("id")): item
        for item in result.get("factor_checks", [])
        if item.get("id")
    }
    ledger_tokens = _tokens(evidence_ledger)
    ranked_paths = []
    for index, path in enumerate(result.get("conditional_paths", [])):
        path_factors = [
            factors[str(factor_id)]
            for factor_id in path.get("factor_ids", [])
            if str(factor_id) in factors
        ]
        score = len(
            ledger_tokens
            & _tokens({"path": path, "factors": path_factors})
        )
        ranked_paths.append((score, -index, path))
    ranked_paths.sort(reverse=True, key=lambda item: (item[0], item[1]))
    selected_paths = []
    selected_factor_ids: list[str] = []
    for _, _, path in ranked_paths:
        path_factor_ids = [
            str(value)
            for value in path.get("factor_ids", [])
            if str(value) in factors
        ]
        added_ids = [
            factor_id
            for factor_id in path_factor_ids
            if factor_id not in selected_factor_ids
        ]
        if (
            selected_paths
            and len(selected_factor_ids) + len(added_ids) > max_factors
        ):
            continue
        selected_paths.append(copy.deepcopy(path))
        selected_factor_ids.extend(added_ids)
        if len(selected_paths) >= max_paths:
            break
    standalone = sorted(
        (
            (
                len(ledger_tokens & _tokens(factor)),
                factor_id,
            )
            for factor_id, factor in factors.items()
            if factor_id not in selected_factor_ids
        ),
        reverse=True,
    )
    for score, factor_id in standalone:
        if len(selected_factor_ids) >= max_factors or score <= 0:
            break
        selected_factor_ids.append(factor_id)
    result["factor_checks"] = [
        copy.deepcopy(factors[factor_id])
        for factor_id in selected_factor_ids
    ]
    result["conditional_paths"] = selected_paths
    selected_edge_ids = {
        str(edge_id)
        for path in selected_paths
        for edge_id in path.get("edge_ids", [])
    }
    result["topology_edges"] = [
        copy.deepcopy(edge)
        for edge in result.get("topology_edges", [])
        if str(edge.get("id") or "") in selected_edge_ids
    ]
    result["routing_after_current_evidence"] = {
        "candidate_path_count": len(ranked_paths),
        "selected_path_ids": [
            str(path.get("id")) for path in selected_paths
        ],
        "selected_factor_ids": selected_factor_ids,
        "max_paths": max_paths,
        "max_factors": max_factors,
    }
    return result


def _live_reasoning_procedure_schema(
    path_ids: list[str],
    factor_ids: list[str],
) -> dict[str, Any]:
    return {
        "name": "live_dag_reasoning_procedure",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "target_operation_check": {"type": "string"},
                "ordered_path_ids": {
                    "type": "array",
                    "minItems": len(path_ids),
                    "maxItems": len(path_ids),
                    "items": {"type": "string", "enum": path_ids},
                },
                "path_execution_steps": {
                    "type": "array",
                    "minItems": len(path_ids),
                    "maxItems": len(path_ids),
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "path_id": {
                                "type": "string",
                                "enum": path_ids,
                            },
                            "factor_order": {
                                "type": "array",
                                "items": {
                                    "type": "string",
                                    "enum": factor_ids,
                                },
                            },
                            "purpose": {"type": "string"},
                            "failure_check": {"type": "string"},
                        },
                        "required": [
                            "path_id",
                            "factor_order",
                            "purpose",
                            "failure_check",
                        ],
                    },
                },
                "current_new_factor_rule": {"type": "string"},
                "target_bridge_plan": {"type": "string"},
                "magnitude_plan": {"type": "string"},
            },
            "required": [
                "target_operation_check",
                "ordered_path_ids",
                "path_execution_steps",
                "current_new_factor_rule",
                "target_bridge_plan",
                "magnitude_plan",
            ],
        },
    }


def _validate_live_reasoning_procedure(
    payload: dict[str, Any],
    *,
    memory: dict[str, Any],
) -> tuple[dict[str, float], list[str]]:
    paths = {
        str(path["id"]): [
            str(value) for value in path.get("factor_ids", [])
        ]
        for path in memory.get("conditional_paths", [])
    }
    expected_path_ids = set(paths)
    ordered_path_ids = [
        str(value) for value in payload.get("ordered_path_ids", [])
    ]
    errors = []
    if (
        len(ordered_path_ids) != len(expected_path_ids)
        or set(ordered_path_ids) != expected_path_ids
    ):
        errors.append(
            "ordered_path_ids must contain every routed path exactly once; "
            f"received={ordered_path_ids}; expected={sorted(expected_path_ids)}"
        )
    steps = payload.get("path_execution_steps", [])
    returned_steps = {
        str(step.get("path_id")): step
        for step in steps
        if isinstance(step, dict)
    }
    if len(returned_steps) != len(steps) or set(returned_steps) != (
        expected_path_ids
    ):
        errors.append(
            "path_execution_steps must contain every routed path exactly "
            f"once; received={sorted(returned_steps)}; "
            f"expected={sorted(expected_path_ids)}"
        )
    for path_id, expected_factor_order in paths.items():
        returned_factor_order = [
            str(value)
            for value in returned_steps.get(path_id, {}).get(
                "factor_order", []
            )
        ]
        if returned_factor_order != expected_factor_order:
            errors.append(
                f"{path_id} must preserve the DAG factor order "
                f"{expected_factor_order}"
            )
    for field in (
        "target_operation_check",
        "current_new_factor_rule",
        "target_bridge_plan",
        "magnitude_plan",
    ):
        if not str(payload.get(field) or "").strip():
            errors.append(f"{field} is empty")
    return {}, errors


def call_live_reasoning_procedure(
    *,
    client: OpenAI,
    model: str,
    question_id: str,
    public_case: dict[str, Any],
    target_operator: dict[str, Any],
    evidence_ledger: dict[str, Any],
    memory: dict[str, Any],
    max_tokens: int,
) -> tuple[dict[str, Any], dict[str, int], float, bool]:
    """Create a current-query procedure from retrieved DAG topology."""
    path_ids = [
        str(path["id"]) for path in memory.get("conditional_paths", [])
    ]
    factor_ids = [
        str(factor["id"]) for factor in memory.get("factor_checks", [])
    ]
    prompt = (
        "Construct the reasoning procedure for this current forecast from the "
        "retrieved outcome-neutral DAG. This is not a historical worked "
        "example. Order the routed paths by relevance to the current evidence "
        "ledger while preserving the exact factor order inside every path. "
        "State how each path should be checked, how its failure condition should "
        "be tested, how competing paths should reach the exact target operation, "
        "and what current numerical evidence would be required for magnitude. "
        "Do not assess whether a factor is supported, do not choose an answer, "
        "and do not emit a probability or target value. The next stage will fill "
        "this procedure using current evidence.\n\n"
        f"CURRENT CASE:\n{json.dumps(public_case, ensure_ascii=False)}\n\n"
        "DETERMINISTIC TARGET OPERATOR:\n"
        f"{json.dumps(target_operator, ensure_ascii=False)}\n\n"
        "CURRENT EVIDENCE LEDGER:\n"
        f"{json.dumps(evidence_ledger, ensure_ascii=False)}\n\n"
        "ROUTED OUTCOME-NEUTRAL DAG:\n"
        f"{json.dumps(memory, ensure_ascii=False)}"
    )
    payload, _, usage, seconds, repaired = _call_with_repair(
        client,
        model=model,
        system=(
            "You create a live execution procedure from a financial DAG. "
            "You do not forecast. Return JSON."
        ),
        prompt=prompt,
        schema=_live_reasoning_procedure_schema(path_ids, factor_ids),
        seed=_seed(question_id, "structured-hgf-live-procedure"),
        max_tokens=max_tokens,
        reasoning_effort="medium",
        validator=lambda candidate: _validate_live_reasoning_procedure(
            candidate,
            memory=memory,
        ),
        semantic_repair_contract=(
            "Use only these exact path IDs. ordered_path_ids is a permutation "
            f"of them and path_execution_steps uses each exactly once: "
            f"{json.dumps(path_ids)}. Never join an ID with '|', never append "
            "a checkpoint ID to a path ID, and never use a checkpoint ID as a "
            "path_id. Copy this exact factor order for each path without adding "
            "or removing any checkpoint: "
            + json.dumps(
                {
                    str(path.get("id")): [
                        str(value) for value in path.get("factor_ids", [])
                    ]
                    for path in memory.get("conditional_paths", [])
                },
                ensure_ascii=False,
            )
        ),
    )
    return payload, usage, seconds, repaired


def _instantiation_schema(
    factor_ids: list[str],
    edge_ids: list[str],
    path_ids: list[str],
) -> dict[str, Any]:
    return {
        "name": "current_dag_instantiation",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "checkpoint_assessments": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "checkpoint_id": {
                                "type": "string",
                            },
                            "status": {
                                "type": "string",
                                "enum": [
                                    "SUPPORTED",
                                    "CONTRADICTED",
                                    "UNKNOWN",
                                ],
                            },
                            "current_state": {
                                "type": "string",
                                "maxLength": 300,
                            },
                            "evidence_ids": {
                                "type": "array",
                                "maxItems": 6,
                                "items": {"type": "string"},
                            },
                            "requirement_satisfied": {"type": "boolean"},
                            "contradiction_present": {"type": "boolean"},
                            "assessment": {
                                "type": "string",
                                "maxLength": 600,
                            },
                        },
                        "required": [
                            "checkpoint_id",
                            "status",
                            "current_state",
                            "evidence_ids",
                            "requirement_satisfied",
                            "contradiction_present",
                            "assessment",
                        ],
                    },
                },
                "edge_assessments": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "edge_id": {
                                "type": "string",
                                "enum": edge_ids,
                            },
                            "status": {
                                "type": "string",
                                "enum": [
                                    "PRESERVED",
                                    "REVERSED",
                                    "CONTRADICTED",
                                    "UNVERIFIED",
                                ],
                            },
                            "current_relation": {
                                "type": "string",
                                "maxLength": 500,
                            },
                            "lag_assessment": {
                                "type": "string",
                                "maxLength": 240,
                            },
                            "confidence": {
                                "type": "string",
                                "enum": ["low", "medium", "high"],
                            },
                            "evidence_ids": {
                                "type": "array",
                                "maxItems": 6,
                                "items": {"type": "string"},
                            },
                            "assessment": {
                                "type": "string",
                                "maxLength": 600,
                            },
                        },
                        "required": [
                            "edge_id",
                            "status",
                            "current_relation",
                            "lag_assessment",
                            "confidence",
                            "evidence_ids",
                            "assessment",
                        ],
                    },
                },
                "path_failure_assessments": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "path_id": {
                                "type": "string",
                            },
                            "failure_condition_status": {
                                "type": "string",
                                "enum": ["PRESENT", "ABSENT", "UNKNOWN"],
                            },
                            "evidence_ids": {
                                "type": "array",
                                "maxItems": 6,
                                "items": {"type": "string"},
                            },
                            "assessment": {
                                "type": "string",
                                "maxLength": 600,
                            },
                        },
                        "required": [
                            "path_id",
                            "failure_condition_status",
                            "evidence_ids",
                            "assessment",
                        ],
                    },
                },
                "current_new_factors": {
                    "type": "array",
                    "maxItems": 3,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "factor": {
                                "type": "string",
                                "maxLength": 180,
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
                            "evidence_ids": {
                                "type": "array",
                                "minItems": 1,
                                "maxItems": 6,
                                "items": {"type": "string"},
                            },
                            "assessment": {
                                "type": "string",
                                "maxLength": 600,
                            },
                        },
                        "required": [
                            "factor",
                            "effect_on_target",
                            "evidence_ids",
                            "assessment",
                        ],
                    },
                },
                "instantiation_summary": {
                    "type": "string",
                    "maxLength": 700,
                },
            },
            "required": [
                "checkpoint_assessments",
                "edge_assessments",
                "path_failure_assessments",
                "current_new_factors",
                "instantiation_summary",
            ],
        },
    }


def _validate_and_derive_instantiation(
    payload: dict[str, Any],
    *,
    memory: dict[str, Any],
    evidence_ids: set[str],
    require_complete: bool = False,
) -> tuple[dict[str, float], list[str]]:
    errors: list[str] = []
    factor_ids = [
        str(item["id"]) for item in memory.get("factor_checks", [])
    ]
    edge_ids = [
        str(item["id"]) for item in memory.get("topology_edges", [])
    ]
    path_ids = [
        str(item["id"]) for item in memory.get("conditional_paths", [])
    ]
    returned_assessments: dict[str, dict[str, Any]] = {}
    raw_assessments = payload.get("checkpoint_assessments", [])
    if not isinstance(raw_assessments, list):
        return {}, ["checkpoint_assessments must be an array of objects"]
    for index, item in enumerate(raw_assessments):
        if not isinstance(item, dict):
            errors.append(
                f"checkpoint_assessments[{index}] must be an object; "
                f"received_type={type(item).__name__}"
            )
            continue
        checkpoint_id = str(item.get("checkpoint_id") or "")
        if (
            checkpoint_id in factor_ids
            and checkpoint_id not in returned_assessments
        ):
            returned_assessments[checkpoint_id] = item
    if require_complete:
        missing_factor_ids = [
            checkpoint_id
            for checkpoint_id in factor_ids
            if checkpoint_id not in returned_assessments
        ]
        if missing_factor_ids:
            errors.append(
                "checkpoint_assessments omitted required IDs: "
                + json.dumps(missing_factor_ids)
            )
    assessments = [
        returned_assessments.get(
            checkpoint_id,
            {
                "checkpoint_id": checkpoint_id,
                "status": "UNKNOWN",
                "current_state": "not assessed",
                "evidence_ids": [],
                "requirement_satisfied": False,
                "contradiction_present": False,
                "assessment": (
                    "The provider omitted this checkpoint, so runtime "
                    "normalization conservatively marks it unknown."
                ),
            },
        )
        for checkpoint_id in factor_ids
    ]
    payload["checkpoint_assessments"] = assessments
    by_factor = {
        str(item.get("checkpoint_id") or ""): item
        for item in assessments
    }
    for item in assessments:
        used = {
            str(value)
            for value in item.get("evidence_ids", [])
            if str(value) in evidence_ids
        }
        item["evidence_ids"] = sorted(used)
        status = str(item.get("status") or "")
        requirement = bool(item.get("requirement_satisfied"))
        contradiction = bool(item.get("contradiction_present"))
        if status in {"SUPPORTED", "CONTRADICTED"} and not used:
            item["status"] = "UNKNOWN"
            item["requirement_satisfied"] = False
            item["contradiction_present"] = False
            status = "UNKNOWN"
        if status == "SUPPORTED" and (not requirement or contradiction):
            item["status"] = "UNKNOWN"
            item["requirement_satisfied"] = False
            item["contradiction_present"] = False
            status = "UNKNOWN"
        if status == "CONTRADICTED" and not contradiction:
            item["status"] = "UNKNOWN"
            item["requirement_satisfied"] = False
            item["contradiction_present"] = False

    returned_edges: dict[str, dict[str, Any]] = {}
    raw_edges = payload.get("edge_assessments", [])
    if not isinstance(raw_edges, list):
        return {}, ["edge_assessments must be an array of objects"]
    for index, item in enumerate(raw_edges):
        if not isinstance(item, dict):
            errors.append(
                f"edge_assessments[{index}] must be an object; "
                f"received_type={type(item).__name__}"
            )
            continue
        edge_id = str(item.get("edge_id") or "")
        if edge_id in edge_ids and edge_id not in returned_edges:
            returned_edges[edge_id] = item
    if require_complete:
        missing_edge_ids = [
            edge_id for edge_id in edge_ids if edge_id not in returned_edges
        ]
        if missing_edge_ids:
            errors.append(
                "edge_assessments omitted required IDs: "
                + json.dumps(missing_edge_ids)
            )
    edge_assessments = [
        returned_edges.get(
            edge_id,
            {
                "edge_id": edge_id,
                "status": "UNVERIFIED",
                "current_relation": "not assessed",
                "lag_assessment": "not assessed",
                "confidence": "low",
                "evidence_ids": [],
                "assessment": (
                    "The provider omitted this edge, so runtime "
                    "normalization conservatively marks it unverified."
                ),
            },
        )
        for edge_id in edge_ids
    ]
    payload["edge_assessments"] = edge_assessments
    for item in edge_assessments:
        used = {
            str(value)
            for value in item.get("evidence_ids", [])
            if str(value) in evidence_ids
        }
        item["evidence_ids"] = sorted(used)
        if item.get("status") != "UNVERIFIED" and not used:
            item["status"] = "UNVERIFIED"
            item["confidence"] = "low"
    edge_by_id = {
        str(item.get("edge_id") or ""): item
        for item in edge_assessments
    }

    returned_failures: dict[str, dict[str, Any]] = {}
    raw_failures = payload.get("path_failure_assessments", [])
    if not isinstance(raw_failures, list):
        return {}, ["path_failure_assessments must be an array of objects"]
    for index, item in enumerate(raw_failures):
        if not isinstance(item, dict):
            errors.append(
                f"path_failure_assessments[{index}] must be an object; "
                f"received_type={type(item).__name__}"
            )
            continue
        path_id = str(item.get("path_id") or "")
        if path_id in path_ids and path_id not in returned_failures:
            returned_failures[path_id] = item
    if require_complete:
        missing_path_ids = [
            path_id
            for path_id in path_ids
            if path_id not in returned_failures
        ]
        if missing_path_ids:
            errors.append(
                "path_failure_assessments omitted required IDs: "
                + json.dumps(missing_path_ids)
            )
    if errors:
        return {}, errors
    failures = [
        returned_failures.get(
            path_id,
            {
                "path_id": path_id,
                "failure_condition_status": "UNKNOWN",
                "evidence_ids": [],
                "assessment": (
                    "The provider omitted this failure check, so runtime "
                    "normalization conservatively marks it unknown."
                ),
            },
        )
        for path_id in path_ids
    ]
    payload["path_failure_assessments"] = failures
    failure_by_path = {
        str(item.get("path_id") or ""): item for item in failures
    }
    for item in failures:
        used = {
            str(value)
            for value in item.get("evidence_ids", [])
            if str(value) in evidence_ids
        }
        item["evidence_ids"] = sorted(used)
        if (
            item.get("failure_condition_status") == "PRESENT"
            and not used
        ):
            item["failure_condition_status"] = "UNKNOWN"
    derived_paths = []
    for path in memory.get("conditional_paths", []):
        path_id = str(path["id"])
        factor_statuses = [
            str(by_factor.get(str(factor_id), {}).get("status") or "UNKNOWN")
            for factor_id in path.get("factor_ids", [])
        ]
        path_edge_ids = [
            str(value) for value in path.get("edge_ids", [])
        ]
        edge_statuses = [
            str(edge_by_id.get(edge_id, {}).get("status") or "UNVERIFIED")
            for edge_id in path_edge_ids
        ]
        failure = str(
            failure_by_path.get(path_id, {}).get(
                "failure_condition_status"
            )
            or "UNKNOWN"
        )
        if (
            failure == "PRESENT"
            or "CONTRADICTED" in factor_statuses
            or any(
                value in {"REVERSED", "CONTRADICTED"}
                for value in edge_statuses
            )
        ):
            status = "BLOCKED"
        elif (
            factor_statuses
            and all(value == "SUPPORTED" for value in factor_statuses)
            and (
                not path_edge_ids
                or all(value == "PRESERVED" for value in edge_statuses)
            )
            and failure == "ABSENT"
        ):
            status = "ACTIVE"
        else:
            status = "UNKNOWN"
        failure_ids = {
            str(value)
            for value in failure_by_path.get(path_id, {}).get(
                "evidence_ids", []
            )
        }
        derived_paths.append(
            {
                "path_id": path_id,
                "status": status,
                "factor_ids": [
                    str(value) for value in path.get("factor_ids", [])
                ],
                "edge_ids": path_edge_ids,
                "edge_statuses": edge_statuses,
                "effect_if_active": path.get("effect_if_active"),
                "failure_condition_status": failure,
                "evidence_ids": sorted(
                    {
                        str(value)
                        for factor_id in path.get("factor_ids", [])
                        for value in by_factor.get(
                            str(factor_id), {}
                        ).get("evidence_ids", [])
                    }
                    | {
                        str(value)
                        for edge_id in path_edge_ids
                        for value in edge_by_id.get(
                            edge_id, {}
                        ).get("evidence_ids", [])
                    }
                    | failure_ids
                ),
                "assessment": failure_by_path.get(path_id, {}).get(
                    "assessment", ""
                ),
            }
        )
    kept_new_factors = []
    for item in payload.get("current_new_factors", []):
        used = {
            str(value)
            for value in item.get("evidence_ids", [])
            if str(value) in evidence_ids
        }
        if not used:
            continue
        item["evidence_ids"] = sorted(used)
        kept_new_factors.append(item)
    payload["current_new_factors"] = kept_new_factors
    payload["derived_path_assessments"] = derived_paths
    payload["active_path_ids"] = [
        item["path_id"]
        for item in derived_paths
        if item["status"] == "ACTIVE"
    ]
    return {}, errors


def call_current_dag_instantiation(
    *,
    client: OpenAI,
    model: str,
    question_id: str,
    public_case: dict[str, Any],
    evidence: list[dict[str, Any]],
    evidence_ledger: dict[str, Any],
    memory: dict[str, Any],
    max_tokens: int,
    require_complete: bool = False,
) -> tuple[dict[str, Any], dict[str, int], float, bool]:
    evidence_ids = {str(item["id"]) for item in evidence}
    factor_ids = [
        str(item["id"]) for item in memory.get("factor_checks", [])
    ]
    edge_ids = [
        str(item["id"]) for item in memory.get("topology_edges", [])
    ]
    path_ids = [
        str(item["id"]) for item in memory.get("conditional_paths", [])
    ]
    prompt = (
        "Instantiate the routed outcome-neutral DAG with the current evidence "
        "ledger. Assess every checkpoint, not only the first factor in a path. "
        "SUPPORTED means the checkpoint's stated evidence requirement is met "
        "now and its contradiction signal is absent. CONTRADICTED means current "
        "evidence activates the stated contradiction. Otherwise use UNKNOWN. "
        "Assess every preserved directed edge as PRESERVED, REVERSED, "
        "CONTRADICTED, or UNVERIFIED using current evidence. PRESERVED requires "
        "evidence for the relation and timing, not only evidence for its two "
        "endpoint nodes. Then test every listed path failure condition. Do not "
        "forecast, choose "
        "an option, or assign probabilities. The runtime will activate a path "
        "only when every checkpoint in it is supported and no failure condition "
        "is present. Add a CURRENT_NEW factor only when current evidence shows "
        "an important driver missing from all routed paths. Use only current "
        "evidence IDs and keep each field under 80 words.\n\n"
        f"CURRENT CASE:\n{json.dumps(public_case, ensure_ascii=False)}\n\n"
        "CURRENT EVIDENCE LEDGER:\n"
        f"{json.dumps(evidence_ledger, ensure_ascii=False)}\n\n"
        "ROUTED OUTCOME-NEUTRAL DAG MEMORY:\n"
        f"{json.dumps(memory, ensure_ascii=False)}\n\n"
        "CURRENT CUTOFF-SAFE EVIDENCE:\n"
        f"{json.dumps(evidence, ensure_ascii=False)}"
    )
    payload, _, usage, seconds, repaired = _call_with_repair(
        client,
        model=model,
        system=(
            "You fill trusted financial DAG checkpoints with current evidence. "
            "You do not produce a forecast. Return JSON."
        ),
        prompt=prompt,
        schema=_instantiation_schema(factor_ids, edge_ids, path_ids),
        seed=_seed(question_id, "structured-hgf-dag-instantiation"),
        max_tokens=max_tokens,
        reasoning_effort="medium",
        validator=lambda candidate: _validate_and_derive_instantiation(
            candidate,
            memory=memory,
            evidence_ids=evidence_ids,
            require_complete=require_complete,
        ),
        semantic_repair_contract=(
            "checkpoint_assessments, edge_assessments, and "
            "path_failure_assessments must be JSON "
            "arrays containing objects only. Use these exact checkpoint IDs "
            f"when assessing DAG factors: {json.dumps(factor_ids)}. Use these "
            "exact edge IDs when assessing preserved relationships: "
            f"{json.dumps(edge_ids)}. Use these "
            "exact path IDs for failure assessments: "
            f"{json.dumps(path_ids)}. Never emit a bare string in either array."
        ),
    )
    return payload, usage, seconds, repaired


def _synthesis_schema(
    active_path_ids: list[str],
    all_path_ids: list[str] | None = None,
) -> dict[str, Any]:
    source_ids = list(all_path_ids or active_path_ids) + [
        "CURRENT_NEW",
        "TARGET_CONTRACT",
    ]
    return {
        "name": "structured_hgf_reasoning_trace",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "target_semantics": {
                    "type": "string",
                    "maxLength": 600,
                },
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
                            "enum": [
                                "supported",
                                "partial",
                                "unsupported",
                            ],
                        },
                        "assessment": {
                            "type": "string",
                            "maxLength": 600,
                        },
                    },
                    "required": [
                        "metric_match",
                        "horizon_match",
                        "magnitude_support",
                        "assessment",
                    ],
                },
                "causal_balance": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "favored_direction": {
                            "type": "string",
                            "enum": [
                                "up",
                                "down",
                                "balanced",
                                "uncertain",
                            ],
                        },
                        "active_path_ids": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "enum": active_path_ids,
                            },
                        },
                        "assessment": {
                            "type": "string",
                            "maxLength": 700,
                        },
                    },
                    "required": [
                        "favored_direction",
                        "active_path_ids",
                        "assessment",
                    ],
                },
                "magnitude_readiness": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "support": {
                            "type": "string",
                            "enum": [
                                "direct",
                                "derived",
                                "direction_only",
                                "insufficient",
                            ],
                        },
                        "evidence_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "assessment": {
                            "type": "string",
                            "maxLength": 600,
                        },
                    },
                    "required": [
                        "support",
                        "evidence_ids",
                        "assessment",
                    ],
                },
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
                                ],
                            },
                            "statement": {
                                "type": "string",
                                "maxLength": 700,
                            },
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
                            "source_id": {
                                "type": "string",
                                "enum": source_ids,
                            },
                        },
                        "required": [
                            "step_type",
                            "statement",
                            "evidence_ids",
                            "effect_on_target",
                            "source_id",
                        ],
                    },
                },
                "counterevidence": {
                    "type": "string",
                    "maxLength": 700,
                },
                "uncertainty": {
                    "type": "string",
                    "maxLength": 600,
                },
            },
            "required": [
                "target_semantics",
                "selected_evidence_ids",
                "evidence_fit",
                "causal_balance",
                "magnitude_readiness",
                "reasoning_steps",
                "counterevidence",
                "uncertainty",
            ],
        },
    }


def _validate_synthesis(
    payload: dict[str, Any],
    *,
    evidence_ids: set[str],
    active_path_ids: set[str],
    all_path_ids: set[str] | None = None,
) -> tuple[dict[str, float], list[str]]:
    errors: list[str] = []
    steps = payload.get("reasoning_steps", [])
    if not isinstance(steps, list):
        return {}, ["reasoning_steps must be an array of objects"]
    selected = {
        str(value) for value in payload.get("selected_evidence_ids", [])
    }
    if not selected:
        errors.append("structured synthesis selected no current evidence")
    if selected - evidence_ids:
        errors.append("structured synthesis cites unknown evidence")
    step_types = set()
    for index, item in enumerate(steps):
        if not isinstance(item, dict):
            errors.append(
                f"reasoning_steps[{index}] must be an object; "
                f"received_type={type(item).__name__}"
            )
            continue
        step_type = str(item.get("step_type") or "")
        step_types.add(step_type)
        used = {str(value) for value in item.get("evidence_ids", [])}
        if used - evidence_ids:
            errors.append("reasoning step cites unknown current evidence")
        source_id = str(item.get("source_id") or "")
        inactive_path_ids = (all_path_ids or active_path_ids) - active_path_ids
        if source_id in inactive_path_ids and step_type != "counterevidence":
            errors.append(
                f"reasoning_steps[{index}] uses inactive path "
                f"{source_id!r} outside a counterevidence step"
            )
        if (
            source_id not in active_path_ids
            and source_id not in {"CURRENT_NEW", "TARGET_CONTRACT"}
            and step_type in {"driver", "mechanism"}
        ):
            allowed_directional_sources = sorted(active_path_ids) + [
                "CURRENT_NEW"
            ]
            errors.append(
                f"reasoning_steps[{index}].source_id={source_id!r} is "
                "invalid for a directional step; allowed source IDs="
                f"{allowed_directional_sources}; checkpoint IDs and empty "
                "source IDs are forbidden"
            )
    for required in ("baseline", "target_bridge"):
        if required not in step_types:
            errors.append(f"structured synthesis lacks {required}")
    balance_paths = {
        str(value)
        for value in payload.get("causal_balance", {}).get(
            "active_path_ids", []
        )
    }
    if not balance_paths <= active_path_ids:
        errors.append(
            "causal_balance.active_path_ids includes non-active values; "
            f"received={sorted(balance_paths)}; "
            f"allowed={sorted(active_path_ids)}"
        )
    magnitude_ids = {
        str(value)
        for value in payload.get("magnitude_readiness", {}).get(
            "evidence_ids", []
        )
    }
    if magnitude_ids - evidence_ids:
        errors.append("magnitude readiness cites unknown evidence")
    return {}, errors


def call_structured_synthesis(
    *,
    client: OpenAI,
    model: str,
    question_id: str,
    public_case: dict[str, Any],
    target_operator: dict[str, Any],
    evidence: list[dict[str, Any]],
    evidence_ledger: dict[str, Any],
    memory: dict[str, Any],
    instantiation: dict[str, Any],
    max_tokens: int,
    use_worked_demonstration: bool = True,
) -> tuple[dict[str, Any], dict[str, int], float, bool]:
    evidence_ids = {str(item["id"]) for item in evidence}
    active_path_ids = {
        str(value) for value in instantiation.get("active_path_ids", [])
    }
    all_path_ids = {
        str(path.get("id"))
        for path in memory.get("conditional_paths", [])
        if path.get("id")
    }
    inactive_path_ids = sorted(all_path_ids - active_path_ids)
    if memory.get("live_reasoning_procedure"):
        procedure_instruction = (
            "Follow the live reasoning procedure derived for this current "
            "question. It preserves the retrieved DAG topology but contains no "
            "historical answer or forecast. "
        )
    else:
        procedure_instruction = (
        "Follow the worked demonstration's order of reasoning, but never copy "
        "its historical entities, values, directions, estimate, or conclusion. "
        if use_worked_demonstration
        else (
            "Use the fixed procedure of target baseline, current drivers, "
            "supported DAG mechanisms, counterevidence, target bridge, and "
            "magnitude readiness. No worked historical demonstration is "
            "available. "
        )
        )
    prompt = (
        "Write the current-case reasoning trace without choosing an option or "
        "assigning probabilities. Start from the factual evidence ledger. Use "
        "only ACTIVE DAG paths as directional mechanisms. BLOCKED and UNKNOWN "
        "paths may appear only as counterevidence or uncertainty. Include "
        "evidence-backed CURRENT_NEW factors when the routed DAG omitted an "
        "important current driver. "
        + procedure_instruction
        + "Reconcile active and competing paths into the "
        "exact target operation. State whether the evidence supports numerical "
        "magnitude or only direction. Do not infer that positive pressure crosses "
        "an upper boundary or that negative pressure crosses a lower boundary. "
        "A separate boundary stage will make the numerical decision. Use only "
        "current evidence IDs and keep every field under 80 words.\n\n"
        f"CURRENT CASE:\n{json.dumps(public_case, ensure_ascii=False)}\n\n"
        "DETERMINISTIC TARGET OPERATOR:\n"
        f"{json.dumps(target_operator, ensure_ascii=False)}\n\n"
        "CURRENT EVIDENCE LEDGER:\n"
        f"{json.dumps(evidence_ledger, ensure_ascii=False)}\n\n"
        "ROUTED REASONING MEMORY:\n"
        f"{json.dumps(memory, ensure_ascii=False)}\n\n"
        "CURRENT DAG INSTANTIATION:\n"
        f"{json.dumps(instantiation, ensure_ascii=False)}\n\n"
        "CURRENT CUTOFF-SAFE EVIDENCE:\n"
        f"{json.dumps(evidence, ensure_ascii=False)}"
    )
    payload, _, usage, seconds, repaired = _call_with_repair(
        client,
        model=model,
        system=(
            "You synthesize independent cutoff-safe financial reasoning from "
            "current evidence and fully instantiated DAG paths. You do not "
            "produce a forecast answer or probabilities. Return JSON."
        ),
        prompt=prompt,
        schema=_synthesis_schema(
            sorted(active_path_ids),
            sorted(all_path_ids),
        ),
        seed=_seed(question_id, "structured-hgf-synthesis"),
        max_tokens=max_tokens,
        validator=lambda candidate: _validate_synthesis(
            candidate,
            evidence_ids=evidence_ids,
            active_path_ids=active_path_ids,
            all_path_ids=all_path_ids,
        ),
        semantic_repair_contract=(
            f"ACTIVE_PATH_IDS={json.dumps(sorted(active_path_ids))}. "
            f"INACTIVE_OR_UNKNOWN_PATH_IDS={json.dumps(inactive_path_ids)}. "
            "reasoning_steps must contain at least one baseline step with "
            "source_id TARGET_CONTRACT and at least one target_bridge step "
            "with source_id TARGET_CONTRACT. Never omit either step, including "
            "when no path is active. "
            "For driver and mechanism steps, source_id must be one of the "
            "ACTIVE_PATH_IDS. CURRENT_NEW must be used for an evidence-backed "
            "directional factor outside the routed graph, including when no "
            "path is active. INACTIVE_OR_UNKNOWN_PATH_IDS may be used only "
            "for counterevidence steps. TARGET_CONTRACT is allowed only for "
            "baseline and target_bridge. Checkpoint IDs are never valid "
            "source_id values. causal_balance.active_path_ids must be a subset "
            "of ACTIVE_PATH_IDS. The routed checkpoint chains are "
            + json.dumps(
                {
                    str(path.get("id")): [
                        str(value) for value in path.get("factor_ids", [])
                    ]
                    for path in memory.get("conditional_paths", [])
                },
                ensure_ascii=False,
            )
        ),
    )
    return payload, usage, seconds, repaired
