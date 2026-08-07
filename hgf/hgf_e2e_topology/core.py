"""Frozen procedural HGF reasoning over a current-instantiated exact topology."""

from __future__ import annotations

import copy
import json
import re
from typing import Any

from hgf.exemplar import _call_with_repair
from hgf.forecast_core import _seed
from openai import OpenAI


def _schema(path_ids: list[str]) -> dict[str, Any]:
    source_ids = [*path_ids, "CURRENT_NEW", "TARGET_CONTRACT"]
    return {
        "name": "procedural_topology_reasoning",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "target_semantics": {"type": "string", "maxLength": 600},
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
                        "assessment": {"type": "string", "maxLength": 600},
                    },
                    "required": [
                        "metric_match",
                        "horizon_match",
                        "magnitude_support",
                        "assessment",
                    ],
                },
                "current_new_factors": {
                    "type": "array",
                    "maxItems": 3,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "factor": {"type": "string", "maxLength": 180},
                            "effect_on_target": {
                                "type": "string",
                                "enum": ["up", "down", "neutral", "mixed", "uncertain"],
                            },
                            "evidence_ids": {
                                "type": "array",
                                "minItems": 1,
                                "maxItems": 6,
                                "items": {"type": "string"},
                            },
                            "assessment": {"type": "string", "maxLength": 500},
                        },
                        "required": [
                            "factor",
                            "effect_on_target",
                            "evidence_ids",
                            "assessment",
                        ],
                    },
                },
                "causal_balance": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "favored_direction": {
                            "type": "string",
                            "enum": ["up", "down", "balanced", "uncertain"],
                        },
                        "used_path_ids": {
                            "type": "array",
                            "items": {"type": "string", "enum": path_ids},
                        },
                        "assessment": {"type": "string", "maxLength": 700},
                    },
                    "required": [
                        "favored_direction",
                        "used_path_ids",
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
                        "assessment": {"type": "string", "maxLength": 600},
                    },
                    "required": ["support", "evidence_ids", "assessment"],
                },
                "reasoning_steps": {
                    "type": "array",
                    "minItems": 3,
                    "maxItems": 10,
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
                            "statement": {"type": "string", "maxLength": 500},
                            "evidence_ids": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "effect_on_target": {
                                "type": "string",
                                "enum": ["up", "down", "neutral", "mixed", "uncertain"],
                            },
                            "source_id": {"type": "string", "enum": source_ids},
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
                "counterevidence": {"type": "string", "maxLength": 700},
                "uncertainty": {"type": "string", "maxLength": 600},
            },
            "required": [
                "target_semantics",
                "selected_evidence_ids",
                "evidence_fit",
                "current_new_factors",
                "causal_balance",
                "magnitude_readiness",
                "reasoning_steps",
                "counterevidence",
                "uncertainty",
            ],
        },
    }


def _filter(values: Any, allowed: set[str]) -> list[str]:
    return sorted({str(value) for value in (values or []) if str(value) in allowed})


def _validate(
    payload: dict[str, Any],
    *,
    path_ids: list[str],
    evidence_ids: set[str],
    graph_evidence_ids: set[str],
) -> tuple[dict[str, float], list[str]]:
    errors: list[str] = []
    required_fields = {
        "target_semantics",
        "selected_evidence_ids",
        "evidence_fit",
        "current_new_factors",
        "causal_balance",
        "magnitude_readiness",
        "reasoning_steps",
        "counterevidence",
        "uncertainty",
    }
    missing = sorted(required_fields - set(payload))
    if missing:
        return {}, [f"missing required reasoning fields: {missing}"]
    if not str(payload.get("target_semantics") or "").strip():
        errors.append("target_semantics must be substantive")
    if not payload.get("selected_evidence_ids"):
        errors.append("selected_evidence_ids must cite current evidence")
    if not isinstance(payload.get("evidence_fit"), dict):
        errors.append("evidence_fit must be an object")
    if not isinstance(payload.get("causal_balance"), dict):
        errors.append("causal_balance must be an object")
    if not isinstance(payload.get("magnitude_readiness"), dict):
        errors.append("magnitude_readiness must be an object")
    if len(payload.get("reasoning_steps") or []) < 3:
        errors.append("reasoning_steps must contain at least three material steps")
    if not str(payload.get("counterevidence") or "").strip():
        errors.append("counterevidence must be explicit")
    if not str(payload.get("uncertainty") or "").strip():
        errors.append("uncertainty must be explicit")
    if errors:
        return {}, errors
    payload.setdefault(
        "target_semantics",
        "Target semantics are defined by the public contract.",
    )
    payload.setdefault(
        "evidence_fit",
        {
            "metric_match": "weak",
            "horizon_match": "weak",
            "magnitude_support": "unsupported",
            "assessment": "No additional evidence-fit assessment was returned.",
        },
    )
    payload.setdefault("current_new_factors", [])
    payload.setdefault(
        "causal_balance",
        {
            "favored_direction": "uncertain",
            "used_path_ids": [],
            "assessment": "No directional balance was returned.",
        },
    )
    payload.setdefault(
        "magnitude_readiness",
        {
            "support": "insufficient",
            "evidence_ids": [],
            "assessment": "No target-period magnitude support was returned.",
        },
    )
    payload.setdefault("reasoning_steps", [])
    payload.setdefault("counterevidence", "No explicit counterevidence was returned.")
    payload.setdefault("uncertainty", "The reasoning output was incomplete.")
    payload["selected_evidence_ids"] = _filter(
        payload.get("selected_evidence_ids"), evidence_ids
    )
    factors = []
    for item in payload.get("current_new_factors") or []:
        item["evidence_ids"] = _filter(item.get("evidence_ids"), evidence_ids)
        if item["evidence_ids"]:
            factors.append(item)
    payload["current_new_factors"] = factors
    used = set(
        _filter(
            (payload.get("causal_balance") or {}).get("used_path_ids"),
            set(path_ids),
        )
    )
    normalizations: list[str] = []
    steps = payload.get("reasoning_steps") or []
    for item in steps:
        item["evidence_ids"] = _filter(item.get("evidence_ids"), evidence_ids)
        raw_source_id = str(item.get("source_id") or "")
        source_ids = [
            value.strip()
            for value in raw_source_id.split(",")
            if value.strip()
        ]
        expanded_source_ids: list[str] = []
        for value in source_ids:
            if value in {"CURRENT_NEW_FACTOR", "CURRENT_CASE"}:
                expanded_source_ids.append("CURRENT_NEW")
                normalizations.append(f"source alias {value} -> CURRENT_NEW")
                continue
            match = re.fullmatch(r"(D\d+):target_bridge", value)
            if match:
                namespace = match.group(1) + ":"
                mapped = [
                    path_id
                    for path_id in path_ids
                    if path_id.startswith(namespace)
                ]
                if mapped:
                    expanded_source_ids.extend(mapped)
                    normalizations.append(
                        f"target bridge node {value} -> {','.join(mapped)}"
                    )
                    continue
            expanded_source_ids.append(value)
        source_ids = expanded_source_ids
        allowed_source_ids = set(path_ids) | {"CURRENT_NEW", "TARGET_CONTRACT"}
        step_type = str(item.get("step_type") or "")
        fallback_source = (
            "TARGET_CONTRACT"
            if step_type in {"baseline", "target_bridge"}
            else "CURRENT_NEW"
        )
        normalized_source_ids: list[str] = []
        for value in source_ids:
            if value in allowed_source_ids:
                normalized_source_ids.append(value)
            else:
                normalized_source_ids.append(fallback_source)
                normalizations.append(
                    f"unknown audit source {value} -> {fallback_source}"
                )
        if not normalized_source_ids:
            normalized_source_ids = [fallback_source]
            normalizations.append(
                f"empty audit source -> {fallback_source}"
            )
        source_ids = list(dict.fromkeys(normalized_source_ids))
        item["source_id"] = source_ids[0]
        if len(source_ids) > 1:
            normalizations.append(
                f"multi-source audit {raw_source_id} -> {source_ids[0]}"
            )
        if step_type in {
            "driver",
            "mechanism",
            "target_bridge",
        }:
            used.update(value for value in source_ids if value in path_ids)
    payload.setdefault("causal_balance", {})["used_path_ids"] = sorted(used)
    baseline_count = sum(
        str(item.get("step_type") or "") == "baseline" for item in steps
    )
    bridge_count = sum(
        str(item.get("step_type") or "") == "target_bridge" for item in steps
    )
    if baseline_count == 0:
        baseline_step = {
            "step_type": "baseline",
            "statement": str(payload.get("target_semantics") or "")[:500],
            "evidence_ids": [],
            "effect_on_target": "uncertain",
            "source_id": "TARGET_CONTRACT",
        }
        if len(steps) >= 10:
            steps[0] = baseline_step
        else:
            steps.insert(0, baseline_step)
        normalizations.append("baseline projected from target_semantics")
    if bridge_count == 0:
        balance = payload.get("causal_balance") or {}
        magnitude = payload.get("magnitude_readiness") or {}
        bridge_statement = " ".join(
            value
            for value in [
                str(balance.get("assessment") or "").strip(),
                str(magnitude.get("assessment") or "").strip(),
            ]
            if value
        )[:500]
        bridge_step = {
            "step_type": "target_bridge",
            "statement": bridge_statement,
            "evidence_ids": _filter(
                magnitude.get("evidence_ids"), evidence_ids
            ),
            "effect_on_target": str(
                balance.get("favored_direction") or "uncertain"
            ),
            "source_id": "TARGET_CONTRACT",
        }
        if len(steps) >= 10:
            steps[-1] = bridge_step
        else:
            steps.append(bridge_step)
        normalizations.append(
            "target_bridge projected from causal_balance and magnitude_readiness"
        )
    payload["reasoning_steps"] = steps
    payload["trace_normalization"] = {
        "applied": bool(normalizations),
        "actions": normalizations,
        "probability_modified": False,
    }
    magnitude = payload.get("magnitude_readiness") or {}
    magnitude["evidence_ids"] = _filter(magnitude.get("evidence_ids"), evidence_ids)
    if not payload["selected_evidence_ids"]:
        grounded = {
            evidence_id
            for item in payload.get("reasoning_steps") or []
            for evidence_id in item.get("evidence_ids") or []
        }
        grounded.update(
            evidence_id
            for item in factors
            for evidence_id in item.get("evidence_ids") or []
        )
        payload["selected_evidence_ids"] = sorted(grounded)
    if not payload["selected_evidence_ids"]:
        payload["selected_evidence_ids"] = sorted(graph_evidence_ids)
    return {}, errors


def _path_assessments(
    instantiated_graph: dict[str, Any], routed_memory: dict[str, Any]
) -> list[dict[str, Any]]:
    node_states = {
        str(item.get("node_id") or ""): item
        for item in instantiated_graph.get("node_states") or []
    }
    path_states = {
        str(item.get("path_id") or ""): item
        for item in instantiated_graph.get("path_states") or []
    }
    result = []
    for path in routed_memory.get("paths", []):
        path_id = str(path["id"])
        candidates = [
            node_states.get(str(node_id), {})
            for node_id in path.get("checkpoint_ids", [])
            if not str(node_id).endswith(":target_bridge")
        ]
        grounded = [
            item
            for item in candidates
            if item.get("relation") in {"ALIGNED", "REVERSED"}
            and item.get("evidence_ids")
        ]
        anchor = grounded[0] if grounded else (candidates[0] if candidates else {})
        state = path_states.get(path_id, {})
        status = str(state.get("status") or "UNRESOLVED")
        result.append(
            {
                "path_id": path_id,
                "anchor_checkpoint_id": str(anchor.get("node_id") or ""),
                "anchor_status": (
                    str(anchor.get("relation"))
                    if anchor.get("relation") in {"ALIGNED", "REVERSED"}
                    else "UNMAPPED"
                ),
                "anchor_state": str(anchor.get("value") or "unknown"),
                "evidence_ids": list(anchor.get("evidence_ids") or []),
                "applicability": {
                    "ACTIVE": "SUPPORTED",
                    "CONTRADICTED": "CONTRADICTED",
                }.get(status, "UNKNOWN"),
                "effect_on_target": str(
                    state.get("effect_on_target") or "uncertain"
                ),
                "topology_trace": " -> ".join(
                    str(value) for value in path.get("checkpoint_ids", [])
                )[:450],
                "assessment": (
                    f"Current graph instantiation marks this path {status.lower()}."
                ),
            }
        )
    return result


def attach_graph_audit(
    reasoning: dict[str, Any],
    *,
    instantiated_graph: dict[str, Any],
    routed_memory: dict[str, Any],
) -> dict[str, Any]:
    result = copy.deepcopy(reasoning)
    result["path_assessments"] = _path_assessments(
        instantiated_graph, routed_memory
    )
    path_by_id = {
        str(path["id"]): path for path in routed_memory.get("paths", [])
    }
    used_ids = {
        str(value)
        for value in result.get("causal_balance", {}).get("used_path_ids", [])
        if str(value) in path_by_id
    }
    used_sources = {
        str(path_by_id[path_id].get("source_question_id") or "")
        for path_id in used_ids
    }
    effects = {
        str(item.get("effect_on_target") or "")
        for item in result["path_assessments"]
        if str(item.get("path_id") or "") in used_ids
    }
    conflict = {"up", "down"} <= effects or "mixed" in effects
    summary = (
        f"Structural use facts: {len(used_ids)} of {len(path_by_id)} routed "
        f"paths contributed across {len(used_sources)} resolved-event DAGs. "
        f"Directional conflict is {'present' if conflict else 'absent'}."
    )
    result["structural_support_summary"] = {
        "used_path_count": len(used_ids),
        "routed_path_count": len(path_by_id),
        "used_source_dag_count": len(used_sources),
        "directional_conflict": conflict,
        "summary": summary,
    }
    evidence_fit = result.setdefault(
        "evidence_fit",
        {
            "metric_match": "weak",
            "horizon_match": "weak",
            "magnitude_support": "unsupported",
            "assessment": "",
        },
    )
    evidence_fit["assessment"] = (
        summary + " " + str(evidence_fit.get("assessment") or "")
    )[:600]
    result["uncertainty"] = (
        summary + " " + str(result.get("uncertainty") or "")
    )[:600]
    return result


def render_reasoning_narrative(reasoning: dict[str, Any]) -> dict[str, Any]:
    """Expose the exact forecast trace as readable prose without new inference."""
    statements = [
        str(item.get("statement") or "").strip()
        for item in reasoning.get("reasoning_steps") or []
        if str(item.get("statement") or "").strip()
    ]
    counterevidence = str(reasoning.get("counterevidence") or "").strip()
    uncertainty = str(reasoning.get("uncertainty") or "").strip()
    if counterevidence and counterevidence not in statements:
        statements.append(counterevidence)
    if uncertainty and uncertainty not in statements:
        statements.append(uncertainty)
    selected_evidence = sorted(
        {
            str(value)
            for item in reasoning.get("reasoning_steps") or []
            for value in item.get("evidence_ids") or []
            if str(value)
        }
    )
    selected_evidence.extend(
        value
        for value in reasoning.get("selected_evidence_ids") or []
        if str(value) not in selected_evidence
    )
    return {
        "schema_version": "reasoning_narrative_view_v1",
        "forecast_analysis": " ".join(statements),
        "selected_evidence_ids": selected_evidence,
        "used_path_ids": copy.deepcopy(
            (reasoning.get("causal_balance") or {}).get("used_path_ids") or []
        ),
        "evidence_backed_claim_count": sum(
            bool(item.get("evidence_ids"))
            for item in reasoning.get("reasoning_steps") or []
        ),
        "derived_from_prediction_reasoning": True,
        "new_inference_added": False,
        "probability_modified": False,
    }


def call_procedural_topology_reasoning(
    *,
    client: OpenAI,
    model: str,
    question_id: str,
    public_case: dict[str, Any],
    target_operator: dict[str, Any],
    evidence: list[dict[str, Any]],
    evidence_ledger: dict[str, Any],
    current_graph: dict[str, Any],
    worked_reasoning_checks: list[dict[str, Any]],
    max_tokens: int,
) -> tuple[dict[str, Any], dict[str, int], float, bool]:
    path_ids = [str(path["id"]) for path in current_graph.get("paths", [])]
    evidence_ids = {str(item["id"]) for item in evidence}
    graph_evidence_ids = {
        str(value)
        for path in current_graph.get("paths", [])
        for group in (path.get("checkpoints", []), path.get("edges", []))
        for item in group
        for value in (item.get("current") or {}).get("evidence_ids", [])
        if str(value) in evidence_ids
    }
    prompt = (
        "Build a complete current forecast reasoning trace using the historical DAGs "
        "as an incomplete structural scaffold. Do not choose an option, assign "
        "probabilities, or emit a target number. Start with the exact target operation "
        "and current baseline. Use helpful nodes, edges, or partial paths to organize "
        "drivers and mechanisms. Add evidence-backed current factors and intermediate "
        "reasoning when the historical graph does not cover them. Test "
        "counterevidence, end with a target bridge, and distinguish "
        "target-period magnitude support from "
        "directional support.\n\n"
        "Current evidence overrides historical structure. A relation marked UNVERIFIED "
        "is uncertain rather than false and may guide a hypothesis, but it is not a "
        "current fact. Never use an explicitly CONTRADICTED relation as positive "
        "support. Use only current article IDs for current factual claims. Select only "
        "the paths that genuinely contributed to the current synthesis in "
        "used_path_ids. "
        "The reasoning may go beyond those paths. When discussing a contradicted "
        "relation, use a counterevidence step rather than a positive driver or "
        "mechanism step.\n\n"
        "The worked reasoning checks show how the same historical DAGs previously "
        "organized a forecast-time argument. Use them after reading current "
        "evidence to "
        "check whether the current trace omitted a baseline, causal link, competing "
        "explanation, or uncertainty. Do not copy their historical facts, direction, "
        "estimate, or conclusion. They are not current evidence.\n\n"
        f"CURRENT CASE:\n{json.dumps(public_case, ensure_ascii=False)}\n\n"
        "DETERMINISTIC TARGET OPERATOR:\n"
        f"{json.dumps(target_operator, ensure_ascii=False)}\n\n"
        "CURRENT EVIDENCE LEDGER:\n"
        f"{json.dumps(evidence_ledger, ensure_ascii=False)}\n\n"
        "CURRENT-INSTANTIATED EXACT SUBGRAPHS:\n"
        f"{json.dumps(current_graph, ensure_ascii=False)}\n\n"
        "ANSWER-FREE WORKED REASONING CHECKS:\n"
        f"{json.dumps(worked_reasoning_checks, ensure_ascii=False)}\n\n"
        "CURRENT CUTOFF-SAFE EVIDENCE:\n"
        f"{json.dumps(evidence, ensure_ascii=False)}"
    )
    reasoning, _, usage, seconds, repaired = _call_with_repair(
        client,
        model=model,
        system=(
            "You are a cutoff-safe financial forecaster. Return reasoning only in "
            "a clear evidence-grounded structure, never an answer or probability."
        ),
        prompt=prompt,
        schema=_schema(path_ids),
        seed=_seed(question_id, "procedural-topology-reasoning-v3"),
        max_tokens=max_tokens,
        validator=lambda candidate: _validate(
            candidate,
            path_ids=path_ids,
            evidence_ids=evidence_ids,
            graph_evidence_ids=graph_evidence_ids,
        ),
    )
    return reasoning, usage, seconds, repaired
