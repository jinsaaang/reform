"""Frozen procedural HGF reasoning over a current-instantiated exact topology."""

from __future__ import annotations

import copy
import json
from typing import Any

from openai import OpenAI

from hgf.exemplar import _call_with_repair
from hgf.forecast_core import _seed


def _schema(path_ids: list[str]) -> dict[str, Any]:
    source_ids = path_ids + ["CURRENT_NEW", "TARGET_CONTRACT"]
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
                        "active_path_ids": {
                            "type": "array",
                            "items": {"type": "string", "enum": path_ids},
                        },
                        "assessment": {"type": "string", "maxLength": 700},
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
                        "assessment": {"type": "string", "maxLength": 600},
                    },
                    "required": ["support", "evidence_ids", "assessment"],
                },
                "reasoning_steps": {
                    "type": "array",
                    "minItems": 3,
                    "maxItems": 6,
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
    payload.setdefault("target_semantics", "Target semantics are defined by the public contract.")
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
            "active_path_ids": [],
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
    active = _filter(
        (payload.get("causal_balance") or {}).get("active_path_ids"),
        set(path_ids),
    )
    payload.setdefault("causal_balance", {})["active_path_ids"] = active
    steps = payload.get("reasoning_steps") or []
    step_types: set[str] = set()
    for item in steps:
        item["evidence_ids"] = _filter(item.get("evidence_ids"), evidence_ids)
        step_type = str(item.get("step_type") or "")
        step_types.add(step_type)
        source_id = str(item.get("source_id") or "")
        if step_type in {"driver", "mechanism"} and source_id in path_ids:
            if source_id not in active:
                item["step_type"] = "counterevidence"
                item["effect_on_target"] = "uncertain"
    if "baseline" not in step_types:
        steps.insert(
            0,
            {
                "step_type": "baseline",
                "statement": str(payload.get("target_semantics") or "")[:500],
                "evidence_ids": [],
                "effect_on_target": "uncertain",
                "source_id": "TARGET_CONTRACT",
            },
        )
    if "target_bridge" not in step_types:
        if len(steps) >= 6:
            steps.pop()
        active_source = active[0] if active else "TARGET_CONTRACT"
        steps.append(
            {
                "step_type": "target_bridge",
                "statement": str(
                    (payload.get("causal_balance") or {}).get("assessment") or ""
                )[:500],
                "evidence_ids": sorted(graph_evidence_ids)[:6],
                "effect_on_target": str(
                    (payload.get("causal_balance") or {}).get("favored_direction")
                    or "uncertain"
                ),
                "source_id": active_source,
            }
        )
    payload["reasoning_steps"] = steps
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
    active_ids = {
        str(value)
        for value in result.get("causal_balance", {}).get("active_path_ids", [])
        if str(value) in path_by_id
    }
    active_sources = {
        str(path_by_id[path_id].get("source_question_id") or "")
        for path_id in active_ids
    }
    effects = {
        str(item.get("effect_on_target") or "")
        for item in result["path_assessments"]
        if str(item.get("path_id") or "") in active_ids
    }
    conflict = {"up", "down"} <= effects or "mixed" in effects
    summary = (
        f"Structural support facts: {len(active_ids)} of {len(path_by_id)} routed "
        f"paths are active across {len(active_sources)} resolved-event DAGs. "
        f"Directional conflict is {'present' if conflict else 'absent'}."
    )
    result["structural_support_summary"] = {
        "active_path_count": len(active_ids),
        "routed_path_count": len(path_by_id),
        "active_source_dag_count": len(active_sources),
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
        "Build the current forecast reasoning trace using the frozen procedural HGF "
        "order and the current-instantiated exact topology. Do not choose an option, "
        "assign probabilities, or emit a target number. Start with the exact target "
        "operation and current baseline. Identify current drivers. Follow active paths "
        "through their preserved or reversed edges to explain mechanisms. Include "
        "evidence-backed current factors absent from the graph. Test counterevidence. "
        "End with a target bridge and state separately whether evidence supports "
        "target-period magnitude or only direction.\n\n"
        "Treat the supplied graph as the current state of the historical "
        "reasoning structure. Keep its path status, direction, lag, support, and "
        "confidence. Direct current evidence overrides structure. Unverified paths may "
        "be used only as uncertainty. Use only current article IDs for factual claims. "
        "Historical answers, probabilities, and worked conclusions are unavailable.\n\n"
        f"CURRENT CASE:\n{json.dumps(public_case, ensure_ascii=False)}\n\n"
        "DETERMINISTIC TARGET OPERATOR:\n"
        f"{json.dumps(target_operator, ensure_ascii=False)}\n\n"
        "CURRENT EVIDENCE LEDGER:\n"
        f"{json.dumps(evidence_ledger, ensure_ascii=False)}\n\n"
        "CURRENT-INSTANTIATED EXACT SUBGRAPHS:\n"
        f"{json.dumps(current_graph, ensure_ascii=False)}\n\n"
        "CURRENT CUTOFF-SAFE EVIDENCE:\n"
        f"{json.dumps(evidence, ensure_ascii=False)}"
    )
    reasoning, _, usage, seconds, repaired = _call_with_repair(
        client,
        model=model,
        system=(
            "You are a cutoff-safe financial forecaster. Return reasoning only in "
            "the frozen procedural HGF order, never an answer or probability."
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
