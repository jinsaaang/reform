"""Compile outcome-neutral causal templates from resolved-event DAGs."""

from __future__ import annotations

import copy
import json
import re
import threading
from pathlib import Path
from typing import Any

from openai import OpenAI

from .exemplar import _call_with_repair
from .forecast_core import _atomic_write, _seed


_CACHE_LOCK = threading.Lock()
_HISTORICAL_STATE_PATTERN = re.compile(
    r"\b(?:19|20)\d{2}\b|"
    r"\b(?:above|below|within) recent range\b|"
    r"\bresolved outcome\b",
    flags=re.IGNORECASE,
)
_REALIZED_FACTOR_PATTERN = re.compile(
    r"\b(?:january|february|march|april|may|june|july|august|"
    r"september|october|november|december)\b|"
    r"\b(?:reported|recorded|added|rose|fell|was|were|had)\b|"
    r"\b\d{1,3}(?:,\d{3})+\b|"
    r"\b\d+(?:\.\d+)?\s*(?:%|percent|percentage points?)\b|"
    r"\bcomparable to\b",
    flags=re.IGNORECASE,
)
_GENERIC_MECHANISM_PATTERN = re.compile(
    r"test whether this historically relevant factor|"
    r"connects? to the target metric",
    flags=re.IGNORECASE,
)


def _redact_historical_markers(value: Any) -> Any:
    if isinstance(value, str):
        return _HISTORICAL_STATE_PATTERN.sub(
            "[historical state removed]",
            value,
        )
    if isinstance(value, list):
        return [_redact_historical_markers(item) for item in value]
    if isinstance(value, dict):
        return {
            key: _redact_historical_markers(item)
            for key, item in value.items()
        }
    return value


def _abstract_factor_label(value: str) -> str:
    cleaned = re.sub(
        r"\b(?:historical|reported|recorded)\b",
        "",
        value,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"\b(?:january|february|march|april|may|june|july|august|"
        r"september|october|november|december)\b",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.split(
        r"\b(?:was|were|rose|fell|added|had|reached)\b",
        cleaned,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    cleaned = re.sub(
        r"\b\d{1,3}(?:,\d{3})+\b|"
        r"\b\d+(?:\.\d+)?\s*(?:%|percent|percentage points?)\b",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,.-")
    return cleaned or "current financial driver"


def _is_deterministically_compiled(
    blueprint: dict[str, Any],
) -> bool:
    model = str(
        blueprint.get("refinement_metadata", {}).get("model") or ""
    )
    return "deterministic_compiler" in model


def _normalize_guidance_blueprint(
    blueprint: dict[str, Any],
) -> dict[str, Any]:
    """Keep conditional mechanisms while removing historical provenance."""
    factor_checks = []
    target_bridge = {}
    for item in blueprint.get("checkpoints", []):
        checkpoint = {
            "id": str(item.get("id") or ""),
            "causal_role": str(item.get("role") or ""),
            "factor": str(item.get("factor") or ""),
            "state_question": (
                "What is the current state of this factor, and which current "
                "evidence establishes it?"
            ),
            "mechanism": str(item.get("mechanism") or ""),
            "effect_if_active": str(
                item.get("expected_direction") or "mixed"
            ),
            "evidence_requirement": str(
                item.get("evidence_requirement") or ""
            ),
            "failure_signal": str(
                item.get("contradiction_signal") or ""
            ),
        }
        if item.get("role") == "target_bridge":
            target_bridge = checkpoint
        elif checkpoint["id"] and checkpoint["factor"]:
            factor_checks.append(checkpoint)
    factor_ids = {item["id"] for item in factor_checks}
    paths = []
    for index, item in enumerate(
        blueprint.get("causal_paths", []),
        start=1,
    ):
        kept_ids = [
            str(value)
            for value in item.get("checkpoint_ids", [])
            if str(value) in factor_ids
        ]
        if not kept_ids:
            continue
        paths.append(
            {
                "id": f"path_{index}",
                "factor_ids": kept_ids,
                "conditional_mechanism": str(
                    item.get("generalized_mechanism") or ""
                ),
                "effect_if_active": str(
                    item.get("expected_direction") or "mixed"
                ),
                "activation_condition": " and ".join(
                    str(value)
                    for value in item.get(
                        "applicability_conditions", []
                    )
                ),
                "failure_conditions": [
                    str(value)
                    for value in item.get("failure_conditions", [])
                ],
            }
        )
    return {
        "schema_version": "outcome_neutral_dag_template_v1",
        "target_operation": str(
            blueprint.get("target_definition", {}).get("metric") or ""
        ),
        "factor_checks": factor_checks,
        "conditional_paths": paths,
        "target_bridge": target_bridge,
        "competing_explanations": [
            {
                "hypothesis": str(item.get("hypothesis") or ""),
                "discriminating_evidence": str(
                    item.get("discriminating_evidence") or ""
                ),
            }
            for item in blueprint.get("alternative_hypotheses", [])[:2]
        ],
        "worked_procedure": [
            "Lock the current target operation and establish its baseline.",
            "Fill each factor slot using current cutoff-safe evidence only.",
            "Activate a conditional path only when every necessary link is "
            "supported in the current case.",
            "Test the path against its failure conditions and a competing "
            "explanation.",
            "Estimate target-period magnitude before mapping to an option.",
        ],
    }


def _graph_view(graph_payload: dict[str, Any]) -> dict[str, Any]:
    """Expose topology and rationale without outcome values or timestamps."""
    graph = graph_payload.get("graph", {})
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    outcome_id = str(
        graph_payload.get("actual_outcome_event_id") or ""
    )

    def node_id(node: dict[str, Any]) -> str:
        return str(node.get("id") or "")

    def edge_source(edge: dict[str, Any]) -> str:
        return str(
            edge.get("source")
            or edge.get("source_event_id")
            or ""
        )

    def edge_target(edge: dict[str, Any]) -> str:
        return str(
            edge.get("target")
            or edge.get("target_event_id")
            or ""
        )

    target_nodes = {
        edge_source(edge)
        for edge in edges
        if edge_target(edge) == outcome_id
    }
    excluded = {
        node_id(node)
        for node in nodes
        if (
            node.get("is_outcome")
            or node.get("is_actual_outcome")
            or node_id(node) == outcome_id
        )
    }
    kept_nodes = []
    for node in nodes:
        current_id = node_id(node)
        if not current_id or current_id in excluded:
            continue
        if current_id in target_nodes:
            kept_nodes.append(
                {
                    "id": "TARGET",
                    "factor_state": "current target-period value",
                    "event_type": "target",
                }
            )
            continue
        kept_nodes.append(
            {
                "id": current_id,
                "factor_state": str(
                    node.get("label") or node.get("title") or ""
                ),
                "event_type": str(node.get("event_type") or ""),
            }
        )
    kept_ids = {str(item["id"]) for item in kept_nodes}
    kept_edges = []
    for edge in edges:
        source = (
            "TARGET"
            if edge_source(edge) in target_nodes
            else edge_source(edge)
        )
        target = (
            "TARGET"
            if edge_target(edge) in target_nodes
            else edge_target(edge)
        )
        if (
            source not in kept_ids
            or target not in kept_ids
            or source == target
        ):
            continue
        kept_edges.append(
            {
                "source": source,
                "target": target,
                "relation": str(
                    edge.get("relationship")
                    or edge.get("relation_type")
                    or ""
                ),
                "rationale": str(
                    edge.get("rationale")
                    or edge.get("reasoning")
                    or ""
                ),
            }
        )
    return {"nodes": kept_nodes, "edges": kept_edges}


def _neutral_template_schema(
    checkpoint_ids: list[str],
) -> dict[str, Any]:
    return {
        "name": "outcome_neutral_dag_template",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "schema_version": {
                    "type": "string",
                    "enum": ["outcome_neutral_dag_template_v1"],
                },
                "target_operation": {
                    "type": "string",
                    "maxLength": 320,
                },
                "factor_checks": {
                    "type": "array",
                    "minItems": 2,
                    "maxItems": 7,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "id": {
                                "type": "string",
                                "enum": checkpoint_ids,
                            },
                            "factor": {
                                "type": "string",
                                "maxLength": 160,
                            },
                            "state_question": {
                                "type": "string",
                                "maxLength": 320,
                            },
                            "mechanism": {
                                "type": "string",
                                "maxLength": 600,
                            },
                            "effect_if_active": {
                                "type": "string",
                                "enum": [
                                    "up",
                                    "down",
                                    "mixed",
                                    "uncertain",
                                ],
                            },
                            "evidence_requirement": {
                                "type": "string",
                                "maxLength": 400,
                            },
                            "failure_signal": {
                                "type": "string",
                                "maxLength": 400,
                            },
                        },
                        "required": [
                            "id",
                            "factor",
                            "state_question",
                            "mechanism",
                            "effect_if_active",
                            "evidence_requirement",
                            "failure_signal",
                        ],
                    },
                },
                "conditional_paths": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 3,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "id": {
                                "type": "string",
                                "maxLength": 100,
                            },
                            "factor_ids": {
                                "type": "array",
                                "minItems": 1,
                                "items": {
                                    "type": "string",
                                    "enum": checkpoint_ids,
                                },
                            },
                            "conditional_mechanism": {
                                "type": "string",
                                "maxLength": 700,
                            },
                            "effect_if_active": {
                                "type": "string",
                                "enum": [
                                    "up",
                                    "down",
                                    "mixed",
                                    "uncertain",
                                ],
                            },
                            "activation_condition": {
                                "type": "string",
                                "maxLength": 400,
                            },
                            "failure_conditions": {
                                "type": "array",
                                "maxItems": 3,
                                "items": {
                                    "type": "string",
                                    "maxLength": 350,
                                },
                            },
                        },
                        "required": [
                            "id",
                            "factor_ids",
                            "conditional_mechanism",
                            "effect_if_active",
                            "activation_condition",
                            "failure_conditions",
                        ],
                    },
                },
                "target_bridge": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "mechanism": {
                            "type": "string",
                            "maxLength": 600,
                        },
                        "magnitude_requirement": {
                            "type": "string",
                            "maxLength": 400,
                        },
                        "failure_signal": {
                            "type": "string",
                            "maxLength": 400,
                        },
                    },
                    "required": [
                        "mechanism",
                        "magnitude_requirement",
                        "failure_signal",
                    ],
                },
                "competing_explanations": {
                    "type": "array",
                    "maxItems": 2,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "hypothesis": {
                                "type": "string",
                                "maxLength": 400,
                            },
                            "discriminating_evidence": {
                                "type": "string",
                                "maxLength": 400,
                            },
                        },
                        "required": [
                            "hypothesis",
                            "discriminating_evidence",
                        ],
                    },
                },
                "worked_procedure": {
                    "type": "array",
                    "minItems": 4,
                    "maxItems": 6,
                    "items": {"type": "string", "maxLength": 320},
                },
            },
            "required": [
                "schema_version",
                "target_operation",
                "factor_checks",
                "conditional_paths",
                "target_bridge",
                "competing_explanations",
                "worked_procedure",
            ],
        },
    }


def _validate_neutral_template(
    payload: dict[str, Any],
    checkpoint_ids: set[str],
) -> tuple[dict[str, float], list[str]]:
    redacted = _redact_historical_markers(payload)
    payload.clear()
    payload.update(redacted)
    errors = []
    factors = payload.get("factor_checks", [])
    factor_ids = [str(item.get("id") or "") for item in factors]
    if len(factor_ids) != len(set(factor_ids)):
        errors.append("factor IDs are not unique")
    if set(factor_ids) - checkpoint_ids:
        errors.append("template introduced unknown factor IDs")
    for path in payload.get("conditional_paths", []):
        if not set(path.get("factor_ids", [])) <= set(factor_ids):
            errors.append("path references a missing factor")
    for factor in factors:
        factor_text = (
            f"{factor.get('factor', '')} "
            f"{factor.get('state_question', '')}"
        )
        if _REALIZED_FACTOR_PATTERN.search(factor_text):
            factor["factor"] = _abstract_factor_label(
                str(factor.get("factor") or "")
            )
            factor["state_question"] = (
                f"What is the current state of {factor['factor']}, "
                "and which current evidence establishes it?"
            )
        if _GENERIC_MECHANISM_PATTERN.search(
            str(factor.get("mechanism") or "")
        ):
            factor["mechanism"] = (
                f"Changes in {factor['factor']} enter the validated "
                "conditional DAG path to the forecast target."
            )
    procedure = payload.get("worked_procedure", [])
    if procedure and not re.search(
        r"magnitude|boundary|option",
        str(procedure[-1]),
        flags=re.IGNORECASE,
    ):
        procedure.append(
            "Assess target-period magnitude and map it to the public option "
            "boundary."
        )
        payload["worked_procedure"] = procedure[-6:]
    if not str(payload.get("target_operation") or "").strip():
        errors.append("target operation is empty")
    return {}, errors


def compile_outcome_neutral_template(
    *,
    client: OpenAI,
    model: str,
    source_question_id: str,
    blueprint: dict[str, Any],
    graph_payload: dict[str, Any],
    cache_dir: Path,
    max_tokens: int,
) -> tuple[dict[str, Any], dict[str, int], float, bool, bool]:
    """Return a cached neutral template, using generation only when needed."""
    if not _is_deterministically_compiled(blueprint):
        return (
            _normalize_guidance_blueprint(blueprint),
            {},
            0.0,
            True,
            False,
        )

    cache_path = cache_dir / f"{source_question_id}.json"
    checkpoint_ids = [
        str(item.get("id"))
        for item in blueprint.get("checkpoints", [])
        if item.get("id") and item.get("role") != "target_bridge"
    ][:7]
    if len(checkpoint_ids) < 2:
        raise ValueError(
            f"{source_question_id} has fewer than two transferable factors"
        )
    checkpoint_roles = {
        str(item.get("id")): str(item.get("role") or "")
        for item in blueprint.get("checkpoints", [])
        if item.get("id")
    }

    def inject_roles(template: dict[str, Any]) -> dict[str, Any]:
        for factor in template.get("factor_checks", []):
            factor["causal_role"] = checkpoint_roles.get(
                str(factor.get("id") or ""),
                "",
            )
        return template

    with _CACHE_LOCK:
        if cache_path.exists():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            _, errors = _validate_neutral_template(
                cached,
                set(checkpoint_ids),
            )
            if not errors:
                return inject_roles(cached), {}, 0.0, True, False

    prompt = (
        "Convert the supplied resolved-event DAG into an outcome-neutral "
        "forecasting template for the same recurring financial target. Preserve "
        "the economic causal relationships and counteracting paths, but replace "
        "realized historical states with timeless factor variables and questions "
        "about their current state. A factor must be a short noun phrase such as "
        "'payroll growth momentum', 'labor demand', or 'policy stance'. It must "
        "not say that a value was reported, rose, fell, or reached a historical "
        "number. A state question must ask whether the variable is strengthening, "
        "weakening, positive, or negative now. It must never compare the current "
        "case with a historical value. Do not retain months, dates, historical "
        "values, named episodes, realized directions, resolved options, or the "
        "old conclusion. "
        "A signed effect is allowed only as a conditional relationship, such as "
        "higher risk aversion raises implied volatility. Do not claim that the "
        "condition currently holds. Every mechanism must state an economic link. "
        "Never use generic wording such as 'test whether the factor connects to "
        "the target'. The worked procedure must describe the order "
        "of reasoning with empty current-evidence slots. It must end by requiring "
        "magnitude support before mapping direction to the public option "
        "boundary.\n\n"
        f"TARGET OPERATION:\n"
        f"{str(blueprint.get('target_definition', {}).get('metric') or '')}"
        "\n\nLOSSY DAG CARDS:\n"
        f"{json.dumps(blueprint, ensure_ascii=False)}"
        "\n\nREDACTED DAG TOPOLOGY:\n"
        f"{json.dumps(_graph_view(graph_payload), ensure_ascii=False)}"
    )
    try:
        template, _, usage, seconds, repaired = _call_with_repair(
            client,
            model=model,
            system=(
                "You faithfully abstract a validated financial causal DAG into "
                "a conditional, outcome-neutral forecasting template. Return "
                "JSON."
            ),
            prompt=prompt,
            schema=_neutral_template_schema(checkpoint_ids),
            seed=_seed(source_question_id, "outcome-neutral-template"),
            max_tokens=max_tokens,
            reasoning_effort="low",
            validator=lambda payload: _validate_neutral_template(
                payload,
                set(checkpoint_ids),
            ),
        )
    except ValueError as exc:
        if "invalid/truncated JSON" not in str(exc):
            raise
        template = _normalize_guidance_blueprint(blueprint)
        usage = {}
        seconds = 0.0
        repaired = True
    with _CACHE_LOCK:
        _atomic_write(cache_path, inject_roles(template))
    return inject_roles(template), usage, seconds, False, repaired


def merge_neutral_templates(
    templates: list[dict[str, Any]],
) -> dict[str, Any]:
    """Namespace and merge only distinct conditional paths."""
    if not templates:
        raise ValueError("at least one neutral template is required")
    merged = {
        "schema_version": "adaptive_neutral_dag_memory_v1",
        "target_operation": templates[0].get("target_operation", ""),
        "factor_checks": [],
        "topology_edges": [],
        "conditional_paths": [],
        "target_bridges": [],
        "competing_explanations": [],
        "worked_procedure": copy.deepcopy(
            templates[0].get("worked_procedure", [])
        ),
    }
    seen_paths: set[str] = set()
    for template_index, template in enumerate(templates, start=1):
        id_map = {}
        edge_id_map = {}
        for factor in template.get("factor_checks", []):
            old_id = str(factor.get("id") or "")
            new_id = f"T{template_index}:{old_id}"
            id_map[old_id] = new_id
            merged["factor_checks"].append(
                {**copy.deepcopy(factor), "id": new_id}
            )
        for edge in template.get("topology_edges", []):
            old_edge_id = str(edge.get("id") or "")
            new_edge_id = f"T{template_index}:{old_edge_id}"
            edge_id_map[old_edge_id] = new_edge_id
            merged["topology_edges"].append(
                {
                    **copy.deepcopy(edge),
                    "id": new_edge_id,
                    "source_checkpoint_id": id_map.get(
                        str(edge.get("source_checkpoint_id") or ""),
                        f"T{template_index}:target_bridge",
                    ),
                    "target_checkpoint_id": id_map.get(
                        str(edge.get("target_checkpoint_id") or ""),
                        f"T{template_index}:target_bridge",
                    ),
                }
            )
        for path in template.get("conditional_paths", []):
            mechanism = str(path.get("conditional_mechanism") or "")
            signature = re.sub(r"\s+", " ", mechanism.lower()).strip()
            if signature in seen_paths:
                continue
            seen_paths.add(signature)
            merged["conditional_paths"].append(
                {
                    **copy.deepcopy(path),
                    "id": (
                        f"T{template_index}:"
                        f"{str(path.get('id') or 'path')}"
                    ),
                    "factor_ids": [
                        id_map[str(value)]
                        for value in path.get("factor_ids", [])
                        if str(value) in id_map
                    ],
                    "edge_ids": [
                        edge_id_map[str(value)]
                        for value in path.get("edge_ids", [])
                        if str(value) in edge_id_map
                    ],
                    "source_factor_id": (
                        id_map[str(path.get("factor_ids", [])[0])]
                        if (
                            path.get("factor_ids")
                            and str(path.get("factor_ids", [])[0])
                            in id_map
                        )
                        else ""
                    ),
                }
            )
        merged["target_bridges"].append(
            copy.deepcopy(template.get("target_bridge", {}))
        )
        merged["competing_explanations"].extend(
            copy.deepcopy(
                template.get("competing_explanations", [])
            )
        )
    merged["conditional_paths"] = merged["conditional_paths"][:6]
    selected_edge_ids = {
        str(edge_id)
        for path in merged["conditional_paths"]
        for edge_id in path.get("edge_ids", [])
    }
    merged["topology_edges"] = [
        edge
        for edge in merged["topology_edges"]
        if str(edge.get("id") or "") in selected_edge_ids
    ]
    merged["competing_explanations"] = (
        merged["competing_explanations"][:3]
    )
    return merged
