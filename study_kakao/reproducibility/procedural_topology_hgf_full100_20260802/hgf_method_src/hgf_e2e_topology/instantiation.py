"""Fill exact hindsight-DAG subgraphs with current evidence without forecasting."""

from __future__ import annotations

import json
import copy
from typing import Any

from openai import OpenAI

from hgf.exemplar import _call_with_repair
from hgf.forecast_core import _seed


def graph_elements(
    routed_memory: dict[str, Any],
) -> tuple[list[str], list[str], dict[str, dict[str, Any]]]:
    path_ids: list[str] = []
    node_ids: list[str] = []
    edges: dict[str, dict[str, Any]] = {}
    for path in routed_memory.get("paths", []):
        path_ids.append(str(path["id"]))
        for checkpoint in path.get("checkpoints", []):
            checkpoint_id = str(checkpoint["id"])
            if checkpoint_id not in node_ids:
                node_ids.append(checkpoint_id)
        for edge in path.get("edges", []):
            edge_id = f"{edge['source_checkpoint_id']}->{edge['target_checkpoint_id']}"
            edges.setdefault(edge_id, edge)
    return path_ids, node_ids, edges


def materialize_current_graph(
    routed_memory: dict[str, Any], instantiated_graph: dict[str, Any]
) -> dict[str, Any]:
    """Merge immutable topology and current state into one non-duplicated graph."""
    node_states = {
        str(item.get("node_id") or ""): item
        for item in instantiated_graph.get("node_states") or []
    }
    edge_states = {
        str(item.get("edge_id") or ""): item
        for item in instantiated_graph.get("edge_states") or []
    }
    path_states = {
        str(item.get("path_id") or ""): item
        for item in instantiated_graph.get("path_states") or []
    }
    paths = []
    for path in routed_memory.get("paths", []):
        checkpoints = []
        for checkpoint in path.get("checkpoints", []):
            node_id = str(checkpoint["id"])
            checkpoints.append(
                {
                    "id": node_id,
                    "role": checkpoint.get("role"),
                    "factor": checkpoint.get("factor"),
                    "mechanism": checkpoint.get("mechanism"),
                    "expected_direction": checkpoint.get("expected_direction"),
                    "evidence_requirement": checkpoint.get("evidence_requirement"),
                    "contradiction_signal": checkpoint.get("contradiction_signal"),
                    "historical_support": checkpoint.get("historical_support"),
                    "current": copy.deepcopy(node_states.get(node_id, {})),
                }
            )
        edges = []
        for edge in path.get("edges", []):
            edge_id = f"{edge['source_checkpoint_id']}->{edge['target_checkpoint_id']}"
            edges.append(
                {
                    "id": edge_id,
                    "source": edge.get("source_checkpoint_id"),
                    "target": edge.get("target_checkpoint_id"),
                    "relationship": edge.get("relationship"),
                    "directionality": edge.get("directionality"),
                    "historical_support": edge.get("support_level"),
                    "rationale": edge.get("rationale"),
                    "current": copy.deepcopy(edge_states.get(edge_id, {})),
                }
            )
        path_id = str(path["id"])
        paths.append(
            {
                "id": path_id,
                "source_question_id": path.get("source_question_id"),
                "role": path.get("source_path_role"),
                "mechanism": path.get("mechanism"),
                "applicability_conditions": copy.deepcopy(
                    path.get("applicability_conditions", [])
                ),
                "failure_conditions": copy.deepcopy(
                    path.get("failure_conditions", [])
                ),
                "checkpoints": checkpoints,
                "edges": edges,
                "current": copy.deepcopy(path_states.get(path_id, {})),
            }
        )
    return {
        "schema_version": "current_instantiated_exact_subgraphs_v1",
        "paths": paths,
        "graph_synthesis": copy.deepcopy(
            instantiated_graph.get("graph_synthesis", {})
        ),
        "contract": {
            "historical_answer": "excluded",
            "historical_probability": "excluded",
            "topology_rewrite": False,
        },
    }


def _schema(
    *, path_ids: list[str], node_ids: list[str], edge_ids: list[str]
) -> dict[str, Any]:
    evidence_ids = {
        "type": "array",
        "maxItems": 4,
        "items": {"type": "string"},
    }
    return {
        "name": "current_dag_instantiation",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "node_states": {
                    "type": "array",
                    "minItems": len(node_ids),
                    "maxItems": len(node_ids),
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "node_id": {"type": "string", "enum": node_ids},
                            "relation": {
                                "type": "string",
                                "enum": [
                                    "ALIGNED",
                                    "REVERSED",
                                    "UNOBSERVED",
                                    "STRUCTURAL",
                                ],
                            },
                            "value": {"type": "string", "maxLength": 120},
                            "time": {
                                "type": "string",
                                "enum": [
                                    "target_period",
                                    "leading",
                                    "lagging",
                                    "structural",
                                    "unknown",
                                ],
                            },
                            "evidence_ids": evidence_ids,
                            "confidence": {
                                "type": "string",
                                "enum": ["high", "medium", "low"],
                            },
                        },
                        "required": [
                            "node_id",
                            "relation",
                            "value",
                            "time",
                            "evidence_ids",
                            "confidence",
                        ],
                    },
                },
                "edge_states": {
                    "type": "array",
                    "minItems": len(edge_ids),
                    "maxItems": len(edge_ids),
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "edge_id": {"type": "string", "enum": edge_ids},
                            "relation": {
                                "type": "string",
                                "enum": [
                                    "PRESERVED",
                                    "REVERSED",
                                    "CONTRADICTED",
                                    "UNVERIFIED",
                                ],
                            },
                            "lag": {
                                "type": "string",
                                "enum": [
                                    "immediate",
                                    "short",
                                    "medium",
                                    "long",
                                    "unknown",
                                ],
                            },
                            "support": {
                                "type": "string",
                                "enum": ["direct", "indirect", "structural", "none"],
                            },
                            "evidence_ids": evidence_ids,
                            "confidence": {
                                "type": "string",
                                "enum": ["high", "medium", "low"],
                            },
                        },
                        "required": [
                            "edge_id",
                            "relation",
                            "lag",
                            "support",
                            "evidence_ids",
                            "confidence",
                        ],
                    },
                },
                "path_states": {
                    "type": "array",
                    "minItems": len(path_ids),
                    "maxItems": len(path_ids),
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "path_id": {"type": "string", "enum": path_ids},
                            "status": {
                                "type": "string",
                                "enum": ["ACTIVE", "CONTRADICTED", "UNRESOLVED"],
                            },
                            "effect_on_target": {
                                "type": "string",
                                "enum": ["up", "down", "neutral", "mixed", "uncertain"],
                            },
                        },
                        "required": ["path_id", "status", "effect_on_target"],
                    },
                },
                "graph_synthesis": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "active_path_ids": {
                            "type": "array",
                            "items": {"type": "string", "enum": path_ids},
                        },
                        "direction": {
                            "type": "string",
                            "enum": ["up", "down", "balanced", "uncertain"],
                        },
                        "assessment": {"type": "string", "maxLength": 550},
                    },
                    "required": ["active_path_ids", "direction", "assessment"],
                },
            },
            "required": [
                "node_states",
                "edge_states",
                "path_states",
                "graph_synthesis",
            ],
        },
    }


def _filter_ids(values: Any, allowed: set[str]) -> list[str]:
    return sorted({str(value) for value in (values or []) if str(value) in allowed})


def validate_instantiation(
    payload: dict[str, Any],
    *,
    evidence_ids: set[str],
    path_ids: list[str],
    node_ids: list[str],
    edge_ids: list[str],
) -> tuple[dict[str, float], list[str]]:
    def complete(
        field: str,
        id_field: str,
        expected: list[str],
        fallback: Any,
    ) -> dict[str, dict[str, Any]]:
        returned: dict[str, dict[str, Any]] = {}
        for item in payload.get(field) or []:
            if not isinstance(item, dict):
                continue
            item_id = str(item.get(id_field) or "")
            if item_id in expected and item_id not in returned:
                item["evidence_ids"] = _filter_ids(
                    item.get("evidence_ids"), evidence_ids
                )
                returned[item_id] = item
        for item_id in expected:
            returned.setdefault(item_id, fallback(item_id))
        payload[field] = [returned[item_id] for item_id in expected]
        return returned

    complete(
        "node_states",
        "node_id",
        node_ids,
        lambda item_id: {
            "node_id": item_id,
            "relation": "UNOBSERVED",
            "value": "unknown",
            "time": "unknown",
            "evidence_ids": [],
            "confidence": "low",
        },
    )
    complete(
        "edge_states",
        "edge_id",
        edge_ids,
        lambda item_id: {
            "edge_id": item_id,
            "relation": "UNVERIFIED",
            "lag": "unknown",
            "support": "none",
            "evidence_ids": [],
            "confidence": "low",
        },
    )
    paths: dict[str, dict[str, Any]] = {}
    for item in payload.get("path_states") or []:
        if not isinstance(item, dict):
            continue
        path_id = str(item.get("path_id") or "")
        if path_id in path_ids and path_id not in paths:
            paths[path_id] = item
    for path_id in path_ids:
        paths.setdefault(
            path_id,
            {
                "path_id": path_id,
                "status": "UNRESOLVED",
                "effect_on_target": "uncertain",
            },
        )
    payload["path_states"] = [paths[path_id] for path_id in path_ids]
    synthesis = payload.get("graph_synthesis") or {}
    synthesis["active_path_ids"] = sorted(
        path_id
        for path_id, item in paths.items()
        if item.get("status") == "ACTIVE"
    )
    return {}, []


def call_graph_instantiation(
    *,
    client: OpenAI,
    model: str,
    question_id: str,
    public_case: dict[str, Any],
    target_operator: dict[str, Any],
    evidence: list[dict[str, Any]],
    evidence_ledger: dict[str, Any],
    routed_memory: dict[str, Any],
    max_tokens: int,
) -> tuple[dict[str, Any], dict[str, int], float, bool]:
    path_ids, node_ids, edge_map = graph_elements(routed_memory)
    edge_ids = list(edge_map)
    evidence_ids = {str(item["id"]) for item in evidence}
    prompt = (
        "Fill the supplied hindsight-DAG subgraphs with current cutoff-safe evidence. "
        "Do not forecast, choose an answer, or assign probabilities. Keep every node "
        "and edge and return only compact state. For nodes, record whether the current "
        "event aligns, reverses, leaves it unobserved, or uses it only structurally, "
        "along with value, timing, evidence, and confidence. For edges, record whether "
        "the original relation is preserved, reversed, contradicted, or unverified, "
        "along with lag and current support. Historical topology is structural "
        "knowledge, not current evidence. Direct current evidence overrides it. An "
        "unobserved mediator does not by itself invalidate an otherwise applicable "
        "mechanism. Mark each complete path active, contradicted, or unresolved.\n\n"
        f"CURRENT CASE:\n{json.dumps(public_case, ensure_ascii=False)}\n\n"
        "DETERMINISTIC TARGET OPERATOR:\n"
        f"{json.dumps(target_operator, ensure_ascii=False)}\n\n"
        "CURRENT EVIDENCE LEDGER:\n"
        f"{json.dumps(evidence_ledger, ensure_ascii=False)}\n\n"
        "ROUTED EXACT SUBGRAPHS:\n"
        f"{json.dumps(routed_memory, ensure_ascii=False)}\n\n"
        "CURRENT CUTOFF-SAFE EVIDENCE:\n"
        f"{json.dumps(evidence, ensure_ascii=False)}"
    )
    graph, _, usage, seconds, repaired = _call_with_repair(
        client,
        model=model,
        system=(
            "You instantiate exact financial DAG subgraphs with current evidence. "
            "You never forecast. Return compact JSON."
        ),
        prompt=prompt,
        schema=_schema(path_ids=path_ids, node_ids=node_ids, edge_ids=edge_ids),
        seed=_seed(question_id, "current-dag-instantiation-v1"),
        max_tokens=max_tokens,
        validator=lambda candidate: validate_instantiation(
            candidate,
            evidence_ids=evidence_ids,
            path_ids=path_ids,
            node_ids=node_ids,
            edge_ids=edge_ids,
        ),
    )
    return graph, usage, seconds, repaired
