"""Shared evidence, schema, scoring, and forecast-DAG primitives."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

from openai import OpenAI

from hgf.generation import completion_parameters, seed_suffix


def _seed(question_id: str, stage: str) -> int:
    """Derive a deterministic per-question, per-stage seed."""
    seed_material = f"{question_id}:{stage}{seed_suffix()}"
    return (
        int(
            hashlib.sha256(seed_material.encode()).hexdigest()[:8],
            16,
        )
        % 2_147_483_647
    )


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _forecast_schema(options: list[str], graph_arm: bool) -> dict[str, Any]:
    properties: dict[str, Any] = {
        "prediction": {"type": "string", "enum": options},
        "option_probabilities": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "option": {"type": "string", "enum": options},
                    "probability": {
                        "type": "number",
                    },
                },
                "required": ["option", "probability"],
            },
        },
        "decision_rationale": {"type": "string"},
        "counterevidence_rationale": {"type": "string"},
    }
    required = [
        "prediction",
        "option_probabilities",
        "decision_rationale",
        "counterevidence_rationale",
    ]
    if graph_arm:
        graph_factor_ids = ["BASE", "D1", "D2", "C1", "M", "B", "T"]
        properties["target_estimate"] = {"type": "string"}
        properties["dominant_factor_ids"] = {
            "type": "array",
            "items": {"type": "string", "enum": graph_factor_ids},
        }
        properties["counter_factor_ids"] = {
            "type": "array",
            "items": {"type": "string", "enum": graph_factor_ids},
        }
        properties["cited_evidence_article_ids"] = {
            "type": "array",
            "items": {"type": "string"},
        }
        required.extend(
            [
                "target_estimate",
                "dominant_factor_ids",
                "counter_factor_ids",
                "cited_evidence_article_ids",
            ]
        )
    else:
        properties["evidence_article_ids"] = {
            "type": "array",
            "items": {"type": "string"},
        }
        required.append("evidence_article_ids")
    return {
        "name": (
            "graph_evidence_probability_forecast"
            if graph_arm
            else "evidence_only_probability_forecast"
        ),
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": properties,
            "required": required,
        },
    }


def _single_dag_plan_schema() -> dict[str, Any]:
    factor = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "label": {"type": "string"},
            "observation": {"type": "string"},
            "temporal_relation": {
                "type": "string",
                "enum": [
                    "historical_baseline",
                    "target_period_signal",
                    "structural_context",
                ],
            },
            "directional_effect": {"type": "string"},
            "checkpoint_id": {"type": "string"},
            "evidence_article_ids": {
                "type": "array",
                "items": {"type": "string"},
            },
            "mechanism_to_synthesis": {"type": "string"},
        },
        "required": [
            "label",
            "observation",
            "temporal_relation",
            "directional_effect",
            "checkpoint_id",
            "evidence_article_ids",
            "mechanism_to_synthesis",
        ],
    }
    return {
        "name": "single_dag_plan",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "baseline": factor,
                "drivers": {"type": "array", "items": factor},
                "countervailing_factors": {
                    "type": "array",
                    "items": factor,
                },
                "synthesis": factor,
                "target_bridge": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "target_metric": {"type": "string"},
                        "level_change_distinction": {"type": "string"},
                        "boundary_mapping": {"type": "string"},
                    },
                    "required": [
                        "target_metric",
                        "level_change_distinction",
                        "boundary_mapping",
                    ],
                },
                "graph_summary": {"type": "string"},
            },
            "required": [
                "baseline",
                "drivers",
                "countervailing_factors",
                "synthesis",
                "target_bridge",
                "graph_summary",
            ],
        },
    }


def _normalize_single_dag_plan(
    plan: dict[str, Any],
    memory: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    """Normalize checkpoint spelling and synthesis provenance only."""
    corrections: list[str] = []
    checkpoint_ids = {
        str(item.get("id")) for item in memory.get("checkpoints", [])
    }
    suffix_lookup = {
        checkpoint_id.split("_", 1)[-1]: checkpoint_id
        for checkpoint_id in checkpoint_ids
    }
    factors = [
        plan.get("baseline", {}),
        *plan.get("drivers", []),
        *plan.get("countervailing_factors", []),
        plan.get("synthesis", {}),
    ]
    for index, factor in enumerate(factors):
        checkpoint_id = str(factor.get("checkpoint_id"))
        if checkpoint_id not in checkpoint_ids:
            matched = suffix_lookup.get(checkpoint_id)
            if matched:
                factor["checkpoint_id"] = matched
                corrections.append(
                    f"factor[{index}] normalized checkpoint ID"
                )
            else:
                factor["checkpoint_id"] = "NONE"
    synthesis = plan.get("synthesis", {})
    if not synthesis.get("evidence_article_ids"):
        inherited: list[str] = []
        for factor in factors[:-1]:
            for article_id in factor.get("evidence_article_ids", []):
                if article_id not in inherited:
                    inherited.append(article_id)
        if inherited:
            synthesis["evidence_article_ids"] = inherited
            corrections.append("synthesis inherited upstream provenance")
    return plan, corrections


def _validate_single_dag_plan(
    plan: dict[str, Any],
    *,
    evidence_ids: set[str],
    checkpoint_ids: set[str],
) -> list[str]:
    errors: list[str] = []
    drivers = plan.get("drivers", [])
    counters = plan.get("countervailing_factors", [])
    if len(drivers) != 2:
        errors.append(f"drivers must contain 2 factors, got {len(drivers)}")
    if len(counters) != 1:
        errors.append(
            "countervailing_factors must contain 1 factor, "
            f"got {len(counters)}"
        )
    baseline = plan.get("baseline", {})
    if baseline.get("temporal_relation") != "historical_baseline":
        errors.append("baseline must be historical_baseline")
    factors = [
        baseline,
        *drivers,
        *counters,
        plan.get("synthesis", {}),
    ]
    used_memory = False
    for index, factor in enumerate(factors):
        citations = set(factor.get("evidence_article_ids", []))
        if not citations:
            errors.append(f"factor[{index}] has no evidence")
        unknown = citations - evidence_ids
        if unknown:
            errors.append(
                f"factor[{index}] cites unknown evidence {sorted(unknown)}"
            )
        checkpoint_id = str(factor.get("checkpoint_id"))
        if checkpoint_id in checkpoint_ids:
            used_memory = True
        elif checkpoint_id != "NONE":
            errors.append(f"factor[{index}] has invalid checkpoint ID")
    if not used_memory:
        errors.append("no retrieved checkpoint instantiated")
    return errors


def _compile_single_dag_plan(
    plan: dict[str, Any],
    *,
    question_text: str,
    checkpoint_ids: list[str],
) -> dict[str, Any]:
    factors = [
        ("BASE", "driver", plan["baseline"]),
        ("D1", "driver", plan["drivers"][0]),
        ("D2", "driver", plan["drivers"][1]),
        ("C1", "countervailing", plan["countervailing_factors"][0]),
        ("M", "mediator", plan["synthesis"]),
    ]
    nodes: list[dict[str, Any]] = []
    assessments: dict[str, dict[str, Any]] = {}
    for node_id, role, factor in factors:
        checkpoint_id = str(factor.get("checkpoint_id"))
        nodes.append(
            {
                "id": node_id,
                "label": factor.get("label"),
                "role": role,
                "observation": factor.get("observation"),
                "effect_on_target": factor.get("directional_effect"),
                "origin": (
                    "memory_instantiated"
                    if checkpoint_id in checkpoint_ids
                    else "current_case_discovered"
                ),
                "temporal_relation": factor.get("temporal_relation"),
                "checkpoint_id": checkpoint_id,
                "evidence_article_ids": factor.get(
                    "evidence_article_ids", []
                ),
            }
        )
        if checkpoint_id in checkpoint_ids:
            assessments[checkpoint_id] = {
                "checkpoint_id": checkpoint_id,
                "status": "SUPPORTED",
                "assessment": factor.get("observation"),
                "evidence_article_ids": factor.get(
                    "evidence_article_ids", []
                ),
            }
    bridge = plan["target_bridge"]
    nodes.extend(
        [
            {
                "id": "B",
                "label": "Exact target metric and boundary bridge",
                "role": "target_bridge",
                "observation": (
                    f"{bridge.get('target_metric')} "
                    f"{bridge.get('level_change_distinction')}"
                ),
                "effect_on_target": bridge.get("boundary_mapping"),
                "origin": "target_definition",
                "temporal_relation": "target_definition",
                "checkpoint_id": "NONE",
                "evidence_article_ids": [],
            },
            {
                "id": "T",
                "label": question_text,
                "role": "target",
                "observation": question_text,
                "effect_on_target": "target_sink",
                "origin": "target_definition",
                "temporal_relation": "target_definition",
                "checkpoint_id": "NONE",
                "evidence_article_ids": [],
            },
        ]
    )
    edges: list[dict[str, Any]] = []
    for node_id, _, factor in factors[:4]:
        edges.append(
            {
                "source": node_id,
                "target": "M",
                "relation": (
                    "anchors_baseline"
                    if node_id == "BASE"
                    else ("counteracts" if node_id == "C1" else "drives")
                ),
                "mechanism": factor.get("mechanism_to_synthesis"),
                "evidence_article_ids": factor.get(
                    "evidence_article_ids", []
                ),
            }
        )
    edges.extend(
        [
            {
                "source": "M",
                "target": "B",
                "relation": "maps_to",
                "mechanism": plan["synthesis"].get(
                    "mechanism_to_synthesis"
                ),
                "evidence_article_ids": plan["synthesis"].get(
                    "evidence_article_ids", []
                ),
            },
            {
                "source": "B",
                "target": "T",
                "relation": "maps_to",
                "mechanism": bridge.get("boundary_mapping"),
                "evidence_article_ids": [],
            },
        ]
    )
    return {
        "checkpoint_assessments": [
            assessments.get(
                checkpoint_id,
                {
                    "checkpoint_id": checkpoint_id,
                    "status": "UNKNOWN",
                    "assessment": "Not instantiated in the current case.",
                    "evidence_article_ids": [],
                },
            )
            for checkpoint_id in checkpoint_ids
        ],
        "nodes": nodes,
        "edges": edges,
        "graph_summary": plan.get("graph_summary"),
    }


def _call_json(
    client: OpenAI,
    *,
    model: str,
    system: str,
    prompt: str,
    schema: dict[str, Any],
    seed: int,
    max_tokens: int,
) -> tuple[dict[str, Any], dict[str, int], float]:
    started = time.monotonic()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_schema", "json_schema": schema},
        seed=seed,
        **completion_parameters(
            model=model,
            stage_max_tokens=max_tokens,
        ),
    )
    raw_content = response.choices[0].message.content or "{}"
    try:
        payload = json.loads(raw_content)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "model returned truncated or invalid JSON "
            f"(characters={len(raw_content)}, line={exc.lineno}, "
            f"column={exc.colno}, tail={raw_content[-160:]!r})"
        ) from exc
    raw_usage = response.usage.model_dump() if response.usage else {}
    usage = {
        "prompt_tokens": int(raw_usage.get("prompt_tokens") or 0),
        "completion_tokens": int(raw_usage.get("completion_tokens") or 0),
        "total_tokens": int(raw_usage.get("total_tokens") or 0),
        "call_count": 1,
    }
    return payload, usage, time.monotonic() - started


def _validate_graph(
    graph: dict[str, Any],
    *,
    evidence_ids: set[str],
    checkpoint_ids: set[str],
) -> list[str]:
    errors: list[str] = []
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    if not 6 <= len(nodes) <= 10:
        errors.append(f"node count must be 6-10, got {len(nodes)}")
    if not 5 <= len(edges) <= 12:
        errors.append(f"edge count must be 5-12, got {len(edges)}")
    node_ids = [str(item.get("id")) for item in nodes]
    node_id_set = set(node_ids)
    if len(node_ids) != len(node_id_set):
        errors.append("duplicate node IDs")
    target_nodes = [item for item in nodes if item.get("role") == "target"]
    if len(target_nodes) != 1:
        errors.append("graph must contain exactly one target")
        return errors
    target_id = str(target_nodes[0].get("id"))
    if any(str(edge.get("source")) == target_id for edge in edges):
        errors.append("target must be a sink")
    if not any(item.get("role") == "target_bridge" for item in nodes):
        errors.append("missing target bridge")
    if not any(item.get("role") == "countervailing" for item in nodes):
        errors.append("missing countervailing factor")
    if not any(
        item.get("origin") == "memory_instantiated"
        for item in nodes
        if item.get("role") != "target"
    ):
        errors.append("no memory checkpoint instantiated")
    for node in target_nodes:
        if node.get("temporal_relation") != "target_definition":
            errors.append("target must have target_definition temporal role")

    adjacency: dict[str, list[str]] = {node_id: [] for node_id in node_ids}
    indegree: dict[str, int] = {node_id: 0 for node_id in node_ids}
    for edge in edges:
        source = str(edge.get("source"))
        target = str(edge.get("target"))
        if source not in node_id_set or target not in node_id_set:
            errors.append(f"edge has unknown endpoint: {source}->{target}")
            continue
        adjacency[source].append(target)
        indegree[target] += 1
        unknown = set(edge.get("evidence_article_ids", [])) - evidence_ids
        if unknown:
            errors.append(f"edge cites unknown evidence: {sorted(unknown)}")

    queue = [node_id for node_id, degree in indegree.items() if degree == 0]
    visited: list[str] = []
    while queue:
        node_id = queue.pop()
        visited.append(node_id)
        for child in adjacency[node_id]:
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(child)
    if len(visited) != len(node_ids):
        errors.append("graph contains a cycle")

    def reaches_target(start: str) -> bool:
        pending = [start]
        seen: set[str] = set()
        while pending:
            current = pending.pop()
            if current == target_id:
                return True
            if current in seen:
                continue
            seen.add(current)
            pending.extend(adjacency.get(current, []))
        return False

    for node in nodes:
        node_id = str(node.get("id"))
        citations = set(node.get("evidence_article_ids", []))
        unknown = citations - evidence_ids
        if unknown:
            errors.append(f"{node_id} cites unknown evidence: {sorted(unknown)}")
        if node.get("role") not in {"target", "target_bridge"} and not citations:
            errors.append(f"{node_id} has no current evidence")
        if node.get("role") != "target" and not reaches_target(node_id):
            errors.append(f"{node_id} does not reach target")
        checkpoint_id = str(node.get("checkpoint_id"))
        if (
            node.get("role") == "target_bridge"
            and node.get("temporal_relation") != "target_definition"
        ):
            errors.append(f"{node_id} target bridge has wrong temporal role")
        if (
            node.get("origin") == "memory_instantiated"
            and checkpoint_id not in checkpoint_ids
        ):
            errors.append(f"{node_id} has invalid checkpoint lineage")

    assessments = graph.get("checkpoint_assessments", [])
    assessed_ids = [str(item.get("checkpoint_id")) for item in assessments]
    if set(assessed_ids) != checkpoint_ids or len(assessed_ids) != len(
        checkpoint_ids
    ):
        errors.append("checkpoint assessment coverage mismatch")
    for item in assessments:
        citations = set(item.get("evidence_article_ids", []))
        if citations - evidence_ids:
            errors.append("checkpoint assessment cites unknown evidence")
        if item.get("status") in {"SUPPORTED", "CONTRADICTED"} and not citations:
            errors.append("decisive checkpoint assessment lacks evidence")
    assessment_by_id = {
        str(item.get("checkpoint_id")): item for item in assessments
    }
    for node in nodes:
        if node.get("origin") != "memory_instantiated":
            continue
        checkpoint_id = str(node.get("checkpoint_id"))
        assessment = assessment_by_id.get(checkpoint_id, {})
        if assessment.get("status") not in {"SUPPORTED", "CONTRADICTED"}:
            errors.append(
                f"{node.get('id')} instantiates an unverified checkpoint"
            )
    return errors


def _probabilities(
    payload: dict[str, Any],
    options: list[str],
) -> tuple[dict[str, float], list[str]]:
    errors: list[str] = []
    rows = payload.get("option_probabilities", [])
    probabilities: dict[str, float] = {}
    for row in rows:
        option = str(row.get("option"))
        if option in probabilities:
            errors.append(f"duplicate probability for {option}")
            continue
        probabilities[option] = float(row.get("probability", -1))
    if set(probabilities) != set(options):
        errors.append("probability options do not match target options")
        return probabilities, errors
    total = sum(probabilities.values())
    if any(value < 0.01 or value > 0.99 for value in probabilities.values()):
        errors.append("probabilities must be between 0.01 and 0.99")
    if abs(total - 1.0) > 0.011:
        errors.append(f"probabilities sum to {total:.6f}")
    prediction = str(payload.get("prediction"))
    max_probability = max(probabilities.values())
    if (
        prediction not in probabilities
        or probabilities[prediction] < max_probability - 1e-9
    ):
        argmax = max(options, key=lambda option: probabilities[option])
        errors.append(f"prediction {prediction} is not argmax {argmax}")
    return probabilities, errors


def _validate_forecast(
    payload: dict[str, Any],
    *,
    options: list[str],
    allowed_ids: set[str],
    graph_arm: bool,
    evidence_ids: set[str] | None = None,
) -> tuple[dict[str, float], list[str]]:
    probabilities, errors = _probabilities(payload, options)
    key = "evidence_article_ids"
    if not graph_arm:
        used_ids = set(payload.get(key, []))
        if not used_ids:
            errors.append(f"{key} is empty")
        unknown = used_ids - allowed_ids
        if unknown:
            errors.append(f"{key} contains unknown IDs: {sorted(unknown)}")
    if graph_arm:
        dominant_ids = set(payload.get("dominant_factor_ids", []))
        counter_ids = set(payload.get("counter_factor_ids", []))
        if not dominant_ids:
            errors.append("dominant_factor_ids is empty")
        if not counter_ids:
            errors.append("counter_factor_ids is empty")
        unknown_factor_ids = (dominant_ids | counter_ids) - allowed_ids
        if unknown_factor_ids:
            errors.append(
                "forecast contains unknown graph factor IDs: "
                f"{sorted(unknown_factor_ids)}"
            )
        cited_evidence = set(payload.get("cited_evidence_article_ids", []))
        if not cited_evidence:
            errors.append("cited_evidence_article_ids is empty")
        unknown_evidence = cited_evidence - (evidence_ids or set())
        if unknown_evidence:
            errors.append(
                "cited_evidence_article_ids contains unknown IDs: "
                f"{sorted(unknown_evidence)}"
            )
    return probabilities, errors


def _ground_truth_option(question: Any) -> str:
    options = [str(option) for option in question.options or []]
    ground_truth = question.ground_truth
    if isinstance(ground_truth, bool):
        value = "yes" if ground_truth else "no"
    else:
        value = str(ground_truth)
    for option in options:
        if option.casefold() == value.casefold():
            return option
    raise ValueError(f"Ground truth {ground_truth!r} not in {options}")
