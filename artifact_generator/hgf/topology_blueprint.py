"""Compile refined DAGs into outcome-redacted, topology-preserving blueprints."""

from __future__ import annotations

from collections import defaultdict, deque
import copy
import json
from pathlib import Path
import re
from typing import Any

from hgf.dag import _redact_answer_labels
from hgf.question_io import family_metadata


_TARGET_MAPPING_RELATIONS = {"classifies_as", "maps_to"}
_COUNTER_RELATIONS = {"counteracts", "inhibits"}
_BASELINE_RELATIONS = {"provides_baseline_for"}
_SUPPORT_RANK = {
    "background_hypothesis": 1,
    "evidence_synthesized": 2,
    "observed": 3,
}
_SUPPORT_LABEL = {
    "background_hypothesis": "weak",
    "evidence_synthesized": "medium",
    "observed": "strong",
}
_CURRENT_VALUE_PLACEHOLDER = "[CURRENT_VALUE_REQUIRED]"
_CURRENT_PERIOD_PLACEHOLDER = "[CURRENT_PERIOD_REQUIRED]"
_MONTH_NAME = (
    r"January|February|March|April|May|June|July|August|September|"
    r"October|November|December|Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|"
    r"Apr(?:il)?|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|"
    r"Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?"
)
_VALUE_PATTERNS = (
    re.compile(
        r"(?<!\w)(?:US\$|\$|USD\s*)[+-]?\d[\d,]*(?:\.\d+)?"
        r"(?:\s*(?:million|billion|trillion))?",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?<![\w-])[+-]?\d[\d,]*(?:\.\d+)?\s*"
        r"(?:%(?!\w)|percent(?:age)?(?:\s+points?)?\b|"
        r"basis\s+points?\b|bps\b|million\b|billion\b|trillion\b|"
        r"index\s+points?\b)",
        re.IGNORECASE,
    ),
    re.compile(r"(?<![\w-])[+-]?\d+\.\d+\b"),
    re.compile(r"(?<![\w-])[+-]?\d{1,3}(?:,\d{3})+(?![\w-])"),
)
_PERIOD_PATTERNS = (
    re.compile(r"\b20\d{2}-\d{2}(?:-\d{2})?\b"),
    re.compile(
        rf"\b(?:{_MONTH_NAME})"
        r"(?:\s+\d{1,2}(?:st|nd|rd|th)?)?"
        r"(?:,?\s+20\d{2})?\b",
    ),
    re.compile(
        r"\b(?:Q[1-4]|first\s+quarter|second\s+quarter|third\s+quarter|"
        r"fourth\s+quarter)(?:\s+(?:of\s+)?20\d{2})?\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b20\d{2}\s+Q[1-4]\b", re.IGNORECASE),
    re.compile(r"\b20\d{2}\b"),
)
_SANITIZED_TEXT_FIELDS = {
    "factor",
    "mechanism",
    "rationale",
    "generalized_mechanism",
    "hypothesis",
    "discriminating_evidence",
}


def _question_value(question: Any, field: str, default: Any = None) -> Any:
    if isinstance(question, dict):
        return question.get(field, default)
    return getattr(question, field, default)


def _node_id(node: dict[str, Any]) -> str:
    return str(node.get("id") or "")


def _node_label(node: dict[str, Any]) -> str:
    return str(node.get("label") or node.get("title") or "")


def _edge_source(edge: dict[str, Any]) -> str:
    return str(edge.get("source") or edge.get("source_event_id") or "")


def _edge_target(edge: dict[str, Any]) -> str:
    return str(edge.get("target") or edge.get("target_event_id") or "")


def _edge_relation(edge: dict[str, Any]) -> str:
    return str(edge.get("relationship") or edge.get("relation_type") or "")


def _edge_rationale(edge: dict[str, Any]) -> str:
    return str(edge.get("rationale") or edge.get("reasoning") or "")


def _as_strings(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [item for item in value.split() if item]
    return [str(item) for item in value if item]


def _sanitize_historical_specific_text(value: str) -> tuple[str, int, int]:
    sanitized = value
    value_count = 0
    period_count = 0
    for pattern in _VALUE_PATTERNS:
        sanitized, count = pattern.subn(
            _CURRENT_VALUE_PLACEHOLDER,
            sanitized,
        )
        value_count += count
    for pattern in _PERIOD_PATTERNS:
        sanitized, count = pattern.subn(
            _CURRENT_PERIOD_PLACEHOLDER,
            sanitized,
        )
        period_count += count
    sanitized = re.sub(
        r"\btarget[- ]quarter\b",
        "target period",
        sanitized,
        flags=re.IGNORECASE,
    )
    return sanitized, value_count, period_count


def _specific_text_fields(payload: Any) -> list[str]:
    values: list[str] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key in _SANITIZED_TEXT_FIELDS and isinstance(value, str):
                values.append(value)
            else:
                values.extend(_specific_text_fields(value))
    elif isinstance(payload, list):
        for value in payload:
            values.extend(_specific_text_fields(value))
    return values


def historical_specific_counts(
    blueprint: dict[str, Any],
) -> dict[str, int]:
    """Count realized values and absolute periods in reusable text fields."""
    value_count = 0
    period_count = 0
    for value in _specific_text_fields(blueprint):
        for pattern in _VALUE_PATTERNS:
            value_count += len(pattern.findall(value))
        for pattern in _PERIOD_PATTERNS:
            period_count += len(pattern.findall(value))
    return {
        "realized_value_count": value_count,
        "absolute_period_count": period_count,
    }


def sanitize_topology_blueprint(
    blueprint: dict[str, Any],
) -> dict[str, Any]:
    """Replace historical values and periods without changing DAG structure."""
    sanitized = copy.deepcopy(blueprint)
    value_replacements = 0
    period_replacements = 0

    def visit(payload: Any) -> None:
        nonlocal value_replacements, period_replacements
        if isinstance(payload, dict):
            for key, value in payload.items():
                if key in _SANITIZED_TEXT_FIELDS and isinstance(value, str):
                    (
                        payload[key],
                        value_count,
                        period_count,
                    ) = _sanitize_historical_specific_text(value)
                    value_replacements += value_count
                    period_replacements += period_count
                else:
                    visit(value)
        elif isinstance(payload, list):
            for value in payload:
                visit(value)

    visit(sanitized)
    refinement_metadata = sanitized.setdefault("refinement_metadata", {})
    refinement_metadata["historical_specifics_sanitized"] = True
    refinement_metadata["sanitizer"] = "minimal_value_period_placeholders_v1"
    refinement_metadata["value_replacement_count"] = value_replacements
    refinement_metadata["period_replacement_count"] = period_replacements
    return sanitized


def _project_reusable_topology(
    graph_payload: dict[str, Any],
) -> dict[str, Any]:
    """Return the reusable ancestors and source edges behind the target value."""
    graph = graph_payload.get("graph", {})
    nodes = list(graph.get("nodes", []))
    edges = list(graph.get("edges", []))
    node_by_id = {_node_id(node): node for node in nodes}
    actual_outcome_id = str(
        graph_payload.get("actual_outcome_event_id") or ""
    )
    if not actual_outcome_id:
        raise ValueError("refined DAG has no actual_outcome_event_id")

    outcome_ids = {
        node_id
        for node_id, node in node_by_id.items()
        if node.get("is_outcome")
        or node.get("is_actual_outcome")
        or node_id == actual_outcome_id
    }
    outcome_incoming_edges = [
        edge
        for edge in edges
        if _edge_target(edge) == actual_outcome_id
        and _edge_source(edge) not in outcome_ids
    ]
    target_mapping_edges = [
        edge
        for edge in outcome_incoming_edges
        if _edge_relation(edge) in _TARGET_MAPPING_RELATIONS
    ]
    direct_outcome_edges = [
        edge
        for edge in outcome_incoming_edges
        if _edge_relation(edge) not in _TARGET_MAPPING_RELATIONS
    ]
    target_anchor_ids = {
        _edge_source(edge) for edge in target_mapping_edges
    }
    direct_target_ids = {
        _edge_source(edge) for edge in direct_outcome_edges
    }
    if not target_anchor_ids and not direct_target_ids:
        raise ValueError("actual outcome has no reusable incoming edge")

    incoming: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for edge in edges:
        incoming[_edge_target(edge)].append(edge)

    ancestor_ids: set[str] = set()
    traversal_seeds = target_anchor_ids | direct_target_ids
    queue = deque(sorted(traversal_seeds))
    visited = set(traversal_seeds)
    while queue:
        target_id = queue.popleft()
        for edge in incoming.get(target_id, []):
            source_id = _edge_source(edge)
            if not source_id or source_id in outcome_ids:
                continue
            ancestor_ids.add(source_id)
            if source_id not in visited:
                visited.add(source_id)
                queue.append(source_id)

    retained_ids = (
        ancestor_ids | direct_target_ids
    ) - target_anchor_ids - outcome_ids
    internal_edges = [
        edge
        for edge in edges
        if _edge_source(edge) in retained_ids
        and _edge_target(edge) in retained_ids
    ]
    terminal_edges = [
        *[
            edge
            for edge in edges
            if _edge_source(edge) in retained_ids
            and _edge_target(edge) in target_anchor_ids
        ],
        *[
            edge
            for edge in direct_outcome_edges
            if _edge_source(edge) in retained_ids
        ],
    ]
    if not retained_ids or not terminal_edges:
        raise ValueError("DAG has no reusable path into the redacted target")

    return {
        "nodes": nodes,
        "edges": edges,
        "node_by_id": node_by_id,
        "actual_outcome_id": actual_outcome_id,
        "outcome_ids": outcome_ids,
        "target_anchor_ids": target_anchor_ids,
        "retained_ids": retained_ids,
        "internal_edges": internal_edges,
        "terminal_edges": terminal_edges,
    }


def _topological_node_order(
    retained_ids: set[str],
    internal_edges: list[dict[str, Any]],
    nodes: list[dict[str, Any]],
) -> list[str]:
    source_order = {
        _node_id(node): index for index, node in enumerate(nodes)
    }
    indegree = {node_id: 0 for node_id in retained_ids}
    outgoing: dict[str, list[str]] = defaultdict(list)
    for edge in internal_edges:
        source_id = _edge_source(edge)
        target_id = _edge_target(edge)
        indegree[target_id] += 1
        outgoing[source_id].append(target_id)

    ready = sorted(
        (node_id for node_id, degree in indegree.items() if degree == 0),
        key=lambda node_id: (source_order.get(node_id, 10**9), node_id),
    )
    ordered: list[str] = []
    while ready:
        node_id = ready.pop(0)
        ordered.append(node_id)
        for target_id in sorted(
            outgoing.get(node_id, []),
            key=lambda value: (source_order.get(value, 10**9), value),
        ):
            indegree[target_id] -= 1
            if indegree[target_id] == 0:
                ready.append(target_id)
                ready.sort(
                    key=lambda value: (
                        source_order.get(value, 10**9),
                        value,
                    )
                )
    if len(ordered) != len(retained_ids):
        raise ValueError("reusable DAG projection is cyclic")
    return ordered


def _historical_support(
    node: dict[str, Any],
    outgoing_edges: list[dict[str, Any]],
) -> str:
    levels = [
        str(node.get("support_level") or ""),
        *[
            str(edge.get("support_level") or "")
            for edge in outgoing_edges
        ],
    ]
    strongest = max(
        levels,
        key=lambda value: _SUPPORT_RANK.get(value, 0),
        default="",
    )
    return _SUPPORT_LABEL.get(strongest, "weak")


def _checkpoint_role(
    *,
    node_id: str,
    incoming_edges: list[dict[str, Any]],
    outgoing_edges: list[dict[str, Any]],
) -> str:
    relations = {_edge_relation(edge) for edge in outgoing_edges}
    if relations & _COUNTER_RELATIONS:
        return "counterevidence"
    if relations & _BASELINE_RELATIONS:
        return "baseline"
    if incoming_edges and outgoing_edges:
        return "mediator"
    return "driver"


def _safe_rationale(
    edge: dict[str, Any],
    resolved_outcome: Any,
    *,
    terminal: bool,
) -> str:
    if terminal:
        return (
            "Historical DAG connected this factor to the redacted target "
            "measurement; the realized value and outcome mapping are omitted."
        )
    return str(
        _redact_answer_labels(
            _edge_rationale(edge),
            resolved_outcome,
        )
    )


def _enumerate_paths(
    *,
    root_checkpoint_ids: list[str],
    topology_edges: list[dict[str, Any]],
    target_bridge_id: str,
    max_paths: int,
) -> tuple[list[dict[str, Any]], bool]:
    outgoing: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for edge in topology_edges:
        outgoing[str(edge["source_checkpoint_id"])].append(edge)

    paths: list[dict[str, Any]] = []
    truncated = False

    def visit(
        checkpoint_id: str,
        checkpoint_path: list[str],
        edge_path: list[dict[str, Any]],
    ) -> None:
        nonlocal truncated
        if len(paths) >= max_paths:
            truncated = True
            return
        if checkpoint_id == target_bridge_id:
            paths.append(
                {
                    "checkpoint_ids": checkpoint_path,
                    "source_edge_ids": [
                        str(edge["source_edge_ids"][0])
                        for edge in edge_path
                    ],
                    "relationships": [
                        str(edge["relationship"]) for edge in edge_path
                    ],
                    "support_score": sum(
                        _SUPPORT_RANK.get(
                            str(edge.get("support_level") or ""),
                            0,
                        )
                        for edge in edge_path
                    )
                    / max(1, len(edge_path)),
                }
            )
            return
        for edge in outgoing.get(checkpoint_id, []):
            target_id = str(edge["target_checkpoint_id"])
            if target_id in checkpoint_path:
                raise ValueError("cycle encountered while enumerating paths")
            visit(
                target_id,
                checkpoint_path + [target_id],
                edge_path + [edge],
            )

    for root_id in root_checkpoint_ids:
        visit(root_id, [root_id], [])

    return paths, truncated


def compile_topology_blueprint(
    *,
    graph_payload: dict[str, Any],
    question: Any,
    audit: dict[str, Any],
    source_graph: Path | str,
    max_paths: int = 32,
) -> dict[str, Any]:
    """Compile one leakage-safe blueprint without inventing DAG adjacency."""
    projection = _project_reusable_topology(graph_payload)
    node_by_id = projection["node_by_id"]
    internal_edges = projection["internal_edges"]
    terminal_edges = projection["terminal_edges"]
    retained_ids = projection["retained_ids"]
    resolved_outcome = _question_value(question, "ground_truth")

    ordered_ids = _topological_node_order(
        retained_ids,
        internal_edges,
        projection["nodes"],
    )
    checkpoint_by_node = {
        node_id: f"checkpoint_{index}"
        for index, node_id in enumerate(ordered_ids, start=1)
    }
    target_bridge_id = "target_bridge"

    incoming_by_node: dict[str, list[dict[str, Any]]] = defaultdict(list)
    outgoing_by_node: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for edge in [*internal_edges, *terminal_edges]:
        source_id = _edge_source(edge)
        outgoing_by_node[source_id].append(edge)
        if _edge_target(edge) in retained_ids:
            incoming_by_node[_edge_target(edge)].append(edge)

    checkpoints = []
    search_factors = []
    for node_id in ordered_ids:
        node = node_by_id[node_id]
        label = str(
            _redact_answer_labels(
                _node_label(node),
                resolved_outcome,
            )
        )
        outgoing_edges = outgoing_by_node[node_id]
        role = _checkpoint_role(
            node_id=node_id,
            incoming_edges=incoming_by_node[node_id],
            outgoing_edges=outgoing_edges,
        )
        internal_rationales = [
            _safe_rationale(
                edge,
                resolved_outcome,
                terminal=False,
            )
            for edge in outgoing_edges
            if _edge_target(edge) in retained_ids
            and _edge_rationale(edge).strip()
        ]
        mechanism = (
            internal_rationales[0]
            if internal_rationales
            else (
                "Re-test whether this factor has a currently supported link "
                "to the exact target metric; do not transfer the historical "
                "outcome or magnitude."
            )
        )
        source_edge_ids = [
            str(edge.get("id"))
            for edge in outgoing_edges
            if edge.get("id")
        ]
        checkpoints.append(
            {
                "id": checkpoint_by_node[node_id],
                "role": role,
                "factor": label,
                "mechanism": mechanism,
                "expected_direction": (
                    "mixed" if role == "counterevidence" else "unknown"
                ),
                "evidence_requirement": (
                    "Current cutoff-safe evidence must directly measure this "
                    "factor and support every retained outgoing relationship."
                ),
                "contradiction_signal": (
                    "Current evidence contradicts the factor, reverses an "
                    "outgoing relationship, or leaves a required link unknown."
                ),
                "historical_support": _historical_support(
                    node,
                    outgoing_edges,
                ),
                "source_event_ids": [node_id],
                "source_edge_ids": source_edge_ids,
                "source_event_type": str(
                    node.get("event_type") or ""
                ),
            }
        )
        search_factors.append(
            {
                "factor": label,
                "why_search": (
                    "This factor occupies a preserved position in a validated "
                    "historical path; verify both the factor and its next edge."
                ),
                "preferred_source_types": [
                    "official release",
                    "dated independent reporting",
                ],
                "source_event_ids": [node_id],
            }
        )

    metadata = family_metadata(question)
    target_metric = str(
        metadata.get("target_metric")
        or "Net transmission into the forecast target"
    )
    terminal_edge_ids = [
        str(edge.get("id"))
        for edge in terminal_edges
        if edge.get("id")
    ]
    checkpoints.append(
        {
            "id": target_bridge_id,
            "role": "target_bridge",
            "factor": target_metric,
            "mechanism": (
                "Map only currently supported paths into the exact target "
                "operator. Directional support alone is not magnitude or "
                "threshold-crossing support."
            ),
            "expected_direction": "mixed",
            "evidence_requirement": (
                "A current target baseline and evidence supporting every edge "
                "used in the bridge."
            ),
            "contradiction_signal": (
                "The current target baseline, magnitude evidence, or option "
                "boundary contradicts the proposed path."
            ),
            "historical_support": "medium",
            "source_event_ids": [],
            "source_edge_ids": terminal_edge_ids,
            "source_event_type": "target_bridge",
        }
    )

    topology_edges = []
    for edge in internal_edges:
        topology_edges.append(
            {
                "source_checkpoint_id": checkpoint_by_node[
                    _edge_source(edge)
                ],
                "target_checkpoint_id": checkpoint_by_node[
                    _edge_target(edge)
                ],
                "relationship": _edge_relation(edge),
                "directionality": (
                    "counteracting"
                    if _edge_relation(edge) in _COUNTER_RELATIONS
                    else "forward"
                ),
                "support_level": str(
                    edge.get("support_level") or ""
                ),
                "rationale": _safe_rationale(
                    edge,
                    resolved_outcome,
                    terminal=False,
                ),
                "source_edge_ids": [str(edge.get("id"))],
                "source_article_ids": _as_strings(
                    edge.get("article_ids")
                    or edge.get("evidence_article_ids")
                ),
                "terminal_to_target_bridge": False,
            }
        )
    for edge in terminal_edges:
        topology_edges.append(
            {
                "source_checkpoint_id": checkpoint_by_node[
                    _edge_source(edge)
                ],
                "target_checkpoint_id": target_bridge_id,
                "relationship": _edge_relation(edge),
                "directionality": (
                    "counteracting"
                    if _edge_relation(edge) in _COUNTER_RELATIONS
                    else "forward"
                ),
                "support_level": str(
                    edge.get("support_level") or ""
                ),
                "rationale": _safe_rationale(
                    edge,
                    resolved_outcome,
                    terminal=True,
                ),
                "source_edge_ids": [str(edge.get("id"))],
                "source_article_ids": _as_strings(
                    edge.get("article_ids")
                    or edge.get("evidence_article_ids")
                ),
                "terminal_to_target_bridge": True,
            }
        )

    incoming_checkpoint_ids = {
        str(edge["target_checkpoint_id"])
        for edge in topology_edges
        if edge["target_checkpoint_id"] != target_bridge_id
    }
    root_checkpoint_ids = [
        checkpoint_by_node[node_id]
        for node_id in ordered_ids
        if checkpoint_by_node[node_id] not in incoming_checkpoint_ids
    ]
    raw_paths, paths_truncated = _enumerate_paths(
        root_checkpoint_ids=root_checkpoint_ids,
        topology_edges=topology_edges,
        target_bridge_id=target_bridge_id,
        max_paths=max_paths,
    )
    if not raw_paths:
        raise ValueError("reusable DAG projection has no root-to-target path")

    non_counter_indices = [
        index
        for index, path in enumerate(raw_paths)
        if not set(path["relationships"]) & _COUNTER_RELATIONS
    ]
    main_index = max(
        non_counter_indices or range(len(raw_paths)),
        key=lambda index: raw_paths[index]["support_score"],
    )
    checkpoint_factor = {
        str(item["id"]): str(item["factor"]) for item in checkpoints
    }
    causal_paths = []
    for index, path in enumerate(raw_paths):
        relationships = list(path["relationships"])
        is_counter = bool(set(relationships) & _COUNTER_RELATIONS)
        first_factor = checkpoint_factor[path["checkpoint_ids"][0]]
        causal_paths.append(
            {
                "checkpoint_ids": path["checkpoint_ids"],
                "source_edge_ids": path["source_edge_ids"],
                "relationships": relationships,
                "path_role": (
                    "counter_path"
                    if is_counter
                    else "main_path"
                    if index == main_index
                    else "supporting_path"
                ),
                "generalized_mechanism": (
                    f"Re-test the preserved relationship chain from "
                    f"{first_factor} to the current target; every edge requires "
                    "current support."
                ),
                "expected_direction": "mixed" if is_counter else "unknown",
                "applicability_conditions": [
                    "Every retained relationship is active in the current case.",
                    "Current cutoff-safe evidence supports each intermediate link.",
                ],
                "failure_conditions": [
                    "Any retained relationship is contradicted or unknown.",
                    "A stronger current mechanism bypasses or dominates this path.",
                ],
            }
        )

    counter_checkpoints = [
        item
        for item in checkpoints
        if item.get("role") == "counterevidence"
    ]
    alternatives = [
        {
            "hypothesis": (
                f"{item['factor']} dominates or reverses the leading current "
                "path."
            ),
            "discriminating_evidence": (
                "Compare current evidence for this counter-path with every "
                "edge on the leading supported path."
            ),
            "source_event_ids": list(item["source_event_ids"]),
        }
        for item in counter_checkpoints[:2]
    ]
    if not alternatives:
        alternatives = [
            {
                "hypothesis": (
                    "A current factor outside the preserved historical topology "
                    "dominates the target."
                ),
                "discriminating_evidence": (
                    "Search for current shocks and target observations not "
                    "explained by any retained path."
                ),
                "source_event_ids": [],
            }
        ]

    question_id = str(_question_value(question, "id") or "")
    options = list(_question_value(question, "options", []) or [])
    blueprint = {
        "schema_version": "hgf_blueprint_topology_v2",
        "question_id": question_id,
        "target_definition": {
            "metric": metadata.get("target_metric"),
            "unit_or_option_space": [str(option) for option in options],
            "forecast_horizon": (
                "Remap the historical target family to the current question's "
                "exact target period."
            ),
        },
        "graph_diagnosis": {
            "usable": len(ordered_ids) >= 2 and bool(causal_paths),
            "summary": (
                f"Outcome-redacted topology preserves {len(ordered_ids)} "
                f"checkpoints, {len(topology_edges)} source edges, and "
                f"{len(causal_paths)} root-to-target paths."
            ),
            "weaknesses": list(audit.get("caveats", []))[:3],
        },
        "search_factors": search_factors,
        "checkpoints": checkpoints,
        "topology": {
            "checkpoint_ids": [
                *[checkpoint_by_node[node_id] for node_id in ordered_ids],
                target_bridge_id,
            ],
            "root_checkpoint_ids": root_checkpoint_ids,
            "target_bridge_id": target_bridge_id,
            "edges": topology_edges,
            "paths_truncated": paths_truncated,
        },
        "causal_paths": causal_paths,
        "alternative_hypotheses": alternatives,
        "forecast_audit_questions": [
            "Does every used checkpoint transition correspond to a preserved edge?",
            "Which preserved edges are supported, contradicted, or unknown now?",
            "Does a counter-path dominate the leading supported path?",
            "Is there evidence for magnitude and boundary crossing, not direction alone?",
        ],
        "refinement_metadata": {
            "model": "deterministic_topology_compiler_v2",
            "source_graph": str(source_graph),
            "source_audit_status": audit.get("status"),
            "outcome_nodes_redacted": len(projection["outcome_ids"]),
            "target_value_anchors_redacted": len(
                projection["target_anchor_ids"]
            ),
        },
    }
    return blueprint


def validate_topology_blueprint(
    blueprint: dict[str, Any],
    graph_payload: dict[str, Any],
) -> dict[str, Any]:
    """Validate edge fidelity, path fidelity, acyclicity, and outcome redaction."""
    projection = _project_reusable_topology(graph_payload)
    errors: list[str] = []
    checkpoints = {
        str(item.get("id")): item
        for item in blueprint.get("checkpoints", [])
    }
    topology = blueprint.get("topology", {})
    topology_edges = list(topology.get("edges", []))
    target_bridge_id = str(topology.get("target_bridge_id") or "")
    raw_edge_by_id = {
        str(edge.get("id")): edge
        for edge in [
            *projection["internal_edges"],
            *projection["terminal_edges"],
        ]
    }
    expected_edge_ids = set(raw_edge_by_id)
    emitted_edge_ids: set[str] = set()

    source_event_by_checkpoint = {
        checkpoint_id: (
            str(item.get("source_event_ids", [""])[0])
            if item.get("source_event_ids")
            else ""
        )
        for checkpoint_id, item in checkpoints.items()
    }
    for edge in topology_edges:
        source_checkpoint_id = str(edge.get("source_checkpoint_id") or "")
        target_checkpoint_id = str(edge.get("target_checkpoint_id") or "")
        source_edge_ids = [
            str(value) for value in edge.get("source_edge_ids", [])
        ]
        if len(source_edge_ids) != 1:
            errors.append(
                "topology edge must cite exactly one source edge: "
                f"{source_checkpoint_id}->{target_checkpoint_id}"
            )
            continue
        source_edge_id = source_edge_ids[0]
        emitted_edge_ids.add(source_edge_id)
        raw_edge = raw_edge_by_id.get(source_edge_id)
        if raw_edge is None:
            errors.append(f"unknown source edge {source_edge_id}")
            continue
        if source_event_by_checkpoint.get(source_checkpoint_id) != _edge_source(
            raw_edge
        ):
            errors.append(f"source mismatch for {source_edge_id}")
        if target_checkpoint_id == target_bridge_id:
            if _edge_target(raw_edge) not in {
                *projection["target_anchor_ids"],
                projection["actual_outcome_id"],
            }:
                errors.append(f"invalid target bridge edge {source_edge_id}")
        elif source_event_by_checkpoint.get(
            target_checkpoint_id
        ) != _edge_target(raw_edge):
            errors.append(f"target mismatch for {source_edge_id}")
        if str(edge.get("relationship") or "") != _edge_relation(raw_edge):
            errors.append(f"relationship mismatch for {source_edge_id}")

    missing_edges = expected_edge_ids - emitted_edge_ids
    extra_edges = emitted_edge_ids - expected_edge_ids
    if missing_edges:
        errors.append(f"missing source edges {sorted(missing_edges)}")
    if extra_edges:
        errors.append(f"extra source edges {sorted(extra_edges)}")

    adjacency = {
        (
            str(edge.get("source_checkpoint_id") or ""),
            str(edge.get("target_checkpoint_id") or ""),
        )
        for edge in topology_edges
    }
    covered_pairs: set[tuple[str, str]] = set()
    for path in blueprint.get("causal_paths", []):
        checkpoint_ids = [
            str(value) for value in path.get("checkpoint_ids", [])
        ]
        for pair in zip(checkpoint_ids, checkpoint_ids[1:]):
            covered_pairs.add(pair)
            if pair not in adjacency:
                errors.append(
                    "causal path invents adjacency "
                    f"{pair[0]}->{pair[1]}"
                )
        if checkpoint_ids and checkpoint_ids[-1] != target_bridge_id:
            errors.append("causal path does not terminate at target_bridge")
    uncovered_pairs = adjacency - covered_pairs
    if uncovered_pairs:
        errors.append(
            "topology edges missing from causal paths "
            f"{sorted(uncovered_pairs)}"
        )

    indegree = {checkpoint_id: 0 for checkpoint_id in checkpoints}
    outgoing: dict[str, list[str]] = defaultdict(list)
    for source_id, target_id in adjacency:
        if source_id not in indegree or target_id not in indegree:
            errors.append(f"unknown checkpoint edge {source_id}->{target_id}")
            continue
        indegree[target_id] += 1
        outgoing[source_id].append(target_id)
    queue = deque(
        sorted(
            checkpoint_id
            for checkpoint_id, degree in indegree.items()
            if degree == 0
        )
    )
    visited = 0
    while queue:
        checkpoint_id = queue.popleft()
        visited += 1
        for target_id in outgoing.get(checkpoint_id, []):
            indegree[target_id] -= 1
            if indegree[target_id] == 0:
                queue.append(target_id)
    if visited != len(checkpoints):
        errors.append("blueprint topology is cyclic")

    leaked_event_ids = {
        event_id
        for item in checkpoints.values()
        for event_id in item.get("source_event_ids", [])
        if event_id in projection["outcome_ids"]
        or event_id in projection["target_anchor_ids"]
    }
    if leaked_event_ids:
        errors.append(
            f"outcome or target-value event IDs leaked {sorted(leaked_event_ids)}"
        )

    reusable_text = json.dumps(
        {
            "graph_diagnosis": blueprint.get("graph_diagnosis", {}),
            "search_factors": blueprint.get("search_factors", []),
            "checkpoints": blueprint.get("checkpoints", []),
            "topology": blueprint.get("topology", {}),
            "causal_paths": blueprint.get("causal_paths", []),
            "alternative_hypotheses": blueprint.get(
                "alternative_hypotheses", []
            ),
            "forecast_audit_questions": blueprint.get(
                "forecast_audit_questions", []
            ),
        },
        ensure_ascii=False,
    ).lower()
    forbidden_text: set[str] = set()
    for node_id in [
        *projection["outcome_ids"],
        *projection["target_anchor_ids"],
    ]:
        label = _node_label(projection["node_by_id"][node_id]).strip()
        if len(label) >= 4:
            forbidden_text.add(label.lower())
        if node_id in projection["outcome_ids"] and ":" in label:
            answer_label = label.split(":", 1)[1].strip()
            if len(answer_label) >= 4:
                forbidden_text.add(answer_label.lower())
    leaked_text = sorted(
        value
        for value in forbidden_text
        if value in reusable_text
        and not re.fullmatch(r"option\s+\d+", value)
    )
    if leaked_text:
        errors.append(f"outcome or target-value text leaked {leaked_text}")

    historical_specifics = historical_specific_counts(blueprint)
    if (
        blueprint.get("refinement_metadata", {}).get(
            "historical_specifics_sanitized"
        )
        and any(historical_specifics.values())
    ):
        errors.append(
            "historical specifics remain after sanitization "
            f"{historical_specifics}"
        )

    edge_coverage = (
        len(expected_edge_ids & emitted_edge_ids) / len(expected_edge_ids)
        if expected_edge_ids
        else 1.0
    )
    path_precision = (
        (len(covered_pairs) - len(covered_pairs - adjacency))
        / len(covered_pairs)
        if covered_pairs
        else 1.0
    )
    return {
        "status": "pass" if not errors else "fail",
        "errors": errors,
        "metrics": {
            "retained_checkpoint_count": len(checkpoints) - 1,
            "source_edge_count": len(expected_edge_ids),
            "emitted_edge_count": len(emitted_edge_ids),
            "edge_coverage": edge_coverage,
            "path_precision": path_precision,
            "causal_path_count": len(
                blueprint.get("causal_paths", [])
            ),
            "outcome_event_leak_count": len(leaked_event_ids),
            "outcome_text_leak_count": len(leaked_text),
            **historical_specifics,
        },
    }
