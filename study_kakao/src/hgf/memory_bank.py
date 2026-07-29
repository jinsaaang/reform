"""Load the validated 200-case final bank into HGF runtime artifacts."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from hgf.dag import _redact_answer_labels
from hgf.question_io import family_metadata
from hgf.package import PACKAGE_ROOT


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve(repo_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def _question_payload(question: Any) -> dict[str, Any]:
    if hasattr(question, "model_dump"):
        return question.model_dump(mode="json")
    return dict(question)


def _canonical_graph(
    *,
    raw_graph: dict[str, Any],
    question: Any,
    evidence_pack: dict[str, Any],
    audit: dict[str, Any],
) -> dict[str, Any]:
    """Adapt a Codex-refined AC graph without changing its causal content."""
    graph = raw_graph.get("graph", {})
    nodes = [
        {
            "id": node["id"],
            "title": node.get("label"),
            "event_type": node.get("event_type"),
            "occurred_date": node.get("occurred_date"),
            "predicted_date": None,
            "is_outcome": bool(node.get("is_outcome")),
            "is_actual_outcome": bool(node.get("is_actual_outcome")),
            "support_level": node.get("support_level"),
            "article_ids": node.get("article_ids", []),
        }
        for node in graph.get("nodes", [])
    ]
    edges = [
        {
            "id": edge.get("id"),
            "source_event_id": edge.get("source"),
            "target_event_id": edge.get("target"),
            "relation_type": edge.get("relationship"),
            "strength": (
                0.9 if edge.get("support_level") == "observed" else 0.6
            ),
            "confidence": (
                0.95 if edge.get("support_level") == "observed" else 0.6
            ),
            "reasoning": edge.get("rationale"),
            "article_ids": edge.get("article_ids", []),
            "support_level": edge.get("support_level"),
        }
        for edge in graph.get("edges", [])
    ]
    evidence = evidence_pack.get("evidence", [])
    checks = audit.get("checks", {})
    return {
        "question": _question_payload(question),
        "actual_outcome_event_id": raw_graph.get(
            "actual_outcome_event_id"
        ),
        "evidence": {
            "satisfied": bool(evidence_pack.get("gate", {}).get("passed")),
            "article_count": len(evidence),
            "missing_requirements": evidence_pack.get("gate", {}).get(
                "failures", []
            ),
            "articles": evidence,
        },
        "graph": {
            "built": True,
            "satisfied": True,
            "nodes": nodes,
            "edges": edges,
            "metrics": {
                "event_count": checks.get("node_count", len(nodes)),
                "hypothesis_count": len(edges),
                "max_depth": checks.get("maximum_depth"),
                "independent_upstream_branches": checks.get(
                    "independent_upstream_branches"
                ),
            },
            "validation": {
                "status": "pass",
                "source_status": audit.get("status"),
                "checks": checks,
                "caveats": audit.get("caveats", []),
            },
        },
    }




def _compiled_blueprint(
    *,
    graph_payload: dict[str, Any],
    question: Any,
    audit: dict[str, Any],
    source_graph: Path,
) -> dict[str, Any]:
    """Compile leakage-safe reusable cards from a validated refined DAG."""
    graph = graph_payload["graph"]
    metadata = family_metadata(question)
    resolved = getattr(question, "ground_truth", None)
    actual_outcome_id = graph_payload.get("actual_outcome_event_id")
    outcome_predecessors = {
        edge.get("source_event_id")
        for edge in graph.get("edges", [])
        if edge.get("target_event_id") == actual_outcome_id
    }
    counter_ids = {
        edge.get("source_event_id")
        for edge in graph.get("edges", [])
        if edge.get("relation_type") == "counteracts"
    }
    incoming_ids = {
        edge.get("target_event_id") for edge in graph.get("edges", [])
    }
    outgoing_ids = {
        edge.get("source_event_id") for edge in graph.get("edges", [])
    }
    causal_relations = {
        "supports",
        "contributes_to",
        "counteracts",
        "drives",
        "influences",
        "leads_to",
        "inhibits",
    }
    causal_node_ids = {
        endpoint
        for edge in graph.get("edges", [])
        if edge.get("relation_type") in causal_relations
        for endpoint in (
            edge.get("source_event_id"),
            edge.get("target_event_id"),
        )
    }

    candidates = []
    for node in graph.get("nodes", []):
        node_id = node.get("id")
        if (
            node.get("is_outcome")
            or node.get("is_actual_outcome")
            or node_id == actual_outcome_id
            or node_id in outcome_predecessors
            or node_id not in causal_node_ids
        ):
            continue
        label = _redact_answer_labels(node.get("title"), resolved)
        if re.search(
            r"target[- ]quarter|target[- ]minus[- ]prior|resolved outcome|"
            r"actual outcome|pre-registered .*boundary|"
            r"(?:growth|return|change|acceleration) was [-+]?\d",
            str(label),
            flags=re.IGNORECASE,
        ):
            continue
        candidates.append((node, label))
    candidates = candidates[:5]
    usable = len(candidates) >= 2

    checkpoints = []
    search_factors = []
    checkpoint_ids = []
    for checkpoint_number, (node, label) in enumerate(candidates, start=1):
        node_id = str(node.get("id"))
        checkpoint_id = f"checkpoint_{checkpoint_number}"
        checkpoint_ids.append(checkpoint_id)
        role = (
            "counterevidence"
            if node_id in counter_ids
            else "mediator"
            if node_id in incoming_ids and node_id in outgoing_ids
            else "driver"
        )
        checkpoints.append(
            {
                "id": checkpoint_id,
                "role": role,
                "factor": label,
                "mechanism": (
                    "Test whether this historically relevant factor is active "
                    "in the current case and connects to the target metric."
                ),
                "expected_direction": (
                    "mixed" if role == "counterevidence" else "unknown"
                ),
                "evidence_requirement": (
                    "Current cutoff-safe official data or dated independent "
                    "reporting directly covering this factor."
                ),
                "contradiction_signal": (
                    "Current evidence shows the factor is absent, reversed, or "
                    "dominated by a competing mechanism."
                ),
                "historical_support": (
                    "strong"
                    if node.get("support_level") == "observed"
                    else "weak"
                ),
                "source_event_ids": [node_id],
                "source_edge_ids": [],
            }
        )
        search_factors.append(
            {
                "factor": label,
                "why_search": (
                    "This factor appeared on a validated historical causal path; "
                    "verify its current-case state rather than copying its outcome."
                ),
                "preferred_source_types": [
                    "official release",
                    "dated independent reporting",
                ],
                "source_event_ids": [node_id],
            }
        )

    target_bridge_id = "target_bridge"
    checkpoints.append(
        {
            "id": target_bridge_id,
            "role": "target_bridge",
            "factor": str(
                metadata.get("target_metric")
                or "Net transmission into the forecast target"
            ),
            "mechanism": (
                "Aggregate supported drivers and counterevidence into the exact "
                "target metric and option space."
            ),
            "expected_direction": "mixed",
            "evidence_requirement": (
                "A recent target baseline plus evidence connecting each active "
                "driver to the target."
            ),
            "contradiction_signal": (
                "The target baseline or current observations diverge from the "
                "proposed driver path."
            ),
            "historical_support": "medium",
            "source_event_ids": [],
            "source_edge_ids": [],
        }
    )
    causal_paths = []
    if checkpoint_ids:
        midpoint = max(1, len(checkpoint_ids) // 2)
        for path_ids in (
            checkpoint_ids[:midpoint],
            checkpoint_ids[midpoint:],
        ):
            if not path_ids:
                continue
            causal_paths.append(
                {
                    "checkpoint_ids": path_ids[:3] + [target_bridge_id],
                    "generalized_mechanism": (
                        "Current evidence must activate the historical factors "
                        "and demonstrate a prospective bridge to the target."
                    ),
                    "expected_direction": "mixed",
                    "applicability_conditions": [
                        "The same economic mechanism is active.",
                        "Current cutoff-safe evidence supports the intermediate link.",
                    ],
                    "failure_conditions": [
                        "A historical factor is absent in the current case.",
                        "Counterevidence dominates the proposed transmission.",
                    ],
                }
            )

    counter_labels = [
        label
        for node, label in candidates
        if str(node.get("id")) in counter_ids
    ]
    alternatives = [
        {
            "hypothesis": (
                f"{label} dominates the main path in the current case."
            ),
            "discriminating_evidence": (
                "Find current observations that directly compare this mechanism "
                "with the leading supported driver."
            ),
            "source_event_ids": [],
        }
        for label in counter_labels[:2]
    ]
    if not alternatives:
        alternatives = [
            {
                "hypothesis": (
                    "A current-case factor outside the retrieved historical path "
                    "dominates the target."
                ),
                "discriminating_evidence": (
                    "Search for recent official surprises and contradictory "
                    "target observations."
                ),
                "source_event_ids": [],
            }
        ]

    return {
        "schema_version": "hgf_blueprint",
        "question_id": getattr(question, "id"),
        "target_definition": {
            "metric": metadata.get("target_metric"),
            "unit_or_option_space": list(
                getattr(question, "options", None) or []
            ),
            "forecast_horizon": (
                "the current question's exact target period; historical fiscal "
                "quarter labels must not be copied"
            ),
        },
        "graph_diagnosis": {
            "usable": usable,
            "summary": (
                "Validated historical graph compiled into reusable prospective "
                "checkpoints."
                if usable
                else
                "Excluded from HGF retrieval because the graph contains only "
                "outcome calculation/classification structure and no reusable "
                "upstream causal branch."
            ),
            "weaknesses": audit.get("caveats", [])[:3],
        },
        "search_factors": search_factors,
        "checkpoints": checkpoints,
        "causal_paths": causal_paths,
        "alternative_hypotheses": alternatives,
        "forecast_audit_questions": [
            "Which checkpoints are supported, contradicted, or unknown now?",
            "Does each retained path have a current evidence-to-target bridge?",
            "What counterevidence or alternative mechanism could dominate?",
            "Would the conclusion change if the weakest link were removed?",
        ],
        "refinement_metadata": {
            "model": "deterministic_compiler_from_codex_refined_dag",
            "source_graph": str(source_graph),
            "validation_status": "pass",
            "source_audit_status": audit.get("status"),
        },
    }


def load_final_memory_bank(
    manifest_path: Path,
    memory_questions: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return one canonical graph and one HGF blueprint for every memory case."""
    manifest_path = manifest_path.resolve()
    repo_root = PACKAGE_ROOT
    manifest = _read(manifest_path)
    entries = {
        str(entry["question_id"]): entry
        for entry in manifest.get("entries", [])
    }
    graphs = []
    blueprints = []
    missing = sorted(set(memory_questions) - set(entries))
    if missing:
        raise FileNotFoundError(
            "Final memory manifest is missing: " + ", ".join(missing)
        )
    for question_id, question in memory_questions.items():
        entry = entries[question_id]
        graph_path = _resolve(repo_root, entry["graph_path"])
        raw_graph = _read(graph_path)
        guidance_path = entry.get("guidance_path")
        if guidance_path:
            graph_payload = raw_graph
            blueprint = _read(_resolve(repo_root, guidance_path))
        else:
            evidence_path = _resolve(repo_root, entry["evidence_path"])
            audit_path = _resolve(repo_root, entry["audit_path"])
            evidence = _read(evidence_path)
            audit = _read(audit_path)
            graph_payload = _canonical_graph(
                raw_graph=raw_graph,
                question=question,
                evidence_pack=evidence,
                audit=audit,
            )
            blueprint = _compiled_blueprint(
                graph_payload=graph_payload,
                question=question,
                audit=audit,
                source_graph=graph_path,
            )
        graphs.append(graph_payload)
        blueprints.append(blueprint)
    return graphs, blueprints
