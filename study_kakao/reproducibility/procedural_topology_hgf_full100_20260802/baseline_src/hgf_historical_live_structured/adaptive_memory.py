"""Adaptive selection and full-fidelity merging of hindsight-DAG memories."""

from __future__ import annotations

import copy
from collections import Counter
from typing import Any

from .exemplar import _transferable_dag_structure
from hgf.memory_retrieval import _blueprint_factor_tokens, _evidence_tokens


_ROUTING_STOPWORDS = {
    "and",
    "can",
    "conditions",
    "current",
    "data",
    "evidence",
    "for",
    "forecast",
    "historical",
    "market",
    "markets",
    "more",
    "outlook",
    "target",
    "than",
    "the",
    "with",
}


def select_adaptive_dags(
    *,
    ranked_blueprints: list[dict[str, Any]],
    evidence: list[dict[str, Any]],
    max_dags: int = 3,
    coverage_threshold: float = 0.8,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Greedily add full DAGs until evidence-linked factor coverage is met."""
    if max_dags <= 0:
        return [], {
            "candidate_dag_ids": [],
            "selected_dag_ids": [],
            "coverage": 0.0,
            "stopping_reason": "max_dags_is_zero",
        }
    evidence_tokens = _evidence_tokens(evidence) - _ROUTING_STOPWORDS
    candidates: list[dict[str, Any]] = []
    for rank, blueprint in enumerate(ranked_blueprints, start=1):
        matched = (
            evidence_tokens
            & _blueprint_factor_tokens(blueprint)
        ) - _ROUTING_STOPWORDS
        candidates.append(
            {
                "rank": rank,
                "blueprint": blueprint,
                "memory_question_id": str(blueprint["question_id"]),
                "matched_terms": matched,
            }
        )
    term_counts = Counter(
        term
        for item in candidates
        for term in item["matched_terms"]
    )
    coverable = {
        term for term, count in term_counts.items() if count >= 2
    }
    if not coverable:
        coverable = set().union(
            *(item["matched_terms"] for item in candidates)
        )
    selected: list[dict[str, Any]] = []
    covered: set[str] = set()
    stopping_reason = "candidate_pool_exhausted"
    for candidate in candidates:
        if len(selected) >= max_dags:
            stopping_reason = "max_dags_reached"
            break
        marginal = (candidate["matched_terms"] & coverable) - covered
        if selected and not marginal:
            continue
        selected.append(candidate)
        covered.update(candidate["matched_terms"] & coverable)
        coverage = (
            len(covered) / len(coverable)
            if coverable
            else 0.0
        )
        if coverage >= coverage_threshold:
            stopping_reason = "coverage_threshold_met"
            break
    if not selected and candidates:
        selected = [candidates[0]]
        stopping_reason = "fallback_to_top_ranked_dag"
    trace = {
        "candidate_dag_ids": [
            item["memory_question_id"] for item in candidates
        ],
        "selected_dag_ids": [
            item["memory_question_id"] for item in selected
        ],
        "coverage": (
            len(covered) / len(coverable)
            if coverable
            else 0.0
        ),
        "covered_term_count": len(covered),
        "coverable_term_count": len(coverable),
        "coverage_threshold": coverage_threshold,
        "max_dags": max_dags,
        "stopping_reason": stopping_reason,
        "selected": [
            {
                "rank": item["rank"],
                "memory_question_id": item["memory_question_id"],
                "matched_terms": sorted(item["matched_terms"]),
            }
            for item in selected
        ],
    }
    return [item["blueprint"] for item in selected], trace


def merge_primary_memory_with_full_dag_structures(
    *,
    primary_memory: dict[str, Any],
    selected_blueprints: list[dict[str, Any]],
    routing_trace: dict[str, Any],
) -> dict[str, Any]:
    """Keep full HGF guidance and add complete same-family DAG structures."""
    if not selected_blueprints:
        raise ValueError("at least one selected DAG is required")
    primary = copy.deepcopy(primary_memory)
    checkpoints = []
    mechanisms = []
    alternatives = []
    audits = []
    source_ids = []
    structure_metadata = []
    for rank, blueprint in enumerate(selected_blueprints, start=1):
        source_id = str(blueprint["question_id"])
        source_ids.append(source_id)
        structure = _transferable_dag_structure(blueprint)
        raw_checkpoints = [
            item
            for item in structure.get("checkpoints", [])
            if item.get("id")
        ][:7]
        mapping = {
            str(item["id"]): (
                f"D{rank}:{str(item['id'])}"
            )
            for item in raw_checkpoints
        }
        structure_metadata.append(
            {
                "rank": rank,
                "source_question_id": source_id,
                "target_definition": structure.get(
                    "target_definition", {}
                ),
                "graph_diagnosis": structure.get(
                    "graph_diagnosis", {}
                ),
            }
        )
        for checkpoint in raw_checkpoints:
            checkpoint_id = str(checkpoint["id"])
            checkpoints.append(
                {
                    "checkpoint_id": mapping[checkpoint_id],
                    "causal_role": str(
                        checkpoint.get("role") or ""
                    ),
                    "factor_role": str(
                        checkpoint.get("factor") or ""
                    ),
                    "evidence_requirement": str(
                        checkpoint.get("evidence_requirement") or ""
                    ),
                    "contradiction_signal": str(
                        checkpoint.get("contradiction_signal") or ""
                    ),
                    "source_question_id": source_id,
                }
            )
        for path in structure.get("causal_paths", [])[:3]:
            kept_ids = [
                mapping[str(value)]
                for value in path.get("checkpoint_ids", [])
                if str(value) in mapping
            ]
            if len(kept_ids) < 2:
                continue
            mechanisms.append(
                {
                    "checkpoint_ids": kept_ids,
                    "mechanism": str(
                        path.get("generalized_mechanism") or ""
                    ),
                    "applicable_when": [
                        str(value)
                        for value in path.get(
                            "applicability_conditions", []
                        )[:3]
                    ],
                    "fails_when": [
                        str(value)
                        for value in path.get(
                            "failure_conditions", []
                        )[:3]
                    ],
                    "source_question_id": source_id,
                }
            )
        for alternative in structure.get(
            "alternative_hypotheses", []
        ):
            alternatives.append(
                {
                    "hypothesis": str(
                        alternative.get("hypothesis") or ""
                    ),
                    "discriminating_evidence": str(
                        alternative.get(
                            "discriminating_evidence"
                        ) or ""
                    ),
                    "source_question_id": source_id,
                }
            )
        for question in structure.get("forecast_audit_questions", []):
            audits.append(
                {
                    "question": str(question),
                    "source_question_id": source_id,
                }
            )
    return {
        "schema_version": "adaptive_full_dag_expert_memory",
        "source_question_ids": source_ids,
        "demonstration_source_question_id": str(
            primary["source_question_id"]
        ),
        "memory_provenance": (
            "The original full worked exemplar and semantic guidance are "
            "preserved. Adaptive routing adds complete reusable checkpoint, "
            "mechanism, alternative, audit, and diagnosis structures from "
            "resolved events in the same recurring family."
        ),
        "task_signature": primary.get("task_signature", {}),
        "expert_reasoning_demonstration": primary.get(
            "expert_reasoning_demonstration", {}
        ),
        "causal_checkpoint_library": checkpoints,
        "mechanism_library": mechanisms,
        "alternative_explanations": alternatives,
        "audit_questions": audits,
        "dag_derived_semantic_lessons": primary.get(
            "dag_derived_semantic_lessons", {}
        ),
        "selected_dag_structures": structure_metadata,
        "adaptive_routing": copy.deepcopy(routing_trace),
        "transfer_rule": (
            "Use current cutoff-safe evidence before memory. The worked "
            "exemplar demonstrates procedure only. Treat every selected full "
            "DAG structure as procedural expertise, not current evidence. "
            "Instantiate only currently supported checkpoints, test "
            "contradiction and failure conditions across structures, and use "
            "CURRENT_NEW when the current case requires an unrepresented "
            "factor. Do not copy historical facts, directions, or outcomes."
        ),
    }
