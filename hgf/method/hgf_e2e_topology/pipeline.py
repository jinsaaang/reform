"""Independent preprocessing for end-to-end topology HGF.

This module deliberately does not import any earlier experimental HGF package.
It reads the canonical question, evidence, and Blueprint artifacts directly.
"""

from __future__ import annotations

import copy
import json
import re
from typing import Any

from hgf.exemplar import _call_with_repair
from hgf.forecast_core import _seed
from openai import OpenAI

_STOPWORDS = {
    "about",
    "after",
    "before",
    "change",
    "current",
    "evidence",
    "factor",
    "financial",
    "forecast",
    "from",
    "growth",
    "historical",
    "monthly",
    "period",
    "target",
    "that",
    "the",
    "this",
    "with",
}

_MEASUREMENT_RELATIONS = {
    "classified_as",
    "classifies_as",
    "classification_rule_for",
    "input_to",
    "maps_to",
    "provides_baseline_for",
    "yields",
}


def _tokens(value: Any) -> set[str]:
    rendered = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return {
        token
        for token in re.findall(r"[a-z0-9]{3,}", rendered.lower())
        if token not in _STOPWORDS
        and not re.fullmatch(r"(?:19|20)\d{2}", token)
        and token not in {"required", "unknown", "supported"}
    }


def add_usage(*items: dict[str, int]) -> dict[str, int]:
    result: dict[str, int] = {}
    for item in items:
        for key, value in item.items():
            if isinstance(value, int):
                result[key] = result.get(key, 0) + value
    return result


def _ledger_schema() -> dict[str, Any]:
    return {
        "name": "e2e_topology_current_evidence_ledger",
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
                        "observation": {"type": "string", "maxLength": 600},
                        "evidence_ids": {
                            "type": "array",
                            "maxItems": 6,
                            "items": {"type": "string"},
                        },
                        "assessment": {"type": "string", "maxLength": 600},
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
                            "signal_id": {"type": "string", "maxLength": 40},
                            "factor": {"type": "string", "maxLength": 180},
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
                            "observation": {"type": "string", "maxLength": 600},
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
                "ledger_summary": {"type": "string", "maxLength": 700},
            },
            "required": [
                "target_baseline",
                "current_signals",
                "data_gaps",
                "ledger_summary",
            ],
        },
    }


def _validate_ledger(
    payload: dict[str, Any], evidence_ids: set[str]
) -> tuple[dict[str, float], list[str]]:
    errors: list[str] = []
    baseline = payload.get("target_baseline", {})
    baseline["evidence_ids"] = sorted(
        {
            str(value)
            for value in baseline.get("evidence_ids", [])
            if str(value) in evidence_ids
        }
    )
    if baseline.get("status") == "observed" and not baseline["evidence_ids"]:
        baseline["status"] = "unavailable"
    signals = []
    seen: set[str] = set()
    for item in payload.get("current_signals", []):
        signal_id = str(item.get("signal_id") or "")
        used = sorted(
            {
                str(value)
                for value in item.get("evidence_ids", [])
                if str(value) in evidence_ids
            }
        )
        if not signal_id or signal_id in seen or not used:
            continue
        item["evidence_ids"] = used
        signals.append(item)
        seen.add(signal_id)
    payload["current_signals"] = signals
    if not signals:
        errors.append("current evidence ledger has no grounded signal")
    if {"prediction", "option_probabilities", "forecast"} & set(payload):
        errors.append("evidence ledger contains a forecast output")
    return {}, errors


def _deterministic_ledger(evidence: list[dict[str, Any]]) -> dict[str, Any]:
    signals = []
    for index, item in enumerate(evidence[:8], start=1):
        evidence_id = str(item.get("id") or "")
        if not evidence_id:
            continue
        signals.append(
            {
                "signal_id": f"fallback_signal_{index}",
                "factor": str(
                    item.get("title") or item.get("source") or "current evidence"
                )[:180],
                "state": "unknown",
                "temporal_role": "structural",
                "observation": str(
                    item.get("excerpt")
                    or item.get("content")
                    or item.get("title")
                    or "Current evidence retained without interpretation."
                )[:600],
                "evidence_ids": [evidence_id],
            }
        )
    if len(signals) < 2:
        raise ValueError("deterministic ledger requires at least two evidence items")
    return {
        "target_baseline": {
            "status": "unavailable",
            "observation": "No target baseline was inferred.",
            "evidence_ids": [],
            "assessment": "Fallback preserves evidence without a baseline claim.",
        },
        "current_signals": signals,
        "data_gaps": ["Signal states were not inferred in fallback."],
        "ledger_summary": "Raw current evidence retained without forecasting.",
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
) -> tuple[dict[str, Any], dict[str, int], float, bool]:
    """Read current evidence without seeing a DAG or producing a forecast."""
    evidence_ids = {str(item["id"]) for item in evidence}
    prompt = (
        "Read the current cutoff-safe evidence before any historical DAG is "
        "shown. Build a compact factual ledger for the exact financial target. "
        "Record the latest usable target baseline, current drivers, their "
        "temporal relation to the target period, and missing information. Do "
        "not forecast, choose an option, assign probabilities, or infer what a "
        "past event implies. Distinguish a level from a change, a broad outlook "
        "from the target period, and a leading signal from a direct target "
        "observation. Use only supplied evidence IDs.\n\n"
        f"CURRENT CASE:\n{json.dumps(public_case, ensure_ascii=False)}\n\n"
        "DETERMINISTIC TARGET OPERATOR:\n"
        f"{json.dumps(target_operator, ensure_ascii=False)}\n\n"
        "CURRENT CUTOFF-SAFE EVIDENCE:\n"
        f"{json.dumps(evidence, ensure_ascii=False)}"
    )
    ledger, _, usage, seconds, repaired = _call_with_repair(
        client,
        model=model,
        system=(
            "You extract a cutoff-safe financial evidence ledger. You never "
            "produce an answer or probability. Return JSON."
        ),
        prompt=prompt,
        schema=_ledger_schema(),
        seed=_seed(question_id, "e2e-topology-evidence-ledger-v1"),
        max_tokens=max_tokens,
        validator=lambda candidate: _validate_ledger(candidate, evidence_ids),
        fallback_factory=lambda _current, _errors: _deterministic_ledger(evidence),
    )
    return ledger, usage, seconds, repaired


_ARTICLE_ID = re.compile(r"\bart_[A-Za-z0-9_]+\b")


def _clean_historical_reasoning(value: Any, *, limit: int = 500) -> str:
    """Keep the reasoning sentence while removing irrelevant article handles."""
    text = _ARTICLE_ID.sub("historical evidence", str(value or ""))
    text = re.sub(r"\s+", " ", text).strip()
    return text[:limit]


def compile_worked_reasoning_check(
    worked_exemplar: dict[str, Any],
) -> dict[str, Any]:
    """Project a cutoff-safe exemplar into an answer-free review pattern."""
    sequence = []
    for value in worked_exemplar.get("expert_reasoning", [])[:7]:
        cleaned = _clean_historical_reasoning(value)
        if cleaned:
            sequence.append(cleaned)
    if not sequence:
        raise ValueError("worked exemplar has no reusable reasoning sequence")
    return {
        "task_signature": copy.deepcopy(worked_exemplar.get("task_signature", {})),
        "historical_target": _clean_historical_reasoning(
            worked_exemplar.get("target_semantics"), limit=350
        ),
        "reasoning_sequence": sequence,
        "counterevidence_check": _clean_historical_reasoning(
            worked_exemplar.get("counterevidence")
        ),
        "uncertainty_check": _clean_historical_reasoning(
            worked_exemplar.get("uncertainty")
        ),
        "structural_lesson": _clean_historical_reasoning(
            worked_exemplar.get("dag_derived_lesson")
        ),
        "contract": (
            "Use this historical trace only to review the organization and "
            "completeness of current reasoning. It is not current evidence and "
            "does not provide the current answer, direction, or probability."
        ),
    }


def compile_topology_memory(
    blueprints: list[dict[str, Any]],
    exemplars_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Namespace validated root-to-target paths without rewriting them."""
    if not blueprints:
        raise ValueError("at least one topology Blueprint is required")
    sources = []
    all_paths = []
    for source_index, blueprint in enumerate(blueprints, start=1):
        if blueprint.get("schema_version") != "hgf_blueprint_topology_v2":
            raise ValueError("end-to-end HGF accepts only topology v2 artifacts")
        source_id = str(blueprint.get("question_id") or "")
        worked_exemplar = exemplars_by_id.get(source_id)
        if not isinstance(worked_exemplar, dict):
            raise ValueError(f"missing fixed worked exemplar for {source_id}")
        namespace = f"D{source_index}"
        checkpoints = {
            str(item["id"]): copy.deepcopy(item)
            for item in blueprint.get("checkpoints", [])
            if item.get("id")
        }
        edges = {
            str(item.get("source_edge_ids", [""])[0]): copy.deepcopy(item)
            for item in blueprint.get("topology", {}).get("edges", [])
            if item.get("source_edge_ids")
        }
        source_paths = []
        for path_index, path in enumerate(
            blueprint.get("causal_paths", []), start=1
        ):
            raw_checkpoint_ids = [
                str(value) for value in path.get("checkpoint_ids", [])
            ]
            raw_edge_ids = [str(value) for value in path.get("source_edge_ids", [])]
            if len(raw_checkpoint_ids) < 2:
                continue
            path_edges = [edges[value] for value in raw_edge_ids if value in edges]
            if len(path_edges) != len(raw_checkpoint_ids) - 1:
                raise ValueError(
                    f"{source_id} path does not preserve one edge per transition"
                )
            namespaced_checkpoints = []
            for checkpoint_id in raw_checkpoint_ids:
                checkpoint = checkpoints[checkpoint_id]
                namespaced_checkpoints.append(
                    {
                        "id": f"{namespace}:{checkpoint_id}",
                        "source_checkpoint_id": checkpoint_id,
                        "role": str(checkpoint.get("role") or ""),
                        "factor": str(checkpoint.get("factor") or ""),
                        "mechanism": str(checkpoint.get("mechanism") or ""),
                        "expected_direction": str(
                            checkpoint.get("expected_direction") or "unknown"
                        ),
                        "evidence_requirement": str(
                            checkpoint.get("evidence_requirement") or ""
                        ),
                        "contradiction_signal": str(
                            checkpoint.get("contradiction_signal") or ""
                        ),
                        "historical_support": str(
                            checkpoint.get("historical_support") or ""
                        ),
                        "source_event_type": str(
                            checkpoint.get("source_event_type") or ""
                        ),
                    }
                )
            namespaced_edges = [
                {
                    "source_checkpoint_id": f"{namespace}:{edge['source_checkpoint_id']}",
                    "target_checkpoint_id": f"{namespace}:{edge['target_checkpoint_id']}",
                    "relationship": str(edge.get("relationship") or ""),
                    "directionality": str(edge.get("directionality") or ""),
                    "support_level": str(edge.get("support_level") or ""),
                    "rationale": str(edge.get("rationale") or ""),
                    "terminal_to_target_bridge": bool(
                        edge.get("terminal_to_target_bridge")
                    ),
                }
                for edge in path_edges
            ]
            compiled = {
                "id": f"{namespace}:path_{path_index}",
                "source_question_id": source_id,
                "source_path_role": str(path.get("path_role") or ""),
                "checkpoints": namespaced_checkpoints,
                "checkpoint_ids": [item["id"] for item in namespaced_checkpoints],
                "edges": namespaced_edges,
                "relationships": list(path.get("relationships", [])),
                "mechanism": str(path.get("generalized_mechanism") or ""),
                "applicability_conditions": list(
                    path.get("applicability_conditions", [])
                ),
                "failure_conditions": list(path.get("failure_conditions", [])),
            }
            source_paths.append(compiled)
            all_paths.append(compiled)
        sources.append(
            {
                "rank": source_index,
                "source_question_id": source_id,
                "target_definition": copy.deepcopy(
                    blueprint.get("target_definition", {})
                ),
                "graph_diagnosis": copy.deepcopy(
                    blueprint.get("graph_diagnosis", {})
                ),
                "worked_reasoning_check": compile_worked_reasoning_check(
                    worked_exemplar
                ),
                "path_count": len(source_paths),
            }
        )
    return {
        "schema_version": "e2e_topology_memory_v1",
        "source_dags": sources,
        "candidate_paths": all_paths,
        "contract": {
            "historical_exemplar": "answer-free reasoning check only",
            "historical_answer": "excluded",
            "topology": "exact validated root-to-target paths",
            "edge_semantics": "conditionally instantiated structural knowledge",
        },
    }


def route_topology_subgraphs(
    memory: dict[str, Any],
    evidence_ledger: dict[str, Any],
    *,
    max_paths: int = 3,
    max_checkpoints: int = 12,
) -> dict[str, Any]:
    """Select complete evidence-relevant paths without topology rewriting."""
    ledger_tokens = _tokens(evidence_ledger)
    ranked = []
    for index, path in enumerate(memory.get("candidate_paths", [])):
        overlap = ledger_tokens & _tokens(path)
        support = sum(
            2 if edge.get("support_level") == "observed" else 1
            for edge in path.get("edges", [])
        )
        measurement_edges = sum(
            str(edge.get("relationship") or "") in _MEASUREMENT_RELATIONS
            for edge in path.get("edges", [])
        )
        causal_edges = len(path.get("edges", [])) - measurement_edges
        placeholder_count = sum(
            str(checkpoint.get("factor") or "").count("[CURRENT_")
            for checkpoint in path.get("checkpoints", [])
        )
        score = (
            3 * len(overlap)
            + 5 * causal_edges
            + support
            - 5 * measurement_edges
            - 2 * placeholder_count
        )
        ranked.append(
            (
                score,
                len(overlap),
                support,
                -index,
                path,
                sorted(overlap),
                causal_edges,
                measurement_edges,
            )
        )
    ranked.sort(reverse=True, key=lambda item: item[:4])

    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    selected_checkpoints: set[str] = set()
    selected_sources: set[str] = set()
    selected_source_order: list[str] = []

    def add(candidate: tuple[Any, ...]) -> bool:
        score, _, _, _, path, overlap, causal_edges, measurement_edges = candidate
        path_id = str(path["id"])
        if path_id in selected_ids:
            return False
        new_checkpoints = set(path.get("checkpoint_ids", [])) - selected_checkpoints
        if selected and len(selected_checkpoints) + len(new_checkpoints) > max_checkpoints:
            return False
        selected.append(
            {
                **copy.deepcopy(path),
                "routing_overlap": overlap,
                "routing_score": score,
                "causal_edge_count": causal_edges,
                "measurement_edge_count": measurement_edges,
            }
        )
        selected_ids.add(path_id)
        selected_checkpoints.update(path.get("checkpoint_ids", []))
        source_id = str(path.get("source_question_id") or "")
        if source_id not in selected_sources:
            selected_source_order.append(source_id)
        selected_sources.add(source_id)
        return True

    if ranked:
        add(ranked[0])
    best_score = ranked[0][0] if ranked else 0
    for candidate in ranked[1:]:
        source_id = str(candidate[4].get("source_question_id") or "")
        if source_id not in selected_sources and candidate[0] >= best_score - 8:
            add(candidate)
        if len(selected) >= max_paths:
            break
    for candidate in ranked:
        if len(selected) >= max_paths:
            break
        add(candidate)
    if not selected:
        raise ValueError("topology memory has no routable path")
    source_by_id = {
        str(source.get("source_question_id") or ""): source
        for source in memory.get("source_dags", [])
    }
    return {
        "schema_version": "e2e_topology_subgraphs_v1",
        "source_dags": [
            {
                key: copy.deepcopy(value)
                for key, value in source.items()
                if key != "worked_reasoning_check"
            }
            for source in memory.get("source_dags", [])
        ],
        "paths": selected,
        "worked_reasoning_checks": [
            {
                "source_question_id": source_id,
                **copy.deepcopy(source_by_id[source_id]["worked_reasoning_check"]),
            }
            for source_id in selected_source_order
            if source_id in source_by_id
        ],
        "routing": {
            "candidate_path_count": len(ranked),
            "selected_path_ids": [str(path["id"]) for path in selected],
            "selected_source_question_ids": sorted(selected_sources),
            "checkpoint_count": len(selected_checkpoints),
            "max_paths": max_paths,
            "max_checkpoints": max_checkpoints,
            "evidence_first": True,
        },
        "contract": copy.deepcopy(memory.get("contract", {})),
    }


def select_forecast_evidence(
    evidence: list[dict[str, Any]],
    ledger: dict[str, Any],
    *,
    limit: int = 14,
) -> list[dict[str, Any]]:
    """Keep every ledger citation, then fill remaining slots by E1 rank."""
    cited = {
        str(value)
        for value in ledger.get("target_baseline", {}).get("evidence_ids", [])
    }
    cited.update(
        str(value)
        for signal in ledger.get("current_signals", [])
        for value in signal.get("evidence_ids", [])
    )
    selected = [item for item in evidence if str(item.get("id")) in cited]
    selected_ids = {str(item.get("id")) for item in selected}
    for item in evidence:
        if len(selected) >= max(limit, len(selected)):
            break
        if str(item.get("id")) not in selected_ids:
            selected.append(item)
            selected_ids.add(str(item.get("id")))
    return selected
