"""Deterministic per-question selection from the fixed exemplar bank."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from hgf.contracts import _is_temporally_eligible
from hgf.exemplar import _exemplar_article_ids
from hgf.memory_retrieval import (
    _blueprint_factor_tokens,
    _evidence_tokens,
    _finance_metadata,
    _resolution_timestamp,
    _tokens,
    select_relevant_blueprints,
)
from hgf.question_io import resolve_forecast_cutoff


def load_fixed_exemplar_bank(
    paths: list[Path],
) -> dict[str, dict[str, Any]]:
    """Load and deduplicate worked exemplars by memory-question ID."""
    bank: dict[str, dict[str, Any]] = {}
    canonical: dict[str, str] = {}
    source: dict[str, Path] = {}
    for root in paths:
        if not root.exists():
            continue
        for path in root.rglob("*.json"):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            worked = payload.get("worked_exemplar")
            memory_id = (
                payload.get("retrieved_memory_question_id")
                or payload.get("source_question_id")
                or payload.get("memory_question_id")
            )
            if not isinstance(worked, dict) or not memory_id:
                continue
            key = str(memory_id)
            rendered = json.dumps(
                worked,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            if key in canonical and canonical[key] != rendered:
                raise ValueError(
                    f"conflicting fixed exemplars for {key}: "
                    f"{source[key]} vs {path}"
                )
            canonical[key] = rendered
            source[key] = path
            bank[key] = worked
    return bank


def rank_blueprints_with_scores(
    *,
    blueprints: list[dict[str, Any]],
    memory_questions: dict[str, Any],
    target_question: Any,
    cutoff: Any,
    evidence: list[dict[str, Any]],
    limit: int,
) -> list[dict[str, Any]]:
    """Return the v22 hybrid retrieval order with score metadata."""
    if limit <= 0:
        return []
    eligible = [
        blueprint
        for blueprint in blueprints
        if (
            str(blueprint.get("question_id")) in memory_questions
            and _is_temporally_eligible(
                memory_questions[str(blueprint["question_id"])],
                cutoff,
            )
            and blueprint.get("graph_diagnosis", {}).get("usable") is not False
        )
    ]
    evidence_tokens = _evidence_tokens(evidence)
    target_metadata = _finance_metadata(target_question)
    target_tokens = _tokens(
        getattr(target_question, "question_text", "")
        + " "
        + str(target_metadata.get("target_metric") or "")
        + " "
        + str(target_metadata.get("subdomain") or "")
    )
    candidates = []
    for blueprint in eligible:
        question_id = str(blueprint["question_id"])
        memory_question = memory_questions[question_id]
        memory_metadata = _finance_metadata(memory_question)
        score = 0
        for field, weight in (
            ("family_id", 20),
            ("entity", 12),
            ("target_metric", 10),
            ("subdomain", 8),
            ("category", 5),
            ("region", 2),
            ("change_unit", 2),
        ):
            target_value = target_metadata.get(field)
            if target_value and target_value == memory_metadata.get(field):
                score += weight
        if getattr(target_question, "question_type", None) == getattr(
            memory_question,
            "question_type",
            None,
        ):
            score += 2
        content = " ".join(
            [
                json.dumps(
                    blueprint.get("target_definition", {}),
                    ensure_ascii=False,
                ),
                " ".join(
                    str(item.get("factor") or "")
                    for item in blueprint.get("search_factors", [])
                ),
                " ".join(
                    str(item.get("factor") or "")
                    for item in blueprint.get("checkpoints", [])
                ),
            ]
        )
        score += min(6, len(target_tokens & _tokens(content)))
        if evidence_tokens:
            score += min(
                12,
                len(
                    evidence_tokens
                    & _blueprint_factor_tokens(blueprint)
                ),
            )
        if score > 0:
            candidates.append(
                {
                    "score": score,
                    "resolution": _resolution_timestamp(memory_question),
                    "family_id": str(
                        memory_metadata.get("family_id") or ""
                    ),
                    "blueprint": blueprint,
                }
            )

    selected = []
    family_counts: dict[str, int] = {}
    while candidates and len(selected) < limit:

        def adjusted(item: dict[str, Any]) -> tuple[float, float, str]:
            family_count = family_counts.get(item["family_id"], 0)
            diversity_penalty = max(0, family_count - 2) * 3
            return (
                item["score"] - diversity_penalty,
                item["resolution"],
                str(item["blueprint"]["question_id"]),
            )

        best = max(candidates, key=adjusted)
        candidates.remove(best)
        adjusted_score = adjusted(best)[0]
        selected.append(
            {
                "rank": len(selected) + 1,
                "memory_question_id": str(
                    best["blueprint"]["question_id"]
                ),
                "score": best["score"],
                "adjusted_score": adjusted_score,
                "resolution_timestamp": best["resolution"],
                "family_id": best["family_id"],
                "blueprint": best["blueprint"],
            }
        )
        family_counts[best["family_id"]] = (
            family_counts.get(best["family_id"], 0) + 1
        )

    parity = select_relevant_blueprints(
        eligible,
        memory_questions,
        target_question,
        limit=limit,
        evidence=evidence,
    )
    if [row["memory_question_id"] for row in selected] != [
        str(blueprint["question_id"]) for blueprint in parity
    ]:
        raise AssertionError("ranking metadata implementation diverged")
    return selected


def choose_top_k_ranked(
    ranked: list[dict[str, Any]],
    *,
    top_k: int,
    evidence_floor: int = 2,
    legacy_window: int = 5,
) -> list[dict[str, Any]]:
    """Extend the v22 top-1 evidence floor into a deterministic top-k list."""
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if not ranked:
        raise ValueError("no eligible exemplar candidates")
    if top_k > len(ranked):
        raise ValueError(
            f"top_k={top_k} exceeds {len(ranked)} eligible candidates"
        )

    first = next(
        (
            row
            for row in ranked[:legacy_window]
            if int(
                row.get(
                    "historical_forecast_time_evidence_count",
                    0,
                )
            )
            >= evidence_floor
        ),
        ranked[0],
    )
    first_reason = (
        "v22_evidence_floor"
        if int(
            first.get(
                "historical_forecast_time_evidence_count",
                0,
            )
        )
        >= evidence_floor
        else "v22_rank_one_fallback"
    )
    ordered: list[tuple[dict[str, Any], str]] = [(first, first_reason)]
    selected_ids = {str(first["memory_question_id"])}

    for row in ranked:
        memory_id = str(row["memory_question_id"])
        evidence_count = int(
            row.get("historical_forecast_time_evidence_count", 0)
        )
        if memory_id in selected_ids or evidence_count < evidence_floor:
            continue
        ordered.append((row, "evidence_floor"))
        selected_ids.add(memory_id)
        if len(ordered) == top_k:
            break

    if len(ordered) < top_k:
        for row in ranked:
            memory_id = str(row["memory_question_id"])
            if memory_id in selected_ids:
                continue
            ordered.append((row, "score_order_backfill"))
            selected_ids.add(memory_id)
            if len(ordered) == top_k:
                break

    selected = []
    for selection_rank, (row, reason) in enumerate(
        ordered[:top_k],
        start=1,
    ):
        payload = copy.copy(row)
        payload["score_rank"] = int(row["rank"])
        payload["rank"] = selection_rank
        payload["selection_reason"] = reason
        selected.append(payload)
    return selected


def select_rule_based_exemplars(
    *,
    blueprints: list[dict[str, Any]],
    graphs_by_id: dict[str, dict[str, Any]],
    memory_questions: dict[str, Any],
    target_question: Any,
    cutoff: Any,
    evidence: list[dict[str, Any]],
    top_k: int,
) -> list[dict[str, Any]]:
    """Score the eligible memory bank and return v22-compatible top-k."""
    ranked = rank_blueprints_with_scores(
        blueprints=blueprints,
        memory_questions=memory_questions,
        target_question=target_question,
        cutoff=cutoff,
        evidence=evidence,
        limit=len(blueprints),
    )
    annotated = []
    for row in ranked:
        memory_id = str(row["memory_question_id"])
        historical_cutoff, _ = resolve_forecast_cutoff(
            memory_questions[memory_id]
        )
        payload = copy.copy(row)
        payload["historical_forecast_time_evidence_count"] = len(
            _exemplar_article_ids(
                graphs_by_id[memory_id],
                historical_cutoff,
            )
        )
        annotated.append(payload)
    return choose_top_k_ranked(annotated, top_k=top_k)


def namespace_expert_memory(
    expert_memory: dict[str, Any],
    *,
    rank: int,
    memory_id: str,
) -> tuple[dict[str, Any], list[str]]:
    """Give checkpoints unique IDs before combining multiple memories."""
    payload = copy.deepcopy(expert_memory)
    prefix = f"M{rank}:{memory_id}:"
    mapping = {
        str(item["checkpoint_id"]): prefix + str(item["checkpoint_id"])
        for item in payload.get("causal_checkpoint_library", [])
    }
    for item in payload.get("causal_checkpoint_library", []):
        item["checkpoint_id"] = mapping[str(item["checkpoint_id"])]
    for mechanism in payload.get("mechanism_library", []):
        mechanism["checkpoint_ids"] = [
            mapping.get(str(value), prefix + str(value))
            for value in mechanism.get("checkpoint_ids", [])
        ]
    payload["rank"] = rank
    payload["source_question_id"] = memory_id
    return payload, list(mapping.values())
