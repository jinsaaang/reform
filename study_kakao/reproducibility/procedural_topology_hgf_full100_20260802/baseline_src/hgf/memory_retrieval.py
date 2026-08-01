"""Retrieve and compile leakage-safe HGF blueprints for role-specific use."""

from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any

from hgf.dag import _finance_metadata
from hgf.contracts import _is_temporally_eligible


_STOPWORDS = {
    "about",
    "after",
    "before",
    "change",
    "during",
    "financial",
    "forecast",
    "from",
    "into",
    "monthly",
    "question",
    "quarterly",
    "range",
    "that",
    "the",
    "this",
    "what",
    "which",
    "will",
    "with",
}

_MONTH_NAMES = (
    "january|february|march|april|june|july|august|september|"
    "october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|"
    "sept|oct|nov|dec"
)


def _tokens(value: Any) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]{3,}", str(value or "").lower())
        if token not in _STOPWORDS
    }


def _generalize_factor(value: Any) -> str:
    """Remove historical period, realized value, and outcome literals."""
    text = str(value or "").strip()
    text = re.sub(
        r"\b(?:question\s+)?resolves?\b[^,;|]*",
        " ",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\b(?:resolved|actual)\s+outcome\b[^,;|]*",
        " ",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\b(?:above|below|within)\s+(?:the\s+)?recent\s+range\b",
        " ",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        rf"\b(?:{_MONTH_NAMES}|May)\s+\d{{1,2}}(?:st|nd|rd|th)?"
        r"(?:,\s*(?:19|20)\d{2})?\b",
        " ",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\b(?:q[1-4]|first|second|third|fourth)\s+quarter"
        r"(?:\s+(?:19|20)\d{2})?\b",
        " ",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\b(?:19|20)\d{2}(?:[-/]\d{1,2}){0,2}\b", " ", text)
    text = re.sub(r"\b\d{1,2}[-/]\d{1,2}(?:[-/]\d{2,4})?\b", " ", text)
    text = re.sub(
        rf"\b(?:{_MONTH_NAMES})\b",
        " ",
        text,
        flags=re.IGNORECASE,
    )
    # "May" is intentionally case-sensitive to avoid removing the modal verb.
    text = re.sub(r"\bMay\b", " ", text)
    text = re.sub(
        r"\b(?:of|at|was|is)?\s*[+-]?(?:\d+(?:\.\d+)?|\.\d+)\s*"
        r"(?:%(?!\w)|(?:percent|percentage[\s-]*points?|"
        r"basis[\s-]*points?|bps?)\b)",
        " ",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\b(?:resolved|outcome)\s*(?:to|as|:)?\b",
        " ",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\boption\s*\d+\b", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"[\s:;,_/|(){}\[\]—–-]+", " ", text).strip(" .")
    return text or "target metric"


def _evidence_tokens(evidence: Any) -> set[str]:
    if evidence is None:
        return set()
    try:
        rendered = json.dumps(
            evidence,
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )
    except (TypeError, ValueError):
        rendered = str(evidence)
    return _tokens(rendered)


def _blueprint_factor_tokens(blueprint: dict[str, Any]) -> set[str]:
    content = []
    for item in blueprint.get("search_factors", []):
        content.extend(
            [
                item.get("factor"),
                item.get("why_search"),
                item.get("preferred_source_types"),
            ]
        )
    for item in blueprint.get("checkpoints", []):
        content.extend(
            [
                item.get("factor"),
                item.get("mechanism"),
                item.get("evidence_requirement"),
                item.get("contradiction_signal"),
            ]
        )
    return _tokens(content)


def _resolution_timestamp(question: Any) -> float:
    raw = getattr(question, "resolution_date", None)
    if raw is None and isinstance(question, dict):
        raw = question.get("resolution_date")
    try:
        if isinstance(raw, datetime):
            return raw.timestamp()
        return datetime.fromisoformat(str(raw).replace("Z", "+00:00")).timestamp()
    except (TypeError, ValueError):
        return 0.0




def select_relevant_blueprints(
    blueprints: list[dict[str, Any]],
    memory_questions: dict[str, Any],
    target_question: Any,
    limit: int = 5,
    evidence: Any | None = None,
) -> list[dict[str, Any]]:
    """Hybrid retrieval with optional evidence-to-factor reranking."""
    if limit <= 0:
        return []
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
    for blueprint in blueprints:
        if blueprint.get("graph_diagnosis", {}).get("usable") is False:
            continue
        question_id = str(blueprint["question_id"])
        memory_question = memory_questions.get(question_id)
        if memory_question is None:
            continue
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
            memory_question, "question_type", None
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
                len(evidence_tokens & _blueprint_factor_tokens(blueprint)),
            )
        if score > 0:
            candidates.append(
                {
                    "score": score,
                    "resolution": _resolution_timestamp(memory_question),
                    "family_id": str(memory_metadata.get("family_id") or ""),
                    "blueprint": blueprint,
                }
            )

    selected = []
    selected_families: dict[str, int] = {}
    while candidates and len(selected) < limit:
        def adjusted(item: dict[str, Any]) -> tuple[float, float, str]:
            family_count = selected_families.get(item["family_id"], 0)
            # Same-family recurrence remains dominant; the small penalty only
            # admits a closely related family when five near-duplicates exist.
            diversity_penalty = max(0, family_count - 2) * 3
            return (
                item["score"] - diversity_penalty,
                item["resolution"],
                str(item["blueprint"]["question_id"]),
            )

        best = max(candidates, key=adjusted)
        candidates.remove(best)
        selected.append(best["blueprint"])
        selected_families[best["family_id"]] = (
            selected_families.get(best["family_id"], 0) + 1
        )
    return selected


def select_compatible_blueprints(
    blueprints: list[dict[str, Any]],
    memory_questions: dict[str, Any],
    target_question: Any,
    *,
    cutoff: datetime,
    evidence: Any | None = None,
    limit: int = 3,
) -> list[dict[str, Any]]:
    """Select one shared, leakage-safe retrieval set for method comparison.

    Compatibility is intentionally stricter than the generic retriever.  Every
    selected memory must be resolved by the target cutoff and must match the
    exact recurring-event family and target metric.  The returned order is
    deterministic for fixed artifacts and evidence.
    """
    target = _finance_metadata(target_question)
    eligible: list[dict[str, Any]] = []
    for blueprint in blueprints:
        question_id = str(blueprint.get("question_id") or "")
        memory_question = memory_questions.get(question_id)
        if memory_question is None or not _is_temporally_eligible(
            memory_question, cutoff
        ):
            continue
        memory = _finance_metadata(memory_question)
        if (
            str(memory.get("family_id") or "")
            != str(target.get("family_id") or "")
            or str(memory.get("target_metric") or "")
            != str(target.get("target_metric") or "")
        ):
            continue
        eligible.append(blueprint)
    selected = select_relevant_blueprints(
        eligible,
        memory_questions,
        target_question,
        limit=limit,
        evidence=evidence,
    )
    if not selected:
        raise ValueError(
            f"no exact-family compatible memory for {target_question.id}"
        )
    return selected


def compile_hgf_search_memory(
    blueprints: list[dict[str, Any]],
    max_factors: int = 8,
) -> str:
    """Aggregate graph-derived factor hints without exposing causal edges."""
    factors: dict[str, dict[str, Any]] = {}
    for blueprint in blueprints:
        for item in blueprint.get("search_factors", []):
            factor = _generalize_factor(item.get("factor"))
            key_tokens = sorted(_tokens(factor))
            key = " ".join(key_tokens) or factor.lower()
            entry = factors.setdefault(
                key,
                {
                    "factor": factor,
                    "why_search": item.get("why_search"),
                    "preferred_source_types": [],
                    "observed_count": 0,
                },
            )
            entry["observed_count"] += 1
            for source_type in item.get("preferred_source_types", []):
                if source_type not in entry["preferred_source_types"]:
                    entry["preferred_source_types"].append(source_type)
    ranked = sorted(
        factors.values(),
        key=lambda item: (-item["observed_count"], str(item["factor"])),
    )[:max_factors]
    return json.dumps(
        {
            "view": "hgf_search_cards",
            "instructions": (
                "Use these as query-expansion and coverage hints only. The target "
                "question remains the search anchor; no historical outcome or edge "
                "is available in this view."
            ),
            "factors": ranked,
        },
        ensure_ascii=False,
        indent=2,
    )
