"""Pinned v1.0.0 values used by the finance manifest audit."""

from typing import Final

MANIFEST_SHA256: Final = (
    "8781718ed4bad08cabaecf820fe5b8ffe9b7d19986ad990b4f0fab359feda269"
)
MANIFEST_SIZE_BYTES: Final = 30743
SOURCE_URL: Final = (
    "https://github.com/cyzus/worldreasoner/releases/download/"
    + "v1.0.0/worldreasoner_public.db"
)
CANONICAL_SQL: Final = (
    "SELECT id, question_type, resolution_date, ground_truth FROM questions "
    + "WHERE lower(domain)='finance' AND graph_built=1 ORDER BY id"
)
LIMITATIONS: Final = (
    "Article bodies are absent from the public release; articles are metadata-only "
    + "and the public search index cannot be rebuilt.",
    "resolution_available_at is absent; resolution_date is a BOOTSTRAP PROXY only "
    + "and must not be treated as actual availability time.",
    "same/shared/derived/near-duplicate relation flags are absent; absence is "
    + "unknown/unavailable, not false.",
)
EXPECTED_MISSING_FIELDS: Final = {
    "causal_explanation": 0,
    "context": 14,
    "domain": 0,
    "estimated_start_time": 4,
    "graph_built_not_one": 0,
    "ground_truth": 0,
    "ground_truth_hash": 37,
    "outcome_event_ids_empty_or_null": 1,
    "question_text": 0,
    "question_type": 0,
    "related_article_ids_empty_or_null": 37,
    "related_event_ids_empty_or_null": 36,
    "resolution_date": 0,
    "target_event_id": 33,
}
EXPECTED_TABLES: Final = frozenset(
    {
        "articles",
        "causal_hypotheses",
        "event_outcome_impacts",
        "events",
        "forecast_events",
        "forecast_hypotheses",
        "forecasts",
        "questions",
    }
)
EXPECTED_ROW_COUNTS: Final = {
    "articles": 14364,
    "causal_hypotheses": 9858,
    "event_outcome_impacts": 9828,
    "events": 9149,
    "forecast_events": 41131,
    "forecast_hypotheses": 28141,
    "forecasts": 11566,
    "questions": 345,
}
REQUIRED_COLUMNS: Final = {
    "questions": frozenset(
        {
            "id",
            "question_text",
            "question_type",
            "domain",
            "source",
            "context",
            "resolution_date",
            "ground_truth",
            "resolution_criteria",
            "options",
            "quantity_unit",
            "outcome_event_ids",
            "graph_built",
        }
    ),
    "events": frozenset(
        {
            "id",
            "title",
            "description",
            "event_type",
            "domain",
            "tags",
            "occurred_date",
            "predicted_date",
            "article_ids",
            "extracted_for_question_id",
            "is_outcome",
            "outcome_scenario",
            "is_actual_outcome",
        }
    ),
    "causal_hypotheses": frozenset(
        {
            "id",
            "source_event_id",
            "target_event_id",
            "relation_type",
            "strength",
            "confidence",
            "time_lag_hours",
            "reasoning",
            "evidence_article_ids",
            "discovered_by_question_ids",
        }
    ),
    "event_outcome_impacts": frozenset(
        {
            "id",
            "event_id",
            "outcome_event_id",
            "question_id",
            "impact_direction",
            "impact_magnitude",
            "confidence",
            "reasoning",
            "evidence_article_ids",
            "causal_chain_hypothesis_ids",
        }
    ),
}
