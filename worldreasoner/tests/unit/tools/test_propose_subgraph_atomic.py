"""Regression tests for atomic and repeat-safe subgraph creation."""

import json
from datetime import datetime, timezone

from src.core.alias_registry import AliasRegistry
from src.domain.models import Article, CausalHypothesis, Event
from src.domain.models.domain import Domain
from src.tools.reasoning.causal_reasoner import CausalReasonerTool
from src.tools.reasoning.event_identifier import EventIdentifierTool
from src.tools.reasoning.propose_subgraph import ProposeSubgraphTool


def _article(test_db) -> Article:
    article = Article(
        id="article-1",
        title="Apple reports quarterly financial results",
        content="Evidence content about Apple quarterly results. " * 10,
        url="https://example.com/apple-results",
        source="Example News",
        published_date=datetime(2026, 4, 30, tzinfo=timezone.utc),
        domain=Domain.FINANCE,
        collected_for_question_id="question-1",
    )
    test_db.save(Article, article)
    return article


def _tool(test_db):
    registry = AliasRegistry()
    event_tool = EventIdentifierTool(
        db_path=str(test_db.db_path),
        question_id="question-1",
        alias_registry=registry,
    )
    reasoner_tool = CausalReasonerTool(
        db_path=str(test_db.db_path),
        question_id="question-1",
        alias_registry=registry,
    )
    return ProposeSubgraphTool(
        event_identifier_tool=event_tool,
        causal_reasoner_tool=reasoner_tool,
        alias_registry=registry,
        db_path=str(test_db.db_path),
        question_id="question-1",
    ), registry


def _payload(article_id: str, relation: str = "causes"):
    return {
        "events": [
            {
                "alias": "E1:ServicesGrowth",
                "title": "Apple reports record services growth",
                "description": (
                    "Apple reported record growth in its high-margin services segment."
                ),
                "domain": "finance",
                "occurred_date": "2026-03-27T12:00:00Z",
                "source_article_ids": [article_id],
            },
            {
                "alias": "E2:MarginExpansion",
                "title": "Apple gross margin expands",
                "description": (
                    "Apple reported a higher consolidated gross margin for the quarter."
                ),
                "domain": "finance",
                "occurred_date": "2026-03-28T12:00:00Z",
                "article_ids": [article_id],
            },
        ],
        "edges": [
            {
                "source": "E1:ServicesGrowth",
                "target": "E2:MarginExpansion",
                "relation": relation,
                "strength": 0.7,
                "confidence": 0.8,
                "reasoning": "A higher services mix supports consolidated margin.",
                "evidence_article_ids": [article_id],
            }
        ],
    }


def test_source_article_alias_is_normalized_and_edge_evidence_is_preserved(test_db):
    article = _article(test_db)
    tool, _ = _tool(test_db)

    result = tool.forward(json.dumps(_payload(article.id)))

    assert result.status == "success", result.failed_items
    assert result.events_created == 2
    assert result.edges_created == 1
    events = test_db.get_many(Event)
    hypotheses = test_db.get_many(CausalHypothesis)
    assert len(events) == 2
    assert all(event.article_ids == [article.id] for event in events)
    assert hypotheses[0].evidence_article_ids == [article.id]


def test_failed_edge_rolls_back_events_and_aliases(test_db):
    article = _article(test_db)
    tool, registry = _tool(test_db)

    result = tool.forward(json.dumps(_payload(article.id, relation="not-a-relation")))

    assert result.status == "error"
    assert result.events_created == 0
    assert result.edges_created == 0
    assert test_db.get_many(Event) == []
    assert test_db.get_many(CausalHypothesis) == []
    assert registry.list_aliases() == {}


def test_identical_invalid_payload_is_reported_as_repeated_without_writes(test_db):
    tool, _ = _tool(test_db)
    payload = _payload("missing-article")

    first = tool.forward(json.dumps(payload))
    second = tool.forward(json.dumps(payload))

    assert first.status == "error"
    assert second.status == "error"
    assert any(item["type"] == "repeated_failure" for item in second.failed_items)
    assert test_db.get_many(Event) == []
    assert test_db.get_many(CausalHypothesis) == []


def test_article_before_event_is_rejected_before_any_write(test_db):
    article = _article(test_db)
    tool, _ = _tool(test_db)
    payload = _payload(article.id)
    payload["events"][0]["occurred_date"] = "2026-05-01T12:00:00Z"

    result = tool.forward(json.dumps(payload))

    assert result.status == "error"
    assert result.events_created == 0
    assert any(
        item.get("field") == "occurred_date"
        and "predates" in item.get("reason", "")
        for item in result.failed_items
    )
    assert test_db.get_many(Event) == []
    assert test_db.get_many(CausalHypothesis) == []


def test_extra_closing_brace_after_valid_payload_is_tolerated(test_db):
    article = _article(test_db)
    tool, _ = _tool(test_db)

    result = tool.forward(json.dumps(_payload(article.id)) + "}")

    assert result.status == "success"
    assert result.events_created == 2
    assert result.edges_created == 1


def test_explanatory_prefix_around_valid_payload_is_tolerated(test_db):
    article = _article(test_db)
    tool, _ = _tool(test_db)
    wrapped = "horrible string with schema: " + json.dumps(_payload(article.id))

    result = tool.forward(wrapped)

    assert result.status == "success", result.failed_items
    assert result.events_created == 2
    assert result.edges_created == 1


def test_non_json_prose_is_still_rejected(test_db):
    tool, _ = _tool(test_db)

    result = tool.forward("This is not a JSON subgraph.")

    assert result.status == "error"
    assert result.failed_items[0]["type"] == "parse_error"
