"""Tests for the leakage-safe Searcher to Forecaster evidence contract."""

from datetime import datetime, timedelta, timezone

from forecaster.data_pipeline.evidence_handoff import build_search_evidence_handoff
from src.domain.models import Article
from src.domain.models.domain import Domain


def _article(article_id: str, published_date: datetime) -> Article:
    return Article(
        id=article_id,
        title=f"Inflation evidence {article_id}",
        content="Verified evidence content. " * 20,
        url=f"https://example.com/{article_id}",
        source="Example",
        published_date=published_date,
        domain=Domain.FINANCE,
        collected_for_question_id="question-1",
    )


def test_handoff_keeps_only_db_verified_pre_cutoff_articles():
    cutoff = datetime(2026, 3, 20, tzinfo=timezone.utc)
    valid = _article("article-valid", cutoff - timedelta(days=1))
    future = _article("article-future", cutoff + timedelta(days=1))
    report = {
        "evidence": [
            {
                "factor": "energy prices",
                "direction": "upward pressure",
                "article_ids": [valid.id, future.id, "invented-id"],
                "dates": ["untrusted"],
            }
        ],
        "remaining_gaps": [],
        "article_count": 999,
    }

    handoff = build_search_evidence_handoff(report, [valid, future], cutoff)

    assert handoff.article_count == 1
    assert handoff.evidence[0].article_ids == [valid.id]
    assert handoff.evidence[0].dates == [valid.published_date.isoformat()]
    assert handoff.evidence[0].direction == "up"
    assert "invented-id" not in handoff.model_dump_json()


def test_handoff_redacts_answer_options_and_discards_searcher_forecast_prose():
    cutoff = datetime(2026, 3, 20, tzinfo=timezone.utc)
    article = _article("article-1", cutoff - timedelta(days=1))
    report = """```json
    {"factor": "prediction: 2.5% to <3.0%", "direction": "mixed",
     "article_ids": ["article-1"], "evidence_summary": "The answer is 2.5% to <3.0%"}
    ```"""

    handoff = build_search_evidence_handoff(
        report,
        [article],
        cutoff,
        prohibited_phrases=["2.5% to <3.0%"],
    )

    serialized = handoff.model_dump_json()
    assert "2.5% to <3.0%" not in serialized
    assert "The answer is" not in serialized
    assert article.title in handoff.evidence[0].evidence_summary


def test_unstructured_report_falls_back_to_all_verified_articles():
    cutoff = datetime(2026, 3, 20, tzinfo=timezone.utc)
    article = _article("article-1", cutoff - timedelta(days=1))

    handoff = build_search_evidence_handoff(
        "Searcher returned malformed prose.",
        [article],
        cutoff,
    )

    assert handoff.article_count == 1
    assert handoff.evidence[0].factor == "additional_collected_evidence"
    assert handoff.evidence[0].article_ids == [article.id]
