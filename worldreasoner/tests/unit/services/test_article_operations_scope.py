"""Tests for target-question evidence isolation in forecast article search."""

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock

from src.domain.models import Article
from src.domain.models.domain import Domain
from src.services.article_operations_service import ArticleOperationsService


def _article(article_id: str, question_id: str) -> Article:
    return Article(
        id=article_id,
        title=f"Forecast evidence article {article_id}",
        content="Evidence content " * 20,
        source="test",
        published_date=datetime(2026, 3, 1, tzinfo=timezone.utc),
        domain=Domain.FINANCE,
        collected_for_question_id=question_id,
    )


def test_search_articles_excludes_other_questions_even_when_index_returns_them(test_db):
    target = _article("target-article", "target-question")
    contaminant = _article("other-article", "other-question")
    test_db.save(Article, target)
    test_db.save(Article, contaminant)

    search = AsyncMock()
    search.search.return_value = [contaminant.id, target.id]
    service = ArticleOperationsService(test_db, search)

    results = asyncio.run(
        service.search_articles(
            query="inflation",
            simulated_date=datetime(2026, 3, 10, tzinfo=timezone.utc),
            domain="finance",
            question_id="target-question",
        )
    )

    assert [article.id for article in results] == [target.id]


def test_search_articles_accepts_cutoff_safe_shared_provenance(test_db):
    shared = _article("shared-article", "owner-question")
    shared.metadata = {"related_question_ids": ["target-question"]}
    test_db.save(Article, shared)

    search = AsyncMock()
    search.search.return_value = [shared.id]
    service = ArticleOperationsService(test_db, search)

    results = asyncio.run(
        service.search_articles(
            query="inflation",
            simulated_date=datetime(2026, 3, 10, tzinfo=timezone.utc),
            domain="finance",
            question_id="target-question",
        )
    )

    assert [article.id for article in results] == [shared.id]
