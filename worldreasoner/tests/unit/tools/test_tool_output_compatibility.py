"""Regression tests for CodeAgent-facing tool output compatibility."""

from datetime import datetime, timezone

from src.core.alias_registry import AliasRegistry
from src.domain.models import Article, Question
from src.domain.models.domain import Domain
from src.tools.base.output_models import ArticleRetrievalOutput
from src.tools.generators.question_articles import QuestionArticlesTool
from src.tools.inspectors.article_retrieval import ArticleRetrievalTool
from tests.conftest import create_test_question


def test_output_model_supports_attribute_mapping_and_membership_access():
    result = ArticleRetrievalOutput(
        id="article-1",
        title="A sufficiently descriptive article title",
        url="https://example.com/article",
        content="full article content",
    )

    assert result.content == "full article content"
    assert result["content"] == "full article content"
    assert result.get("content") == "full article content"
    assert "content" in result
    assert "missing" not in result


def test_question_articles_preserves_alias_and_url(test_db):
    question_id = "question-1"
    test_db.save(
        Question,
        create_test_question(
            id=question_id,
            resolution_date=datetime(2026, 5, 1, tzinfo=timezone.utc),
        ),
    )
    article = Article(
        id="article-1",
        title="Apple reports quarterly financial results",
        content="Evidence content " * 20,
        url="https://example.com/apple-results",
        source="Example News",
        published_date=datetime(2026, 4, 30, tzinfo=timezone.utc),
        domain=Domain.FINANCE,
        collected_for_question_id=question_id,
    )
    test_db.save(Article, article)

    registry = AliasRegistry()
    tool = QuestionArticlesTool(
        db_path=str(test_db.db_path),
        question_id=question_id,
        alias_registry=registry,
    )

    result = tool.forward()
    repeated = tool.forward()

    assert result.total == 1
    assert result.articles[0].alias == "A1:AppleReportsQuarterly"
    assert repeated.articles[0].alias == result.articles[0].alias
    assert result.articles[0].url == article.url
    assert registry.resolve(result.articles[0].alias) == article.id
    assert registry.list_aliases() == {
        "A1:AppleReportsQuarterly": article.id
    }


def test_alias_registry_restores_article_counter_after_clear():
    registry = AliasRegistry()
    registry.register("A7:ExistingArticle", "article-7")
    registry.clear()
    registry.register("A3:RestoredArticle", "article-3")

    assert (
        registry.generate_article_alias("A new article", "article-4")
        == "A4:ANewArticle"
    )


def test_article_retrieval_bounds_agent_context(test_db):
    content = ("introductory text " * 1_000) + "gross margin reached 49.3 percent"
    article = Article(
        id="article-long",
        title="Apple publishes a detailed earnings report",
        content=content,
        url="https://example.com/long-report",
        source="Example News",
        published_date=datetime(2026, 4, 30, tzinfo=timezone.utc),
        domain=Domain.FINANCE,
    )
    test_db.save(Article, article)
    tool = ArticleRetrievalTool(db_path=str(test_db.db_path))

    result = tool.forward(
        article_id=article.id,
        query="gross margin",
        max_chars=2_000,
    )

    assert result.content_truncated is True
    assert result.original_char_count == len(content)
    assert len(result.content) <= 2_100
    assert "gross margin" in result.content
