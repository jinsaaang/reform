"""Publication-date resolution regression tests for ArticleCollectorTool."""

from datetime import datetime, timezone

import pytest

from src.domain.models import Article, Question
from tests.conftest import create_test_question
from src.tools.base.output_models import WebFetchOutput
from src.tools.collectors.article_collector import ArticleCollectorTool


class _FakeWebVisitor:
    def __init__(self, output: WebFetchOutput):
        self.output = output

    def forward(self, url: str, timeout: int = 15) -> WebFetchOutput:
        return self.output


def _collector(test_db, web_output: WebFetchOutput) -> ArticleCollectorTool:
    tool = ArticleCollectorTool(db_path=str(test_db.db_path), question_id="question-1")
    tool.web_visitor = _FakeWebVisitor(web_output)
    return tool


def test_page_metadata_replaces_incorrect_agent_date_hint(test_db):
    output = WebFetchOutput(
        url="https://example.com/apple-results",
        title="Apple results",
        content="Quarterly results and financial details. " * 10,
        metadata={"datePublished": "2026-04-30T13:30:00-07:00"},
        success=True,
    )
    tool = _collector(test_db, output)

    result = tool.forward(
        url=output.url,
        title="Apple reports quarterly financial results",
        source="Apple",
        published_date="2026-03-28T00:00:00Z",
        domain="finance",
    )

    stored = test_db.get(Article, result.id)
    assert result.status == "created"
    assert stored.published_date.date().isoformat() == "2026-04-30"
    assert result.published_date_source == "metadata:datePublished"
    assert result.warnings and "replaced" in result.warnings[0]


@pytest.mark.parametrize(
    ("metadata", "expected_source"),
    [
        ({"DC.date": "2023-12-04"}, "metadata:DC.date"),
        (
            {"citation_publication_date": "2023/08/16"},
            "metadata:citation_publication_date",
        ),
        ({"prism.publicationDate": "2023/08/16"}, "metadata:prism.publicationDate"),
    ],
)
def test_standard_publisher_metadata_dates_are_accepted(
    test_db,
    metadata,
    expected_source,
):
    output = WebFetchOutput(
        url="https://example.com/research-note",
        title="Research note",
        content="A substantive fixed-income research note. " * 20,
        metadata=metadata,
        success=True,
    )
    tool = _collector(test_db, output)

    result = tool.forward(
        url=output.url,
        title=output.title,
        source="Publisher",
        domain="finance",
    )

    assert result.status == "created"
    assert result.published_date_source == expected_source


def test_content_date_is_used_when_search_has_no_date(test_db):
    output = WebFetchOutput(
        url="https://example.com/apple-results",
        title="Apple results",
        content=(
            "PRESS RELEASE 30 April 2026\n"
            "Apple announced quarterly financial results. " * 10
        ),
        metadata={},
        success=True,
    )
    tool = _collector(test_db, output)

    result = tool.forward(
        url=output.url,
        title="Apple reports quarterly financial results",
        source="Apple",
        published_date=None,
        domain="finance",
    )

    assert result.id != "error"
    assert result.published_date.startswith("2026-04-30")
    assert result.published_date_source == "content"


def test_reporting_period_date_is_not_treated_as_publication_date(test_db):
    output = WebFetchOutput(
        url="https://example.com/apple-margin-history",
        title="Apple margin history",
        content=(
            "Quarter ended December 27, 2025. "
            "The company reported its historical gross margin. " * 10
        ),
        metadata={},
        success=True,
    )
    tool = _collector(test_db, output)

    result = tool.forward(
        url=output.url,
        title=output.title,
        source="Example",
        published_date=None,
        domain="finance",
    )

    assert result.id == "error"
    assert "Could not determine publication date" in result.status
    assert test_db.get_many(Article) == []


def test_filing_date_wins_over_reporting_period_date(test_db):
    output = WebFetchOutput(
        url="https://example.com/apple-quarterly-report",
        title="Apple quarterly report",
        content=(
            "10-Q Period: Q2 FY2026\n"
            "Apple Quarterly Report for Q2 Ended March 28, 2026\n"
            "Filed May 1, 2026For Securities: AAPL\n"
            "Financial details and gross margin discussion. " * 10
        ),
        metadata={},
        success=True,
    )
    tool = _collector(test_db, output)

    result = tool.forward(
        url=output.url,
        title=output.title,
        source="Example",
        published_date=None,
        domain="finance",
    )

    assert result.published_date.startswith("2026-05-01")
    assert result.published_date_source == "content"


def test_verified_sec_filing_hint_wins_over_period_date_in_url(test_db):
    output = WebFetchOutput(
        url=(
            "https://www.sec.gov/Archives/edgar/data/50863/"
            "000005086326000011/intc-20251227.htm"
        ),
        title="Intel annual filing",
        content="Intel annual report for the year ended December 27, 2025. " * 10,
        metadata={},
        success=True,
    )
    tool = _collector(test_db, output)

    result = tool.forward(
        url=output.url,
        title=output.title,
        source="SEC EDGAR",
        published_date="2026-01-23",
        domain="finance",
    )

    assert result.published_date.startswith("2026-01-23")
    assert result.published_date_source == "verified_filing_hint"


def test_leading_business_wire_timestamp_is_publication_date(test_db):
    output = WebFetchOutput(
        url="https://example.com/business-wire-release",
        title="Apple reports second quarter results",
        content=(
            "Apr 30, 2026 4:30 PM Eastern Daylight Time\n"
            "Apple reports second quarter results\n"
            "Fiscal quarter ended March 28, 2026. " * 10
        ),
        metadata={},
        success=True,
    )
    tool = _collector(test_db, output)

    result = tool.forward(
        url=output.url,
        title=output.title,
        source="Business Wire",
        published_date=None,
        domain="finance",
    )

    assert result.published_date.startswith("2026-04-30")
    assert result.published_date_source == "content"


def test_article_at_or_after_question_cutoff_is_not_stored(test_db):
    test_db.save(
        Question,
        create_test_question(
            id="question-1",
            estimated_start_time=datetime(2026, 3, 28, tzinfo=timezone.utc),
            resolution_date=datetime(2026, 5, 1, 23, 59, 59, tzinfo=timezone.utc),
        ),
    )
    output = WebFetchOutput(
        url="https://example.com/late-analysis",
        title="Late analysis",
        content="Published June 18, 2026\nLate analysis content. " * 10,
        metadata={},
        success=True,
    )
    tool = _collector(test_db, output)

    result = tool.forward(
        url=output.url,
        title=output.title,
        source="Example",
        published_date=None,
        domain="finance",
    )

    assert result.id == "error"
    assert "at or after the evidence cutoff" in result.status
    assert test_db.get_many(Article) == []


def test_article_before_estimated_start_is_valid_pre_cutoff_evidence(test_db):
    test_db.save(
        Question,
        create_test_question(
            id="question-1",
            estimated_start_time=datetime(2026, 3, 28, tzinfo=timezone.utc),
            resolution_date=datetime(2026, 4, 19, tzinfo=timezone.utc),
        ),
    )
    output = WebFetchOutput(
        url="https://example.com/intel-2025-10-k",
        title="Intel 2025 annual report",
        content="Intel annual filing and historical financial details. " * 10,
        metadata={"datePublished": "2026-01-23T00:00:00Z"},
        success=True,
    )
    tool = _collector(test_db, output)

    result = tool.forward(
        url=output.url,
        title=output.title,
        source="SEC EDGAR",
        published_date="2026-01-23",
        domain="finance",
    )

    assert result.status == "created"
    assert result.published_date.startswith("2026-01-23")
    assert result.warnings is None
    assert len(test_db.get_many(Article)) == 1


def test_unresolved_date_fails_instead_of_using_current_or_question_date(test_db):
    output = WebFetchOutput(
        url="https://example.com/no-date",
        title="Undated article",
        content="An article with no publication date in its content. " * 10,
        metadata={},
        success=True,
    )
    tool = _collector(test_db, output)

    result = tool.forward(
        url=output.url,
        title="An article without a publication date",
        source="Example",
        published_date=None,
        domain="finance",
    )

    assert result.id == "error"
    assert "Could not determine publication date" in result.status
    assert test_db.get_many(Article) == []


def test_existing_shared_url_is_rejected_against_each_question_cutoff(test_db):
    early_question = create_test_question(
        id="early-question",
        resolution_date=datetime(2026, 3, 1, tzinfo=timezone.utc),
    )
    late_question = create_test_question(
        id="late-question",
        resolution_date=datetime(2026, 5, 1, tzinfo=timezone.utc),
    )
    test_db.save(Question, early_question)
    test_db.save(Question, late_question)
    existing = Article(
        id="shared-article",
        title="April company filing",
        content="Company filing content. " * 20,
        url="https://example.com/shared-filing",
        source="Example",
        published_date=datetime(2026, 4, 1, tzinfo=timezone.utc),
        domain="finance",
        collected_for_question_id=late_question.id,
        metadata={"related_question_ids": [late_question.id]},
    )
    test_db.save(Article, existing)
    collector = ArticleCollectorTool(
        db_path=str(test_db.db_path), question_id=early_question.id
    )

    result = collector.forward(
        url=existing.url,
        title=existing.title,
        source=existing.source,
        domain="finance",
    )

    assert result.id == "error"
    assert "existing article publication date" in result.status
    saved = test_db.get(Article, existing.id)
    assert early_question.id not in saved.metadata["related_question_ids"]
