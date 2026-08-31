"""Tests for the bounded search plan and deterministic coverage ledger."""

from datetime import datetime, timedelta, timezone

from src.domain.models import Article, Domain, Question
from src.tools.search_orchestration import SearchCoverageTracker, SearchCoverageTool
from tests.conftest import create_test_question


def _plan():
    return [
        {
            "name": "official release",
            "role": "official",
            "queries": ["Intel official earnings release"],
        },
        {
            "name": "countervailing demand",
            "role": "countervailing",
            "queries": ["Intel weak demand risk"],
        },
    ]


def test_tracker_rejects_duplicate_calls_and_enforces_query_budget():
    tracker = SearchCoverageTracker(
        db_path=None,
        question_id="q",
        min_articles=10,
        max_queries=2,
    )
    tracker.register_plan(_plan())

    assert tracker.allow_search("Intel earnings", 1, "google_news")[0]
    tracker.record_search(
        query="Intel earnings",
        page=1,
        provider="google_news",
        factor="official release",
    )
    assert not tracker.allow_search("Intel earnings", 1, "google_news")[0]
    assert tracker.allow_search("Intel earnings", 1, "ddgs")[0]
    tracker.record_search(
        query="Intel earnings",
        page=1,
        provider="ddgs",
        factor="official release",
    )
    allowed, reason = tracker.allow_search("Intel risk", 1, "ddgs")
    assert allowed is False
    assert "budget exhausted" in reason


def test_tracker_reserves_parallel_query_budget_before_completion():
    tracker = SearchCoverageTracker(
        db_path=None,
        question_id="q",
        min_articles=10,
        max_queries=1,
    )
    tracker.register_plan(_plan())

    assert tracker.allow_search("Intel earnings", 1, "ddgs")[0]
    allowed, reason = tracker.allow_search("Intel demand", 1, "ddgs")
    assert allowed is False
    assert "budget exhausted" in reason


def test_tracker_can_extend_budget_for_a_retry_without_forgetting_attempts():
    tracker = SearchCoverageTracker(
        db_path=None,
        question_id="q",
        min_articles=10,
        max_queries=1,
    )
    tracker.register_plan(_plan())
    tracker.record_search(
        query="Intel earnings",
        page=1,
        provider="ddgs",
        factor="official release",
    )

    assert tracker.allow_search("Intel demand", 1, "ddgs")[0] is False

    tracker.extend_query_budget(1)

    assert tracker.allow_search("Intel demand", 1, "ddgs")[0] is True
    assert tracker.allow_search("Intel earnings", 1, "ddgs")[0] is False
    snapshot = tracker.snapshot()
    assert snapshot["query_budget"] == 2
    assert snapshot["queries_used"] == 1


def test_tracker_credits_articles_collected_after_the_search(test_db):
    cutoff = datetime(2026, 3, 28, tzinfo=timezone.utc)
    question = create_test_question(id="q", resolution_date=cutoff)
    test_db.save(Question, question)
    tracker = SearchCoverageTracker(
        db_path=str(test_db.db_path),
        question_id=question.id,
        cutoff=cutoff,
        min_articles=1,
    )
    tracker.register_plan(_plan())
    tracker.record_search(
        query="Intel official earnings release",
        factor="official release",
        provider="ddgs",
        result_urls=["https://example.com/intel-release"],
        raw_result_count=1,
    )

    test_db.save(
        Article,
        Article(
            id="article-1",
            title="Intel releases earnings",
            url="https://example.com/intel-release",
            content="Verified evidence. " * 20,
            source="example.com",
            published_date=cutoff - timedelta(days=1),
            domain=Domain.FINANCE,
            collected_for_question_id=question.id,
        ),
    )

    snapshot = tracker.snapshot()
    official = next(
        factor for factor in snapshot["factors"] if factor["name"] == "official release"
    )
    assert snapshot["article_count"] == 1
    assert snapshot["coverage_target_met"] is True
    assert official["article_ids"] == ["article-1"]
    assert snapshot["recommended_queries"] == [
        {
            "factor": "countervailing demand",
            "query": "Intel weak demand risk",
        }
    ]
    assert "factor gaps remain" in snapshot["message"]
    assert "No verified evidence for factor: countervailing demand" in snapshot["gaps"]
    allowed, reason = tracker.allow_search("Intel weak demand risk")
    assert allowed is False
    assert "target already met" in reason
    assert tracker.allow_search(
        "Intel weak demand risk",
        factor="countervailing demand",
    )[0]
    arbitrary_allowed, arbitrary_reason = tracker.allow_search(
        "Intel unrelated arbitrary query",
        factor="countervailing demand",
    )
    assert arbitrary_allowed is False
    assert "registered query" in arbitrary_reason


def test_over_target_factor_gap_accepts_transport_side_cutoff_only(test_db):
    cutoff = datetime(2026, 3, 28, tzinfo=timezone.utc)
    question = create_test_question(id="q", resolution_date=cutoff)
    test_db.save(Question, question)
    test_db.save(
        Article,
        Article(
            id="article-1",
            title="Existing evidence",
            url="https://example.com/existing",
            content="Verified evidence. " * 20,
            source="example.com",
            published_date=cutoff - timedelta(days=1),
            domain=Domain.FINANCE,
            collected_for_question_id=question.id,
        ),
    )
    tracker = SearchCoverageTracker(
        db_path=str(test_db.db_path),
        question_id=question.id,
        cutoff=cutoff,
        min_articles=1,
    )
    tracker.register_plan(_plan())

    assert tracker.allow_search(
        "Intel weak demand risk before:2026-03-28",
        factor="countervailing demand",
    )[0]


def test_question_only_search_stops_at_global_article_target(test_db):
    cutoff = datetime(2026, 3, 28, tzinfo=timezone.utc)
    question = create_test_question(id="q", resolution_date=cutoff)
    test_db.save(Question, question)
    test_db.save(
        Article,
        Article(
            id="article-1",
            title="Existing evidence",
            url="https://example.com/existing",
            content="Verified evidence. " * 20,
            source="example.com",
            published_date=cutoff - timedelta(days=1),
            domain=Domain.FINANCE,
            collected_for_question_id=question.id,
        ),
    )
    tracker = SearchCoverageTracker(
        db_path=str(test_db.db_path),
        question_id=question.id,
        cutoff=cutoff,
        min_articles=1,
        allow_over_target_factor_search=False,
    )
    tracker.register_plan(_plan())

    allowed, reason = tracker.allow_search(
        "Intel weak demand risk",
        factor="countervailing demand",
    )
    assert allowed is False
    assert "stop searching" in reason
    snapshot = tracker.snapshot()
    assert snapshot["recommended_queries"] == []
    assert "Stop searching" in snapshot["message"]


def test_transport_date_bound_cannot_bypass_duplicate_query_check():
    tracker = SearchCoverageTracker(
        db_path=None,
        question_id="q",
        min_articles=10,
        max_queries=3,
    )
    tracker.register_plan(_plan())
    tracker.record_search(
        query="Intel weak demand risk",
        factor="countervailing demand",
    )

    allowed, reason = tracker.allow_search(
        "Intel weak demand risk before:2026-03-28",
        factor="countervailing demand",
    )

    assert allowed is False
    assert "Duplicate" in reason


def test_coverage_tool_registers_and_reports_plan():
    tracker = SearchCoverageTracker(db_path=None, question_id="q")
    result = SearchCoverageTool(tracker).forward("register_plan", _plan())

    assert result.plan_registered is True
    assert result.queries_used == 0
    assert len(result.factors) == 2


def test_retry_plan_keeps_original_factor_identities():
    tracker = SearchCoverageTracker(db_path=None, question_id="q")
    tracker.register_plan(_plan())

    tracker.register_plan(
        [
            {
                "name": "newly invented retry factor",
                "role": "supporting",
                "queries": ["new retry query"],
            },
            {
                "name": "official release",
                "role": "official",
                "queries": ["Intel SEC filing"],
            },
        ]
    )

    snapshot = tracker.snapshot()
    assert [factor["name"] for factor in snapshot["factors"]] == [
        "official release",
        "countervailing demand",
    ]
