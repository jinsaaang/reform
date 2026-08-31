"""Test auto-collection feature in WebSearchTool."""

import logging
import time
import pytest
from datetime import datetime
from unittest.mock import Mock
from src.tools.collectors.web_search import WebSearchTool
from src.tools.collectors.web_search import (
    _max_fetch_workers,
    _max_google_news_resolve_workers,
)
from src.tools.base.output_models import WebSearchOutput
from src.tools.search_orchestration import SearchCoverageTracker


@pytest.fixture
def mock_db():
    """Mock database for testing."""
    return Mock()


@pytest.fixture
def mock_question():
    """Mock question with resolution date."""
    question = Mock()
    question.id = "test_question_123"
    question.resolution_date = datetime(2026, 1, 1)  # Future date
    return question


@pytest.fixture
def sample_search_results():
    """Real-world sample search results matching the user's output."""
    return [
        {
            "title": "Wildfires, floods and extreme heat: These are the biggest weather stories of 2025",
            "url": "https://www.msn.com/en-us/weather/topstories/these-are-the-biggest-weather-stories-of-2025/ar-AA1Tm5zH",
            "content": "Devastating wildfire, flooding and extreme heat events took place over the past year, several resulting in mass fatalities.",
            "engines": ["startpage news", "bing news"],
            "publishedDate": "2025-12-31T15:49:00",  # Has date
        },
        {
            "title": "Five Things to Know About Climate Change in 2025",
            "url": "https://www.climatecentral.org/climate-matters/five-things-to-know-about-climate-change-in-2025",
            "content": "In 2025, carbon pollution made 89% of record high daily temperatures set across 247 major U.S. cities more likely",
            "engines": ["bing news"],
            "publishedDate": None,  # No date
        },
        {
            "title": "Oceans are supercharging hurricanes past Category 5 | ScienceDaily",
            "url": "https://sciencedaily.com/releases/2025/12/251225080725.htm",
            "content": "Deep ocean hot spots packed with heat are making the strongest hurricanes and typhoons more likely",
            "engines": ["brave.news"],
            "publishedDate": None,  # No date
        },
        {
            "title": "2025 was so hot it pushed Earth past critical climate change mark",
            "url": "https://www.cbsnews.com/news/climate-change-2025-critical-mark-eclipsed/",
            "content": "2025 was the third hottest year on record and pushed Earth past a critical climate change mark",
            "engines": ["brave.news"],
            "publishedDate": None,  # No date
        },
    ]


def test_auto_collect_with_missing_dates(
    mock_db, mock_question, sample_search_results, caplog
):
    """Missing search dates are delegated to ArticleCollector page extraction."""
    caplog.set_level(logging.DEBUG)

    mock_db.get.return_value = mock_question
    mock_collector_instance = Mock()

    tool = WebSearchTool(
        db_path=None,
        collector=None,
        question_id="test_question_123",
        auto_collect_enabled=False,
        max_auto_collect=5,
        domain="general",
    )

    tool.db = mock_db
    tool.question_id = mock_question.id
    tool.question = mock_question
    tool.auto_collect_enabled = True
    tool.question_resolution_date = mock_question.resolution_date
    tool.article_collector = mock_collector_instance

    summary = tool._auto_collect_articles(sample_search_results)

    print("\n" + "=" * 60)
    print("Auto-collection summary:")
    print(summary)
    print("=" * 60)

    if caplog.text:
        print("\nCaptured logs:")
        print(caplog.text)

    assert "auto-collected 4 article" in summary.lower()
    assert mock_collector_instance.forward.call_count == 4
    calls_by_url = {
        call.kwargs["url"]: call.kwargs
        for call in mock_collector_instance.forward.call_args_list
    }
    first_url = sample_search_results[0]["url"]
    assert calls_by_url[first_url]["title"] == sample_search_results[0]["title"]
    assert calls_by_url[first_url]["source"] == "msn.com"


def test_search_appends_question_upper_cutoff_without_lower_bound():
    tool = WebSearchTool()
    tool.question_resolution_date = datetime(2025, 8, 1)

    effective = tool._with_upper_cutoff("NVIDIA earnings outlook")

    assert effective == "NVIDIA earnings outlook before:2025-08-01"
    assert "after:" not in effective


def test_search_preserves_explicit_before_operator():
    tool = WebSearchTool()
    tool.question_resolution_date = datetime(2025, 8, 1)

    assert tool._with_upper_cutoff("CPI before:2025-07-01") == (
        "CPI before:2025-07-01"
    )


def test_hindsight_upper_only_mode_removes_caller_lower_bound():
    tool = WebSearchTool(enforce_upper_only_dates=True)
    tool.question_resolution_date = datetime(2025, 8, 1)

    assert tool._with_upper_cutoff(
        "Walmart Q4 2022 after:2023-01-31"
    ) == "Walmart Q4 2022 before:2025-08-01"


def test_search_reports_and_tracks_effective_bounded_query(monkeypatch):
    tracker = SearchCoverageTracker(
        db_path=None, question_id="q", min_articles=1, max_queries=2
    )
    tool = WebSearchTool(coverage_tracker=tracker)
    tool.question_resolution_date = datetime(2025, 8, 1)
    monkeypatch.setattr(tool, "_get_structured_results", lambda *args, **kwargs: [])

    result = tool.forward("AI infrastructure demand")

    assert result.query == "AI infrastructure demand before:2025-08-01"
    assert tracker.snapshot()["queries_used"] == 1
    assert not tracker.allow_search(result.query)[0]


def test_auto_cascades_when_google_results_store_no_usable_articles(monkeypatch):
    tool = WebSearchTool(domain="finance")
    tool.auto_collect_enabled = True
    tool.question_resolution_date = datetime(2025, 8, 1)
    tool.article_collector = Mock()
    tool.article_collector.forward.return_value = Mock(id="article-ddgs")
    google_result = {
        "title": "Unrelated cooking recipes",
        "url": "https://example.com/recipes",
        "content": "Recipes and kitchen tips",
        "engines": ["google-news-rss"],
        "publishedDate": "2025-07-01T00:00:00",
    }
    ddgs_result = {
        "title": "NASDAQ performance improves",
        "url": "https://example.com/nasdaq",
        "content": "NASDAQ performance improved as technology shares advanced",
        "engines": ["ddgs:bing"],
        "publishedDate": "2025-07-02T00:00:00",
    }
    monkeypatch.setattr(
        tool,
        "_get_structured_results",
        lambda *args, **kwargs: [google_result],
    )
    monkeypatch.setattr(
        tool,
        "_get_fallback_structured_results",
        lambda *args, **kwargs: [ddgs_result]
        if kwargs.get("provider") == "ddgs"
        else [],
    )

    result = tool.forward("NASDAQ performance")

    assert result.count == 2
    assert tool.article_collector.forward.call_count == 1
    assert tool.article_collector.forward.call_args.kwargs["url"] == (
        "https://example.com/nasdaq"
    )
    assert "DDGS usable-result fallback" in result.collection_summary


def test_auto_does_not_cascade_while_timed_out_fetches_unwind(monkeypatch):
    tool = WebSearchTool(domain="finance")
    tool.auto_collect_enabled = True
    tool.question_resolution_date = datetime(2025, 8, 1)
    tool.article_collector = Mock()
    google_result = {
        "title": "NASDAQ performance report",
        "url": "https://example.com/slow",
        "content": "NASDAQ performance and technology shares",
        "engines": ["google-news-rss"],
        "publishedDate": "2025-07-01T00:00:00",
    }
    fallback = Mock(return_value=[])
    monkeypatch.setattr(
        tool,
        "_get_structured_results",
        lambda *args, **kwargs: [google_result],
    )
    monkeypatch.setattr(
        tool,
        "_auto_collect_articles",
        lambda *args, **kwargs: (
            "\n---\n**Auto-collected 0 article(s):**\n"
            "(Skipped 1: 1 timeout.)"
        ),
    )
    monkeypatch.setattr(tool, "_get_fallback_structured_results", fallback)

    result = tool.forward("NASDAQ performance")

    assert result.count == 1
    fallback.assert_not_called()


def test_auto_cascades_when_google_and_ddgs_are_future_dated(monkeypatch):
    tool = WebSearchTool()
    tool.question_resolution_date = datetime(2025, 8, 1)
    future = {
        "title": "Future result",
        "link": "https://example.com/future",
        "publishedDate": "2026-01-01T00:00:00Z",
    }
    unknown_date = {
        "title": "Undated result",
        "link": "https://example.com/undated",
        "description": "No reliable publication date",
    }
    monkeypatch.setattr(tool, "_search_google_news", lambda *_: [future])
    monkeypatch.setattr(tool, "_search_ddgs", lambda *_: [future])
    monkeypatch.setattr(tool, "_search_gdelt", lambda *_: [unknown_date])

    results = tool._get_fallback_structured_results("AI demand", provider="auto")

    assert [result["url"] for result in results] == ["https://example.com/undated"]
    assert results[0]["engines"] == ["gdelt"]


def test_explicit_provider_keeps_unknown_dates_and_drops_known_future_dates(monkeypatch):
    tool = WebSearchTool()
    tool.question_resolution_date = datetime(2025, 8, 1)
    monkeypatch.setattr(
        tool,
        "_search_ddgs",
        lambda *_: [
            {"title": "Future", "href": "https://example.com/future", "date": "2026-01-01"},
            {"title": "Unknown", "href": "https://example.com/unknown"},
        ],
    )

    results = tool._get_fallback_structured_results("AI demand", provider="ddgs")

    assert [result["url"] for result in results] == ["https://example.com/unknown"]


def test_auto_collect_with_invalid_date_format(mock_db, mock_question, caplog):
    """Invalid search dates are delegated to ArticleCollector page extraction."""
    caplog.set_level(logging.DEBUG)

    results_with_bad_date = [
        {
            "title": "Test Article",
            "url": "https://example.com/article",
            "content": "Test content",
            "engines": ["test"],
            "publishedDate": "invalid-date-format",
        }
    ]

    mock_db.get.return_value = mock_question
    mock_collector_instance = Mock()

    tool = WebSearchTool(
        db_path=None,
        collector=None,
        question_id="test_question_123",
        auto_collect_enabled=False,
        max_auto_collect=5,
        domain="general",
    )

    tool.db = mock_db
    tool.question_id = mock_question.id
    tool.question = mock_question
    tool.auto_collect_enabled = True
    tool.question_resolution_date = mock_question.resolution_date
    tool.article_collector = mock_collector_instance

    summary = tool._auto_collect_articles(results_with_bad_date)

    print("\n" + "=" * 60)
    print("Auto-collection summary (bad date):")
    print(summary)
    print("=" * 60)

    assert "auto-collected 1 article" in summary.lower()
    assert mock_collector_instance.forward.call_count == 1
    assert mock_collector_instance.forward.call_args.kwargs["published_date"] is None


def test_auto_collect_with_date_after_resolution(mock_db, mock_question, caplog):
    """Test that articles published after resolution date are skipped."""
    caplog.set_level(logging.DEBUG)

    results_future = [
        {
            "title": "Future Article",
            "url": "https://example.com/future",
            "content": "Future content",
            "engines": ["test"],
            "publishedDate": "2026-06-01T00:00:00",  # After resolution date (2026-01-01)
        }
    ]

    mock_db.get.return_value = mock_question
    mock_collector_instance = Mock()

    tool = WebSearchTool(
        db_path=None,
        collector=None,
        question_id="test_question_123",
        auto_collect_enabled=False,
        max_auto_collect=5,
        domain="general",
    )

    tool.db = mock_db
    tool.question_id = mock_question.id
    tool.question = mock_question
    tool.auto_collect_enabled = True
    tool.question_resolution_date = mock_question.resolution_date
    tool.article_collector = mock_collector_instance

    summary = tool._auto_collect_articles(results_future)

    print("\n" + "=" * 60)
    print("Auto-collection summary (future date):")
    print(summary)
    print("=" * 60)

    assert "after resolution" in summary.lower()
    assert mock_collector_instance.forward.call_count == 0


def test_auto_collect_does_not_wait_for_timed_out_worker(
    mock_db, mock_question, monkeypatch
):
    import src.tools.collectors.web_search as web_search_module

    result = {
        "title": "Slow Article",
        "url": "https://example.com/slow",
        "content": "Slow content",
        "publishedDate": "2025-01-01T00:00:00",
    }
    mock_collector_instance = Mock()

    def slow_forward(**kwargs):
        time.sleep(1)
        return Mock(id="stored")

    mock_collector_instance.forward.side_effect = slow_forward
    monkeypatch.setattr(
        web_search_module, "ARTICLE_COLLECT_TIMEOUT_SECONDS", 0.05
    )

    tool = WebSearchTool(
        db_path=None,
        collector=None,
        question_id="test_question_123",
        auto_collect_enabled=False,
        max_auto_collect=1,
        domain="general",
    )
    tool.db = mock_db
    tool.question_id = mock_question.id
    tool.question = mock_question
    tool.auto_collect_enabled = True
    tool.question_resolution_date = mock_question.resolution_date
    tool.article_collector = mock_collector_instance

    started = time.monotonic()
    summary = tool._auto_collect_articles([result])
    elapsed = time.monotonic() - started

    assert elapsed < 0.5
    assert "1 timeout" in summary.lower()


def test_fallback_search_returns_normalized_structured_results(monkeypatch):
    monkeypatch.setenv("WEB_SEARCH_FALLBACK_PROVIDER", "smolagents")
    tool = WebSearchTool(auto_collect_enabled=False)
    tool.use_searxng = False
    tool.fallback_tool = Mock()
    tool.fallback_tool.engine = "duckduckgo"
    tool.fallback_tool.search.return_value = [
        {
            "title": "Apple reports quarterly results",
            "link": "https://www.apple.com/newsroom/example",
            "description": "Apple published its quarterly results.",
        }
    ]

    result = tool.forward("Apple quarterly results")

    assert isinstance(result, WebSearchOutput)
    assert result.count == 1
    assert result.results[0].title == "Apple reports quarterly results"
    assert result.results[0].url == "https://www.apple.com/newsroom/example"
    assert result.results[0].source == "apple.com"
    assert result.results[0].published_date is None


def test_per_call_provider_override_bypasses_auto_cascade(monkeypatch):
    monkeypatch.setenv("WEB_SEARCH_FALLBACK_PROVIDER", "auto")
    tool = WebSearchTool(auto_collect_enabled=False)
    tool._search_google_news = Mock(side_effect=AssertionError("must not run"))
    tool._search_ddgs = Mock(
        return_value=[
            {
                "title": "Intel annual filing",
                "href": "https://www.intc.com/annual-report",
                "body": "Historical filing and baseline data.",
            }
        ]
    )

    result = tool.forward("Intel annual filing", provider="ddgs")

    assert result.count == 1
    tool._search_ddgs.assert_called_once()


def test_query_relevance_rejects_unrelated_search_drift():
    unrelated = {
        "title": "Shiba Inu classified as a digital commodity",
        "content": "U.S. regulators issued a new decision affecting SHIB price.",
    }
    relevant = {
        "title": "Supply disruption raises U.S. inflation risk",
        "content": "Shipping delays could raise CPI through higher goods prices.",
    }

    query = "supply chain disruptions U.S. CPI March 2026"

    assert WebSearchTool._is_relevant_to_query(unrelated, query) is False
    assert WebSearchTool._is_relevant_to_query(relevant, query) is True


@pytest.mark.parametrize(
    ("url", "title", "description", "expected"),
    [
        (
            "https://www.cnbc.com/2026/04/30/apple-earnings.html",
            "Apple earnings",
            "Quarter ended March 28, 2026.",
            "2026-04-30",
        ),
        (
            "https://example.com/apple-earnings",
            "AAPL earnings report on 4/30/2026",
            "Quarter ended March 28, 2026.",
            "2026-04-30",
        ),
        (
            "https://example.com/apple-earnings",
            "Apple reports quarterly results",
            "Apple announced results for the quarter ended March 28, 2026.",
            None,
        ),
        (
            "https://example.com/apple-earnings",
            "Apple Q2 analysis for quarter ended 2026-03-28",
            "Financial analysis of the completed quarter.",
            None,
        ),
        (
            "https://example.com/apple-earnings",
            "Apple reports quarterly results",
            "Published April 30, 2026 with complete financial statements.",
            "2026-04-30",
        ),
    ],
)
def test_search_date_inference_is_conservative(
    url, title, description, expected
):
    assert WebSearchTool._infer_published_date(
        url=url,
        title=title,
        description=description,
    ) == expected


def test_fallback_search_uses_requested_duckduckgo_page(monkeypatch):
    monkeypatch.setenv("WEB_SEARCH_FALLBACK_PROVIDER", "smolagents")
    tool = WebSearchTool(auto_collect_enabled=False)
    tool.use_searxng = False
    tool.fallback_tool = Mock()
    tool.fallback_tool.engine = "duckduckgo"
    tool._search_duckduckgo_page = Mock(
        return_value=[
            {
                "title": "A second-page result",
                "link": "https://example.com/2026/04/30/result",
                "description": "Additional evidence.",
            }
        ]
    )

    result = tool.forward("Apple earnings", page=2)

    tool._search_duckduckgo_page.assert_called_once_with("Apple earnings", 2)
    assert result.results[0].title == "A second-page result"
    assert result.results[0].published_date == "2026-04-30"


def test_fallback_search_uses_ddgs_directly_by_default(monkeypatch):
    monkeypatch.delenv("WEB_SEARCH_FALLBACK_PROVIDER", raising=False)
    tool = WebSearchTool(auto_collect_enabled=False)
    tool._search_ddgs = Mock(
        return_value=[
            {
                "title": "Direct DDGS result",
                "href": "https://example.com/2026/04/30/result",
                "body": "Additional evidence.",
            }
        ]
    )
    tool.fallback_tool = Mock()
    tool.fallback_tool.search.side_effect = AssertionError(
        "smolagents fallback should not run before DDGS"
    )

    result = tool.forward("Apple earnings")

    tool._search_ddgs.assert_called_once_with("Apple earnings", 1)
    tool.fallback_tool.search.assert_not_called()
    assert result.results[0].title == "Direct DDGS result"


def test_ddgs_strips_unsupported_date_operators(monkeypatch):
    calls = []

    class FakeDDGS:
        def text(self, query, **kwargs):
            calls.append((query, kwargs))
            return [
                {
                    "title": "Result",
                    "href": "https://example.com/result",
                    "body": "Evidence",
                }
            ]

    monkeypatch.setattr("ddgs.DDGS", FakeDDGS)

    tool = WebSearchTool(auto_collect_enabled=False)
    results = tool._search_ddgs(
        "US CPI after:2026-01-01 before:2026-02-01", 1
    )

    assert calls[0][0] == "US CPI"
    assert calls[0][1]["backend"] == "bing"
    assert results[0]["title"] == "Result"


def test_finance_search_uses_gdelt_with_cutoff_and_no_implicit_lower_bound(
    monkeypatch,
):
    captured = {}

    class FakeResponse:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {
                "articles": [
                    {
                        "title": "Inflation outlook",
                        "url": "https://example.com/inflation-outlook",
                        "seendate": "20260110T120000Z",
                    }
                ]
            }

    def fake_get(url, **kwargs):
        captured["url"] = url
        captured.update(kwargs)
        return FakeResponse()

    monkeypatch.setenv("WEB_SEARCH_FALLBACK_PROVIDER", "gdelt")
    monkeypatch.setattr("httpx.get", fake_get)
    question = Mock()
    question.estimated_start_time = datetime(2025, 12, 31)
    question.resolution_date = datetime(2026, 2, 13)
    question.metadata = {"finfactorbench": {"region": "US"}}
    tool = WebSearchTool(domain="finance", auto_collect_enabled=False)
    tool.question = question

    result = tool.forward("US CPI inflation")

    assert tool._fallback_provider() == "gdelt"
    assert "startdatetime" not in captured["params"]
    assert captured["params"]["enddatetime"] == "20260213235959"
    assert "sourcecountry:US" in captured["params"]["query"]
    assert result.results[0].published_date == "2026-01-10T12:00:00Z"


def test_finance_auto_search_uses_google_news_rss(monkeypatch):
    rss = b"""<?xml version="1.0" encoding="UTF-8"?>
    <rss version="2.0"><channel><item>
      <title>Canadian CPI growth edged lower in January - RBC</title>
      <link>https://news.google.com/rss/articles/test-finance-auto</link>
      <pubDate>Tue, 17 Feb 2026 08:00:00 GMT</pubDate>
      <description>Canadian inflation analysis.</description>
    </item></channel></rss>"""

    class FakeResponse:
        content = rss

        def raise_for_status(self):
            return None

    monkeypatch.delenv("WEB_SEARCH_FALLBACK_PROVIDER", raising=False)
    monkeypatch.setattr("httpx.get", lambda *args, **kwargs: FakeResponse())
    monkeypatch.setattr(
        "googlenewsdecoder.new_decoderv1",
        lambda _url: {
            "status": True,
            "decoded_url": "https://www.rbc.com/canada-cpi-january-2026",
        },
    )
    question = Mock()
    question.question_text = "What bucket will Canada CPI inflation fall in?"
    question.estimated_start_time = datetime(2025, 12, 31)
    question.resolution_date = datetime(2026, 2, 17)
    question.metadata = {
        "finfactorbench": {"region": "CA", "subdomain": "inflation"}
    }
    tool = WebSearchTool(domain="finance", auto_collect_enabled=False)
    tool.question = question

    result = tool.forward(
        "Canada CPI January 2026 after:2025-12-31 before:2026-02-17"
    )

    assert tool._fallback_provider() == "auto"
    assert result.count == 1
    assert result.results[0].source == "rbc.com"
    assert result.results[0].published_date == "2026-02-17T08:00:00+00:00"


def test_google_news_resolves_dated_headline_via_ddgs_when_decoder_is_throttled(
    monkeypatch,
):
    rss = b"""<?xml version="1.0" encoding="UTF-8"?>
    <rss version="2.0"><channel><item>
      <title>What is going on with the 30-year Treasury yield? - Marketplace</title>
      <link>https://news.google.com/rss/articles/throttled</link>
      <pubDate>Thu, 22 May 2025 07:00:00 GMT</pubDate>
      <description>Long-term Treasury yield analysis.</description>
      <source url="https://www.marketplace.org">Marketplace</source>
    </item></channel></rss>"""

    class FakeResponse:
        content = rss

        def raise_for_status(self):
            return None

    class FakeDDGS:
        def text(self, query, **kwargs):
            assert "30-year Treasury yield" in query
            return [
                {
                    "title": "What is going on with the 30-year Treasury yield?",
                    "href": (
                        "https://www.marketplace.org/story/2025/05/22/"
                        "whats-going-on-with-the-30year-treasury-yield"
                    ),
                    "body": "Long-term Treasury yield analysis.",
                }
            ]

    monkeypatch.setattr("httpx.get", lambda *args, **kwargs: FakeResponse())
    monkeypatch.setattr(
        "googlenewsdecoder.new_decoderv1",
        lambda _url: {"status": False, "message": "429 Too Many Requests"},
    )
    monkeypatch.setattr("ddgs.DDGS", FakeDDGS)
    tool = WebSearchTool(domain="finance", auto_collect_enabled=False)

    results = tool._search_google_news("30 year Treasury yield May 2025", 1)

    assert len(results) == 1
    assert results[0]["link"].startswith("https://www.marketplace.org/story/")
    assert results[0]["publishedDate"] == "2025-05-22T07:00:00+00:00"


def test_google_news_keeps_resolved_entries_when_one_headline_lookup_fails(
    monkeypatch,
):
    rss = b"""<?xml version="1.0" encoding="UTF-8"?>
    <rss version="2.0"><channel>
      <item>
        <title>Resolved Treasury analysis - Publisher A</title>
        <link>https://news.google.com/rss/articles/resolved</link>
        <pubDate>Thu, 10 Aug 2023 07:00:00 GMT</pubDate>
      </item>
      <item>
        <title>Unresolved Treasury analysis - Publisher B</title>
        <link>https://news.google.com/rss/articles/unresolved</link>
        <pubDate>Fri, 11 Aug 2023 07:00:00 GMT</pubDate>
      </item>
    </channel></rss>"""

    class FakeResponse:
        content = rss

        def raise_for_status(self):
            return None

    class FailingDDGS:
        def text(self, *_args, **_kwargs):
            raise RuntimeError("temporary search transport failure")

    monkeypatch.setattr("httpx.get", lambda *args, **kwargs: FakeResponse())
    monkeypatch.setattr(
        "googlenewsdecoder.new_decoderv1",
        lambda url: (
            {
                "status": True,
                "decoded_url": "https://publisher-a.example/resolved",
            }
            if url.endswith("/resolved")
            else {"status": False, "message": "decoder throttled"}
        ),
    )
    monkeypatch.setattr("ddgs.DDGS", FailingDDGS)
    tool = WebSearchTool(domain="finance", auto_collect_enabled=False)

    results = tool._search_google_news("Treasury yield August 2023", 1)

    assert len(results) == 1
    assert results[0]["link"] == "https://publisher-a.example/resolved"
    assert results[0]["publishedDate"] == "2023-08-10T07:00:00+00:00"


def test_auto_search_cascades_when_indexes_return_no_results(monkeypatch):
    monkeypatch.setenv("WEB_SEARCH_FALLBACK_PROVIDER", "auto")
    monkeypatch.setenv("WEB_SEARCH_AUTO_GDELT", "true")
    tool = WebSearchTool(domain="finance", auto_collect_enabled=False)
    tool._search_google_news = Mock(return_value=[])
    tool._search_ddgs = Mock(return_value=[])
    tool._search_gdelt = Mock(
        return_value=[
            {
                "title": "Eli Lilly reports annual revenue",
                "href": "https://example.com/lilly-revenue",
                "body": "Eli Lilly financial results and annual revenue.",
            }
        ]
    )

    result = tool.forward(
        "Eli Lilly annual revenue after:2026-01-01 before:2026-02-12"
    )

    assert result.count == 1
    tool._search_google_news.assert_called_once()
    tool._search_gdelt.assert_called_once()
    tool._search_ddgs.assert_called_once()


def test_auto_search_can_reserve_gdelt_for_explicit_fallback(monkeypatch):
    monkeypatch.setenv("WEB_SEARCH_FALLBACK_PROVIDER", "auto")
    monkeypatch.setenv("WEB_SEARCH_AUTO_GDELT", "false")
    tool = WebSearchTool(domain="finance", auto_collect_enabled=False)
    tool._search_google_news = Mock(return_value=[])
    tool._search_ddgs = Mock(return_value=[])
    tool._search_gdelt = Mock(return_value=[])

    result = tool.forward("Eli Lilly annual revenue before:2026-02-12")

    assert result.count == 0
    tool._search_google_news.assert_called_once()
    tool._search_ddgs.assert_called_once()
    tool._search_gdelt.assert_not_called()

    tool.forward(
        "Eli Lilly annual revenue before:2026-02-12",
        provider="gdelt",
    )
    tool._search_gdelt.assert_called_once()


def test_auto_search_skips_gdelt_when_ddgs_succeeds(monkeypatch):
    monkeypatch.setenv("WEB_SEARCH_FALLBACK_PROVIDER", "auto")
    tool = WebSearchTool(domain="finance", auto_collect_enabled=False)
    tool._search_google_news = Mock(return_value=[])
    tool._search_ddgs = Mock(
        return_value=[
            {
                "title": "Federal Reserve policy outlook",
                "href": "https://example.com/fed-outlook",
                "body": "Federal Reserve policy and interest-rate outlook.",
            }
        ]
    )
    tool._search_gdelt = Mock(return_value=[])

    result = tool.forward("Federal Reserve policy outlook 2026")

    assert result.count == 1
    tool._search_ddgs.assert_called_once()
    tool._search_gdelt.assert_not_called()


def test_google_news_region_filter_rejects_cross_country_noise():
    question = Mock()
    question.question_text = "What bucket will Canada CPI inflation fall in?"
    question.metadata = {
        "finfactorbench": {"region": "Canada", "subdomain": "inflation"}
    }
    tool = WebSearchTool(domain="finance", auto_collect_enabled=False)
    tool.question = question

    assert tool._google_news_entry_matches_region(
        {
            "title": "Canadian CPI growth edged lower in January",
            "summary": "Inflation analysis",
            "source": {"href": "https://example.com"},
        }
    )
    assert tool._google_news_entry_matches_region(
        {
            "title": "Outlook",
            "summary": "Economic projections",
            "source": {
                "href": "https://www.bankofcanada.ca",
                "title": "Bank of Canada",
            },
        }
    )
    assert not tool._google_news_entry_matches_region(
        {
            "title": "US CPI Preview: Release Time and Forecast",
            "summary": "Federal Reserve policy",
            "source": {"href": "https://example.com"},
        }
    )
    assert not tool._google_news_entry_matches_region(
        {
            "title": "A record-breaking year for CEO pay in Canada",
            "summary": "Executive compensation report",
            "source": {"href": "https://example.ca"},
        }
    )
    assert tool._finance_region_code() == "CA"


def test_finance_v3_metadata_namespace_is_used_for_search_context():
    question = Mock()
    question.question_text = "Which range will the US 2-year yield change fall into?"
    question.metadata = {
        "finance": {
            "region": "US",
            "category": "monetary_policy",
            "entity": "United States 2-year Treasury yield",
            "target_period": "2025-02",
        }
    }
    tool = WebSearchTool(domain="finance", auto_collect_enabled=False)
    tool.question = question

    assert tool._finance_region_code() == "US"
    assert tool._finance_metadata()["entity"] == (
        "United States 2-year Treasury yield"
    )
    assert not tool._google_news_entry_matches_region(
        {
            "title": "Reserve Bank of Australia holds rates steady",
            "summary": "Australian monetary policy news",
            "source": {"href": "https://example.com"},
        }
    )


def test_google_news_region_filter_does_not_reject_us_company_news():
    question = Mock()
    question.question_text = "Which revenue bucket will Intel report?"
    question.metadata = {
        "finfactorbench": {
            "region": "US",
            "original_domain": "corporate_earnings",
            "entity": "INTEL CORP (INTC)",
        }
    }
    tool = WebSearchTool(domain="finance", auto_collect_enabled=False)
    tool.question = question

    assert tool._google_news_entry_matches_region(
        {
            "title": "Intel files annual report and updates revenue outlook",
            "summary": "The chipmaker discussed demand and operating trends.",
            "source": {"href": "https://www.intc.com"},
        }
    )


def test_question_context_does_not_enable_auto_collect_without_opt_in(
    mock_db, mock_question, monkeypatch
):
    mock_db.get.return_value = mock_question
    monkeypatch.setattr("src.core.database.GenericDatabase", lambda _path: mock_db)

    tool = WebSearchTool(
        db_path="unused-test.db",
        question_id=mock_question.id,
        auto_collect_enabled=False,
    )

    assert tool.question is mock_question
    assert tool.auto_collect_enabled is False
    assert tool.article_collector is None


def test_browser_fetch_worker_limit_is_conservative_and_bounded(monkeypatch):
    monkeypatch.delenv("WEB_SEARCH_MAX_FETCH_WORKERS", raising=False)
    assert _max_fetch_workers() == 2

    monkeypatch.setenv("WEB_SEARCH_MAX_FETCH_WORKERS", "1")
    assert _max_fetch_workers() == 1

    monkeypatch.setenv("WEB_SEARCH_MAX_FETCH_WORKERS", "100")
    assert _max_fetch_workers() == 8


def test_google_news_resolve_worker_limit_is_separate_and_bounded(monkeypatch):
    monkeypatch.delenv("GOOGLE_NEWS_RESOLVE_WORKERS", raising=False)
    assert _max_google_news_resolve_workers() == 2

    monkeypatch.setenv("GOOGLE_NEWS_RESOLVE_WORKERS", "1")
    assert _max_google_news_resolve_workers() == 1

    monkeypatch.setenv("GOOGLE_NEWS_RESOLVE_WORKERS", "100")
    assert _max_google_news_resolve_workers() == 4


def test_search_result_snippets_are_bounded_without_changing_url_or_date():
    long_description = "market analysis " * 500

    result = WebSearchTool._normalize_result(
        {
            "title": "Treasury market update",
            "url": "https://example.com/2024/03/15/treasury-update",
            "description": long_description,
            "publishedDate": "2024-03-15T08:00:00Z",
        }
    )

    assert result.url == "https://example.com/2024/03/15/treasury-update"
    assert result.published_date == "2024-03-15T08:00:00Z"
    assert len(result.description) <= 800
    assert result.description.endswith("[truncated]")


def test_search_date_inference_uses_full_snippet_before_display_compaction():
    description = (
        "Background context. " * 80
        + " Published on March 15, 2024. Treasury market analysis."
    )

    result = WebSearchTool._normalize_result(
        {
            "title": "Treasury market update",
            "url": "https://example.com/treasury-update",
            "description": description,
        }
    )

    assert len(result.description) <= 800
    assert result.published_date == "2024-03-15"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
