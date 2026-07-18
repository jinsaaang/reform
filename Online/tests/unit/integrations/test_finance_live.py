"""Offline contract tests for the live finance provider adapters."""

from datetime import datetime, timezone
from time import struct_time
from types import SimpleNamespace

from feedparser import FeedParserDict
import pytest

from src.domain.finance.provider import (
    FinanceRunMode,
    GuidedSearchRequest,
    HistoricalSearchGuidance,
    InitialSearchRequest,
    SearchQueryIntent,
    SearchSourcePolicy,
)
from src.integrations.finance_live import LiveForecastProvider, LiveSearchProvider
import src.integrations.finance_live_forecast as live_forecast
from src.integrations.finance_rss_search import NewsRssSearch, compact_news_query
from src.tools.base.output_models import RssFetchOutput, RssFeedItem
from src.tools.base.output_models import WebFetchOutput
import src.tools.collectors.rss_fetch as rss_fetch
from src.tools.collectors.rss_fetch import RssFetchTool
from tests.fixtures.finance_pipeline import make_target
from tests.unit.agents.test_finance_forecast_agent import _forecast_input
from tests.unit.domain.finance._factories import make_episode


class _FakeSearchTool:
    def __init__(self, response: str) -> None:
        self.response = response
        self.queries: list[str] = []

    def forward(self, query: str, **_: object) -> str:
        self.queries.append(query)
        return self.response


class _FakeFetchTool:
    def __init__(self, content: str) -> None:
        self.content = content
        self.urls: list[str] = []

    def forward(self, url: str, **_: object) -> WebFetchOutput:
        self.urls.append(url)
        return WebFetchOutput(
            url=url,
            title="Quarterly filing",
            content=self.content,
            success=True,
        )


class _FakeRssFetcher:
    def __init__(self, output: RssFetchOutput) -> None:
        self.output = output
        self.feed_urls: list[str] = []

    def forward(self, feed_url: str, max_items: int = 10) -> RssFetchOutput:
        self.feed_urls.append(feed_url)
        return self.output


class _FakeLLMClient:
    def __init__(self, response: str = "{}") -> None:
        self.response = response
        self.calls: list[tuple[list[dict[str, str]], dict[str, object]]] = []

    def acomplete(
        self,
        messages: list[dict[str, str]],
        response_format: dict[str, object] | None = None,
    ) -> str:
        self.calls.append((messages, response_format or {}))
        return self.response


def test_compact_news_query_removes_resolution_boilerplate() -> None:
    """Given a dated threshold question, RSS receives only discovery terms."""
    query = compact_news_query(
        "Will the S&P 500 index close at or above 7,000 on December 31, 2026?"
    )

    assert query == "S&P 500 7000 forecast 2026"


def test_compact_news_query_normalizes_us_threshold_and_year() -> None:
    query = compact_news_query("Will U.S. CPI be below 3.0 percent in December 2026?")

    assert query == "US CPI 3 percent forecast 2026"


def test_rss_fetch_preserves_utc_for_feedparser_timestamp(monkeypatch) -> None:
    """Given RSS structured time, conversion retains an explicit UTC offset."""
    entry = FeedParserDict(
        title="Rate outlook",
        link="https://example.com/rates",
        published="Fri, 17 Jul 2026 12:00:00 GMT",
        published_parsed=struct_time((2026, 7, 17, 12, 0, 0, 4, 198, 0)),
        summary="Policy update",
    )
    monkeypatch.setattr(
        rss_fetch.feedparser,
        "parse",
        lambda _: SimpleNamespace(bozo=False, entries=[entry]),
    )

    output = RssFetchTool().forward("https://example.com/feed", max_items=1)

    assert output.items[0].published == "2026-07-17T12:00:00+00:00"


def test_rss_output_preserves_items_wire_name() -> None:
    """The warning-free internal field keeps the existing serialized contract."""
    output = RssFetchOutput(feed_url="https://example.com", total_items=0, items=[])

    assert output.items == []
    assert output.model_dump() == {
        "feed_url": "https://example.com",
        "total_items": 0,
        "items": [],
    }
    assert "items" in RssFetchOutput.model_json_schema()["properties"]
    assert "feed_items" not in RssFetchOutput.model_json_schema()["properties"]


def test_live_search_emits_exact_body_and_temporal_provenance() -> None:
    """Given dated search results, only a strict pre-cutoff body is emitted."""
    search = _FakeSearchTool(
        """# Search Results\n\n## 1. Filing\n**URL:** https://example.com/filing\n**Description:** Revenue improved.\n**Published Date:** 2026-05-01T00:00:00+00:00\n\n## 2. Undated\n**URL:** https://example.com/undated\n**Description:** No date.\n"""
    )
    fetch = _FakeFetchTool("Revenue improved.\n")
    provider = LiveSearchProvider(
        search_tool=search,
        fetch_tool=fetch,
        clock=lambda: datetime(2026, 5, 2, tzinfo=timezone.utc),
        result_limit=3,
    )

    serialized = provider.search(
        InitialSearchRequest(
            target_profile=make_target(),
            source_policy=SearchSourcePolicy.LIVE_SEARCH,
            query_intents=(SearchQueryIntent.DIRECT_TARGET,),
        )
    )

    from src.domain.finance.provider import RawSearchCandidate, SearchProviderEnvelope

    envelope = SearchProviderEnvelope.model_validate_json(serialized)
    candidate = RawSearchCandidate.model_validate_json(envelope.candidate_payloads[0])
    assert candidate.exact_body == "Revenue improved.\n"
    assert (
        candidate.content_hash
        == "sha256:a0b1ee25755e0bf4e9940afdec8c5cb4110acfd8b1cb2912dd9ac35cf7ae5240"
    )
    assert candidate.available_at == "2026-05-01T00:00:00+00:00"
    assert candidate.retrieved_at == "2026-05-02T00:00:00+00:00"
    assert candidate.canonical_source_id == "live:url:https://example.com/filing"
    assert fetch.urls == ["https://example.com/filing"]


def test_live_search_rejects_historical_mode_before_tool_use() -> None:
    """Given a historical request, live search fails closed without fetching."""
    search = _FakeSearchTool("")
    fetch = _FakeFetchTool("body")
    provider = LiveSearchProvider(search_tool=search, fetch_tool=fetch)

    with pytest.raises(Exception):
        provider.search(
            InitialSearchRequest(
                run_mode=FinanceRunMode.HISTORICAL_BACKTEST,
                target_profile=make_target(),
                source_policy=SearchSourcePolicy.LIVE_SEARCH,
                query_intents=(SearchQueryIntent.DIRECT_TARGET,),
            )
        )

    assert search.queries == []
    assert fetch.urls == []


def test_guided_search_query_contains_identity_and_mechanisms_only() -> None:
    """Given guided history, query construction excludes outcomes/full episodes."""
    search = _FakeSearchTool("# Search Results")
    fetch = _FakeFetchTool("body")
    provider = LiveSearchProvider(search_tool=search, fetch_tool=fetch)
    episode = make_episode()

    provider.search(
        GuidedSearchRequest(
            target_profile=make_target(),
            source_policy=SearchSourcePolicy.LIVE_SEARCH,
            query_intents=(SearchQueryIntent.HISTORICAL_GAP,),
            historical_guidance=(
                HistoricalSearchGuidance(
                    reference=episode.reference,
                    matched_terms=("revenue",),
                    mechanism_hints=("demand",),
                ),
            ),
        )
    )

    query = search.queries[0]
    assert str(episode.dag_id) in query
    assert "demand" in query
    assert "historical_outcome" not in query
    assert "GPU demand and semiconductor revenue" not in query


def test_live_search_uses_dated_news_rss_without_searxng(monkeypatch) -> None:
    """Given no SearXNG, RSS supplies dated hits and rejects naive timestamps."""
    monkeypatch.delenv("SEARXNG_BASE_URL", raising=False)
    rss = _FakeRssFetcher(
        RssFetchOutput(
            feed_url="unused",
            total_items=2,
            items=[
                RssFeedItem(
                    title="Dated filing - Source",
                    link=(
                        "https://www.bing.com/news/apiclick.aspx?"
                        "url=https%3A%2F%2Fexample.com%2Frss-filing"
                    ),
                    published="2026-05-01T00:00:00+00:00",
                    summary="Revenue improved.",
                ),
                RssFeedItem(
                    title="Naive date - Source",
                    link="https://example.com/naive",
                    published="2026-05-01T00:00:00",
                    summary="Should be excluded.",
                ),
            ],
        )
    )
    search = _FakeSearchTool("should not be used")
    fetch = _FakeFetchTool("Revenue improved.\n")
    provider = LiveSearchProvider(
        search_tool=search,
        rss_search=NewsRssSearch(rss),
        fetch_tool=fetch,
        clock=lambda: datetime(2026, 5, 2, tzinfo=timezone.utc),
        result_limit=2,
    )

    serialized = provider.search(
        InitialSearchRequest(
            target_profile=make_target(),
            source_policy=SearchSourcePolicy.LIVE_SEARCH,
            query_intents=(SearchQueryIntent.DIRECT_TARGET,),
        )
    )

    from src.domain.finance.provider import RawSearchCandidate, SearchProviderEnvelope

    envelope = SearchProviderEnvelope.model_validate_json(serialized)
    candidate = RawSearchCandidate.model_validate_json(envelope.candidate_payloads[0])
    assert candidate.citation == "https://example.com/rss-filing"
    assert fetch.urls == ["https://example.com/rss-filing"]
    assert search.queries == []
    assert rss.feed_urls[0].startswith("https://www.bing.com/news/search?")
    assert "q=NVIDIA+revenue+analyst+expectations+forecast" in rss.feed_urls[0]
    assert "before%3A" not in rss.feed_urls[0]


def test_live_forecast_sends_only_typed_input_and_strict_schema() -> None:
    """Given typed input, the LLM boundary receives only its JSON serialization."""
    client = _FakeLLMClient()
    provider = LiveForecastProvider(client=client)

    assert provider.forecast(_forecast_input()) == "{}"

    messages, response_format = client.calls[0]
    assert [message["role"] for message in messages] == ["system", "user"]
    assert messages[1] == {
        "role": "user",
        "content": _forecast_input().model_dump_json(),
    }
    assert response_format["type"] == "json_schema"
    schema = response_format["json_schema"]
    assert isinstance(schema, dict)
    assert schema["name"] == "ForecastResult"
    assert schema["strict"] is True
    output_schema = schema["schema"]
    assert isinstance(output_schema, dict)
    assert output_schema["additionalProperties"] is False
    pending: list[object] = [output_schema]
    while pending:
        node = pending.pop()
        if isinstance(node, dict):
            if node.get("type") == "object":
                assert node.get("additionalProperties") is False
            pending.extend(node.values())
        elif isinstance(node, list):
            pending.extend(node)


def test_default_live_forecast_client_selects_reasoning_model(monkeypatch) -> None:
    """Given default settings, construction selects the reasoning model contract."""
    captured: dict[str, object] = {}

    def capture_client(config: dict[str, object]) -> _FakeLLMClient:
        captured.update(config)
        return _FakeLLMClient()

    monkeypatch.setattr(live_forecast, "LiteLLMClient", capture_client)

    _ = LiveForecastProvider()

    assert captured["model"] == "openrouter/openai/gpt-5.6-sol"
    assert captured["reasoning_effort"] == "high"
