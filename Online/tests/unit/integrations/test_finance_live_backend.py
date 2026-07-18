"""Backend-selection regressions for the live finance search adapter."""

from datetime import UTC, datetime

import pytest

from src.domain.finance.experiment import FinanceSearchBackend
from src.domain.finance.provider import (
    InitialSearchRequest,
    SearchQueryIntent,
    SearchSourcePolicy,
)
from src.integrations.finance_live_search import LiveSearchProvider
from src.integrations.finance_rss_search import NewsRssSearch, RssSearchTool
from src.tools.base.output_models import (
    RssFeedItem,
    RssFetchOutput,
    WebFetchOutput,
)
from tests.fixtures.finance_pipeline import make_target


class _WebSearchSpy:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def forward(self, query: str) -> str:
        self.queries.append(query)
        return """# Search Results

## 1. Filing
**URL:** https://example.test/filing
**Description:** Revenue improved.
**Published Date:** 2026-05-01T00:00:00+00:00
"""


class _RssSearchSpy(RssSearchTool):
    def __init__(self) -> None:
        self.urls: list[str] = []

    def forward(self, feed_url: str, max_items: int = 10) -> RssFetchOutput:
        self.urls.append(feed_url)
        return RssFetchOutput(
            feed_url=feed_url,
            total_items=1,
            items=[
                RssFeedItem(
                    title="RSS filing",
                    link="https://example.test/rss-filing",
                    published="2026-05-01T00:00:00+00:00",
                    summary="Revenue improved.",
                )
            ][:max_items],
        )


class _FetchSpy:
    def __init__(self) -> None:
        self.urls: list[str] = []

    def forward(self, url: str, timeout: int = 15) -> WebFetchOutput:
        del timeout
        self.urls.append(url)
        return WebFetchOutput(
            url=url,
            title="Filing",
            content="Revenue improved.\n",
            success=True,
        )


def _request() -> InitialSearchRequest:
    return InitialSearchRequest(
        target_profile=make_target(),
        source_policy=SearchSourcePolicy.LIVE_SEARCH,
        query_intents=(SearchQueryIntent.DIRECT_TARGET,),
    )


def test_automatic_backend_keeps_using_web_search_when_searxng_is_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given
    monkeypatch.setenv("SEARXNG_BASE_URL", "https://searxng.example.test")
    web = _WebSearchSpy()
    rss = _RssSearchSpy()
    fetch = _FetchSpy()
    provider = LiveSearchProvider(
        search_tool=web,
        rss_search=NewsRssSearch(rss),
        fetch_tool=fetch,
        clock=lambda: datetime(2026, 5, 2, tzinfo=UTC),
    )

    # When
    _ = provider.search(_request())

    # Then
    assert len(web.queries) == 1
    assert rss.urls == []
    assert fetch.urls == ["https://example.test/filing"]


def test_explicit_bing_backend_ignores_conflicting_searxng_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given
    monkeypatch.setenv("SEARXNG_BASE_URL", "https://searxng.example.test")
    web = _WebSearchSpy()
    rss = _RssSearchSpy()
    fetch = _FetchSpy()
    provider = LiveSearchProvider(
        search_tool=web,
        rss_search=NewsRssSearch(rss),
        fetch_tool=fetch,
        result_limit=5,
        fetch_timeout=20,
        clock=lambda: datetime(2026, 5, 2, tzinfo=UTC),
        backend=FinanceSearchBackend.BING_NEWS_RSS_V1,
    )

    # When
    _ = provider.search(_request())

    # Then
    assert provider.backend is FinanceSearchBackend.BING_NEWS_RSS_V1
    assert web.queries == []
    assert len(rss.urls) == 1
    assert fetch.urls == ["https://example.test/rss-filing"]
