"""No-key Bing News RSS search used when SearXNG is unavailable."""

import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Final
from urllib.parse import parse_qs, urlencode, urlsplit

from src.tools.base.output_models import RssFetchOutput
from src.tools.collectors.rss_fetch import RssFetchTool

_TOKEN_PATTERN: Final = re.compile(r"[A-Za-z]+(?:&[A-Za-z]+)?|\d+(?:[,.]\d+)*%?")
_CALENDAR_DATE_PATTERN: Final = re.compile(
    r"\b(?:january|february|march|april|may|june|july|august|september|"
    r"october|november|december)\s+\d{1,2}\b",
    re.IGNORECASE,
)
_STOP_WORDS: Final = frozenset(
    {
        "a",
        "above",
        "an",
        "and",
        "at",
        "below",
        "between",
        "be",
        "close",
        "december",
        "dollars",
        "end",
        "exceed",
        "in",
        "index",
        "its",
        "july",
        "least",
        "on",
        "once",
        "or",
        "the",
        "u",
        "s",
        "will",
    }
)


class RssSearchTool:
    """Structural RSS fetch capability for deterministic adapter tests."""

    def forward(self, feed_url: str, max_items: int = 10) -> RssFetchOutput:
        """Fetch and parse one RSS feed."""


@dataclass(frozen=True, slots=True)
class RssSearchHit:
    """Dated result metadata that can enter the existing body fetch path."""

    title: str
    url: str
    claim: str
    published_at: datetime


def news_feed_url(query: str) -> str:
    """Build a Bing News RSS search URL from a typed query."""
    params = urlencode(
        {
            "q": query,
            "format": "rss",
        }
    )
    return f"https://www.bing.com/news/search?{params}"


def compact_news_query(question_text: str) -> str:
    """Reduce binary resolution wording to stable news-discovery terms."""
    normalized_question = re.sub(r"\bU\.S\.", "US", question_text)
    normalized_question = _CALENDAR_DATE_PATTERN.sub("", normalized_question)
    terms: list[str] = []
    years: list[str] = []
    seen: set[str] = set()
    for token in _TOKEN_PATTERN.findall(normalized_question):
        normalized = token.replace(",", "")
        if normalized.endswith(".0"):
            normalized = normalized[:-2]
        key = normalized.casefold().strip(".%")
        if key in _STOP_WORDS or key in seen:
            continue
        seen.add(key)
        if len(normalized) == 4 and normalized.startswith("20"):
            years.append(normalized)
        else:
            terms.append(normalized)
    return " ".join((*terms[:11], "forecast", *years[:1]))


def _source_url(raw_url: str) -> str:
    """Recover the publisher URL exposed by Bing's RSS wrapper."""
    parsed = urlsplit(raw_url)
    if parsed.netloc.casefold().endswith("bing.com"):
        source = parse_qs(parsed.query).get("url", [])
        if source and urlsplit(source[0]).scheme.casefold() in {"http", "https"}:
            return source[0]
    return raw_url


def _parse_aware_timestamp(raw: str) -> datetime | None:
    """Accept only source timestamps carrying an explicit timezone."""
    try:
        parsed = datetime.fromisoformat(raw.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None
    return parsed.astimezone(UTC)


@dataclass(frozen=True, slots=True)
class NewsRssSearch:
    """Fetch dated news RSS metadata without an API key."""

    fetcher: RssSearchTool

    def __init__(self, fetcher: RssSearchTool | None = None) -> None:
        object.__setattr__(self, "fetcher", fetcher or RssFetchTool())

    def search(self, query: str, result_limit: int) -> tuple[RssSearchHit, ...]:
        """Return only URL-bearing, timezone-aware RSS items."""
        output = self.fetcher.forward(
            news_feed_url(query),
            max_items=result_limit,
        )
        hits: list[RssSearchHit] = []
        for item in output.items[:result_limit]:
            published_at = _parse_aware_timestamp(item.published)
            if not item.title.strip() or not item.link.strip() or published_at is None:
                continue
            claim = item.summary.strip() or item.title.strip()
            hits.append(
                RssSearchHit(
                    title=item.title.strip(),
                    url=_source_url(item.link.strip()),
                    claim=claim,
                    published_at=published_at,
                )
            )
        return tuple(hits)


__all__ = [
    "NewsRssSearch",
    "RssSearchHit",
    "RssSearchTool",
    "compact_news_query",
    "news_feed_url",
]
