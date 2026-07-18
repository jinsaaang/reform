"""Live SearchProvider backed by web tools with explicit backend selection."""

import os
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from hashlib import sha256
from typing import Final, Literal, Protocol
from urllib.parse import urlsplit, urlunsplit

from src.agents.finance_search_agent import SearchProvider, SearchProviderError
from src.domain.finance.experiment import FinanceSearchBackend
from src.domain.finance.provider import (
    GuidedSearchRequest,
    RawSearchCandidate,
    SearchProviderEnvelope,
    SearchRequest,
    SearchSourcePolicy,
    hash_exact_utf8_body,
)
from src.integrations.finance_rss_search import NewsRssSearch, compact_news_query
from src.tools.base.output_models import WebFetchOutput
from src.tools.collectors.web_fetch import WebFetchTool
from src.tools.collectors.web_search import WebSearchTool

_SECTION_PATTERN: Final = re.compile(
    r"(?ms)^##\s+\d+\.\s+(?P<title>[^\n]+)\n(?P<body>.*?)(?=^##\s+\d+\.\s+|\Z)"
)
_FIELD_PATTERNS: Final = {
    "url": re.compile(r"(?m)^\*\*URL:\*\*\s*(?P<value>\S+)\s*$"),
    "description": re.compile(r"(?m)^\*\*Description:\*\*\s*(?P<value>[^\n]*)\s*$"),
    "published": re.compile(r"(?m)^\*\*Published Date:\*\*\s*(?P<value>[^\n]*)\s*$"),
}


class SearchTool(Protocol):
    """Minimal capability needed from WorldReasoner's WebSearchTool."""

    def forward(self, query: str) -> str: ...


class FetchTool(Protocol):
    """Minimal capability needed from WorldReasoner's WebFetchTool."""

    def forward(self, url: str, timeout: int = 15) -> WebFetchOutput: ...


@dataclass(frozen=True, slots=True)
class LiveSearchConfigurationError(ValueError):
    """Reject invalid live-search adapter limits before execution."""

    detail: str

    def __str__(self) -> str:
        return f"invalid live-search configuration: {self.detail}"


def _new_search_tool() -> SearchTool:
    return WebSearchTool()


@dataclass(frozen=True, slots=True)
class _SearchHit:
    title: str
    url: str
    claim: str
    published_at: datetime


def _parse_timestamp(raw: str) -> datetime | None:
    """Parse an explicit source timestamp, rejecting malformed values."""
    normalized = raw.strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None
    return parsed.astimezone(UTC)


def _canonical_url(raw_url: str) -> str | None:
    """Normalize a web URL into a stable source identity."""
    parsed = urlsplit(raw_url.strip().rstrip(".,;"))
    if parsed.scheme.casefold() not in {"http", "https"} or not parsed.netloc:
        return None
    parts = (
        parsed.scheme.casefold(),
        parsed.netloc.casefold(),
        parsed.path or "/",
        parsed.query,
        "",
    )
    return urlunsplit(parts)


def _field(section: str, name: str) -> str | None:
    pattern = _FIELD_PATTERNS[name]
    match = pattern.search(section)
    return match.group("value").strip() if match else None


def _parse_hits(markdown: str) -> tuple[_SearchHit, ...]:
    """Parse only dated, URL-bearing WebSearchTool result sections."""
    hits: list[_SearchHit] = []
    for section_match in _SECTION_PATTERN.finditer(markdown):
        title = section_match.group("title").strip()
        section = section_match.group("body")
        raw_url = _field(section, "url")
        raw_published = _field(section, "published")
        if not raw_url or not raw_published:
            continue
        url = _canonical_url(raw_url)
        published_at = _parse_timestamp(raw_published)
        if url is None or published_at is None:
            continue
        claim = _field(section, "description") or title
        if not claim:
            continue
        hits.append(_SearchHit(title, url, claim, published_at))
    return tuple(hits)


def _query_for(request: SearchRequest) -> str:
    """Build a query from the sanitized target and outcome-free guidance."""
    profile = request.target_profile
    parts = [profile.question_text, *profile.context]
    if profile.resolution_rule:
        parts.append(profile.resolution_rule)
    parts.extend(intent.value.replace("_", " ") for intent in request.query_intents)
    if isinstance(request, GuidedSearchRequest):
        for item in request.historical_guidance:
            parts.extend((*item.matched_terms, *item.mechanism_hints))
        parts.extend(str(item.reference.dag_id) for item in request.historical_guidance)
    parts.append(f"before:{profile.cutoff.date().isoformat()}")
    return " ".join(part.strip() for part in parts if part.strip())


def _candidate(hit: _SearchHit, body: str, retrieved_at: datetime) -> str:
    """Serialize one exact-body candidate with stable source/version IDs."""
    content_hash = hash_exact_utf8_body(body)
    source_id = f"live:url:{hit.url}"
    version_id = f"{source_id}:version:{content_hash.removeprefix('sha256:')}"
    encoded_id = f"{source_id}\0{version_id}".encode("utf-8")
    candidate_id = f"live:{sha256(encoded_id).hexdigest()}"
    return RawSearchCandidate(
        candidate_id=candidate_id,
        claim=hit.claim,
        citation=hit.url,
        canonical_source_id=source_id,
        source_version_id=version_id,
        exact_body=body,
        content_hash=content_hash,
        available_at=hit.published_at.isoformat(),
        retrieved_at=retrieved_at.isoformat(),
        direction="unknown",
        context_slot="live_search",
    ).model_dump_json()


@dataclass(frozen=True, slots=True)
class LiveSearchProvider(SearchProvider):
    """Use WebSearchTool/WebFetchTool without exposing graph capabilities."""

    search_tool: SearchTool = field(default_factory=_new_search_tool)
    rss_search: NewsRssSearch | None = None
    fetch_tool: FetchTool = field(default_factory=WebFetchTool)
    result_limit: int = 5
    fetch_timeout: int = 20
    clock: Callable[[], datetime] = field(default=lambda: datetime.now(UTC))
    backend: Literal[FinanceSearchBackend.BING_NEWS_RSS_V1] | None = None

    def __post_init__(self) -> None:
        if self.result_limit <= 0:
            raise LiveSearchConfigurationError("result_limit must be positive")
        if self.fetch_timeout <= 0:
            raise LiveSearchConfigurationError("fetch_timeout must be positive")

    def search(self, request: SearchRequest) -> str:
        """Run one live pass and return a strict SearchProviderEnvelope."""
        if request.source_policy is not SearchSourcePolicy.LIVE_SEARCH:
            raise SearchProviderError(
                code="unsupported_source_policy",
                detail="live adapter requires LIVE_SEARCH",
            )
        if request.run_mode.value != "current_unresolved":
            raise SearchProviderError(
                code="historical_live_unsupported",
                detail="live search has no frozen snapshot guarantee",
            )
        query = _query_for(request)
        use_rss = self.backend is FinanceSearchBackend.BING_NEWS_RSS_V1 or (
            self.backend is None
            and not os.getenv("SEARXNG_BASE_URL")
            and bool(self.rss_search or isinstance(self.search_tool, WebSearchTool))
        )
        if use_rss:
            rss_search = self.rss_search or NewsRssSearch()
            try:
                hits = tuple(
                    _SearchHit(item.title, item.url, item.claim, item.published_at)
                    for item in rss_search.search(
                        compact_news_query(request.target_profile.question_text),
                        self.result_limit,
                    )
                )
            except (OSError, RuntimeError, TypeError, ValueError) as error:
                raise SearchProviderError(
                    "rss_search_failed", type(error).__name__
                ) from error
        else:
            try:
                markdown = self.search_tool.forward(query)
            except (OSError, RuntimeError, TypeError, ValueError) as error:
                raise SearchProviderError(
                    "search_failed", type(error).__name__
                ) from error
            hits = _parse_hits(markdown)[: self.result_limit]
        try:
            retrieved_at = self.clock()
        except (RuntimeError, TypeError, ValueError) as error:
            raise SearchProviderError("clock_failed", type(error).__name__) from error
        if retrieved_at.tzinfo is None or retrieved_at.utcoffset() is None:
            raise SearchProviderError("clock_invalid", "clock returned a naive time")
        retrieved_at = retrieved_at.astimezone(UTC)
        payloads: list[str] = []
        for hit in hits:
            if hit.published_at >= request.target_profile.cutoff.astimezone(UTC):
                continue
            if retrieved_at < hit.published_at:
                continue
            try:
                fetched = self.fetch_tool.forward(hit.url, timeout=self.fetch_timeout)
            except (OSError, RuntimeError, TypeError, ValueError) as error:
                raise SearchProviderError(
                    "fetch_failed", type(error).__name__
                ) from error
            if not fetched.success or not fetched.content:
                continue
            payloads.append(_candidate(hit, fetched.content, retrieved_at))
        envelope = SearchProviderEnvelope(candidate_payloads=tuple(payloads))
        return envelope.model_dump_json()


__all__ = [
    "FetchTool",
    "LiveSearchConfigurationError",
    "LiveSearchProvider",
    "SearchTool",
]
