"""Deterministic metadata-only search over the immutable public finance seed."""

import json
import re
import sqlite3
import unicodedata
from contextlib import closing
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable, Final

from src.agents.finance_search_agent import SearchProvider, SearchProviderError
from src.domain.finance.provider import (
    FinanceRunMode,
    RawSearchCandidate,
    SearchProviderEnvelope,
    SearchRequest,
    SearchSourcePolicy,
    canonical_snapshot_id,
    hash_exact_utf8_body,
)

_TOKEN_PATTERN: Final = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")
_STOPWORDS: Final = frozenset(
    {
        "a",
        "an",
        "and",
        "at",
        "be",
        "before",
        "by",
        "for",
        "from",
        "in",
        "is",
        "of",
        "on",
        "or",
        "the",
        "to",
        "will",
    }
)


@dataclass(frozen=True, slots=True)
class _ArticleMetadata:
    article_id: str
    title: str
    url: str | None
    source: str
    published_at: datetime
    record_created_at: datetime
    domain: str
    tags: tuple[str, ...]


def _tokens(text: str) -> frozenset[str]:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return frozenset(_TOKEN_PATTERN.findall(normalized)) - _STOPWORDS


def _parse_tags(raw: str | None) -> tuple[str, ...]:
    if not raw:
        return ()
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return ()
    if not isinstance(parsed, list):
        return ()
    return tuple(item for item in parsed if isinstance(item, str) and item.strip())


def _published_at(raw: str) -> datetime | None:
    try:
        value = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if value.tzinfo is None or value.utcoffset() is None:
        return None
    return value.astimezone(UTC)


def _article_body(article: _ArticleMetadata) -> str:
    """Return the exact metadata record exposed to the evidence boundary."""
    return json.dumps(
        {
            "article_id": article.article_id,
            "domain": article.domain,
            "published_at": article.published_at.isoformat(),
            "record_created_at": article.record_created_at.isoformat(),
            "source": article.source,
            "tags": article.tags,
            "title": article.title,
            "url": article.url,
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


@dataclass(frozen=True, slots=True)
class PublicDbMetadataSearchProvider(SearchProvider):
    """Search only dated metadata in the pinned DB; never fetch the current web."""

    db_path: Path
    result_limit: int = 5
    clock: Callable[[], datetime] = field(default=lambda: datetime.now(UTC))

    def __post_init__(self) -> None:
        if self.result_limit <= 0:
            raise ValueError("result_limit must be positive")

    def _articles(self, cutoff: datetime) -> tuple[_ArticleMetadata, ...]:
        uri = f"{self.db_path.resolve().as_uri()}?mode=ro&immutable=1"
        sql = """
        SELECT id,title,url,source,published_date,created_at,domain,tags
        FROM articles WHERE lower(domain)='finance' ORDER BY id
        """
        try:
            with closing(sqlite3.connect(uri, uri=True)) as connection:
                rows = connection.execute(sql).fetchall()
        except sqlite3.Error as error:
            raise SearchProviderError(
                "public_db_read_failed", type(error).__name__
            ) from error
        articles: list[_ArticleMetadata] = []
        for row in rows:
            published_at = _published_at(row[4])
            record_created_at = _published_at(row[5])
            if (
                published_at is None
                or record_created_at is None
                or published_at >= cutoff
            ):
                continue
            articles.append(
                _ArticleMetadata(
                    article_id=row[0],
                    title=row[1],
                    url=row[2],
                    source=row[3],
                    published_at=published_at,
                    record_created_at=record_created_at,
                    domain=row[6],
                    tags=_parse_tags(row[7]),
                )
            )
        return tuple(articles)

    @staticmethod
    def _score(article: _ArticleMetadata, query_terms: frozenset[str]) -> int:
        title_terms = _tokens(article.title)
        source_terms = _tokens(article.source)
        tag_terms = _tokens(" ".join(article.tags))
        return (
            4 * len(query_terms & title_terms)
            + 2 * len(query_terms & tag_terms)
            + len(query_terms & source_terms)
        )

    def _candidate(self, article: _ArticleMetadata, retrieved_at: datetime) -> str:
        body = _article_body(article)
        content_hash = hash_exact_utf8_body(body)
        source_id = f"public-db:article:{article.article_id}"
        version_id = f"{source_id}:published:{article.published_at.isoformat()}"
        citation = article.url or f"worldreasoner://article/{article.article_id}"
        return RawSearchCandidate(
            candidate_id=f"public-db-metadata:{article.article_id}",
            claim=article.title,
            citation=citation,
            canonical_source_id=source_id,
            source_version_id=version_id,
            exact_body=body,
            body_snapshot_id=canonical_snapshot_id(
                source_id,
                version_id,
                content_hash,
            ),
            snapshot_captured_at=article.record_created_at.isoformat(),
            snapshot_available_at=article.record_created_at.isoformat(),
            available_at=article.published_at.isoformat(),
            retrieved_at=retrieved_at.isoformat(),
            content_hash=content_hash,
            direction="unknown",
            context_slot="public_db_metadata_proxy",
        ).model_dump_json()

    def search(self, request: SearchRequest) -> str:
        """Return the most lexically relevant pre-cutoff metadata records."""
        if request.source_policy is not SearchSourcePolicy.PUBLIC_DB_METADATA:
            raise SearchProviderError(
                "unsupported_source_policy",
                "public DB adapter requires PUBLIC_DB_METADATA",
            )
        if request.run_mode is not FinanceRunMode.HISTORICAL_BACKTEST:
            raise SearchProviderError(
                "unsupported_run_mode",
                "public DB metadata is a retrospective backtest proxy",
            )
        cutoff = request.target_profile.cutoff.astimezone(UTC)
        retrieved_at = self.clock()
        if retrieved_at.tzinfo is None or retrieved_at.utcoffset() is None:
            raise SearchProviderError("clock_invalid", "clock returned a naive time")
        retrieved_at = retrieved_at.astimezone(UTC)
        query_terms = _tokens(
            " ".join(
                (
                    request.target_profile.question_text,
                    *request.target_profile.context,
                    request.target_profile.resolution_rule or "",
                )
            )
        )
        ranked = sorted(
            self._articles(cutoff),
            key=lambda article: (
                -self._score(article, query_terms),
                -article.published_at.timestamp(),
                article.article_id,
            ),
        )
        payloads = tuple(
            self._candidate(article, retrieved_at)
            for article in ranked[: self.result_limit]
            if article.published_at <= retrieved_at
        )
        return SearchProviderEnvelope(candidate_payloads=payloads).model_dump_json()


__all__ = ["PublicDbMetadataSearchProvider"]
