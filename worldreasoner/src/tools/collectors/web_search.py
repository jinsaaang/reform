from datetime import datetime
from dotenv import load_dotenv
from email.utils import parsedate_to_datetime
from html.parser import HTMLParser
import concurrent.futures
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, List, Dict, Any
from urllib.parse import unquote, urlparse

import httpx
from smolagents.tools import Tool
from smolagents import WebSearchTool as SmolWebSearchTool

from src.tools.base.output_models import WebSearchOutput, WebSearchResultItem
from src.tools.base.schema_helper import pydantic_to_output_schema
from src.utils.logging import logger

load_dotenv()

# Keep article collection timeout consistent across all ArticleCollector calls.
ARTICLE_COLLECT_TIMEOUT_SECONDS = 15
DEFAULT_MAX_FETCH_WORKERS = 2
DEFAULT_GOOGLE_NEWS_RESOLVE_WORKERS = 2
GDELT_MIN_INTERVAL_SECONDS = 5.1
GDELT_MAX_RETRIES = 2
_GDELT_LOCK = threading.Lock()
_GDELT_CACHE: Dict[str, List[Dict[str, Any]]] = {}
_GDELT_LAST_REQUEST_AT = 0.0
_GOOGLE_NEWS_URL_CACHE: Dict[str, str] = {}


def _max_fetch_workers() -> int:
    """Return a conservative per-process browser fan-out limit."""
    raw_value = os.getenv("WEB_SEARCH_MAX_FETCH_WORKERS", "")
    try:
        configured = int(raw_value) if raw_value else DEFAULT_MAX_FETCH_WORKERS
    except ValueError:
        logger.warning(
            "Ignoring invalid WEB_SEARCH_MAX_FETCH_WORKERS="
            f"{raw_value!r}; using {DEFAULT_MAX_FETCH_WORKERS}"
        )
        configured = DEFAULT_MAX_FETCH_WORKERS
    return max(1, min(configured, 8))


def _max_google_news_resolve_workers() -> int:
    """Limit lightweight RSS URL resolution separately from browser crawling."""
    raw_value = os.getenv("GOOGLE_NEWS_RESOLVE_WORKERS", "")
    try:
        configured = (
            int(raw_value)
            if raw_value
            else DEFAULT_GOOGLE_NEWS_RESOLVE_WORKERS
        )
    except ValueError:
        logger.warning(
            "Ignoring invalid GOOGLE_NEWS_RESOLVE_WORKERS="
            f"{raw_value!r}; using {DEFAULT_GOOGLE_NEWS_RESOLVE_WORKERS}"
        )
        configured = DEFAULT_GOOGLE_NEWS_RESOLVE_WORKERS
    return max(1, min(configured, 4))


_SEARCH_RELEVANCE_STOPWORDS = {
    "about", "after", "annual", "before", "could", "economic", "economy",
    "effect", "financial", "forecast", "impact", "latest", "market", "march",
    "news", "outlook", "report", "reports", "the", "this", "what", "will",
}
_SEARCH_RELEVANCE_SIGNAL_TERMS = {
    "cpi", "hicp", "inflation", "oil", "energy", "fuel", "gas", "wage",
    "wages", "payroll", "unemployment", "revenue", "margin", "earnings",
    "inventory", "inventories", "production", "tariff", "tariffs", "rates",
    "rate", "supply", "shipping", "credit", "spread", "volatility",
}


class _DuckDuckGoNextFormParser(HTMLParser):
    """Extract the opaque fields required by DuckDuckGo Lite pagination."""

    def __init__(self) -> None:
        super().__init__()
        self.in_next_form = False
        self.fields: Dict[str, str] = {}

    def handle_starttag(self, tag: str, attrs: list[tuple[str, Optional[str]]]) -> None:
        attributes = dict(attrs)
        if tag == "form" and "next_form" in (attributes.get("class") or "").split():
            self.in_next_form = True
        elif self.in_next_form and tag == "input" and attributes.get("name"):
            self.fields[attributes["name"]] = attributes.get("value") or ""

    def handle_endtag(self, tag: str) -> None:
        if tag == "form" and self.in_next_form:
            self.in_next_form = False


class WebSearchTool(Tool):
    """
    A unified web search tool that uses SearXNG if configured, otherwise falls back to default web search.

    If SEARXNG_BASE_URL is set in environment variables, this tool will use a SearXNG instance
    for privacy-focused meta-search. Otherwise, it uses the default smolagents WebSearchTool.
    """

    name: str = "WebSearchTool"
    description: str = (
        "Performs a web search using either SearXNG or default search. "
        "Returns the same structured results object for every backend, with "
        "title, URL, description, source, and optional published date.\n\n"
        "CUSTOM DATE RANGES:\n"
        "To define a custom date range, include search operators directly in your query string:\n"
        "  - after:YYYY-MM-DD  (e.g., after:2025-01-01)\n"
        "  - before:YYYY-MM-DD (e.g., before:2025-12-31)\n"
        "  - Example: 'AI news after:2025-11-01 before:2025-11-30'"
    )

    is_initialized: bool = False

    inputs = {
        "query": {"type": "string", "description": "The search query string."},
        "categories": {
            "type": "string",
            "description": "Optional categories to search (e.g., 'general', 'news', 'images')",
            "nullable": True,
        },
        "language": {
            "type": "string",
            "description": "Optional language code (e.g., 'en', 'fr')",
            "nullable": True,
        },
        "page": {
            "type": "integer",
            "description": "Optional page number for results (default: 1)",
            "nullable": True,
        },
        "factor": {
            "type": "string",
            "description": "Exact planned factor name from search_coverage.",
            "nullable": True,
        },
        "provider": {
            "type": "string",
            "enum": ["auto", "google_news", "ddgs", "gdelt", "smolagents"],
            "description": "Optional backend override for this focused search.",
            "nullable": True,
        },
    }
    output_type = "object"
    output_schema = pydantic_to_output_schema(WebSearchOutput)

    def __init__(
        self,
        db_path: str = None,
        collector=None,
        question_id: Optional[str] = None,
        auto_collect_enabled: bool = False,
        max_auto_collect: int = 10,
        domain: str = "general",
        coverage_tracker=None,
        enforce_upper_only_dates: bool = False,
    ):
        """Initialize WebSearchTool with optional auto-collect.

        Args:
            auto_collect_enabled: If True, automatically collect articles with publishedDate < question_resolution_date
            question_id, question_resolution_date: Required if auto_collect_enabled=True
            db, db_path, collector, domain, max_auto_collect: Passed to ArticleCollectorTool if enabled
            enforce_upper_only_dates: Remove caller-supplied ``after:`` operators
                so hindsight evidence can use the full history before resolution.
        """
        # Initialize attributes that may be used later
        self.auto_collect_enabled = False
        self.question_id = question_id
        self.question = None
        self.question_resolution_date = None

        self.db = None
        if db_path:
            from src.core.database import GenericDatabase

            self.db = GenericDatabase(db_path)
        if self.db and question_id:
            from src.domain.models.question import Question

            self.question = self.db.get(Question, question_id)
            # A question supplies the temporal/provenance context, but must not
            # silently turn collection on.  The evidence agent already calls
            # ArticleCollectorTool for selected search results; enabling both
            # paths fetches every page twice.
            self.auto_collect_enabled = bool(
                auto_collect_enabled and self.question is not None
            )
            self.question_resolution_date = (
                self.question.resolution_date if self.question else None
            )

        self.max_auto_collect = max_auto_collect
        self.domain = domain
        self.coverage_tracker = coverage_tracker
        self.enforce_upper_only_dates = enforce_upper_only_dates

        # Initialize article collector if enabled
        if self.auto_collect_enabled:
            from src.tools.collectors.article_collector import ArticleCollectorTool

            self.article_collector = ArticleCollectorTool(
                db=self.db,
                db_path=db_path,
                collector=collector,
                question_id=self.question_id,
            )
            logger.info(
                f"Auto-collect enabled (question_id={question_id}, max={max_auto_collect})"
            )
        else:
            self.article_collector = None

        # Set up search backend
        self.searxng_base_url = os.getenv("SEARXNG_BASE_URL")
        self.use_searxng = bool(self.searxng_base_url)

        if self.use_searxng:
            logger.info(f"Using SearXNG at {self.searxng_base_url}")
            self.client = httpx.Client(
                base_url=self.searxng_base_url,
                timeout=30.0,
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                    "Accept": "application/json",
                    "Accept-Language": "en-US,en;q=0.9",
                },
                follow_redirects=True,
            )
        else:
            fallback_provider = self._fallback_provider()
            logger.info(f"Using fallback web search provider {fallback_provider}")
            self.client = None

    def _fallback_provider(self) -> str:
        """Choose a news index for finance while preserving the general default."""
        default_provider = "auto" if self.domain == "finance" else "ddgs"
        return os.getenv(
            "WEB_SEARCH_FALLBACK_PROVIDER", default_provider
        ).lower()

    def _query_mode(self) -> str:
        """Return the configured transport-side query expansion mode."""
        default_mode = "finance_variants" if self.domain == "finance" else "original"
        return os.getenv("WEB_SEARCH_QUERY_MODE", default_mode).lower()

    @staticmethod
    def _auto_gdelt_enabled() -> bool:
        """Whether ``auto`` may invoke the slow GDELT transport.

        Explicit ``provider='gdelt'`` requests remain available regardless of
        this flag. This lets latency-sensitive forecasting runs reserve GDELT
        for a final, deliberate gap-filling query.
        """
        return os.getenv("WEB_SEARCH_AUTO_GDELT", "true").strip().lower() not in {
            "0",
            "false",
            "no",
            "off",
        }

    def _finance_metadata(self) -> Dict[str, Any]:
        """Return finance metadata from the active or legacy dataset schema."""
        if self.question is None:
            return {}
        root = self.question.metadata or {}
        for namespace in ("finance", "finfactorbench"):
            candidate = root.get(namespace, {})
            if isinstance(candidate, dict) and candidate:
                return candidate
        return {}

    def _finance_region_code(self) -> Optional[str]:
        """Normalize benchmark country names and short codes for search."""
        metadata = self._finance_metadata()
        if not metadata.get("region"):
            return None
        raw_region = str(metadata["region"]).strip().lower()
        aliases = {
            "us": "US",
            "united states": "US",
            "united states of america": "US",
            "ca": "CA",
            "canada": "CA",
            "uk": "UK",
            "united kingdom": "UK",
            "great britain": "UK",
            "jp": "JP",
            "japan": "JP",
            "au": "AU",
            "australia": "AU",
            "ea": "EA",
            "euro area": "EA",
            "eurozone": "EA",
            "in": "IN",
            "india": "IN",
        }
        return aliases.get(raw_region, raw_region.upper())

    def forward(
        self,
        query: str,
        categories: Optional[str] = None,
        language: Optional[str] = None,
        page: Optional[int] = 1,
        factor: Optional[str] = None,
        provider: Optional[str] = None,
    ) -> WebSearchOutput:
        """
        Perform a web search using either SearXNG or the default search tool.

        Args:
            query: The search query string
            categories: Optional categories for SearXNG (ignored for default search)
            language: Optional language code for SearXNG (ignored for default search)
            page: Optional page number for SearXNG (ignored for default search)

        Returns:
            Backend-independent structured search results
        """
        query = self._with_upper_cutoff(query)
        page_number = max(1, page or 1)
        if self.coverage_tracker:
            allowed, reason = self.coverage_tracker.allow_search(
                query, page_number, provider, factor
            )
            if not allowed:
                return WebSearchOutput(
                    query=query,
                    results=[],
                    count=0,
                    error=f"Search orchestration rejected call: {reason}",
                )

        try:
            structured_results = self._get_structured_results(
                query,
                categories,
                language,
                page_number,
                provider=provider,
            )
            if not structured_results and self.use_searxng:
                structured_results = self._get_fallback_structured_results(
                    query, page=page_number, provider=provider
                )
        except Exception as exc:
            logger.error(f"Web search failed for '{query}': {exc}")
            if self.coverage_tracker:
                self.coverage_tracker.record_search(
                    query=query,
                    page=page_number,
                    provider=provider,
                    factor=factor,
                    error=str(exc),
                )
            return WebSearchOutput(
                query=query,
                results=[],
                count=0,
                error=str(exc),
            )

        collection_summary = None
        if self.auto_collect_enabled and structured_results:
            collection_summary = self._auto_collect_articles(
                structured_results, query=query
            )
            provider_mode = (provider or self._fallback_provider()).lower()
            if provider_mode == "auto" and not self._collection_target_met(
                collection_summary
            ) and not self._collection_had_timeout(collection_summary):
                # Raw Google News results are not useful when every item is
                # irrelevant, duplicate, future-dated, or unfetchable. Cascade
                # within the same bounded search action based on *persisted*
                # evidence progress, rather than forcing the model to spend
                # another query on the same factor.
                current_engines = {
                    str(engine).lower()
                    for result in structured_results
                    for engine in (result.get("engines") or [])
                }
                if any("google-news" in engine for engine in current_engines):
                    fallback_providers = (
                        ("ddgs", "gdelt")
                        if self._auto_gdelt_enabled()
                        else ("ddgs",)
                    )
                elif any("ddgs" in engine for engine in current_engines):
                    fallback_providers = (
                        ("gdelt",) if self._auto_gdelt_enabled() else ()
                    )
                else:
                    fallback_providers = ()

                known_urls = {
                    str(result.get("url") or "")
                    for result in structured_results
                    if result.get("url")
                }
                summaries = [collection_summary]
                for fallback_provider in fallback_providers:
                    try:
                        fallback_results = self._get_fallback_structured_results(
                            query,
                            page=page_number,
                            provider=fallback_provider,
                        )
                    except Exception as exc:
                        logger.warning(
                            f"{fallback_provider.upper()} usable-result fallback "
                            f"failed ({type(exc).__name__})"
                        )
                        continue
                    new_results = [
                        item
                        for item in fallback_results
                        if item.get("url") and item.get("url") not in known_urls
                    ]
                    if not new_results:
                        continue
                    known_urls.update(str(item["url"]) for item in new_results)
                    structured_results.extend(new_results)
                    fallback_summary = self._auto_collect_articles(
                        new_results,
                        query=query,
                    )
                    summaries.append(
                        f"**{fallback_provider.upper()} usable-result fallback**"
                        f"{fallback_summary}"
                    )
                    if self._collection_had_timeout(fallback_summary):
                        break
                    if self._collection_target_met(fallback_summary):
                        break
                collection_summary = "\n\n".join(summaries)

        normalized = [self._normalize_result(result) for result in structured_results]
        if self.coverage_tracker:
            self.coverage_tracker.record_search(
                query=query,
                page=page_number,
                provider=provider,
                factor=factor,
                result_urls=[item.url for item in normalized if item.url],
                raw_result_count=len(normalized),
            )
        return WebSearchOutput(
            query=query,
            results=normalized,
            count=len(normalized),
            collection_summary=collection_summary,
        )

    def _collection_target_met(self, latest_summary: Optional[str]) -> bool:
        """Report persisted coverage, or collection progress without a tracker."""
        if self.coverage_tracker:
            return bool(self.coverage_tracker.snapshot()["coverage_target_met"])
        match = re.search(
            r"\*\*Auto-collected\s+(\d+)\s+article",
            latest_summary or "",
            re.IGNORECASE,
        )
        return bool(match and int(match.group(1)) > 0)

    @staticmethod
    def _collection_had_timeout(summary: Optional[str]) -> bool:
        """Avoid overlapping provider batches while timed-out fetches unwind."""
        return bool(
            re.search(
                r"\b[1-9]\d*\s+timeout\b",
                summary or "",
                re.IGNORECASE,
            )
        )

    def _with_upper_cutoff(self, query: str) -> str:
        """Append the question cutoff only when the caller omitted ``before:``.

        Search remains upper-bounded for hindsight evidence without inventing a
        lower date bound or overriding a deliberately narrower caller cutoff.
        """
        if self.enforce_upper_only_dates:
            query = re.sub(
                r"\bafter:\d{4}-\d{2}-\d{2}\b",
                "",
                query,
                flags=re.IGNORECASE,
            )
            query = " ".join(query.split())
        if not self.question_resolution_date:
            return query
        if re.search(r"\bbefore:\d{4}-\d{2}-\d{2}\b", query, flags=re.IGNORECASE):
            return query
        cutoff_date = self.question_resolution_date.date().isoformat()
        return f"{query.rstrip()} before:{cutoff_date}"

    @staticmethod
    def _normalize_result(result: Dict[str, Any]) -> WebSearchResultItem:
        """Convert backend-specific keys to the public search-result contract."""
        from urllib.parse import urlparse

        url = result.get("url") or result.get("link") or ""
        raw_title = result.get("title") or "No title"
        raw_description = result.get("content") or result.get("description") or ""
        title = WebSearchTool._compact_search_text(
            raw_title,
            limit=300,
        )
        description = WebSearchTool._compact_search_text(
            raw_description,
            limit=800,
        )
        source = result.get("source")
        if not source and url:
            source = urlparse(url).netloc.removeprefix("www.") or None
        published_date = result.get("publishedDate") or result.get("published_date")
        if not published_date:
            published_date = WebSearchTool._infer_published_date(
                url=url,
                title=str(raw_title),
                description=str(raw_description),
            )
        return WebSearchResultItem(
            title=title,
            url=url,
            description=description,
            source=source,
            published_date=published_date,
        )

    @staticmethod
    def _compact_search_text(value: Any, *, limit: int) -> str:
        """Keep result-list context bounded; full articles remain in the DB."""
        compact = " ".join(str(value).split())
        if len(compact) <= limit:
            return compact
        return compact[: max(0, limit - 15)].rstrip() + " … [truncated]"

    @staticmethod
    def _infer_published_date(
        *, url: str, title: str, description: str
    ) -> Optional[str]:
        """Infer a conservative publication-date hint from search metadata.

        DuckDuckGo Lite does not expose a dedicated publication-date field even
        when an exact date is present in the URL or headline.  Returning that
        visible date prevents evidence agents from discarding otherwise valid
        results.  ArticleCollectorTool still verifies this hint against the page.
        """

        def valid_iso(year: str, month: str, day: str) -> Optional[str]:
            try:
                return datetime(int(year), int(month), int(day)).date().isoformat()
            except ValueError:
                return None

        numeric_ymd = re.compile(
            r"(?<!\d)(20\d{2})[\-_/](0?[1-9]|1[0-2])[\-_/](0?[1-9]|[12]\d|3[01])(?!\d)"
        )
        numeric_mdy = re.compile(
            r"(?<!\d)(0?[1-9]|1[0-2])[\-/](0?[1-9]|[12]\d|3[01])[\-/](20\d{2})(?!\d)"
        )
        month_names = {
            name.lower(): index
            for index, names in enumerate(
                (
                    (),
                    ("jan", "january"),
                    ("feb", "february"),
                    ("mar", "march"),
                    ("apr", "april"),
                    ("may",),
                    ("jun", "june"),
                    ("jul", "july"),
                    ("aug", "august"),
                    ("sep", "sept", "september"),
                    ("oct", "october"),
                    ("nov", "november"),
                    ("dec", "december"),
                )
            )
            for name in names
        }
        named_date = re.compile(
            r"\b(" + "|".join(map(re.escape, month_names))
            + r")\.?\s+(0?[1-9]|[12]\d|3[01]),?\s+(20\d{2})\b",
            re.IGNORECASE,
        )

        def first_date(text: str) -> Optional[str]:
            match = numeric_ymd.search(text)
            if match:
                return valid_iso(*match.groups())
            match = numeric_mdy.search(text)
            if match:
                month, day, year = match.groups()
                return valid_iso(year, month, day)
            match = named_date.search(text)
            if match:
                month, day, year = match.groups()
                return valid_iso(year, str(month_names[month.lower()]), day)
            return None

        # A full date embedded in a canonical URL is usually publication routing
        # metadata.  Headline dates can instead be fiscal-period end dates, so
        # require an explicit publication-like preposition there.
        if inferred := first_date(unquote(url)):
            return inferred
        title_context = re.search(
            r"\b(?:on|published|posted|released|filed|updated)\b"
            r"\s*(?:[:\-]\s*)?(.+)$",
            title,
            re.IGNORECASE,
        )
        if title_context and (inferred := first_date(title_context.group(1))):
            return inferred

        # Snippets often contain event-period dates.  Only accept a date that is
        # explicitly presented as publication/release metadata or leads the text.
        leading = re.match(r"^\s*(.{0,24})", description)
        if leading and (inferred := first_date(leading.group(1))):
            return inferred
        contextual = re.search(
            r"\b(?:(?:published|posted|released|filed|updated)"
            r"\s*(?:(?:on|at)\s+|[:\-]\s*)?|announced\s+on\s+)"
            r"((?:20\d{2}[\-_/]\d{1,2}[\-_/]\d{1,2})|"
            r"(?:\d{1,2}[\-/]\d{1,2}[\-/]20\d{2})|"
            r"(?:(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|"
            r"jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|"
            r"oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\.?\s+"
            r"\d{1,2},?\s+20\d{2}))",
            description,
            re.IGNORECASE,
        )
        if contextual:
            return first_date(contextual.group(1))
        return None

    def _get_fallback_structured_results(
        self,
        query: str,
        page: Optional[int] = 1,
        provider: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Return structured results from the configured fallback provider."""
        target_page = max(1, page or 1)
        provider = (provider or self._fallback_provider()).lower()

        # DuckDuckGo Lite currently rejects or stalls automated requests in the
        # finance runtime.  Use the already-supported DDGS transport directly by
        # default instead of paying that failure latency on every agent search.
        if provider == "auto":
            try:
                raw_results = self._filter_results_before_cutoff(
                    self._search_google_news(query, target_page)
                )
                if not raw_results:
                    raise RuntimeError("Google News RSS returned no cutoff-safe results")
                engine = "google-news-rss"
            except Exception as exc:
                logger.warning(
                    "Google News RSS search failed "
                    f"({type(exc).__name__}); retrying with DDGS"
                )
                try:
                    raw_results = self._filter_results_before_cutoff(
                        self._search_ddgs(query, target_page)
                    )
                    if not raw_results:
                        raise RuntimeError("DDGS returned no cutoff-safe results")
                    engine = f"ddgs:{os.getenv('DDGS_BACKEND', 'bing')}"
                except Exception as ddgs_exc:
                    if self._auto_gdelt_enabled():
                        logger.warning(
                            "DDGS search failed "
                            f"({type(ddgs_exc).__name__}); retrying with GDELT"
                        )
                        raw_results = self._filter_results_before_cutoff(
                            self._search_gdelt(query, target_page)
                        )
                        engine = "gdelt"
                    else:
                        logger.warning(
                            "DDGS search failed "
                            f"({type(ddgs_exc).__name__}); automatic GDELT disabled"
                        )
                        raw_results = []
                        engine = f"ddgs:{os.getenv('DDGS_BACKEND', 'bing')}"
        elif provider == "google_news":
            raw_results = self._filter_results_before_cutoff(
                self._search_google_news(query, target_page)
            )
            engine = "google-news-rss"
        elif provider == "gdelt":
            raw_results = self._filter_results_before_cutoff(
                self._search_gdelt(query, target_page)
            )
            engine = "gdelt"
        elif provider == "ddgs":
            raw_results = self._filter_results_before_cutoff(
                self._search_ddgs(query, target_page)
            )
            engine = f"ddgs:{os.getenv('DDGS_BACKEND', 'bing')}"
        else:
            if provider != "smolagents":
                logger.warning(
                    f"Unknown WEB_SEARCH_FALLBACK_PROVIDER={provider}; using DDGS"
                )
                raw_results = self._filter_results_before_cutoff(
                    self._search_ddgs(query, target_page)
                )
                engine = f"ddgs:{os.getenv('DDGS_BACKEND', 'bing')}"
            else:
                if not hasattr(self, "fallback_tool"):
                    self.fallback_tool = SmolWebSearchTool()
                engine = getattr(self.fallback_tool, "engine", "fallback")
                try:
                    if target_page > 1 and engine == "duckduckgo":
                        raw_results = self._search_duckduckgo_page(
                            query, target_page
                        )
                    else:
                        raw_results = self.fallback_tool.search(query)
                    raw_results = self._filter_results_before_cutoff(raw_results)
                    if not raw_results:
                        raw_results = self._filter_results_before_cutoff(
                            self._search_ddgs(query, target_page)
                        )
                        engine = f"ddgs:{os.getenv('DDGS_BACKEND', 'bing')}"
                except Exception as exc:
                    logger.warning(
                        f"Primary fallback search failed ({type(exc).__name__}); "
                        "retrying with DDGS"
                    )
                    raw_results = self._filter_results_before_cutoff(
                        self._search_ddgs(query, target_page)
                    )
                    engine = f"ddgs:{os.getenv('DDGS_BACKEND', 'bing')}"
        return [
            {
                "title": result.get("title", "No title"),
                "url": result.get("link") or result.get("href", ""),
                "content": result.get("description") or result.get("body", ""),
                "engines": [engine],
                "publishedDate": result.get("publishedDate"),
            }
            for result in raw_results
        ]

    def _filter_results_before_cutoff(
        self, results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Drop only results with a reliably known date at or after the cutoff."""
        if not self.question_resolution_date:
            return results
        cutoff_date = self.question_resolution_date.date()
        filtered = []
        for result in results:
            published = (
                result.get("publishedDate")
                or result.get("published_date")
                or result.get("date")
            )
            if not published:
                published = self._infer_published_date(
                    url=str(result.get("url") or result.get("link") or result.get("href") or ""),
                    title=str(result.get("title") or ""),
                    description=str(result.get("content") or result.get("description") or result.get("body") or ""),
                )
            known_date = self._parse_known_date(published)
            if known_date is not None and known_date >= cutoff_date:
                continue
            filtered.append(result)
        return filtered

    @staticmethod
    def _parse_known_date(value: Any):
        if not value:
            return None
        if isinstance(value, datetime):
            return value.date()
        try:
            from src.tools.collectors.article_collector import ArticleCollectorTool

            parsed = ArticleCollectorTool._parse_date_candidate(str(value))
        except (TypeError, ValueError):
            return None
        return parsed.date() if parsed is not None else None

    def _search_google_news(
        self, query: str, page: int
    ) -> List[Dict[str, Any]]:
        """Search dated Google News RSS results and resolve publisher URLs."""
        import feedparser
        from googlenewsdecoder import new_decoderv1

        page_size = 10
        target_page = max(1, page)
        region = self._finance_region_code() or "US"
        locale = {
            "CA": ("en-CA", "CA", "CA:en"),
            "UK": ("en-GB", "GB", "GB:en"),
            "AU": ("en-AU", "AU", "AU:en"),
            "IN": ("en-IN", "IN", "IN:en"),
        }.get(region, ("en-US", "US", "US:en"))

        response = httpx.get(
            "https://news.google.com/rss/search",
            params={
                "q": query,
                "hl": locale[0],
                "gl": locale[1],
                "ceid": locale[2],
            },
            timeout=30.0,
            follow_redirects=True,
        )
        response.raise_for_status()
        entries = feedparser.parse(response.content).entries
        entries = [
            entry
            for entry in entries
            if self._google_news_entry_matches_region(entry)
        ]
        start = (target_page - 1) * page_size
        page_entries = entries[start : start + page_size]
        if self.auto_collect_enabled:
            # Resolving a throttled Google News URL may require one exact-title
            # lookup. Do not resolve more entries than this call can collect.
            page_entries = page_entries[: self.max_auto_collect]

        def resolve(entry: Any) -> Optional[Dict[str, Any]]:
            google_url = entry.get("link") or ""
            if not google_url:
                return None
            decoded_url = _GOOGLE_NEWS_URL_CACHE.get(google_url)
            if decoded_url is None:
                skip_decoder = (
                    os.getenv("WEB_SEARCH_SKIP_GOOGLE_DECODER", "false").lower()
                    == "true"
                )
                decoded = (
                    {"status": False}
                    if skip_decoder
                    else new_decoderv1(google_url)
                )
                if decoded.get("status") and decoded.get("decoded_url"):
                    decoded_url = decoded["decoded_url"]
                else:
                    # Google's batchexecute decoder is frequently rate-limited.
                    # The RSS entry still provides a dated headline and publisher,
                    # so resolve that exact headline through DDGS and prefer the
                    # matching publisher domain. This preserves dated news
                    # evidence instead of falling through to undated web pages.
                    decoded_url = self._resolve_google_news_entry_via_ddgs(entry)
                if not decoded_url:
                    return None
                _GOOGLE_NEWS_URL_CACHE[google_url] = decoded_url

            published_date = None
            if entry.get("published"):
                try:
                    published_date = parsedate_to_datetime(
                        entry["published"]
                    ).isoformat()
                except (TypeError, ValueError):
                    published_date = None
            return {
                "title": entry.get("title") or "No title",
                "link": decoded_url,
                "description": entry.get("summary") or "",
                "publishedDate": published_date,
            }

        with ThreadPoolExecutor(
            max_workers=max(
                1,
                min(len(page_entries), _max_google_news_resolve_workers()),
            )
        ) as executor:
            resolved = list(executor.map(resolve, page_entries))
        results = [result for result in resolved if result is not None]
        if not results and page_entries:
            raise ValueError("Google News RSS publisher URL resolution failed")
        return results

    @staticmethod
    def _resolve_google_news_entry_via_ddgs(entry: Any) -> Optional[str]:
        """Resolve a dated RSS headline when Google's URL decoder is throttled."""
        from ddgs import DDGS

        title = str(entry.get("title") or "").strip()
        if not title:
            return None
        source = entry.get("source") or {}
        source_url = source.get("href", "") if hasattr(source, "get") else ""
        source_title = source.get("title", "") if hasattr(source, "get") else ""
        source_host = urlparse(source_url).netloc.lower().removeprefix("www.")

        headline = title
        publisher_suffix = f" - {source_title}"
        if source_title and headline.lower().endswith(publisher_suffix.lower()):
            headline = headline[: -len(publisher_suffix)].strip()

        try:
            candidates = DDGS().text(
                f'"{headline}"',
                backend=os.getenv("DDGS_BACKEND", "bing"),
                max_results=5,
            )
        except Exception as exc:
            # A single publisher-URL lookup must not discard every other dated
            # RSS result resolved successfully in the same batch. The caller
            # will omit only this entry and retain the usable results.
            logger.warning(
                "Google News headline resolution failed "
                f"({type(exc).__name__}): {headline}"
            )
            return None
        ranked: list[tuple[int, str]] = []
        headline_tokens = WebSearchTool._relevance_tokens(headline)
        for candidate in candidates or []:
            url = candidate.get("href") or candidate.get("url") or ""
            host = urlparse(url).netloc.lower().removeprefix("www.")
            if not url or not host or host.endswith("google.com"):
                continue
            candidate_title = str(candidate.get("title") or "")
            overlap = len(
                headline_tokens
                & WebSearchTool._relevance_tokens(candidate_title)
            )
            domain_match = bool(
                source_host
                and (
                    host == source_host
                    or host.endswith("." + source_host)
                    or source_host.endswith("." + host)
                )
            )
            score = (100 if domain_match else 0) + overlap
            ranked.append((score, url))
        if not ranked:
            return None
        score, url = max(ranked, key=lambda item: item[0])
        return url if score >= 2 else None

    def _google_news_entry_matches_region(self, entry: Any) -> bool:
        """Reject obvious cross-country noise before expensive page crawling."""
        if self.domain != "finance" or self.question is None:
            return True
        metadata = self._finance_metadata()
        # Country terms are useful disambiguators for macro releases, but they
        # are not a valid relevance test for company news.  A headline such as
        # "Intel files annual report" need not say "U.S.", "Fed", or "BLS".
        # Preserve the legacy behavior when older rows do not identify their
        # category, and apply strict region filtering only to country-specific
        # macro and monetary-policy questions when the category is known.
        category = str(
            metadata.get("original_domain") or metadata.get("domain") or ""
        ).strip().lower()
        if category and category not in {"macro", "monetary_policy"}:
            return True
        region = self._finance_region_code()
        region_patterns = {
            "US": r"united states|u\.s\.|\bamerican\b|federal reserve|"
            r"\bfed\b|cpi-u|\bbls\b",
            "CA": r"\bcanada\b|\bcanadian\b|bank of canada|\bboc\b|"
            r"nova scotia|\bstatcan\b",
            "UK": r"united kingdom|\bbritish\b|\bbritain\b|\bu\.k\.\b|"
            r"bank of england|\bboe\b|\bons\b",
            "JP": r"\bjapan\b|\bjapanese\b|bank of japan|\bboj\b|\btokyo\b",
            "AU": r"\baustralia\b|\baustralian\b|reserve bank of australia|"
            r"\brba\b",
            "EA": r"euro area|\beurozone\b|european central bank|\becb\b|"
            r"\bhicp\b|\beurostat\b",
            "IN": r"\bindia\b|\bindian\b|reserve bank of india|\brbi\b",
        }
        trusted_hosts = {
            "US": ("bls.gov", "federalreserve.gov", "clevelandfed.org"),
            "CA": ("bankofcanada.ca", "statcan.gc.ca", "novascotia.ca", "rbc.com"),
            "UK": ("bankofengland.co.uk", "ons.gov.uk"),
            "JP": ("boj.or.jp", "stat.go.jp"),
            "AU": ("rba.gov.au", "abs.gov.au"),
            "EA": ("ecb.europa.eu", "eurostat.ec.europa.eu"),
            "IN": ("rbi.org.in", "mospi.gov.in"),
        }
        pattern = region_patterns.get(region)
        if pattern is None:
            return True
        source = entry.get("source") or {}
        source_title = source.get("title", "") if hasattr(source, "get") else ""
        source_url = source.get("href", "") if hasattr(source, "get") else ""
        text = " ".join(
            (
                entry.get("title") or "",
                entry.get("summary") or "",
                source_title,
                source_url,
            )
        ).lower()
        region_matches = bool(re.search(pattern, text)) or any(
            host in source_url.lower() for host in trusted_hosts.get(region, ())
        )
        if not region_matches:
            return False

        is_inflation_question = (
            str(metadata.get("subdomain", "")).lower() == "inflation"
            or "inflation" in str(self.question.question_text).lower()
        )
        if not is_inflation_question:
            return True
        topic_pattern = (
            r"\binflation\b|\bcpi(?:-u)?\b|consumer price|\bprices?\b|"
            r"interest rate|central bank|bank of (?:canada|england|japan)|"
            r"reserve bank|\bhicp\b"
        )
        return bool(re.search(topic_pattern, text))

    def _search_gdelt(self, query: str, page: int) -> List[Dict[str, Any]]:
        """Search GDELT's dated news index with the cutoff as the hard upper bound."""
        page_size = 10
        target_page = max(1, page)
        max_records = min(250, page_size * target_page)

        after_match = re.search(r"\bafter:(\d{4}-\d{2}-\d{2})\b", query)
        before_match = re.search(r"\bbefore:(\d{4}-\d{2}-\d{2})\b", query)
        start_date = after_match.group(1) if after_match else None
        end_date = before_match.group(1) if before_match else None
        if self.question is not None:
            # ``estimated_start_time`` is when the question became forecastable,
            # not the earliest admissible evidence date.  Leave GDELT's lower
            # bound open unless the caller deliberately supplied ``after:``.
            end_date = end_date or self.question.resolution_date.date().isoformat()
        if not end_date:
            raise ValueError("GDELT search requires an evidence cutoff")

        lowered = query.lower()
        if "inflation" in lowered or re.search(r"\bcpi(?:-u)?\b", lowered):
            gdelt_query = '(inflation OR "consumer price index")'
            region = self._finance_region_code()
            source_country = {
                "US": "US",
                "CA": "CA",
                "UK": "UK",
                "JP": "JA",
                "AU": "AS",
                "IN": "IN",
            }.get(region)
            if source_country:
                gdelt_query += f" sourcecountry:{source_country}"
            elif region == "EA":
                gdelt_query += " Europe"

            factor_terms = []
            for factor in (
                "energy",
                "food",
                "shelter",
                "rent",
                "wage",
                "labor",
                "tariff",
                "supply chain",
                "federal reserve",
            ):
                if factor in lowered:
                    factor_terms.append(
                        f'"{factor}"' if " " in factor else factor
                    )
            if factor_terms:
                gdelt_query += f" ({' OR '.join(factor_terms)})"
        else:
            cleaned = re.sub(
                r"\b(?:after|before):\d{4}-\d{2}-\d{2}\b", "", query
            )
            words = re.findall(r"[A-Za-z][A-Za-z0-9]{3,}", cleaned)
            stop_words = {
                "after",
                "before",
                "about",
                "news",
                "report",
                "forecast",
                "January",
                "February",
                "March",
                "April",
                "2025",
                "2026",
            }
            terms = [word for word in words if word not in stop_words][:8]
            if not terms:
                raise ValueError("GDELT query has no searchable terms")
            gdelt_query = " ".join(terms)

        params = {
            "query": gdelt_query,
            "mode": "ArtList",
            "maxrecords": max_records,
            "enddatetime": end_date.replace("-", "") + "235959",
            "format": "json",
            "sort": "HybridRel",
        }
        if start_date:
            params["startdatetime"] = start_date.replace("-", "") + "000000"
        articles = self._request_gdelt(params)
        if not isinstance(articles, list):
            raise ValueError("GDELT response did not contain an article list")

        results = []
        for article in articles:
            seen_date = article.get("seendate")
            published_date = None
            if seen_date:
                try:
                    published_date = datetime.strptime(
                        seen_date, "%Y%m%dT%H%M%SZ"
                    ).isoformat() + "Z"
                except ValueError:
                    published_date = None
            results.append(
                {
                    "title": article.get("title") or "No title",
                    "link": article.get("url") or "",
                    "description": "",
                    "publishedDate": published_date,
                }
            )

        start = (target_page - 1) * page_size
        return results[start : start + page_size]

    @staticmethod
    def _request_gdelt(params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Call GDELT with process-wide throttling, retry, and response caching."""
        global _GDELT_LAST_REQUEST_AT

        cache_key = json.dumps(params, sort_keys=True)
        with _GDELT_LOCK:
            cached = _GDELT_CACHE.get(cache_key)
            if cached is not None:
                return [dict(article) for article in cached]

            last_error: Optional[Exception] = None
            for attempt in range(GDELT_MAX_RETRIES):
                elapsed = time.monotonic() - _GDELT_LAST_REQUEST_AT
                wait_seconds = GDELT_MIN_INTERVAL_SECONDS - elapsed
                if wait_seconds > 0:
                    time.sleep(wait_seconds)

                try:
                    response = httpx.get(
                        "https://api.gdeltproject.org/api/v2/doc/doc",
                        params=params,
                        timeout=30.0,
                        follow_redirects=True,
                    )
                    _GDELT_LAST_REQUEST_AT = time.monotonic()
                    if response.status_code == 429 and attempt + 1 < GDELT_MAX_RETRIES:
                        continue
                    response.raise_for_status()
                    payload = response.json()
                    articles = payload.get("articles", [])
                    if not isinstance(articles, list):
                        raise ValueError(
                            "GDELT response did not contain an article list"
                        )
                    _GDELT_CACHE[cache_key] = [dict(article) for article in articles]
                    return [dict(article) for article in articles]
                except (httpx.HTTPError, ValueError) as exc:
                    last_error = exc
                    if attempt + 1 >= GDELT_MAX_RETRIES:
                        raise

            if last_error is not None:
                raise last_error
            return []

    def _search_ddgs(self, query: str, page: int) -> List[Dict[str, Any]]:
        """Use the ddgs multi-backend client as a structured fallback."""
        from ddgs import DDGS

        page_size = 10
        target_page = max(1, page)
        backend = os.getenv("DDGS_BACKEND", "bing")
        max_results = page_size * target_page
        variants = self._ddgs_query_variants(query)
        per_variant_results: List[List[Dict[str, Any]]] = []
        last_error: Optional[Exception] = None
        for variant in variants:
            try:
                variant_results = DDGS().text(
                    variant,
                    max_results=max_results,
                    backend=backend,
                )
                per_variant_results.append(variant_results or [])
            except Exception as exc:
                logger.warning(
                    f"DDGS query variation failed ({type(exc).__name__}): {variant}"
                )
                last_error = exc

        results = []
        seen_urls = set()
        for result_index in range(max_results):
            for variant_results in per_variant_results:
                if result_index >= len(variant_results):
                    continue
                result = variant_results[result_index]
                url = result.get("href") or result.get("link") or ""
                if not url or url in seen_urls:
                    continue
                seen_urls.add(url)
                results.append(result)
                if len(results) >= max_results:
                    break
            if len(results) >= max_results:
                break

        if not results and last_error is not None:
            raise last_error
        start = (target_page - 1) * page_size
        return results[start : start + page_size]

    def _ddgs_query_variants(self, query: str) -> List[str]:
        """Build deterministic finance-search variations without changing prompts."""
        cleaned = re.sub(
            r"\b(?:after|before):\d{4}-\d{2}-\d{2}\b", "", query
        )
        cleaned = " ".join(cleaned.split())
        if self._query_mode() == "original" or self.domain != "finance":
            return [cleaned]

        variants = [cleaned]
        lowered = cleaned.lower()
        metadata = self._finance_metadata()

        entity = str(metadata.get("entity") or "").strip()
        target_period = str(metadata.get("target_period") or "").strip()
        signal_candidates = metadata.get("candidate_outcome_relevant_signals", [])
        if isinstance(signal_candidates, list):
            for signal_item in signal_candidates[:3]:
                if not isinstance(signal_item, dict):
                    continue
                signal = str(signal_item.get("signal") or "").strip()
                if signal:
                    variants.append(
                        " ".join(part for part in (entity, signal, target_period) if part)
                    )

        if "inflation" in lowered or re.search(r"\bcpi(?:-u)?\b", lowered):
            region = self._finance_region_code()
            country = {
                "US": "United States",
                "CA": "Canada",
                "UK": "United Kingdom",
                "JP": "Japan",
                "AU": "Australia",
                "EA": "Euro area",
                "IN": "India",
            }.get(region, "")
            target_period = metadata.get("target_period") or "2026"
            subject = " ".join(
                part for part in (country, str(target_period)) if part
            )
            variants.extend(
                [
                    f"{subject} inflation outlook",
                    f"{subject} consumer price index forecast",
                    f"{subject} inflation energy food shelter wages",
                    f"{subject} inflation preview analysis",
                ]
            )

        unique_variants = []
        for variant in variants:
            normalized = " ".join(variant.split())
            if normalized and normalized not in unique_variants:
                unique_variants.append(normalized)
        return unique_variants

    def _search_duckduckgo_page(
        self, query: str, page: int
    ) -> List[Dict[str, Any]]:
        """Fetch a real DuckDuckGo Lite result page instead of repeating page 1."""
        import requests

        headers = {"User-Agent": "Mozilla/5.0"}
        with requests.Session() as session:
            response = session.get(
                "https://lite.duckduckgo.com/lite/",
                params={"q": query},
                headers=headers,
                timeout=30,
            )
            response.raise_for_status()

            for _ in range(1, page):
                next_form = _DuckDuckGoNextFormParser()
                next_form.feed(response.text)
                if not next_form.fields:
                    return []
                response = session.post(
                    "https://lite.duckduckgo.com/lite/",
                    data=next_form.fields,
                    headers={**headers, "Referer": response.url},
                    timeout=30,
                )
                response.raise_for_status()

        parser = self.fallback_tool._create_duckduckgo_parser()
        parser.feed(response.text)
        return parser.results

    def _get_structured_results(
        self,
        query: str,
        categories: Optional[str] = None,
        language: Optional[str] = None,
        page: Optional[int] = 1,
        provider: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get structured search results before markdown formatting.

        This method provides programmatic access to search result metadata,
        including publishedDate when available from SearXNG.

        Args:
            query: The search query string
            categories: Optional categories for SearXNG
            language: Optional language code for SearXNG
            page: Optional page number for SearXNG

        Returns:
            List of result dicts with keys: title, url, content, engines, publishedDate
            publishedDate may be None if not available
            Empty list if search fails or using fallback tool
        """
        if provider and provider != "auto":
            return self._get_fallback_structured_results(
                query, page=page, provider=provider
            )
        if not self.use_searxng:
            return self._get_fallback_structured_results(
                query, page=page, provider=provider
            )

        try:
            params = {
                "q": query,
                "format": "json",
                "page": page or 1,
            }

            # Add optional parameters if provided
            if categories:
                params["categories"] = categories
            if language:
                params["language"] = language

            response = self.client.get("/search", params=params)

            # If JSON format is forbidden (403), fall back
            if response.status_code == 403:
                logger.warning(
                    "SearXNG JSON format restricted, no structured results available"
                )
                return []

            response.raise_for_status()

            # Parse and return structured results
            try:
                data = json.loads(response.text)
                results = data.get("results", [])

                # Build structured result list
                structured = []
                for result in results:
                    structured.append(
                        {
                            "title": result.get("title", "No title"),
                            "url": result.get("url", ""),
                            "content": result.get(
                                "content", "No description available"
                            ),
                            "engines": result.get("engines", []),
                            "publishedDate": result.get("publishedDate", None),
                        }
                    )
                return structured
            except json.JSONDecodeError:
                logger.error("Failed to parse SearXNG JSON response")
                return []

        except httpx.HTTPStatusError as e:
            logger.error(f"SearXNG returned status {e.response.status_code}")
            return []
        except httpx.RequestError as e:
            logger.error(
                f"Failed to connect to SearXNG at {self.searxng_base_url}: {str(e)}"
            )
            return []
        except Exception as e:
            logger.error(f"Error querying SearXNG: {str(e)}")
            return []

    def _search_with_searxng(
        self,
        query: str,
        categories: Optional[str] = None,
        language: Optional[str] = None,
        page: Optional[int] = 1,
    ) -> str:
        """
        Perform a search using SearXNG instance.

        Args:
            query: The search query string
            categories: Optional categories to search
            language: Optional language code
            page: Optional page number

        Returns:
            Formatted markdown string with search results (+ collection summary if auto-collect enabled)
        """
        # Get structured results (DRY: reuse query logic)
        structured_results = self._get_structured_results(
            query, categories, language, page
        )

        # If no results (error or fallback), use fallback tool
        if not structured_results:
            if hasattr(self, "fallback_tool"):
                return self.fallback_tool.forward(query=query)
            else:
                self.fallback_tool = SmolWebSearchTool()
                return self.fallback_tool.forward(query=query)

        # Format structured results as markdown
        search_results = self._format_search_results_from_list(
            query, structured_results
        )

        # If auto-collect disabled, return search results only
        if not self.auto_collect_enabled:
            return search_results

        # Auto-collect eligible articles
        collection_summary = self._auto_collect_articles(
            structured_results, query=query
        )

        # Append collection summary to search results
        return f"{search_results}\n\n{collection_summary}"

    def _format_search_results_from_list(
        self, query: str, results: List[Dict[str, Any]]
    ) -> str:
        """
        Format structured search results into a readable markdown string.

        Args:
            query: The search query
            results: List of structured result dicts

        Returns:
            Formatted markdown string with search results
        """
        if not results:
            return f"No results found for query: '{query}'"

        # Build formatted output
        output = [f"# Search Results for: {query}\n"]

        # Limit to top 10 results for readability
        for i, result in enumerate(results[:10], 1):
            title = result.get("title", "No title")
            url = result.get("url", "")
            content = result.get("content", "No description available")
            engines = result.get("engines", [])
            published_date = result.get("publishedDate", None)

            # Clean up content - remove extra whitespace and limit length
            content = " ".join(content.split())
            if len(content) > 200:
                content = content[:197] + "..."

            output.append(f"## {i}. {title}")
            output.append(f"**URL:** {url}")
            output.append(f"**Description:** {content}")
            if published_date:
                output.append(f"**Published Date:** {published_date}")

            if engines:
                output.append(f"**Sources:** {', '.join(engines)}")
            output.append("")  # Empty line for spacing

        return "\n".join(output)

    def _format_search_results(self, data: dict) -> str:
        """
        Format SearXNG JSON results into a readable markdown string.

        Args:
            data: Parsed JSON response from SearXNG

        Returns:
            Formatted markdown string with search results
        """
        query = data.get("query", "")
        results = data.get("results", [])

        # Convert to structured list format and reuse formatting logic (DRY)
        structured_results = []
        for result in results:
            structured_results.append(
                {
                    "title": result.get("title", "No title"),
                    "url": result.get("url", ""),
                    "content": result.get("content", "No description available"),
                    "engines": result.get("engines", []),
                    "publishedDate": result.get("publishedDate", None),
                }
            )

        return self._format_search_results_from_list(query, structured_results)

    @staticmethod
    def _relevance_tokens(value: str) -> set[str]:
        tokens = {
            token
            for token in re.findall(r"[a-z][a-z0-9-]{2,}", value.lower())
            if token not in _SEARCH_RELEVANCE_STOPWORDS
            and not re.fullmatch(r"20\d{2}", token)
        }
        return tokens

    @classmethod
    def _is_relevant_to_query(cls, result: Dict[str, Any], query: str) -> bool:
        """Reject obvious search-engine drift before fetching and persistence."""
        query_tokens = cls._relevance_tokens(query)
        if not query_tokens:
            return True
        result_text = " ".join(
            str(result.get(field) or "")
            for field in ("title", "content", "description")
        )
        overlap = query_tokens & cls._relevance_tokens(result_text)
        return len(overlap) >= 2 or bool(overlap & _SEARCH_RELEVANCE_SIGNAL_TERMS)

    def _auto_collect_articles(
        self,
        structured_results: List[Dict[str, Any]],
        query: Optional[str] = None,
    ) -> str:
        """Automatically collect articles that meet temporal criteria.

        Args:
            structured_results: List of structured search result dicts

        Returns:
            Brief summary of collection activity
        """
        # Safety check: ensure resolution date is set
        if not self.question_resolution_date:
            logger.warning("Auto-collect skipped: question_resolution_date is not set")
            return "\n---\n**Auto-collected 0 article(s)** (auto-collect disabled: no resolution date)"

        import concurrent.futures
        from urllib.parse import urlparse

        from src.tools.collectors.article_collector import ArticleCollectorTool

        collected = []
        skipped = {
            "after_resolution": 0,
            "irrelevant": 0,
            "error": 0,
            "timeout": 0,
        }

        eligible_items = []

        for result in structured_results:
            if len(eligible_items) >= self.max_auto_collect:
                break

            url, title, published_date_str = (
                result.get("url"),
                result.get("title", ""),
                result.get("publishedDate"),
            )

            if not url:
                skipped["error"] += 1
                continue

            if query and not self._is_relevant_to_query(result, query):
                skipped["irrelevant"] += 1
                continue

            try:
                published_date = ArticleCollectorTool._parse_date_candidate(
                    published_date_str
                )

                if published_date is not None:
                    # Normalize both datetimes for comparison.
                    pub_dt = published_date.replace(tzinfo=None)
                    res_dt = self.question_resolution_date.replace(tzinfo=None)

                    if pub_dt >= res_dt:
                        skipped["after_resolution"] += 1
                        continue

                # Collect article
                hostname = urlparse(url).netloc.removeprefix("www.")
                source = result.get("source") or hostname or "Unknown"

                eligible_items.append({
                    "url": url,
                    "title": title,
                    "source": source,
                    "published_date": published_date.isoformat()
                    if published_date
                    else None,
                    "domain": self.domain,
                    "author": None,
                })

            except Exception as e:
                logger.warning(f"Auto-collect validation skipped {url}: {type(e).__name__}: {e}")
                skipped["error"] += 1

        if eligible_items:
            def fetch_article(item):
                result = self.article_collector.forward(
                    url=item["url"],
                    title=item["title"],
                    source=item["source"],
                    published_date=item["published_date"],
                    domain=item["domain"],
                    author=item["author"],
                    timeout=ARTICLE_COLLECT_TIMEOUT_SECONDS,
                )
                if getattr(result, "id", None) == "error":
                    raise RuntimeError(getattr(result, "status", "collection failed"))
                return item["title"]

            # Execute concurrently with a timeout consistent with ArticleCollector fetch timeout.
            worker_count = min(len(eligible_items), _max_fetch_workers())
            executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=worker_count
            )
            try:
                future_to_item = {executor.submit(fetch_article, item): item for item in eligible_items}

                done, not_done = concurrent.futures.wait(
                    future_to_item.keys(), timeout=float(ARTICLE_COLLECT_TIMEOUT_SECONDS)
                )

                for future in done:
                    item = future_to_item[future]
                    try:
                        title = future.result()
                        collected.append(title[:30] + "..." if len(title) > 30 else title)
                    except Exception as e:
                        logger.warning(f"Auto-collect failed for {item['url']}: {type(e).__name__}: {e}")
                        skipped["error"] += 1

                for future in not_done:
                    item = future_to_item[future]
                    future.cancel()
                    logger.warning(f"Auto-collect timed out for {item['url']}")
                    skipped["timeout"] += 1
            finally:
                # shutdown(wait=True), including the ThreadPoolExecutor context
                # manager default, can block forever on a wedged browser fetch.
                # WebFetchTool enforces its own total deadline; this outer layer
                # must return immediately once its collection budget expires.
                executor.shutdown(wait=False, cancel_futures=True)

        # Build concise summary
        total_skipped = sum(skipped.values())
        summary = f"\n---\n**Auto-collected {len(collected)} article(s):**"
        if collected:
            summary += "\n" + "\n".join([f"- {t}" for t in collected])

        if total_skipped > 0:
            details = [
                f"{v} {k.replace('_', ' ')}" for k, v in skipped.items() if v > 0
            ]
            summary += f"\n\n(Skipped {total_skipped}: {', '.join(details)}. **Please manually collect relevant articles if they were skipped.**)"

        return summary

    def __del__(self):
        """Clean up HTTP client when tool is destroyed."""
        if hasattr(self, "client") and self.client:
            try:
                self.client.close()
            except Exception:
                pass
