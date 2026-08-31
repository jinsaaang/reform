"""Article collection tool using web search for scraping."""

import hashlib
import re
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Dict, Optional, Tuple
from urllib.parse import urlparse

from src.config import get_config
from src.domain.models import Article, Domain
from src.utils.logging import logger
from src.utils.enums import enum_to_list, parse_domain
from src.utils.date_utils import ensure_timezone_aware
from src.domain.models.id_generator import generate_article_id
from src.tools.base.schema_helper import pydantic_to_output_schema
from src.tools.collectors.web_fetch import WebFetchTool
from src.tools.base.base import CollectorAwareTool
from src.tools.base.output_models import ArticleOutput


class ArticleCollectorTool(CollectorAwareTool[Article]):
    """Fetches and stores article data from URLs into Article objects.

    This tool helps the agent:
    1. Internally fetch full article content from a URL (using WebFetchTool)
    2. Convert content into structured Article format
    3. Generate unique article IDs
    4. Handle deduplication via content hashing
    5. Calculate metadata (word count, reading time, etc.)

    IMPORTANT: This tool ONLY needs the URL and metadata.
    The agent should use web_search to find article URLs, then pass ONLY the URL
    to this tool. This tool will internally fetch the full content, avoiding
    expensive token usage from passing large article text through the LLM.
    """

    name = "article_collector"
    description = f"""Fetches and stores article data from a URL to database.

    Use this tool AFTER you've found article URLs using web_search.
    IMPORTANT: make sure the published_date is correct.
    This tool will internally fetch the full article content to save tokens.
    """

    # Auto-generate inputs from Enum classes (single source of truth)
    inputs = {
        "url": {
            "type": "string",
            "description": "Source URL to fetch the article from",
        },
        "title": {
            "type": "string",
            "description": "Article headline/title from search results",
        },
        "source": {"type": "string", "description": "Publication name"},
        "domain": {
            "type": "string",
            "description": f"Domain category - one of: {', '.join(enum_to_list(Domain))}",
            "enum": enum_to_list(Domain),
            "nullable": True,
        },
        "published_date": {
            "type": "string",
            "description": (
                "Optional publication-date hint. The collector verifies it against "
                "page metadata, URL, and article content."
            ),
            "nullable": True,
        },
        "author": {"type": "string", "description": "Author name", "nullable": True},
        "timeout": {
            "type": "integer",
            "description": "Web fetch timeout in seconds (default: 15)",
            "nullable": True,
        },
    }
    output_type = "object"
    output_schema = pydantic_to_output_schema(ArticleOutput)

    def __init__(
        self,
        db=None,
        db_path: str = None,
        collector=None,
        question_id: Optional[str] = None,
    ):
        """Initialize the article collector.

        Args:
            db: Optional Database instance for cross-run deduplication
            db_path: Optional path to database file (creates new Database with schema if provided)
            collector: Optional ResultCollector[Article] for storing results.
                      If provided, articles are added to the collector instead of internal storage.
            question_id: Question ID for provenance tracking (sets collected_for_question_id)
        """
        super().__init__(collector)
        self.config = None
        self.seen_hashes = set()  # For in-memory deduplication within this run
        self.web_visitor = WebFetchTool()  # Internal tool for fetching content
        self.question_id = question_id  # Provenance context

        logger.info(
            f"ArticleCollectorTool initialized with collector: {collector is not None}, question_id: {question_id}"
        )

        # Database for cross-run deduplication (optional)
        self.db = None
        if db:
            self.db = db
        elif db_path:
            # Lazy import to avoid circular dependency
            from src.core.database import GenericDatabase

            self.db = GenericDatabase(db_path)
            # Ensure schema is initialized
            self.db.create_table(Article)

    def setup(self):
        """Load configuration (called on first use)."""
        if self.config is None:
            self.config = get_config()

    def forward(
        self,
        url: str,
        title: str,
        source: str,
        published_date: Optional[str] = None,
        domain: str = "general",
        author: Optional[str] = None,
        timeout: int = 15,
    ) -> ArticleOutput:
        """Fetch article content from URL and store as structured Article.

        Args:
            url: Source URL to fetch article from
            title: Article headline from search results
            source: Publication name
            domain: Article domain category (string, will be converted to enum)
            published_date: Publication date (ISO format with time zone)
            author: Optional author name
            timeout: Web fetch timeout in seconds

        Returns:
            ArticleOutput: Pydantic model with article metadata and status
        """
        # STAGE 1: Fast URL-based deduplication (before fetching content)
        # This is the most efficient check - prevents unnecessary web scraping
        if self.db:
            # Normalize URL for better matching (remove trailing slash, fragments)
            normalized_url = self._normalize_url(url)

            # Use GenericDatabase's get_many with filter
            existing_articles = self.db.get_many(
                Article, filters={"url": normalized_url}
            )
            if existing_articles:
                existing = existing_articles[0]

                cutoff = self._get_question_cutoff()
                if (
                    cutoff is not None
                    and existing.published_date is not None
                    and ensure_timezone_aware(existing.published_date) >= cutoff
                ):
                    return ArticleOutput(
                        id="error",
                        title=existing.title,
                        url=existing.url,
                        source=existing.source,
                        status=(
                            "error: existing article publication date is at or after "
                            f"the evidence cutoff ({existing.published_date.isoformat()} "
                            f">= {cutoff.isoformat()})"
                        ),
                        published_date=existing.published_date.isoformat(),
                        warnings=["existing_article_at_or_after_evidence_cutoff"],
                    )

                # Preserve the original owner while recording many-to-many
                # question provenance for report families that share evidence.
                if self.question_id:
                    metadata = dict(existing.metadata or {})
                    related_ids = set(metadata.get("related_question_ids", []))
                    if existing.collected_for_question_id:
                        related_ids.add(existing.collected_for_question_id)
                    related_ids.add(self.question_id)
                    metadata["related_question_ids"] = sorted(related_ids)
                    existing.metadata = metadata
                    if existing.collected_for_question_id is None:
                        existing.collected_for_question_id = self.question_id
                    self.db.save(Article, existing)
                    logger.debug(
                        f"Updated question provenance on existing article {existing.id}"
                    )

                # Add to collector even if duplicate (for current pipeline run)
                # Note: Check 'is not None' because ResultCollector.__bool__ returns False when empty
                if self.collector is not None:
                    self.collector.add(existing)
                    logger.debug(
                        f"Added existing article {existing.id} to collector (duplicate URL, total: {self.collector.count()})"
                    )
                else:
                    self._fallback_items.append(existing)
                    logger.debug(
                        f"Added existing article {existing.id} to internal list (duplicate URL, total: {len(self._fallback_items)})"
                    )

                return ArticleOutput(
                    id=existing.id,
                    title=existing.title,
                    url=existing.url,
                    source=existing.source,
                    status="existing",
                    word_count=existing.word_count,
                    published_date=existing.published_date.isoformat()
                    if existing.published_date
                    else None,
                )

        # STAGE 2: Fetch content (only if URL not found)
        # This avoids passing large content through the LLM
        try:
            web_output = self.web_visitor.forward(url, timeout=timeout)

            # WebFetchTool returns WebFetchOutput object now
            if not web_output.success or not web_output.content:
                # Return error as ArticleOutput
                error_msg = web_output.error or "Empty content fetched"
                return ArticleOutput(
                    id="error",
                    title=title,
                    url=url,
                    status=f"error: {error_msg}",
                )

            content = web_output.content
            if len(content.strip()) < 100:
                return ArticleOutput(
                    id="error",
                    title=title,
                    url=url,
                    status=f"error: Content too short ({len(content)} chars)",
                )
        except Exception as e:
            return ArticleOutput(
                id="error",
                title=title,
                url=url,
                status=f"error: {str(e)}",
            )

        pub_date, date_source, date_warning = self._resolve_published_date(
            url=url,
            content=content,
            metadata=web_output.metadata or {},
            hinted_date=published_date,
        )
        if pub_date is None:
            return ArticleOutput(
                id="error",
                title=title,
                url=url,
                source=source,
                status=(
                    "error: Could not determine publication date from page "
                    "metadata, URL, article content, or supplied hint"
                ),
                warnings=["published_date_unresolved"],
            )

        # Forecast evidence has one hard temporal rule: it must predate the
        # information cutoff.  ``estimated_start_time`` describes when the
        # target question became forecastable; it is not a lower bound on
        # admissible evidence.  Prior filings and historical base rates are
        # often the most useful forecast inputs.
        time_window_validation = None
        if self.question_id and self.db:
            from src.domain.models import Question
            from src.utils.date_utils import validate_date_against_question_window

            question = self.db.get(Question, self.question_id)
            if question:
                evidence_cutoff = ensure_timezone_aware(question.resolution_date)
                if pub_date >= evidence_cutoff:
                    return ArticleOutput(
                        id="error",
                        title=title,
                        url=url,
                        source=source,
                        status=(
                            "error: publication date is at or after the evidence "
                            f"cutoff ({pub_date.isoformat()} >= "
                            f"{evidence_cutoff.isoformat()})"
                        ),
                        published_date=pub_date.isoformat(),
                        published_date_source=date_source,
                        warnings=["at_or_after_evidence_cutoff"],
                    )
                time_window_validation = validate_date_against_question_window(
                    date=pub_date,
                    question_start_time=None,
                    question_resolution_date=question.resolution_date,
                    entity_type="Article",
                )

        # STAGE 3: Content hash deduplication (catches syndicated/republished articles)
        # Check if we've already seen this content (in-memory for current run)
        content_hash = self._compute_content_hash(content)
        if content_hash in self.seen_hashes:
            logger.debug(
                f"Skipping duplicate content hash: {content_hash} for URL: {url}"
            )
            return ArticleOutput(
                id="duplicate",
                title=title,
                url=url,
                status="duplicate: Same content already exists",
            )

        # Also check database for content hash (cross-run syndication detection)
        if self.db:
            # Query by content_hash if your Article model has this field
            # For now, we'll skip this to avoid performance issues
            # In production, you might want to add a content_hash field and index
            pass

        self.seen_hashes.add(content_hash)

        # Validate and convert domain
        domain_enum = parse_domain(domain, default=Domain.GENERAL)
        if domain_enum is None:
            domain_enum = Domain.GENERAL

        # Generate unique ID
        article_id = generate_article_id(domain_enum, pub_date, len(self.seen_hashes))

        # Extract domain from URL if not provided
        parsed_url = urlparse(url)
        source_domain = parsed_url.netloc

        # Store normalized URL for consistency
        normalized_url = self._normalize_url(url)

        # Build metadata with provenance info
        metadata = {
            "published_date_source": date_source,
            "content_hash": content_hash,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
        }
        modified_keys = {"datemodified", "date_modified", "article:modified_time"}
        for key, value in self._flatten_metadata(web_output.metadata or {}):
            normalized_key = key.lower().replace("-", "_")
            if normalized_key in modified_keys or "datemodified" in normalized_key:
                metadata["page_modified_at"] = str(value)
                break
        if published_date:
            metadata["published_date_hint"] = published_date
        if date_warning:
            metadata["published_date_warning"] = date_warning
        if self.question_id:
            metadata["evidence_type"] = "hindsight"
            metadata["related_question_ids"] = [self.question_id]

        # Create Article object
        article = Article(
            id=article_id,
            title=title,
            content=content,
            url=normalized_url,  # Use normalized URL for consistency
            source=source,
            author=author or "Unknown",
            published_date=pub_date,
            domain=domain_enum,
            tags=[domain_enum.value, source_domain],
            event_ids=[],  # Initialize with empty list (will be populated later in pipeline)
            is_synthetic=False,
            language="en",
            collected_for_question_id=self.question_id,  # Provenance tracking
            metadata=metadata,
        )

        # Calculate metadata
        article.word_count = len(article.content.split())
        article.reading_time_minutes = max(1, article.word_count // 200)

        # Store article using unified collector interface
        self.store_result(article, context=f"Article {article.id}")

        # Persist to database if available
        if self.db:
            self.db.save(Article, article)
            logger.debug(f"Article {article.id} persisted to database")

        # Return ArticleOutput Pydantic model with summary metadata
        # Note: We don't include full content to save tokens
        warnings = []
        if date_warning:
            warnings.append(date_warning)
        if time_window_validation:
            warnings.extend(time_window_validation.get("warnings", []))

        return ArticleOutput(
            id=article.id,
            title=article.title,
            url=article.url,
            source=article.source,
            # Warnings are carried separately.  Keep the status in the
            # documented created/updated/existing vocabulary so agents can
            # reliably count successful collections.
            status="created",
            word_count=article.word_count,
            published_date=article.published_date.isoformat(),
            published_date_source=date_source,
            warnings=warnings or None,
        )

    @classmethod
    def _resolve_published_date(
        cls,
        url: str,
        content: str,
        metadata: Dict[str, Any],
        hinted_date: Optional[str],
    ) -> Tuple[Optional[datetime], Optional[str], Optional[str]]:
        """Resolve publication date from page-owned evidence before LLM hints."""
        candidates = []

        metadata_keys = {
            "datepublished",
            "date_published",
            "article:published_time",
            "article_published_time",
            "citation_publication_date",
            "dc_date",
            "og_published_time",
            "prism_publicationdate",
            "prism_publication_date",
            "published_time",
            "publish_date",
            "published",
        }
        for key, value in cls._flatten_metadata(metadata):
            normalized_key = re.sub(
                r"[^a-z0-9]+",
                "_",
                key.lower(),
            ).strip("_")
            if normalized_key in metadata_keys or "datepublished" in normalized_key:
                candidates.append((value, f"metadata:{key}"))

        hint_parsed = cls._parse_date_candidate(hinted_date)
        parsed_url = urlparse(url)
        is_sec_filing = (
            parsed_url.netloc.lower().endswith("sec.gov")
            and "/archives/edgar/data/" in parsed_url.path.lower()
        )
        # EDGAR exhibit filenames commonly contain the financial period end
        # (for example ``intc-20251227.htm``), not the filing/publication date.
        # A curated filing-availability hint is therefore more reliable than
        # URL/content date heuristics for this specific source type.
        if is_sec_filing and hint_parsed:
            candidates.append((hinted_date, "verified_filing_hint"))

        url_date = cls._extract_date_from_url(url)
        if url_date:
            candidates.append((url_date, "url"))

        content_date = cls._extract_date_from_content(content)
        if content_date:
            candidates.append((content_date, "content"))

        if hinted_date and not (is_sec_filing and hint_parsed):
            candidates.append((hinted_date, "agent_hint"))

        resolved = None
        resolved_source = None
        for value, source in candidates:
            parsed = cls._parse_date_candidate(value)
            if parsed is not None:
                resolved = parsed
                resolved_source = source
                break

        if resolved is None:
            return None, None, None

        warning = None
        if hint_parsed and resolved_source != "agent_hint":
            if hint_parsed.date() != resolved.date():
                warning = (
                    "Published-date hint "
                    f"{hint_parsed.date().isoformat()} replaced by "
                    f"{resolved.date().isoformat()} from {resolved_source}"
                )

        return resolved, resolved_source, warning

    def _get_question_cutoff(self) -> Optional[datetime]:
        """Return the active question cutoff used for every create/reuse path."""
        if not self.db or not self.question_id:
            return None
        from src.domain.models import Question

        question = self.db.get(Question, self.question_id)
        if not question:
            return None
        return ensure_timezone_aware(question.resolution_date)

    @staticmethod
    def _flatten_metadata(metadata: Dict[str, Any]):
        """Yield scalar metadata entries from shallow or nested mappings."""
        stack = [("", metadata)]
        while stack:
            prefix, value = stack.pop()
            if isinstance(value, dict):
                for key, nested in value.items():
                    name = f"{prefix}.{key}" if prefix else str(key)
                    stack.append((name, nested))
            elif isinstance(value, (str, datetime)):
                yield prefix, value

    @classmethod
    def _parse_date_candidate(cls, value: Any) -> Optional[datetime]:
        if isinstance(value, datetime):
            return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        if not isinstance(value, str) or not value.strip():
            return None

        raw = value.strip()
        try:
            parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
        except ValueError:
            pass

        try:
            parsed = parsedate_to_datetime(raw)
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
        except (TypeError, ValueError, OverflowError):
            pass

        normalized = re.sub(r"\b([A-Za-z]{3})\.", r"\1", raw)
        formats = (
            "%B %d, %Y",
            "%b %d, %Y",
            "%d %B %Y",
            "%d %b %Y",
            "%Y/%m/%d",
            "%Y-%m-%d",
        )
        for fmt in formats:
            try:
                return datetime.strptime(normalized, fmt).replace(tzinfo=timezone.utc)
            except ValueError:
                continue
        return None

    @classmethod
    def _extract_date_from_url(cls, url: str) -> Optional[str]:
        patterns = (
            r"(?<!\d)(20\d{2})[/-](0?[1-9]|1[0-2])[/-](0?[1-9]|[12]\d|3[01])(?!\d)",
            r"(?<!\d)(20\d{2})(0[1-9]|1[0-2])([0-2]\d|3[01])(?!\d)",
        )
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                year, month, day = match.groups()
                candidate = f"{year}-{int(month):02d}-{int(day):02d}"
                if cls._parse_date_candidate(candidate):
                    return candidate
        return None

    @classmethod
    def _extract_date_from_content(cls, content: str) -> Optional[str]:
        head = content[:5_000]
        month_names = (
            "January|February|March|April|May|June|July|August|"
            "September|October|November|December|"
            "Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec"
        )
        date_patterns = (
            rf"(?:{month_names})\.?\s+\d{{1,2}},\s+20\d{{2}}",
            rf"\d{{1,2}}\s+(?:{month_names})\.?\s+20\d{{2}}",
            r"20\d{2}-\d{2}-\d{2}",
        )
        general_publication_markers = (
            r"published(?:\s+(?:on|at))?",
            r"publication\s+date",
            r"posted(?:\s+(?:on|at))?",
            r"updated(?:\s+(?:on|at))?",
            r"filed(?:\s+(?:on|at))?",
            r"release\s+date",
        )
        general_marker = "(?:" + "|".join(general_publication_markers) + ")"
        release_marker = r"(?:press\s+release|news\s+release)"
        date = "(?:" + "|".join(date_patterns) + ")"
        contextual_patterns = (
            rf"{general_marker}[^\n]{{0,80}}?(?P<date>{date})(?!\d)",
            rf"{release_marker}[^\n]{{0,40}}?(?P<date>{date})(?!\d)",
        )
        for pattern in contextual_patterns:
            match = re.search(pattern, head, re.IGNORECASE)
            if match and cls._parse_date_candidate(match.group("date")):
                return match.group("date")

        # Business-wire style pages often put the publication timestamp at the
        # very start of the extracted article, before the headline and without
        # a "published" label.  Restrict this fallback to the opening 24
        # characters so a reporting-period date in the headline/body is not
        # mistaken for publication time.
        for pattern in date_patterns:
            match = re.search(
                rf"^\s*(?P<date>{pattern})(?!\d)", head, re.IGNORECASE
            )
            if match and cls._parse_date_candidate(match.group("date")):
                return match.group("date")
        return None

    def _normalize_url(self, url: str) -> str:
        """Normalize URL for consistent duplicate detection.

        Removes:
        - Trailing slashes
        - URL fragments (#section)
        - Common tracking parameters
        - Converts http to https

        Args:
            url: Raw URL

        Returns:
            Normalized URL
        """
        from urllib.parse import urlparse, urlunparse, parse_qs, urlencode

        parsed = urlparse(url)

        # Convert http to https
        scheme = "https" if parsed.scheme == "http" else parsed.scheme

        # Remove trailing slash from path
        path = parsed.path.rstrip("/")

        # Remove common tracking parameters
        if parsed.query:
            params = parse_qs(parsed.query)
            # Remove common tracking params
            tracking_params = {
                "utm_source",
                "utm_medium",
                "utm_campaign",
                "utm_content",
                "utm_term",
                "fbclid",
                "gclid",
                "ref",
                "source",
            }
            cleaned_params = {
                k: v for k, v in params.items() if k not in tracking_params
            }
            query = urlencode(cleaned_params, doseq=True) if cleaned_params else ""
        else:
            query = ""

        # Reconstruct URL without fragment
        normalized = urlunparse((scheme, parsed.netloc, path, parsed.params, query, ""))

        return normalized

    def _compute_content_hash(self, content: str) -> str:
        """Compute SHA-256 hash of normalized content for deduplication.

        Args:
            content: Article content to hash

        Returns:
            Hexadecimal hash string
        """
        normalized = " ".join(content.lower().split())
        return hashlib.sha256(normalized.encode()).hexdigest()

    def reset_deduplication(self):
        """Reset the deduplication cache."""
        self.seen_hashes.clear()
