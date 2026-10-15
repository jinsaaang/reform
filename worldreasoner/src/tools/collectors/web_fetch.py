"""Advanced web fetching tool using crawl4ai for robust content extraction."""

import asyncio
import concurrent.futures
import json
import os
import threading
import time
from typing import Dict, Any, Optional
from smolagents import Tool
from src.tools.base.schema_helper import pydantic_to_output_schema
from src.tools.base.output_models import WebFetchOutput


class WebFetchTool(Tool):
    """Fetch and extract clean content from web pages using crawl4ai.

    This tool provides advanced web scraping capabilities:
    - JavaScript rendering support
    - Clean markdown extraction
    - Metadata extraction (title, description, etc.)
    - Better handling of dynamic content than simple HTTP requests

    Uses crawl4ai for robust content extraction with proper handling of:
    - SPAs (Single Page Applications)
    - Dynamic content loading
    - Content cleaning and formatting
    """

    name = "web_fetch"
    description = """Fetch and extract clean content from a web page URL.

    This tool uses advanced web scraping to handle modern websites with JavaScript.
    It returns clean markdown content suitable for LLM processing.

    IMPORTANT: this tool doesn't automatically store articles to the database.
    """

    inputs = {
        "url": {"type": "string", "description": "URL to fetch content from"},
        "timeout": {
            "type": "integer",
            "description": "Timeout in seconds (default: 15)",
            "nullable": True,
        },
        "timestamp": {
            "type": "string",
            "description": "Optional timestamp (YYYYMMDDhhmmss) to fetch from Internet Archive",
            "nullable": True,
        },
    }
    output_type = "object"
    output_schema = pydantic_to_output_schema(WebFetchOutput)

    # Evidence agents commonly inspect a URL with web_fetch immediately before
    # passing it to ArticleCollectorTool, which owns another WebFetchTool
    # instance.  Share successful results within the current process so that
    # this normal pipeline sequence does not download/render the page twice.
    _success_cache: Dict[str, WebFetchOutput] = {}
    _failure_cache: Dict[str, tuple[float, WebFetchOutput]] = {}
    _cache_lock = threading.RLock()
    _max_cache_entries = 128
    _failure_cache_ttl_seconds = 300
    # The caller-provided timeout covers page navigation.  Keep a small grace
    # window for browser startup/cleanup, but never let the whole fetch wait
    # forever outside the page-level timeout.
    _total_timeout_grace_seconds = 5.0

    def __init__(self):
        """Initialize the web fetch tool."""
        super().__init__()
        self._crawler = None

    @staticmethod
    def _browser_cpu_args() -> list[str]:
        """Keep one fallback page render from monopolizing the host CPU."""
        try:
            raster_threads = int(
                os.getenv("WEB_FETCH_BROWSER_RASTER_THREADS", "1")
            )
        except ValueError:
            raster_threads = 1
        raster_threads = max(1, min(raster_threads, 2))
        return [
            "--renderer-process-limit=1",
            f"--num-raster-threads={raster_threads}",
        ]

    @classmethod
    def clear_cache(cls) -> None:
        """Clear process-local fetch caches (primarily for tests)."""
        with cls._cache_lock:
            cls._success_cache.clear()
            cls._failure_cache.clear()

    @classmethod
    def _cache_key(cls, url: str, timestamp: Optional[str]) -> str:
        return f"{url.rstrip('/')}|{timestamp or ''}"

    @classmethod
    def _get_cached(
        cls, url: str, timestamp: Optional[str]
    ) -> Optional[WebFetchOutput]:
        key = cls._cache_key(url, timestamp)
        with cls._cache_lock:
            output = cls._success_cache.get(key)
            if output is not None:
                return output.model_copy(deep=True)
            failure = cls._failure_cache.get(key)
            if failure is None:
                return None
            stored_at, output = failure
            if time.monotonic() - stored_at > cls._failure_cache_ttl_seconds:
                del cls._failure_cache[key]
                return None
            return output.model_copy(deep=True)

    @classmethod
    def _cache_output(
        cls, url: str, timestamp: Optional[str], output: WebFetchOutput
    ) -> None:
        key = cls._cache_key(url, timestamp)
        with cls._cache_lock:
            if output.success:
                cls._success_cache[key] = output.model_copy(deep=True)
                while len(cls._success_cache) > cls._max_cache_entries:
                    oldest_key = next(iter(cls._success_cache))
                    del cls._success_cache[oldest_key]
            else:
                cls._failure_cache[key] = (
                    time.monotonic(),
                    output.model_copy(deep=True),
                )
                while len(cls._failure_cache) > cls._max_cache_entries:
                    oldest_key = next(iter(cls._failure_cache))
                    del cls._failure_cache[oldest_key]

    @staticmethod
    def _to_output(result: Dict[str, Any], requested_url: str) -> WebFetchOutput:
        content = result.get("markdown", "") or ""
        if len(content) > 50_000:
            content = content[:50_000] + "\n\n[Content truncated at 50,000 characters]"
        return WebFetchOutput(
            url=result.get("url", requested_url),
            title=result.get("title", ""),
            content=content,
            metadata=result.get("metadata", {}),
            success=result.get("success", False),
            error=result.get("error"),
        )

    async def _fetch_with_deadline(
        self, url: str, timeout: int, timestamp: Optional[str]
    ) -> Dict[str, Any]:
        """Run the complete HTTP/browser path under one hard deadline."""
        total_timeout = max(float(timeout), 0.1) + self._total_timeout_grace_seconds
        try:
            return await asyncio.wait_for(
                self._fetch_async(url, timeout, timestamp),
                timeout=total_timeout,
            )
        except asyncio.TimeoutError:
            return {
                "url": url,
                "success": False,
                "error": f"Fetch exceeded the {total_timeout:g}s total deadline",
            }

    async def _fast_fetch_async(self, url: str, timeout: int = 10) -> Dict[str, Any]:
        """Try a fast fetch with AI-friendly headers.

        Args:
            url: URL to fetch
            timeout: Timeout in seconds

        Returns:
            Dictionary with content if successful, else None
        """
        try:
            import httpx

            headers = {
                "Accept": "text/markdown, text/plain, */*",
                "User-Agent": "MyAI-Agent/1.0 (Hybrid Fetch Controller)",
            }
            async with httpx.AsyncClient(
                timeout=timeout, follow_redirects=True
            ) as client:
                response = await client.get(url, headers=headers)

                # If we got markdown back directly, great!
                content_type = response.headers.get("Content-Type", "").lower()
                if response.status_code == 200 and (
                    "markdown" in content_type or "text/plain" in content_type
                ):
                    return {
                        "url": str(response.url),
                        "title": "",  # Hard to get without parsing
                        "markdown": response.text,
                        "metadata": {"method": "fast_fetch_markdown"},
                        "success": True,
                    }

                if response.status_code == 200 and "html" in content_type:
                    extracted = self._extract_static_html(response.text)
                    if extracted is not None:
                        title, markdown, metadata = extracted
                        metadata["method"] = "fast_fetch_html"
                        return {
                            "url": str(response.url),
                            "title": title,
                            "markdown": markdown,
                            "metadata": metadata,
                            "success": True,
                        }

                return None
        except Exception:
            return None

    @staticmethod
    def _extract_static_html(
        html: str,
    ) -> Optional[tuple[str, str, Dict[str, Any]]]:
        """Extract substantive static HTML before paying for a browser render."""
        from bs4 import BeautifulSoup
        from markdownify import markdownify

        soup = BeautifulSoup(html, "html.parser")
        visible_text = " ".join(soup.stripped_strings)
        blocked_markers = (
            "enable javascript and cookies to continue",
            "verify you are human",
            "checking your browser",
            "attention required! | cloudflare",
            "access denied",
            "consent-page",
        )
        lowered = visible_text.lower()
        if any(marker in lowered for marker in blocked_markers):
            return None

        metadata: Dict[str, Any] = {}
        for tag in soup.find_all("meta"):
            key = (
                tag.get("property")
                or tag.get("name")
                or tag.get("itemprop")
            )
            value = tag.get("content")
            if key and value:
                metadata[str(key)] = str(value)
        for script in soup.find_all(
            "script",
            attrs={"type": "application/ld+json"},
        ):
            try:
                payload = json.loads(script.string or script.get_text() or "")
            except (TypeError, json.JSONDecodeError):
                continue
            stack = [payload]
            while stack:
                item = stack.pop()
                if isinstance(item, list):
                    stack.extend(item)
                    continue
                if not isinstance(item, dict):
                    continue
                for key, value in item.items():
                    if isinstance(value, (dict, list)):
                        stack.append(value)
                    elif key in {
                        "datePublished",
                        "dateCreated",
                        "headline",
                    } and value:
                        metadata.setdefault(key, str(value))

        title = str(
            metadata.get("og:title")
            or metadata.get("twitter:title")
            or metadata.get("headline")
            or (soup.title.string if soup.title and soup.title.string else "")
        ).strip()
        container = (
            soup.find("article")
            or soup.find("main")
            or soup.find(attrs={"role": "main"})
            or soup.body
        )
        if container is None:
            return None
        for tag in container.find_all(
            ["script", "style", "noscript", "svg", "nav", "header", "footer", "form"]
        ):
            tag.decompose()
        clean_text = " ".join(container.stripped_strings)
        if len(clean_text) < 200 or len(clean_text.split()) < 35:
            return None
        clean_markdown = markdownify(str(container), heading_style="ATX").strip()
        if len(clean_markdown) < 200:
            return None
        return title, clean_markdown, metadata

    async def _archive_fetch_async(
        self, url: str, timestamp: Optional[str] = None
    ) -> Optional[str]:
        """Check Internet Archive for a snapshot.

        Args:
            url: URL to fetch
            timestamp: Target timestamp (YYYYMMDDhhmmss). If None, defaults to latest.

        Returns:
            The raw snapshot URL if found, else None
        """
        try:
            import httpx

            # If no timestamp provided, we omit it to get the 'latest'
            query_ts = ""
            if timestamp:
                # Clean timestamp to YYYYMMDD style for API
                query_ts = (
                    timestamp.replace("-", "")
                    .replace(":", "")
                    .replace("T", "")
                    .split(".")[0]
                )

            api_url = f"http://archive.org/wayback/available?url={url}"
            if query_ts:
                api_url += f"&timestamp={query_ts}"

            async with httpx.AsyncClient(timeout=10, follow_redirects=True) as client:
                resp = await client.get(api_url)
                if resp.status_code == 200:
                    data = resp.json()
                    closest = data.get("archived_snapshots", {}).get("closest")
                    if closest and closest.get("available"):
                        snapshot_url = closest["url"]
                        snapshot_ts = closest["timestamp"]
                        # The 'id_' suffix gives us raw HTML without the Wayback toolbar
                        return snapshot_url.replace(
                            f"/{snapshot_ts}/", f"/{snapshot_ts}id_/"
                        )
        except Exception:
            pass
        return None

    async def _fetch_async(
        self, url: str, timeout: int = 15, timestamp: Optional[str] = None
    ) -> Dict[str, Any]:
        """Async implementation of web fetching with hybrid strategy.

        Args:
            url: URL to fetch
            timeout: Timeout in seconds
            timestamp: Optional timestamp for historical fetch

        Returns:
            Dictionary with fetched content and metadata
        """
        # 1. Option A: Explicit Archive Fetch (Handled first if timestamp is provided)
        effective_url = url
        is_archived = False
        method = "live_fast"

        if timestamp:
            archive_url = await self._archive_fetch_async(url, timestamp)
            if archive_url:
                effective_url = archive_url
                is_archived = True
                method = "archive_explicit"

        # 2. Option B: Live Fast Fetch (AI-friendly headers)
        if not is_archived:
            fast_result = await self._fast_fetch_async(url, timeout=min(timeout, 10))
            if fast_result:
                return fast_result

            # 3. Option C: Optional latest-archive fallback. Keep this opt-in:
            # silently replacing a live page with today's latest snapshot is both
            # slower and unsafe for historical evidence windows. Explicit
            # timestamp requests above continue to use Internet Archive.
            if os.getenv("WEB_FETCH_IMPLICIT_ARCHIVE", "false").lower() == "true":
                archive_url = await self._archive_fetch_async(url)
                if archive_url:
                    effective_url = archive_url
                    is_archived = True
                    method = "archive_fallback_latest"

        # 4. Final Option: Robust Fetch with crawl4ai (Live browser)
        try:
            from crawl4ai import (
                AsyncWebCrawler,
                BrowserConfig,
                CrawlerRunConfig,
                CacheMode,
            )
            from crawl4ai.content_filter_strategy import PruningContentFilter
            from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator

            # Configure browser with stealth and random UA for robust fallback
            browser_config = BrowserConfig(
                headless=True,
                verbose=False,
                user_agent_mode="random",
                enable_stealth=True,
                extra_args=self._browser_cpu_args(),
            )

            # Use PruningContentFilter to strip noise and get clean fit_markdown
            md_generator = DefaultMarkdownGenerator(
                content_filter=PruningContentFilter(
                    threshold=0.45, threshold_type="fixed"
                )
            )

            # JS to bypass common consent walls (like Yahoo)
            js_bypass = """
            const bypassConsent = () => {
                const agreeButtons = [
                    'button[name="agree"]', // Yahoo
                    '#L2AGLb', // Google
                    '.consent-give', // Common
                    'button:contains("Accept all")',
                    'button:contains("Agree")'
                ];
                for (const selector of agreeButtons) {
                    try {
                        const btn = document.querySelector(selector);
                        if (btn) {
                            console.log("Found consent button: " + selector);
                            btn.click();
                            return true;
                        }
                    } catch (e) {}
                }
                return false;
            };
            bypassConsent();
            """

            # Configure crawler run (bypass cache, set timeout, magic mode)
            # For Archive URLs, we disable 'magic' and complex waiting as they can break on Wayback wrappers
            crawler_config = CrawlerRunConfig(
                cache_mode=CacheMode.BYPASS,
                page_timeout=timeout * 1000,
                markdown_generator=md_generator,
                magic=not is_archived,  # Disable magic for Archive URLs to avoid script interference
                js_code=js_bypass if not is_archived else None,
                wait_for="body:not(.wizard):not(.consent-page)"
                if not is_archived
                else None,
                delay_before_return_html=2.0 if not is_archived else 0.5,
            )

            # Fetch the page using context manager
            async with AsyncWebCrawler(config=browser_config) as crawler:
                result = await crawler.arun(effective_url, config=crawler_config)

            if not result.success:
                return {
                    "url": url,
                    "success": False,
                    "error": result.error_message or "Failed to fetch page",
                }

            # Extract metadata
            metadata = {}
            if result.metadata:
                # Preserve page metadata such as datePublished and
                # article:published_time. ArticleCollector uses it to establish
                # publication dates without asking the LLM to guess.
                metadata = dict(result.metadata)
                metadata.update(
                    {
                        "method": method
                        if is_archived
                        else "robust_fetch_crawl4ai",
                        "is_archived": is_archived,
                        "original_url": url if is_archived else None,
                    }
                )

            # Extract clean markdown: prefer fit_markdown (noise-filtered), fall back to raw_markdown
            md_content = result.markdown
            if hasattr(md_content, "fit_markdown") and md_content.fit_markdown:
                clean_markdown = md_content.fit_markdown
            elif hasattr(md_content, "raw_markdown") and md_content.raw_markdown:
                clean_markdown = md_content.raw_markdown
            else:
                clean_markdown = str(md_content) if md_content else ""

            # Build response
            response = {
                "url": result.url,
                "title": result.metadata.get("title", "") if result.metadata else "",
                "markdown": clean_markdown,
                "metadata": metadata,
                "success": True,
            }

            return response

        except ImportError:
            return {
                "url": url,
                "success": False,
                "error": "crawl4ai is not installed. Install it with: pip install crawl4ai",
            }
        except Exception as e:
            return {
                "url": url,
                "success": False,
                "error": f"Error fetching URL: {str(e)}",
            }

    def forward(
        self, url: str, timeout: int = 15, timestamp: Optional[str] = None
    ) -> WebFetchOutput:
        """Fetch web page content.

        Args:
            url: URL to fetch
            timeout: Maximum time to wait in seconds
            timestamp: Optional timestamp for historical fetch

        Returns:
            WebFetchOutput Pydantic model with fetched content
        """
        cached = self._get_cached(url, timestamp)
        if cached is not None:
            return cached

        # Check if we're already in an async context
        try:
            asyncio.get_running_loop()
            # We're in an async context - run coroutine in a new thread with its own event loop
            def run_in_thread():
                """Run the async function in a new thread with a new event loop."""
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                try:
                    return new_loop.run_until_complete(
                        self._fetch_with_deadline(url, timeout, timestamp)
                    )
                finally:
                    new_loop.close()

            executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            try:
                future = executor.submit(run_in_thread)
                emergency_timeout = (
                    max(float(timeout), 0.1)
                    + self._total_timeout_grace_seconds
                    + 2.0
                )
                try:
                    result = future.result(timeout=emergency_timeout)
                except concurrent.futures.TimeoutError:
                    future.cancel()
                    result = {
                        "url": url,
                        "success": False,
                        "error": (
                            f"Fetch worker exceeded the {emergency_timeout:g}s "
                            "emergency deadline"
                        ),
                    }
            finally:
                # A ThreadPoolExecutor context manager performs shutdown(wait=True),
                # which defeats the timeout when a browser worker is wedged.
                executor.shutdown(wait=False, cancel_futures=True)
        except RuntimeError:
            # No event loop running, safe to create one
            result = asyncio.run(
                self._fetch_with_deadline(url, timeout, timestamp)
            )

        output = self._to_output(result, url)
        self._cache_output(url, timestamp, output)
        return output

    async def forward_async(
        self, url: str, timeout: int = 15, timestamp: Optional[str] = None
    ) -> WebFetchOutput:
        """Async version of forward for use in async contexts.

        Args:
            url: URL to fetch
            timeout: Maximum time to wait in seconds
            timestamp: Optional timestamp for historical fetch

        Returns:
            WebFetchOutput Pydantic model with fetched content
        """
        cached = self._get_cached(url, timestamp)
        if cached is not None:
            return cached

        result = await self._fetch_with_deadline(url, timeout, timestamp)
        output = self._to_output(result, url)
        self._cache_output(url, timestamp, output)
        return output
