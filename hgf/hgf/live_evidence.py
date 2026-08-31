"""Collect cutoff-safe current evidence without a persistent SQLite database.

This adapter reuses the bundled WorldReasoner search transport.  It exposes
only target metadata and Blueprint factor labels to search, then returns a
plain evidence list compatible with the canonical ReFoRM forecasting pipeline.
No DAG paths, relations, historical answers, or target outcomes are sent to
the search backend.
"""

from __future__ import annotations

import hashlib
import html
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


_PLACEHOLDER_RE = re.compile(r"\[(?:CURRENT_[A-Z_]+|PAST_OUTCOME_REDACTED)\]")


def _worldreasoner_root() -> Path:
    return Path(__file__).resolve().parents[2] / "worldreasoner"


def _load_worldreasoner_tools() -> tuple[type[Any], type[Any]]:
    root = _worldreasoner_root()
    if not root.is_dir():
        raise RuntimeError(
            f"bundled WorldReasoner source is missing: {root}"
        )
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        from src.tools.collectors.web_fetch import WebFetchTool
        from src.tools.collectors.web_search import WebSearchTool
    except ImportError as exc:
        raise RuntimeError(
            "live evidence requires requirements-generation.txt"
        ) from exc
    return WebSearchTool, WebFetchTool


def _metadata(question: Any) -> dict[str, Any]:
    root = getattr(question, "metadata", None) or {}
    if not isinstance(root, dict):
        return {}
    for namespace in ("finance", "finfactorbench", "benchmark"):
        value = root.get(namespace)
        if isinstance(value, dict) and value:
            return value
    return root


def _clean_factor(value: Any) -> str:
    text = _PLACEHOLDER_RE.sub("", str(value or ""))
    return " ".join(text.replace("[", " ").replace("]", " ").split())


def _search_period(metadata: dict[str, Any]) -> str:
    value = str(metadata.get("target_period") or "").strip()
    match = re.search(r"(20\d{2})-(\d{2})-(\d{2})", value)
    if match and "quarter" in value.casefold():
        year, month = int(match.group(1)), int(match.group(2))
        return f"Q{(month - 1) // 3 + 1} {year}"
    month_match = re.search(r"(20\d{2})-(\d{2})", value)
    if month_match:
        return f"{month_match.group(1)}-{month_match.group(2)}"
    return value


def _metric_search_terms(metric: str) -> str:
    lowered = metric.casefold()
    if "revenue" in lowered:
        return "revenue expectations earnings"
    if "cpi" in lowered or "price" in lowered:
        return "inflation outlook price drivers"
    if "yield" in lowered or "rate" in lowered:
        return "rate outlook yield drivers"
    return metric


def _plain_text(value: Any) -> str:
    without_tags = re.sub(r"<[^>]+>", " ", str(value or ""))
    return " ".join(html.unescape(without_tags).split())


def build_live_queries(
    question: Any,
    blueprints: list[dict[str, Any]],
    cutoff: datetime,
    *,
    limit: int,
) -> list[str]:
    """Build bounded searches from the target and factor labels only."""
    metadata = _metadata(question)
    entity = re.sub(
        r"\s*\([^)]*\)\s*$", "", str(metadata.get("entity") or "").strip()
    )
    metric = str(metadata.get("target_metric") or "").strip()
    period = _search_period(metadata)
    base = " ".join(
        part for part in (entity, period, _metric_search_terms(metric)) if part
    )
    candidates = [base or str(question.question_text)]
    for blueprint in blueprints:
        for row in blueprint.get("search_factors", []):
            factor = _clean_factor(row.get("factor"))
            if factor:
                candidates.append(
                    " ".join(part for part in (entity, period, factor) if part)
                )

    before = cutoff.astimezone(UTC).date().isoformat()
    queries: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        normalized = " ".join(candidate.split())
        key = normalized.casefold()
        if not normalized or key in seen:
            continue
        seen.add(key)
        queries.append(f"{normalized} before:{before}")
        if len(queries) >= max(1, limit):
            break
    return queries


def _parse_date(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        parsed = value
    else:
        text = str(value or "").strip()
        if not text:
            return None
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


def _fetch_excerpt(fetch_type: type[Any], url: str, timeout: int) -> str:
    output = fetch_type().forward(url=url, timeout=timeout)
    if not getattr(output, "success", False):
        return ""
    return " ".join(str(getattr(output, "content", "") or "").split())[:2000]


def collect_live_evidence(
    *,
    question: Any,
    cutoff: datetime,
    blueprints: list[dict[str, Any]],
    provider: str = "auto",
    query_limit: int = 6,
    result_limit: int = 80,
    fetch_limit: int = 12,
    fetch_workers: int = 2,
    fetch_timeout: int = 15,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Search, enforce the cutoff, deduplicate, and return in-memory evidence."""
    WebSearchTool, WebFetchTool = _load_worldreasoner_tools()
    queries = build_live_queries(
        question,
        blueprints,
        cutoff,
        limit=query_limit,
    )
    tool = WebSearchTool(domain="finance", enforce_upper_only_dates=True)
    raw: list[dict[str, Any]] = []
    errors: list[str] = []
    for query in queries:
        output = tool.forward(query=query, provider=provider, page=1)
        if getattr(output, "error", None):
            errors.append(f"{query}: {output.error}")
        for item in getattr(output, "results", []):
            payload = item.model_dump() if hasattr(item, "model_dump") else dict(item)
            payload["query"] = query
            raw.append(payload)

    normalized_cutoff = cutoff if cutoff.tzinfo is not None else cutoff.replace(tzinfo=UTC)
    by_url: dict[str, dict[str, Any]] = {}
    dropped_unknown_date = 0
    dropped_after_cutoff = 0
    for item in raw:
        url = str(item.get("url") or "").strip()
        published = _parse_date(item.get("published_date"))
        if not url or published is None:
            dropped_unknown_date += 1
            continue
        if published >= normalized_cutoff:
            dropped_after_cutoff += 1
            continue
        existing = by_url.get(url)
        if existing is None or published > existing["_published"]:
            by_url[url] = {**item, "_published": published}

    ordered = sorted(
        by_url.values(),
        key=lambda item: item["_published"],
        reverse=True,
    )[: max(1, result_limit)]
    fetched: dict[str, str] = {}
    fetch_candidates = ordered[: max(0, fetch_limit)]
    if fetch_candidates:
        with ThreadPoolExecutor(max_workers=max(1, fetch_workers)) as executor:
            futures = {
                executor.submit(
                    _fetch_excerpt,
                    WebFetchTool,
                    str(item["url"]),
                    fetch_timeout,
                ): str(item["url"])
                for item in fetch_candidates
            }
            for future in as_completed(futures):
                url = futures[future]
                try:
                    excerpt = future.result()
                except Exception:
                    excerpt = ""
                if excerpt:
                    fetched[url] = excerpt

    evidence: list[dict[str, Any]] = []
    for item in ordered:
        url = str(item["url"])
        published = item["_published"]
        identifier = hashlib.sha256(
            f"{url}|{published.isoformat()}".encode("utf-8")
        ).hexdigest()[:20]
        evidence.append(
            {
                "id": f"live_{identifier}",
                "title": str(item.get("title") or url),
                "source": str(item.get("source") or "web"),
                "published_date": published.isoformat(),
                "excerpt": fetched.get(url)
                or _plain_text(item.get("description"))[:1000],
                "url": url,
                "search_query": str(item.get("query") or ""),
            }
        )
    if not evidence:
        raise RuntimeError(
            "live search returned no evidence with a verifiable date before the cutoff"
        )
    manifest = {
        "schema_version": "hgf_live_evidence_v1",
        "provider": provider,
        "cutoff": normalized_cutoff.isoformat(),
        "queries": queries,
        "raw_result_count": len(raw),
        "eligible_result_count": len(evidence),
        "fetched_page_count": len(fetched),
        "dropped_unknown_date": dropped_unknown_date,
        "dropped_at_or_after_cutoff": dropped_after_cutoff,
        "errors": errors,
    }
    return evidence, manifest
