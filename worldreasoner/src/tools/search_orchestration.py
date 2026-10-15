"""Bounded, shared search planning and coverage state.

The LLM chooses factors and queries.  This module only records the plan,
prevents duplicate/unbounded searches, and reports which planned factors still
lack cutoff-safe evidence.  It deliberately contains no LLM calls.
"""

from __future__ import annotations

import re
import sqlite3
from threading import Lock
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field
from smolagents import Tool

from src.tools.base.schema_helper import pydantic_to_output_schema
from src.utils.date_utils import ensure_timezone_aware


SEARCH_FACTOR_ROLES = (
    "official",
    "baseline",
    "supporting",
    "countervailing",
    "mechanism",
    "other",
)


class SearchCoverageOutput(BaseModel):
    """Serializable plan/coverage state returned to a search agent."""

    status: str
    plan_registered: bool
    query_budget: int
    queries_used: int
    article_count: int
    target_article_count: int
    coverage_target_met: bool
    factors: List[Dict[str, Any]] = Field(default_factory=list)
    gaps: List[str] = Field(default_factory=list)
    recommended_queries: List[Dict[str, str]] = Field(default_factory=list)
    unassigned_article_count: int = 0
    message: str = ""


def _normalized_text(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _query_identity(value: Any) -> str:
    """Normalize a query while ignoring only transport-side date bounds."""
    without_dates = re.sub(
        r"\b(?:after|before):\d{4}-\d{2}-\d{2}\b",
        "",
        str(value or ""),
        flags=re.IGNORECASE,
    )
    return _normalized_text(without_dates)


def _tokens(value: Any) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9][a-z0-9._%-]+", _normalized_text(value))
        if len(token) > 2
    }


class SearchCoverageTracker:
    """In-memory coverage ledger shared by one bounded search run."""

    def __init__(
        self,
        *,
        db_path: Optional[str],
        question_id: str,
        min_articles: int = 10,
        max_queries: int = 10,
        cutoff=None,
        allow_over_target_factor_search: bool = True,
    ) -> None:
        self.db_path = db_path
        self.question_id = question_id
        self.min_articles = max(1, min_articles)
        self.max_queries = max(1, max_queries)
        self.cutoff = ensure_timezone_aware(cutoff) if cutoff else None
        self.allow_over_target_factor_search = allow_over_target_factor_search
        self._factors: Dict[str, Dict[str, Any]] = {}
        self._attempts: List[Dict[str, Any]] = []
        self._attempt_keys: set[tuple[str, int, str]] = set()
        self._inflight_keys: set[tuple[str, int, str]] = set()
        self._lock = Lock()

    def register_plan(self, factors: List[Dict[str, Any]]) -> None:
        """Register the initial plan or enrich matching factors on a retry.

        A retry agent may call ``register_plan`` again even though the shared
        ledger already has a plan.  Keep the original factor identities stable
        in that case: merge queries only for exact factor-name matches and
        ignore newly invented factor names.  This preserves cumulative coverage
        without letting retries expand beyond the six-factor contract.
        """
        minimum = 1 if self._factors else 2
        if not isinstance(factors, list) or not minimum <= len(factors) <= 6:
            raise ValueError(
                f"Search plan must contain {minimum}-6 factors for this action"
            )

        with self._lock:
            has_existing_plan = bool(self._factors)
            for raw in factors:
                if not isinstance(raw, dict):
                    raise ValueError("Each search factor must be an object")
                name = " ".join(str(raw.get("name") or "").split())
                if not name:
                    raise ValueError("Each search factor requires a non-empty name")
                role = _normalized_text(raw.get("role") or "other")
                if role not in SEARCH_FACTOR_ROLES:
                    raise ValueError(
                        f"Unsupported factor role {role!r}; expected one of "
                        + ", ".join(SEARCH_FACTOR_ROLES)
                    )
                queries = raw.get("queries") or []
                if isinstance(queries, str):
                    queries = [queries]
                normalized_queries = []
                for query in queries:
                    cleaned = " ".join(str(query or "").split())
                    if cleaned and cleaned not in normalized_queries:
                        normalized_queries.append(cleaned)
                if not normalized_queries:
                    raise ValueError(f"Search factor {name!r} requires at least one query")

                key = _normalized_text(name)
                existing = self._factors.get(key)
                if existing:
                    for query in normalized_queries:
                        if query not in existing["queries"]:
                            existing["queries"].append(query)
                    continue
                if has_existing_plan:
                    continue
                self._factors[key] = {
                    "name": name,
                    "role": role,
                    "queries": normalized_queries[:3],
                    "article_ids": set(),
                }

    def _attempt_key(
        self, query: str, page: int, provider: Optional[str]
    ) -> tuple[str, int, str]:
        # The tracker has one immutable cutoff. Transport adapters may append
        # or remove that same date bound, which must not turn one logical query
        # into multiple budget-consuming attempts.
        return (_query_identity(query), max(1, page or 1), provider or "auto")

    def allow_search(
        self,
        query: str,
        page: int = 1,
        provider: Optional[str] = None,
        factor: Optional[str] = None,
    ) -> tuple[bool, str]:
        """Reject completed, duplicate, or over-budget search calls."""
        articles = self._eligible_articles() if self.db_path else []
        if len(articles) >= self.min_articles:
            if not self.allow_over_target_factor_search:
                return (
                    False,
                    f"Verified article target already met ({self.min_articles}+); "
                    "stop searching and synthesize the evidence map.",
                )
            # Meeting the global article target must stop open-ended searching,
            # but HGF may still need one of its explicitly planned factor gaps
            # connected to cutoff-safe evidence already in (or newly added to)
            # the DB.  Only an exact registered factor name can justify such a
            # bounded over-target call; arbitrary/query-inferred searches remain
            # blocked.
            self._refresh_article_assignments(articles)
            eligible_ids = {article.id for article in articles}
            factor_key = _normalized_text(factor)
            factor_item = self._factors.get(factor_key)
            is_registered_query = bool(
                factor_item
                and _query_identity(query)
                in {_query_identity(item) for item in factor_item["queries"]}
            )
            factor_uncovered = bool(
                factor_item
                and is_registered_query
                and not (factor_item["article_ids"] & eligible_ids)
            )
            if not factor_uncovered:
                return (
                    False,
                    f"Verified article target already met ({self.min_articles}+); "
                    "only a registered query for an explicitly planned uncovered "
                    "factor may be searched.",
                )
        key = self._attempt_key(query, page, provider)
        with self._lock:
            if key in self._attempt_keys or key in self._inflight_keys:
                return False, "Duplicate query/page/provider combination"
            if len(self._attempt_keys) + len(self._inflight_keys) >= self.max_queries:
                return False, f"Search query budget exhausted ({self.max_queries})"
            # Reserve before the network call so parallel tool calls cannot all
            # pass the same bounded budget check.
            self._inflight_keys.add(key)
        return True, ""

    def extend_query_budget(self, additional_queries: int) -> None:
        """Open one bounded retry pass without forgetting prior search attempts."""
        if additional_queries <= 0:
            raise ValueError("additional_queries must be positive")
        with self._lock:
            self.max_queries += additional_queries

    def _infer_factor_key(self, factor: Optional[str], query: str) -> Optional[str]:
        requested = _normalized_text(factor)
        if requested in self._factors:
            return requested
        query_tokens = _tokens(query)
        best_key = None
        best_score = 0
        for key, item in self._factors.items():
            candidate_tokens = _tokens(item["name"])
            for planned_query in item["queries"]:
                candidate_tokens.update(_tokens(planned_query))
            score = len(query_tokens & candidate_tokens)
            if score > best_score:
                best_key = key
                best_score = score
        return best_key

    def record_search(
        self,
        *,
        query: str,
        page: int = 1,
        provider: Optional[str] = None,
        factor: Optional[str] = None,
        result_urls: Optional[List[str]] = None,
        article_ids: Optional[List[str]] = None,
        raw_result_count: int = 0,
        error: Optional[str] = None,
    ) -> None:
        """Record one completed search and its verified article associations."""
        key = self._attempt_key(query, page, provider)
        factor_key = self._infer_factor_key(factor, query)
        with self._lock:
            if key in self._attempt_keys:
                self._inflight_keys.discard(key)
                return
            self._inflight_keys.discard(key)
            self._attempt_keys.add(key)
            ids = {str(item) for item in (article_ids or []) if item}
            if factor_key and factor_key in self._factors:
                self._factors[factor_key]["article_ids"].update(ids)
            self._attempts.append(
                {
                    "query": " ".join(query.split()),
                    "page": key[1],
                    "provider": key[2],
                    "factor_key": factor_key,
                    "result_urls": sorted(set(result_urls or [])),
                    "raw_result_count": raw_result_count,
                    "article_ids": ids,
                    "error": error,
                }
            )

    def _eligible_articles(self):
        if not self.db_path:
            return []
        from src.core.database import GenericDatabase
        from src.domain.models import Article, Question

        db = GenericDatabase(self.db_path)
        try:
            question = db.get(Question, self.question_id)
            articles = db.get_many(Article)
        except sqlite3.OperationalError as exc:
            # A per-round tracker can be constructed just before the runner
            # initializes its database tables. Treat that brief bootstrap state
            # as zero coverage rather than failing the first search.
            if "no such table" in str(exc).lower():
                return []
            raise
        cutoff = self.cutoff or (
            ensure_timezone_aware(question.resolution_date) if question else None
        )
        eligible = []
        for article in articles:
            linked = (
                article.collected_for_question_id == self.question_id
                or self.question_id
                in (article.metadata or {}).get("related_question_ids", [])
            )
            if not linked or not article.published_date:
                continue
            if cutoff and ensure_timezone_aware(article.published_date) >= cutoff:
                continue
            eligible.append(article)
        return eligible

    def _refresh_article_assignments(self, articles) -> None:
        """Resolve URLs again so manual collection after search is also credited."""
        article_id_by_url = {article.url: article.id for article in articles if article.url}
        with self._lock:
            for attempt in self._attempts:
                resolved_ids = {
                    article_id_by_url[url]
                    for url in attempt["result_urls"]
                    if url in article_id_by_url
                }
                attempt["article_ids"].update(resolved_ids)
                factor_key = attempt.get("factor_key")
                if factor_key and factor_key in self._factors:
                    self._factors[factor_key]["article_ids"].update(resolved_ids)

    def snapshot(self) -> Dict[str, Any]:
        articles = self._eligible_articles()
        self._refresh_article_assignments(articles)
        eligible_ids = {article.id for article in articles}
        tried_queries = {
            _normalized_text(attempt["query"]) for attempt in self._attempts
        }

        factors = []
        covered_ids: set[str] = set()
        recommended_queries = []
        gaps = []
        coverage_target_met = len(articles) >= self.min_articles
        for item in self._factors.values():
            verified_ids = sorted(item["article_ids"] & eligible_ids)
            covered_ids.update(verified_ids)
            factors.append(
                {
                    "name": item["name"],
                    "role": item["role"],
                    "article_count": len(verified_ids),
                    "article_ids": verified_ids,
                }
            )
            if not verified_ids:
                gaps.append(f"No verified evidence for factor: {item['name']}")
                if (
                    not coverage_target_met
                    or self.allow_over_target_factor_search
                ):
                    for query in item["queries"]:
                        if _normalized_text(query) not in tried_queries:
                            recommended_queries.append(
                                {"factor": item["name"], "query": query}
                            )
                            break

        if not coverage_target_met:
            gaps.insert(
                0,
                f"Verified article target not met: {len(articles)} < {self.min_articles}",
            )
        factor_gaps_remain = bool(
            self._factors
            and any(factor["article_count"] == 0 for factor in factors)
        )
        return {
            "status": "ready" if self._factors else "plan_required",
            "plan_registered": bool(self._factors),
            "query_budget": self.max_queries,
            "queries_used": len(self._attempt_keys),
            "article_count": len(articles),
            "target_article_count": self.min_articles,
            "coverage_target_met": coverage_target_met,
            "factors": factors,
            "gaps": gaps,
            "recommended_queries": recommended_queries,
            "unassigned_article_count": len(eligible_ids - covered_ids),
            "message": (
                "Coverage target met. Stop searching and synthesize the "
                "available evidence map; report any remaining factor gaps."
                if coverage_target_met
                and not self.allow_over_target_factor_search
                else
                "Coverage target met, but planned factor gaps remain. Search only "
                "the recommended uncovered factors, using their exact factor names."
                if coverage_target_met and factor_gaps_remain
                else "Coverage and planned factors are covered. Stop searching and "
                "synthesize the evidence map."
                if coverage_target_met
                else "Search the uncovered factors only; do not repeat tried "
                "query/page/provider combinations."
            ),
        }


class SearchCoverageTool(Tool):
    """Register a bounded search plan and inspect deterministic coverage gaps."""

    name = "search_coverage"
    description = (
        "Register a 2-6 factor search plan before searching, then inspect which "
        "factors still lack verified cutoff-safe articles. This tool does not search "
        "or call another model. Reuse the exact factor names in web-search calls."
    )
    inputs = {
        "action": {
            "type": "string",
            "enum": ["register_plan", "status"],
            "description": "Register/extend the plan, or inspect current coverage.",
        },
        "factors": {
            "type": "array",
            "description": "Required only for register_plan.",
            "nullable": True,
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "role": {
                        "type": "string",
                        "enum": list(SEARCH_FACTOR_ROLES),
                    },
                    "queries": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": ["name", "role", "queries"],
            },
        },
    }
    output_type = "object"
    output_schema = pydantic_to_output_schema(SearchCoverageOutput)

    def __init__(self, tracker: SearchCoverageTracker) -> None:
        super().__init__()
        self.tracker = tracker

    def forward(
        self,
        action: str,
        factors: Optional[List[Dict[str, Any]]] = None,
    ) -> SearchCoverageOutput:
        if action == "register_plan":
            self.tracker.register_plan(factors or [])
        elif action != "status":
            raise ValueError(f"Unsupported search coverage action: {action}")
        return SearchCoverageOutput(**self.tracker.snapshot())
