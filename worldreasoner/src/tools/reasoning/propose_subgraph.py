"""Tool to batch-create events and edges using structured JSON input."""

import hashlib
import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from smolagents import Tool

from src.tools.base.base import ToolResponseMixin
from src.tools.base.schema_helper import pydantic_to_output_schema
from src.tools.base.output_models import SubgraphOutput
from src.core.alias_registry import AliasRegistry
from src.domain.models import Article, CausalHypothesis, Event
from src.domain.models.forecast_graph import ForecastEvent, ForecastHypothesis


# Using local model to enforce structured input schema
# Note: we mirror the schema inside the tool so smolagents auto-generates the input dict
class ProposeSubgraphTool(Tool, ToolResponseMixin):
    """Tool to batch create events and edges for a causal graph."""

    name = "propose_subgraph"
    description = """Batch create events and edges in a single call to build the causal graph.

    This avoids having to call event_identifier and causal_reasoner dozens of times manually.
    Provide a JSON object with 'events' (list of event definitions) and 'edges' (list of causal links).
    """

    inputs = {
        "subgraph_json": {
            "type": "string",
            "description": "JSON object with 'events' and 'edges'. Events: {alias, title, description, domain, occurred_date, article_ids}. Edges: {source, target, relation, strength, confidence, reasoning}. 'source' and 'target' should match event aliases.",
        }
    }
    output_type = "object"
    output_schema = pydantic_to_output_schema(SubgraphOutput)

    @staticmethod
    def _clean_edge_error(reason: str) -> str:
        """Normalize internal tool errors into concise user-facing messages."""
        if not reason:
            return "Unknown validation error"

        cleaned = " ".join(str(reason).split())
        lowered = cleaned.lower()

        if lowered.startswith("chronology violation"):
            return cleaned.replace(
                "Chronology violation:",
                "Invalid chronology:",
                1,
            )
        return cleaned

    def __init__(
        self,
        event_identifier_tool,
        causal_reasoner_tool,
        alias_registry: AliasRegistry,
        db_path: str = None,
        question_id: Optional[str] = None,
    ):
        """Initialize the tool.

        Args:
            event_identifier_tool: Instance of EventIdentifierTool
            causal_reasoner_tool: Instance of CausalReasonerTool
            alias_registry: Registry to map semantic names to event UUIDs
        """
        super().__init__()
        self.question_id = question_id
        self.event_tool = event_identifier_tool
        self.reasoner_tool = causal_reasoner_tool
        self.alias_registry = alias_registry
        self.event_model = (
            ForecastEvent if hasattr(event_identifier_tool, "forecast_db") else Event
        )
        self.hypothesis_model = (
            ForecastHypothesis
            if hasattr(causal_reasoner_tool, "forecast_db")
            else CausalHypothesis
        )

        from src.core.database import GenericDatabase

        self.db = GenericDatabase(db_path) if db_path else None
        self._last_failure_signature: Optional[str] = None
        self._repeated_failure_count = 0

    @staticmethod
    def _parse_event_date(value: Any) -> Optional[datetime]:
        if not isinstance(value, str) or not value.strip():
            return None
        try:
            parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
            return parsed
        except ValueError:
            return None

    @staticmethod
    def _as_id_list(value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        return [item.strip() for item in str(value).split(",") if item.strip()]

    def _resolve_article_ids(self, value: Any) -> List[str]:
        raw_ids = self._as_id_list(value)
        if not self.alias_registry:
            return raw_ids
        return [self.alias_registry.resolve(article_id) or article_id for article_id in raw_ids]

    def _validate_subgraph(
        self, subgraph: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """Normalize common model variants and reject invalid batches before writes."""
        events = subgraph.get("events")
        edges = subgraph.get("edges")
        failures: List[Dict[str, Any]] = []

        if not isinstance(events, list):
            failures.append(
                {"type": "schema", "field": "events", "reason": "Must be a list"}
            )
            events = []
        if not isinstance(edges, list):
            failures.append(
                {"type": "schema", "field": "edges", "reason": "Must be a list"}
            )
            edges = []

        normalized_events = []
        aliases = set()
        event_dates: Dict[str, datetime] = {}

        for index, original in enumerate(events):
            if not isinstance(original, dict):
                failures.append(
                    {
                        "type": "event",
                        "index": index,
                        "reason": "Event must be an object",
                    }
                )
                continue

            event = dict(original)
            alias = event.get("alias")
            if not isinstance(alias, str) or not alias.strip():
                failures.append(
                    {
                        "type": "event",
                        "index": index,
                        "field": "alias",
                        "reason": "Missing event alias",
                    }
                )
                continue
            if alias in aliases:
                failures.append(
                    {
                        "type": "event",
                        "index": index,
                        "alias": alias,
                        "reason": "Duplicate event alias",
                    }
                )
                continue
            aliases.add(alias)

            # Gemini sometimes mirrors EventIdentifierTool's field name. Accept
            # it at this boundary and keep article_ids as the canonical batch key.
            article_value = event.get("article_ids")
            if article_value is None and "source_article_ids" in event:
                article_value = event.get("source_article_ids")
            resolved_article_ids = self._resolve_article_ids(article_value)
            event["article_ids"] = resolved_article_ids
            event.pop("source_article_ids", None)

            if self.db:
                missing_ids = [
                    article_id
                    for article_id in resolved_article_ids
                    if self.db.get(Article, article_id) is None
                ]
                if missing_ids:
                    failures.append(
                        {
                            "type": "event",
                            "index": index,
                            "alias": alias,
                            "field": "article_ids",
                            "reason": f"Unknown article IDs: {', '.join(missing_ids)}",
                        }
                    )

            occurred_date = self._parse_event_date(event.get("occurred_date"))
            if occurred_date is None:
                failures.append(
                    {
                        "type": "event",
                        "index": index,
                        "alias": alias,
                        "field": "occurred_date",
                        "reason": "Missing or invalid ISO datetime",
                    }
                )
            else:
                event_dates[alias] = occurred_date
                if self.db:
                    invalid_date_articles = []
                    for article_id in resolved_article_ids:
                        article = self.db.get(Article, article_id)
                        published = (
                            article.published_date if article is not None else None
                        )
                        if (
                            published is not None
                            and published.date() < occurred_date.date()
                        ):
                            invalid_date_articles.append(article_id)
                    if invalid_date_articles:
                        failures.append(
                            {
                                "type": "event",
                                "index": index,
                                "alias": alias,
                                "field": "occurred_date",
                                "reason": (
                                    "Source article predates the proposed event: "
                                    + ", ".join(invalid_date_articles)
                                    + ". Use the exact event time reported by the "
                                    "article; do not replay this batch unchanged."
                                ),
                            }
                        )

            normalized_events.append(event)

        normalized_edges = []
        adjacency: Dict[str, List[str]] = {}
        for index, original in enumerate(edges):
            if not isinstance(original, dict):
                failures.append(
                    {
                        "type": "edge",
                        "index": index,
                        "reason": "Edge must be an object",
                    }
                )
                continue

            edge = dict(original)
            source = edge.get("source")
            target = edge.get("target")
            if not source or not target:
                failures.append(
                    {
                        "type": "edge",
                        "index": index,
                        "reason": "Missing source or target",
                    }
                )
                continue

            for field, endpoint in (("source", source), ("target", target)):
                resolved = self.alias_registry.resolve(endpoint) if self.alias_registry else endpoint
                is_batch_alias = endpoint in aliases
                is_existing_event = bool(
                    self.db and resolved and self.db.get(self.event_model, resolved)
                )
                if not is_batch_alias and not is_existing_event:
                    failures.append(
                        {
                            "type": "edge",
                            "index": index,
                            "field": field,
                            "reason": f"Unknown event endpoint: {endpoint}",
                        }
                    )

            source_date = event_dates.get(source)
            target_date = event_dates.get(target)
            if source_date is None and self.db:
                resolved_source = self.alias_registry.resolve(source) if self.alias_registry else source
                source_event = (
                    self.db.get(self.event_model, resolved_source)
                    if resolved_source
                    else None
                )
                source_date = (
                    source_event.occurred_date or source_event.predicted_date
                    if source_event
                    else None
                )
            if target_date is None and self.db:
                resolved_target = self.alias_registry.resolve(target) if self.alias_registry else target
                target_event = (
                    self.db.get(self.event_model, resolved_target)
                    if resolved_target
                    else None
                )
                target_date = (
                    target_event.occurred_date or target_event.predicted_date
                    if target_event
                    else None
                )
            if source_date and target_date and source_date > target_date:
                failures.append(
                    {
                        "type": "edge",
                        "index": index,
                        "source": source,
                        "target": target,
                        "reason": (
                            "Invalid chronology: cause occurs after effect "
                            f"({source_date.isoformat()} > {target_date.isoformat()})"
                        ),
                    }
                )

            evidence_ids = self._resolve_article_ids(edge.get("evidence_article_ids"))
            edge["evidence_article_ids"] = evidence_ids
            if self.db:
                missing_evidence = [
                    article_id
                    for article_id in evidence_ids
                    if self.db.get(Article, article_id) is None
                ]
                if missing_evidence:
                    failures.append(
                        {
                            "type": "edge",
                            "index": index,
                            "field": "evidence_article_ids",
                            "reason": (
                                "Unknown article IDs: " + ", ".join(missing_evidence)
                            ),
                        }
                    )

            adjacency.setdefault(source, []).append(target)
            normalized_edges.append(edge)

        def has_cycle() -> bool:
            visiting = set()
            visited = set()

            def visit(node: str) -> bool:
                if node in visiting:
                    return True
                if node in visited:
                    return False
                visiting.add(node)
                for child in adjacency.get(node, []):
                    if visit(child):
                        return True
                visiting.remove(node)
                visited.add(node)
                return False

            return any(visit(node) for node in list(adjacency))

        if has_cycle():
            failures.append(
                {
                    "type": "graph",
                    "field": "edges",
                    "reason": "Cycle detected in proposed batch",
                }
            )

        return {"events": normalized_events, "edges": normalized_edges}, failures

    def _failure_output(
        self,
        subgraph: Any,
        failures: List[Dict[str, Any]],
    ) -> SubgraphOutput:
        canonical = json.dumps(subgraph, sort_keys=True, default=str)
        failure_text = json.dumps(failures, sort_keys=True, default=str)
        signature = hashlib.sha256(f"{canonical}|{failure_text}".encode()).hexdigest()
        if signature == self._last_failure_signature:
            self._repeated_failure_count += 1
            failures = list(failures) + [
                {
                    "type": "repeated_failure",
                    "reason": (
                        "This identical invalid payload was already rejected. "
                        "Change only the reported fields before retrying."
                    ),
                    "repeat_count": self._repeated_failure_count,
                }
            ]
        else:
            self._last_failure_signature = signature
            self._repeated_failure_count = 1

        return SubgraphOutput(
            status="error",
            events_created=0,
            edges_created=0,
            failed_items=failures,
            alias_map=self.alias_registry.list_aliases(),
        )

    def _rollback(
        self,
        before_events: Dict[str, Event],
        touched_event_ids: List[str],
        created_hypothesis_ids: List[str],
        aliases_before: Dict[str, str],
    ) -> None:
        if self.db:
            for hypothesis_id in created_hypothesis_ids:
                    self.db.delete(self.hypothesis_model, hypothesis_id)
            for event_id in touched_event_ids:
                if event_id in before_events:
                    self.db.save(self.event_model, before_events[event_id])
                else:
                    self.db.delete(self.event_model, event_id)
        if self.alias_registry:
            self.alias_registry.clear()
            for alias, item_id in aliases_before.items():
                self.alias_registry.register(alias, item_id)

    def forward(self, subgraph_json: str) -> SubgraphOutput:
        """Batch create events and edges."""
        try:
            subgraph = json.loads(subgraph_json)
        except json.JSONDecodeError as e:
            # Tool-calling models occasionally append one or more unmatched
            # closing braces to an otherwise valid JSON object. Accept only
            # that narrow, unambiguous suffix; prose or any other trailing data
            # remains an error.
            try:
                candidate = subgraph_json.lstrip()
                object_start = candidate.find("{")
                if object_start < 0:
                    raise e
                candidate = candidate[object_start:]
                subgraph, end = json.JSONDecoder().raw_decode(candidate)
                trailing = candidate[end:].strip()
                if trailing and set(trailing) != {"}"}:
                    raise e
            except (json.JSONDecodeError, ValueError):
                return SubgraphOutput(
                    status="error",
                    events_created=0,
                    edges_created=0,
                    failed_items=[
                        {"type": "parse_error", "reason": f"Invalid JSON: {str(e)}"}
                    ],
                )

        if not isinstance(subgraph, dict):
            return self._failure_output(
                subgraph,
                [{"type": "schema", "reason": "Top-level JSON must be an object"}],
            )

        subgraph, validation_failures = self._validate_subgraph(subgraph)
        if validation_failures:
            return self._failure_output(subgraph, validation_failures)

        events = subgraph.get("events", [])
        edges = subgraph.get("edges", [])

        events_created = 0
        edges_created = 0
        failed_items = []
        before_events = (
            {
                event.id: event.model_copy(deep=True)
                for event in self.db.get_many(self.event_model)
            }
            if self.db
            else {}
        )
        aliases_before = self.alias_registry.list_aliases()
        touched_event_ids: List[str] = []
        created_hypothesis_ids: List[str] = []

        # 1. Process Events
        for ev in events:
            alias = ev.get("alias")
            if not alias:
                failed_items.append(
                    {"type": "event", "item": ev, "reason": "Missing 'alias' field"}
                )
                continue

            try:
                # Call EventIdentifierTool
                # Map structured input back to tool arguments
                title = ev.get("title", "Unknown Event")
                desc = ev.get("description", "")
                domain = ev.get("domain", "politics")

                art_ids = ev.get("article_ids", [])
                if isinstance(art_ids, list):
                    raw_ids = art_ids
                else:
                    raw_ids = [a.strip() for a in str(art_ids).split(",") if a.strip()]

                art_ids_str = ",".join(raw_ids)

                occurred_date = ev.get("occurred_date", "")

                # We do NOT pass alias to event_identifier inputs directly,
                # but we will register the resulting ID in our alias registry!

                resp = self.event_tool.forward(
                    title=title,
                    description=desc,
                    domain=domain,
                    source_article_ids=art_ids_str,
                    occurred_date=occurred_date,
                )

                # EventIdentifier returns EventOutput on success or a JSON error string on failure
                if isinstance(resp, str):
                    import json as _json
                    try:
                        err = _json.loads(resp)
                        reason = err.get("message") or err.get("error") or resp
                    except Exception:
                        reason = resp
                    failed_items.append({"type": "event", "alias": alias, "reason": reason})
                    continue

                event_id = getattr(resp, "id", None)
                if not event_id:
                    event_data = getattr(resp, "event", None)
                    if isinstance(event_data, dict):
                        event_id = event_data.get("id")
                event_status = str(getattr(resp, "status", "")).strip()

                # EventIdentifierTool may return EventOutput(id="error", status="error: ...")
                # for validation failures. Treat these as failures and do not register aliases.
                if event_id == "error" or event_status.lower().startswith("error"):
                    reason = event_status or "Event validation failed"
                    failed_items.append(
                        {
                            "type": "event",
                            "alias": alias,
                            "reason": reason,
                        }
                    )
                    continue

                if not event_id:
                    failed_items.append(
                        {
                            "type": "event",
                            "alias": alias,
                            "reason": "Tool failed to return an ID",
                        }
                    )
                else:
                    self.alias_registry.register(alias, event_id)
                    touched_event_ids.append(event_id)
                    events_created += 1

            except Exception as e:
                failed_items.append({"type": "event", "alias": alias, "reason": str(e)})

        # 2. Process Edges
        for ed in edges:
            source_alias = ed.get("source")
            target_alias = ed.get("target")

            if not source_alias or not target_alias:
                failed_items.append(
                    {
                        "type": "edge",
                        "item": ed,
                        "reason": "Missing source or target alias",
                    }
                )
                continue

            # Resolve aliases to UUIDs
            source_id = self.alias_registry.resolve(source_alias) or source_alias
            target_id = self.alias_registry.resolve(target_alias) or target_alias

            # Alias resolver can carry unresolved values as "error".
            if source_id == "error" or target_id == "error":
                unresolved = source_alias if source_id == "error" else target_alias
                failed_items.append(
                    {
                        "type": "edge",
                        "source": source_alias,
                        "target": target_alias,
                        "reason": (
                            f"Unresolved event alias '{unresolved}'. "
                            "Create that event first, then retry this edge."
                        ),
                    }
                )
                continue

            try:
                # Call CausalReasonerTool
                resp = self.reasoner_tool.forward(
                    source_event_id=source_id,
                    target_event_id=target_id,
                    relation_type=ed.get("relation", "causes"),
                    strength=float(ed.get("strength", 0.5)),
                    confidence=float(ed.get("confidence", 0.5)),
                    reasoning=ed.get("reasoning", ""),
                    evidence_article_ids=",".join(
                        ed.get("evidence_article_ids", [])
                    ),
                )

                # CausalReasonerTool returns HypothesisOutput on success or a JSON error string
                if isinstance(resp, str):
                    import json as _json
                    try:
                        err = _json.loads(resp)
                        reason = err.get("message") or err.get("error") or resp
                    except Exception:
                        reason = resp
                    failed_items.append(
                        {
                            "type": "edge",
                            "source": source_alias,
                            "target": target_alias,
                            "reason": self._clean_edge_error(reason),
                        }
                    )
                elif getattr(resp, "status", "error") == "error":
                    error_msg = getattr(resp, "error", "Unknown validation error")
                    failed_items.append(
                        {
                            "type": "edge",
                            "source": source_alias,
                            "target": target_alias,
                            "reason": self._clean_edge_error(error_msg),
                        }
                    )
                else:
                    hypothesis_id = getattr(resp, "hypothesis_id", None)
                    if hypothesis_id:
                        created_hypothesis_ids.append(hypothesis_id)
                    edges_created += 1

            except Exception as e:
                failed_items.append(
                    {
                        "type": "edge",
                        "source": source_alias,
                        "target": target_alias,
                        "reason": str(e),
                    }
                )

        if failed_items:
            self._rollback(
                before_events=before_events,
                touched_event_ids=touched_event_ids,
                created_hypothesis_ids=created_hypothesis_ids,
                aliases_before=aliases_before,
            )
            return self._failure_output(subgraph, failed_items)

        self._last_failure_signature = None
        self._repeated_failure_count = 0

        return SubgraphOutput(
            status="success",
            events_created=events_created,
            edges_created=edges_created,
            failed_items=failed_items,
            alias_map=self.alias_registry.list_aliases(),
        )
