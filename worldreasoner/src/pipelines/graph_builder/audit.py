"""Lightweight deterministic audit for hindsight causal graphs."""

from collections import defaultdict
from typing import Any, Dict, List, Set

from src.core.database import GenericDatabase
from src.domain.models import CausalHypothesis, Event, Question
from src.services.question_monitor_service import QuestionMonitorService
from src.utils.date_utils import ensure_timezone_aware


class GraphAuditPipeline:
    """Validate the minimum causal, temporal, and evidence invariants."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.db = GenericDatabase(db_path)

    def audit_question(self, question_id: str) -> Dict[str, Any]:
        """Audit the graph for a specific question."""
        question = self.db.get(Question, question_id)
        if not question:
            return {
                "status": "error",
                "message": f"Question {question_id} not found.",
            }

        events = {event.id: event for event in self.db.get_many(Event)}
        hypotheses = [
            hypothesis
            for hypothesis in self.db.get_many(CausalHypothesis)
            if question_id in hypothesis.discovered_by_question_ids
        ]
        issues: List[str] = []

        outcome_candidates = {
            event.id: event
            for event in events.values()
            if event.id in question.outcome_event_ids
            or (
                event.extracted_for_question_id == question_id
                and event.is_outcome
            )
        }
        actual_outcomes = [
            event for event in outcome_candidates.values() if event.is_actual_outcome
        ]
        actual_outcome_id = actual_outcomes[0].id if len(actual_outcomes) == 1 else None
        if len(actual_outcomes) != 1:
            issues.append(
                "Expected exactly one actual outcome event, found "
                f"{len(actual_outcomes)}."
            )

        event_ids: Set[str] = set()
        adjacency: Dict[str, Set[str]] = defaultdict(set)
        for hypothesis in hypotheses:
            event_ids.update(
                (hypothesis.source_event_id, hypothesis.target_event_id)
            )
            adjacency[hypothesis.source_event_id].add(hypothesis.target_event_id)
            source = events.get(hypothesis.source_event_id)
            target = events.get(hypothesis.target_event_id)
            if not source or not target:
                issues.append(
                    f"Hypothesis {hypothesis.id} references a missing event."
                )
                continue

            source_date = source.occurred_date or source.predicted_date
            target_date = target.occurred_date or target.predicted_date
            if source_date and target_date and ensure_timezone_aware(
                source_date
            ) > ensure_timezone_aware(target_date):
                issues.append(
                    "Chronology violation: "
                    f"{source.title} ({source_date}) > {target.title} ({target_date})."
                )

        has_cycle = self._has_cycle(adjacency)
        if has_cycle:
            issues.append("Cycle detected in the question causal graph.")

        valid_article_ids = {
            article.id
            for article in QuestionMonitorService(self.db).get_evidence_articles(
                question_id
            )
        }
        for event_id in sorted(event_ids):
            event = events.get(event_id)
            if not event or event.is_outcome:
                continue
            if not (event.occurred_date or event.predicted_date):
                issues.append(f"Event {event_id} has no event date.")
            if not valid_article_ids.intersection(event.article_ids or []):
                issues.append(
                    f"Event {event_id} has no cutoff-safe evidence for this question."
                )

        if actual_outcome_id:
            has_incoming = any(
                hypothesis.target_event_id == actual_outcome_id
                for hypothesis in hypotheses
            )
            if not has_incoming:
                issues.append(
                    f"Actual outcome event {actual_outcome_id} has no incoming causal links (orphan)."
                )
            if actual_outcome_id in adjacency:
                issues.append(
                    f"Actual outcome event {actual_outcome_id} has outgoing causal links."
                )

        return {
            "status": "pass" if not issues else "fail",
            "issues": issues,
            "actual_outcome_event_id": actual_outcome_id,
            "events_count": len(event_ids),
            "hypotheses_count": len(hypotheses),
        }

    @staticmethod
    def _has_cycle(adjacency: Dict[str, Set[str]]) -> bool:
        visiting: Set[str] = set()
        visited: Set[str] = set()

        def visit(node: str) -> bool:
            if node in visiting:
                return True
            if node in visited:
                return False
            visiting.add(node)
            if any(visit(child) for child in adjacency.get(node, set())):
                return True
            visiting.remove(node)
            visited.add(node)
            return False

        return any(visit(node) for node in list(adjacency))
