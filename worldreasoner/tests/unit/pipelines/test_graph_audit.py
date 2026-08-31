"""Tests for the deterministic hindsight DAG validation gate."""

from datetime import datetime, timezone

from src.domain.models import (
    Article,
    CausalHypothesis,
    CausalRelationType,
    Domain,
    Event,
    EventStatus,
    EventType,
    Question,
)
from src.pipelines.graph_builder.audit import GraphAuditPipeline
from tests.conftest import create_test_question


def _save_grounded_graph(test_db) -> None:
    resolution = datetime(2026, 4, 1, tzinfo=timezone.utc)
    question = create_test_question(
        id="q_graph_audit",
        domain=Domain.FINANCE,
        ground_truth=True,
        resolution_date=resolution,
        outcome_event_ids=["evt_outcome"],
    )
    article = Article(
        id="art_grounding",
        title="A dated source reports the causal event",
        content="Grounding evidence. " * 20,
        url="https://example.com/grounding",
        source="Example",
        published_date=datetime(2026, 3, 2, tzinfo=timezone.utc),
        domain=Domain.FINANCE,
        collected_for_question_id=question.id,
    )
    cause = Event(
        id="evt_cause",
        title="Company announces a concrete operational change",
        description="The company announced a concrete operational change before resolution.",
        event_type=EventType.DECISION,
        domain=Domain.FINANCE,
        status=EventStatus.OCCURRED,
        occurred_date=datetime(2026, 3, 1, tzinfo=timezone.utc),
        article_ids=[article.id],
        extracted_for_question_id=question.id,
    )
    outcome = Event(
        id="evt_outcome",
        title="The forecast question resolves positively",
        description="The known positive resolution of the forecast question occurred.",
        event_type=EventType.OUTCOME,
        domain=Domain.FINANCE,
        status=EventStatus.OCCURRED,
        occurred_date=resolution,
        is_outcome=True,
        is_actual_outcome=True,
        extracted_for_question_id=question.id,
    )
    hypothesis = CausalHypothesis(
        id="hyp_cause_outcome",
        source_event_id=cause.id,
        target_event_id=outcome.id,
        relation_type=CausalRelationType.CAUSES,
        strength=0.7,
        confidence=0.8,
        reasoning="The operational change directly contributed to the outcome.",
        evidence_article_ids=[article.id],
        discovered_by_question_ids=[question.id],
    )
    test_db.save(Question, question)
    test_db.save(Article, article)
    test_db.save(Event, cause)
    test_db.save(Event, outcome)
    test_db.save(CausalHypothesis, hypothesis)


def test_graph_audit_accepts_grounded_connected_dag(test_db):
    _save_grounded_graph(test_db)

    result = GraphAuditPipeline(str(test_db.db_path)).audit_question(
        "q_graph_audit"
    )

    assert result["status"] == "pass"
    assert result["actual_outcome_event_id"] == "evt_outcome"


def test_graph_audit_rejects_node_without_question_evidence(test_db):
    _save_grounded_graph(test_db)
    cause = test_db.get(Event, "evt_cause")
    cause.article_ids = []
    test_db.save(Event, cause)

    result = GraphAuditPipeline(str(test_db.db_path)).audit_question(
        "q_graph_audit"
    )

    assert result["status"] == "fail"
    assert any("no cutoff-safe evidence" in issue for issue in result["issues"])


def test_graph_audit_rejects_cycle(test_db):
    _save_grounded_graph(test_db)
    test_db.save(
        CausalHypothesis,
        CausalHypothesis(
            id="hyp_cycle",
            source_event_id="evt_outcome",
            target_event_id="evt_cause",
            relation_type=CausalRelationType.CAUSES,
            strength=0.5,
            confidence=0.5,
            reasoning="Invalid reverse link used to verify cycle rejection.",
            discovered_by_question_ids=["q_graph_audit"],
        ),
    )

    result = GraphAuditPipeline(str(test_db.db_path)).audit_question(
        "q_graph_audit"
    )

    assert result["status"] == "fail"
    assert "Cycle detected in the question causal graph." in result["issues"]
