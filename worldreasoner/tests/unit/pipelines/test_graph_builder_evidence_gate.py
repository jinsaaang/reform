"""Regression tests for the graph builder's evidence hard gate."""

from datetime import datetime, timezone

from scripts.finance.run_dag_sample import export_question_graph
from src.config.pipeline import EvidenceSatisfactionConfig
from src.core.database import GenericDatabase
from src.domain.models import (
    Article,
    CausalHypothesis,
    Domain,
    Event,
    EventOutcomeImpact,
    Question,
)
from src.pipelines.graph_builder.pipeline import GraphBuilderPipeline
from src.services.question_monitor_service import EvidenceSatisfaction
from tests.conftest import create_test_question


def test_graph_builder_rejects_underfilled_evidence_without_calling_model(tmp_path):
    db_path = tmp_path / "graph-gate.sqlite"
    db = GenericDatabase(str(db_path))
    db.create_table(Question)
    db.create_table(Article)
    db.create_table(Event)
    db.create_table(CausalHypothesis)
    db.create_table(EventOutcomeImpact)
    question = create_test_question(
        id="underfilled",
        resolution_date=datetime(2026, 4, 1, tzinfo=timezone.utc),
        causal_explanation="A completed explanation is not enough by itself.",
    )
    db.save(Question, question)
    db.save(
        Article,
        Article(
            id="only_article",
            title="One valid article",
            content="Evidence content. " * 10,
            url="https://example.com/one",
            source="Example",
            published_date=datetime(2026, 3, 1, tzinfo=timezone.utc),
            domain=Domain.FINANCE,
            collected_for_question_id=question.id,
        ),
    )

    pipeline = GraphBuilderPipeline(
        db_path=str(db_path),
        min_evidence_articles=2,
    )

    assert pipeline._process_single_question(question) is False
    saved = db.get(Question, question.id)
    assert saved.graph_built is False
    assert "1 < 2" in saved.graph_build_error


def test_graph_builder_rejects_missing_actual_outcome_without_calling_model(
    tmp_path, monkeypatch
):
    db_path = tmp_path / "missing-outcome.sqlite"
    db = GenericDatabase(str(db_path))
    for model in (Question, Article, Event, CausalHypothesis, EventOutcomeImpact):
        db.create_table(model)
    question = create_test_question(
        id="missing-outcome",
        ground_truth=True,
        resolution_date=datetime(2026, 4, 1, tzinfo=timezone.utc),
        causal_explanation="A complete explanation with evidence is available.",
    )
    db.save(Question, question)
    db.save(
        Article,
        Article(
            id="article-one",
            title="One valid article for the resolved question",
            content="Evidence content. " * 10,
            url="https://example.com/article-one",
            source="Example",
            published_date=datetime(2026, 3, 1, tzinfo=timezone.utc),
            domain=Domain.FINANCE,
            collected_for_question_id=question.id,
        ),
    )

    def fail_if_model_is_created(**_kwargs):
        raise AssertionError("Graph model should not run without an actual outcome")

    monkeypatch.setattr(
        "src.pipelines.graph_builder.pipeline.GraphBuilderAgentFactory.create",
        fail_if_model_is_created,
    )

    pipeline = GraphBuilderPipeline(
        db_path=str(db_path),
        min_evidence_articles=1,
    )

    assert pipeline._process_single_question(question) is False
    saved = db.get(Question, question.id)
    assert saved.graph_built is False
    assert "exactly one actual outcome" in saved.graph_build_error


def test_graph_builder_repairs_initial_structural_failure_once(tmp_path, monkeypatch):
    db_path = tmp_path / "repair-structural-failure.sqlite"
    db = GenericDatabase(str(db_path))
    for model in (Question, Article, Event, CausalHypothesis, EventOutcomeImpact):
        db.create_table(model)

    question = create_test_question(
        id="repair-structural-failure",
        ground_truth=True,
        resolution_date=datetime(2026, 4, 1, tzinfo=timezone.utc),
        causal_explanation="A complete explanation with evidence is available.",
        outcome_event_ids=["actual-outcome"],
    )
    db.save(Question, question)
    db.save(
        Event,
        Event(
            id="actual-outcome",
            title="Actual outcome occurs",
            description="The actual outcome for this test question occurred.",
            event_type="outcome",
            domain=Domain.GENERAL,
            occurred_date=datetime(2026, 4, 1, tzinfo=timezone.utc),
            status="occurred",
            extracted_for_question_id=question.id,
            is_outcome=True,
            is_actual_outcome=True,
        ),
    )

    graph_checks = iter(
        [
            EvidenceSatisfaction(
                is_satisfied=False,
                graph_depth=0,
                article_count=0,
                hypothesis_count=0,
                missing_requirements=["events (6 < 8)"],
            ),
            EvidenceSatisfaction(
                is_satisfied=True,
                graph_depth=3,
                article_count=0,
                hypothesis_count=8,
                missing_requirements=[],
            ),
        ]
    )
    monkeypatch.setattr(
        "src.pipelines.graph_builder.pipeline.QuestionMonitorService.has_sufficient_evidence_articles",
        lambda _self, _question_id: True,
    )
    monkeypatch.setattr(
        "src.pipelines.graph_builder.pipeline.QuestionMonitorService.check_graph_satisfaction",
        lambda _self, _question_id: next(graph_checks),
    )
    monkeypatch.setattr(
        "src.pipelines.graph_builder.pipeline.GraphAuditPipeline.audit_question",
        lambda _self, _question_id: {"status": "pass", "issues": []},
    )

    class RecordingAgent:
        def __init__(self, *, repairs=False):
            self.prompts = []
            self.repairs = repairs

        def run(self, prompt):
            self.prompts.append(prompt)
            if self.repairs:
                saved = db.get(Question, question.id)
                saved.graph_built = True
                db.save(Question, saved)

    initial_agent = RecordingAgent()
    repair_agent = RecordingAgent(repairs=True)
    created_agents = iter([initial_agent, repair_agent])
    monkeypatch.setattr(
        "src.pipelines.graph_builder.pipeline.GraphBuilderAgentFactory.create",
        lambda **_kwargs: next(created_agents),
    )

    pipeline = GraphBuilderPipeline(
        db_path=str(db_path),
        min_evidence_articles=1,
        min_graph_depth=3,
        min_events=8,
    )

    assert pipeline._process_single_question(question) is True
    assert len(initial_agent.prompts) == 1
    assert len(repair_agent.prompts) == 1
    assert "events (6 < 8)" in repair_agent.prompts[0]
    assert db.get(Question, question.id).graph_build_error is None


def test_graph_builder_normalizes_missing_agent_completion_flag(
    tmp_path, monkeypatch
):
    db_path = tmp_path / "normalize-valid-graph.sqlite"
    db = GenericDatabase(str(db_path))
    for model in (Question, Article, Event, CausalHypothesis, EventOutcomeImpact):
        db.create_table(model)

    question = create_test_question(
        id="normalize-valid-graph",
        ground_truth=True,
        resolution_date=datetime(2026, 4, 1, tzinfo=timezone.utc),
        causal_explanation="A complete explanation with evidence is available.",
        outcome_event_ids=["actual-outcome"],
    )
    db.save(Question, question)
    db.save(
        Event,
        Event(
            id="actual-outcome",
            title="Actual outcome occurs",
            description="The actual outcome for this test question occurred.",
            event_type="outcome",
            domain=Domain.GENERAL,
            occurred_date=datetime(2026, 4, 1, tzinfo=timezone.utc),
            status="occurred",
            extracted_for_question_id=question.id,
            is_outcome=True,
            is_actual_outcome=True,
        ),
    )
    monkeypatch.setattr(
        "src.pipelines.graph_builder.pipeline.QuestionMonitorService.has_sufficient_evidence_articles",
        lambda _self, _question_id: True,
    )
    monkeypatch.setattr(
        "src.pipelines.graph_builder.pipeline.QuestionMonitorService.check_graph_satisfaction",
        lambda _self, _question_id: EvidenceSatisfaction(
            is_satisfied=True,
            graph_depth=3,
            article_count=10,
            hypothesis_count=8,
            missing_requirements=[],
        ),
    )
    monkeypatch.setattr(
        "src.pipelines.graph_builder.pipeline.GraphAuditPipeline.audit_question",
        lambda _self, _question_id: {"status": "pass", "issues": []},
    )

    class ValidGraphWithoutFinalMark:
        def run(self, _prompt):
            return None

    create_calls = []

    def create_agent(**_kwargs):
        create_calls.append(True)
        return ValidGraphWithoutFinalMark()

    monkeypatch.setattr(
        "src.pipelines.graph_builder.pipeline.GraphBuilderAgentFactory.create",
        create_agent,
    )

    pipeline = GraphBuilderPipeline(
        db_path=str(db_path),
        min_evidence_articles=1,
        min_graph_depth=3,
        min_events=8,
    )

    assert pipeline._process_single_question(question) is True
    assert len(create_calls) == 1
    saved = db.get(Question, question.id)
    assert saved.graph_built is True
    assert saved.graph_build_error is None


def test_graph_builder_reuses_valid_persisted_graph_before_model_call(
    tmp_path, monkeypatch
):
    db_path = tmp_path / "reuse-valid-persisted-graph.sqlite"
    db = GenericDatabase(str(db_path))
    for model in (Question, Article, Event, CausalHypothesis, EventOutcomeImpact):
        db.create_table(model)

    question = create_test_question(
        id="reuse-valid-persisted-graph",
        ground_truth=True,
        resolution_date=datetime(2026, 4, 1, tzinfo=timezone.utc),
        causal_explanation="A complete explanation with evidence is available.",
        outcome_event_ids=["actual-outcome"],
    )
    source = Event(
        id="source-event",
        title="Upstream driver",
        description="A grounded upstream driver occurred.",
        event_type="indicator",
        domain=Domain.FINANCE,
        occurred_date=datetime(2026, 3, 1, tzinfo=timezone.utc),
        status="occurred",
        extracted_for_question_id=question.id,
    )
    outcome = Event(
        id="actual-outcome",
        title="Actual outcome occurs",
        description="The actual outcome for this test question occurred.",
        event_type="outcome",
        domain=Domain.FINANCE,
        occurred_date=datetime(2026, 4, 1, tzinfo=timezone.utc),
        status="occurred",
        extracted_for_question_id=question.id,
        is_outcome=True,
        is_actual_outcome=True,
    )
    db.save(Question, question)
    db.save(Event, source)
    db.save(Event, outcome)
    db.save(
        CausalHypothesis,
        CausalHypothesis(
            id="persisted-hypothesis",
            source_event_id=source.id,
            target_event_id=outcome.id,
            relation_type="causes",
            strength=0.8,
            confidence=0.8,
            reasoning="The upstream driver contributed to the outcome.",
            discovered_by_question_ids=[question.id],
        ),
    )
    monkeypatch.setattr(
        "src.pipelines.graph_builder.pipeline.QuestionMonitorService.has_sufficient_evidence_articles",
        lambda _self, _question_id: True,
    )
    monkeypatch.setattr(
        "src.pipelines.graph_builder.pipeline.QuestionMonitorService.check_graph_satisfaction",
        lambda _self, _question_id: EvidenceSatisfaction(
            is_satisfied=True,
            graph_depth=3,
            article_count=10,
            hypothesis_count=8,
            missing_requirements=[],
        ),
    )
    monkeypatch.setattr(
        "src.pipelines.graph_builder.pipeline.GraphAuditPipeline.audit_question",
        lambda _self, _question_id: {"status": "pass", "issues": []},
    )
    monkeypatch.setattr(
        "src.pipelines.graph_builder.pipeline.GraphBuilderAgentFactory.create",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("A valid persisted graph must not call the model")
        ),
    )

    pipeline = GraphBuilderPipeline(
        db_path=str(db_path),
        min_evidence_articles=1,
        min_graph_depth=3,
        min_events=8,
    )

    assert pipeline._process_single_question(question) is True
    saved = db.get(Question, question.id)
    assert saved.graph_built is True
    assert saved.graph_build_error is None


def test_graph_builder_starts_persisted_invalid_graph_with_targeted_repair(
    tmp_path, monkeypatch
):
    db_path = tmp_path / "repair-persisted-graph.sqlite"
    db = GenericDatabase(str(db_path))
    for model in (Question, Article, Event, CausalHypothesis, EventOutcomeImpact):
        db.create_table(model)

    question = create_test_question(
        id="repair-persisted-graph",
        ground_truth=True,
        resolution_date=datetime(2026, 4, 1, tzinfo=timezone.utc),
        causal_explanation="A complete explanation with evidence is available.",
        outcome_event_ids=["actual-outcome"],
    )
    source = Event(
        id="unsupported-source",
        title="Unsupported upstream node",
        description="A persisted node needs targeted repair.",
        event_type="indicator",
        domain=Domain.FINANCE,
        occurred_date=datetime(2026, 3, 1, tzinfo=timezone.utc),
        status="occurred",
        extracted_for_question_id=question.id,
    )
    outcome = Event(
        id="actual-outcome",
        title="Actual outcome occurs",
        description="The actual outcome for this test question occurred.",
        event_type="outcome",
        domain=Domain.FINANCE,
        occurred_date=datetime(2026, 4, 1, tzinfo=timezone.utc),
        status="occurred",
        extracted_for_question_id=question.id,
        is_outcome=True,
        is_actual_outcome=True,
    )
    db.save(Question, question)
    db.save(Event, source)
    db.save(Event, outcome)
    db.save(
        CausalHypothesis,
        CausalHypothesis(
            id="persisted-invalid-hypothesis",
            source_event_id=source.id,
            target_event_id=outcome.id,
            relation_type="causes",
            strength=0.8,
            confidence=0.8,
            reasoning="The node was linked before its evidence was validated.",
            discovered_by_question_ids=[question.id],
        ),
    )
    graph_checks = iter(
        [
            EvidenceSatisfaction(
                is_satisfied=False,
                graph_depth=1,
                article_count=10,
                hypothesis_count=1,
                missing_requirements=["graph_depth (1 < 3)"],
            ),
            EvidenceSatisfaction(
                is_satisfied=True,
                graph_depth=3,
                article_count=10,
                hypothesis_count=8,
                missing_requirements=[],
            ),
        ]
    )
    audit_checks = iter(
        [
            {
                "status": "fail",
                "issues": [
                    "Event unsupported-source has no cutoff-safe evidence for this question."
                ],
            },
            {"status": "pass", "issues": []},
        ]
    )
    monkeypatch.setattr(
        "src.pipelines.graph_builder.pipeline.QuestionMonitorService.has_sufficient_evidence_articles",
        lambda _self, _question_id: True,
    )
    monkeypatch.setattr(
        "src.pipelines.graph_builder.pipeline.QuestionMonitorService.check_graph_satisfaction",
        lambda _self, _question_id: next(graph_checks),
    )
    monkeypatch.setattr(
        "src.pipelines.graph_builder.pipeline.GraphAuditPipeline.audit_question",
        lambda _self, _question_id: next(audit_checks),
    )

    prompts = []

    class RepairingAgent:
        def run(self, prompt):
            prompts.append(prompt)
            saved = db.get(Question, question.id)
            saved.graph_built = True
            db.save(Question, saved)

    monkeypatch.setattr(
        "src.pipelines.graph_builder.pipeline.GraphBuilderAgentFactory.create",
        lambda **_kwargs: RepairingAgent(),
    )
    pipeline = GraphBuilderPipeline(
        db_path=str(db_path),
        min_evidence_articles=1,
        min_graph_depth=3,
        min_events=8,
    )

    assert pipeline._process_single_question(question) is True
    assert len(prompts) == 1
    assert "failed structural or deterministic validation" in prompts[0]
    assert "unsupported-source has no cutoff-safe evidence" in prompts[0]


def test_graph_export_excludes_shared_article_at_cutoff(tmp_path):
    db_path = tmp_path / "graph-export-cutoff.sqlite"
    db = GenericDatabase(str(db_path))
    question = create_test_question(
        id="target",
        resolution_date=datetime(2026, 4, 1, tzinfo=timezone.utc),
    )
    db.create_table(Question)
    db.create_table(Article)
    db.create_table(Event)
    db.create_table(CausalHypothesis)
    db.create_table(EventOutcomeImpact)
    db.save(Question, question)
    db.save(
        Article,
        Article(
            id="late_shared",
            title="Article at cutoff",
            content="Late article. " * 10,
            url="https://example.com/late",
            source="Example",
            published_date=question.resolution_date,
            domain=Domain.FINANCE,
            collected_for_question_id="another-question",
            metadata={"related_question_ids": [question.id]},
        ),
    )

    payload = export_question_graph(
        db,
        question.id,
        EvidenceSatisfactionConfig(min_articles=1),
    )

    assert payload["evidence"]["article_count"] == 0
    assert payload["evidence"]["articles"] == []
