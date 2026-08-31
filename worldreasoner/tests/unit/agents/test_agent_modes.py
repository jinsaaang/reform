"""Tests for code-free tool-calling modes used by the finance runner."""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from smolagents import ActionStep
from smolagents.memory import Timing

from tests.conftest import create_test_question
from src.agents.base import BaseAgent, create_llm_model
from src.agents.graph_builder_agent import (
    GraphBuilderAgentFactory,
    _compact_graph_builder_history,
)
from src.agents.hindsight_agent import HindsightAgent
from src.config import Config
from src.config.pipeline import EvidenceSatisfactionConfig
from src.domain.models import Article, Domain, Question


class _CapturedAgent:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.name = kwargs.get("name", "captured")


def test_hindsight_model_override_is_shared_by_both_agents(test_db, monkeypatch):
    question = create_test_question(
        id="model-override-question",
        resolution_date=datetime.now(timezone.utc) - timedelta(days=1),
    )
    test_db.save(Question, question)
    captured_models = []

    def capture_model(config):
        captured_models.append(config.llm.model)
        return SimpleNamespace(model_id=config.llm.model)

    monkeypatch.setattr(
        "src.agents.hindsight_agent.create_llm_model", capture_model
    )
    monkeypatch.setattr("src.agents.hindsight_agent.ToolCallingAgent", _CapturedAgent)
    monkeypatch.setattr(
        BaseAgent, "__init__", lambda self, **kwargs: setattr(self, "base_kwargs", kwargs)
    )

    agent = HindsightAgent(
        config=Config(),
        db_path=str(test_db.db_path),
        question_id=question.id,
        is_code=False,
        evidence_agent_is_code=False,
        model_id="deepseek/deepseek-v4-pro",
    )

    assert captured_models == ["deepseek/deepseek-v4-pro"]
    assert agent.base_kwargs["config"].llm.model == "deepseek/deepseek-v4-pro"
    evidence_agent = agent.base_kwargs["managed_agents"][0]
    assert evidence_agent.kwargs["model"].model_id == "deepseek/deepseek-v4-pro"


def test_hindsight_tool_mode_uses_auto_collect_without_article_code_tool(
    test_db, monkeypatch
):
    question = create_test_question(
        id="finance-question",
        domain=Domain.FINANCE,
        cutoff_date=datetime.now(timezone.utc) - timedelta(days=30),
        resolution_date=datetime.now(timezone.utc) - timedelta(days=1),
    )
    test_db.save(Question, question)

    monkeypatch.setattr(
        "src.agents.hindsight_agent.create_llm_model",
        lambda _config: SimpleNamespace(model_id="google/gemini-2.5-flash"),
    )
    monkeypatch.setattr(
        "src.agents.hindsight_agent.ToolCallingAgent", _CapturedAgent
    )

    def capture_base_init(self, **kwargs):
        self.base_kwargs = kwargs

    monkeypatch.setattr(BaseAgent, "__init__", capture_base_init)

    agent = HindsightAgent(
        config=SimpleNamespace(),
        db_path=str(test_db.db_path),
        question_id=question.id,
        is_code=False,
        evidence_agent_is_code=False,
        domain="finance",
    )

    evidence_agent = agent.base_kwargs["managed_agents"][0]
    evidence_tools = {tool.name: tool for tool in evidence_agent.kwargs["tools"]}
    manager_tool_names = {tool.name for tool in agent.base_kwargs["tools"]}

    assert agent.base_kwargs["is_code"] is False
    assert set(evidence_tools) == {
        "article_inspector",
        "search_coverage",
        "WebSearchTool",
    }
    assert evidence_tools["WebSearchTool"].auto_collect_enabled is True
    assert evidence_tools["WebSearchTool"].domain == "finance"
    assert evidence_tools["WebSearchTool"].enforce_upper_only_dates is True
    assert (
        evidence_tools["WebSearchTool"].coverage_tracker
        is agent.search_coverage_tracker
    )
    assert "article_collector" not in manager_tool_names
    assert "search_coverage" in manager_tool_names
    assert "additional_authorized_imports" not in evidence_agent.kwargs
    assert "executor_kwargs" not in evidence_agent.kwargs


def test_graph_builder_tool_mode_uses_native_tool_calls(test_db, monkeypatch):
    monkeypatch.setattr(
        GraphBuilderAgentFactory,
        "_create_model",
        lambda *_args, **_kwargs: SimpleNamespace(
            model_id="google/gemini-2.5-flash"
        ),
    )
    monkeypatch.setattr(
        "src.agents.graph_builder_agent.ToolCallingAgent", _CapturedAgent
    )

    agent = GraphBuilderAgentFactory.create(
        model_id="google/gemini-2.5-flash",
        db_path=str(test_db.db_path),
        question_id="finance-question",
        agent_mode="tool",
    )

    assert agent.kwargs["max_tool_threads"] == 1
    assert "additional_authorized_imports" not in agent.kwargs
    assert "executor_kwargs" not in agent.kwargs


def test_graph_builder_code_mode_bounds_tool_output_history(test_db, monkeypatch):
    monkeypatch.setattr(
        GraphBuilderAgentFactory,
        "_create_model",
        lambda *_args, **_kwargs: SimpleNamespace(
            model_id="google/gemini-2.5-flash"
        ),
    )
    monkeypatch.setattr("src.agents.graph_builder_agent.CodeAgent", _CapturedAgent)

    agent = GraphBuilderAgentFactory.create(
        model_id="google/gemini-2.5-flash",
        db_path=str(test_db.db_path),
        question_id="finance-question",
        agent_mode="code",
    )

    assert agent.kwargs["max_print_outputs_length"] == 20_000
    assert _compact_graph_builder_history in agent.kwargs["step_callbacks"]


def test_graph_builder_caps_only_pathological_model_output(test_db, monkeypatch):
    model = SimpleNamespace(
        model_id="google/gemini-2.5-flash",
        kwargs={"max_tokens": 100_000},
    )
    monkeypatch.setattr(
        GraphBuilderAgentFactory,
        "_create_model",
        lambda *_args, **_kwargs: model,
    )
    monkeypatch.setattr("src.agents.graph_builder_agent.CodeAgent", _CapturedAgent)

    GraphBuilderAgentFactory.create(
        model_id="google/gemini-2.5-flash",
        db_path=str(test_db.db_path),
        question_id="finance-question",
        agent_mode="code",
        max_output_tokens=24_000,
    )

    assert model.kwargs["max_tokens"] == 24_000


def test_graph_builder_history_compacts_only_intermediate_actions():
    steps = [
        ActionStep(
            step_number=index,
            timing=Timing(start_time=float(index), end_time=float(index + 1)),
            model_output=f"model-output-{index}",
            code_action=f"code-{index}",
            observations=f"observation-{index}",
        )
        for index in range(1, 7)
    ]
    agent = SimpleNamespace(memory=SimpleNamespace(steps=steps))

    _compact_graph_builder_history(steps[-1], agent=agent)

    assert steps[0].model_output == "model-output-1"
    assert steps[1].model_output == "model-output-2"
    assert steps[-2].model_output == "model-output-5"
    assert steps[-1].model_output == "model-output-6"
    for step in steps[2:4]:
        assert step.code_action is None
        assert step.observations is None
        assert "compacted" in step.model_output


def test_openrouter_model_enables_overflow_only_context_compression(monkeypatch):
    captured = {}

    def capture_model(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(model_id=kwargs["model_id"])

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr("src.agents.base.OpenAIServerModel", capture_model)

    model = create_llm_model(
        Config(), model_id="google/gemini-2.5-flash", temperature=0.2
    )

    assert model.model_id == "google/gemini-2.5-flash"
    assert captured["extra_body"] == {
        "plugins": [{"id": "context-compression"}]
    }


def test_hindsight_keeps_search_agent_when_existing_evidence_is_underfilled(
    test_db, monkeypatch
):
    question = create_test_question(
        id="underfilled-question",
        resolution_date=datetime.now(timezone.utc) - timedelta(days=1),
    )
    test_db.save(Question, question)
    test_db.save(
        Article,
        Article(
            id="article-1",
            title="One evidence article for an underfilled question",
            content="Evidence content. " * 20,
            source="test",
            published_date=question.resolution_date - timedelta(days=1),
            domain=Domain.FINANCE,
            collected_for_question_id=question.id,
        ),
    )
    monkeypatch.setattr(
        "src.agents.hindsight_agent.create_llm_model",
        lambda _config: SimpleNamespace(model_id="google/gemini-2.5-flash"),
    )
    monkeypatch.setattr("src.agents.hindsight_agent.ToolCallingAgent", _CapturedAgent)
    monkeypatch.setattr(
        BaseAgent, "__init__", lambda self, **kwargs: setattr(self, "base_kwargs", kwargs)
    )

    agent = HindsightAgent(
        config=SimpleNamespace(),
        db_path=str(test_db.db_path),
        question_id=question.id,
        is_code=False,
        evidence_agent_is_code=False,
        satisfaction_config=EvidenceSatisfactionConfig(min_articles=2),
    )

    assert len(agent.base_kwargs["managed_agents"]) == 1


def test_hindsight_skips_search_agent_only_when_article_target_is_met(
    test_db, monkeypatch
):
    question = create_test_question(
        id="covered-question",
        resolution_date=datetime.now(timezone.utc) - timedelta(days=1),
    )
    test_db.save(Question, question)
    test_db.save(
        Article,
        Article(
            id="article-1",
            title="Evidence article meeting the configured target",
            content="Evidence content. " * 20,
            source="test",
            published_date=question.resolution_date - timedelta(days=1),
            domain=Domain.FINANCE,
            collected_for_question_id=question.id,
        ),
    )
    monkeypatch.setattr(
        "src.agents.hindsight_agent.create_llm_model",
        lambda _config: SimpleNamespace(model_id="google/gemini-2.5-flash"),
    )
    monkeypatch.setattr("src.agents.hindsight_agent.ToolCallingAgent", _CapturedAgent)
    monkeypatch.setattr(
        BaseAgent, "__init__", lambda self, **kwargs: setattr(self, "base_kwargs", kwargs)
    )

    agent = HindsightAgent(
        config=SimpleNamespace(),
        db_path=str(test_db.db_path),
        question_id=question.id,
        is_code=False,
        evidence_agent_is_code=False,
        satisfaction_config=EvidenceSatisfactionConfig(min_articles=1),
    )

    assert agent.base_kwargs["managed_agents"] == []


def test_hindsight_manager_only_mode_skips_search_agent_when_underfilled(
    test_db, monkeypatch
):
    question = create_test_question(
        id="fallback-manager-only-question",
        resolution_date=datetime.now(timezone.utc) - timedelta(days=1),
    )
    test_db.save(Question, question)
    monkeypatch.setattr(
        "src.agents.hindsight_agent.create_llm_model",
        lambda _config: SimpleNamespace(model_id="google/gemini-2.5-flash"),
    )
    monkeypatch.setattr("src.agents.hindsight_agent.ToolCallingAgent", _CapturedAgent)
    monkeypatch.setattr(
        BaseAgent, "__init__", lambda self, **kwargs: setattr(self, "base_kwargs", kwargs)
    )

    agent = HindsightAgent(
        config=SimpleNamespace(),
        db_path=str(test_db.db_path),
        question_id=question.id,
        is_code=False,
        evidence_agent_is_code=False,
        satisfaction_config=EvidenceSatisfactionConfig(min_articles=10),
        enable_evidence_agent=False,
    )

    assert agent.base_kwargs["managed_agents"] == []


def test_graph_builder_rejects_unknown_agent_mode():
    try:
        GraphBuilderAgentFactory.create(
            model_id="google/gemini-2.5-flash",
            agent_mode="unknown",
        )
    except ValueError as exc:
        assert "agent_mode" in str(exc)
    else:
        raise AssertionError("Expected invalid agent_mode to raise ValueError")
