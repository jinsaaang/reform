"""Forecast graph tools initialize the tables they write to."""

import json

from src.core.alias_registry import AliasRegistry
from src.core.database import GenericDatabase
from src.domain.models import Event
from src.domain.models.forecast_graph import ForecastEvent, ForecastHypothesis
from src.tools.reasoning.forecast_causal_reasoner import ForecastCausalReasonerTool
from src.tools.reasoning.forecast_event_identifier import ForecastEventIdentifierTool
from src.tools.reasoning.propose_subgraph import ProposeSubgraphTool


def test_forecast_writers_create_required_tables(tmp_path):
    db_path = tmp_path / "forecast.sqlite"

    ForecastEventIdentifierTool(
        question_db_path=str(db_path),
        forecast_db_path=str(db_path),
        session_id="session-1",
    )
    ForecastCausalReasonerTool(
        forecast_db_path=str(db_path),
        session_id="session-1",
    )

    db = GenericDatabase(str(db_path))
    assert db.get_many(ForecastEvent) == []
    assert db.get_many(ForecastHypothesis) == []


def _forecast_subgraph_tool(db_path):
    db = GenericDatabase(str(db_path))
    db.create_table(Event)
    registry = AliasRegistry()
    event_tool = ForecastEventIdentifierTool(
        question_db_path=str(db_path),
        forecast_db_path=str(db_path),
        session_id="session-1",
    )
    reasoner_tool = ForecastCausalReasonerTool(
        forecast_db_path=str(db_path),
        session_id="session-1",
    )
    return ProposeSubgraphTool(
        event_identifier_tool=event_tool,
        causal_reasoner_tool=reasoner_tool,
        alias_registry=registry,
        db_path=str(db_path),
    )


def _forecast_payload(relation="causes"):
    return {
        "events": [
            {
                "alias": "E1",
                "title": "Energy prices rise",
                "description": "Energy prices rise before the forecast horizon.",
                "domain": "finance",
                "occurred_date": "2026-03-01T00:00:00Z",
            },
            {
                "alias": "E2",
                "title": "Inflation increases",
                "description": "Inflation increases at the forecast horizon.",
                "domain": "finance",
                "occurred_date": "2026-03-31T00:00:00Z",
            },
        ],
        "edges": [
            {
                "source": "E1",
                "target": "E2",
                "relation": relation,
                "strength": 0.7,
                "confidence": 0.8,
                "reasoning": "Higher energy costs feed into consumer prices.",
            }
        ],
    }


def test_batch_tool_reads_forecast_event_ids_from_nested_output(tmp_path):
    db_path = tmp_path / "forecast.sqlite"
    tool = _forecast_subgraph_tool(db_path)

    result = tool.forward(json.dumps(_forecast_payload()))

    assert result.status == "success", result.failed_items
    assert result.events_created == 2
    assert result.edges_created == 1
    assert set(result.alias_map) == {"E1", "E2"}


def test_failed_forecast_batch_rolls_back_forecast_tables(tmp_path):
    db_path = tmp_path / "forecast.sqlite"
    tool = _forecast_subgraph_tool(db_path)

    result = tool.forward(json.dumps(_forecast_payload("not-a-relation")))

    db = GenericDatabase(str(db_path))
    assert result.status == "error"
    assert db.get_many(ForecastEvent) == []
    assert db.get_many(ForecastHypothesis) == []


def test_later_batch_can_reference_existing_forecast_event_id(tmp_path):
    db_path = tmp_path / "forecast.sqlite"
    tool = _forecast_subgraph_tool(db_path)
    first = tool.forward(json.dumps(_forecast_payload()))
    existing_target_id = first.alias_map["E2"]
    second_payload = {
        "events": [
            {
                "alias": "E3",
                "title": "Wage growth remains elevated",
                "description": "Wage growth adds pressure before the target period.",
                "domain": "finance",
                "occurred_date": "2026-03-15T00:00:00Z",
            }
        ],
        "edges": [
            {
                "source": "E3",
                "target": existing_target_id,
                "relation": "amplifies",
                "strength": 0.6,
                "confidence": 0.7,
                "reasoning": "Elevated wage growth can amplify inflation.",
            }
        ],
    }

    second = tool.forward(json.dumps(second_payload))

    assert second.status == "success", second.failed_items
    assert second.events_created == 1
    assert second.edges_created == 1


def test_causal_reasoner_rejects_unknown_forecast_event_ids(tmp_path):
    db_path = tmp_path / "forecast.sqlite"
    tool = ForecastCausalReasonerTool(
        forecast_db_path=str(db_path), session_id="session-1"
    )

    result = tool.forward(
        source_event_id="missing-source",
        target_event_id="missing-target",
        relation_type="causes",
        strength=0.7,
        confidence=0.8,
        reasoning="This edge should not be persisted.",
    )

    db = GenericDatabase(str(db_path))
    assert result.hypothesis_id == "error"
    assert db.get_many(ForecastHypothesis) == []
