"""Tests for the resolved-DAG memory compiler."""

import json
from types import SimpleNamespace

import pytest

from forecaster.memory.dag import (
    compile_forecast_memory,
    compile_search_memory,
    load_memory_graphs,
    select_relevant_memory_graphs,
)


def graph_payload(question_id: str, satisfied: bool = True):
    return {
        "evidence": {"satisfied": True},
        "question": {
            "id": question_id,
            "question_text": f"Past question {question_id}",
            "ground_truth": "2.0% to <2.5%",
            "resolution_date": "2026-02-13T23:59:59Z",
        },
        "graph": {
            "built": satisfied,
            "satisfied": satisfied,
            "validation": {"status": "pass" if satisfied else "fail"},
            "metrics": {"event_count": 2, "hypothesis_count": 1},
            "nodes": [
                {
                    "id": "factor",
                    "title": "Energy-price disinflation",
                    "occurred_date": "2026-01-15T00:00:00Z",
                    "event_type": "economic",
                    "is_actual_outcome": False,
                },
                {
                    "id": "outcome",
                    "title": "CPI lands in resolved bucket",
                    "occurred_date": "2026-02-13T00:00:00Z",
                    "event_type": "outcome",
                    "is_actual_outcome": True,
                },
            ],
            "edges": [
                {
                    "source_event_id": "factor",
                    "target_event_id": "outcome",
                    "relation_type": "causes",
                    "strength": 0.8,
                    "confidence": 0.9,
                    "reasoning": (
                        "Lower energy prices reduced headline CPI into the "
                        "2.0% to <2.5% bucket."
                    ),
                }
            ],
        },
    }


def test_relevant_memory_selection_prefers_same_entity_and_is_bounded():
    graphs = [graph_payload(f"q_{index}") for index in range(10)]
    for index, graph in enumerate(graphs):
        graph["question"]["metadata"] = {
            "finfactorbench": {
                "entity": "Apple" if index in (3, 7) else f"Entity {index}",
                "subdomain": "earnings",
                "original_domain": "corporate_earnings",
            }
        }
    question = SimpleNamespace(
        question_text="Will Apple report higher quarterly revenue?",
        metadata={
            "finfactorbench": {
                "entity": "Apple",
                "subdomain": "earnings",
                "original_domain": "corporate_earnings",
            }
        },
    )

    selected = select_relevant_memory_graphs(graphs, question, limit=3)

    assert len(selected) == 3
    assert [item["question"]["id"] for item in selected[:2]] == ["q_3", "q_7"]


def test_search_view_excludes_all_outcome_typed_nodes():
    payload = graph_payload("q_outcomes")
    payload["graph"]["nodes"].append(
        {
            "id": "alternative-outcome",
            "title": "Option 3: 2.5% to <3.0%",
            "occurred_date": "2026-02-13T00:00:00Z",
            "event_type": "outcome",
            "is_actual_outcome": False,
        }
    )

    search_memory = json.loads(compile_search_memory([payload]))

    serialized = json.dumps(search_memory)
    assert "Option 3: 2.5% to <3.0%" not in serialized


def test_load_memory_graphs_follows_authorized_manifest_order(tmp_path):
    for question_id in ("q_1", "q_2"):
        (tmp_path / f"{question_id}.json").write_text(
            json.dumps(graph_payload(question_id)), encoding="utf-8"
        )

    graphs = load_memory_graphs(tmp_path, ["q_2", "q_1"])

    assert [payload["question"]["id"] for payload in graphs] == ["q_2", "q_1"]


def test_load_memory_graphs_rejects_missing_or_unsatisfied_graphs(tmp_path):
    with pytest.raises(FileNotFoundError, match="q_missing"):
        load_memory_graphs(tmp_path, ["q_missing"])

    (tmp_path / "q_bad.json").write_text(
        json.dumps(graph_payload("q_bad", satisfied=False)), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="not quality-satisfied"):
        load_memory_graphs(tmp_path, ["q_bad"])

    invalid = graph_payload("q_invalid")
    invalid["graph"]["validation"] = {"status": "fail"}
    (tmp_path / "q_invalid.json").write_text(
        json.dumps(invalid), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="failed deterministic audit"):
        load_memory_graphs(tmp_path, ["q_invalid"])


def test_compile_memories_expose_role_specific_views_without_past_outcome():
    graphs = [graph_payload("q_1")]

    search_memory = json.loads(compile_search_memory(graphs))
    forecast_memory = json.loads(compile_forecast_memory(graphs))

    assert search_memory["view"] == "search_factor_catalog"
    family = search_memory["factor_families"][0]
    assert family["factor_family"] == "energy_and_transport"
    assert "Energy-price disinflation" in family["example_factors"]
    assert "past_outcome" not in json.dumps(search_memory)
    assert "causal_edges" not in search_memory

    assert forecast_memory["view"] == "forecast_causal_mechanisms"
    mechanism = forecast_memory["examples"][0]["mechanism_patterns"][0]
    assert mechanism["source_factor"] == "Energy-price disinflation"
    assert mechanism["target_mechanism"] == "TARGET_OUTCOME"
    assert "2.0% to <2.5%" not in json.dumps(forecast_memory)
    assert "CPI lands in resolved bucket" not in json.dumps(forecast_memory)
    assert "[PAST_OUTCOME_REDACTED]" in mechanism["reasoning_pattern"]
