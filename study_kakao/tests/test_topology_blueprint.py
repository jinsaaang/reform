from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

from hgf.topology_blueprint import (
    compile_topology_blueprint,
    historical_specific_counts,
    sanitize_topology_blueprint,
    validate_topology_blueprint,
)


def _graph() -> dict:
    return {
        "actual_outcome_event_id": "outcome",
        "graph": {
            "nodes": [
                {
                    "id": "outcome",
                    "label": "Resolved outcome: below recent range",
                    "event_type": "outcome",
                    "is_outcome": True,
                    "is_actual_outcome": True,
                    "support_level": "observed",
                },
                {
                    "id": "target_value",
                    "label": "January index change was -0.471",
                    "event_type": "indicator",
                    "is_outcome": False,
                    "is_actual_outcome": False,
                    "support_level": "observed",
                },
                {
                    "id": "driver",
                    "label": "Debt-ceiling conditions softened tightening",
                    "event_type": "external_shock",
                    "is_outcome": False,
                    "is_actual_outcome": False,
                    "support_level": "observed",
                },
                {
                    "id": "mediator",
                    "label": "Liquidity supported financial conditions",
                    "event_type": "indicator",
                    "is_outcome": False,
                    "is_actual_outcome": False,
                    "support_level": "evidence_synthesized",
                },
                {
                    "id": "counter",
                    "label": "Credit-crunch concerns remained",
                    "event_type": "external_shock",
                    "is_outcome": False,
                    "is_actual_outcome": False,
                    "support_level": "observed",
                },
            ],
            "edges": [
                {
                    "id": "edge_driver",
                    "source": "driver",
                    "target": "mediator",
                    "relationship": "supports",
                    "support_level": "evidence_synthesized",
                    "rationale": "Softer tightening may support liquidity.",
                    "article_ids": ["article_1"],
                },
                {
                    "id": "edge_counter",
                    "source": "counter",
                    "target": "mediator",
                    "relationship": "counteracts",
                    "support_level": "background_hypothesis",
                    "rationale": "Credit stress may offset liquidity support.",
                    "article_ids": ["article_2"],
                },
                {
                    "id": "edge_target",
                    "source": "mediator",
                    "target": "target_value",
                    "relationship": "contributes_to",
                    "support_level": "background_hypothesis",
                    "rationale": "Conditions contributed to the -0.471 value.",
                    "article_ids": ["article_3"],
                },
                {
                    "id": "edge_mapping",
                    "source": "target_value",
                    "target": "outcome",
                    "relationship": "maps_to",
                    "support_level": "observed",
                    "rationale": "The value maps to below recent range.",
                    "article_ids": [],
                },
            ],
        },
    }


def _question() -> SimpleNamespace:
    return SimpleNamespace(
        id="memory_question",
        ground_truth="below recent range",
        options=[
            "below recent range",
            "within recent range",
            "above recent range",
        ],
        metadata={
            "finance": {
                "target_metric": "monthly financial-stress-index change",
            }
        },
    )


def _blueprint() -> dict:
    return compile_topology_blueprint(
        graph_payload=_graph(),
        question=_question(),
        audit={"status": "pass", "caveats": []},
        source_graph=Path("data/dags/memory_question/graph.json"),
    )


def _direct_outcome_graph() -> dict:
    graph = copy.deepcopy(_graph())
    graph["graph"]["nodes"] = [
        node
        for node in graph["graph"]["nodes"]
        if node["id"] != "target_value"
    ]
    graph["graph"]["edges"] = [
        edge
        for edge in graph["graph"]["edges"]
        if edge["id"] != "edge_mapping"
    ]
    target_edge = next(
        edge
        for edge in graph["graph"]["edges"]
        if edge["id"] == "edge_target"
    )
    target_edge["target"] = "outcome"
    target_edge["relationship"] = "correlates"
    return graph


def test_compiler_preserves_branches_without_inventing_sequence() -> None:
    blueprint = _blueprint()
    source_by_checkpoint = {
        item["id"]: (
            item["source_event_ids"][0]
            if item["source_event_ids"]
            else "target_bridge"
        )
        for item in blueprint["checkpoints"]
    }
    emitted_pairs = {
        (
            source_by_checkpoint[edge["source_checkpoint_id"]],
            source_by_checkpoint[edge["target_checkpoint_id"]],
        )
        for edge in blueprint["topology"]["edges"]
    }
    assert emitted_pairs == {
        ("driver", "mediator"),
        ("counter", "mediator"),
        ("mediator", "target_bridge"),
    }
    assert ("mediator", "counter") not in emitted_pairs

    source_paths = {
        tuple(source_by_checkpoint[value] for value in path["checkpoint_ids"])
        for path in blueprint["causal_paths"]
    }
    assert source_paths == {
        ("driver", "mediator", "target_bridge"),
        ("counter", "mediator", "target_bridge"),
    }


def test_compiler_preserves_direct_outcome_parent_as_checkpoint() -> None:
    graph = _direct_outcome_graph()
    blueprint = compile_topology_blueprint(
        graph_payload=graph,
        question=_question(),
        audit={"status": "pass", "caveats": []},
        source_graph=Path("data/dags/memory_question/graph.json"),
    )
    source_by_checkpoint = {
        item["id"]: (
            item["source_event_ids"][0]
            if item["source_event_ids"]
            else "target_bridge"
        )
        for item in blueprint["checkpoints"]
    }
    emitted_pairs = {
        (
            source_by_checkpoint[edge["source_checkpoint_id"]],
            source_by_checkpoint[edge["target_checkpoint_id"]],
        )
        for edge in blueprint["topology"]["edges"]
    }
    assert ("mediator", "target_bridge") in emitted_pairs
    assert validate_topology_blueprint(blueprint, graph)["status"] == "pass"


def test_compiler_redacts_outcome_and_realized_target_value() -> None:
    blueprint = _blueprint()
    source_event_ids = {
        event_id
        for item in blueprint["checkpoints"]
        for event_id in item["source_event_ids"]
    }
    assert "outcome" not in source_event_ids
    assert "target_value" not in source_event_ids

    reusable_text = json.dumps(
        {
            "checkpoints": blueprint["checkpoints"],
            "topology": blueprint["topology"],
            "causal_paths": blueprint["causal_paths"],
        },
        ensure_ascii=False,
    )
    assert "-0.471" not in reusable_text
    assert "below recent range" not in reusable_text


def test_sanitizer_removes_specifics_without_changing_topology() -> None:
    raw = _blueprint()
    raw["checkpoints"][0]["factor"] = (
        "The 10-year yield fell 0.25 percentage points in December 2024"
    )
    raw["topology"]["edges"][0]["rationale"] = (
        "The index may fall by -0.471 after 256,000 jobs on 2024-12-31."
    )

    sanitized = sanitize_topology_blueprint(raw)

    assert sanitized["checkpoints"][0]["factor"] == (
        "The 10-year yield fell [CURRENT_VALUE_REQUIRED] in "
        "[CURRENT_PERIOD_REQUIRED]"
    )
    assert sanitized["topology"]["edges"][0]["rationale"] == (
        "The index may fall by [CURRENT_VALUE_REQUIRED] after "
        "[CURRENT_VALUE_REQUIRED] jobs on [CURRENT_PERIOD_REQUIRED]."
    )
    assert sanitized["topology"]["checkpoint_ids"] == (
        raw["topology"]["checkpoint_ids"]
    )
    assert sanitized["topology"]["root_checkpoint_ids"] == (
        raw["topology"]["root_checkpoint_ids"]
    )
    assert [
        (
            edge["source_checkpoint_id"],
            edge["target_checkpoint_id"],
            edge["relationship"],
            edge["source_edge_ids"],
        )
        for edge in sanitized["topology"]["edges"]
    ] == [
        (
            edge["source_checkpoint_id"],
            edge["target_checkpoint_id"],
            edge["relationship"],
            edge["source_edge_ids"],
        )
        for edge in raw["topology"]["edges"]
    ]
    assert [
        (path["checkpoint_ids"], path["source_edge_ids"])
        for path in sanitized["causal_paths"]
    ] == [
        (path["checkpoint_ids"], path["source_edge_ids"])
        for path in raw["causal_paths"]
    ]
    assert historical_specific_counts(sanitized) == {
        "realized_value_count": 0,
        "absolute_period_count": 0,
    }


def test_sanitized_blueprint_still_passes_topology_validation() -> None:
    sanitized = sanitize_topology_blueprint(_blueprint())
    validation = validate_topology_blueprint(sanitized, _graph())
    assert validation["status"] == "pass"
    assert validation["metrics"]["edge_coverage"] == 1.0
    assert validation["metrics"]["path_precision"] == 1.0
    assert validation["metrics"]["realized_value_count"] == 0
    assert validation["metrics"]["absolute_period_count"] == 0


def test_topology_validator_proves_edge_and_path_fidelity() -> None:
    blueprint = _blueprint()
    validation = validate_topology_blueprint(blueprint, _graph())
    assert validation["status"] == "pass"
    assert validation["metrics"]["edge_coverage"] == 1.0
    assert validation["metrics"]["path_precision"] == 1.0
    assert validation["metrics"]["outcome_event_leak_count"] == 0
    assert validation["metrics"]["outcome_text_leak_count"] == 0


def test_topology_validator_rejects_invented_path_adjacency() -> None:
    blueprint = copy.deepcopy(_blueprint())
    path = blueprint["causal_paths"][0]
    path["checkpoint_ids"].insert(1, blueprint["topology"]["root_checkpoint_ids"][1])
    validation = validate_topology_blueprint(blueprint, _graph())
    assert validation["status"] == "fail"
    assert any(
        "invents adjacency" in error for error in validation["errors"]
    )
