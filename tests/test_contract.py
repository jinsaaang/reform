from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest
from hgf.boundary import _neutral_boundary_payload, _validate_boundary_forecast
from hgf_e2e_topology.core import _schema, _validate
from hgf_e2e_topology.pipeline import compile_worked_reasoning_check
from hgf_e2e_topology_sidecar.run import _stage

from hgf import exemplar


def _step(step_type: str, source_id: str) -> dict[str, object]:
    return {
        "step_type": step_type,
        "statement": f"{step_type} statement",
        "evidence_ids": ["e1"],
        "effect_on_target": "up",
        "source_id": source_id,
    }


def _payload(steps: list[dict[str, object]]) -> dict[str, object]:
    return {
        "target_semantics": "exact target",
        "selected_evidence_ids": ["e1"],
        "evidence_fit": {
            "metric_match": "direct",
            "horizon_match": "direct",
            "magnitude_support": "partial",
            "assessment": "Evidence matches the target.",
        },
        "current_new_factors": [],
        "causal_balance": {
            "favored_direction": "up",
            "used_path_ids": [],
            "assessment": "Current evidence favors an increase.",
        },
        "magnitude_readiness": {
            "support": "direction_only",
            "evidence_ids": ["e1"],
            "assessment": "Only direction is supported.",
        },
        "reasoning_steps": steps,
        "counterevidence": "Counterevidence was checked.",
        "uncertainty": "Magnitude remains uncertain.",
    }


def test_worked_check_keeps_reasoning_but_excludes_answer_fields() -> None:
    worked = {
        "task_signature": {"target_operation": "change"},
        "target_semantics": "Historical target",
        "expert_reasoning": [
            "Demand increased (art_example_1).",
            "Higher demand tightened supply (art_example_1, art_example_2).",
        ],
        "counterevidence": "Supply could recover.",
        "uncertainty": "The magnitude is uncertain.",
        "dag_derived_lesson": "Check demand before inferring price pressure.",
        "prospective_target_estimate": "above range",
        "option_mapping": "Choose above range.",
    }
    check = compile_worked_reasoning_check(worked)
    assert check["reasoning_sequence"] == [
        "Demand increased (historical evidence).",
        "Higher demand tightened supply (historical evidence, historical evidence).",
    ]
    assert "prospective_target_estimate" not in check
    assert "option_mapping" not in check
    assert "art_example" not in str(check)


def test_reasoning_uses_same_ten_step_budget_as_baselines() -> None:
    reasoning_schema = _schema(["p1"])["schema"]["properties"]["reasoning_steps"]
    assert reasoning_schema["maxItems"] == 10


def test_path_can_be_used_without_a_hard_active_path_contract() -> None:
    steps = [
        _step("baseline", "TARGET_CONTRACT"),
        _step("mechanism", "p1"),
        _step("counterevidence", "CURRENT_NEW"),
        _step("target_bridge", "TARGET_CONTRACT"),
    ]
    payload = _payload(steps)
    payload["causal_balance"]["used_path_ids"] = ["p1"]
    _, errors = _validate(
        payload,
        path_ids=["p1"],
        evidence_ids={"e1"},
        graph_evidence_ids={"e1"},
    )
    assert errors == []
    assert payload["reasoning_steps"] == steps
    assert payload["causal_balance"]["used_path_ids"] == ["p1"]


def test_validator_projects_missing_protocol_steps_from_returned_fields() -> None:
    steps = [
        _step("driver", "CURRENT_NEW"),
        _step("mechanism", "p1"),
        _step("counterevidence", "CURRENT_NEW"),
    ]
    payload = _payload(steps)
    _, errors = _validate(
        payload,
        path_ids=["p1"],
        evidence_ids={"e1"},
        graph_evidence_ids={"e1"},
    )
    assert errors == []
    assert payload["reasoning_steps"][0]["step_type"] == "baseline"
    assert payload["reasoning_steps"][-1]["step_type"] == "target_bridge"
    assert payload["trace_normalization"]["applied"] is True
    assert payload["trace_normalization"]["probability_modified"] is False


def test_multiple_path_ids_are_normalized_as_audit_metadata() -> None:
    steps = [
        _step("baseline", "TARGET_CONTRACT"),
        _step("mechanism", "p1, p2"),
        _step("counterevidence", "CURRENT_NEW"),
        _step("target_bridge", "TARGET_CONTRACT"),
    ]
    payload = _payload(steps)
    _, errors = _validate(
        payload,
        path_ids=["p1", "p2"],
        evidence_ids={"e1"},
        graph_evidence_ids={"e1"},
    )
    assert errors == []
    assert payload["reasoning_steps"][1]["source_id"] == "p1"
    assert payload["causal_balance"]["used_path_ids"] == ["p1", "p2"]


def test_node_and_current_case_aliases_are_normalized() -> None:
    steps = [
        _step("baseline", "TARGET_CONTRACT"),
        _step("driver", "CURRENT_CASE"),
        _step("mechanism", "D1:target_bridge"),
        _step("target_bridge", "TARGET_CONTRACT"),
    ]
    payload = _payload(steps)
    _, errors = _validate(
        payload,
        path_ids=["D1:path_1", "D1:path_2"],
        evidence_ids={"e1"},
        graph_evidence_ids={"e1"},
    )
    assert errors == []
    assert payload["reasoning_steps"][1]["source_id"] == "CURRENT_NEW"
    assert payload["reasoning_steps"][2]["source_id"] == "D1:path_1"
    assert payload["causal_balance"]["used_path_ids"] == [
        "D1:path_1",
        "D1:path_2",
    ]


def test_unknown_audit_labels_fall_back_without_claiming_dag_use() -> None:
    steps = [
        _step("baseline", "SOME_TARGET_LABEL"),
        _step("driver", "CURRENT_EVIDENCE_LEDGER"),
        _step("mechanism", "UNREGISTERED_STRUCTURE"),
        _step("target_bridge", "FINAL_MAPPING"),
    ]
    payload = _payload(steps)
    _, errors = _validate(
        payload,
        path_ids=["p1"],
        evidence_ids={"e1"},
        graph_evidence_ids={"e1"},
    )
    assert errors == []
    assert [item["source_id"] for item in payload["reasoning_steps"]] == [
        "TARGET_CONTRACT",
        "CURRENT_NEW",
        "CURRENT_NEW",
        "TARGET_CONTRACT",
    ]
    assert payload["causal_balance"]["used_path_ids"] == []
    assert payload["trace_normalization"]["probability_modified"] is False


def test_boundary_normalizes_only_redundant_labels() -> None:
    options = [
        "below recent range",
        "within recent range",
        "above recent range",
    ]
    contract = {
        "target_metric": "monthly change",
        "target_period": "2025-10",
        "change_unit": "percent",
        "intervals": {
            "below recent range": "(-infinity, -1.0)",
            "within recent range": "[-1.0, 1.0)",
            "above recent range": "[1.0, +infinity)",
        },
    }
    payload = _neutral_boundary_payload(options=options, contract=contract)
    payload["mapped_option"] = "above recent range"
    payload["prediction"] = "above recent range"
    probabilities_before = list(payload["option_probabilities"])
    _, errors = _validate_boundary_forecast(
        payload,
        options=options,
        contract=contract,
        evidence_ids=set(),
        reasoning_policy="boundary_only",
        validation_policy="strict",
    )
    assert errors == []
    assert payload["mapped_option"] == "within recent range"
    assert payload["prediction"] == "within recent range"
    assert payload["option_probabilities"] == probabilities_before


def test_repair_budget_is_four_medium_attempts() -> None:
    source = inspect.getsource(exemplar._call_with_repair)
    assert "range(4)" in source
    assert 'reasoning_effort_override="minimal"' not in source


def test_empty_reasoning_payload_is_not_silently_projected() -> None:
    _, errors = _validate(
        {},
        path_ids=["p1"],
        evidence_ids={"e1"},
        graph_evidence_ids={"e1"},
    )
    assert errors
    assert "missing required reasoning fields" in errors[0]


def test_length_terminated_json_call_is_an_execution_failure() -> None:
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason="length",
                message=SimpleNamespace(content=None),
            )
        ],
        usage=None,
    )
    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **_: response)
        )
    )
    with pytest.raises(ValueError, match="invalid/truncated JSON"):
        exemplar._call_json(
            client,
            model="minimax/minimax-m2.5",
            system="system",
            prompt="prompt",
            schema={"name": "test", "strict": True, "schema": {}},
            seed=0,
            max_tokens=32,
        )


def test_raw_recorder_names_the_current_reasoning_stage() -> None:
    messages = [
        {"role": "system", "content": "Return reasoning only."},
        {
            "role": "user",
            "content": "Build a complete current forecast reasoning trace.",
        },
    ]
    assert _stage(messages) == "procedural_reasoning"
