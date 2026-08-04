from __future__ import annotations

from hgf.boundary import _validate_boundary_forecast
from hgf_e2e_topology.core import _schema, _validate
from hgf_e2e_topology.pipeline import compile_worked_reasoning_check


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


def test_incomplete_reasoning_is_rejected_before_audit_defaults() -> None:
    payload = _payload(
        [
            _step("baseline", "TARGET_CONTRACT"),
            _step("driver", "CURRENT_NEW"),
            _step("target_bridge", "TARGET_CONTRACT"),
        ]
    )
    del payload["counterevidence"]
    _, errors = _validate(
        payload,
        path_ids=["p1"],
        evidence_ids={"e1"},
        graph_evidence_ids={"e1"},
    )
    assert errors == ["missing required reasoning fields: ['counterevidence']"]
    assert "counterevidence" not in payload


def test_short_reasoning_and_empty_grounding_are_rejected() -> None:
    payload = _payload(
        [
            _step("driver", "CURRENT_NEW"),
            _step("target_bridge", "TARGET_CONTRACT"),
        ]
    )
    payload["selected_evidence_ids"] = []
    _, errors = _validate(
        payload,
        path_ids=["p1"],
        evidence_ids={"e1"},
        graph_evidence_ids={"e1"},
    )
    assert "selected_evidence_ids must cite current evidence" in errors
    assert "reasoning_steps must contain at least three material steps" in errors


def _boundary_payload(
    *,
    central: float,
    mapped_option: str,
    support: str,
    probabilities: tuple[float, float, float],
) -> dict[str, object]:
    options = [
        "below recent range",
        "within recent range",
        "above recent range",
    ]
    return {
        "target_operation_check": "Exact target operation checked.",
        "directional_signal": "uncertain",
        "magnitude_assessment": {
            "support": support,
            "evidence_ids": ["e1"],
            "rationale": "Magnitude support is explicitly assessed.",
        },
        "latent_target_estimate": {
            "low": central - 1.0,
            "central": central,
            "high": central + 1.0,
            "unit": "percent",
            "basis": "Cutoff-safe evidence and a neutral anchor.",
        },
        "boundary_checks": [
            {
                "option": option,
                "interval": "public interval",
                "compatibility": "plausible",
                "rationale": "Compared with the central estimate.",
            }
            for option in options
        ],
        "mapped_option": mapped_option,
        "prediction": mapped_option,
        "option_probabilities": [
            {"option": option, "probability": probability}
            for option, probability in zip(options, probabilities, strict=True)
        ],
        "uncertainty": "Uncertainty is high.",
    }


def _three_way_contract(lower: float, upper: float) -> dict[str, object]:
    return {
        "intervals": {
            "below recent range": f"(-infinity, {lower})",
            "within recent range": f"[{lower}, {upper})",
            "above recent range": f"[{upper}, +infinity)",
        }
    }


def _boundary_errors(
    payload: dict[str, object], contract: dict[str, object]
) -> list[str]:
    _, errors = _validate_boundary_forecast(
        payload,
        options=[
            "below recent range",
            "within recent range",
            "above recent range",
        ],
        contract=contract,
        evidence_ids={"e1"},
        reasoning_policy="boundary_only",
        validation_policy="strict",
        prospective_anchor_support="none",
    )
    return errors


def test_weak_magnitude_allows_arithmetic_outer_mapping_with_low_confidence() -> None:
    payload = _boundary_payload(
        central=0.0,
        mapped_option="below recent range",
        support="insufficient",
        probabilities=(0.45, 0.45, 0.10),
    )
    assert _boundary_errors(payload, _three_way_contract(0.313386, 2.041064)) == []


def test_zero_at_exclusive_upper_boundary_maps_above_with_low_confidence() -> None:
    payload = _boundary_payload(
        central=0.0,
        mapped_option="above recent range",
        support="insufficient",
        probabilities=(0.30, 0.30, 0.40),
    )
    assert _boundary_errors(payload, _three_way_contract(-0.115, 0.0)) == []


def test_weak_outer_mapping_rejects_overconfidence_not_the_option() -> None:
    payload = _boundary_payload(
        central=0.0,
        mapped_option="below recent range",
        support="insufficient",
        probabilities=(0.60, 0.30, 0.10),
    )
    errors = _boundary_errors(payload, _three_way_contract(0.313386, 2.041064))
    assert len(errors) == 1
    assert "cap every option probability at 0.45" in errors[0]


def test_weak_support_does_not_excuse_wrong_arithmetic_mapping() -> None:
    payload = _boundary_payload(
        central=0.0,
        mapped_option="within recent range",
        support="insufficient",
        probabilities=(0.40, 0.45, 0.15),
    )
    errors = _boundary_errors(payload, _three_way_contract(0.313386, 2.041064))
    assert any("expected 'below recent range'" in error for error in errors)


def test_repair_feedback_fixes_probabilities_not_arithmetic_mapping() -> None:
    payload = _boundary_payload(
        central=0.0,
        mapped_option="below recent range",
        support="insufficient",
        probabilities=(0.40, 0.45, 0.15),
    )
    errors = _boundary_errors(payload, _three_way_contract(0.313386, 2.041064))
    assert len(errors) == 1
    assert "fixes both mapped_option and prediction" in errors[0]
    assert "Change option_probabilities" in errors[0]
    assert "at or below 0.45" in errors[0]
