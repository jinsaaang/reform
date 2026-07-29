from __future__ import annotations

from hgf.runner import (
    _dag_semantic_lesson_validator,
    _inject_target_operator_step,
    _reasoning_schema,
    _reasoning_validator,
    compile_current_target_operator,
    compile_dag_expert_memory,
)


def _blueprint(*, usable: bool = True) -> dict:
    return {
        "graph_diagnosis": {"usable": usable, "summary": "ok"},
        "target_definition": {"metric": "monthly return"},
        "checkpoints": [
            {
                "id": "baseline",
                "role": "baseline",
                "factor": "recent target state",
                "mechanism": "anchors the forecast",
                "expected_direction": "mixed",
                "evidence_requirement": "recent observation",
                "contradiction_signal": "structural break",
                "historical_support": "high",
            },
            {
                "id": "driver",
                "role": "driver",
                "factor": "demand",
                "mechanism": "demand changes target",
                "expected_direction": "up",
                "evidence_requirement": "current demand evidence",
                "contradiction_signal": "demand decline",
                "historical_support": "high",
            },
            {
                "id": "target_bridge",
                "role": "target_bridge",
                "factor": "monthly return",
                "mechanism": "maps balance to return",
                "expected_direction": "mixed",
                "evidence_requirement": "target bridge",
                "contradiction_signal": "price divergence",
                "historical_support": "high",
            },
        ],
        "causal_paths": [
            {
                "checkpoint_ids": ["baseline", "driver", "target_bridge"],
                "generalized_mechanism": "baseline updated by demand",
                "expected_direction": "up",
                "applicability_conditions": ["demand is current"],
                "failure_conditions": ["supply dominates"],
            }
        ],
        "alternative_hypotheses": [
            {
                "hypothesis": "supply dominates",
                "discriminating_evidence": "inventory build",
            }
        ],
        "forecast_audit_questions": ["Does the bridge reach the target?"],
    }


def _worked_exemplar() -> dict:
    return {
        "task_signature": {
            "category": "energy",
            "target_operation": "return",
            "option_geometry": "range",
        },
        "target_semantics": "historical monthly return",
        "expert_reasoning": [
            "Establish a baseline.",
            "Assess demand.",
            "Bridge to the target.",
        ],
        "counterevidence": "Supply may dominate.",
        "uncertainty": "Magnitude is uncertain.",
        "dag_derived_lesson": "Reconcile demand and supply.",
    }


def test_compiler_uses_only_usable_dag() -> None:
    payload = compile_dag_expert_memory(
        source_question_id="q1",
        blueprint=_blueprint(),
        worked_exemplar=_worked_exemplar(),
    )
    assert payload["source_question_id"] == "q1"
    assert len(payload["causal_checkpoint_library"]) == 3
    assert payload["mechanism_library"][0]["checkpoint_ids"] == [
        "baseline",
        "driver",
        "target_bridge",
    ]


def test_compiler_rejects_unusable_dag() -> None:
    try:
        compile_dag_expert_memory(
            source_question_id="q1",
            blueprint=_blueprint(usable=False),
            worked_exemplar=_worked_exemplar(),
        )
    except ValueError as exc:
        assert "not causally reusable" in str(exc)
    else:
        raise AssertionError("unusable DAG must be rejected")


def test_schema_requires_checkpoint_mapping() -> None:
    schema = _reasoning_schema(
        ["below", "within", "above"],
        ["baseline", "driver", "target_bridge"],
    )
    step = schema["schema"]["properties"]["reasoning_steps"]["items"]
    assert "source_checkpoint_id" in step["required"]


def test_validator_requires_one_dag_checkpoint() -> None:
    validator = _reasoning_validator(
        options=["no", "yes"],
        evidence_ids={"e1"},
        checkpoint_ids={"baseline", "driver", "target_bridge"},
        target_bridge_checkpoint_ids={"target_bridge"},
    )
    payload = {
        "target_semantics": "target",
        "selected_evidence_ids": ["e1"],
        "evidence_fit": {
            "metric_match": "direct",
            "horizon_match": "direct",
            "magnitude_support": "partial",
            "assessment": "current evidence partially supports magnitude",
        },
        "reasoning_steps": [
            {
                "step_type": "baseline",
                "statement": "baseline",
                "evidence_ids": ["e1"],
                "effect_on_target": "neutral",
                "source_checkpoint_id": "CURRENT_NEW",
            },
            {
                "step_type": "driver",
                "statement": "driver",
                "evidence_ids": ["e1"],
                "effect_on_target": "up",
                "source_checkpoint_id": "CURRENT_NEW",
            },
            {
                "step_type": "target_bridge",
                "statement": "bridge",
                "evidence_ids": ["e1"],
                "effect_on_target": "up",
                "source_checkpoint_id": "TARGET_CONTRACT",
            },
        ],
        "counterevidence": "counter",
        "target_estimate": "estimate",
        "option_mapping": "yes",
        "prediction": "yes",
        "option_probabilities": [
            {"option": "no", "probability": 0.35},
            {"option": "yes", "probability": 0.65},
        ],
        "uncertainty": "uncertain",
    }
    _, errors = validator(payload)
    assert any("at least one" in error for error in errors)


def test_acceleration_operator_is_explicitly_injected() -> None:
    operator = compile_current_target_operator(
        {
            "target_metric": "quarterly revenue growth acceleration",
            "target_period": "2025-Q1",
            "change_unit": "percentage points",
            "comparison_rule": "target YoY growth >= prior-quarter YoY growth",
            "resolution_rule": "compare sequential YoY growth rates",
        }
    )
    assert "does not imply" in operator["semantic_guard"]
    reasoning = {
        "reasoning_steps": [
            {
                "step_type": "baseline",
                "statement": "current baseline",
                "evidence_ids": [],
                "effect_on_target": "neutral",
                "source_checkpoint_id": "TARGET_CONTRACT",
            }
        ]
    }
    _inject_target_operator_step(reasoning, operator)
    statement = reasoning["reasoning_steps"][0]["statement"]
    assert "growth acceleration" in statement
    assert "prior-quarter YoY growth" in statement


def test_dag_semantic_lessons_require_all_disciplines() -> None:
    _, errors = _dag_semantic_lesson_validator(
        {
            "target_semantics_lesson": "Keep the exact target operation.",
            "evidence_selection_lesson": "Use checkpoint requirements.",
            "causal_reasoning_lesson": "Follow the supported mechanism.",
            "counterevidence_lesson": "Test the competing path.",
            "calibration_lesson": "",
        }
    )
    assert errors == ["calibration_lesson is empty"]
