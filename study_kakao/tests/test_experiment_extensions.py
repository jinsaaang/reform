from __future__ import annotations

import copy
import math

from hgf.experiment_ablation import (
    _condition_schema,
    transform_expert_memory,
)
from hgf.experiment_common import TOPK_VALUES, validate_condition_matrix
from hgf.experiment_judge import (
    _blind_forecast,
    _judgment_validator,
    _summary,
)
from hgf.experiment_stats import (
    mean_std,
    question_bootstrap_ci,
    spearman,
)
from hgf.experiment_topk import namespace_expert_memory


def _memory() -> dict:
    return {
        "expert_reasoning_demonstration": {
            "counterevidence": "counter",
            "uncertainty": "uncertain",
        },
        "causal_checkpoint_library": [
            {
                "checkpoint_id": "C1",
                "causal_role": "driver",
            },
            {
                "checkpoint_id": "C2",
                "causal_role": "target_bridge",
            },
        ],
        "mechanism_library": [
            {
                "checkpoint_ids": ["C1", "C2"],
                "fails_when": ["failure"],
            }
        ],
        "alternative_explanations": [{"hypothesis": "alternative"}],
        "dag_derived_semantic_lessons": {
            "counterevidence_lesson": "counter lesson",
            "calibration_lesson": "calibration lesson",
        },
    }


def test_ablation_transforms_are_copy_only_and_component_specific() -> None:
    original = _memory()
    frozen = copy.deepcopy(original)

    no_counter = transform_expert_memory(
        original,
        "without_counterevidence",
    )
    assert original == frozen
    assert "counterevidence" not in no_counter["expert_reasoning_demonstration"]
    assert "fails_when" not in no_counter["mechanism_library"][0]
    assert "alternative_explanations" not in no_counter
    assert (
        "counterevidence_lesson"
        not in no_counter["dag_derived_semantic_lessons"]
    )
    assert no_counter["causal_checkpoint_library"] == frozen[
        "causal_checkpoint_library"
    ]

    no_bridge = transform_expert_memory(original, "without_target_bridge")
    assert [item["checkpoint_id"] for item in no_bridge[
        "causal_checkpoint_library"
    ]] == ["C1"]
    assert no_bridge["mechanism_library"][0]["checkpoint_ids"] == ["C1"]

    no_uncertainty = transform_expert_memory(
        original,
        "without_uncertainty",
    )
    assert "uncertainty" not in no_uncertainty[
        "expert_reasoning_demonstration"
    ]
    assert "calibration_lesson" not in no_uncertainty[
        "dag_derived_semantic_lessons"
    ]


def test_ablation_schemas_cannot_reintroduce_removed_fields() -> None:
    options = ["low", "middle", "high"]
    no_counter = _condition_schema(
        options,
        ["C1"],
        "without_counterevidence",
    )
    assert "counterevidence" not in no_counter["schema"]["properties"]
    step_enum = no_counter["schema"]["properties"]["reasoning_steps"][
        "items"
    ]["properties"]["step_type"]["enum"]
    assert "counterevidence" not in step_enum

    no_bridge = _condition_schema(
        options,
        ["C1"],
        "without_target_bridge",
    )
    assert "target_estimate" not in no_bridge["schema"]["properties"]
    assert "option_mapping" not in no_bridge["schema"]["properties"]
    bridge_enum = no_bridge["schema"]["properties"]["reasoning_steps"][
        "items"
    ]["properties"]["step_type"]["enum"]
    assert "target_bridge" not in bridge_enum

    no_uncertainty = _condition_schema(
        options,
        ["C1"],
        "without_uncertainty",
    )
    assert "uncertainty" not in no_uncertainty["schema"]["properties"]


def test_topk_namespaces_checkpoint_ids_without_mutating_source() -> None:
    original = _memory()
    frozen = copy.deepcopy(original)
    namespaced, ids = namespace_expert_memory(
        original,
        rank=2,
        memory_id="memory-x",
    )
    assert original == frozen
    assert ids == ["M2:memory-x:C1", "M2:memory-x:C2"]
    assert namespaced["mechanism_library"][0]["checkpoint_ids"] == ids
    assert TOPK_VALUES == (1, 3, 5, 7)


def test_blind_forecast_excludes_identity_and_ground_truth() -> None:
    row = {
        "method": "hgf",
        "model": "forecaster",
        "ground_truth": "yes",
        "reasoning": {"steps": []},
        "probabilities": {"yes": 0.7, "no": 0.3},
    }
    assert _blind_forecast(row) == {
        "selected_outcome": "yes",
        "reasoning": {"steps": []},
    }


def test_judge_validator_enforces_paper_reasoning_contract() -> None:
    judgment = {
        "evidence_items": [
            {
                "requirement": "Revenue increased",
                "cited_evidence_ids": ["e1"],
                "supported_at_forecast_time": True,
                "rationale": "The cited filing states the increase.",
            }
        ],
        "invalid_reasoning": False,
        "invalid_reasons": {
            "unsupported_decisive_claim": False,
            "forecast_time_violation": False,
            "selected_outcome_not_justified": False,
        },
        "invalid_reasoning_rationale": "All decisive claims are supported.",
    }
    payload = {
        label: copy.deepcopy(judgment)
        for label in ("forecast_a", "forecast_b")
    }
    assert _judgment_validator(
        payload,
        allowed_evidence_ids={"e1"},
    ) == ({}, [])
    payload["forecast_a"]["invalid_reasoning"] = True
    assert _judgment_validator(payload)[1]


def test_judge_summary_reports_paper_rates() -> None:
    rows = [
        {
            "parse_retries": 0,
            "judgments": {
                "raw_dag": {
                    "required_evidence_items": 2,
                    "supported_evidence_items": 1,
                    "invalid_reasoning": True,
                    "correct_after_unblinding": True,
                },
                "full_hgf": {
                    "required_evidence_items": 2,
                    "supported_evidence_items": 2,
                    "invalid_reasoning": False,
                    "correct_after_unblinding": True,
                },
            },
        }
    ]
    summary = _summary(rows)
    assert summary["conditions"]["raw_dag"]["evidence_coverage"]["rate"] == 0.5
    assert summary["conditions"]["raw_dag"]["invalid_reasoning"]["rate"] == 1.0
    assert (
        summary["conditions"]["raw_dag"]["invalid_among_correct"]["rate"]
        == 1.0
    )
    assert (
        summary["conditions"]["full_hgf"]["invalid_among_correct"]["rate"]
        == 0.0
    )


def test_statistics_handle_ties_and_grouped_question_bootstrap() -> None:
    assert math.isclose(spearman([1, 2, 2, 4], [1, 3, 3, 5]), 1.0)
    assert mean_std([1, 2, 3]) == {
        "n": 3,
        "mean": 2.0,
        "std": 1.0,
    }
    interval = question_bootstrap_ci(
        {"q1": [1.0, 1.0], "q2": [0.0, 0.0]},
        iterations=200,
        seed=7,
    )
    assert interval["n_questions"] == 2
    assert interval["estimate"] == 0.5
    assert 0 <= interval["lower_95"] <= interval["upper_95"] <= 1


def test_exact_condition_matrix_validation() -> None:
    payload = {
        "results": [
            {
                "status": "success",
                "question_id": question_id,
                "method": method,
            }
            for question_id in ("q1", "q2")
            for method in ("a", "b")
        ]
    }
    assert validate_condition_matrix(
        payload,
        question_ids=("q1", "q2"),
        conditions=("a", "b"),
    ) == []
    payload["results"].pop()
    assert validate_condition_matrix(
        payload,
        question_ids=("q1", "q2"),
        conditions=("a", "b"),
    )
