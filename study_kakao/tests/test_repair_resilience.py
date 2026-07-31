from __future__ import annotations

from importlib import import_module

import pytest


def _api():
    return import_module("hgf.repair_resilience")


def test_reasoning_schema_contract_enforces_non_empty_fields_and_exact_probability_row_count() -> None:
    api = _api()

    schema = api.forecast_reasoning_schema(
        options=["below", "within", "above"],
    )
    properties = schema["schema"]["properties"]
    step = properties["reasoning_steps"]["items"]["properties"]

    assert properties["target_semantics"]["minLength"] == 1
    assert properties["counterevidence"]["minLength"] == 1
    assert properties["target_estimate"]["minLength"] == 1
    assert properties["option_mapping"]["minLength"] == 1
    assert properties["uncertainty"]["minLength"] == 1
    assert step["statement"]["minLength"] == 1
    assert properties["option_probabilities"]["minItems"] == 3
    assert properties["option_probabilities"]["maxItems"] == 3


def test_conservative_repair_merge_preserves_prior_substantive_fields_when_repair_returns_empty_values() -> None:
    api = _api()

    original = {
        "target_semantics": "exact monthly return for the target period",
        "counterevidence": "inventory data weakens the bullish path",
        "target_estimate": "-1% to +1%",
        "option_mapping": "within recent range",
        "uncertainty": "magnitude support is partial",
        "reasoning_steps": [
            {
                "step_type": "baseline",
                "statement": "Current baseline remains near neutral.",
                "evidence_ids": ["e1"],
                "effect_on_target": "neutral",
            },
            {
                "step_type": "target_bridge",
                "statement": "The direct bridge keeps the estimate near range.",
                "evidence_ids": ["e1"],
                "effect_on_target": "mixed",
            },
        ],
        "prediction": "within recent range",
        "option_probabilities": [
            {"option": "below recent range", "probability": 0.2},
            {"option": "within recent range", "probability": 0.6},
            {"option": "above recent range", "probability": 0.2},
        ],
    }
    repaired = {
        "target_semantics": "",
        "counterevidence": "",
        "target_estimate": "",
        "option_mapping": "",
        "uncertainty": "",
        "reasoning_steps": [
            {
                "step_type": "baseline",
                "statement": "",
                "evidence_ids": [],
                "effect_on_target": "neutral",
            }
        ],
        "prediction": "within recent range",
        "option_probabilities": [],
    }

    merged = api.conservative_repair_merge(
        original=original,
        repaired=repaired,
    )

    assert merged["target_semantics"] == original["target_semantics"]
    assert merged["counterevidence"] == original["counterevidence"]
    assert merged["target_estimate"] == original["target_estimate"]
    assert merged["option_mapping"] == original["option_mapping"]
    assert merged["uncertainty"] == original["uncertainty"]
    assert merged["reasoning_steps"][0]["statement"] == original["reasoning_steps"][0]["statement"]


def test_neutral_probability_serialization_fills_missing_and_duplicate_rows_deterministically() -> None:
    api = _api()

    rows = [
        {"option": "yes", "probability": 0.9},
        {"option": "yes", "probability": 0.1},
    ]

    serialized = api.serialize_neutral_probabilities(
        rows=rows,
        options=["no", "yes", "maybe"],
    )

    assert serialized == [
        {"option": "no", "probability": pytest.approx(1 / 3)},
        {"option": "yes", "probability": pytest.approx(1 / 3)},
        {"option": "maybe", "probability": pytest.approx(1 / 3)},
    ]


def test_neutral_probability_serialization_sums_to_one_for_empty_rows() -> None:
    api = _api()

    serialized = api.serialize_neutral_probabilities(
        rows=[],
        options=["a", "b", "c", "d"],
    )

    assert [row["option"] for row in serialized] == ["a", "b", "c", "d"]
    assert sum(row["probability"] for row in serialized) == pytest.approx(1.0)
    assert all(row["probability"] == pytest.approx(0.25) for row in serialized)


def test_neutral_reasoning_fallback_is_explicit_and_checkpoint_safe() -> None:
    api = _api()

    payload = api.neutral_reasoning_payload(
        options=["below", "within", "above"],
        target_semantics="monthly change for the target period",
        include_checkpoint_mapping=True,
    )

    assert payload["generation_fallback"].startswith("neutral_reasoning")
    assert payload["evidence_fit"]["magnitude_support"] == "unsupported"
    assert payload["selected_evidence_ids"] == []
    assert {
        step["source_checkpoint_id"]
        for step in payload["reasoning_steps"]
    } <= {"CURRENT_NEW", "TARGET_CONTRACT"}
    assert sum(
        row["probability"] for row in payload["option_probabilities"]
    ) == pytest.approx(1.0)
