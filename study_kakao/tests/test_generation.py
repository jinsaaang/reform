from __future__ import annotations

from types import SimpleNamespace

from hgf.exemplar import (
    _call_json as exemplar_call_json,
    _ensure_baseline_reasoning_step,
    _normalize_probability_rows,
)
from hgf.forecast_core import _call_json as forecast_call_json
from hgf.forecast_core import _seed
from hgf.generation import configure_generation


class _Usage:
    def model_dump(self):
        return {
            "prompt_tokens": 3,
            "completion_tokens": 2,
            "total_tokens": 5,
        }


class _Completions:
    def __init__(self) -> None:
        self.requests = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="{}"),
                )
            ],
            usage=_Usage(),
        )


def _client() -> SimpleNamespace:
    return SimpleNamespace(
        chat=SimpleNamespace(completions=_Completions()),
    )


def test_generation_controls_reach_both_json_callers() -> None:
    configure_generation(
        reasoning_effort="medium",
        max_output_tokens=8000,
        run_seed=2,
    )
    try:
        for caller, model in (
            (forecast_call_json, "google/gemini-2.5-flash"),
            (exemplar_call_json, "openai/gpt-5-mini"),
        ):
            client = _client()
            caller(
                client,
                model=model,
                system="system",
                prompt="prompt",
                schema={"name": "test", "schema": {"type": "object"}},
                seed=1,
                max_tokens=123,
            )
            request = client.chat.completions.requests[0]
            assert request["max_tokens"] == 8000
            assert request["extra_body"] == {
                "reasoning": {"effort": "medium"}
            }
            assert ("temperature" in request) is not model.startswith(
                "openai/gpt-5"
            )
    finally:
        configure_generation()


def test_run_seed_namespaces_repeated_experiments() -> None:
    configure_generation(run_seed=0)
    canonical = _seed("question", "stage")
    configure_generation(run_seed=1)
    repeated = _seed("question", "stage")
    configure_generation()

    assert canonical != repeated
    assert canonical == _seed("question", "stage")


def test_provider_serialization_is_normalized_without_changing_ranking() -> None:
    payload = {
        "prediction": "b",
        "option_probabilities": [
            {"option": "a", "probability": "20%"},
            {"option": "b", "probability": "50%"},
            {"option": "c", "probability": "30%"},
        ],
    }
    _normalize_probability_rows(payload, ["a", "b", "c"])

    assert payload["prediction"] == "b"
    assert payload["option_probabilities"] == [
        {"option": "a", "probability": 0.2},
        {"option": "b", "probability": 0.5},
        {"option": "c", "probability": 0.3},
    ]


def test_missing_baseline_step_is_restored_structurally() -> None:
    payload = {
        "target_semantics": "monthly change in the exact target metric",
        "reasoning_steps": [
            {
                "step_type": "driver",
                "statement": "supported driver",
                "evidence_ids": ["e1"],
                "effect_on_target": "up",
            },
            {
                "step_type": "target_bridge",
                "statement": "bridge",
                "evidence_ids": ["e1"],
                "effect_on_target": "up",
            },
        ],
    }
    _ensure_baseline_reasoning_step(
        payload,
        source_checkpoint_id="TARGET_CONTRACT",
    )

    baseline = payload["reasoning_steps"][0]
    assert baseline["step_type"] == "baseline"
    assert baseline["source_checkpoint_id"] == "TARGET_CONTRACT"
    assert baseline["evidence_ids"] == ["e1"]


def test_string_reasoning_steps_preserve_text_in_structured_form() -> None:
    payload = {
        "target_semantics": "monthly change in the exact target metric",
        "reasoning_steps": ["A provider-serialized driver statement."],
    }
    _ensure_baseline_reasoning_step(payload)

    baseline, driver = payload["reasoning_steps"]
    assert baseline["step_type"] == "baseline"
    assert driver == {
        "step_type": "driver",
        "statement": "A provider-serialized driver statement.",
        "evidence_ids": [],
        "effect_on_target": "uncertain",
    }


def test_string_probability_rows_are_removed_before_validation() -> None:
    payload = {
        "prediction": "a",
        "option_probabilities": [
            "a: 50%",
            {"option": "b", "probability": 0.3},
            {"option": "c", "probability": 0.2},
        ],
    }
    _normalize_probability_rows(payload, ["a", "b", "c"])

    assert payload["option_probabilities"] == [
        {"option": "b", "probability": 0.3},
        {"option": "c", "probability": 0.2},
    ]
