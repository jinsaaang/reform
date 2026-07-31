from __future__ import annotations

from importlib import import_module

def _api():
    return import_module("hgf.forecast_safety")


def test_memory_compatibility_requires_matching_family_and_exact_target_metric() -> None:
    api = _api()

    target = api.ForecastTarget(
        family_id="energy",
        target_metric="monthly return",
    )
    same_memory = api.MemoryMetadata(
        family_id="energy",
        target_metric="monthly return",
    )
    other_family = api.MemoryMetadata(
        family_id="equity",
        target_metric="monthly return",
    )
    other_metric = api.MemoryMetadata(
        family_id="energy",
        target_metric="quarterly revenue growth acceleration",
    )

    assert api.is_memory_compatible(target, same_memory) is True
    assert api.is_memory_compatible(target, other_family) is False
    assert api.is_memory_compatible(target, other_metric) is False


def test_checkpoint_requirement_stays_required_for_accepted_compatible_memory() -> None:
    api = _api()

    decision = api.checkpoint_requirement(
        memory_accepted=True,
        memory_compatible=True,
        magnitude_support="supported",
    )

    assert decision == "required"


def test_checkpoint_requirement_is_optional_for_incompatible_memory() -> None:
    api = _api()

    decision = api.checkpoint_requirement(
        memory_accepted=True,
        memory_compatible=False,
        magnitude_support="supported",
    )

    assert decision == "optional"


def test_checkpoint_requirement_is_optional_when_magnitude_support_is_unsupported() -> None:
    api = _api()

    decision = api.checkpoint_requirement(
        memory_accepted=True,
        memory_compatible=True,
        magnitude_support="unsupported",
    )

    assert decision == "optional"


def test_explicit_prediction_breaks_exact_probability_tie_for_accuracy_scoring() -> None:
    api = _api()

    probabilities = {
        "no": 0.50,
        "yes": 0.50,
    }

    accuracy, _ = api.score_forecast(
        probabilities=probabilities,
        explicit_prediction="yes",
        ground_truth="yes",
        options=["no", "yes"],
    )

    assert accuracy == 1.0
