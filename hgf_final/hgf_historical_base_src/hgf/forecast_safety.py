"""Deterministic safety gates for transferring hindsight memory.

The helpers in this module deliberately avoid model calls and resolved test
outcomes. They decide whether a retrieved memory is structurally compatible,
how strongly a checkpoint must be enforced, and how raw forecasts are scored.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence


@dataclass(frozen=True)
class ForecastTarget:
    family_id: str
    target_metric: str


@dataclass(frozen=True)
class MemoryMetadata:
    family_id: str
    target_metric: str


_WEAK_SUPPORT = {
    "",
    "weak",
    "unsupported",
    "insufficient",
    "direction_only",
}


def _normalized_label(value: str) -> str:
    return " ".join(str(value).strip().casefold().split())


def is_memory_compatible(
    target: ForecastTarget,
    memory: MemoryMetadata,
) -> bool:
    """Require the same recurring family and exact target operation."""
    return (
        _normalized_label(target.family_id)
        == _normalized_label(memory.family_id)
        and _normalized_label(target.target_metric)
        == _normalized_label(memory.target_metric)
    )


def checkpoint_requirement(
    *,
    memory_accepted: bool,
    memory_compatible: bool,
    magnitude_support: str,
) -> str:
    """Return whether a historical checkpoint is mandatory in the trace."""
    support = _normalized_label(magnitude_support)
    if (
        memory_accepted
        and memory_compatible
        and support not in _WEAK_SUPPORT
    ):
        return "required"
    return "optional"


def score_forecast(
    *,
    probabilities: Mapping[str, float],
    explicit_prediction: str | None,
    ground_truth: str,
    options: Sequence[str],
) -> tuple[float, float]:
    """Score a forecast while respecting an explicit prediction on exact ties."""
    if not options:
        raise ValueError("options must not be empty")
    max_probability = max(float(probabilities[option]) for option in options)
    tied = [
        option
        for option in options
        if abs(float(probabilities[option]) - max_probability) <= 1e-12
    ]
    predicted = (
        str(explicit_prediction)
        if explicit_prediction in tied
        else tied[0]
    )
    accuracy = float(predicted == ground_truth)
    brier = sum(
        (
            float(probabilities[option])
            - (1.0 if option == ground_truth else 0.0)
        )
        ** 2
        for option in options
    ) / len(options)
    return accuracy, brier
