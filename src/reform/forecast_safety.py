"""Deterministic scoring of a raw forecast.

Scoring never calls a model and never reads a resolved outcome before the
forecast has been produced.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence


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
