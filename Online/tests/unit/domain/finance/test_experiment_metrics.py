"""Behavioral contracts for ex-ante finance experiment metrics."""

from decimal import Decimal

import pytest
from pydantic import ValidationError

from src.domain.finance.experiment_metrics import (
    ArgmaxState,
    DeltaDirection,
    ExAnteTieReason,
    ForecastPair,
    JudgePanelMetrics,
    MajorityFlip,
    PanelPreference,
    compare_positive_probabilities,
)


def test_delta_is_second_minus_first_with_explicit_majority_flip() -> None:
    # Given
    first = Decimal("0.6")
    second = Decimal("0.4")

    # When
    comparison = compare_positive_probabilities(ForecastPair.A_B, first, second)

    # Then
    assert comparison.delta == Decimal("-0.2")
    assert comparison.direction is DeltaDirection.FIRST_HIGHER
    assert comparison.majority_flip is MajorityFlip.POSITIVE_TO_NEGATIVE


@pytest.mark.parametrize(
    ("first", "second", "first_state", "second_state", "flip"),
    (
        (
            "0.4",
            "0.6",
            ArgmaxState.NEGATIVE,
            ArgmaxState.POSITIVE,
            MajorityFlip.NEGATIVE_TO_POSITIVE,
        ),
        (
            "0.5",
            "0.6",
            ArgmaxState.TIE,
            ArgmaxState.POSITIVE,
            MajorityFlip.TIE_TO_POSITIVE,
        ),
        (
            "0.5",
            "0.4",
            ArgmaxState.TIE,
            ArgmaxState.NEGATIVE,
            MajorityFlip.TIE_TO_NEGATIVE,
        ),
        (
            "0.6",
            "0.5",
            ArgmaxState.POSITIVE,
            ArgmaxState.TIE,
            MajorityFlip.POSITIVE_TO_TIE,
        ),
        (
            "0.4",
            "0.5",
            ArgmaxState.NEGATIVE,
            ArgmaxState.TIE,
            MajorityFlip.NEGATIVE_TO_TIE,
        ),
    ),
)
def test_argmax_and_every_majority_transition_are_explicit(
    first: str,
    second: str,
    first_state: ArgmaxState,
    second_state: ArgmaxState,
    flip: MajorityFlip,
) -> None:
    # Given
    first_probability = Decimal(first)
    second_probability = Decimal(second)

    # When
    comparison = compare_positive_probabilities(
        ForecastPair.A_C,
        first_probability,
        second_probability,
    )

    # Then
    assert comparison.first_argmax is first_state
    assert comparison.second_argmax is second_state
    assert comparison.majority_flip is flip


def test_tolerance_marks_probability_and_majority_as_ties() -> None:
    # Given
    first = Decimal("0.5000000004")
    second = Decimal("0.4999999996")

    # When
    comparison = compare_positive_probabilities(ForecastPair.B_C, first, second)

    # Then
    assert comparison.direction is DeltaDirection.UNCHANGED
    assert comparison.first_argmax is ArgmaxState.TIE
    assert comparison.second_argmax is ArgmaxState.TIE
    assert comparison.majority_flip is MajorityFlip.NONE


def test_probability_comparison_rejects_values_outside_unit_interval() -> None:
    # Given
    invalid = Decimal("1.000000001")

    # When
    with pytest.raises(ValidationError):
        compare_positive_probabilities(ForecastPair.A_B, invalid, Decimal("0.5"))

    # Then
    assert invalid > 1


def test_panel_formulas_exclude_inconsistent_member_from_evaluable_count() -> None:
    # Given
    one_half = Decimal(1) / Decimal(2)
    two_thirds = Decimal(2) / Decimal(3)

    # When
    panel = JudgePanelMetrics(
        overall_preference=PanelPreference.TIE,
        preference_eligible=True,
        tie_reason=ExAnteTieReason.NO_CONSENSUS,
        first_votes=1,
        second_votes=0,
        tie_votes=1,
        inconsistent_count=1,
        invalid_count=0,
        evaluable_count=2,
        two_parse_valid_count=3,
        attempted_call_count=6,
        invalid_rate=Decimal(0),
        inconsistent_rate=Decimal(1) / Decimal(3),
        agreement=one_half,
        order_consistency=two_thirds,
    )

    # Then
    assert panel.evaluable_count == 2
    assert panel.two_parse_valid_count == 3
    assert panel.agreement == one_half
    assert panel.order_consistency == two_thirds


def test_panel_rejects_forged_agreement_and_order_consistency() -> None:
    # Given
    forged = Decimal("0.9")

    # When
    with pytest.raises(ValidationError):
        JudgePanelMetrics(
            overall_preference=PanelPreference.SECOND,
            preference_eligible=True,
            tie_reason=None,
            first_votes=1,
            second_votes=2,
            tie_votes=0,
            inconsistent_count=0,
            invalid_count=0,
            evaluable_count=3,
            two_parse_valid_count=3,
            attempted_call_count=6,
            invalid_rate=Decimal(0),
            inconsistent_rate=Decimal(0),
            agreement=forged,
            order_consistency=forged,
        )

    # Then
    assert forged != Decimal(2) / Decimal(3)
