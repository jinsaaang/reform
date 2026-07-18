"""Behavioral contracts for deterministic ex-ante Markdown rendering."""

from decimal import Decimal

import pytest

from src.services.finance_experiment_reporting import (
    build_ex_ante_pair_summaries,
    format_percentage,
    format_percentage_points,
    render_ex_ante_report,
)
from tests.fixtures.finance_experiment_artifact import make_persisted_suite


def test_percentage_rendering_uses_decimal_half_even() -> None:
    # Given
    probability = Decimal("0.12345")

    # When
    rendered = format_percentage(probability)

    # Then
    assert rendered == "12.34%"


@pytest.mark.parametrize(
    ("value", "expected"),
    (
        ("-0.000049", "-0.00%p"),
        ("-0.00005", "-0.00%p"),
        ("-0.000051", "-0.01%p"),
        ("0.000049", "+0.00%p"),
        ("0.00005", "+0.00%p"),
        ("0.000051", "+0.01%p"),
    ),
)
def test_percentage_point_rendering_preserves_signed_zero(
    value: str,
    expected: str,
) -> None:
    assert format_percentage_points(Decimal(value)) == expected


def test_unsigned_percentage_rendering_still_normalizes_negative_zero() -> None:
    assert format_percentage(Decimal("-0.00005")) == "0.00%"


def test_report_contains_exact_deltas_and_explicit_denominators() -> None:
    # Given
    suite = make_persisted_suite()

    # When
    report = render_ex_ante_report(suite)

    # Then
    assert "-20.00%p" in report
    assert "-50.00%p" in report
    assert "completed panels: 2/2" in report
    assert "invalid members: 0/6" in report
    assert "no-quorum panels: 0/2" in report
    assert "agreement observations: 2/2" in report
    assert "order-consistency observations: 2/2" in report


def test_ex_ante_report_has_no_truth_or_resolution_score_surface() -> None:
    # Given
    suite = make_persisted_suite()

    # When
    rendered = render_ex_ante_report(suite).casefold()

    # Then
    assert "brier" not in rendered
    assert "realized" not in rendered
    assert "resolved" not in rendered
    assert "ground truth" not in rendered
    assert "outcome" not in rendered


def test_pair_summaries_preserve_judge_counts_and_rates() -> None:
    # Given
    suite = make_persisted_suite()

    # When
    summaries = build_ex_ante_pair_summaries(suite)

    # Then
    assert len(summaries) == 3
    assert summaries[0].completed_panel_count == 2
    assert summaries[0].judge_member_count == 6
    assert summaries[0].invalid_member_count == 0
    assert summaries[0].agreement_mean == Decimal("0.6666666666666666666666666667")
    assert summaries[0].order_consistency_mean == Decimal("1")
