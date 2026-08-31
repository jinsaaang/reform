"""Focused event deduplication regression tests."""

from src.tools.reasoning.event_identifier import EventIdentifierTool


def test_opposite_threshold_outcomes_are_not_deduplicated():
    assert EventIdentifierTool._has_threshold_polarity_conflict(
        "Apple gross margin is below 49.0%",
        "Apple gross margin is at least 49.0%",
    )


def test_same_threshold_polarity_can_still_be_deduplicated():
    assert not EventIdentifierTool._has_threshold_polarity_conflict(
        "Apple gross margin exceeds 49.0%",
        "Apple gross margin is at least 49.0%",
    )
