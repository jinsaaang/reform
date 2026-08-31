"""Tests for chronological question-family experiment selection."""

from datetime import datetime, timedelta, timezone

import pytest

from forecaster.data_pipeline.family import (
    FamilySpec,
    build_chronological_split,
    select_family,
)
from src.domain.models import Domain
from src.domain.models.question_helpers import ForecastSlot
from tests.conftest import create_test_question


SPEC = FamilySpec("macro", "inflation", "bucketed_numeric")


def family_question(index: int):
    resolution = datetime(2026, 1, 10, tzinfo=timezone.utc) + timedelta(
        days=index * 7
    )
    return create_test_question(
        id=f"q_{index:02d}",
        domain=Domain.FINANCE,
        estimated_start_time=resolution - timedelta(days=50),
        resolution_date=resolution,
        metadata={
            "finfactorbench": {
                "original_domain": "macro",
                "subdomain": "inflation",
                "original_question_type": "bucketed_numeric",
            }
        },
    )


def test_select_family_is_chronological_and_not_randomized():
    questions = [family_question(index) for index in (3, 1, 2)]

    selected = select_family(questions, SPEC)

    assert [question.id for question in selected] == ["q_01", "q_02", "q_03"]


def test_build_chronological_split_uses_temporal_embargo_and_70_30_order():
    questions = [family_question(index) for index in range(11)]

    manifest = build_chronological_split(
        questions,
        SPEC,
        memory_count=7,
        test_count=3,
        forecast_slot=ForecastSlot.LATE,
    )

    assert manifest["memory_question_ids"] == [f"q_{index:02d}" for index in range(7)]
    assert [row["question_id"] for row in manifest["test"]] == [
        "q_08",
        "q_09",
        "q_10",
    ]
    assert manifest["cohort_question_ids"] == [
        *[f"q_{index:02d}" for index in range(7)],
        "q_08",
        "q_09",
        "q_10",
    ]
    assert manifest["embargo_question_ids"] == ["q_07"]
    assert manifest["selection_policy"]["randomized"] is False
    assert all(
        row["eligible_memory_question_ids"] == manifest["memory_question_ids"]
        for row in manifest["test"]
    )


def test_build_chronological_split_rejects_when_no_leakage_safe_test_exists():
    questions = [family_question(index) for index in range(10)]
    questions[-1].estimated_start_time = questions[6].resolution_date

    with pytest.raises(ValueError, match="Not enough leakage-safe"):
        build_chronological_split(
            questions,
            SPEC,
            memory_count=7,
            test_count=3,
            min_evidence_window_days=0,
            forecast_slot=ForecastSlot.EARLY,
        )


def test_build_chronological_split_excludes_short_evidence_windows():
    questions = [family_question(index) for index in range(12)]
    questions[0].estimated_start_time = questions[0].resolution_date - timedelta(days=5)

    manifest = build_chronological_split(questions, SPEC)

    assert "q_00" not in manifest["cohort_question_ids"]
    assert manifest["cohort_question_ids"][-1] == "q_11"
