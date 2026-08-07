"""Minimal question model and forecast-time helpers for the public bundle."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Question(BaseModel):
    """Self-contained subset of the WorldReasoner question schema.

    The benchmark JSONL rows retain every original field through ``extra=allow``.
    Only fields read by the HGF and baseline runners are declared here.
    """

    id: str
    question_text: str
    question_type: str
    domain: str
    source: str
    difficulty: int
    resolution_date: datetime
    estimated_start_time: datetime | None = None
    ground_truth: Any | None = None
    context: str | None = None
    resolution_criteria: str | None = None
    options: list[str] | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] | None = None

    model_config = ConfigDict(extra="allow")


class ForecastSlot(StrEnum):
    EARLY = "early"
    MID = "mid"
    LATE = "late"


SLOT_FRACTIONS = {
    ForecastSlot.EARLY: 0.20,
    ForecastSlot.MID: 0.50,
    ForecastSlot.LATE: 0.80,
}
MIN_EFFECTIVE_SLOT_WINDOW_DAYS = 7


def _aware(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


def calculate_forecast_context_window(
    question: Question,
) -> tuple[datetime, datetime]:
    """Return the same question-defined window used by WorldReasoner."""
    window_end = _aware(question.resolution_date) - timedelta(seconds=1)
    if question.estimated_start_time is not None:
        window_start = _aware(question.estimated_start_time)
    else:
        candidate = _aware(question.created_at)
        window_start = (
            candidate
            if candidate < window_end
            else window_end - timedelta(days=30)
        )
    if window_start >= window_end:
        raise ValueError(
            f"Invalid forecast window for {question.id}: "
            f"{window_start=} >= {window_end=}"
        )
    return window_start, window_end


def get_forecast_date_for_slot(
    question: Question,
    slot: ForecastSlot = ForecastSlot.MID,
    db: Any | None = None,
) -> dict[str, Any]:
    """Interpolate a named forecast slot within the question's time window."""
    del db
    window_start, window_end = calculate_forecast_context_window(question)
    minimum_start = window_end - timedelta(days=MIN_EFFECTIVE_SLOT_WINDOW_DAYS)
    effective_start = min(window_start, minimum_start)
    span = window_end - effective_start
    simulated_date = effective_start + span * SLOT_FRACTIONS[slot]
    return {
        "simulated_date": simulated_date,
        "window_start": effective_start,
        "window_end": window_end,
        "slot": slot.value,
        "horizon_days": span.days,
    }
