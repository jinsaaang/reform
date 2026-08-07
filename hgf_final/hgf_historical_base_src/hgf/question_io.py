"""Deterministic family selection and chronological splitting for finance runs."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from hgf.models import Question
from hgf.models import ForecastSlot, get_forecast_date_for_slot


_AS_OF_DATE_RE = re.compile(r"\bAs\s+of\s+(20\d{2}-\d{2}-\d{2})\b", re.IGNORECASE)


def read_questions(path: Path) -> list[Question]:
    with path.open(encoding="utf-8") as handle:
        return [Question.model_validate_json(line) for line in handle if line.strip()]


def family_metadata(question: Question) -> dict[str, Any]:
    metadata = question.metadata or {}
    if not isinstance(metadata, dict):
        return {}
    for namespace in ("finance", "finfactorbench", "benchmark"):
        candidate = metadata.get(namespace)
        if isinstance(candidate, dict):
            return candidate
    return metadata


def resolve_forecast_cutoff(
    question: Question,
    fallback_slot: ForecastSlot = ForecastSlot.LATE,
) -> tuple[datetime, str]:
    """Resolve the forecast-time cutoff, preferring the question's explicit date.

    FinFactorBench rows encode this date both in ``forecast_date_options`` and
    in the leading ``As of YYYY-MM-DD`` clause.  Slot interpolation is retained
    only for datasets that do not provide an explicit forecast date.
    """
    metadata = family_metadata(question)
    raw_options = metadata.get("forecast_date_options", [])
    option_values = raw_options if isinstance(raw_options, list) else [raw_options]
    candidates = [
        ("metadata.forecast_date_options", value) for value in option_values
    ]
    text_match = _AS_OF_DATE_RE.search(question.question_text or "")
    if text_match:
        candidates.append(("question_text.as_of", text_match.group(1)))

    for source, raw_value in candidates:
        value = str(raw_value or "").strip()
        if not value:
            continue
        try:
            cutoff = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            continue
        if cutoff.tzinfo is None:
            cutoff = cutoff.replace(tzinfo=timezone.utc)
        if cutoff >= question.resolution_date:
            raise ValueError(
                f"Explicit forecast cutoff {cutoff.isoformat()} must predate "
                f"resolution {question.resolution_date.isoformat()} for {question.id}"
            )
        return cutoff, source

    setup = get_forecast_date_for_slot(question, fallback_slot)
    return setup["simulated_date"], f"forecast_slot.{fallback_slot.value}"
