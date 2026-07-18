"""Typed parsing for finance CLI run inputs."""

from datetime import datetime
from enum import StrEnum, unique
from hashlib import sha256
from typing import assert_never, final

from pydantic import ValidationError
from typing_extensions import override

from src.domain.finance.memory import QuestionId, QuestionKind
from src.domain.finance.provider import FinanceRunMode
from src.domain.finance.search import TargetProfile

_OFFLINE_RULE = "Offline fixture resolution rule."


@unique
class CliMode(StrEnum):
    """User-facing run-mode aliases kept separate from provider enum values."""

    CURRENT = "current"
    HISTORICAL = "historical"


@final
class FinanceCliInputError(ValueError):
    """Typed malformed argument diagnostic at the CLI boundary."""

    def __init__(self, detail: str) -> None:
        super().__init__(detail)
        self.detail: str = detail

    @override
    def __str__(self) -> str:
        return self.detail


def parse_mode(raw_mode: str) -> FinanceRunMode:
    """Parse a closed CLI mode into the production run-mode enum."""
    try:
        mode = CliMode(raw_mode.casefold())
    except ValueError as error:
        raise FinanceCliInputError(
            "mode must be exactly current or historical"
        ) from error
    match mode:
        case CliMode.CURRENT:
            return FinanceRunMode.CURRENT_UNRESOLVED
        case CliMode.HISTORICAL:
            return FinanceRunMode.HISTORICAL_BACKTEST
        case _ as unreachable:
            assert_never(unreachable)


def parse_cutoff(raw_cutoff: str) -> datetime:
    """Parse an aware ISO-8601 cutoff; naive dates fail closed."""
    try:
        cutoff = datetime.fromisoformat(raw_cutoff.replace("Z", "+00:00"))
    except ValueError as error:
        raise FinanceCliInputError(
            "cutoff must be an ISO-8601 timestamp with timezone"
        ) from error
    if cutoff.tzinfo is None or cutoff.utcoffset() is None:
        raise FinanceCliInputError("cutoff must include an explicit timezone offset")
    return cutoff


def build_target(
    question: str,
    context: str,
    cutoff: datetime,
    identifier_prefix: str = "offline-current",
) -> TargetProfile:
    """Create a sanitized binary target whose ID is derived only from input text."""
    if not question.strip():
        raise FinanceCliInputError("question must not be empty")
    if not context.strip():
        raise FinanceCliInputError("context must not be empty")
    normalized_question = question.strip()
    normalized_context = context.strip()
    question_id = QuestionId(
        f"{identifier_prefix}-{sha256(normalized_question.encode('utf-8')).hexdigest()}"
    )
    try:
        return TargetProfile(
            question_id=question_id,
            question_text=normalized_question,
            question_type=QuestionKind.BINARY,
            domain="finance",
            context=(normalized_context,),
            cutoff=cutoff,
            outcome_space=("Yes", "No"),
            resolution_rule=_OFFLINE_RULE,
        )
    except ValidationError as error:
        raise FinanceCliInputError(f"invalid target profile: {error}") from error


__all__ = [
    "FinanceCliInputError",
    "build_target",
    "parse_cutoff",
    "parse_mode",
]
