"""Runtime guards for exhaustively handled finance experiment variants."""

from typing import Never

from pydantic import BaseModel

ClosedVariant = BaseModel | str | None


def runtime_persisted_variant(value: BaseModel) -> BaseModel:
    """Retain the runtime guard while callers match a closed static union."""
    return value


def unexpected_persisted_variant(value: BaseModel) -> Never:
    """Fail closed if runtime data escapes the declared discriminated union."""
    raise AssertionError(f"unexpected persisted variant: {type(value).__name__}")


def runtime_closed_variant(value: ClosedVariant) -> ClosedVariant:
    """Widen a closed runtime variant before an exhaustive match."""
    return value


def unexpected_closed_variant(value: ClosedVariant) -> Never:
    """Fail closed if a widened runtime variant escapes its contract."""
    raise AssertionError(f"unexpected closed variant: {type(value).__name__}")


__all__ = [
    "runtime_closed_variant",
    "runtime_persisted_variant",
    "unexpected_closed_variant",
    "unexpected_persisted_variant",
]
