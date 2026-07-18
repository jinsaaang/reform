"""Typed, truth-free metrics for ex-ante finance experiments."""

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum, unique
from typing import Annotated, ClassVar, Final

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_core import PydanticCustomError

from src.domain.finance.artifact import ForecastArm
from src.domain.finance.experiment_panel_metrics import (
    ExAntePairSummary,
    ExAnteTieReason,
    JudgePanelMetrics,
    PanelPreference,
)
from src.domain.finance.experiment_pairs import ForecastPair, PAIR_ARMS

METRIC_TOLERANCE: Final = Decimal("1e-9")
ProbabilityDecimal = Annotated[Decimal, Field(ge=Decimal(0), le=Decimal(1))]


class _MetricModel(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=True,
    )


@unique
class ArgmaxState(StrEnum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    TIE = "tie"


@unique
class DeltaDirection(StrEnum):
    FIRST_HIGHER = "first_higher"
    SECOND_HIGHER = "second_higher"
    UNCHANGED = "unchanged"


@unique
class MajorityFlip(StrEnum):
    NONE = "none"
    NEGATIVE_TO_POSITIVE = "negative_to_positive"
    POSITIVE_TO_NEGATIVE = "positive_to_negative"
    TIE_TO_POSITIVE = "tie_to_positive"
    TIE_TO_NEGATIVE = "tie_to_negative"
    POSITIVE_TO_TIE = "positive_to_tie"
    NEGATIVE_TO_TIE = "negative_to_tie"


_FLIPS: Final = {
    (ArgmaxState.NEGATIVE, ArgmaxState.POSITIVE): (MajorityFlip.NEGATIVE_TO_POSITIVE),
    (ArgmaxState.POSITIVE, ArgmaxState.NEGATIVE): (MajorityFlip.POSITIVE_TO_NEGATIVE),
    (ArgmaxState.TIE, ArgmaxState.POSITIVE): MajorityFlip.TIE_TO_POSITIVE,
    (ArgmaxState.TIE, ArgmaxState.NEGATIVE): MajorityFlip.TIE_TO_NEGATIVE,
    (ArgmaxState.POSITIVE, ArgmaxState.TIE): MajorityFlip.POSITIVE_TO_TIE,
    (ArgmaxState.NEGATIVE, ArgmaxState.TIE): MajorityFlip.NEGATIVE_TO_TIE,
}


class ProbabilityComparison(_MetricModel):
    """Exact second-minus-first positive-label comparison."""

    pair: ForecastPair
    first_arm: ForecastArm
    second_arm: ForecastArm
    first_positive_probability: ProbabilityDecimal
    second_positive_probability: ProbabilityDecimal
    delta: Decimal
    direction: DeltaDirection
    first_argmax: ArgmaxState
    second_argmax: ArgmaxState
    majority_flip: MajorityFlip

    @model_validator(mode="after")
    def validate_derived_fields(self) -> "ProbabilityComparison":
        expected = _derive_comparison(
            self.first_positive_probability,
            self.second_positive_probability,
        )
        actual = _DerivedComparison(
            delta=self.delta,
            direction=self.direction,
            first_argmax=self.first_argmax,
            second_argmax=self.second_argmax,
            majority_flip=self.majority_flip,
        )
        if actual != expected:
            raise PydanticCustomError(
                "comparison_derivation_mismatch",
                "comparison fields must match exact probability inputs",
            )
        return self


def _argmax(probability: Decimal) -> ArgmaxState:
    if abs(probability - Decimal("0.5")) <= METRIC_TOLERANCE:
        return ArgmaxState.TIE
    return (
        ArgmaxState.POSITIVE if probability > Decimal("0.5") else ArgmaxState.NEGATIVE
    )


@dataclass(frozen=True, slots=True)
class _DerivedComparison:
    delta: Decimal
    direction: DeltaDirection
    first_argmax: ArgmaxState
    second_argmax: ArgmaxState
    majority_flip: MajorityFlip


def _derive_comparison(
    first: Decimal,
    second: Decimal,
) -> _DerivedComparison:
    delta = second - first
    first_state = _argmax(first)
    second_state = _argmax(second)
    if abs(delta) <= METRIC_TOLERANCE:
        direction = DeltaDirection.UNCHANGED
    else:
        direction = (
            DeltaDirection.SECOND_HIGHER if delta > 0 else DeltaDirection.FIRST_HIGHER
        )
    return _DerivedComparison(
        delta=delta,
        direction=direction,
        first_argmax=first_state,
        second_argmax=second_state,
        majority_flip=_FLIPS.get((first_state, second_state), MajorityFlip.NONE),
    )


def compare_positive_probabilities(
    pair: ForecastPair,
    first_probability: Decimal | float,
    second_probability: Decimal | float,
) -> ProbabilityComparison:
    """Compute exact ex-ante pair metrics from declared probabilities."""
    first = Decimal(str(first_probability))
    second = Decimal(str(second_probability))
    derived = _derive_comparison(first, second)
    first_arm, second_arm = PAIR_ARMS[pair]
    return ProbabilityComparison(
        pair=pair,
        first_arm=first_arm,
        second_arm=second_arm,
        first_positive_probability=first,
        second_positive_probability=second,
        delta=derived.delta,
        direction=derived.direction,
        first_argmax=derived.first_argmax,
        second_argmax=derived.second_argmax,
        majority_flip=derived.majority_flip,
    )


__all__ = [
    "ArgmaxState",
    "DeltaDirection",
    "ExAntePairSummary",
    "ExAnteTieReason",
    "ForecastPair",
    "JudgePanelMetrics",
    "METRIC_TOLERANCE",
    "MajorityFlip",
    "PAIR_ARMS",
    "PanelPreference",
    "ProbabilityComparison",
    "compare_positive_probabilities",
]
