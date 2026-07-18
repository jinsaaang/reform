"""Typed ex-ante three-member panel metrics and denominators."""

from decimal import Decimal
from enum import StrEnum, unique
from typing import Annotated, ClassVar

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_core import PydanticCustomError

from src.domain.finance.experiment_pairs import ForecastPair

ProbabilityDecimal = Annotated[Decimal, Field(ge=Decimal(0), le=Decimal(1))]


class _PanelMetricModel(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=True,
    )


@unique
class PanelPreference(StrEnum):
    FIRST = "first"
    SECOND = "second"
    TIE = "tie"


@unique
class ExAnteTieReason(StrEnum):
    NO_CONSENSUS = "no_consensus"
    NO_QUORUM = "no_quorum"


class JudgePanelMetrics(_PanelMetricModel):
    """Closed aggregate of one fixed three-member reasoning panel."""

    overall_preference: PanelPreference
    preference_eligible: bool
    tie_reason: ExAnteTieReason | None
    first_votes: Annotated[int, Field(ge=0, le=3)]
    second_votes: Annotated[int, Field(ge=0, le=3)]
    tie_votes: Annotated[int, Field(ge=0, le=3)]
    inconsistent_count: Annotated[int, Field(ge=0, le=3)]
    invalid_count: Annotated[int, Field(ge=0, le=3)]
    evaluable_count: Annotated[int, Field(ge=0, le=3)]
    two_parse_valid_count: Annotated[int, Field(ge=0, le=3)]
    attempted_call_count: Annotated[int, Field(ge=0, le=6)]
    invalid_rate: ProbabilityDecimal = Decimal(0)
    inconsistent_rate: ProbabilityDecimal = Decimal(0)
    agreement: ProbabilityDecimal | None
    order_consistency: ProbabilityDecimal | None

    @model_validator(mode="after")
    def validate_counts_and_rates(self) -> "JudgePanelMetrics":
        member_count = (
            self.first_votes
            + self.second_votes
            + self.tie_votes
            + self.inconsistent_count
            + self.invalid_count
        )
        expected_evaluable = self.first_votes + self.second_votes + self.tie_votes
        expected_two_parse_valid = expected_evaluable + self.inconsistent_count
        rates_match = self.invalid_rate == Decimal(self.invalid_count) / Decimal(
            3
        ) and self.inconsistent_rate == Decimal(self.inconsistent_count) / Decimal(3)
        if (
            member_count != 3
            or self.evaluable_count != expected_evaluable
            or self.two_parse_valid_count != expected_two_parse_valid
            or self.attempted_call_count != 6
            or not rates_match
        ):
            raise PydanticCustomError(
                "panel_metric_mismatch",
                "panel counts, rates, and call denominators must agree",
            )
        expected_agreement = (
            None
            if expected_evaluable == 0
            else Decimal(max(self.first_votes, self.second_votes, self.tie_votes))
            / Decimal(expected_evaluable)
        )
        if self.agreement != expected_agreement:
            raise PydanticCustomError(
                "panel_agreement_mismatch",
                "agreement must use the evaluable-member denominator",
            )
        expected_order_consistency = (
            None
            if expected_two_parse_valid == 0
            else Decimal(expected_evaluable) / Decimal(expected_two_parse_valid)
        )
        if self.order_consistency != expected_order_consistency:
            raise PydanticCustomError(
                "panel_order_consistency_mismatch",
                "order consistency must use the two-parse-valid denominator",
            )
        expected_eligible = expected_evaluable >= 2
        if self.preference_eligible != expected_eligible:
            raise PydanticCustomError(
                "panel_quorum_mismatch",
                "preference eligibility requires at least two evaluable members",
            )
        expected_preference = PanelPreference.TIE
        if self.first_votes >= 2:
            expected_preference = PanelPreference.FIRST
        elif self.second_votes >= 2:
            expected_preference = PanelPreference.SECOND
        expected_tie_reason = (
            None
            if expected_preference is not PanelPreference.TIE
            else (
                ExAnteTieReason.NO_CONSENSUS
                if expected_eligible
                else ExAnteTieReason.NO_QUORUM
            )
        )
        if (
            self.overall_preference is not expected_preference
            or self.tie_reason is not expected_tie_reason
        ):
            raise PydanticCustomError(
                "panel_tie_reason_mismatch",
                "panel preference and closed tie reason must match vote counts",
            )
        return self


class ExAntePairSummary(_PanelMetricModel):
    """Explicit ex-ante denominators and panel diagnostics for one pair."""

    pair: ForecastPair
    scheduled_trial_count: Annotated[int, Field(ge=0)]
    completed_panel_count: Annotated[int, Field(ge=0)]
    preference_eligible_panel_count: Annotated[int, Field(ge=0)]
    no_quorum_panel_count: Annotated[int, Field(ge=0)]
    judge_member_count: Annotated[int, Field(ge=0)]
    invalid_member_count: Annotated[int, Field(ge=0)]
    inconsistent_member_count: Annotated[int, Field(ge=0)]
    agreement_observation_count: Annotated[int, Field(ge=0)]
    order_consistency_observation_count: Annotated[int, Field(ge=0)]
    agreement_mean: ProbabilityDecimal | None
    order_consistency_mean: ProbabilityDecimal | None


__all__ = [
    "ExAntePairSummary",
    "ExAnteTieReason",
    "JudgePanelMetrics",
    "PanelPreference",
]
