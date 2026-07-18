"""Typed post-resolution binary scoring and aggregation artifacts."""

from decimal import Decimal
from enum import StrEnum, unique
from typing import Annotated, ClassVar, Literal
from uuid import UUID

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

from src.domain.finance.artifact import ForecastArm
from src.domain.finance.experiment_metrics import ForecastPair
from src.domain.finance.memory import OutcomeLabel, QuestionId

Score = Annotated[Decimal, Field(ge=Decimal(0), le=Decimal(1))]
Count = Annotated[int, Field(ge=0)]


class _ResolutionMetricModel(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=True,
    )


@unique
class BrierDirection(StrEnum):
    BETTER = "better"
    WORSE = "worse"
    UNCHANGED = "unchanged"


class ArmBrierScore(_ResolutionMetricModel):
    arm: ForecastArm
    positive_probability: Score
    brier: Score
    correct: bool


class PairBrierScore(_ResolutionMetricModel):
    pair: ForecastPair
    first_brier: Score
    second_brier: Score
    first_correct: bool
    second_correct: bool
    direction: BrierDirection


class ResolutionTrialAnalysis(_ResolutionMetricModel):
    trial_id: UUID
    question_id: QuestionId
    repetition_index: Count
    resolved_positive: bool
    arm_scores: tuple[ArmBrierScore, ...]
    pair_scores: tuple[PairBrierScore, ...]


class QuestionPairResolutionAggregate(_ResolutionMetricModel):
    question_id: QuestionId
    pair: ForecastPair
    successful_trial_count: Count
    eligible_trial_count: Count
    mean_first_brier: Score
    mean_second_brier: Score
    mean_first_accuracy: Score
    mean_second_accuracy: Score
    direction: BrierDirection


class PairResolutionAggregate(_ResolutionMetricModel):
    pair: ForecastPair
    resolved_question_count: Count
    suite_question_count: Count
    successful_question_count: Count
    eligible_question_count: Count
    successful_trial_count: Count
    eligible_trial_count: Count
    macro_first_brier: Score | None
    macro_second_brier: Score | None
    macro_first_accuracy: Score | None
    macro_second_accuracy: Score | None
    direction: BrierDirection | None


class ResolutionAuditEntry(_ResolutionMetricModel):
    """Persistable outcome metadata with only a digest of the source string."""

    question_id: QuestionId
    outcome_label: OutcomeLabel
    resolved_at: AwareDatetime
    resolution_source_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class ResolutionManifestAudit(_ResolutionMetricModel):
    schema_version: Literal["finance-resolution-audit/v1"] = (
        "finance-resolution-audit/v1"
    )
    suite_id: UUID
    experiment_manifest_id: str = Field(min_length=1)
    suite_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    entries: tuple[ResolutionAuditEntry, ...]


class FinanceResolutionAnalysis(_ResolutionMetricModel):
    """Distinct derived artifact bound to a verified immutable ex-ante suite."""

    schema_version: Literal["finance-resolution-analysis/v1"] = (
        "finance-resolution-analysis/v1"
    )
    source_suite_id: UUID
    source_manifest_id: str = Field(min_length=1)
    source_suite_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    resolution: ResolutionManifestAudit
    resolved_question_count: Count
    suite_question_count: Count
    trials: tuple[ResolutionTrialAnalysis, ...]
    question_pair_aggregates: tuple[QuestionPairResolutionAggregate, ...]
    pair_aggregates: tuple[PairResolutionAggregate, ...]


__all__ = [
    "ArmBrierScore",
    "BrierDirection",
    "FinanceResolutionAnalysis",
    "PairBrierScore",
    "PairResolutionAggregate",
    "QuestionPairResolutionAggregate",
    "ResolutionAuditEntry",
    "ResolutionManifestAudit",
    "ResolutionTrialAnalysis",
]
