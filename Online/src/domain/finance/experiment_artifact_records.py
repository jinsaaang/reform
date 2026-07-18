"""Sanitized arm, provider-exchange, and pair persistence records."""

from decimal import Decimal
from enum import StrEnum, unique
from typing import Annotated, ClassVar, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_core import PydanticCustomError

from src.domain.finance.artifact import ForecastArm
from src.domain.finance.experiment_metrics import (
    ForecastPair,
    JudgePanelMetrics,
    ProbabilityComparison,
)
from src.domain.finance.experiment_results import (
    ArmFailureReason,
    ArmUnavailableReason,
    ProviderAttemptTelemetry,
    TreatmentAudit,
)
from src.domain.finance.judge_views import (
    SanitizedCandidateView,
    SanitizedEvidenceView,
    SanitizedMemoryView,
    SanitizedTargetView,
)
from src.domain.finance.sanitized_artifact import FinanceAliasAudit

Sha256Digest: TypeAlias = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
PersistedProbability: TypeAlias = Annotated[
    Decimal,
    Field(ge=Decimal(0), le=Decimal(1)),
]


class _PersistedRecordModel(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=True,
        str_strip_whitespace=True,
    )


@unique
class ProviderExchangeKind(StrEnum):
    SEARCH = "search"
    FORECAST = "forecast"
    JUDGE = "judge"


class ProviderExchangeDigest(_PersistedRecordModel):
    """Typed telemetry plus hashes of exact provider input and response bytes."""

    exchange_id: str = Field(min_length=1)
    kind: ProviderExchangeKind
    input_sha256: Sha256Digest
    response_sha256: Sha256Digest | None
    attempt: ProviderAttemptTelemetry

    @model_validator(mode="after")
    def validate_response_hash_presence(self) -> "ProviderExchangeDigest":
        response_exists = self.attempt.output_bytes is not None
        if response_exists != (self.response_sha256 is not None):
            raise PydanticCustomError(
                "provider_response_hash_mismatch",
                "response hash presence must match typed output telemetry",
            )
        return self


class SanitizedArmReasoning(_PersistedRecordModel):
    """Complete aliased public reasoning for one successful arm."""

    schema_version: Literal["finance-sanitized-arm-reasoning/v1"] = (
        "finance-sanitized-arm-reasoning/v1"
    )
    target: SanitizedTargetView
    evidence: tuple[SanitizedEvidenceView, ...]
    memory: tuple[SanitizedMemoryView, ...]
    candidate: SanitizedCandidateView
    alias_audit: FinanceAliasAudit


class _PersistedArm(_PersistedRecordModel):
    arm: ForecastArm
    exchanges: tuple[ProviderExchangeDigest, ...]

    @model_validator(mode="after")
    def validate_forecast_exchanges(self) -> "_PersistedArm":
        if any(
            item.kind is not ProviderExchangeKind.FORECAST for item in self.exchanges
        ):
            raise PydanticCustomError(
                "arm_exchange_kind",
                "arm exchanges must be forecast provider interactions",
            )
        return self


class PersistedArmSucceeded(_PersistedArm):
    status: Literal["succeeded"] = "succeeded"
    treatment: TreatmentAudit
    positive_probability: PersistedProbability
    reasoning: SanitizedArmReasoning

    @model_validator(mode="after")
    def validate_treatment_arm(self) -> "PersistedArmSucceeded":
        if self.treatment.arm is not self.arm:
            raise PydanticCustomError(
                "persisted_arm_treatment_mismatch",
                "treatment must identify the persisted arm",
            )
        return self


class PersistedArmUnavailable(_PersistedArm):
    status: Literal["unavailable"] = "unavailable"
    treatment: TreatmentAudit | None
    reason: ArmUnavailableReason

    @model_validator(mode="after")
    def validate_treatment_arm(self) -> "PersistedArmUnavailable":
        if self.treatment is not None and self.treatment.arm is not self.arm:
            raise PydanticCustomError(
                "persisted_arm_treatment_mismatch",
                "treatment must identify the persisted arm",
            )
        return self


class PersistedArmFailed(_PersistedArm):
    status: Literal["failed"] = "failed"
    treatment: TreatmentAudit
    reason: ArmFailureReason

    @model_validator(mode="after")
    def validate_treatment_arm(self) -> "PersistedArmFailed":
        if self.treatment.arm is not self.arm:
            raise PydanticCustomError(
                "persisted_arm_treatment_mismatch",
                "treatment must identify the persisted arm",
            )
        return self


PersistedArmResult: TypeAlias = Annotated[
    PersistedArmSucceeded | PersistedArmUnavailable | PersistedArmFailed,
    Field(discriminator="status"),
]


@unique
class PairUnavailableReason(StrEnum):
    ARM_UNAVAILABLE = "arm_unavailable"
    PANEL_UNAVAILABLE = "panel_unavailable"
    SANITIZATION_REJECTED = "sanitization_rejected"


class PersistedPairCompleted(_PersistedRecordModel):
    status: Literal["completed"] = "completed"
    comparison: ProbabilityComparison
    panel: JudgePanelMetrics
    judge_exchanges: tuple[ProviderExchangeDigest, ...]

    @model_validator(mode="after")
    def validate_judge_exchanges(self) -> "PersistedPairCompleted":
        if len(self.judge_exchanges) != self.panel.attempted_call_count or any(
            item.kind is not ProviderExchangeKind.JUDGE for item in self.judge_exchanges
        ):
            raise PydanticCustomError(
                "panel_exchange_mismatch",
                "completed panels require one digest per judge call",
            )
        return self


class PersistedPairUnavailable(_PersistedRecordModel):
    status: Literal["unavailable"] = "unavailable"
    pair: ForecastPair
    reason: PairUnavailableReason
    judge_exchanges: tuple[ProviderExchangeDigest, ...] = ()

    @model_validator(mode="after")
    def validate_judge_exchanges(self) -> "PersistedPairUnavailable":
        if any(
            item.kind is not ProviderExchangeKind.JUDGE for item in self.judge_exchanges
        ):
            raise PydanticCustomError(
                "panel_exchange_mismatch",
                "unavailable panels may contain only judge exchange digests",
            )
        return self


PersistedPairResult: TypeAlias = Annotated[
    PersistedPairCompleted | PersistedPairUnavailable,
    Field(discriminator="status"),
]


__all__ = [
    "PairUnavailableReason",
    "PersistedArmFailed",
    "PersistedArmResult",
    "PersistedArmSucceeded",
    "PersistedArmUnavailable",
    "PersistedPairCompleted",
    "PersistedPairResult",
    "PersistedPairUnavailable",
    "ProviderExchangeDigest",
    "ProviderExchangeKind",
    "SanitizedArmReasoning",
]
