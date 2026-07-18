"""Terminal arm, provider-attempt, trial, and suite result contracts."""

from enum import StrEnum, unique
from typing import Annotated, ClassVar, Literal, TypeAlias
from uuid import UUID

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator
from pydantic_core import PydanticCustomError

from src.domain.finance.artifact import ForecastArm
from src.domain.finance.experiment_manifest import FinanceExperimentManifest
from src.domain.finance.experiment_telemetry import (
    ProviderAttemptStatus,
    ProviderAttemptTelemetry,
    ProviderSeedEffective,
    ProviderSeedNotRequested,
    ProviderSeedUnsupported,
    ProviderUsageReported,
    ProviderUsageUnavailable,
)
from src.domain.finance.forecast import ForecastResult
from src.domain.finance.memory import QuestionId

Sha256Digest: TypeAlias = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class _ResultModel(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=True,
        str_strip_whitespace=True,
    )


class TreatmentAudit(_ResultModel):
    """Mechanical proof of each arm's evidence and memory inputs."""

    arm: ForecastArm
    evidence_snapshot_digest: Sha256Digest | None
    evidence_snapshot_bytes: Annotated[int, Field(ge=0)]
    evidence_item_count: Annotated[int, Field(ge=0)]
    historical_memory_episode_count: Annotated[int, Field(ge=0)]

    @model_validator(mode="after")
    def validate_arm_inputs(self) -> "TreatmentAudit":
        """Make Direct/Search-only/Search+DAG input mixing unrepresentable."""
        valid_by_arm = {
            ForecastArm.DIRECT: (
                self.evidence_snapshot_digest is None
                and self.evidence_snapshot_bytes == 0
                and self.evidence_item_count == 0
                and self.historical_memory_episode_count == 0
            ),
            ForecastArm.SEARCH_ONLY: (
                self.evidence_snapshot_digest is not None
                and self.evidence_snapshot_bytes > 0
                and self.evidence_item_count > 0
                and self.historical_memory_episode_count == 0
            ),
            ForecastArm.SEARCH_DAG: (
                self.evidence_snapshot_digest is not None
                and self.evidence_snapshot_bytes > 0
                and self.evidence_item_count > 0
                and self.historical_memory_episode_count > 0
            ),
        }
        valid = valid_by_arm[self.arm]
        if not valid:
            raise PydanticCustomError(
                "arm_input_inconsistency",
                "treatment inputs do not match the declared forecast arm",
            )
        return self


@unique
class ArmUnavailableReason(StrEnum):
    NOT_ATTEMPTED = "not_attempted"
    SEARCH_UNAVAILABLE = "search_unavailable"
    NO_ADMITTED_EVIDENCE = "no_admitted_evidence"
    MEMORY_UNAVAILABLE = "memory_unavailable"


@unique
class ArmFailureReason(StrEnum):
    PROVIDER_ERROR = "provider_error"
    MALFORMED_OUTPUT = "malformed_output"
    OUTCOME_SPACE_MISMATCH = "outcome_space_mismatch"
    INVALID_INPUT = "invalid_input"


class _ArmTerminal(_ResultModel):
    arm: ForecastArm
    attempts: tuple[ProviderAttemptTelemetry, ...]


class ArmSucceeded(_ArmTerminal):
    status: Literal["succeeded"] = "succeeded"
    treatment: TreatmentAudit
    forecast: ForecastResult


class ArmUnavailable(_ArmTerminal):
    status: Literal["unavailable"] = "unavailable"
    treatment: TreatmentAudit | None
    reason: ArmUnavailableReason


class ArmFailed(_ArmTerminal):
    status: Literal["failed"] = "failed"
    treatment: TreatmentAudit
    reason: ArmFailureReason


ArmResult: TypeAlias = Annotated[
    ArmSucceeded | ArmUnavailable | ArmFailed,
    Field(discriminator="status"),
]


class FinanceExperimentTrial(_ResultModel):
    """One question/repetition with exactly one terminal record per arm."""

    trial_id: UUID
    question_id: QuestionId
    repetition_index: Annotated[int, Field(ge=0)]
    preparation_attempts: tuple[ProviderAttemptTelemetry, ...]
    arms: tuple[ArmResult, ...] = Field(min_length=3, max_length=3)

    @model_validator(mode="after")
    def validate_three_arm_trial(self) -> "FinanceExperimentTrial":
        """Require A/B/C once and byte-identical fixed evidence for B/C."""
        by_arm = {record.arm: record for record in self.arms}
        if set(by_arm) != set(ForecastArm) or len(by_arm) != 3:
            raise PydanticCustomError(
                "invalid_trial_arms",
                "each trial requires one direct, search_only, and search_dag record",
            )
        for record in self.arms:
            treatment = record.treatment
            if treatment is not None and treatment.arm is not record.arm:
                raise PydanticCustomError(
                    "arm_treatment_mismatch",
                    "arm result and treatment audit must identify the same arm",
                )
        search_only = by_arm[ForecastArm.SEARCH_ONLY].treatment
        search_dag = by_arm[ForecastArm.SEARCH_DAG].treatment
        if (
            search_only is not None
            and search_dag is not None
            and (
                search_only.evidence_snapshot_digest
                != search_dag.evidence_snapshot_digest
                or search_only.evidence_snapshot_bytes
                != search_dag.evidence_snapshot_bytes
            )
        ):
            raise PydanticCustomError(
                "fixed_snapshot_mismatch",
                "search_only and search_dag must share snapshot digest and bytes",
            )
        return self


@unique
class FinanceSuiteStatus(StrEnum):
    COMPLETE = "complete"
    PARTIAL = "partial"
    FAILED = "failed"


class FinanceExperimentSuite(_ResultModel):
    """Ex-ante suite with a manifest-bound, explicit terminal status."""

    schema_version: Literal["finance-experiment-suite/v1"]
    suite_id: UUID
    created_at: AwareDatetime
    status: FinanceSuiteStatus
    manifest: FinanceExperimentManifest
    trials: tuple[FinanceExperimentTrial, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_schedule_and_status(self) -> "FinanceExperimentSuite":
        """Require the ordered schedule and derive complete/partial/failed truth."""
        expected = tuple(
            (question.question_id, repetition)
            for question in self.manifest.questions
            for repetition in range(self.manifest.repetitions)
        )
        actual = tuple(
            (trial.question_id, trial.repetition_index) for trial in self.trials
        )
        if actual != expected:
            raise PydanticCustomError(
                "suite_schedule_mismatch",
                "suite trials must exactly follow manifest question/repetition order",
            )
        records = tuple(record for trial in self.trials for record in trial.arms)
        success_count = sum(record.status == "succeeded" for record in records)
        incomplete_count = sum(record.status != "succeeded" for record in records)
        expected_status = (
            FinanceSuiteStatus.FAILED
            if success_count == 0
            else FinanceSuiteStatus.PARTIAL
            if incomplete_count > 0
            else FinanceSuiteStatus.COMPLETE
        )
        if self.status is not expected_status:
            raise PydanticCustomError(
                "suite_status_mismatch",
                "suite status does not match terminal arm records",
            )
        return self


__all__ = [
    "ArmFailed",
    "ArmFailureReason",
    "ArmResult",
    "ArmSucceeded",
    "ArmUnavailable",
    "ArmUnavailableReason",
    "FinanceExperimentSuite",
    "FinanceExperimentTrial",
    "FinanceSuiteStatus",
    "ProviderAttemptStatus",
    "ProviderAttemptTelemetry",
    "ProviderSeedEffective",
    "ProviderSeedNotRequested",
    "ProviderSeedUnsupported",
    "ProviderUsageReported",
    "ProviderUsageUnavailable",
    "Sha256Digest",
    "TreatmentAudit",
]
