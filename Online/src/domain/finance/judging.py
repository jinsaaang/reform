"""Strict result algebra contracts for blind finance judging."""

from enum import StrEnum, unique
from typing import Annotated, ClassVar, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_core import PydanticCustomError

from src.domain.finance.experiment_manifest import Seed
from src.domain.finance.experiment_telemetry import (
    ProviderAttemptStatus,
    ProviderAttemptTelemetry,
)
from src.domain.finance.judge_views import NeutralCandidate


class _JudgeModel(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=True,
        str_strip_whitespace=True,
    )


@unique
class JudgeDiagnosticDimension(StrEnum):
    """Seven preregistered, non-weighted reasoning diagnostics."""

    FINANCE_MECHANISM_PLAUSIBILITY = "finance_mechanism_plausibility"
    EVIDENCE_GROUNDING = "evidence_grounding"
    SCENARIO_COHERENCE = "scenario_coherence"
    COUNTEREVIDENCE = "counterevidence"
    ASSUMPTIONS_TRIGGERS_DISCONFIRMERS = "assumptions_triggers_disconfirmers"
    UNCERTAINTY_QUALIFICATION = "uncertainty_qualification"
    ANALOG_USE_LIMITATIONS = "analog_use_limitations"


@unique
class JudgePreference(StrEnum):
    ANSWER_A = "answer_a"
    ANSWER_B = "answer_b"
    TIE = "tie"


@unique
class JudgeDiagnosticAssessment(StrEnum):
    ANSWER_A = "answer_a"
    ANSWER_B = "answer_b"
    EQUAL = "equal"
    INSUFFICIENT = "insufficient"


class JudgeDiagnostic(_JudgeModel):
    dimension: JudgeDiagnosticDimension
    assessment: JudgeDiagnosticAssessment


def _require_complete_diagnostics(
    diagnostics: tuple[JudgeDiagnostic, ...],
) -> None:
    dimensions = tuple(item.dimension for item in diagnostics)
    if len(dimensions) != len(set(dimensions)) or set(dimensions) != set(
        JudgeDiagnosticDimension
    ):
        raise PydanticCustomError(
            "invalid_judge_diagnostics",
            "all seven judge diagnostic dimensions are required exactly once",
        )


class JudgeProviderResponse(_JudgeModel):
    """Parsed structured provider response; exact raw output is never retained."""

    preference: JudgePreference
    diagnostics: tuple[JudgeDiagnostic, ...] = Field(min_length=7, max_length=7)

    @model_validator(mode="after")
    def validate_seven_dimensions(self) -> "JudgeProviderResponse":
        """Require every preregistered dimension exactly once."""
        _require_complete_diagnostics(self.diagnostics)
        return self


@unique
class JudgeCallOrientation(StrEnum):
    CANONICAL = "canonical"
    SWAPPED = "swapped"


@unique
class MappedJudgePreference(StrEnum):
    CANDIDATE_1 = "candidate_1"
    CANDIDATE_2 = "candidate_2"
    TIE = "tie"


@unique
class JudgeCallFailureReason(StrEnum):
    PROVIDER_ERROR = "provider_error"
    MALFORMED_JSON = "malformed_json"
    SCHEMA_ERROR = "schema_error"
    FORBIDDEN_CONTENT = "forbidden_content"


@unique
class JudgeCallFailureClass(StrEnum):
    """Closed, persistence-safe failure classifications."""

    JUDGE_PROVIDER_ERROR = "JudgeProviderError"
    JSON_DECODE_ERROR = "JSONDecodeError"
    VALIDATION_ERROR = "ValidationError"


class _JudgeCall(_JudgeModel):
    orientation: JudgeCallOrientation
    requested_seed: Seed
    answer_a_candidate: NeutralCandidate
    answer_b_candidate: NeutralCandidate


class JudgeCallSucceeded(_JudgeCall):
    status: Literal["succeeded"] = "succeeded"
    attempt: ProviderAttemptTelemetry
    mapped_preference: MappedJudgePreference
    diagnostics: tuple[JudgeDiagnostic, ...] = Field(min_length=7, max_length=7)

    @model_validator(mode="after")
    def validate_successful_attempt(self) -> "JudgeCallSucceeded":
        if self.attempt.status is not ProviderAttemptStatus.SUCCEEDED:
            raise PydanticCustomError(
                "judge_call_attempt_mismatch",
                "successful judge calls require successful attempt telemetry",
            )
        _require_complete_diagnostics(self.diagnostics)
        return self


class JudgeCallInvalid(_JudgeCall):
    status: Literal["invalid"] = "invalid"
    attempt: ProviderAttemptTelemetry
    reason: JudgeCallFailureReason
    failure_class: JudgeCallFailureClass

    @model_validator(mode="after")
    def validate_invalid_attempt(self) -> "JudgeCallInvalid":
        if self.attempt.status is ProviderAttemptStatus.SUCCEEDED:
            raise PydanticCustomError(
                "judge_call_attempt_mismatch",
                "invalid judge calls require failed or invalid attempt telemetry",
            )
        return self


JudgeCallRecord: TypeAlias = Annotated[
    JudgeCallSucceeded | JudgeCallInvalid,
    Field(discriminator="status"),
]


class _SingleJudgeTerminal(_JudgeModel):
    member_id: str = Field(min_length=1)
    calls: tuple[JudgeCallRecord, ...] = Field(min_length=2, max_length=2)

    @model_validator(mode="after")
    def validate_two_order_calls(self) -> "_SingleJudgeTerminal":
        """Require canonical then swapped calls with one common seed."""
        orientations = tuple(call.orientation for call in self.calls)
        if orientations != (
            JudgeCallOrientation.CANONICAL,
            JudgeCallOrientation.SWAPPED,
        ):
            raise PydanticCustomError(
                "invalid_order_swap_calls",
                "single-judge result requires canonical then swapped calls",
            )
        if self.calls[0].requested_seed != self.calls[1].requested_seed:
            raise PydanticCustomError(
                "judge_seed_mismatch",
                "both answer orders must use one common requested seed",
            )
        return self


class SingleJudgeWinner(_SingleJudgeTerminal):
    status: Literal["valid"] = "valid"
    winner: NeutralCandidate


class SingleJudgeTie(_SingleJudgeTerminal):
    status: Literal["tie"] = "tie"


class SingleJudgeInconsistent(_SingleJudgeTerminal):
    status: Literal["inconsistent"] = "inconsistent"


class SingleJudgeInvalid(_SingleJudgeTerminal):
    status: Literal["invalid"] = "invalid"


SingleJudgeResult: TypeAlias = Annotated[
    SingleJudgeWinner | SingleJudgeTie | SingleJudgeInconsistent | SingleJudgeInvalid,
    Field(discriminator="status"),
]


class ThreeMemberJudgePanelResult(_JudgeModel):
    """Fixed-size terminal member records before panel aggregation."""

    schema_version: Literal["finance-three-member-judge-panel/v1"] = (
        "finance-three-member-judge-panel/v1"
    )
    members: tuple[SingleJudgeResult, ...] = Field(min_length=3, max_length=3)

    @model_validator(mode="after")
    def validate_unique_members(self) -> "ThreeMemberJudgePanelResult":
        member_ids = tuple(member.member_id for member in self.members)
        if len(member_ids) != len(set(member_ids)):
            raise PydanticCustomError(
                "duplicate_judge_panel_member",
                "the fixed judge panel requires three unique members",
            )
        return self


__all__ = [
    "JudgeCallFailureClass",
    "JudgeCallFailureReason",
    "JudgeCallInvalid",
    "JudgeCallOrientation",
    "JudgeCallRecord",
    "JudgeCallSucceeded",
    "JudgeDiagnostic",
    "JudgeDiagnosticAssessment",
    "JudgeDiagnosticDimension",
    "JudgePreference",
    "JudgeProviderResponse",
    "MappedJudgePreference",
    "SingleJudgeInconsistent",
    "SingleJudgeInvalid",
    "SingleJudgeResult",
    "SingleJudgeTie",
    "SingleJudgeWinner",
    "ThreeMemberJudgePanelResult",
]
