"""Typed source and failure contracts for finance judge-view construction."""

from dataclasses import dataclass, field
from enum import StrEnum, unique

from typing_extensions import override

from src.domain.finance.experiment_results import ArmResult, ArmSucceeded
from src.domain.finance.forecast import ForecasterInput
from src.domain.finance.judge_views import NeutralCandidate
from src.domain.finance.sanitized_artifact import TransientForbiddenValueRegistry


@dataclass(frozen=True, slots=True)
class JudgeCandidateSource:
    """One terminal arm record paired with its exact typed provider input."""

    arm_result: ArmResult
    forecast_input: ForecasterInput


@dataclass(frozen=True, slots=True)
class JudgeViewRequest:
    """Two candidates plus transient values that must not survive."""

    candidates: tuple[JudgeCandidateSource, JudgeCandidateSource]
    transient_forbidden: TransientForbiddenValueRegistry = field(
        default_factory=TransientForbiddenValueRegistry
    )


@dataclass(frozen=True, slots=True)
class CanonicalJudgeCandidate:
    """Validated successful source under an order-independent neutral alias."""

    alias: NeutralCandidate
    arm_result: ArmSucceeded
    forecast_input: ForecasterInput


@unique
class JudgeViewFailureReason(StrEnum):
    TARGET_MISMATCH = "target_mismatch"
    ARM_NOT_SUCCEEDED = "arm_not_succeeded"
    DUPLICATE_CANDIDATE = "duplicate_candidate"
    TREATMENT_MISMATCH = "treatment_mismatch"
    POST_CUTOFF_EVIDENCE = "post_cutoff_evidence"
    CONFLICTING_EVIDENCE = "conflicting_evidence"
    CONFLICTING_MEMORY = "conflicting_memory"
    INVALID_SOURCE = "invalid_source"
    INVALID_ALIAS_SALT = "invalid_alias_salt"
    FORBIDDEN_CONTENT = "forbidden_content"


@dataclass(frozen=True, slots=True)
class JudgeViewError(Exception):
    """Closed judge-view rejection without source or provider text."""

    reason: JudgeViewFailureReason
    failure_class: str

    @override
    def __str__(self) -> str:
        return f"judge view rejected: {self.reason.value} ({self.failure_class})"


__all__ = [
    "CanonicalJudgeCandidate",
    "JudgeCandidateSource",
    "JudgeViewError",
    "JudgeViewFailureReason",
    "JudgeViewRequest",
]
