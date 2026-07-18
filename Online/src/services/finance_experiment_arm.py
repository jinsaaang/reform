"""Independent finance experiment arm execution."""

from dataclasses import dataclass
from typing import Final

from pydantic import ValidationError

from src.agents.finance_forecast_agent import (
    FinanceForecastAgent,
    ForecastAgentSucceeded,
)
from src.domain.finance.experiment import (
    ArmFailed,
    ArmFailureReason,
    ArmResult,
    ArmSucceeded,
    ArmUnavailable,
    ArmUnavailableReason,
    FinanceBinaryQuestion,
    FinanceEvidenceSnapshot,
    FinanceExperimentManifest,
    ForecastArm,
    TreatmentAudit,
)
from src.domain.finance.forecast import ForecasterInput
from src.domain.finance.memory import ResolvedDagEpisode
from src.domain.finance.pipeline import ForecastAbstentionReason
from src.domain.finance.search import EvidencePack, TargetProfile


@dataclass(frozen=True, slots=True)
class FinanceArmForecastRequest:
    """Typed treatment inputs for one independently terminal arm."""

    arm: ForecastArm
    target_profile: TargetProfile
    evidence_pack: EvidencePack
    historical_memory: tuple[ResolvedDagEpisode, ...]
    treatment: TreatmentAudit


def target_profile(
    manifest: FinanceExperimentManifest,
    question: FinanceBinaryQuestion,
) -> TargetProfile:
    """Build the current target without a realized outcome or DAG."""
    return TargetProfile(
        question_id=question.question_id,
        question_text=question.question_text,
        question_type=question.question_type,
        domain=question.domain,
        context=question.context,
        cutoff=question.forecast_cutoff or manifest.cutoff,
        outcome_space=question.outcome_space,
        resolution_rule=question.resolution_rule,
    )


def direct_treatment() -> TreatmentAudit:
    """Return the mechanically empty Direct treatment audit."""
    return TreatmentAudit(
        arm=ForecastArm.DIRECT,
        evidence_snapshot_digest=None,
        evidence_snapshot_bytes=0,
        evidence_item_count=0,
        historical_memory_episode_count=0,
    )


def snapshot_treatment(
    arm: ForecastArm,
    snapshot: FinanceEvidenceSnapshot,
    memory_count: int,
) -> TreatmentAudit:
    """Bind an evidence-bearing arm to exact frozen snapshot metadata."""
    return TreatmentAudit(
        arm=arm,
        evidence_snapshot_digest=snapshot.digest,
        evidence_snapshot_bytes=snapshot.byte_length,
        evidence_item_count=len(snapshot.items),
        historical_memory_episode_count=memory_count,
    )


def unavailable_search_arms(
    reason: ArmUnavailableReason,
) -> tuple[ArmUnavailable, ArmUnavailable]:
    """Return terminal B/C records when A0 preparation cannot complete."""
    return (
        ArmUnavailable(
            arm=ForecastArm.SEARCH_ONLY,
            attempts=(),
            treatment=None,
            reason=reason,
        ),
        ArmUnavailable(
            arm=ForecastArm.SEARCH_DAG,
            attempts=(),
            treatment=None,
            reason=reason,
        ),
    )


def memory_unavailable() -> ArmUnavailable:
    """Return C unavailable while preserving completed A/B records."""
    return ArmUnavailable(
        arm=ForecastArm.SEARCH_DAG,
        attempts=(),
        treatment=None,
        reason=ArmUnavailableReason.MEMORY_UNAVAILABLE,
    )


_FAILURE_REASONS: Final[dict[ForecastAbstentionReason, ArmFailureReason]] = {
    ForecastAbstentionReason.PROVIDER_ERROR: ArmFailureReason.PROVIDER_ERROR,
    ForecastAbstentionReason.MALFORMED_OUTPUT: ArmFailureReason.MALFORMED_OUTPUT,
    ForecastAbstentionReason.OUTCOME_SPACE_MISMATCH: (
        ArmFailureReason.OUTCOME_SPACE_MISMATCH
    ),
    ForecastAbstentionReason.UNKNOWN_EVIDENCE_REFERENCE: (
        ArmFailureReason.INVALID_INPUT
    ),
    ForecastAbstentionReason.UNKNOWN_HISTORICAL_REFERENCE: (
        ArmFailureReason.INVALID_INPUT
    ),
}


def _failure_reason(reason: ForecastAbstentionReason) -> ArmFailureReason:
    return _FAILURE_REASONS[reason]


def execute_finance_forecast_arm(
    forecaster: FinanceForecastAgent,
    request: FinanceArmForecastRequest,
) -> ArmResult:
    """Return one terminal record without affecting sibling arms."""
    try:
        forecast_input = ForecasterInput(
            target_profile=request.target_profile,
            evidence_pack=request.evidence_pack,
            historical_memory=request.historical_memory,
        )
    except ValidationError:
        return ArmFailed(
            arm=request.arm,
            attempts=(),
            treatment=request.treatment,
            reason=ArmFailureReason.INVALID_INPUT,
        )
    result = forecaster.forecast(forecast_input)
    if isinstance(result, ForecastAgentSucceeded):
        return ArmSucceeded(
            arm=request.arm,
            attempts=(),
            treatment=request.treatment,
            forecast=result.forecast_result,
        )
    return ArmFailed(
        arm=request.arm,
        attempts=(),
        treatment=request.treatment,
        reason=_failure_reason(result.reason),
    )


__all__ = [
    "FinanceArmForecastRequest",
    "direct_treatment",
    "execute_finance_forecast_arm",
    "memory_unavailable",
    "snapshot_treatment",
    "target_profile",
    "unavailable_search_arms",
]
