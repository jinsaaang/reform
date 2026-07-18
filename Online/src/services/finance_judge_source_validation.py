"""Validate and canonicalize typed sources before finance judge sanitization."""

from dataclasses import dataclass
from typing import Final

from src.domain.finance.artifact import ForecastArm
from src.domain.finance.experiment_results import ArmSucceeded
from src.domain.finance.experiment_snapshot import FinanceEvidenceSnapshot
from src.domain.finance.judge_source import (
    CanonicalJudgeCandidate,
    JudgeCandidateSource,
    JudgeViewError,
    JudgeViewFailureReason,
    JudgeViewRequest,
)
from src.domain.finance.judge_views import NeutralCandidate
from src.domain.finance.memory import DagId, EvidenceId, ResolvedDagEpisode
from src.domain.finance.search import EvidenceItem

_ARM_RANK: Final = {
    ForecastArm.DIRECT: 0,
    ForecastArm.SEARCH_ONLY: 1,
    ForecastArm.SEARCH_DAG: 2,
}


@dataclass(frozen=True, slots=True)
class ValidatedJudgeSources:
    candidates: tuple[CanonicalJudgeCandidate, CanonicalJudgeCandidate]
    evidence: tuple[EvidenceItem, ...]
    memory: tuple[ResolvedDagEpisode, ...]


def _successful(source: JudgeCandidateSource) -> ArmSucceeded:
    if isinstance(source.arm_result, ArmSucceeded):
        return source.arm_result
    raise JudgeViewError(
        reason=JudgeViewFailureReason.ARM_NOT_SUCCEEDED,
        failure_class=type(source.arm_result).__name__,
    )


def _arm_rank(arm: ForecastArm) -> int:
    return _ARM_RANK[arm]


def _arm_inputs_match(
    arm: ForecastArm,
    has_evidence: bool,
    has_memory: bool,
) -> bool:
    valid_by_arm = {
        ForecastArm.DIRECT: not has_evidence and not has_memory,
        ForecastArm.SEARCH_ONLY: has_evidence and not has_memory,
        ForecastArm.SEARCH_DAG: has_evidence and has_memory,
    }
    return valid_by_arm[arm]


def _validate_treatment(source: JudgeCandidateSource, result: ArmSucceeded) -> None:
    forecast_input = source.forecast_input
    evidence = forecast_input.evidence_pack.items
    memory = forecast_input.historical_memory
    snapshot = FinanceEvidenceSnapshot(items=evidence)
    actual_digest = snapshot.digest if evidence else None
    actual_bytes = snapshot.byte_length if evidence else 0
    audit = result.treatment
    matches_audit = (
        audit.evidence_snapshot_digest == actual_digest
        and audit.evidence_snapshot_bytes == actual_bytes
        and audit.evidence_item_count == len(evidence)
        and audit.historical_memory_episode_count == len(memory)
    )
    matches_arm = _arm_inputs_match(result.arm, bool(evidence), bool(memory))
    evidence_ids = {item.evidence_id for item in evidence}
    memory_ids = {item.dag_id for item in memory}
    referenced_evidence = {
        item for scenario in result.forecast.scenarios for item in scenario.evidence_ids
    }
    referenced_memory = {
        item.dag_id
        for scenario in result.forecast.scenarios
        for item in scenario.historical_dag_references
    }
    if (
        not matches_audit
        or not matches_arm
        or not referenced_evidence.issubset(evidence_ids)
        or not referenced_memory.issubset(memory_ids)
    ):
        raise JudgeViewError(
            reason=JudgeViewFailureReason.TREATMENT_MISMATCH,
            failure_class="TreatmentAuditMismatch",
        )
    if any(
        item.available_at >= forecast_input.target_profile.cutoff for item in evidence
    ):
        raise JudgeViewError(
            reason=JudgeViewFailureReason.POST_CUTOFF_EVIDENCE,
            failure_class="EvidenceTimingMismatch",
        )


def _canonical_candidates(
    request: JudgeViewRequest,
) -> tuple[CanonicalJudgeCandidate, CanonicalJudgeCandidate]:
    results = tuple(_successful(source) for source in request.candidates)
    if results[0].arm is results[1].arm:
        raise JudgeViewError(
            reason=JudgeViewFailureReason.DUPLICATE_CANDIDATE,
            failure_class="DuplicateForecastArm",
        )
    for source, result in zip(request.candidates, results, strict=True):
        _validate_treatment(source, result)
    if (
        request.candidates[0].forecast_input.target_profile
        != request.candidates[1].forecast_input.target_profile
    ):
        raise JudgeViewError(
            reason=JudgeViewFailureReason.TARGET_MISMATCH,
            failure_class="TargetProfileMismatch",
        )
    ordered = sorted(
        zip(request.candidates, results, strict=True),
        key=lambda item: _arm_rank(item[1].arm),
    )
    return (
        CanonicalJudgeCandidate(
            alias=NeutralCandidate.CANDIDATE_1,
            arm_result=ordered[0][1],
            forecast_input=ordered[0][0].forecast_input,
        ),
        CanonicalJudgeCandidate(
            alias=NeutralCandidate.CANDIDATE_2,
            arm_result=ordered[1][1],
            forecast_input=ordered[1][0].forecast_input,
        ),
    )


def _merged_evidence(
    candidates: tuple[CanonicalJudgeCandidate, CanonicalJudgeCandidate],
) -> tuple[EvidenceItem, ...]:
    by_id: dict[EvidenceId, EvidenceItem] = {}
    for candidate in candidates:
        for item in candidate.forecast_input.evidence_pack.items:
            known = by_id.get(item.evidence_id)
            if known is not None and known != item:
                raise JudgeViewError(
                    reason=JudgeViewFailureReason.CONFLICTING_EVIDENCE,
                    failure_class="EvidenceIdentityConflict",
                )
            by_id[item.evidence_id] = item
    return FinanceEvidenceSnapshot(items=tuple(by_id.values())).items


def _merged_memory(
    candidates: tuple[CanonicalJudgeCandidate, CanonicalJudgeCandidate],
) -> tuple[ResolvedDagEpisode, ...]:
    by_id: dict[DagId, ResolvedDagEpisode] = {}
    ordered: list[ResolvedDagEpisode] = []
    for candidate in candidates:
        for episode in candidate.forecast_input.historical_memory:
            known = by_id.get(episode.dag_id)
            if known is not None and known != episode:
                raise JudgeViewError(
                    reason=JudgeViewFailureReason.CONFLICTING_MEMORY,
                    failure_class="MemoryIdentityConflict",
                )
            if known is None:
                by_id[episode.dag_id] = episode
                ordered.append(episode)
    return tuple(ordered)


def validate_judge_sources(request: JudgeViewRequest) -> ValidatedJudgeSources:
    """Return canonical candidates and candidate-order-independent context."""
    candidates = _canonical_candidates(request)
    return ValidatedJudgeSources(
        candidates=candidates,
        evidence=_merged_evidence(candidates),
        memory=_merged_memory(candidates),
    )


__all__ = ["ValidatedJudgeSources", "validate_judge_sources"]
