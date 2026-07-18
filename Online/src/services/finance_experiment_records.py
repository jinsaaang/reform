"""Persisted arm projections for finance experiment batches."""

from dataclasses import dataclass
from decimal import Decimal
from typing import assert_never

from src.domain.finance.experiment_artifact import (
    PersistedArmFailed,
    PersistedArmResult,
    PersistedArmSucceeded,
    PersistedArmUnavailable,
    SanitizedArmReasoning,
)
from src.domain.finance.experiment_results import (
    ArmFailed,
    ArmResult,
    ArmSucceeded,
    ArmUnavailable,
)
from src.domain.finance.judge_source import CanonicalJudgeCandidate
from src.domain.finance.judge_views import NeutralCandidate
from src.domain.finance.memory import OutcomeLabel
from src.domain.finance.experiment_variant_guard import (
    runtime_closed_variant,
    unexpected_closed_variant,
)
from src.services.finance_aliases import build_run_alias_table
from src.services.finance_artifact_sanitizer import validate_persisted_artifact
from src.services.finance_experiment_recording import ForecastObservation
from src.services.finance_judge_alias_sources import (
    JudgeAliasInputs,
    collect_judge_alias_sources,
)
from src.services.finance_judge_candidate_view import (
    build_sanitized_candidate_view,
    build_sanitized_evidence_views,
    build_sanitized_target_view,
)
from src.services.finance_judge_memory_view import build_sanitized_memory_views


@dataclass(frozen=True, slots=True)
class FinanceExperimentRecordError(Exception):
    """A successful arm lacks its exact transient provider input."""

    detail: str

    def __str__(self) -> str:
        return f"finance experiment record construction failed: {self.detail}"


@dataclass(frozen=True, slots=True)
class ArmPersistenceRequest:
    result: ArmResult
    positive_label: OutcomeLabel
    observation: ForecastObservation | None
    alias_salt: bytes


def _reasoning(
    result: ArmSucceeded,
    observation: ForecastObservation,
    alias_salt: bytes,
) -> SanitizedArmReasoning:
    forecast_input = observation.forecast_input
    candidate = CanonicalJudgeCandidate(
        alias=NeutralCandidate.CANDIDATE_1,
        arm_result=result,
        forecast_input=forecast_input,
    )
    evidence = forecast_input.evidence_pack.items
    memory = forecast_input.historical_memory
    aliases = build_run_alias_table(
        alias_salt,
        collect_judge_alias_sources(
            JudgeAliasInputs(
                candidates=(candidate, candidate),
                evidence=evidence,
                memory=memory,
                transient_identities=(),
            )
        ),
    )
    reasoning = SanitizedArmReasoning(
        target=build_sanitized_target_view(
            forecast_input.target_profile,
            aliases,
        ),
        evidence=build_sanitized_evidence_views(evidence, aliases),
        memory=build_sanitized_memory_views(memory, aliases),
        candidate=build_sanitized_candidate_view(candidate, aliases),
        alias_audit=aliases.audit,
    )
    validate_persisted_artifact(reasoning.model_dump_json())
    return reasoning


def _positive_probability(
    result: ArmSucceeded,
    positive_label: OutcomeLabel,
) -> Decimal:
    matching = tuple(
        item
        for item in result.forecast.outcome_probabilities
        if item.label == positive_label
    )
    if len(matching) != 1:
        raise FinanceExperimentRecordError("positive label is absent or duplicated")
    return Decimal(str(matching[0].probability))


def build_persisted_arm(
    request: ArmPersistenceRequest,
) -> PersistedArmResult:
    """Project one terminal runtime arm through the shared sanitizer."""
    result = runtime_closed_variant(request.result)
    match result:
        case ArmSucceeded():
            observation = request.observation
            if observation is None or observation.arm is not result.arm:
                raise FinanceExperimentRecordError(
                    "successful arm observation is unavailable"
                )
            return PersistedArmSucceeded(
                arm=result.arm,
                exchanges=(observation.exchange,),
                treatment=result.treatment,
                positive_probability=_positive_probability(
                    result,
                    request.positive_label,
                ),
                reasoning=_reasoning(result, observation, request.alias_salt),
            )
        case ArmUnavailable():
            return PersistedArmUnavailable(
                arm=result.arm,
                exchanges=(),
                treatment=result.treatment,
                reason=result.reason,
            )
        case ArmFailed():
            exchanges = (
                () if request.observation is None else (request.observation.exchange,)
            )
            return PersistedArmFailed(
                arm=result.arm,
                exchanges=exchanges,
                treatment=result.treatment,
                reason=result.reason,
            )
        case _ as unreachable:
            assert_never(unexpected_closed_variant(unreachable))


__all__ = [
    "ArmPersistenceRequest",
    "FinanceExperimentRecordError",
    "build_persisted_arm",
]
