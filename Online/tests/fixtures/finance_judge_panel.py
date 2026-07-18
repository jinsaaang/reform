"""Deterministic blind-judge fixtures for panel and adapter tests."""

import base64
from datetime import UTC, datetime
from enum import StrEnum, unique

from src.agents.finance_reasoning_judge import (
    JudgeProviderError,
    TransientJudgeProviderCompletion,
)
from src.domain.finance.experiment_manifest import Seed
from src.domain.finance.experiment_telemetry import (
    ProviderAttemptStatus,
    ProviderAttemptTelemetry,
    ProviderSeedEffective,
    ProviderUsageUnavailable,
)
from src.domain.finance.forecast import OutcomeProbability
from src.domain.finance.judge_views import (
    BlindJudgePayload,
    NeutralCandidate,
    SanitizedCandidateView,
    SanitizedJudgePair,
    SanitizedScenarioView,
    SanitizedTargetView,
)
from src.domain.finance.judging import (
    JudgeDiagnostic,
    JudgeDiagnosticAssessment,
    JudgeDiagnosticDimension,
    JudgePreference,
    JudgeProviderResponse,
)
from src.domain.finance.memory import OutcomeLabel, QuestionKind
from src.domain.finance.sanitized_artifact import FinanceAliasAudit


@unique
class MemberOutcome(StrEnum):
    CANDIDATE_1 = "candidate_1"
    CANDIDATE_2 = "candidate_2"
    TIE = "tie"
    INCONSISTENT = "inconsistent"
    INVALID = "invalid"


def make_judge_attempt(
    requested_seed: Seed,
    status: ProviderAttemptStatus = ProviderAttemptStatus.SUCCEEDED,
) -> ProviderAttemptTelemetry:
    failure_class = None if status is ProviderAttemptStatus.SUCCEEDED else "OSError"
    return ProviderAttemptTelemetry(
        attempt_index=0,
        provider_name="offline-judge",
        status=status,
        latency_ms=1,
        input_bytes=10,
        output_bytes=20 if status is ProviderAttemptStatus.SUCCEEDED else None,
        seed=ProviderSeedEffective(requested_seed=requested_seed),
        usage=ProviderUsageUnavailable(),
        failure_class=failure_class,
    )


def make_judge_response(preference: JudgePreference) -> str:
    diagnostics = tuple(
        JudgeDiagnostic(
            dimension=dimension,
            assessment=JudgeDiagnosticAssessment.EQUAL,
        )
        for dimension in JudgeDiagnosticDimension
    )
    return JudgeProviderResponse(
        preference=preference,
        diagnostics=diagnostics,
    ).model_dump_json()


def _candidate(alias: NeutralCandidate) -> SanitizedCandidateView:
    outcomes = (
        OutcomeProbability(label=OutcomeLabel("No"), probability=0.4),
        OutcomeProbability(label=OutcomeLabel("Yes"), probability=0.6),
    )
    return SanitizedCandidateView(
        alias=alias,
        scenarios=(
            SanitizedScenarioView(
                alias=f"{alias.value}_scenario_001",
                name="Base case",
                reasoning_steps=("The public indicator moves.",),
                probability=1.0,
                conditional_outcomes=outcomes,
                evidence_aliases=(),
                memory_aliases=(),
                assumptions=(),
                triggers=(),
                disconfirmers=(),
                uncertainty="Residual uncertainty remains.",
            ),
        ),
        outcome_probabilities=outcomes,
        explanation="One complete public reasoning trace.",
    )


def make_judge_pair() -> SanitizedJudgePair:
    return SanitizedJudgePair(
        target=SanitizedTargetView(
            question_text="Will the benchmark cross its threshold?",
            question_type=QuestionKind.BINARY,
            domain="finance",
            context=(),
            cutoff=datetime(2026, 7, 18, tzinfo=UTC),
            outcome_space=("No", "Yes"),
            resolution_rule="Use the official benchmark.",
        ),
        evidence=(),
        memory=(),
        candidate_1=_candidate(NeutralCandidate.CANDIDATE_1),
        candidate_2=_candidate(NeutralCandidate.CANDIDATE_2),
        alias_audit=FinanceAliasAudit(
            salt_base64=base64.b64encode(b"s" * 32).decode("ascii"),
            mappings=(),
        ),
    )


class TerminalJudgeProvider:
    """Mutable fake that records both ordered calls and returns one terminal vote."""

    def __init__(self, outcome: MemberOutcome) -> None:
        self.outcome = outcome
        self.payloads: list[BlindJudgePayload] = []
        self.seeds: list[Seed] = []

    def judge(
        self,
        payload: BlindJudgePayload,
        requested_seed: Seed,
    ) -> TransientJudgeProviderCompletion:
        self.payloads.append(payload)
        self.seeds.append(requested_seed)
        if self.outcome is MemberOutcome.INVALID:
            raise JudgeProviderError(
                attempt=make_judge_attempt(
                    requested_seed,
                    ProviderAttemptStatus.FAILED,
                )
            )
        preference = self._preference(payload.answer_a.alias)
        return TransientJudgeProviderCompletion(
            serialized_response=make_judge_response(preference),
            attempt=make_judge_attempt(requested_seed),
        )

    def _preference(self, answer_a: NeutralCandidate) -> JudgePreference:
        targets = {
            MemberOutcome.CANDIDATE_1: NeutralCandidate.CANDIDATE_1,
            MemberOutcome.CANDIDATE_2: NeutralCandidate.CANDIDATE_2,
        }
        target = targets.get(self.outcome)
        if target is not None:
            return (
                JudgePreference.ANSWER_A
                if answer_a is target
                else JudgePreference.ANSWER_B
            )
        if self.outcome is MemberOutcome.TIE:
            return JudgePreference.TIE
        if self.outcome is MemberOutcome.INCONSISTENT:
            return JudgePreference.ANSWER_A
        raise AssertionError("invalid outcomes fail before preference mapping")


__all__ = [
    "MemberOutcome",
    "TerminalJudgeProvider",
    "make_judge_attempt",
    "make_judge_pair",
    "make_judge_response",
]
