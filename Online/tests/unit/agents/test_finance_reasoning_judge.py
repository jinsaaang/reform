"""Two-order finance reasoning judge tests."""

import base64
from datetime import UTC, datetime
from typing import TypeAlias

import pytest

from src.agents.finance_reasoning_judge import (
    FinanceReasoningJudge,
    JudgeProviderError,
    ReasoningJudgeRequest,
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
    JudgeCallFailureClass,
    JudgeCallInvalid,
    JudgeDiagnostic,
    JudgeDiagnosticAssessment,
    JudgeDiagnosticDimension,
    JudgePreference,
    JudgeProviderResponse,
    SingleJudgeInvalid,
    SingleJudgeWinner,
)
from src.domain.finance.memory import OutcomeLabel, QuestionKind
from src.domain.finance.sanitized_artifact import FinanceAliasAudit

ScriptItem: TypeAlias = str | JudgeProviderError


def _attempt(
    status: ProviderAttemptStatus = ProviderAttemptStatus.SUCCEEDED,
    failure_class: str | None = None,
) -> ProviderAttemptTelemetry:
    return ProviderAttemptTelemetry(
        attempt_index=0,
        provider_name="offline-judge",
        status=status,
        latency_ms=1,
        input_bytes=10,
        output_bytes=20 if status is ProviderAttemptStatus.SUCCEEDED else None,
        seed=ProviderSeedEffective(requested_seed=73),
        usage=ProviderUsageUnavailable(),
        failure_class=failure_class,
    )


class ScriptedJudgeProvider:
    """Mutable only to capture the exact sequential calls under test."""

    def __init__(self, script: tuple[ScriptItem, ScriptItem]) -> None:
        self._script = script
        self.payloads: list[BlindJudgePayload] = []
        self.seeds: list[Seed] = []

    def judge(
        self,
        payload: BlindJudgePayload,
        requested_seed: Seed,
    ) -> TransientJudgeProviderCompletion:
        self.payloads.append(payload)
        self.seeds.append(requested_seed)
        item = self._script[len(self.payloads) - 1]
        if isinstance(item, JudgeProviderError):
            raise item
        return TransientJudgeProviderCompletion(
            serialized_response=item,
            attempt=_attempt(),
        )


def _diagnostics() -> tuple[JudgeDiagnostic, ...]:
    return tuple(
        JudgeDiagnostic(
            dimension=dimension,
            assessment=JudgeDiagnosticAssessment.EQUAL,
        )
        for dimension in JudgeDiagnosticDimension
    )


def _response(preference: JudgePreference) -> str:
    return JudgeProviderResponse(
        preference=preference,
        diagnostics=_diagnostics(),
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


def _pair() -> SanitizedJudgePair:
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


@pytest.mark.parametrize(
    ("first", "second", "expected_status", "expected_winner"),
    (
        (JudgePreference.ANSWER_A, JudgePreference.ANSWER_B, "valid", "candidate_1"),
        (JudgePreference.ANSWER_B, JudgePreference.ANSWER_A, "valid", "candidate_2"),
        (JudgePreference.TIE, JudgePreference.TIE, "tie", None),
        (JudgePreference.ANSWER_A, JudgePreference.ANSWER_A, "inconsistent", None),
        (JudgePreference.ANSWER_B, JudgePreference.ANSWER_B, "inconsistent", None),
        (JudgePreference.ANSWER_A, JudgePreference.TIE, "inconsistent", None),
        (JudgePreference.TIE, JudgePreference.ANSWER_A, "inconsistent", None),
        (JudgePreference.ANSWER_B, JudgePreference.TIE, "inconsistent", None),
        (JudgePreference.TIE, JudgePreference.ANSWER_B, "inconsistent", None),
    ),
)
def test_exact_two_order_mapping_table(
    first: JudgePreference,
    second: JudgePreference,
    expected_status: str,
    expected_winner: str | None,
) -> None:
    # Given
    provider = ScriptedJudgeProvider((_response(first), _response(second)))
    request = ReasoningJudgeRequest(_pair(), "judge-1", 73)

    # When
    result = FinanceReasoningJudge(provider).judge(request)

    # Then
    assert result.status == expected_status
    assert len(provider.payloads) == 2
    assert provider.seeds == [73, 73]
    assert provider.payloads[0].answer_a.alias is NeutralCandidate.CANDIDATE_1
    assert provider.payloads[1].answer_a.alias is NeutralCandidate.CANDIDATE_2
    if isinstance(result, SingleJudgeWinner):
        assert result.winner.value == expected_winner
    else:
        assert expected_winner is None


def test_second_order_is_attempted_after_first_provider_failure() -> None:
    # Given
    raw_error = "TimeoutError: RAW_PROVIDER_ERROR_TEXT_7c2a"
    failure = JudgeProviderError(
        attempt=_attempt(ProviderAttemptStatus.FAILED, raw_error),
    )
    provider = ScriptedJudgeProvider((failure, _response(JudgePreference.TIE)))

    # When
    result = FinanceReasoningJudge(provider).judge(
        ReasoningJudgeRequest(_pair(), "judge-1", 73)
    )

    # Then
    assert isinstance(result, SingleJudgeInvalid)
    assert len(provider.payloads) == 2
    serialized = result.model_dump_json()
    assert raw_error not in serialized
    first_call = result.calls[0]
    assert isinstance(first_call, JudgeCallInvalid)
    assert first_call.failure_class is JudgeCallFailureClass.JUDGE_PROVIDER_ERROR
    assert first_call.attempt.failure_class == "JudgeProviderError"


@pytest.mark.parametrize("malformed", ("not-json", "missing-diagnostic"))
def test_malformed_json_and_missing_diagnostic_are_invalid(malformed: str) -> None:
    # Given
    valid = JudgeProviderResponse(
        preference=JudgePreference.TIE,
        diagnostics=_diagnostics(),
    )
    invalid_schema = valid.model_copy(update={"diagnostics": _diagnostics()[:-1]})
    first = "{" if malformed == "not-json" else invalid_schema.model_dump_json()
    provider = ScriptedJudgeProvider((first, _response(JudgePreference.TIE)))

    # When
    result = FinanceReasoningJudge(provider).judge(
        ReasoningJudgeRequest(_pair(), "judge-1", 73)
    )

    # Then
    assert isinstance(result, SingleJudgeInvalid)
    assert len(provider.payloads) == 2
