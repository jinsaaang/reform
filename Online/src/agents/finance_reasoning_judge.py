"""Exact two-order finance reasoning judge without raw-output persistence."""

import json
from dataclasses import dataclass
from typing import Final, Protocol

from pydantic import ValidationError
from typing_extensions import override

from src.domain.finance.experiment_manifest import Seed
from src.domain.finance.experiment_telemetry import (
    ProviderAttemptStatus,
    ProviderAttemptTelemetry,
)
from src.domain.finance.judge_views import (
    BlindJudgePayload,
    NeutralCandidate,
    SanitizedJudgePair,
)
from src.domain.finance.judging import (
    JudgeCallFailureClass,
    JudgeCallFailureReason,
    JudgeCallInvalid,
    JudgeCallOrientation,
    JudgeCallRecord,
    JudgeCallSucceeded,
    JudgePreference,
    JudgeProviderResponse,
    MappedJudgePreference,
    SingleJudgeInconsistent,
    SingleJudgeInvalid,
    SingleJudgeResult,
    SingleJudgeTie,
    SingleJudgeWinner,
)
from src.services.finance_judge_view import build_blind_judge_payload

_PREFERENCE_MAPPING: Final = {
    (JudgeCallOrientation.CANONICAL, JudgePreference.ANSWER_A): (
        MappedJudgePreference.CANDIDATE_1
    ),
    (JudgeCallOrientation.CANONICAL, JudgePreference.ANSWER_B): (
        MappedJudgePreference.CANDIDATE_2
    ),
    (JudgeCallOrientation.CANONICAL, JudgePreference.TIE): (MappedJudgePreference.TIE),
    (JudgeCallOrientation.SWAPPED, JudgePreference.ANSWER_A): (
        MappedJudgePreference.CANDIDATE_2
    ),
    (JudgeCallOrientation.SWAPPED, JudgePreference.ANSWER_B): (
        MappedJudgePreference.CANDIDATE_1
    ),
    (JudgeCallOrientation.SWAPPED, JudgePreference.TIE): (MappedJudgePreference.TIE),
}
_CANDIDATE_ORDER: Final = {
    JudgeCallOrientation.CANONICAL: (
        NeutralCandidate.CANDIDATE_1,
        NeutralCandidate.CANDIDATE_2,
    ),
    JudgeCallOrientation.SWAPPED: (
        NeutralCandidate.CANDIDATE_2,
        NeutralCandidate.CANDIDATE_1,
    ),
}
_OTHER_ORIENTATION: Final = {
    JudgeCallOrientation.CANONICAL: JudgeCallOrientation.SWAPPED,
    JudgeCallOrientation.SWAPPED: JudgeCallOrientation.CANONICAL,
}
_WINNER_MAPPING: Final = {
    MappedJudgePreference.CANDIDATE_1: NeutralCandidate.CANDIDATE_1,
    MappedJudgePreference.CANDIDATE_2: NeutralCandidate.CANDIDATE_2,
    MappedJudgePreference.TIE: None,
}


@dataclass(frozen=True, slots=True)
class TransientJudgeProviderCompletion:
    """In-memory raw completion paired with safe attempt telemetry."""

    serialized_response: str
    attempt: ProviderAttemptTelemetry


class JudgeProvider(Protocol):
    """Only capability available to the blind reasoning judge."""

    def judge(
        self,
        payload: BlindJudgePayload,
        requested_seed: Seed,
    ) -> TransientJudgeProviderCompletion:
        """Return exactly one ordered structured completion."""
        ...


@dataclass(frozen=True, slots=True)
class JudgeProviderError(Exception):
    """Expected adapter failure containing no provider error text."""

    attempt: ProviderAttemptTelemetry

    @override
    def __str__(self) -> str:
        return "judge provider failed"


@dataclass(frozen=True, slots=True)
class ReasoningJudgeRequest:
    pair: SanitizedJudgePair
    member_id: str
    requested_seed: Seed


def _invalid_attempt(
    attempt: ProviderAttemptTelemetry,
    failure_class: JudgeCallFailureClass,
) -> ProviderAttemptTelemetry:
    return attempt.model_copy(
        update={
            "status": ProviderAttemptStatus.INVALID,
            "failure_class": failure_class.value,
        }
    )


def _mapped_preference(
    preference: JudgePreference,
    orientation: JudgeCallOrientation,
) -> MappedJudgePreference:
    return _PREFERENCE_MAPPING[(orientation, preference)]


def _candidate_order(
    orientation: JudgeCallOrientation,
) -> tuple[NeutralCandidate, NeutralCandidate]:
    return _CANDIDATE_ORDER[orientation]


@dataclass(frozen=True, slots=True)
class FinanceReasoningJudge:
    """Attempt canonical and swapped calls and apply the exact mapping table."""

    provider: JudgeProvider

    def _call(
        self,
        request: ReasoningJudgeRequest,
        orientation: JudgeCallOrientation,
    ) -> JudgeCallRecord:
        answer_a, answer_b = _candidate_order(orientation)
        payload = build_blind_judge_payload(request.pair, orientation)
        try:
            completion = self.provider.judge(payload, request.requested_seed)
        except JudgeProviderError as error:
            failure_class = JudgeCallFailureClass.JUDGE_PROVIDER_ERROR
            return JudgeCallInvalid(
                orientation=orientation,
                requested_seed=request.requested_seed,
                answer_a_candidate=answer_a,
                answer_b_candidate=answer_b,
                attempt=error.attempt.model_copy(
                    update={"failure_class": failure_class.value}
                ),
                reason=JudgeCallFailureReason.PROVIDER_ERROR,
                failure_class=failure_class,
            )
        try:
            json.loads(completion.serialized_response)
        except json.JSONDecodeError:
            failure_class = JudgeCallFailureClass.JSON_DECODE_ERROR
            return JudgeCallInvalid(
                orientation=orientation,
                requested_seed=request.requested_seed,
                answer_a_candidate=answer_a,
                answer_b_candidate=answer_b,
                attempt=_invalid_attempt(completion.attempt, failure_class),
                reason=JudgeCallFailureReason.MALFORMED_JSON,
                failure_class=failure_class,
            )
        try:
            response = JudgeProviderResponse.model_validate_json(
                completion.serialized_response
            )
        except ValidationError:
            failure_class = JudgeCallFailureClass.VALIDATION_ERROR
            return JudgeCallInvalid(
                orientation=orientation,
                requested_seed=request.requested_seed,
                answer_a_candidate=answer_a,
                answer_b_candidate=answer_b,
                attempt=_invalid_attempt(completion.attempt, failure_class),
                reason=JudgeCallFailureReason.SCHEMA_ERROR,
                failure_class=failure_class,
            )
        return JudgeCallSucceeded(
            orientation=orientation,
            requested_seed=request.requested_seed,
            answer_a_candidate=answer_a,
            answer_b_candidate=answer_b,
            attempt=completion.attempt,
            mapped_preference=_mapped_preference(response.preference, orientation),
            diagnostics=response.diagnostics,
        )

    def judge(self, request: ReasoningJudgeRequest) -> SingleJudgeResult:
        """Always issue both calls, then map them without coercion or retry."""
        return self.judge_in_order(request, JudgeCallOrientation.CANONICAL)

    def judge_in_order(
        self,
        request: ReasoningJudgeRequest,
        first_orientation: JudgeCallOrientation,
    ) -> SingleJudgeResult:
        """Execute the requested order while retaining canonical storage order."""
        executed = tuple(
            self._call(request, orientation)
            for orientation in (
                first_orientation,
                _OTHER_ORIENTATION[first_orientation],
            )
        )
        by_orientation = {call.orientation: call for call in executed}
        calls = (
            by_orientation[JudgeCallOrientation.CANONICAL],
            by_orientation[JudgeCallOrientation.SWAPPED],
        )
        if any(isinstance(call, JudgeCallInvalid) for call in calls):
            return SingleJudgeInvalid(member_id=request.member_id, calls=calls)
        first, second = calls
        if not isinstance(first, JudgeCallSucceeded) or not isinstance(
            second,
            JudgeCallSucceeded,
        ):
            raise AssertionError("exhaustive judge-call narrowing failed")
        preferences = (first.mapped_preference, second.mapped_preference)
        if preferences == (MappedJudgePreference.TIE, MappedJudgePreference.TIE):
            return SingleJudgeTie(member_id=request.member_id, calls=calls)
        if first.mapped_preference is second.mapped_preference:
            winner = _WINNER_MAPPING[first.mapped_preference]
            if winner is None:
                return SingleJudgeTie(member_id=request.member_id, calls=calls)
            return SingleJudgeWinner(
                member_id=request.member_id,
                calls=calls,
                winner=winner,
            )
        return SingleJudgeInconsistent(member_id=request.member_id, calls=calls)
