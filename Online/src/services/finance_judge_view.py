"""Blind finance judge-view construction from successful arm records."""

import secrets
from dataclasses import dataclass
from typing import Final

from pydantic import ValidationError

from src.domain.finance.judge_source import (
    JudgeCandidateSource,
    JudgeViewError,
    JudgeViewFailureReason,
    JudgeViewRequest,
)
from src.domain.finance.judge_views import (
    BlindJudgePayload,
    SanitizedJudgePair,
)
from src.domain.finance.judging import JudgeCallOrientation
from src.domain.finance.sanitized_artifact import (
    ArtifactSanitizationError,
    ArtifactSanitizationFailureReason,
)
from src.services.finance_aliases import build_run_alias_table
from src.services.finance_artifact_sanitizer import validate_persisted_artifact
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
from src.services.finance_judge_source_validation import validate_judge_sources

_ARTIFACT_FAILURE_MAPPING: Final = {
    ArtifactSanitizationFailureReason.INVALID_ALIAS_SALT: (
        JudgeViewFailureReason.INVALID_ALIAS_SALT
    ),
    ArtifactSanitizationFailureReason.UNKNOWN_IDENTIFIER: (
        JudgeViewFailureReason.INVALID_SOURCE
    ),
    ArtifactSanitizationFailureReason.FORBIDDEN_KEY: (
        JudgeViewFailureReason.FORBIDDEN_CONTENT
    ),
    ArtifactSanitizationFailureReason.FORBIDDEN_VALUE: (
        JudgeViewFailureReason.FORBIDDEN_CONTENT
    ),
    ArtifactSanitizationFailureReason.CREDENTIAL_PATTERN: (
        JudgeViewFailureReason.FORBIDDEN_CONTENT
    ),
    ArtifactSanitizationFailureReason.MALFORMED_JSON: (
        JudgeViewFailureReason.FORBIDDEN_CONTENT
    ),
    ArtifactSanitizationFailureReason.AMBIGUOUS_IDENTIFIER: (
        JudgeViewFailureReason.FORBIDDEN_CONTENT
    ),
}


def _artifact_reason(error: ArtifactSanitizationError) -> JudgeViewFailureReason:
    return _ARTIFACT_FAILURE_MAPPING[error.reason]


@dataclass(frozen=True, slots=True)
class FinanceJudgeViewBuilder:
    """Create one run-local sanitized pair using exactly 32 salt bytes."""

    alias_salt: bytes

    @classmethod
    def for_new_run(cls) -> "FinanceJudgeViewBuilder":
        return cls(alias_salt=secrets.token_bytes(32))

    def build(self, request: JudgeViewRequest) -> SanitizedJudgePair:
        validated = validate_judge_sources(request)
        candidates = validated.candidates
        evidence = validated.evidence
        memory = validated.memory
        alias_inputs = JudgeAliasInputs(
            candidates=candidates,
            evidence=evidence,
            memory=memory,
            transient_identities=request.transient_forbidden.original_identifiers,
        )
        try:
            aliases = build_run_alias_table(
                self.alias_salt,
                collect_judge_alias_sources(alias_inputs),
            )
            pair = SanitizedJudgePair(
                target=build_sanitized_target_view(
                    candidates[0].forecast_input.target_profile,
                    aliases,
                ),
                evidence=build_sanitized_evidence_views(evidence, aliases),
                memory=build_sanitized_memory_views(memory, aliases),
                candidate_1=build_sanitized_candidate_view(candidates[0], aliases),
                candidate_2=build_sanitized_candidate_view(candidates[1], aliases),
                alias_audit=aliases.audit,
            )
            validate_persisted_artifact(
                pair.model_dump_json(),
                aliases.forbidden_registry(request.transient_forbidden),
            )
        except ArtifactSanitizationError as error:
            raise JudgeViewError(
                reason=_artifact_reason(error),
                failure_class=type(error).__name__,
            ) from error
        except ValidationError as error:
            raise JudgeViewError(
                reason=JudgeViewFailureReason.INVALID_SOURCE,
                failure_class=type(error).__name__,
            ) from error
        return pair


def build_blind_judge_payload(
    pair: SanitizedJudgePair,
    orientation: JudgeCallOrientation,
) -> BlindJudgePayload:
    """Project a canonical pair into one of the exact two answer orders."""
    ordered_by_orientation = {
        JudgeCallOrientation.CANONICAL: (pair.candidate_1, pair.candidate_2),
        JudgeCallOrientation.SWAPPED: (pair.candidate_2, pair.candidate_1),
    }
    ordered = ordered_by_orientation[orientation]
    payload = BlindJudgePayload(
        target=pair.target,
        evidence=pair.evidence,
        memory=pair.memory,
        answer_a=ordered[0],
        answer_b=ordered[1],
    )
    validate_persisted_artifact(payload.model_dump_json())
    return payload


__all__ = [
    "FinanceJudgeViewBuilder",
    "JudgeCandidateSource",
    "JudgeViewError",
    "JudgeViewFailureReason",
    "JudgeViewRequest",
    "build_blind_judge_payload",
]
