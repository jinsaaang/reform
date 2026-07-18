"""TDD contract gates for blind finance judging."""

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from importlib.util import find_spec

import pytest
from pydantic import ValidationError

from src.domain.finance.experiment_telemetry import ProviderAttemptStatus
from src.domain.finance.judge_views import NeutralCandidate
from src.domain.finance.judging import (
    JudgeCallFailureClass,
    JudgeCallFailureReason,
    JudgeCallInvalid,
    JudgeCallOrientation,
    JudgeCallSucceeded,
    JudgeDiagnostic,
    JudgeDiagnosticAssessment,
    JudgeDiagnosticDimension,
    JudgePreference,
    JudgeProviderResponse,
    MappedJudgePreference,
    SingleJudgeTie,
    ThreeMemberJudgePanelResult,
)
from src.domain.finance.memory import (
    ArticleId,
    CausalEdgeId,
    EventId,
    HistoricalCausalEdge,
    HistoricalImpact,
    ImpactId,
    QuestionId,
    QuestionKind,
    ResolvedDagEpisode,
)
from src.domain.finance.search import (
    EvidenceDirection,
    EvidenceId,
    EvidenceItem,
    TargetProfile,
)
from tests.unit.domain.finance._experiment_factories import make_attempt
from tests.unit.domain.finance._factories import make_episode

EVIDENCE_SENTINEL = "ORIGINAL_EVIDENCE_SENTINEL_7f9c"
DAG_SENTINEL = "ORIGINAL_DAG_SENTINEL_4a2e"


def make_judge_target(identifier: str = "current-target") -> TargetProfile:
    return TargetProfile(
        question_id=QuestionId(identifier),
        question_text="Will the public benchmark cross its threshold?",
        question_type=QuestionKind.BINARY,
        domain="finance",
        context=("Use only information available before the cutoff.",),
        cutoff=datetime(2026, 7, 18, tzinfo=UTC),
        outcome_space=("No", "Yes"),
        resolution_rule="Resolve from the official benchmark.",
    )


def make_judge_evidence(content_hash: str = "a" * 64) -> EvidenceItem:
    cutoff = make_judge_target().cutoff
    return EvidenceItem(
        evidence_id=EvidenceId(EVIDENCE_SENTINEL),
        claim=f"Claim cites {EVIDENCE_SENTINEL} and {DAG_SENTINEL}.",
        citation=f"https://example.test/{EVIDENCE_SENTINEL}",
        available_at=cutoff - timedelta(hours=2),
        retrieved_at=cutoff + timedelta(hours=1),
        content_hash=content_hash,
        direction=EvidenceDirection.SUPPORTS,
        context_slot="current_state",
    )


def make_bounded_judge_episode() -> ResolvedDagEpisode:
    base = make_episode(DAG_SENTINEL)
    nodes = tuple(
        replace(
            base.nodes[0],
            event_id=EventId(f"{DAG_SENTINEL}-node-{index:02d}"),
            title=f"Node {index} from {DAG_SENTINEL}",
            source_article_ids=(ArticleId(f"article-{index:02d}"),),
            is_outcome=index == 1,
        )
        for index in range(14)
    )
    edges = tuple(
        HistoricalCausalEdge(
            edge_id=CausalEdgeId(f"{DAG_SENTINEL}-edge-{index:02d}"),
            source_event_id=nodes[0].event_id,
            target_event_id=nodes[13 if index == 0 else 1].event_id,
            relation_type="influences",
            strength=0.7,
            confidence=0.8,
            time_lag_hours=None,
            reasoning=f"Edge {index} cites {DAG_SENTINEL}",
            evidence_article_ids=(),
        )
        for index in range(18)
    )
    impacts = tuple(
        HistoricalImpact(
            impact_id=ImpactId(f"{DAG_SENTINEL}-impact-{index:02d}"),
            event_id=nodes[0].event_id,
            outcome_event_id=nodes[1].event_id,
            direction="positive",
            magnitude=0.5,
            confidence=0.7,
            reasoning=f"Impact {index} cites {DAG_SENTINEL}",
            evidence_article_ids=(),
            causal_edge_ids=(edges[0 if index == 0 else 1].edge_id,),
        )
        for index in range(10)
    )
    return replace(
        base,
        source_profile=replace(
            base.source_profile,
            question_text="Historical rate question",
        ),
        historical_outcome=replace(
            base.historical_outcome,
            value=f"Resolved Yes under {DAG_SENTINEL}",
        ),
        nodes=nodes,
        edges=edges,
        impacts=impacts,
    )


def test_judging_contract_exists() -> None:
    # Given / When
    specification = find_spec("src.domain.finance.judging")

    # Then
    assert specification is not None


def test_judging_domain_exposes_closed_contracts() -> None:
    # Given
    expected_dimensions = {
        "finance_mechanism_plausibility",
        "evidence_grounding",
        "scenario_coherence",
        "counterevidence",
        "assumptions_triggers_disconfirmers",
        "uncertainty_qualification",
        "analog_use_limitations",
    }

    # When
    actual_dimensions = {dimension.value for dimension in JudgeDiagnosticDimension}

    # Then
    assert actual_dimensions == expected_dimensions


def test_sanitized_view_and_audit_contracts_exist() -> None:
    # Given / When
    specifications = (
        find_spec("src.domain.finance.judge_views"),
        find_spec("src.domain.finance.sanitized_artifact"),
    )

    # Then
    assert all(specification is not None for specification in specifications)


def _diagnostics() -> tuple[JudgeDiagnostic, ...]:
    return tuple(
        JudgeDiagnostic(
            dimension=dimension,
            assessment=JudgeDiagnosticAssessment.EQUAL,
        )
        for dimension in JudgeDiagnosticDimension
    )


def test_provider_response_requires_all_seven_diagnostics() -> None:
    # Given
    diagnostics = _diagnostics()

    # When
    response = JudgeProviderResponse(
        preference=JudgePreference.TIE,
        diagnostics=diagnostics,
    )

    # Then
    assert response.diagnostics == diagnostics


def test_provider_response_rejects_missing_diagnostic() -> None:
    # Given / When / Then
    with pytest.raises(ValidationError):
        JudgeProviderResponse(
            preference=JudgePreference.TIE,
            diagnostics=_diagnostics()[:-1],
        )


@pytest.mark.parametrize(
    "diagnostics",
    (
        (),
        _diagnostics()[:-1],
        (_diagnostics()[0],) * 7,
    ),
)
def test_success_call_rejects_noncanonical_diagnostics(
    diagnostics: tuple[JudgeDiagnostic, ...],
) -> None:
    # Given / When / Then
    with pytest.raises(ValidationError):
        JudgeCallSucceeded(
            orientation=JudgeCallOrientation.CANONICAL,
            requested_seed=42,
            answer_a_candidate=NeutralCandidate.CANDIDATE_1,
            answer_b_candidate=NeutralCandidate.CANDIDATE_2,
            attempt=make_attempt(),
            mapped_preference=MappedJudgePreference.TIE,
            diagnostics=diagnostics,
        )


def test_invalid_call_rejects_free_form_failure_class() -> None:
    # Given
    raw_error = "TimeoutError: RAW_PROVIDER_ERROR_TEXT_7c2a"
    failed_attempt = make_attempt().model_copy(
        update={
            "status": ProviderAttemptStatus.FAILED,
            "output_bytes": None,
            "failure_class": raw_error,
        }
    )
    valid = JudgeCallInvalid(
        orientation=JudgeCallOrientation.CANONICAL,
        requested_seed=42,
        answer_a_candidate=NeutralCandidate.CANDIDATE_1,
        answer_b_candidate=NeutralCandidate.CANDIDATE_2,
        attempt=failed_attempt.model_copy(
            update={"failure_class": JudgeCallFailureClass.JUDGE_PROVIDER_ERROR.value}
        ),
        reason=JudgeCallFailureReason.PROVIDER_ERROR,
        failure_class=JudgeCallFailureClass.JUDGE_PROVIDER_ERROR,
    )
    tampered = valid.model_dump_json().replace(
        JudgeCallFailureClass.JUDGE_PROVIDER_ERROR.value,
        raw_error,
    )

    # When / Then
    with pytest.raises(ValidationError):
        JudgeCallInvalid.model_validate_json(tampered)


def _tie(member_id: str) -> SingleJudgeTie:
    calls = (
        JudgeCallSucceeded(
            orientation=JudgeCallOrientation.CANONICAL,
            requested_seed=42,
            answer_a_candidate=NeutralCandidate.CANDIDATE_1,
            answer_b_candidate=NeutralCandidate.CANDIDATE_2,
            attempt=make_attempt(),
            mapped_preference=MappedJudgePreference.TIE,
            diagnostics=_diagnostics(),
        ),
        JudgeCallSucceeded(
            orientation=JudgeCallOrientation.SWAPPED,
            requested_seed=42,
            answer_a_candidate=NeutralCandidate.CANDIDATE_2,
            answer_b_candidate=NeutralCandidate.CANDIDATE_1,
            attempt=make_attempt(),
            mapped_preference=MappedJudgePreference.TIE,
            diagnostics=_diagnostics(),
        ),
    )
    return SingleJudgeTie(member_id=member_id, calls=calls)


def test_panel_requires_three_unique_members() -> None:
    # Given / When / Then
    with pytest.raises(ValidationError):
        ThreeMemberJudgePanelResult(
            members=(_tie("judge-1"), _tie("judge-2"), _tie("judge-2")),
        )
