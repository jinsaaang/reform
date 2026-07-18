"""Behavioral tests for deterministic three-member judge aggregation."""

from collections.abc import Sequence

import pytest

from src.agents.finance_reasoning_judge import FinanceReasoningJudge
from src.core.finance_judge_panel import (
    FinanceJudgePanelMember,
    FinanceJudgePanelRequest,
    run_finance_judge_panel,
)
from src.domain.finance.judge_views import NeutralCandidate
from src.domain.finance.judging import (
    JudgeCallOrientation,
    MappedJudgePreference,
)
from src.services.finance_judge_panel import PanelTieReason
from tests.fixtures.finance_judge_panel import (
    MemberOutcome,
    TerminalJudgeProvider,
    make_judge_pair,
)


def _run_panel(
    outcomes: Sequence[MemberOutcome],
    scheduling_seed: int = 1,
):
    providers = tuple(TerminalJudgeProvider(outcome) for outcome in outcomes)
    members = tuple(
        FinanceJudgePanelMember(
            member_id=f"judge-{index}",
            judge=FinanceReasoningJudge(provider),
            requested_seed=100 + index,
        )
        for index, provider in enumerate(providers, start=1)
    )
    result = run_finance_judge_panel(
        FinanceJudgePanelRequest(
            pair=make_judge_pair(),
            scheduling_seed=scheduling_seed,
            members=members,
        )
    )
    return result, providers


@pytest.mark.parametrize(
    ("outcomes", "preference", "tie_reason", "eligible"),
    (
        (
            (MemberOutcome.CANDIDATE_1,) * 2 + (MemberOutcome.CANDIDATE_2,),
            MappedJudgePreference.CANDIDATE_1,
            None,
            True,
        ),
        (
            (MemberOutcome.CANDIDATE_1,) * 2 + (MemberOutcome.INVALID,),
            MappedJudgePreference.CANDIDATE_1,
            None,
            True,
        ),
        (
            (
                MemberOutcome.CANDIDATE_1,
                MemberOutcome.CANDIDATE_2,
                MemberOutcome.TIE,
            ),
            MappedJudgePreference.TIE,
            PanelTieReason.NO_CONSENSUS,
            True,
        ),
        (
            (MemberOutcome.CANDIDATE_1,) + (MemberOutcome.INVALID,) * 2,
            MappedJudgePreference.TIE,
            PanelTieReason.NO_QUORUM,
            False,
        ),
        (
            (MemberOutcome.TIE,) * 3,
            MappedJudgePreference.TIE,
            PanelTieReason.NO_CONSENSUS,
            True,
        ),
        (
            (MemberOutcome.INVALID,) * 3,
            MappedJudgePreference.TIE,
            PanelTieReason.NO_QUORUM,
            False,
        ),
    ),
)
def test_panel_requires_two_matching_decisive_votes(
    outcomes: tuple[MemberOutcome, MemberOutcome, MemberOutcome],
    preference: MappedJudgePreference,
    tie_reason: PanelTieReason | None,
    eligible: bool,
) -> None:
    # Given / When
    result, _ = _run_panel(outcomes)

    # Then
    assert result.overall_preference is preference
    assert result.tie_reason is tie_reason
    assert result.preference_eligible is eligible


@pytest.mark.parametrize(
    (
        "outcomes",
        "invalid_rate",
        "inconsistent_rate",
        "agreement",
        "order_consistency",
    ),
    (
        (
            (MemberOutcome.CANDIDATE_1,) * 2 + (MemberOutcome.CANDIDATE_2,),
            0,
            0,
            2 / 3,
            1,
        ),
        ((MemberOutcome.CANDIDATE_1,) * 2 + (MemberOutcome.INVALID,), 1 / 3, 0, 1, 1),
        (
            (
                MemberOutcome.CANDIDATE_1,
                MemberOutcome.CANDIDATE_2,
                MemberOutcome.TIE,
            ),
            0,
            0,
            1 / 3,
            1,
        ),
        (
            (
                MemberOutcome.CANDIDATE_1,
                MemberOutcome.TIE,
                MemberOutcome.INCONSISTENT,
            ),
            0,
            1 / 3,
            1 / 2,
            2 / 3,
        ),
        ((MemberOutcome.INVALID,) * 3, 1, 0, None, None),
    ),
)
def test_panel_reports_exact_preregistered_rates(
    outcomes: tuple[MemberOutcome, MemberOutcome, MemberOutcome],
    invalid_rate: float,
    inconsistent_rate: float,
    agreement: float | None,
    order_consistency: float | None,
) -> None:
    # Given / When
    result, _ = _run_panel(outcomes)

    # Then
    assert result.invalid_rate == invalid_rate
    assert result.inconsistent_rate == inconsistent_rate
    assert result.agreement == agreement
    assert result.order_consistency == order_consistency


def test_panel_replays_seeded_member_and_orientation_order() -> None:
    # Given
    outcomes = (MemberOutcome.CANDIDATE_1,) * 3

    # When
    first, first_providers = _run_panel(outcomes, scheduling_seed=1)
    replay, _ = _run_panel(outcomes, scheduling_seed=1)

    # Then
    assert first.execution_audit == replay.execution_audit
    assert tuple(item.member_id for item in first.execution_audit) == (
        "judge-2",
        "judge-3",
        "judge-1",
    )
    assert tuple(item.first_orientation for item in first.execution_audit) == (
        JudgeCallOrientation.SWAPPED,
        JudgeCallOrientation.CANONICAL,
        JudgeCallOrientation.SWAPPED,
    )
    assert first.attempted_call_count == 6
    assert [len(provider.payloads) for provider in first_providers] == [2, 2, 2]
    assert [provider.seeds for provider in first_providers] == [
        [101, 101],
        [102, 102],
        [103, 103],
    ]
    for member in first.members.members:
        assert tuple(call.orientation for call in member.calls) == (
            JudgeCallOrientation.CANONICAL,
            JudgeCallOrientation.SWAPPED,
        )


def test_existing_single_judge_path_remains_canonical_first() -> None:
    # Given
    provider = TerminalJudgeProvider(MemberOutcome.CANDIDATE_1)
    judge = FinanceReasoningJudge(provider)
    from src.agents.finance_reasoning_judge import ReasoningJudgeRequest

    # When
    result = judge.judge(ReasoningJudgeRequest(make_judge_pair(), "judge-1", 101))

    # Then
    assert provider.payloads[0].answer_a.alias is NeutralCandidate.CANDIDATE_1
    assert result.calls[0].orientation is JudgeCallOrientation.CANONICAL
