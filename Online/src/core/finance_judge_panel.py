"""Seeded local scheduling for the fixed finance reasoning judge panel."""

import random
from dataclasses import dataclass
from typing import Protocol

from src.agents.finance_reasoning_judge import ReasoningJudgeRequest
from src.domain.finance.experiment_manifest import Seed
from src.domain.finance.judge_views import SanitizedJudgePair
from src.domain.finance.judging import (
    JudgeCallOrientation,
    SingleJudgeResult,
    ThreeMemberJudgePanelResult,
)
from src.services.finance_judge_panel import (
    FinanceJudgePanelResult,
    PanelMemberExecutionAudit,
    aggregate_judge_panel,
)


class OrderedReasoningJudge(Protocol):
    """Two-order judge capability with an explicit first orientation."""

    def judge_in_order(
        self,
        request: ReasoningJudgeRequest,
        first_orientation: JudgeCallOrientation,
    ) -> SingleJudgeResult: ...


@dataclass(frozen=True, slots=True)
class FinanceJudgePanelMember:
    """One unique panel member and its common two-orientation seed."""

    member_id: str
    judge: OrderedReasoningJudge
    requested_seed: Seed


@dataclass(frozen=True, slots=True)
class FinanceJudgePanelConfigError(Exception):
    """The panel request cannot represent exactly three unique members."""

    member_ids: tuple[str, ...]

    def __str__(self) -> str:
        return "finance judge panel requires exactly three unique members"


@dataclass(frozen=True, slots=True)
class FinanceJudgePanelRequest:
    """One sanitized pair plus its locally seeded three-member schedule."""

    pair: SanitizedJudgePair
    scheduling_seed: Seed
    members: tuple[FinanceJudgePanelMember, ...]

    def __post_init__(self) -> None:
        member_ids = tuple(member.member_id for member in self.members)
        if len(member_ids) != 3 or len(set(member_ids)) != 3:
            raise FinanceJudgePanelConfigError(member_ids)


def run_finance_judge_panel(
    request: FinanceJudgePanelRequest,
) -> FinanceJudgePanelResult:
    """Execute six calls in a replayable local member/orientation schedule."""
    rng = random.Random(request.scheduling_seed)
    scheduled = list(request.members)
    rng.shuffle(scheduled)
    orientations = tuple(JudgeCallOrientation)
    results: list[SingleJudgeResult] = []
    audit: list[PanelMemberExecutionAudit] = []
    for execution_index, member in enumerate(scheduled):
        first_orientation = rng.choice(orientations)
        audit.append(
            PanelMemberExecutionAudit(
                member_id=member.member_id,
                execution_index=execution_index,
                first_orientation=first_orientation,
            )
        )
        results.append(
            member.judge.judge_in_order(
                ReasoningJudgeRequest(
                    pair=request.pair,
                    member_id=member.member_id,
                    requested_seed=member.requested_seed,
                ),
                first_orientation,
            )
        )
    return aggregate_judge_panel(
        ThreeMemberJudgePanelResult(members=tuple(results)),
        tuple(audit),
    )


__all__ = [
    "FinanceJudgePanelConfigError",
    "FinanceJudgePanelMember",
    "FinanceJudgePanelRequest",
    "OrderedReasoningJudge",
    "run_finance_judge_panel",
]
