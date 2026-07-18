"""Preregistered aggregation for a fixed three-member finance judge panel."""

from enum import StrEnum, unique
from typing import Annotated, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_core import PydanticCustomError

from src.domain.finance.judge_views import NeutralCandidate
from src.domain.finance.judging import (
    JudgeCallOrientation,
    MappedJudgePreference,
    SingleJudgeInconsistent,
    SingleJudgeTie,
    SingleJudgeWinner,
    ThreeMemberJudgePanelResult,
)


class _PanelModel(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=True,
        str_strip_whitespace=True,
    )


@unique
class PanelTieReason(StrEnum):
    NO_CONSENSUS = "no_consensus"
    NO_QUORUM = "no_quorum"


class PanelMemberExecutionAudit(_PanelModel):
    """Actual seeded schedule for one member's two ordered calls."""

    member_id: str = Field(min_length=1)
    execution_index: Annotated[int, Field(ge=0, le=2)]
    first_orientation: JudgeCallOrientation


class FinanceJudgePanelResult(_PanelModel):
    """Terminal aggregation with explicit denominators and schedule audit."""

    schema_version: Literal["finance-judge-panel-result/v1"] = (
        "finance-judge-panel-result/v1"
    )
    overall_preference: MappedJudgePreference
    tie_reason: PanelTieReason | None
    preference_eligible: bool
    members: ThreeMemberJudgePanelResult
    execution_audit: tuple[PanelMemberExecutionAudit, ...] = Field(
        min_length=3,
        max_length=3,
    )
    candidate_1_votes: Annotated[int, Field(ge=0, le=3)]
    candidate_2_votes: Annotated[int, Field(ge=0, le=3)]
    tie_votes: Annotated[int, Field(ge=0, le=3)]
    inconsistent_count: Annotated[int, Field(ge=0, le=3)]
    invalid_count: Annotated[int, Field(ge=0, le=3)]
    evaluable_count: Annotated[int, Field(ge=0, le=3)]
    two_parse_valid_count: Annotated[int, Field(ge=0, le=3)]
    attempted_call_count: Literal[6]
    invalid_rate: Annotated[float, Field(ge=0.0, le=1.0)]
    inconsistent_rate: Annotated[float, Field(ge=0.0, le=1.0)]
    agreement: Annotated[float, Field(ge=0.0, le=1.0)] | None
    order_consistency: Annotated[float, Field(ge=0.0, le=1.0)] | None

    @model_validator(mode="after")
    def validate_aggregation(self) -> "FinanceJudgePanelResult":
        """Reject count, denominator, outcome, or schedule drift."""
        terminal_total = (
            self.candidate_1_votes
            + self.candidate_2_votes
            + self.tie_votes
            + self.inconsistent_count
            + self.invalid_count
        )
        if terminal_total != 3:
            raise PydanticCustomError(
                "panel_count_mismatch",
                "panel terminal member counts must total three",
            )
        if self.evaluable_count != (
            self.candidate_1_votes + self.candidate_2_votes + self.tie_votes
        ) or self.two_parse_valid_count != (
            self.evaluable_count + self.inconsistent_count
        ):
            raise PydanticCustomError(
                "panel_denominator_mismatch",
                "panel denominators must follow the preregistered formulas",
            )
        audited_ids = tuple(item.member_id for item in self.execution_audit)
        member_ids = tuple(item.member_id for item in self.members.members)
        if audited_ids != member_ids or tuple(
            item.execution_index for item in self.execution_audit
        ) != (0, 1, 2):
            raise PydanticCustomError(
                "panel_schedule_mismatch",
                "execution audit must follow the actual terminal member order",
            )
        is_tie = self.overall_preference is MappedJudgePreference.TIE
        if is_tie != (self.tie_reason is not None):
            raise PydanticCustomError(
                "panel_tie_reason_mismatch",
                "only an overall tie carries one explicit tie reason",
            )
        has_quorum = self.tie_reason is not PanelTieReason.NO_QUORUM
        if self.preference_eligible is not has_quorum:
            raise PydanticCustomError(
                "panel_preference_eligibility_mismatch",
                "no-quorum panels are excluded from preference denominators",
            )
        return self


def aggregate_judge_panel(
    members: ThreeMemberJudgePanelResult,
    execution_audit: tuple[PanelMemberExecutionAudit, ...],
) -> FinanceJudgePanelResult:
    """Apply the locked two-decisive-vote rule and exact rate formulas."""
    candidate_1_votes = 0
    candidate_2_votes = 0
    tie_votes = 0
    inconsistent_count = 0
    invalid_count = 0
    for member in members.members:
        if isinstance(member, SingleJudgeWinner):
            if member.winner is NeutralCandidate.CANDIDATE_1:
                candidate_1_votes += 1
            else:
                candidate_2_votes += 1
            continue
        if isinstance(member, SingleJudgeTie):
            tie_votes += 1
            continue
        if isinstance(member, SingleJudgeInconsistent):
            inconsistent_count += 1
            continue
        invalid_count += 1
    evaluable_count = candidate_1_votes + candidate_2_votes + tie_votes
    two_parse_valid_count = evaluable_count + inconsistent_count
    if candidate_1_votes >= 2:
        preference = MappedJudgePreference.CANDIDATE_1
        tie_reason = None
    elif candidate_2_votes >= 2:
        preference = MappedJudgePreference.CANDIDATE_2
        tie_reason = None
    else:
        preference = MappedJudgePreference.TIE
        tie_reason = (
            PanelTieReason.NO_CONSENSUS
            if evaluable_count >= 2
            else PanelTieReason.NO_QUORUM
        )
    agreement = (
        max(candidate_1_votes, candidate_2_votes, tie_votes) / evaluable_count
        if evaluable_count
        else None
    )
    order_consistency = (
        evaluable_count / two_parse_valid_count if two_parse_valid_count else None
    )
    return FinanceJudgePanelResult(
        overall_preference=preference,
        tie_reason=tie_reason,
        preference_eligible=tie_reason is not PanelTieReason.NO_QUORUM,
        members=members,
        execution_audit=execution_audit,
        candidate_1_votes=candidate_1_votes,
        candidate_2_votes=candidate_2_votes,
        tie_votes=tie_votes,
        inconsistent_count=inconsistent_count,
        invalid_count=invalid_count,
        evaluable_count=evaluable_count,
        two_parse_valid_count=two_parse_valid_count,
        attempted_call_count=6,
        invalid_rate=invalid_count / 3,
        inconsistent_rate=inconsistent_count / 3,
        agreement=agreement,
        order_consistency=order_consistency,
    )


__all__ = [
    "FinanceJudgePanelResult",
    "PanelMemberExecutionAudit",
    "PanelTieReason",
    "aggregate_judge_panel",
]
