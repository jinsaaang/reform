"""Pairwise panel execution and persisted metric projection."""

from dataclasses import dataclass
from decimal import Decimal
from typing import assert_never

from src.agents.finance_reasoning_judge import FinanceReasoningJudge
from src.core.finance_judge_panel import (
    FinanceJudgePanelMember,
    FinanceJudgePanelRequest,
    run_finance_judge_panel,
)
from src.domain.finance.experiment_artifact import PersistedPairCompleted
from src.domain.finance.experiment_manifest import JudgeMember
from src.domain.finance.experiment_metrics import (
    ExAnteTieReason,
    ForecastPair,
    JudgePanelMetrics,
    PanelPreference,
    compare_positive_probabilities,
)
from src.domain.finance.judge_source import JudgeCandidateSource, JudgeViewRequest
from src.domain.finance.judging import MappedJudgePreference
from src.domain.finance.experiment_variant_guard import (
    runtime_closed_variant,
    unexpected_closed_variant,
)
from src.services.finance_experiment_contracts import (
    FinanceTrialSeedInputs,
    JudgeProviderFactory,
    derive_judge_seed,
    derive_panel_seed,
)
from src.services.finance_experiment_recording import RecordingJudgeProvider
from src.services.finance_judge_panel import PanelTieReason
from src.services.finance_judge_view import FinanceJudgeViewBuilder


@dataclass(frozen=True, slots=True)
class PairPanelRequest:
    pair: ForecastPair
    candidates: tuple[JudgeCandidateSource, JudgeCandidateSource]
    positive_probabilities: tuple[Decimal, Decimal]
    judges: tuple[JudgeMember, ...]
    seed_inputs: FinanceTrialSeedInputs
    provider_factory: JudgeProviderFactory
    alias_salt: bytes
    exchange_prefix: str


def _preference(value: MappedJudgePreference) -> PanelPreference:
    candidate = runtime_closed_variant(value)
    match candidate:
        case MappedJudgePreference.CANDIDATE_1:
            return PanelPreference.FIRST
        case MappedJudgePreference.CANDIDATE_2:
            return PanelPreference.SECOND
        case MappedJudgePreference.TIE:
            return PanelPreference.TIE
        case _ as unreachable:
            assert_never(unexpected_closed_variant(unreachable))


def _tie_reason(value: PanelTieReason | None) -> ExAnteTieReason | None:
    candidate = runtime_closed_variant(value)
    match candidate:
        case None:
            return None
        case PanelTieReason.NO_CONSENSUS:
            return ExAnteTieReason.NO_CONSENSUS
        case PanelTieReason.NO_QUORUM:
            return ExAnteTieReason.NO_QUORUM
        case _ as unreachable:
            assert_never(unexpected_closed_variant(unreachable))


def run_persisted_pair_panel(
    request: PairPanelRequest,
) -> PersistedPairCompleted:
    """Sanitize one pair, execute six calls, and persist exact denominators."""
    pair_view = FinanceJudgeViewBuilder(request.alias_salt).build(
        JudgeViewRequest(request.candidates)
    )
    recorders: dict[str, RecordingJudgeProvider] = {}
    members: list[FinanceJudgePanelMember] = []
    for member in request.judges:
        requested_seed = derive_judge_seed(
            request.seed_inputs,
            request.pair,
            member.member_id,
        )
        settings = member.settings.model_copy(update={"requested_seed": requested_seed})
        recorder = RecordingJudgeProvider(
            request.provider_factory(settings),
            f"{request.exchange_prefix}:{member.member_id}",
        )
        recorders[member.member_id] = recorder
        members.append(
            FinanceJudgePanelMember(
                member_id=member.member_id,
                judge=FinanceReasoningJudge(recorder),
                requested_seed=requested_seed,
            )
        )
    result = run_finance_judge_panel(
        FinanceJudgePanelRequest(
            pair=pair_view,
            scheduling_seed=derive_panel_seed(
                request.seed_inputs,
                request.pair,
            ),
            members=tuple(members),
        )
    )
    metrics = JudgePanelMetrics(
        overall_preference=_preference(result.overall_preference),
        preference_eligible=result.preference_eligible,
        tie_reason=_tie_reason(result.tie_reason),
        first_votes=result.candidate_1_votes,
        second_votes=result.candidate_2_votes,
        tie_votes=result.tie_votes,
        inconsistent_count=result.inconsistent_count,
        invalid_count=result.invalid_count,
        evaluable_count=result.evaluable_count,
        two_parse_valid_count=result.two_parse_valid_count,
        attempted_call_count=result.attempted_call_count,
        invalid_rate=Decimal(result.invalid_count) / Decimal(3),
        inconsistent_rate=Decimal(result.inconsistent_count) / Decimal(3),
        agreement=(
            None
            if result.evaluable_count == 0
            else Decimal(
                max(
                    result.candidate_1_votes,
                    result.candidate_2_votes,
                    result.tie_votes,
                )
            )
            / Decimal(result.evaluable_count)
        ),
        order_consistency=(
            None
            if result.two_parse_valid_count == 0
            else Decimal(result.evaluable_count) / Decimal(result.two_parse_valid_count)
        ),
    )
    exchanges = tuple(
        exchange
        for audit in result.execution_audit
        for exchange in recorders[audit.member_id].exchanges
    )
    first_probability, second_probability = request.positive_probabilities
    return PersistedPairCompleted(
        comparison=compare_positive_probabilities(
            request.pair,
            first_probability,
            second_probability,
        ),
        panel=metrics,
        judge_exchanges=exchanges,
    )


__all__ = ["PairPanelRequest", "run_persisted_pair_panel"]
