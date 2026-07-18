"""Deterministic sanitized Todo 5 experiment artifact fixture."""

from base64 import b64encode
from datetime import UTC, datetime
from decimal import Decimal
from uuid import UUID

from src.domain.finance.experiment import (
    FinanceSuiteStatus,
    ForecastArm,
    TreatmentAudit,
)
from src.domain.finance.experiment_artifact import (
    PersistedArmSucceeded,
    PersistedExperimentTrial,
    PersistedFinanceExperimentSuite,
    PersistedPairCompleted,
    ProviderExchangeDigest,
    ProviderExchangeKind,
    SanitizedArmReasoning,
)
from src.domain.finance.experiment_metrics import (
    ForecastPair,
    JudgePanelMetrics,
    PanelPreference,
    compare_positive_probabilities,
)
from src.domain.finance.forecast import OutcomeProbability
from src.domain.finance.judge_views import (
    NeutralCandidate,
    SanitizedCandidateView,
    SanitizedScenarioView,
    SanitizedTargetView,
)
from src.domain.finance.memory import OutcomeLabel
from src.domain.finance.sanitized_artifact import FinanceAliasAudit
from tests.fixtures.finance_experiment_sanitized_sources import (
    make_source_sanitized_pair,
)
from tests.unit.domain.finance._experiment_factories import make_attempt, make_manifest


def _exchange(identifier: str, kind: ProviderExchangeKind) -> ProviderExchangeDigest:
    return ProviderExchangeDigest(
        exchange_id=identifier,
        kind=kind,
        input_sha256="1" * 64,
        response_sha256="2" * 64,
        attempt=make_attempt(),
    )


def _reasoning(arm: ForecastArm, probability: Decimal) -> SanitizedArmReasoning:
    question = make_manifest().questions[0]
    if arm is ForecastArm.DIRECT:
        target = SanitizedTargetView(
            question_text=question.question_text,
            question_type=question.question_type,
            domain=question.domain,
            context=question.context,
            cutoff=make_manifest().cutoff,
            outcome_space=question.outcome_space,
            resolution_rule=question.resolution_rule,
        )
        evidence = ()
        memory = ()
        source_candidate = SanitizedCandidateView(
            alias=NeutralCandidate.CANDIDATE_1,
            scenarios=(
                SanitizedScenarioView(
                    alias="scenario_1",
                    name="Base case",
                    reasoning_steps=("The public indicator informs the base rate.",),
                    probability=1.0,
                    conditional_outcomes=(
                        OutcomeProbability(
                            label=OutcomeLabel("No"),
                            probability=0.5,
                        ),
                        OutcomeProbability(
                            label=OutcomeLabel("Yes"),
                            probability=0.5,
                        ),
                    ),
                    evidence_aliases=(),
                    memory_aliases=(),
                    assumptions=("The official series remains available.",),
                    triggers=("The indicator crosses its threshold.",),
                    disconfirmers=("The indicator reverses.",),
                    uncertainty="Publication timing remains uncertain.",
                ),
            ),
            outcome_probabilities=(
                OutcomeProbability(label=OutcomeLabel("No"), probability=0.5),
                OutcomeProbability(label=OutcomeLabel("Yes"), probability=0.5),
            ),
            explanation="The base case receives the full scenario weight.",
        )
        audit = FinanceAliasAudit(
            salt_base64=b64encode(b"e" * 32).decode("ascii"),
            mappings=(),
        )
    else:
        target, evidence, pair_memory, candidate_1, candidate_2, audit = (
            make_source_sanitized_pair()
        )
        memory = pair_memory if arm is ForecastArm.SEARCH_DAG else ()
        source_candidate = candidate_2 if arm is ForecastArm.SEARCH_DAG else candidate_1
    outcomes = (
        OutcomeProbability(
            label=OutcomeLabel("No"),
            probability=float(1 - probability),
        ),
        OutcomeProbability(
            label=OutcomeLabel("Yes"),
            probability=float(probability),
        ),
    )
    candidate = source_candidate.model_copy(
        update={
            "scenarios": tuple(
                scenario.model_copy(update={"conditional_outcomes": outcomes})
                for scenario in source_candidate.scenarios
            ),
            "outcome_probabilities": outcomes,
        }
    )
    return SanitizedArmReasoning(
        target=target,
        evidence=evidence,
        memory=memory,
        candidate=candidate,
        alias_audit=audit,
    )


def _arm(arm: ForecastArm, probability: Decimal) -> PersistedArmSucceeded:
    digest = None if arm is ForecastArm.DIRECT else "a" * 64
    evidence_bytes = 0 if arm is ForecastArm.DIRECT else 512
    evidence_count = 0 if arm is ForecastArm.DIRECT else 1
    memory_count = 1 if arm is ForecastArm.SEARCH_DAG else 0
    return PersistedArmSucceeded(
        arm=arm,
        treatment=TreatmentAudit(
            arm=arm,
            evidence_snapshot_digest=digest,
            evidence_snapshot_bytes=evidence_bytes,
            evidence_item_count=evidence_count,
            historical_memory_episode_count=memory_count,
        ),
        positive_probability=probability,
        reasoning=_reasoning(arm, probability),
        exchanges=(_exchange(f"{arm.value}-forecast", ProviderExchangeKind.FORECAST),),
    )


def _panel(preference: PanelPreference) -> JudgePanelMetrics:
    first_votes = 1 if preference is PanelPreference.SECOND else 2
    second_votes = 2 if preference is PanelPreference.SECOND else 1
    return JudgePanelMetrics(
        overall_preference=preference,
        preference_eligible=True,
        tie_reason=None,
        first_votes=first_votes,
        second_votes=second_votes,
        tie_votes=0,
        inconsistent_count=0,
        invalid_count=0,
        evaluable_count=3,
        two_parse_valid_count=3,
        attempted_call_count=6,
        agreement=Decimal("0.6666666666666666666666666667"),
        order_consistency=Decimal("1"),
    )


def make_persisted_suite() -> PersistedFinanceExperimentSuite:
    """Build one two-repetition, three-arm, three-panel sanitized suite."""
    manifest = make_manifest()
    question = manifest.questions[0]
    manifest = manifest.model_copy(
        update={"questions": (question,), "repetitions": 2},
    )
    probability_rows = (
        (Decimal("0.6"), Decimal("0.4"), Decimal("0.5")),
        (Decimal("0.8"), Decimal("0.3"), Decimal("0.4")),
    )
    trials: list[PersistedExperimentTrial] = []
    for repetition, probabilities in enumerate(probability_rows):
        arms = tuple(
            _arm(arm, probability)
            for arm, probability in zip(ForecastArm, probabilities, strict=True)
        )
        pair_rows = (
            (ForecastPair.A_B, probabilities[0], probabilities[1]),
            (ForecastPair.B_C, probabilities[1], probabilities[2]),
            (ForecastPair.A_C, probabilities[0], probabilities[2]),
        )
        pairs = tuple(
            PersistedPairCompleted(
                comparison=compare_positive_probabilities(pair, first, second),
                panel=_panel(PanelPreference.SECOND),
                judge_exchanges=tuple(
                    _exchange(
                        f"{pair.value}-judge-{index}",
                        ProviderExchangeKind.JUDGE,
                    )
                    for index in range(6)
                ),
            )
            for pair, first, second in pair_rows
        )
        trials.append(
            PersistedExperimentTrial(
                trial_id=UUID(f"00000000-0000-0000-0000-{repetition + 1:012d}"),
                question_id=question.question_id,
                repetition_index=repetition,
                preparation_exchanges=(),
                arms=arms,
                pairs=pairs,
            )
        )
    return PersistedFinanceExperimentSuite(
        suite_id=UUID("50000000-0000-0000-0000-000000000005"),
        created_at=datetime(2026, 7, 18, 2, tzinfo=UTC),
        status=FinanceSuiteStatus.COMPLETE,
        manifest=manifest,
        trials=tuple(trials),
    )


__all__ = ["make_persisted_suite"]
