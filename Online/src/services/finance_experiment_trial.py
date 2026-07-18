"""One persisted trial within a three-arm finance experiment batch."""

from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal
from typing import Protocol, assert_never
from uuid import UUID

from src.agents.finance_forecast_agent import FinanceForecastAgent
from src.agents.finance_search_agent import FinanceSearchAgent
from src.core.finance_experiment_runner import FinanceExperimentRunner
from src.core.finance_experiment_runtime import (
    FinanceExperimentTrialRequest,
    HistoricalRetriever,
)
from src.domain.finance.artifact import ForecastArm
from src.domain.finance.experiment_artifact import (
    PairUnavailableReason,
    PersistedArmResult,
    PersistedArmSucceeded,
    PersistedExperimentTrial,
    PersistedPairResult,
    PersistedPairUnavailable,
)
from src.domain.finance.experiment_manifest import (
    FinanceBinaryQuestion,
    FinanceExperimentManifest,
)
from src.domain.finance.experiment_pairs import ForecastPair, PAIR_ARMS
from src.domain.finance.experiment_results import (
    ArmFailed,
    ArmSucceeded,
    ArmUnavailable,
)
from src.domain.finance.judge_source import JudgeCandidateSource
from src.domain.finance.retrieval import HistoricalEpisodeSource
from src.domain.finance.experiment_variant_guard import (
    runtime_closed_variant,
    unexpected_closed_variant,
)
from src.services.finance_experiment_contracts import (
    FinanceExperimentProviders,
    FinanceTrialSeedInputs,
    derive_forecast_seed,
)
from src.services.finance_experiment_panels import (
    PairPanelRequest,
    run_persisted_pair_panel,
)
from src.services.finance_experiment_recording import (
    ForecastObservation,
    RecordingForecastProvider,
    RecordingSearchProvider,
)
from src.services.finance_experiment_records import (
    ArmPersistenceRequest,
    build_persisted_arm,
)
from src.services.finance_judge_view import JudgeViewError


class TrialUUIDFactory(Protocol):
    def __call__(self) -> UUID: ...


@dataclass(frozen=True, slots=True)
class FinanceExperimentTrialContext:
    manifest: FinanceExperimentManifest
    providers: FinanceExperimentProviders
    repository: HistoricalEpisodeSource
    retriever: HistoricalRetriever
    monotonic_clock: Callable[[], float]
    uuid_factory: TrialUUIDFactory
    alias_salt: bytes


@dataclass(frozen=True, slots=True)
class _PreverifiedIdentity:
    def verify_identity(self) -> None:
        return None


def _observations_by_arm(
    observations: list[ForecastObservation],
) -> dict[ForecastArm, ForecastObservation]:
    return {observation.arm: observation for observation in observations}


def _panel_inputs(
    runtime_arms: tuple[ArmSucceeded | ArmUnavailable | ArmFailed, ...],
    persisted_arms: tuple[PersistedArmResult, ...],
    observations: dict[ForecastArm, ForecastObservation],
) -> tuple[
    dict[ForecastArm, JudgeCandidateSource],
    dict[ForecastArm, Decimal],
]:
    sources: dict[ForecastArm, JudgeCandidateSource] = {}
    probabilities: dict[ForecastArm, Decimal] = {}
    for runtime_arm, persisted_arm in zip(
        runtime_arms,
        persisted_arms,
        strict=True,
    ):
        candidate = runtime_closed_variant(runtime_arm)
        match candidate:
            case ArmSucceeded():
                observation = observations[runtime_arm.arm]
                sources[runtime_arm.arm] = JudgeCandidateSource(
                    runtime_arm,
                    observation.forecast_input,
                )
                if not isinstance(persisted_arm, PersistedArmSucceeded):
                    raise AssertionError("successful arm projection drifted")
                probabilities[runtime_arm.arm] = persisted_arm.positive_probability
            case ArmUnavailable() | ArmFailed():
                continue
            case _ as unreachable:
                assert_never(unexpected_closed_variant(unreachable))
    return sources, probabilities


def _run_panels(
    context: FinanceExperimentTrialContext,
    seeds: FinanceTrialSeedInputs,
    trial_id: UUID,
    sources: dict[ForecastArm, JudgeCandidateSource],
    probabilities: dict[ForecastArm, Decimal],
) -> tuple[PersistedPairResult, ...]:
    pairs: list[PersistedPairResult] = []
    for pair in ForecastPair:
        first_arm, second_arm = PAIR_ARMS[pair]
        first = sources.get(first_arm)
        second = sources.get(second_arm)
        first_probability = probabilities.get(first_arm)
        second_probability = probabilities.get(second_arm)
        if (
            first is None
            or second is None
            or first_probability is None
            or second_probability is None
        ):
            pairs.append(
                PersistedPairUnavailable(
                    pair=pair,
                    reason=PairUnavailableReason.ARM_UNAVAILABLE,
                )
            )
            continue
        try:
            completed = run_persisted_pair_panel(
                PairPanelRequest(
                    pair=pair,
                    candidates=(first, second),
                    positive_probabilities=(
                        first_probability,
                        second_probability,
                    ),
                    judges=context.manifest.judges,
                    seed_inputs=seeds,
                    provider_factory=(context.providers.judge_provider_factory),
                    alias_salt=context.alias_salt,
                    exchange_prefix=f"{trial_id}:{pair.value}:judge",
                )
            )
        except JudgeViewError:
            pairs.append(
                PersistedPairUnavailable(
                    pair=pair,
                    reason=PairUnavailableReason.SANITIZATION_REJECTED,
                )
            )
        else:
            pairs.append(completed)
    return tuple(pairs)


def run_persisted_experiment_trial(
    context: FinanceExperimentTrialContext,
    question: FinanceBinaryQuestion,
    repetition_index: int,
) -> PersistedExperimentTrial:
    """Execute and project one ordered A/B/C trial."""
    trial_id = context.uuid_factory()
    seeds = FinanceTrialSeedInputs(
        context.manifest.root_seed,
        question.question_id,
        repetition_index,
    )
    forecast_settings = context.manifest.forecast.model_copy(
        update={"requested_seed": derive_forecast_seed(seeds)}
    )
    forecast_recorder = RecordingForecastProvider(
        context.providers.forecast_provider_factory(forecast_settings),
        f"{trial_id}:forecast",
    )
    search_recorder = RecordingSearchProvider(
        context.providers.search_provider,
        f"{trial_id}:search",
        context.monotonic_clock,
    )
    runner = FinanceExperimentRunner(
        searcher=FinanceSearchAgent(search_recorder),
        forecaster=FinanceForecastAgent(forecast_recorder),
        repository=context.repository,
        retriever=context.retriever,
        identity_verifier=_PreverifiedIdentity(),
    )
    trial_run = runner.run_trial(
        FinanceExperimentTrialRequest(
            manifest=context.manifest,
            question=question,
            repetition_index=repetition_index,
            trial_id=trial_id,
        )
    )
    observations = _observations_by_arm(forecast_recorder.observations)
    persisted_arms = tuple(
        build_persisted_arm(
            ArmPersistenceRequest(
                result=arm,
                positive_label=question.positive_label,
                observation=observations.get(arm.arm),
                alias_salt=context.alias_salt,
            )
        )
        for arm in trial_run.trial.arms
    )
    sources, probabilities = _panel_inputs(
        trial_run.trial.arms,
        persisted_arms,
        observations,
    )
    return PersistedExperimentTrial(
        trial_id=trial_id,
        question_id=question.question_id,
        repetition_index=repetition_index,
        preparation_exchanges=tuple(search_recorder.exchanges),
        arms=persisted_arms,
        pairs=_run_panels(context, seeds, trial_id, sources, probabilities),
    )


__all__ = [
    "FinanceExperimentTrialContext",
    "run_persisted_experiment_trial",
]
