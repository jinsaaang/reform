"""Deterministic offline providers for Todo 6 batch-service tests."""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from uuid import UUID

from src.agents.finance_reasoning_judge import TransientJudgeProviderCompletion
from src.domain.finance.experiment_manifest import (
    FinanceExperimentManifest,
    SafeCompletionSettings,
    Seed,
)
from src.domain.finance.experiment_telemetry import (
    ProviderAttemptStatus,
    ProviderAttemptTelemetry,
    ProviderSeedEffective,
    ProviderUsageUnavailable,
)
from src.domain.finance.forecast import (
    ForecasterInput,
    ForecastResult,
    OutcomeProbability,
    Scenario,
)
from src.domain.finance.judge_views import BlindJudgePayload, NeutralCandidate
from src.domain.finance.judging import JudgePreference
from src.domain.finance.memory import OutcomeLabel, ScenarioId
from src.domain.finance.provider import SearchRequest
from src.integrations.finance_live_forecast import (
    TransientForecastProviderCompletion,
)
from src.services.finance_experiment import (
    ExperimentForecastProvider,
    FinanceExperimentDependencies,
    FinanceExperimentProviderBuilder,
    FinanceExperimentProviders,
    JudgeProviderFactory,
)
from src.services.historical_dag_retriever import HistoricalDagRetriever
from tests.fixtures.finance_experiment_runner import (
    RecordingIdentityVerifier,
    StaticRepository,
    search_envelope,
)
from tests.fixtures.finance_judge_panel import make_judge_response
from tests.unit.domain.finance._factories import make_episode


@dataclass(frozen=True, slots=True)
class OfflineExperimentState:
    """Mutable-list accumulator for observable provider construction and calls."""

    events: list[str] = field(default_factory=list)
    search_requests: list[SearchRequest] = field(default_factory=list)
    forecast_inputs: list[ForecasterInput] = field(default_factory=list)
    forecast_settings: list[SafeCompletionSettings] = field(default_factory=list)
    judge_payloads: list[BlindJudgePayload] = field(default_factory=list)
    judge_settings: list[SafeCompletionSettings] = field(default_factory=list)
    judge_seeds: list[Seed] = field(default_factory=list)
    provider_build_count: list[int] = field(default_factory=list)
    malformed_forecasts: bool = False


def _attempt(
    seed: Seed, provider_name: str, output_bytes: int
) -> ProviderAttemptTelemetry:
    return ProviderAttemptTelemetry(
        attempt_index=0,
        provider_name=provider_name,
        status=ProviderAttemptStatus.SUCCEEDED,
        latency_ms=1,
        input_bytes=32,
        output_bytes=output_bytes,
        seed=ProviderSeedEffective(requested_seed=seed),
        usage=ProviderUsageUnavailable(),
    )


@dataclass(frozen=True, slots=True)
class OfflineSearchProvider:
    state: OfflineExperimentState
    response: str

    def search(self, request: SearchRequest) -> str:
        self.state.search_requests.append(request)
        self.state.events.append("search")
        return self.response


@dataclass(frozen=True, slots=True)
class OfflineForecastProvider(ExperimentForecastProvider):
    settings: SafeCompletionSettings
    state: OfflineExperimentState

    def complete(
        self,
        forecast_input: ForecasterInput,
    ) -> TransientForecastProviderCompletion:
        seed = self.settings.requested_seed
        if seed is None:
            raise AssertionError("service must derive one forecast seed")
        self.state.forecast_inputs.append(forecast_input)
        if self.state.malformed_forecasts:
            result = "not-json"
            return TransientForecastProviderCompletion(
                serialized_response=result,
                attempt=_attempt(
                    seed,
                    f"offline:{self.settings.model}",
                    len(result),
                ),
            )
        evidence = forecast_input.evidence_pack.items
        memory = forecast_input.historical_memory
        positive = 0.6 if memory else 0.5 if evidence else 0.4
        outcomes = (
            OutcomeProbability(label=OutcomeLabel("No"), probability=1 - positive),
            OutcomeProbability(label=OutcomeLabel("Yes"), probability=positive),
        )
        result = ForecastResult(
            scenarios=(
                Scenario(
                    scenario_id=ScenarioId(
                        f"scenario-{len(self.state.forecast_inputs):03d}"
                    ),
                    name="Offline case",
                    reasoning_steps=("The fixed offline inputs determine the case.",),
                    probability=1.0,
                    conditional_outcomes=outcomes,
                    evidence_ids=tuple(item.evidence_id for item in evidence),
                    historical_dag_references=tuple(
                        episode.reference for episode in memory
                    ),
                    assumptions=("The fixture remains internally consistent.",),
                    triggers=("The official indicator moves.",),
                    disconfirmers=("The indicator reverses.",),
                    uncertainty="Future public information remains unknown.",
                ),
            ),
            outcome_probabilities=outcomes,
            explanation="One deterministic offline scenario is retained.",
        ).model_dump_json()
        return TransientForecastProviderCompletion(
            serialized_response=result,
            attempt=_attempt(seed, f"offline:{self.settings.model}", len(result)),
        )


@dataclass(frozen=True, slots=True)
class OfflineForecastFactory:
    state: OfflineExperimentState

    def __call__(
        self,
        settings: SafeCompletionSettings,
    ) -> ExperimentForecastProvider:
        self.state.forecast_settings.append(settings)
        return OfflineForecastProvider(settings, self.state)


@dataclass(frozen=True, slots=True)
class OfflineJudgeProvider:
    settings: SafeCompletionSettings
    state: OfflineExperimentState

    def judge(
        self,
        payload: BlindJudgePayload,
        requested_seed: Seed,
    ) -> TransientJudgeProviderCompletion:
        self.state.judge_payloads.append(payload)
        self.state.judge_seeds.append(requested_seed)
        preference = (
            JudgePreference.ANSWER_A
            if payload.answer_a.alias is NeutralCandidate.CANDIDATE_1
            else JudgePreference.ANSWER_B
        )
        result = make_judge_response(preference)
        return TransientJudgeProviderCompletion(
            serialized_response=result,
            attempt=_attempt(requested_seed, "offline:judge", len(result)),
        )


@dataclass(frozen=True, slots=True)
class OfflineJudgeFactory(JudgeProviderFactory):
    state: OfflineExperimentState

    def __call__(self, settings: SafeCompletionSettings) -> OfflineJudgeProvider:
        self.state.judge_settings.append(settings)
        return OfflineJudgeProvider(settings, self.state)


@dataclass(frozen=True, slots=True)
class OfflineProviderBuilder(FinanceExperimentProviderBuilder):
    state: OfflineExperimentState
    search_response: str = field(default_factory=search_envelope)

    def __call__(
        self,
        manifest: FinanceExperimentManifest,
    ) -> FinanceExperimentProviders:
        del manifest
        self.state.provider_build_count.append(1)
        return FinanceExperimentProviders(
            search_provider=OfflineSearchProvider(self.state, self.search_response),
            forecast_provider_factory=OfflineForecastFactory(self.state),
            judge_provider_factory=OfflineJudgeFactory(self.state),
        )


class UUIDSequence:
    """Deterministic UUID allocator; mutation is its documented purpose."""

    __slots__ = ("_next",)

    def __init__(self) -> None:
        self._next = 1

    def __call__(self) -> UUID:
        value = UUID(int=self._next)
        self._next += 1
        return value


def fixed_clock() -> datetime:
    return datetime(2026, 7, 18, 1, tzinfo=UTC)


def make_offline_dependencies(
    state: OfflineExperimentState,
) -> FinanceExperimentDependencies:
    events = state.events
    repository = StaticRepository((make_episode(),), events)
    return FinanceExperimentDependencies(
        provider_builder=OfflineProviderBuilder(state),
        identity_verifier=RecordingIdentityVerifier(events),
        repository=repository,
        retriever=HistoricalDagRetriever(repository),
        clock=fixed_clock,
        monotonic_clock=lambda: 1.0,
        uuid_factory=UUIDSequence(),
        alias_salt_factory=lambda: b"s" * 32,
    )


__all__ = [
    "OfflineExperimentState",
    "OfflineProviderBuilder",
    "fixed_clock",
    "make_offline_dependencies",
]
