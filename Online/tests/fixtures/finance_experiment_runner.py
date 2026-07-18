"""Typed spies for fixed-pack finance experiment runner tests."""

from dataclasses import replace
from pathlib import Path

from src.core.finance_experiment_runtime import FinanceExperimentRetrievalError
from src.domain.finance.artifact import ForecastArm
from src.domain.finance.forecast import (
    ForecasterInput,
    ForecastResult,
    OutcomeProbability,
    Scenario,
)
from src.domain.finance.memory import OutcomeLabel, ResolvedDagEpisode, ScenarioId
from src.domain.finance.provider import hash_exact_utf8_body
from src.domain.finance.retrieval import (
    HistoricalDagQuery,
    HistoricalDagRetrieval,
    HistoricalEpisodeSource,
    SeedAssetMismatchError,
)
from src.services.historical_dag_retriever import HistoricalDagRetriever
from tests.fixtures.finance_pipeline import BASE_CANDIDATE, provider_envelope


class RecordingIdentityVerifier:
    """Record successful identity-only preflight."""

    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.call_count = 0

    def verify_identity(self) -> None:
        self.events.append("metadata_preflight")
        self.call_count += 1


class FailingIdentityVerifier:
    """Reject a stale DB identity with the repository's typed error."""

    def __init__(self) -> None:
        self.call_count = 0

    def verify_identity(self) -> None:
        self.call_count += 1
        raise SeedAssetMismatchError(
            path=Path("fixture.db"),
            attribute="sha256",
            expected="a" * 64,
            actual="b" * 64,
        )


class RecordingRepository:
    """Record the one deferred immutable episode load."""

    def __init__(self, source: HistoricalEpisodeSource, events: list[str]) -> None:
        self.source = source
        self.events = events
        self.call_count = 0

    def load_episodes(self) -> tuple[ResolvedDagEpisode, ...]:
        self.events.append("repository_load")
        self.call_count += 1
        return self.source.load_episodes()


class StaticRepository:
    """Return one fixed canonical tuple while recording calls."""

    def __init__(
        self,
        episodes: tuple[ResolvedDagEpisode, ...],
        events: list[str],
    ) -> None:
        self.episodes = episodes
        self.events = events
        self.call_count = 0

    def load_episodes(self) -> tuple[ResolvedDagEpisode, ...]:
        self.events.append("repository_load")
        self.call_count += 1
        return self.episodes


class FailingRepository:
    """Raise a typed repository identity failure after Search-only."""

    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.call_count = 0

    def load_episodes(self) -> tuple[ResolvedDagEpisode, ...]:
        self.events.append("repository_load")
        self.call_count += 1
        raise SeedAssetMismatchError(
            path=Path("fixture.db"),
            attribute="sha256",
            expected="a" * 64,
            actual="b" * 64,
        )


class RecordingRetriever:
    """Record pure retrieval over the caller's canonical tuple."""

    def __init__(self, source: HistoricalEpisodeSource, events: list[str]) -> None:
        self.delegate = HistoricalDagRetriever(source)
        self.events = events
        self.call_count = 0
        self.results: list[HistoricalDagRetrieval] = []

    def retrieve_from(
        self,
        episodes: tuple[ResolvedDagEpisode, ...],
        query: HistoricalDagQuery,
    ) -> HistoricalDagRetrieval:
        self.events.append("historical_retrieval")
        self.call_count += 1
        result = self.delegate.retrieve_from(episodes, query)
        self.results.append(result)
        return result


class FailingRetriever:
    """Raise the runner's narrow expected retriever failure."""

    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.call_count = 0

    def retrieve_from(
        self,
        episodes: tuple[ResolvedDagEpisode, ...],
        query: HistoricalDagQuery,
    ) -> HistoricalDagRetrieval:
        del episodes, query
        self.events.append("historical_retrieval")
        self.call_count += 1
        raise FinanceExperimentRetrievalError("fixture retriever unavailable")


class StaticRetriever:
    """Return a chosen retrieval for integrity and eligibility failures."""

    def __init__(self, result: HistoricalDagRetrieval, events: list[str]) -> None:
        self.result = result
        self.events = events
        self.call_count = 0

    def retrieve_from(
        self,
        episodes: tuple[ResolvedDagEpisode, ...],
        query: HistoricalDagQuery,
    ) -> HistoricalDagRetrieval:
        del episodes, query
        self.events.append("historical_retrieval")
        self.call_count += 1
        return self.result


class MemoryAwareForecastProvider:
    """Emit references from actual historical memory, never pack metadata."""

    def __init__(
        self,
        events: list[str],
        malformed_arm: ForecastArm | None = None,
    ) -> None:
        self.events = events
        self.malformed_arm = malformed_arm
        self.inputs: list[ForecasterInput] = []

    @property
    def call_count(self) -> int:
        return len(self.inputs)

    def forecast(self, forecast_input: ForecasterInput) -> str:
        self.inputs.append(forecast_input)
        arm = _input_arm(forecast_input)
        self.events.append(f"forecast:{arm.value}")
        if arm is self.malformed_arm:
            return "not-json"
        outcomes = tuple(
            OutcomeProbability(label=OutcomeLabel(label), probability=0.5)
            for label in forecast_input.target_profile.outcome_space
        )
        scenario = Scenario(
            scenario_id=ScenarioId(f"scenario-{arm.value}"),
            name="Balanced case",
            reasoning_steps=("The admitted inputs leave both outcomes plausible.",),
            probability=1.0,
            conditional_outcomes=outcomes,
            evidence_ids=tuple(
                item.evidence_id for item in forecast_input.evidence_pack.items
            ),
            historical_dag_references=tuple(
                episode.reference for episode in forecast_input.historical_memory
            ),
            assumptions=("The official series remains available.",),
            triggers=("The target indicator moves materially.",),
            disconfirmers=("The indicator reverses before resolution.",),
            uncertainty="Future public information remains unknown.",
        )
        return ForecastResult(
            scenarios=(scenario,),
            outcome_probabilities=outcomes,
            explanation="One balanced scenario determines the distribution.",
        ).model_dump_json()


def _input_arm(forecast_input: ForecasterInput) -> ForecastArm:
    if forecast_input.historical_memory:
        return ForecastArm.SEARCH_DAG
    if forecast_input.evidence_pack.items:
        return ForecastArm.SEARCH_ONLY
    return ForecastArm.DIRECT


def unordered_candidate_payloads() -> tuple[str, ...]:
    """Return later evidence first so the snapshot must reorder it."""
    earlier_body = "An earlier official observation.\n"
    earlier = replace(
        BASE_CANDIDATE,
        candidate_id="evidence-0",
        claim="An earlier official observation.",
        citation="fixture://official/earlier",
        canonical_source_id="official:earlier",
        source_version_id="official:earlier:v1",
        exact_body=earlier_body,
        available_at="2024-11-01T00:00:00+00:00",
        retrieved_at="2024-11-02T00:00:00+00:00",
        content_hash=hash_exact_utf8_body(earlier_body),
    )
    return (BASE_CANDIDATE.payload(), earlier.payload())


def search_envelope(
    candidates: tuple[str, ...] | None = None,
) -> str:
    selected = unordered_candidate_payloads() if candidates is None else candidates
    return provider_envelope(selected)


__all__ = [
    "FailingIdentityVerifier",
    "FailingRepository",
    "FailingRetriever",
    "MemoryAwareForecastProvider",
    "RecordingIdentityVerifier",
    "RecordingRepository",
    "RecordingRetriever",
    "StaticRepository",
    "StaticRetriever",
    "search_envelope",
]
