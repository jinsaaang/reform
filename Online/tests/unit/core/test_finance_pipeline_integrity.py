"""Gate regressions for run-mode and canonical-memory pipeline integrity."""

from dataclasses import replace
from typing import assert_never, final

import pytest

from src.agents.finance_forecast_agent import FinanceForecastAgent
from src.agents.finance_search_agent import FinanceSearchAgent
from src.core.finance_pipeline import FinancePipelineAgents, FinanceReasoningPipeline
from src.domain.finance.memory import DagId, ResolvedDagEpisode
from src.domain.finance.pipeline import (
    PipelineAbstained,
    PipelineAbstentionReason,
    PipelineResult,
    PipelineRunRequest,
    PipelineStage,
    PipelineSucceeded,
)
from src.domain.finance.provider import FinanceRunMode, SearchSourcePolicy
from src.domain.finance.retrieval import (
    EligibilityPolicy,
    HistoricalDagQuery,
    HistoricalDagRetrieval,
    RankedHistoricalDag,
    TopK,
)
from src.services.historical_dag_retriever import HistoricalDagRetriever
from tests.fixtures.finance_pipeline import (
    BASE_CANDIDATE,
    FixtureCandidate,
    FixtureForecastProvider,
    FixtureSearchProvider,
    make_target,
    provider_envelope,
)
from tests.unit.domain.finance._factories import make_episode


@final
class _CountingRepository:
    """Mutable test spy records canonical source loads."""

    def __init__(self, episodes: tuple[ResolvedDagEpisode, ...]):
        self.episodes = episodes
        self.call_count = 0

    def load_episodes(self) -> tuple[ResolvedDagEpisode, ...]:
        self.call_count += 1
        return self.episodes


@final
class _TamperingRetriever:
    def __init__(self, canonical: ResolvedDagEpisode):
        self.tampered = replace(
            canonical,
            dag_id=DagId("dag-forged"),
            historical_outcome=replace(
                canonical.historical_outcome,
                value="No",
            ),
        )

    def retrieve_from(
        self,
        episodes: tuple[ResolvedDagEpisode, ...],
        query: HistoricalDagQuery,
    ) -> HistoricalDagRetrieval:
        del episodes, query
        ranked = RankedHistoricalDag(
            episode=self.tampered,
            score=1,
            score_components=(),
            matched_terms=("finance",),
            audit_markers=(),
        )
        return HistoricalDagRetrieval(
            selected=(ranked,),
            ranked_candidates=(ranked,),
            excluded=(),
        )


@final
class _StaticRetriever:
    def __init__(self, result: HistoricalDagRetrieval):
        self.result = result

    def retrieve_from(
        self,
        episodes: tuple[ResolvedDagEpisode, ...],
        query: HistoricalDagQuery,
    ) -> HistoricalDagRetrieval:
        del episodes, query
        return self.result


def _request(
    run_mode: FinanceRunMode,
    top_k_value: int = 1,
) -> PipelineRunRequest:
    return PipelineRunRequest(
        target_profile=make_target(),
        run_mode=run_mode,
        source_policy=SearchSourcePolicy.OFFLINE_FIXTURE,
        eligibility_policy=EligibilityPolicy.PUBLIC_DB_BOOTSTRAP,
        top_k=TopK(top_k_value),
    )


def _pipeline(
    repository: _CountingRepository,
    retriever: HistoricalDagRetriever | _TamperingRetriever | _StaticRetriever,
    candidate: FixtureCandidate = BASE_CANDIDATE,
) -> tuple[
    FinanceReasoningPipeline,
    FixtureSearchProvider,
    FixtureForecastProvider,
]:
    search = FixtureSearchProvider(
        (
            provider_envelope((candidate.payload(),)),
            provider_envelope(()),
        )
    )
    forecast = FixtureForecastProvider()
    return (
        FinanceReasoningPipeline(
            FinancePipelineAgents(
                FinanceSearchAgent(search),
                FinanceForecastAgent(forecast),
            ),
            repository,
            retriever,
        ),
        search,
        forecast,
    )


def _abstained(result: PipelineResult) -> PipelineAbstained:
    match result:
        case PipelineAbstained():
            return result
        case PipelineSucceeded():
            raise AssertionError("pipeline unexpectedly succeeded")
        case unreachable:
            assert_never(unreachable)


class TestCanonicalMemoryIntegrity:
    def test_should_load_canonical_episode_source_exactly_once(self) -> None:
        # Given: one canonical source shared by pipeline and concrete retriever
        repository = _CountingRepository((make_episode(),))
        pipeline, _, _ = _pipeline(repository, HistoricalDagRetriever(repository))

        # When: the full historical pipeline executes
        result = pipeline.run(_request(FinanceRunMode.HISTORICAL_BACKTEST))

        # Then: the canonical source is loaded exactly once
        assert result.status == "succeeded"
        assert repository.call_count == 1

    def test_should_reject_same_id_tampered_memory_before_forecast(self) -> None:
        # Given: retriever returns a forged DAG/outcome under a canonical episode ID
        canonical = make_episode()
        repository = _CountingRepository((canonical,))
        pipeline, _, forecast = _pipeline(
            repository,
            _TamperingRetriever(canonical),
        )

        # When: retrieval crosses the canonical-memory boundary
        result = _abstained(pipeline.run(_request(FinanceRunMode.HISTORICAL_BACKTEST)))

        # Then: exact inequality abstains before guided Search or Forecast
        assert result.reason is PipelineAbstentionReason.RETRIEVAL_INCONSISTENT
        assert forecast.call_count == 0

    def test_should_reject_duplicate_selected_before_guided_search(self) -> None:
        # Given: canonical ranking is returned with its selected entry duplicated
        episodes = (make_episode(),)
        repository = _CountingRepository(episodes)
        query = HistoricalDagQuery(
            target_profile=make_target(),
            policy=EligibilityPolicy.PUBLIC_DB_BOOTSTRAP,
            top_k=TopK(1),
        )
        canonical = HistoricalDagRetriever(repository).retrieve_from(episodes, query)
        duplicated = replace(
            canonical,
            selected=(canonical.selected[0], canonical.selected[0]),
        )
        pipeline, search, forecast = _pipeline(
            repository,
            _StaticRetriever(duplicated),
        )

        # When: the malformed selected collection crosses the retrieval boundary
        result = _abstained(pipeline.run(_request(FinanceRunMode.HISTORICAL_BACKTEST)))

        # Then: the pipeline abstains at retrieval without leaking validation errors
        assert result.reason is PipelineAbstentionReason.RETRIEVAL_INCONSISTENT
        assert repository.call_count == 1
        assert search.call_count == 1
        assert forecast.call_count == 0

    @pytest.mark.parametrize("selected_count", (1, 0))
    def test_should_reject_shortened_selected_prefix_before_guided_search(
        self,
        selected_count: int,
    ) -> None:
        # Given: three eligible candidates but TopK(2) is shortened by retriever
        episodes = tuple(make_episode(episode_id=name) for name in ("a", "b", "c"))
        repository = _CountingRepository(episodes)
        query = HistoricalDagQuery(
            target_profile=make_target(),
            policy=EligibilityPolicy.PUBLIC_DB_BOOTSTRAP,
            top_k=TopK(2),
        )
        canonical = HistoricalDagRetriever(repository).retrieve_from(episodes, query)
        assert len(canonical.ranked_candidates) == 3
        shortened = replace(
            canonical,
            selected=canonical.ranked_candidates[:selected_count],
        )
        pipeline, search, forecast = _pipeline(repository, _StaticRetriever(shortened))

        # When: the shortened prefix crosses the retrieval boundary
        result = _abstained(
            pipeline.run(_request(FinanceRunMode.HISTORICAL_BACKTEST, 2))
        )

        # Then: integrity stops at retrieval before guided Search or Forecast
        assert result.reason is PipelineAbstentionReason.RETRIEVAL_INCONSISTENT
        assert result.stage_order[-1] is PipelineStage.HISTORICAL_RETRIEVAL
        assert repository.call_count == 1
        assert search.call_count == 1
        assert forecast.call_count == 0


class TestRunModeEvidencePolicy:
    def test_should_send_distinct_search_requests_for_each_run_mode(self) -> None:
        # Given: otherwise equal current and historical pipeline requests
        episode = make_episode()
        current_repo = _CountingRepository((episode,))
        history_repo = _CountingRepository((episode,))
        current, current_search, _ = _pipeline(
            current_repo,
            HistoricalDagRetriever(current_repo),
        )
        history, history_search, _ = _pipeline(
            history_repo,
            HistoricalDagRetriever(history_repo),
        )

        # When: both run modes execute the same sanitized target
        _ = current.run(_request(FinanceRunMode.CURRENT_UNRESOLVED))
        _ = history.run(_request(FinanceRunMode.HISTORICAL_BACKTEST))

        # Then: both initial and guided provider requests encode the mode
        assert current_search.requests[0] != history_search.requests[0]
        assert current_search.requests[1] != history_search.requests[1]
        for current_request, history_request in zip(
            current_search.requests,
            history_search.requests,
            strict=True,
        ):
            assert current_request.run_mode is FinanceRunMode.CURRENT_UNRESOLVED
            assert history_request.run_mode is FinanceRunMode.HISTORICAL_BACKTEST
            assert current_request.model_dump(exclude={"run_mode"}) == (
                history_request.model_dump(exclude={"run_mode"})
            )

    def test_should_reject_historical_snapshot_captured_at_cutoff(self) -> None:
        # Given: body was pre-cutoff but asserted snapshot capture is at cutoff
        at_cutoff = replace(
            BASE_CANDIDATE,
            candidate_id="snapshot-at-cutoff",
            snapshot_captured_at=make_target().cutoff.isoformat(),
            snapshot_available_at=make_target().cutoff.isoformat(),
        )
        repository = _CountingRepository((make_episode(),))
        pipeline, _, forecast = _pipeline(
            repository,
            HistoricalDagRetriever(repository),
            at_cutoff,
        )

        # When: historical/backtest evidence policy evaluates the snapshot
        result = _abstained(pipeline.run(_request(FinanceRunMode.HISTORICAL_BACKTEST)))

        # Then: snapshot equality fails closed before Forecast
        assert result.reason is PipelineAbstentionReason.NO_ADMITTED_EVIDENCE
        assert forecast.call_count == 0
