"""Offline end-to-end tests for the deterministic finance pipeline."""

from dataclasses import replace
from pathlib import Path
from typing import assert_never, final

from src.agents.finance_forecast_agent import FinanceForecastAgent
from src.agents.finance_search_agent import FinanceSearchAgent
from src.core.finance_pipeline import (
    FinancePipelineAgents,
    FinanceReasoningPipeline,
)
from src.core.finance_seed_repository import FinanceSeedRepository
from src.domain.finance.memory import QuestionId, ResolvedDagEpisode
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
    AuditMarker,
    EligibilityPolicy,
    HistoricalDagQuery,
    HistoricalDagRetrieval,
    TopK,
)
from src.domain.finance.search import TargetProfile
from src.services.historical_dag_retriever import HistoricalDagRetriever
from tests.fixtures.finance_pipeline import (
    BASE_CANDIDATE,
    FixtureForecastProvider,
    FixtureSearchProvider,
    make_target,
    mixed_search_responses,
    provider_envelope,
)

_DB_PATH = Path("data/releases/worldreasoner/v1.0.0/worldreasoner_public.db")
_EXPECTED_STAGES = (
    PipelineStage.INITIAL_SEARCH,
    PipelineStage.IMMUTABLE_EPISODE_LOAD,
    PipelineStage.HISTORICAL_RETRIEVAL,
    PipelineStage.GUIDED_SEARCH,
    PipelineStage.EVIDENCE_ADMISSION,
    PipelineStage.FORECASTER,
)


def _pipeline(
    search_responses: tuple[str, ...],
    forecast_output: str | None = None,
) -> tuple[
    FinanceReasoningPipeline,
    FixtureSearchProvider,
    FixtureForecastProvider,
    list[str],
]:
    events: list[str] = []
    search_provider = FixtureSearchProvider(search_responses, events)
    forecast_provider = FixtureForecastProvider(forecast_output, events)
    repository = FinanceSeedRepository(_DB_PATH)
    return (
        FinanceReasoningPipeline(
            agents=FinancePipelineAgents(
                searcher=FinanceSearchAgent(search_provider),
                forecaster=FinanceForecastAgent(forecast_provider),
            ),
            repository=repository,
            retriever=HistoricalDagRetriever(repository),
        ),
        search_provider,
        forecast_provider,
        events,
    )


def _request(
    policy: EligibilityPolicy = EligibilityPolicy.PUBLIC_DB_BOOTSTRAP,
    target: TargetProfile | None = None,
) -> PipelineRunRequest:
    return PipelineRunRequest(
        target_profile=target or make_target(),
        run_mode=FinanceRunMode.HISTORICAL_BACKTEST,
        source_policy=SearchSourcePolicy.OFFLINE_FIXTURE,
        eligibility_policy=policy,
        top_k=TopK(3),
    )


def _succeeded(result: PipelineResult) -> PipelineSucceeded:
    match result:
        case PipelineSucceeded():
            return result
        case PipelineAbstained():
            raise AssertionError(result.reason)
        case unreachable:
            assert_never(unreachable)


def _abstained(result: PipelineResult) -> PipelineAbstained:
    match result:
        case PipelineAbstained():
            return result
        case PipelineSucceeded():
            raise AssertionError("pipeline unexpectedly succeeded")
        case unreachable:
            assert_never(unreachable)


class _EmptyRepository:
    def load_episodes(self) -> tuple[ResolvedDagEpisode, ...]:
        return ()


@final
class _UnusedRetriever:
    """Retriever spy that must remain uncalled when memory is empty."""

    def __init__(self) -> None:
        self.call_count = 0

    def retrieve_from(
        self,
        episodes: tuple[ResolvedDagEpisode, ...],
        query: HistoricalDagQuery,
    ) -> HistoricalDagRetrieval:
        del episodes, query
        self.call_count += 1
        return HistoricalDagRetrieval(selected=(), ranked_candidates=(), excluded=())


class TestFinanceReasoningPipeline:
    def test_should_run_real_db_stages_in_exact_order_with_typed_trace(self) -> None:
        # Given: mixed temporal evidence, duplicate evidence, and the real seed DB
        pipeline, search_provider, forecast_provider, events = _pipeline(
            mixed_search_responses()
        )

        # When: the complete offline pipeline runs
        result = _succeeded(pipeline.run(_request()))

        # Then: every stage and isolated provider call is observable in order
        assert result.stage_order == _EXPECTED_STAGES
        assert events == ["search:initial", "search:guided", "forecast"]
        assert search_provider.call_count == 2 and forecast_provider.call_count == 1
        assert len(result.selected_historical_dags) == 3
        assert all(
            selected.audit_markers == (AuditMarker.UNVERIFIED_RELATION_METADATA,)
            for selected in result.selected_historical_dags
        )
        assert tuple(
            str(item.evidence_id) for item in result.searcher_result.evidence_pack.items
        ) == ("evidence-1", "evidence-2")
        assert result.forecast_result.outcome_probabilities[0].probability == 0.5
        serialized = result.model_dump_json()
        assert "ground_truth" not in serialized and "current_dag" not in serialized

    def test_should_produce_equal_result_for_repeated_real_db_run(self) -> None:
        # Given: two fresh deterministic provider/pipeline instances
        first_pipeline, _, _, _ = _pipeline(mixed_search_responses())
        second_pipeline, _, _, _ = _pipeline(mixed_search_responses())

        # When: each executes the same request
        first = first_pipeline.run(_request())
        second = second_pipeline.run(_request())

        # Then: complete typed results are equal
        assert first == second

    def test_should_change_retrieved_history_when_target_context_changes(self) -> None:
        # Given: NVIDIA and oil targets with otherwise identical run policy
        oil_target = TargetProfile(
            question_id=QuestionId("current-oil-target"),
            question_text="Will OPEC production raise crude oil prices?",
            question_type=make_target().question_type,
            domain="finance",
            context=("OPEC crude oil refinery supply production",),
            cutoff=make_target().cutoff,
            outcome_space=("Yes", "No"),
            resolution_rule="Use the official market settlement.",
        )
        first_pipeline, _, _, _ = _pipeline(mixed_search_responses())
        second_pipeline, _, _, _ = _pipeline(mixed_search_responses())

        # When: both targets traverse the real retriever
        chips = _succeeded(first_pipeline.run(_request()))
        oil = _succeeded(second_pipeline.run(_request(target=oil_target)))

        # Then: context changes the top immutable episode
        assert (
            chips.selected_historical_dags[0].episode_id
            != oil.selected_historical_dags[0].episode_id
        )

    def test_should_abstain_without_forecast_when_all_evidence_is_post_cutoff(
        self,
    ) -> None:
        # Given: both Searcher passes return only post-cutoff bodies
        post = replace(
            BASE_CANDIDATE,
            available_at="2026-06-02T00:00:00+00:00",
            retrieved_at="2026-06-03T00:00:00+00:00",
        )
        pipeline, search_provider, forecast_provider, _ = _pipeline(
            (provider_envelope((post.payload(),)), provider_envelope(()))
        )

        # When: the pipeline applies evidence admission
        result = _abstained(pipeline.run(_request()))

        # Then: it abstains and never enters the Forecast provider
        assert result.reason is PipelineAbstentionReason.NO_ADMITTED_EVIDENCE
        assert search_provider.call_count == 2 and forecast_provider.call_count == 0

    def test_should_abstain_before_guided_search_when_memory_is_empty(self) -> None:
        # Given: an empty immutable source and a retriever spy
        events: list[str] = []
        search_provider = FixtureSearchProvider((provider_envelope(()),), events)
        forecast_provider = FixtureForecastProvider(events=events)
        retriever = _UnusedRetriever()
        pipeline = FinanceReasoningPipeline(
            FinancePipelineAgents(
                FinanceSearchAgent(search_provider),
                FinanceForecastAgent(forecast_provider),
            ),
            _EmptyRepository(),
            retriever,
        )

        # When: the pipeline reaches immutable memory loading
        result = _abstained(pipeline.run(_request()))

        # Then: no retrieval, guided search, or forecast is attempted
        assert result.reason is PipelineAbstentionReason.NO_IMMUTABLE_MEMORY
        assert search_provider.call_count == 1
        assert retriever.call_count == 0 and forecast_provider.call_count == 0

    def test_should_report_strict_policy_no_memory_without_relaxation(self) -> None:
        # Given: public DB relation metadata under STRICT policy
        pipeline, search_provider, forecast_provider, _ = _pipeline(
            (provider_envelope(()),)
        )

        # When: the hard relation gate runs
        result = _abstained(pipeline.run(_request(EligibilityPolicy.STRICT)))

        # Then: all memory is excluded and policy is not silently relaxed
        assert result.reason is PipelineAbstentionReason.NO_ELIGIBLE_MEMORY
        assert len(result.retrieval_exclusions) == 1
        assert search_provider.call_count == 1 and forecast_provider.call_count == 0

    def test_should_abstain_on_malformed_forecast_without_partial_success(self) -> None:
        # Given: valid search evidence and malformed Forecast provider output
        responses = (
            provider_envelope((BASE_CANDIDATE.payload(),)),
            provider_envelope(()),
        )
        pipeline, _, forecast_provider, _ = _pipeline(responses, "not-json")

        # When: the final provider output is validated
        result = _abstained(pipeline.run(_request()))

        # Then: the terminal value is a typed abstention with one provider call
        assert result.reason is PipelineAbstentionReason.FORECAST_OUTPUT_INVALID
        assert forecast_provider.call_count == 1
