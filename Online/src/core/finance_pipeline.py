from dataclasses import dataclass, replace
from typing import Protocol, assert_never

from pydantic import ValidationError

from src.agents import finance_forecast_agent as forecast_agent
from src.agents import finance_search_agent as search_agent
from src.domain.finance import forecast, pipeline, provider, retrieval
from src.domain.finance.memory import ResolvedDagEpisode


class HistoricalRetriever(Protocol):
    def retrieve_from(
        self,
        episodes: tuple[ResolvedDagEpisode, ...],
        query: retrieval.HistoricalDagQuery,
    ) -> retrieval.HistoricalDagRetrieval: ...


@dataclass(frozen=True, slots=True)
class FinancePipelineAgents:
    searcher: search_agent.FinanceSearchAgent
    forecaster: forecast_agent.FinanceForecastAgent


@dataclass(frozen=True, slots=True)
class _TraceState:
    request: pipeline.PipelineRunRequest
    stages: tuple[pipeline.PipelineStage, ...] = ()
    passes: tuple[pipeline.QueryPassTrace, ...] = ()
    selected: tuple[pipeline.SelectedHistoricalDagTrace, ...] = ()
    exclusions: tuple[pipeline.RetrievalExclusionSummary, ...] = ()
    audit: tuple[pipeline.EvidenceAuditRecord, ...] = ()
    provenance: tuple[pipeline.EvidenceProvenance, ...] = ()

    def stage(self, stage: pipeline.PipelineStage) -> "_TraceState":
        return replace(self, stages=(*self.stages, stage))

    def abstain(
        self,
        reason: pipeline.PipelineAbstentionReason,
        detail: str,
    ) -> pipeline.PipelineAbstained:
        return pipeline.PipelineAbstained(
            target_profile=self.request.target_profile,
            run_mode=self.request.run_mode,
            stage_order=self.stages,
            query_passes=self.passes,
            selected_historical_dags=self.selected,
            retrieval_exclusions=self.exclusions,
            evidence_audit=self.audit,
            evidence_provenance=self.provenance,
            reason=reason,
            detail=detail,
        )


@dataclass(frozen=True, slots=True)
class FinanceReasoningPipeline:
    """Execute the fixed two-pass memory-to-forecast stage sequence."""

    agents: FinancePipelineAgents
    repository: retrieval.HistoricalEpisodeSource
    retriever: HistoricalRetriever

    def run(self, request: pipeline.PipelineRunRequest) -> pipeline.PipelineResult:
        """Run the complete deterministic finance reasoning pipeline."""
        state = _TraceState(request).stage(pipeline.PipelineStage.INITIAL_SEARCH)
        initial = self.agents.searcher.search(
            provider.InitialSearchRequest(
                run_mode=request.run_mode,
                target_profile=request.target_profile,
                source_policy=request.source_policy,
                query_intents=tuple(provider.SearchQueryIntent)[:4],
            )
        )
        state = replace(state, passes=(*state.passes, initial.trace))
        match initial:
            case pipeline.SearchPassFailed():
                return state.abstain(
                    pipeline.PipelineAbstentionReason.INITIAL_SEARCH_FAILED,
                    initial.trace.failure_reason.value,
                )
            case pipeline.SearchPassCompleted():
                pass
            case unreachable:
                assert_never(unreachable)

        state = state.stage(pipeline.PipelineStage.IMMUTABLE_EPISODE_LOAD)
        try:
            episodes = self.repository.load_episodes()
        except (
            retrieval.DuplicateEpisodeIdError,
            retrieval.SeedAssetMismatchError,
            retrieval.SeedDataError,
            retrieval.SeedDatabaseReadError,
            retrieval.SeedSchemaError,
            retrieval.UnsafeSeedPathError,
            retrieval.WritableSeedAssetError,
        ) as error:
            return state.abstain(
                pipeline.PipelineAbstentionReason.SEED_MEMORY_UNAVAILABLE,
                str(error),
            )
        if not episodes:
            return state.abstain(
                pipeline.PipelineAbstentionReason.NO_IMMUTABLE_MEMORY,
                "immutable episode source returned no finance memory",
            )

        state = state.stage(pipeline.PipelineStage.HISTORICAL_RETRIEVAL)
        result = self.retriever.retrieve_from(
            episodes,
            retrieval.HistoricalDagQuery(
                target_profile=request.target_profile,
                policy=request.eligibility_policy,
                top_k=request.top_k,
            ),
        )
        if not pipeline.retrieval_matches_canonical_memory(
            episodes,
            result,
            request.top_k,
        ):
            return state.abstain(
                pipeline.PipelineAbstentionReason.RETRIEVAL_INCONSISTENT,
                "retrieval audit differs from canonical immutable memory",
            )
        selected = tuple(
            pipeline.SelectedHistoricalDagTrace(
                episode_id=item.episode.episode_id,
                question_id=item.episode.question_id,
                dag_id=item.episode.dag_id,
                score=item.score,
                matched_terms=item.matched_terms,
                audit_markers=item.audit_markers,
                resolution_date_proxy=(
                    item.episode.historical_resolution.resolution_date_proxy
                ),
                resolution_provenance=(item.episode.historical_resolution.provenance),
            )
            for item in result.selected
        )
        exclusions = tuple(
            pipeline.RetrievalExclusionSummary(
                reason=reason,
                episode_ids=ids,
                count=len(ids),
            )
            for reason in retrieval.ExclusionReason
            if (
                ids := tuple(
                    item.episode.episode_id
                    for item in result.excluded
                    if reason in item.decision.exclusion_reasons
                )
            )
        )
        state = replace(state, selected=selected, exclusions=exclusions)
        if not result.selected:
            return state.abstain(
                pipeline.PipelineAbstentionReason.NO_ELIGIBLE_MEMORY,
                "historical retrieval selected no eligible memory",
            )
        state = state.stage(pipeline.PipelineStage.GUIDED_SEARCH)
        guided = self.agents.searcher.search(
            provider.GuidedSearchRequest(
                run_mode=request.run_mode,
                target_profile=request.target_profile,
                source_policy=request.source_policy,
                query_intents=tuple(provider.SearchQueryIntent)[4:]
                + tuple(provider.SearchQueryIntent)[1:3],
                historical_guidance=self.agents.searcher.build_guidance(
                    result.selected
                ),
            )
        )
        state = replace(state, passes=(*state.passes, guided.trace))
        match guided:
            case pipeline.SearchPassFailed():
                return state.abstain(
                    pipeline.PipelineAbstentionReason.GUIDED_SEARCH_FAILED,
                    guided.trace.failure_reason.value,
                )
            case pipeline.SearchPassCompleted():
                pass
            case unreachable:
                assert_never(unreachable)

        state = state.stage(pipeline.PipelineStage.EVIDENCE_ADMISSION)
        completion = self.agents.searcher.merge_and_admit(
            pipeline.SearchMergeInput(
                target_profile=request.target_profile,
                historical_dag_references=tuple(
                    item.episode.reference for item in result.selected
                ),
                passes=(initial, guided),
            )
        )
        state = replace(
            state,
            audit=completion.evidence_audit,
            provenance=completion.evidence_provenance,
        )
        if not completion.searcher_result.evidence_pack.items:
            return state.abstain(
                pipeline.PipelineAbstentionReason.NO_ADMITTED_EVIDENCE,
                "no exact body was available strictly before target cutoff",
            )
        try:
            provider_input = forecast.ForecasterInput(
                target_profile=request.target_profile,
                evidence_pack=completion.searcher_result.evidence_pack,
                historical_memory=tuple(item.episode for item in result.selected),
            )
        except ValidationError:
            return state.abstain(
                pipeline.PipelineAbstentionReason.FORECAST_INPUT_INVALID,
                "selected history violates ForecasterInput contract",
            )
        state = state.stage(pipeline.PipelineStage.FORECASTER)
        forecasted = self.agents.forecaster.forecast(provider_input)
        match forecasted:
            case forecast_agent.ForecastAgentSucceeded():
                return pipeline.PipelineSucceeded(
                    target_profile=request.target_profile,
                    run_mode=request.run_mode,
                    stage_order=state.stages,
                    query_passes=state.passes,
                    selected_historical_dags=state.selected,
                    retrieval_exclusions=state.exclusions,
                    evidence_audit=state.audit,
                    evidence_provenance=state.provenance,
                    searcher_result=completion.searcher_result,
                    forecast_result=forecasted.forecast_result,
                )
            case forecast_agent.ForecastAgentAbstained(
                reason=pipeline.ForecastAbstentionReason.PROVIDER_ERROR
            ):
                return state.abstain(
                    pipeline.PipelineAbstentionReason.FORECAST_PROVIDER_ERROR,
                    forecasted.detail,
                )
            case forecast_agent.ForecastAgentAbstained():
                return state.abstain(
                    pipeline.PipelineAbstentionReason.FORECAST_OUTPUT_INVALID,
                    forecasted.reason.value,
                )
            case unreachable:
                assert_never(unreachable)
