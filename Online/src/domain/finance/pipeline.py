"""Immutable cross-stage audit, trace, and pipeline result models."""

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum, unique
from typing import Annotated, ClassVar, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field

from src.domain.finance import forecast, memory, retrieval, search
from src.domain.finance.provider import (
    EvidenceAuditReason,
    FinanceRunMode,
    HistoricalSearchGuidance,
    SearchPassFailureReason,
    SearchPassKind,
    SearchQueryIntent,
    SearchRequest,
    SearchSourcePolicy,
)


@unique
class EvidenceDecision(StrEnum):
    ADMITTED = "admitted"
    EXCLUDED = "excluded"


@unique
class ForecastAbstentionReason(StrEnum):
    PROVIDER_ERROR = "provider_error"
    MALFORMED_OUTPUT = "malformed_output"
    OUTCOME_SPACE_MISMATCH = "outcome_space_mismatch"
    UNKNOWN_EVIDENCE_REFERENCE = "unknown_evidence_reference"
    UNKNOWN_HISTORICAL_REFERENCE = "unknown_historical_reference"


@unique
class PipelineAbstentionReason(StrEnum):
    INITIAL_SEARCH_FAILED = "initial_search_failed"
    SEED_MEMORY_UNAVAILABLE = "seed_memory_unavailable"
    NO_IMMUTABLE_MEMORY = "no_immutable_memory"
    NO_ELIGIBLE_MEMORY = "no_eligible_memory"
    RETRIEVAL_INCONSISTENT = "retrieval_inconsistent"
    GUIDED_SEARCH_FAILED = "guided_search_failed"
    NO_ADMITTED_EVIDENCE = "no_admitted_evidence"
    FORECAST_INPUT_INVALID = "forecast_input_invalid"
    FORECAST_PROVIDER_ERROR = "forecast_provider_error"
    FORECAST_OUTPUT_INVALID = "forecast_output_invalid"


@unique
class PipelineStage(StrEnum):
    INITIAL_SEARCH = "initial_search"
    IMMUTABLE_EPISODE_LOAD = "immutable_episode_load"
    HISTORICAL_RETRIEVAL = "historical_retrieval"
    GUIDED_SEARCH = "guided_search"
    EVIDENCE_ADMISSION = "evidence_admission"
    FORECASTER = "forecaster"


@dataclass(frozen=True, slots=True)
class PipelineRunRequest:
    target_profile: search.TargetProfile
    run_mode: FinanceRunMode
    source_policy: SearchSourcePolicy
    eligibility_policy: retrieval.EligibilityPolicy
    top_k: retrieval.TopK


class _TraceModel(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")


class _QueryPassBase(_TraceModel):
    pass_kind: SearchPassKind
    run_mode: FinanceRunMode
    source_policy: SearchSourcePolicy
    query_intents: tuple[SearchQueryIntent, ...]
    historical_dag_references: tuple[memory.HistoricalDagReference, ...]
    matched_terms: tuple[str, ...]
    mechanism_hints: tuple[str, ...]


class QueryPassCompleted(_QueryPassBase):
    status: Literal["completed"] = "completed"
    candidate_count: int = Field(ge=0)


class QueryPassFailed(_QueryPassBase):
    status: Literal["failed"] = "failed"
    failure_reason: SearchPassFailureReason


QueryPassTrace: TypeAlias = Annotated[
    QueryPassCompleted | QueryPassFailed,
    Field(discriminator="status"),
]


class EvidenceAuditRecord(_TraceModel):
    sequence: int = Field(ge=0)
    candidate_key: str = Field(min_length=1)
    source_pass: SearchPassKind
    decision: EvidenceDecision
    reason: EvidenceAuditReason
    duplicate_of: memory.EvidenceId | None = None
    canonical_source_id: str | None = None
    source_version_id: str | None = None
    body_snapshot_id: str | None = None
    content_hash: str | None = None


class EvidenceProvenance(_TraceModel):
    evidence_id: memory.EvidenceId
    source_passes: tuple[SearchPassKind, ...]
    canonical_source_id: str = Field(min_length=1)
    source_version_id: str = Field(min_length=1)
    body_snapshot_id: str | None = None
    content_hash: str = Field(min_length=1)


class SelectedHistoricalDagTrace(_TraceModel):
    episode_id: memory.EpisodeId
    question_id: memory.QuestionId
    dag_id: memory.DagId
    score: int
    matched_terms: tuple[str, ...]
    audit_markers: tuple[retrieval.AuditMarker, ...]
    resolution_date_proxy: datetime
    resolution_provenance: memory.ResolutionProvenance


class RetrievalExclusionSummary(_TraceModel):
    reason: retrieval.ExclusionReason
    episode_ids: tuple[memory.EpisodeId, ...]
    count: int = Field(ge=1)


@dataclass(frozen=True, slots=True)
class SearchPassCompleted:
    request: SearchRequest
    candidate_payloads: tuple[str, ...]
    trace: QueryPassCompleted


@dataclass(frozen=True, slots=True)
class SearchPassFailed:
    request: SearchRequest
    trace: QueryPassFailed


SearchPassResult: TypeAlias = SearchPassCompleted | SearchPassFailed


@dataclass(frozen=True, slots=True)
class SearchMergeInput:
    target_profile: search.TargetProfile
    historical_dag_references: tuple[memory.HistoricalDagReference, ...]
    passes: tuple[SearchPassCompleted, ...]


@dataclass(frozen=True, slots=True)
class SearchCompletion:
    searcher_result: search.SearcherResult
    evidence_audit: tuple[EvidenceAuditRecord, ...]
    evidence_provenance: tuple[EvidenceProvenance, ...]


def build_historical_guidance(
    selected: tuple[retrieval.RankedHistoricalDag, ...],
) -> tuple[HistoricalSearchGuidance, ...]:
    """Reduce selected episodes to outcome-free provider guidance."""
    return tuple(
        HistoricalSearchGuidance(
            reference=item.episode.reference,
            matched_terms=item.matched_terms,
            mechanism_hints=tuple(
                sorted(
                    {node.event_type for node in item.episode.nodes}
                    | {tag for node in item.episode.nodes for tag in node.tags}
                    | {
                        edge.relation_type
                        for edge in item.episode.edges
                        if edge.relation_type
                    }
                )
            )[:16],
        )
        for item in selected
    )


def retrieval_matches_canonical_memory(
    episodes: tuple[memory.ResolvedDagEpisode, ...],
    result: retrieval.HistoricalDagRetrieval,
    top_k: retrieval.TopK,
) -> bool:
    """Require retrieval audit values to equal the one canonical source load."""
    canonical = {episode.episode_id: episode for episode in episodes}
    if len(canonical) != len(episodes):
        return False
    selected_ids = tuple(item.episode.episode_id for item in result.selected)
    ranked_episodes = tuple(item.episode for item in result.ranked_candidates)
    ranked_ids = tuple(episode.episode_id for episode in ranked_episodes)
    excluded_episodes = tuple(item.episode for item in result.excluded)
    excluded_ids = tuple(episode.episode_id for episode in excluded_episodes)
    if any(
        len(ids) != len(set(ids)) for ids in (selected_ids, ranked_ids, excluded_ids)
    ):
        return False
    ranked_id_set = set(ranked_ids)
    excluded_id_set = set(excluded_ids)
    if ranked_id_set & excluded_id_set:
        return False
    if ranked_id_set | excluded_id_set != set(canonical):
        return False
    if any(
        canonical.get(episode.episode_id) != episode
        for episode in (*ranked_episodes, *excluded_episodes)
    ):
        return False
    expected_count = min(top_k.value, len(result.ranked_candidates))
    return result.selected == result.ranked_candidates[:expected_count]


class _PipelineTrace(_TraceModel):
    target_profile: search.TargetProfile
    run_mode: FinanceRunMode
    stage_order: tuple[PipelineStage, ...]
    query_passes: tuple[QueryPassTrace, ...]
    selected_historical_dags: tuple[SelectedHistoricalDagTrace, ...]
    retrieval_exclusions: tuple[RetrievalExclusionSummary, ...]
    evidence_audit: tuple[EvidenceAuditRecord, ...]
    evidence_provenance: tuple[EvidenceProvenance, ...]


class PipelineSucceeded(_PipelineTrace):
    status: Literal["succeeded"] = "succeeded"
    searcher_result: search.SearcherResult
    forecast_result: forecast.ForecastResult


class PipelineAbstained(_PipelineTrace):
    status: Literal["abstained"] = "abstained"
    reason: PipelineAbstentionReason
    detail: str = Field(min_length=1)


PipelineResult: TypeAlias = Annotated[
    PipelineSucceeded | PipelineAbstained,
    Field(discriminator="status"),
]
