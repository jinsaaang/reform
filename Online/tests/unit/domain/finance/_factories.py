"""Small real-value factories shared by finance contract tests."""

from datetime import datetime, timezone

from src.domain.finance.memory import (
    CausalEdgeId,
    DagId,
    EpisodeId,
    EpisodeRelationMetadata,
    EventId,
    HistoricalCausalEdge,
    HistoricalEventNode,
    HistoricalOutcomeMetadata,
    HistoricalQuestionProfile,
    HistoricalResolutionMetadata,
    QuestionId,
    QuestionKind,
    RelationState,
    ResolutionProvenance,
    ResolvedDagEpisode,
)


def make_episode(
    episode_id: str = "episode-alpha",
    question_text: str = "Will semiconductor earnings beat expectations?",
    context: str | None = "GPU demand and semiconductor revenue",
    resolution_year: int = 2024,
    relations: EpisodeRelationMetadata | None = None,
) -> ResolvedDagEpisode:
    """Build a compact immutable historical episode with real domain values."""
    question_id = QuestionId(f"question-{episode_id}")
    node_id = EventId(f"event-{episode_id}")
    return ResolvedDagEpisode(
        episode_id=EpisodeId(episode_id),
        question_id=question_id,
        dag_id=DagId(f"dag-{episode_id}"),
        source_profile=HistoricalQuestionProfile(
            question_text=question_text,
            question_type=QuestionKind.BINARY,
            domain="finance",
            source="fixture",
            context=context,
            outcome_space=("Yes", "No"),
            resolution_rule="Resolve from the official quarterly filing.",
            quantity_unit=None,
        ),
        historical_resolution=HistoricalResolutionMetadata(
            resolution_date_proxy=datetime(
                resolution_year,
                1,
                1,
                tzinfo=timezone.utc,
            ),
            provenance=ResolutionProvenance.BOOTSTRAP_RESOLUTION_DATE_PROXY,
        ),
        historical_outcome=HistoricalOutcomeMetadata(
            value="Yes",
            outcome_event_ids=(EventId(f"outcome-{episode_id}"),),
        ),
        nodes=(
            HistoricalEventNode(
                event_id=node_id,
                title=question_text,
                description=context or "No historical context",
                event_type="financial",
                domain="finance",
                tags=("finance",),
                occurred_at=None,
                predicted_at=None,
                source_article_ids=(),
                is_outcome=False,
                outcome_scenario=None,
                is_actual_outcome=None,
            ),
        ),
        edges=(
            HistoricalCausalEdge(
                edge_id=CausalEdgeId(f"edge-{episode_id}"),
                source_event_id=node_id,
                target_event_id=EventId(f"outcome-{episode_id}"),
                relation_type="influences",
                strength=0.7,
                confidence=0.8,
                time_lag_hours=None,
                reasoning=context or "Unspecified mechanism",
                evidence_article_ids=(),
            ),
        ),
        impacts=(),
        relation_metadata=relations
        or EpisodeRelationMetadata(
            same_underlying_event=RelationState.KNOWN_FALSE,
            shared_resolution=RelationState.KNOWN_FALSE,
            derived_question=RelationState.KNOWN_FALSE,
            near_duplicate=RelationState.KNOWN_FALSE,
        ),
    )
