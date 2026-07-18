"""Immutable values for resolved historical finance DAG memory."""

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum, unique
from typing import Annotated, ClassVar, NewType, TypeAlias

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, Json

EpisodeId = NewType("EpisodeId", str)
QuestionId = NewType("QuestionId", str)
DagId = NewType("DagId", str)
EventId = NewType("EventId", str)
CausalEdgeId = NewType("CausalEdgeId", str)
ImpactId = NewType("ImpactId", str)
ArticleId = NewType("ArticleId", str)
EvidenceId = NewType("EvidenceId", str)
ScenarioId = NewType("ScenarioId", str)
OutcomeLabel = NewType("OutcomeLabel", str)
HistoricalOutcomeValue: TypeAlias = bool | int | float | str
NonEmptySeedText: TypeAlias = Annotated[str, Field(min_length=1)]


@unique
class QuestionKind(StrEnum):
    """Closed set of forecast question shapes in the public seed."""

    BINARY = "binary"
    MCQ = "mcq"
    QUANTITY = "quantity"
    TIMEFRAME = "timeframe"


@unique
class ResolutionProvenance(StrEnum):
    """Provenance of the historical resolution eligibility timestamp."""

    BOOTSTRAP_RESOLUTION_DATE_PROXY = "BOOTSTRAP_RESOLUTION_DATE_PROXY"


@unique
class RelationState(StrEnum):
    """Three-valued state for candidate-to-target event relations."""

    KNOWN_TRUE = "known_true"
    KNOWN_FALSE = "known_false"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True, slots=True)
class HistoricalQuestionProfile:
    """Source-question features preserved without historical resolution data."""

    question_text: str
    question_type: QuestionKind
    domain: str
    source: str
    context: str | None
    outcome_space: tuple[str, ...]
    resolution_rule: str | None
    quantity_unit: str | None


@dataclass(frozen=True, slots=True)
class HistoricalResolutionMetadata:
    """Explicitly proxied historical resolution metadata."""

    resolution_date_proxy: datetime
    provenance: ResolutionProvenance


@dataclass(frozen=True, slots=True)
class HistoricalOutcomeMetadata:
    """Resolved value namespaced to a historical memory episode."""

    value: HistoricalOutcomeValue
    outcome_event_ids: tuple[EventId, ...]


@dataclass(frozen=True, slots=True)
class HistoricalEventNode:
    """Raw immutable event-node fields exposed by the public database."""

    event_id: EventId
    title: str
    description: str
    event_type: str
    domain: str
    tags: tuple[str, ...]
    occurred_at: datetime | None
    predicted_at: datetime | None
    source_article_ids: tuple[ArticleId, ...]
    is_outcome: bool
    outcome_scenario: str | None
    is_actual_outcome: bool | None


@dataclass(frozen=True, slots=True)
class HistoricalCausalEdge:
    """Raw immutable causal-hypothesis fields for a historical DAG."""

    edge_id: CausalEdgeId
    source_event_id: EventId
    target_event_id: EventId
    relation_type: str | None
    strength: float
    confidence: float
    time_lag_hours: float | None
    reasoning: str
    evidence_article_ids: tuple[ArticleId, ...]


@dataclass(frozen=True, slots=True)
class HistoricalImpact:
    """Raw immutable event-to-outcome impact fields for one episode."""

    impact_id: ImpactId
    event_id: EventId
    outcome_event_id: EventId
    direction: str
    magnitude: float
    confidence: float
    reasoning: str
    evidence_article_ids: tuple[ArticleId, ...]
    causal_edge_ids: tuple[CausalEdgeId, ...]


@dataclass(frozen=True, slots=True)
class EpisodeRelationMetadata:
    """Candidate-to-target hard-exclusion metadata without boolean collapse."""

    same_underlying_event: RelationState
    shared_resolution: RelationState
    derived_question: RelationState
    near_duplicate: RelationState

    @classmethod
    def public_db_unavailable(cls) -> "EpisodeRelationMetadata":
        """Represent the public release's absent relation flags truthfully."""
        return cls(
            same_underlying_event=RelationState.UNAVAILABLE,
            shared_resolution=RelationState.UNAVAILABLE,
            derived_question=RelationState.UNAVAILABLE,
            near_duplicate=RelationState.UNAVAILABLE,
        )


@dataclass(frozen=True, slots=True)
class HistoricalDagReference:
    """Stable identity-only reference to an already-built historical DAG."""

    episode_id: EpisodeId
    question_id: QuestionId
    dag_id: DagId


@dataclass(frozen=True, slots=True)
class ResolvedDagEpisode:
    """Full immutable historical episode reconstructed from raw DAG records."""

    episode_id: EpisodeId
    question_id: QuestionId
    dag_id: DagId
    source_profile: HistoricalQuestionProfile
    historical_resolution: HistoricalResolutionMetadata
    historical_outcome: HistoricalOutcomeMetadata
    nodes: tuple[HistoricalEventNode, ...]
    edges: tuple[HistoricalCausalEdge, ...]
    impacts: tuple[HistoricalImpact, ...]
    relation_metadata: EpisodeRelationMetadata

    @property
    def reference(self) -> HistoricalDagReference:
        """Return a stable non-graph identity reference."""
        return HistoricalDagReference(
            episode_id=self.episode_id,
            question_id=self.question_id,
            dag_id=self.dag_id,
        )


class SeedBoundaryRow(BaseModel):
    """Frozen validated base for rows crossing the immutable SQLite boundary."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")


class SeedQuestionRow(SeedBoundaryRow):
    question_id: NonEmptySeedText
    question_text: str
    question_type: QuestionKind
    domain: str
    source: str
    context: str | None
    resolution_date: AwareDatetime
    ground_truth: Json[HistoricalOutcomeValue]
    resolution_rule: str | None
    options: Json[tuple[str, ...]] | None
    quantity_unit: str | None
    outcome_event_ids: Json[tuple[str, ...]]


class SeedEventRow(SeedBoundaryRow):
    question_id: NonEmptySeedText
    event_id: NonEmptySeedText
    title: str
    description: str
    event_type: str
    domain: str
    tags: Json[tuple[str, ...]] | None
    occurred_at: AwareDatetime | None
    predicted_at: AwareDatetime | None
    article_ids: Json[tuple[str, ...]] | None
    is_outcome: bool
    outcome_scenario: str | None
    is_actual_outcome: bool | None


class SeedCausalEdgeRow(SeedBoundaryRow):
    question_id: NonEmptySeedText
    edge_id: NonEmptySeedText
    source_event_id: NonEmptySeedText
    target_event_id: NonEmptySeedText
    relation_type: str | None
    strength: float
    confidence: float
    time_lag_hours: float | None
    reasoning: str
    evidence_article_ids: Json[tuple[str, ...]] | None


class SeedImpactRow(SeedBoundaryRow):
    question_id: NonEmptySeedText
    impact_id: NonEmptySeedText
    event_id: NonEmptySeedText
    outcome_event_id: NonEmptySeedText
    direction: str
    magnitude: float
    confidence: float
    reasoning: str
    evidence_article_ids: Json[tuple[str, ...]] | None
    causal_edge_ids: Json[tuple[str, ...]] | None


class SeedSchemaVersionRow(SeedBoundaryRow):
    version: int


class SeedSchemaNameRow(SeedBoundaryRow):
    name: str
