"""Identity-blind views shared by judging and persisted reasoning."""

from enum import StrEnum, unique
from typing import Annotated, ClassVar, Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

from src.domain.finance.forecast import OutcomeProbability, Probability
from src.domain.finance.memory import HistoricalOutcomeValue, QuestionKind
from src.domain.finance.sanitized_artifact import FinanceAliasAudit
from src.domain.finance.search import EvidenceDirection


class _JudgeViewModel(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=True,
        str_strip_whitespace=True,
    )


@unique
class NeutralCandidate(StrEnum):
    """Order-independent candidate identities with no treatment meaning."""

    CANDIDATE_1 = "candidate_1"
    CANDIDATE_2 = "candidate_2"


class SanitizedTargetView(_JudgeViewModel):
    """Current target without stable identity or realized outcome."""

    question_text: str = Field(min_length=1)
    question_type: QuestionKind
    domain: str = Field(min_length=1)
    context: tuple[str, ...]
    cutoff: AwareDatetime
    outcome_space: tuple[str, ...] = Field(min_length=1)
    resolution_rule: str | None


class SanitizedEvidenceView(_JudgeViewModel):
    """Citation-bearing evidence without source identity hashes."""

    alias: str = Field(min_length=1)
    claim: str = Field(min_length=1)
    citation: str = Field(min_length=1)
    available_at: AwareDatetime
    retrieved_at: AwareDatetime
    direction: EvidenceDirection
    context_slot: str = Field(min_length=1)


class SanitizedHistoricalQuestionView(_JudgeViewModel):
    """Public historical target description without its stable ID."""

    question_text: str = Field(min_length=1)
    question_type: QuestionKind
    domain: str = Field(min_length=1)
    source: str = Field(min_length=1)
    context: str | None
    outcome_space: tuple[str, ...]
    resolution_rule: str | None
    quantity_unit: str | None


class SanitizedMemoryNodeView(_JudgeViewModel):
    alias: str = Field(min_length=1)
    title: str
    description: str
    event_type: str
    domain: str
    tags: tuple[str, ...]
    occurred_at: AwareDatetime | None
    predicted_at: AwareDatetime | None
    source_citation_aliases: tuple[str, ...]
    is_outcome: bool
    outcome_scenario: str | None
    is_actual_outcome: bool | None


class SanitizedMemoryEdgeView(_JudgeViewModel):
    alias: str = Field(min_length=1)
    source_node_alias: str = Field(min_length=1)
    target_node_alias: str = Field(min_length=1)
    relation_type: str | None
    strength: float
    confidence: float
    time_lag_hours: float | None
    reasoning: str
    evidence_citation_aliases: tuple[str, ...]


class SanitizedMemoryImpactView(_JudgeViewModel):
    alias: str = Field(min_length=1)
    event_alias: str = Field(min_length=1)
    outcome_event_alias: str = Field(min_length=1)
    direction: str
    magnitude: float
    confidence: float
    reasoning: str
    evidence_citation_aliases: tuple[str, ...]
    causal_edge_aliases: tuple[str, ...]


class SanitizedMemoryView(_JudgeViewModel):
    """Retriever-ranked, bounded historical episode without dangling links."""

    alias: str = Field(min_length=1)
    question: SanitizedHistoricalQuestionView
    resolution_date: AwareDatetime
    resolved_historical_outcome: HistoricalOutcomeValue
    nodes: tuple[SanitizedMemoryNodeView, ...]
    edges: tuple[SanitizedMemoryEdgeView, ...]
    impacts: tuple[SanitizedMemoryImpactView, ...]
    omitted_node_count: Annotated[int, Field(ge=0)]
    omitted_edge_count: Annotated[int, Field(ge=0)]
    omitted_impact_count: Annotated[int, Field(ge=0)]


class SanitizedScenarioView(_JudgeViewModel):
    alias: str = Field(min_length=1)
    name: str = Field(min_length=1)
    reasoning_steps: tuple[str, ...] = Field(min_length=1)
    probability: Probability
    conditional_outcomes: tuple[OutcomeProbability, ...] = Field(min_length=1)
    evidence_aliases: tuple[str, ...]
    memory_aliases: tuple[str, ...]
    assumptions: tuple[str, ...]
    triggers: tuple[str, ...]
    disconfirmers: tuple[str, ...]
    uncertainty: str = Field(min_length=1)


class SanitizedCandidateView(_JudgeViewModel):
    """Complete public forecast reasoning under a neutral candidate alias."""

    alias: NeutralCandidate
    scenarios: tuple[SanitizedScenarioView, ...] = Field(min_length=1)
    outcome_probabilities: tuple[OutcomeProbability, ...] = Field(min_length=1)
    explanation: str = Field(min_length=1)


class SanitizedJudgePair(_JudgeViewModel):
    """Persistence-safe canonical pair plus non-reversible alias audit."""

    schema_version: Literal["finance-sanitized-reasoning/v1"] = (
        "finance-sanitized-reasoning/v1"
    )
    target: SanitizedTargetView
    evidence: tuple[SanitizedEvidenceView, ...]
    memory: tuple[SanitizedMemoryView, ...]
    candidate_1: SanitizedCandidateView
    candidate_2: SanitizedCandidateView
    alias_audit: FinanceAliasAudit


class BlindJudgePayload(_JudgeViewModel):
    """Exact identity-blind payload sent to one ordered judge call."""

    schema_version: Literal["finance-blind-judge-payload/v1"] = (
        "finance-blind-judge-payload/v1"
    )
    target: SanitizedTargetView
    evidence: tuple[SanitizedEvidenceView, ...]
    memory: tuple[SanitizedMemoryView, ...]
    answer_a: SanitizedCandidateView
    answer_b: SanitizedCandidateView


__all__ = [
    "BlindJudgePayload",
    "NeutralCandidate",
    "SanitizedCandidateView",
    "SanitizedEvidenceView",
    "SanitizedHistoricalQuestionView",
    "SanitizedJudgePair",
    "SanitizedMemoryEdgeView",
    "SanitizedMemoryImpactView",
    "SanitizedMemoryNodeView",
    "SanitizedMemoryView",
    "SanitizedScenarioView",
    "SanitizedTargetView",
]
