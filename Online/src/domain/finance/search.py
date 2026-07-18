"""Sanitized current-target and flat Searcher boundary models."""

from enum import StrEnum, unique
from typing import Annotated, ClassVar

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator
from pydantic_core import PydanticCustomError

from src.domain.finance.memory import (
    EvidenceId,
    HistoricalDagReference,
    QuestionId,
    QuestionKind,
    ResolvedDagEpisode,
)


@unique
class EvidenceDirection(StrEnum):
    """Direction of an admitted evidence claim relative to the target."""

    SUPPORTS = "supports"
    WEAKENS = "weakens"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class TargetProfile(BaseModel):
    """Sanitized current target; hindsight and current DAGs are unrepresentable."""

    model_config: ClassVar[ConfigDict] = ConfigDict(
        frozen=True,
        extra="forbid",
        str_strip_whitespace=True,
    )

    question_id: Annotated[QuestionId, Field(min_length=1)]
    question_text: str = Field(min_length=1)
    question_type: QuestionKind
    domain: str = Field(min_length=1)
    context: tuple[str, ...] = ()
    cutoff: AwareDatetime
    outcome_space: tuple[str, ...] = Field(min_length=1)
    resolution_rule: str | None = None


class EvidenceItem(BaseModel):
    """One admitted flat claim with reproducible timing and provenance."""

    model_config: ClassVar[ConfigDict] = ConfigDict(
        frozen=True,
        extra="forbid",
        str_strip_whitespace=True,
    )

    evidence_id: Annotated[EvidenceId, Field(min_length=1)]
    claim: str = Field(min_length=1)
    citation: str = Field(min_length=1)
    available_at: AwareDatetime
    retrieved_at: AwareDatetime
    content_hash: str = Field(min_length=1)
    direction: EvidenceDirection
    context_slot: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_timing(self) -> "EvidenceItem":
        """Require retrieval to occur no earlier than evidence availability."""
        if self.retrieved_at < self.available_at:
            raise PydanticCustomError(
                "evidence_timing",
                "retrieved_at must not precede available_at",
            )
        return self


class EvidencePack(BaseModel):
    """Flat admitted evidence plus identity-only historical DAG references."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    items: tuple[EvidenceItem, ...]
    historical_dag_references: tuple[HistoricalDagReference, ...]

    @model_validator(mode="after")
    def validate_unique_ids(self) -> "EvidencePack":
        """Reject ambiguous duplicate evidence and historical references."""
        evidence_ids = tuple(item.evidence_id for item in self.items)
        if len(evidence_ids) != len(set(evidence_ids)):
            raise PydanticCustomError(
                "duplicate_evidence_id",
                "evidence IDs must be unique",
            )
        dag_ids = tuple(
            reference.dag_id for reference in self.historical_dag_references
        )
        if len(dag_ids) != len(set(dag_ids)):
            raise PydanticCustomError(
                "duplicate_historical_dag_id",
                "historical DAG references must be unique",
            )
        return self


class SearcherInput(BaseModel):
    """Contract for a Searcher that consumes, but does not build, history."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    target_profile: TargetProfile
    retrieved_historical_memory: tuple[ResolvedDagEpisode, ...]


class SearcherResult(BaseModel):
    """Searcher output with no probability or preferred-outcome surface."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    target_profile: TargetProfile
    evidence_pack: EvidencePack
    historical_dag_references: tuple[HistoricalDagReference, ...]
    limitations: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_historical_references(self) -> "SearcherResult":
        """Require the ordered reference surface to match the evidence pack."""
        if (
            self.historical_dag_references
            != self.evidence_pack.historical_dag_references
        ):
            raise PydanticCustomError(
                "inconsistent_historical_dag_references",
                "Searcher result references must match its evidence pack",
            )
        return self
