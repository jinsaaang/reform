"""Outcome-only resolution manifests bound to verified ex-ante suites."""

from typing import Annotated, ClassVar, Literal, TypeAlias
from uuid import UUID

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator
from pydantic_core import PydanticCustomError

from src.domain.finance.memory import OutcomeLabel, QuestionId

Sha256Digest: TypeAlias = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class _ResolutionModel(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=True,
        str_strip_whitespace=True,
    )


class FinanceResolutionEntry(_ResolutionModel):
    """One resolved target with source and aware resolution time."""

    question_id: Annotated[QuestionId, Field(min_length=1)]
    outcome_label: Annotated[OutcomeLabel, Field(min_length=1)]
    resolved_at: AwareDatetime
    resolution_source: str = Field(min_length=1)


class FinanceResolutionManifest(_ResolutionModel):
    """Partial outcome set cryptographically bound to one ex-ante suite."""

    schema_version: Literal["finance-resolution-manifest/v1"]
    suite_id: UUID
    experiment_manifest_id: str = Field(min_length=1)
    suite_sha256: Sha256Digest
    entries: tuple[FinanceResolutionEntry, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_questions(self) -> "FinanceResolutionManifest":
        """Reject resolving one question/trial more than once."""
        question_ids = tuple(entry.question_id for entry in self.entries)
        if len(question_ids) != len(set(question_ids)):
            raise PydanticCustomError(
                "duplicate_resolution_question",
                "a question may appear only once in a resolution manifest",
            )
        return self


__all__ = [
    "FinanceResolutionEntry",
    "FinanceResolutionManifest",
]
