"""Versioned, immutable artifacts for auditable finance forecast runs."""

from datetime import datetime
from enum import StrEnum, unique
from typing import ClassVar, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_core import PydanticCustomError

from src.domain.finance.pipeline import PipelineResult


@unique
class ForecastArm(StrEnum):
    """Experimental condition that produced one forecast result."""

    DIRECT = "direct"
    SEARCH_ONLY = "search_only"
    SEARCH_DAG = "search_dag"


class _ArtifactModel(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=True,
        str_strip_whitespace=True,
    )


class FinanceRunMetadata(_ArtifactModel):
    """Reproducibility metadata that contains no provider credentials."""

    run_id: UUID
    created_at: datetime
    arm: ForecastArm
    forecast_model: str = Field(min_length=1)
    reasoning_effort: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_created_at_is_aware(self) -> "FinanceRunMetadata":
        """Reject ambiguous local timestamps at the persistence boundary."""
        if self.created_at.tzinfo is None or self.created_at.utcoffset() is None:
            raise PydanticCustomError(
                "naive_created_at",
                "artifact creation timestamp must include a timezone",
            )
        return self


class FinanceReasoningRunArtifact(_ArtifactModel):
    """Full public reasoning trace plus the metadata needed to compare arms."""

    schema_version: Literal["finance-reasoning-run/v1"] = "finance-reasoning-run/v1"
    metadata: FinanceRunMetadata
    pipeline_result: PipelineResult


__all__ = [
    "FinanceReasoningRunArtifact",
    "FinanceRunMetadata",
    "ForecastArm",
]
