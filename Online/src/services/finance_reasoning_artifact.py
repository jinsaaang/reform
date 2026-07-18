"""Durable JSON persistence for validated finance reasoning run artifacts."""

import os
from datetime import UTC, datetime
from pathlib import Path
from typing import final
from uuid import UUID, uuid4

from pydantic import ValidationError
from typing_extensions import override

from src.domain.finance.artifact import (
    FinanceReasoningRunArtifact,
    FinanceRunMetadata,
    ForecastArm,
)
from src.domain.finance.pipeline import PipelineResult


@final
class FinanceReasoningArtifactError(RuntimeError):
    """Typed artifact persistence or validation failure."""

    def __init__(self, detail: str) -> None:
        super().__init__(detail)
        self.detail: str = detail

    @override
    def __str__(self) -> str:
        return self.detail


def build_finance_reasoning_artifact(
    result: PipelineResult,
    *,
    arm: ForecastArm,
    forecast_model: str,
    reasoning_effort: str,
    created_at: datetime | None = None,
    run_id: UUID | None = None,
) -> FinanceReasoningRunArtifact:
    """Build one versioned envelope around the complete validated pipeline trace."""
    return FinanceReasoningRunArtifact(
        metadata=FinanceRunMetadata(
            run_id=run_id or uuid4(),
            created_at=created_at or datetime.now(UTC),
            arm=arm,
            forecast_model=forecast_model,
            reasoning_effort=reasoning_effort,
        ),
        pipeline_result=result,
    )


def save_finance_reasoning_artifact(
    output_dir: Path,
    result: PipelineResult,
    *,
    arm: ForecastArm,
    forecast_model: str,
    reasoning_effort: str,
) -> Path:
    """Atomically persist one complete reasoning artifact and return its path."""
    artifact = build_finance_reasoning_artifact(
        result,
        arm=arm,
        forecast_model=forecast_model,
        reasoning_effort=reasoning_effort,
    )
    run_id = artifact.metadata.run_id.hex
    timestamp = artifact.metadata.created_at.strftime("%Y%m%dT%H%M%S%fZ")
    directory = output_dir.expanduser().resolve()
    destination = directory / f"{timestamp}_{run_id}.json"
    temporary = directory / f".{run_id}.tmp"
    serialized = artifact.model_dump_json(indent=2) + "\n"
    try:
        directory.mkdir(parents=True, exist_ok=True)
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            _ = handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        _ = temporary.replace(destination)
    except OSError as error:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise FinanceReasoningArtifactError(
            f"failed to persist reasoning artifact: {type(error).__name__}"
        ) from error
    return destination


def load_finance_reasoning_artifact(path: Path) -> FinanceReasoningRunArtifact:
    """Load and validate an existing reasoning artifact."""
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise FinanceReasoningArtifactError(
            f"failed to read reasoning artifact: {type(error).__name__}"
        ) from error
    try:
        return FinanceReasoningRunArtifact.model_validate_json(payload)
    except ValidationError as error:
        raise FinanceReasoningArtifactError(
            "reasoning artifact failed schema validation"
        ) from error


__all__ = [
    "FinanceReasoningArtifactError",
    "build_finance_reasoning_artifact",
    "load_finance_reasoning_artifact",
    "save_finance_reasoning_artifact",
]
