"""Filesystem boundaries and suite binding for finance experiment manifests."""

from dataclasses import dataclass
from enum import StrEnum, unique
from pathlib import Path

from pydantic import ValidationError
from typing_extensions import override

from src.domain.finance.experiment import (
    FinanceExperimentManifest,
    FinanceExperimentSuite,
    FinanceResolutionManifest,
)


@dataclass(frozen=True, slots=True)
class FinanceManifestReadError(Exception):
    """A manifest file could not be read as UTF-8 text."""

    path: Path
    failure_class: str

    @override
    def __str__(self) -> str:
        return f"unable to read finance manifest {self.path}: {self.failure_class}"


@dataclass(frozen=True, slots=True)
class FinanceManifestValidationError(Exception):
    """A manifest failed its strict Pydantic boundary."""

    path: Path
    error_count: int

    @override
    def __str__(self) -> str:
        return f"invalid finance manifest {self.path}: {self.error_count} error(s)"


@unique
class ResolutionBindingReason(StrEnum):
    """Closed failures checked before post-resolution output is started."""

    SUITE_ID_MISMATCH = "suite_id_mismatch"
    MANIFEST_ID_MISMATCH = "manifest_id_mismatch"
    SUITE_HASH_MISMATCH = "suite_hash_mismatch"
    UNKNOWN_QUESTION = "unknown_question"
    INVALID_OUTCOME_LABEL = "invalid_outcome_label"


@dataclass(frozen=True, slots=True)
class ResolutionBindingError(Exception):
    """A resolution manifest does not match its verified ex-ante suite."""

    reason: ResolutionBindingReason

    @override
    def __str__(self) -> str:
        return f"resolution manifest binding failed: {self.reason.value}"


def load_finance_experiment_manifest(path: Path) -> FinanceExperimentManifest:
    """Read and validate one strict experiment manifest without side effects."""
    try:
        payload = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise FinanceManifestReadError(path, type(error).__name__) from error
    try:
        return FinanceExperimentManifest.model_validate_json(payload)
    except ValidationError as error:
        raise FinanceManifestValidationError(path, error.error_count()) from error


def load_finance_resolution_manifest(path: Path) -> FinanceResolutionManifest:
    """Read and validate one strict outcome-only resolution manifest."""
    try:
        payload = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise FinanceManifestReadError(path, type(error).__name__) from error
    try:
        return FinanceResolutionManifest.model_validate_json(payload)
    except ValidationError as error:
        raise FinanceManifestValidationError(path, error.error_count()) from error


def bind_resolution_manifest(
    resolution: FinanceResolutionManifest,
    suite: FinanceExperimentSuite,
    verified_suite_sha256: str,
) -> FinanceResolutionManifest:
    """Prove suite identity, hash, question membership, and outcome labels."""
    if resolution.suite_id != suite.suite_id:
        raise ResolutionBindingError(ResolutionBindingReason.SUITE_ID_MISMATCH)
    if resolution.experiment_manifest_id != suite.manifest.manifest_id:
        raise ResolutionBindingError(ResolutionBindingReason.MANIFEST_ID_MISMATCH)
    if resolution.suite_sha256 != verified_suite_sha256:
        raise ResolutionBindingError(ResolutionBindingReason.SUITE_HASH_MISMATCH)
    targets = {question.question_id: question for question in suite.manifest.questions}
    for entry in resolution.entries:
        target = targets.get(entry.question_id)
        if target is None:
            raise ResolutionBindingError(ResolutionBindingReason.UNKNOWN_QUESTION)
        if entry.outcome_label not in target.outcome_space:
            raise ResolutionBindingError(ResolutionBindingReason.INVALID_OUTCOME_LABEL)
    return resolution


__all__ = [
    "FinanceManifestReadError",
    "FinanceManifestValidationError",
    "ResolutionBindingError",
    "ResolutionBindingReason",
    "bind_resolution_manifest",
    "load_finance_experiment_manifest",
    "load_finance_resolution_manifest",
]
