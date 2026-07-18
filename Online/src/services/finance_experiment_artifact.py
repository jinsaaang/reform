"""Atomic persistence and verification for sanitized ex-ante suites."""

from dataclasses import dataclass
from enum import StrEnum, unique
from pathlib import Path
from typing import Final

from pydantic import ValidationError
from typing_extensions import override

from src.domain.finance.experiment_artifact import PersistedFinanceExperimentSuite
from src.domain.finance.sanitized_artifact import TransientForbiddenValueRegistry
from src.services.finance_artifact_sanitizer import validate_persisted_artifact
from src.services.finance_experiment_reporting import render_ex_ante_report
from src.services.finance_hashed_bundle import (
    HashedBundleContent,
    HashedBundleError,
    HashedBundleFailure,
    HashedBundleReceipt,
    ensure_new_bundle_destination,
    load_hashed_bundle,
    persist_hashed_bundle,
)


@unique
class FinanceExperimentArtifactFailure(StrEnum):
    DESTINATION_EXISTS = "destination_exists"
    PARENT_UNAVAILABLE = "parent_unavailable"
    WRITE_FAILED = "write_failed"
    READ_FAILED = "read_failed"
    INVALID_LAYOUT = "invalid_layout"
    HASH_MISMATCH = "hash_mismatch"
    SCHEMA_INVALID = "schema_invalid"
    REPORT_MISMATCH = "report_mismatch"
    CLEANUP_FAILED = "cleanup_failed"


_FAILURE_MAPPING: Final = {
    HashedBundleFailure.DESTINATION_EXISTS: (
        FinanceExperimentArtifactFailure.DESTINATION_EXISTS
    ),
    HashedBundleFailure.PARENT_UNAVAILABLE: (
        FinanceExperimentArtifactFailure.PARENT_UNAVAILABLE
    ),
    HashedBundleFailure.WRITE_FAILED: FinanceExperimentArtifactFailure.WRITE_FAILED,
    HashedBundleFailure.READ_FAILED: FinanceExperimentArtifactFailure.READ_FAILED,
    HashedBundleFailure.INVALID_LAYOUT: (
        FinanceExperimentArtifactFailure.INVALID_LAYOUT
    ),
    HashedBundleFailure.HASH_MISMATCH: (FinanceExperimentArtifactFailure.HASH_MISMATCH),
    HashedBundleFailure.CLEANUP_FAILED: (
        FinanceExperimentArtifactFailure.CLEANUP_FAILED
    ),
}


class FinanceExperimentArtifactError(Exception):
    __slots__ = ("failure_class", "path", "reason")

    def __init__(
        self,
        reason: FinanceExperimentArtifactFailure,
        path: Path,
        failure_class: str | None = None,
    ) -> None:
        super().__init__(reason, path, failure_class)
        self.reason = reason
        self.path = path
        self.failure_class = failure_class

    @override
    def __str__(self) -> str:
        suffix = f" ({self.failure_class})" if self.failure_class else ""
        return f"finance experiment artifact failed: {self.reason.value}{suffix}"


@dataclass(frozen=True, slots=True)
class VerifiedFinanceExperimentBundle:
    suite: PersistedFinanceExperimentSuite
    receipt: HashedBundleReceipt


def _translate(error: HashedBundleError) -> FinanceExperimentArtifactError:
    return FinanceExperimentArtifactError(
        reason=_FAILURE_MAPPING[error.reason],
        path=error.path,
        failure_class=error.failure_class,
    )


def ensure_bundle_destination_available(destination: Path) -> None:
    """Reject existing destinations and parents that cannot contain a bundle."""
    try:
        ensure_new_bundle_destination(destination)
    except HashedBundleError as error:
        raise _translate(error) from error


def persist_finance_experiment_bundle(
    destination: Path,
    suite: PersistedFinanceExperimentSuite,
    *,
    registry: TransientForbiddenValueRegistry | None = None,
) -> HashedBundleReceipt:
    """Validate, render, hash, fsync, and atomically publish one ex-ante bundle."""
    serialized = suite.model_dump_json(indent=2) + "\n"
    validate_persisted_artifact(serialized, registry)
    content = HashedBundleContent(
        data_filename="suite.json",
        data_bytes=serialized.encode("utf-8"),
        report_bytes=render_ex_ante_report(suite).encode("utf-8"),
    )
    try:
        return persist_hashed_bundle(destination, content)
    except HashedBundleError as error:
        raise _translate(error) from error


def load_finance_experiment_bundle(
    directory: Path,
) -> VerifiedFinanceExperimentBundle:
    """Hash-check, sanitize, schema-load, and reproduce one ex-ante report."""
    try:
        verified = load_hashed_bundle(directory, "suite.json")
    except HashedBundleError as error:
        raise _translate(error) from error
    try:
        serialized = verified.data_bytes.decode("utf-8")
    except UnicodeError as error:
        raise FinanceExperimentArtifactError(
            reason=FinanceExperimentArtifactFailure.SCHEMA_INVALID,
            path=directory / "suite.json",
            failure_class=type(error).__name__,
        ) from error
    validate_persisted_artifact(serialized)
    try:
        suite = PersistedFinanceExperimentSuite.model_validate_json(serialized)
    except ValidationError as error:
        raise FinanceExperimentArtifactError(
            reason=FinanceExperimentArtifactFailure.SCHEMA_INVALID,
            path=directory / "suite.json",
            failure_class=type(error).__name__,
        ) from error
    expected_report = render_ex_ante_report(suite).encode("utf-8")
    if verified.report_bytes != expected_report:
        raise FinanceExperimentArtifactError(
            reason=FinanceExperimentArtifactFailure.REPORT_MISMATCH,
            path=directory / "report.md",
        )
    return VerifiedFinanceExperimentBundle(suite=suite, receipt=verified.receipt)


__all__ = [
    "FinanceExperimentArtifactError",
    "FinanceExperimentArtifactFailure",
    "VerifiedFinanceExperimentBundle",
    "ensure_bundle_destination_available",
    "load_finance_experiment_bundle",
    "persist_finance_experiment_bundle",
]
