"""Verified, separately invoked post-resolution finance analysis."""

from dataclasses import dataclass
from enum import StrEnum, unique
from pathlib import Path

from pydantic import ValidationError
from typing_extensions import override

from src.domain.finance.experiment_artifact import PersistedFinanceExperimentSuite
from src.domain.finance.experiment_resolution import FinanceResolutionManifest
from src.domain.finance.experiment_resolution_metrics import FinanceResolutionAnalysis
from src.domain.finance.sanitized_artifact import ArtifactSanitizationError
from src.services.finance_artifact_sanitizer import validate_persisted_artifact
from src.services.finance_experiment_artifact import (
    FinanceExperimentArtifactError,
    load_finance_experiment_bundle,
)
from src.services.finance_experiment_manifest import (
    FinanceManifestReadError,
    FinanceManifestValidationError,
    ResolutionBindingError,
    ResolutionBindingReason,
    load_finance_resolution_manifest,
)
from src.services.finance_hashed_bundle import (
    HashedBundleContent,
    HashedBundleError,
    HashedBundleReceipt,
    load_hashed_bundle,
    persist_hashed_bundle,
)
from src.services.finance_resolution_metrics import (
    binary_accuracy,
    binary_brier,
    build_finance_resolution_analysis,
)
from src.services.finance_resolution_reporting import render_resolution_report


@unique
class FinanceResolutionAnalysisFailure(StrEnum):
    SOURCE_BUNDLE_INVALID = "source_bundle_invalid"
    RESOLUTION_MANIFEST_INVALID = "resolution_manifest_invalid"
    BINDING_MISMATCH = "binding_mismatch"
    SANITIZATION_REJECTED = "sanitization_rejected"
    PERSISTENCE_FAILED = "persistence_failed"
    SCHEMA_INVALID = "schema_invalid"
    REPORT_MISMATCH = "report_mismatch"


class FinanceResolutionAnalysisError(Exception):
    __slots__ = ("failure_class", "reason")

    def __init__(
        self,
        reason: FinanceResolutionAnalysisFailure,
        failure_class: str,
    ) -> None:
        super().__init__(reason, failure_class)
        self.reason = reason
        self.failure_class = failure_class

    @override
    def __str__(self) -> str:
        return (
            "finance resolution analysis failed: "
            f"{self.reason.value} ({self.failure_class})"
        )


@dataclass(frozen=True, slots=True)
class VerifiedFinanceResolutionAnalysisBundle:
    analysis: FinanceResolutionAnalysis
    receipt: HashedBundleReceipt


def _bind_resolution(
    resolution: FinanceResolutionManifest,
    suite: PersistedFinanceExperimentSuite,
    verified_suite_sha256: str,
) -> None:
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


def analyze_finance_resolution(
    source_bundle: Path,
    resolution_manifest: Path,
    destination: Path,
) -> HashedBundleReceipt:
    """Verify ex-ante bytes, bind outcomes, and write one distinct derived bundle."""
    try:
        resolution = load_finance_resolution_manifest(resolution_manifest)
    except (FinanceManifestReadError, FinanceManifestValidationError) as error:
        raise FinanceResolutionAnalysisError(
            reason=FinanceResolutionAnalysisFailure.RESOLUTION_MANIFEST_INVALID,
            failure_class=type(error).__name__,
        ) from error
    return analyze_finance_resolution_manifest(source_bundle, resolution, destination)


def analyze_finance_resolution_manifest(
    source_bundle: Path,
    resolution: FinanceResolutionManifest,
    destination: Path,
) -> HashedBundleReceipt:
    """Analyze an in-memory outcome manifest loaded after suite completion."""
    try:
        verified = load_finance_experiment_bundle(source_bundle)
    except (FinanceExperimentArtifactError, ArtifactSanitizationError) as error:
        raise FinanceResolutionAnalysisError(
            reason=FinanceResolutionAnalysisFailure.SOURCE_BUNDLE_INVALID,
            failure_class=type(error).__name__,
        ) from error
    try:
        _bind_resolution(resolution, verified.suite, verified.receipt.data_sha256)
    except ResolutionBindingError as error:
        raise FinanceResolutionAnalysisError(
            reason=FinanceResolutionAnalysisFailure.BINDING_MISMATCH,
            failure_class=error.reason.value,
        ) from error
    analysis = build_finance_resolution_analysis(
        verified.suite,
        verified.receipt.data_sha256,
        resolution,
    )
    serialized = analysis.model_dump_json(indent=2) + "\n"
    try:
        validate_persisted_artifact(serialized)
    except ArtifactSanitizationError as error:
        raise FinanceResolutionAnalysisError(
            reason=FinanceResolutionAnalysisFailure.SANITIZATION_REJECTED,
            failure_class=type(error).__name__,
        ) from error
    content = HashedBundleContent(
        data_filename="analysis.json",
        data_bytes=serialized.encode("utf-8"),
        report_bytes=render_resolution_report(analysis).encode("utf-8"),
    )
    try:
        return persist_hashed_bundle(destination, content)
    except HashedBundleError as error:
        raise FinanceResolutionAnalysisError(
            reason=FinanceResolutionAnalysisFailure.PERSISTENCE_FAILED,
            failure_class=error.reason.value,
        ) from error


def load_finance_resolution_analysis_bundle(
    directory: Path,
) -> VerifiedFinanceResolutionAnalysisBundle:
    """Verify hashes, schema, sanitizer, and deterministic derived report."""
    try:
        verified = load_hashed_bundle(directory, "analysis.json")
    except HashedBundleError as error:
        raise FinanceResolutionAnalysisError(
            reason=FinanceResolutionAnalysisFailure.PERSISTENCE_FAILED,
            failure_class=error.reason.value,
        ) from error
    try:
        serialized = verified.data_bytes.decode("utf-8")
        validate_persisted_artifact(serialized)
        analysis = FinanceResolutionAnalysis.model_validate_json(serialized)
    except UnicodeError as error:
        raise FinanceResolutionAnalysisError(
            reason=FinanceResolutionAnalysisFailure.SCHEMA_INVALID,
            failure_class=type(error).__name__,
        ) from error
    except ArtifactSanitizationError as error:
        raise FinanceResolutionAnalysisError(
            reason=FinanceResolutionAnalysisFailure.SANITIZATION_REJECTED,
            failure_class=type(error).__name__,
        ) from error
    except ValidationError as error:
        raise FinanceResolutionAnalysisError(
            reason=FinanceResolutionAnalysisFailure.SCHEMA_INVALID,
            failure_class=type(error).__name__,
        ) from error
    expected_report = render_resolution_report(analysis).encode("utf-8")
    if verified.report_bytes != expected_report:
        raise FinanceResolutionAnalysisError(
            reason=FinanceResolutionAnalysisFailure.REPORT_MISMATCH,
            failure_class="DeterministicReportMismatch",
        )
    return VerifiedFinanceResolutionAnalysisBundle(
        analysis=analysis,
        receipt=verified.receipt,
    )


__all__ = [
    "FinanceResolutionAnalysisError",
    "FinanceResolutionAnalysisFailure",
    "VerifiedFinanceResolutionAnalysisBundle",
    "analyze_finance_resolution",
    "analyze_finance_resolution_manifest",
    "binary_accuracy",
    "binary_brier",
    "load_finance_resolution_analysis_bundle",
]
