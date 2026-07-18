"""Batch orchestration for persisted three-arm finance experiments."""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from enum import IntEnum, unique
from pathlib import Path
from typing import Protocol, assert_never
from uuid import UUID

from pydantic import ValidationError

from src.core.finance_experiment_runtime import (
    HistoricalRetriever,
    SeedIdentityVerifier,
)
from src.domain.finance.experiment_artifact import (
    PersistedExperimentTrial,
    PersistedFinanceExperimentSuite,
)
from src.domain.finance.experiment_manifest import FinanceExperimentManifest
from src.domain.finance.experiment_results import FinanceSuiteStatus
from src.domain.finance import experiment_variant_guard
from src.domain.finance.retrieval import (
    DuplicateEpisodeIdError,
    HistoricalEpisodeSource,
    SeedAssetMismatchError,
    SeedDatabaseReadError,
    SeedDataError,
    SeedSchemaError,
    UnsafeSeedPathError,
    WritableSeedAssetError,
)
from src.domain.finance.sanitized_artifact import ArtifactSanitizationError
from src.services.finance_experiment_artifact import (
    FinanceExperimentArtifactError,
    ensure_bundle_destination_available,
    persist_finance_experiment_bundle,
)
from src.services.finance_experiment_contracts import (
    FinanceExperimentProviderBuildError,
    FinanceExperimentProviderBuilder,
)
from src.services.finance_experiment_trial import (
    FinanceExperimentTrialContext,
    run_persisted_experiment_trial,
)
from src.services.finance_hashed_bundle import HashedBundleReceipt
from src.services.finance_seed_identity import FinanceSeedSidecarError


class UUIDFactory(Protocol):
    def __call__(self) -> UUID: ...


class AliasSaltFactory(Protocol):
    def __call__(self) -> bytes: ...


@dataclass(frozen=True, slots=True)
class FinanceExperimentDependencies:
    provider_builder: FinanceExperimentProviderBuilder
    identity_verifier: SeedIdentityVerifier
    repository: HistoricalEpisodeSource
    retriever: HistoricalRetriever
    clock: Callable[[], datetime]
    monotonic_clock: Callable[[], float]
    uuid_factory: UUIDFactory
    alias_salt_factory: AliasSaltFactory


@dataclass(frozen=True, slots=True)
class FinanceExperimentRunRequest:
    manifest: FinanceExperimentManifest
    destination: Path


@unique
class FinanceExperimentExitCode(IntEnum):
    SUCCESS = 0
    PARTIAL_RECORDED = 1
    FAILED_RECORDED = 2
    PREFLIGHT_FAILED = 3
    PERSISTENCE_FAILED = 4


@dataclass(frozen=True, slots=True)
class FinanceExperimentRunResult:
    exit_code: FinanceExperimentExitCode
    suite_status: FinanceSuiteStatus | None
    recorded: bool
    destination: Path | None
    suite_id: UUID | None
    receipt: HashedBundleReceipt | None
    failure_class: str | None


def _suite_status(trials: tuple[PersistedExperimentTrial, ...]) -> FinanceSuiteStatus:
    success_count = sum(
        arm.status == "succeeded" for trial in trials for arm in trial.arms
    )
    unavailable_pair = any(
        pair.status == "unavailable" for trial in trials for pair in trial.pairs
    )
    terminal_count = sum(len(trial.arms) for trial in trials)
    if success_count == 0:
        return FinanceSuiteStatus.FAILED
    if success_count < terminal_count or unavailable_pair:
        return FinanceSuiteStatus.PARTIAL
    return FinanceSuiteStatus.COMPLETE


def _exit_code(status: FinanceSuiteStatus) -> FinanceExperimentExitCode:
    match experiment_variant_guard.runtime_closed_variant(status):
        case FinanceSuiteStatus.COMPLETE:
            return FinanceExperimentExitCode.SUCCESS
        case FinanceSuiteStatus.PARTIAL:
            return FinanceExperimentExitCode.PARTIAL_RECORDED
        case FinanceSuiteStatus.FAILED:
            return FinanceExperimentExitCode.FAILED_RECORDED
        case _ as unreachable:
            assert_never(
                experiment_variant_guard.unexpected_closed_variant(unreachable)
            )


_PREFLIGHT_ERRORS = (
    ValidationError,
    FinanceExperimentArtifactError,
    FinanceSeedSidecarError,
    DuplicateEpisodeIdError,
    SeedAssetMismatchError,
    SeedDatabaseReadError,
    SeedDataError,
    SeedSchemaError,
    UnsafeSeedPathError,
    WritableSeedAssetError,
)


def _preflight(
    request: FinanceExperimentRunRequest,
    dependencies: FinanceExperimentDependencies,
) -> FinanceExperimentManifest | None:
    try:
        manifest = FinanceExperimentManifest.model_validate_json(
            request.manifest.model_dump_json()
        )
        ensure_bundle_destination_available(request.destination)
        dependencies.identity_verifier.verify_identity()
    except _PREFLIGHT_ERRORS:
        return None
    return manifest


def _failed_result(
    exit_code: FinanceExperimentExitCode,
    failure_class: str,
    suite_id: UUID | None = None,
) -> FinanceExperimentRunResult:
    return FinanceExperimentRunResult(
        exit_code,
        None,
        False,
        None,
        suite_id,
        None,
        failure_class,
    )


def run_finance_experiment(
    request: FinanceExperimentRunRequest,
    dependencies: FinanceExperimentDependencies,
) -> FinanceExperimentRunResult:
    """Preflight, execute the ordered schedule, and atomically persist it."""
    manifest = _preflight(request, dependencies)
    if manifest is None:
        return _failed_result(
            FinanceExperimentExitCode.PREFLIGHT_FAILED,
            "PreflightRejected",
        )
    created_at = dependencies.clock()
    alias_salt = dependencies.alias_salt_factory()
    if (
        created_at.tzinfo is None
        or created_at.utcoffset() is None
        or len(alias_salt) != 32
    ):
        return _failed_result(
            FinanceExperimentExitCode.PREFLIGHT_FAILED,
            "InvalidInjectedBoundary",
        )
    suite_id = dependencies.uuid_factory()
    try:
        providers = dependencies.provider_builder(manifest)
    except FinanceExperimentProviderBuildError as error:
        return _failed_result(
            FinanceExperimentExitCode.PREFLIGHT_FAILED,
            type(error).__name__,
        )
    context = FinanceExperimentTrialContext(
        manifest=manifest,
        providers=providers,
        repository=dependencies.repository,
        retriever=dependencies.retriever,
        monotonic_clock=dependencies.monotonic_clock,
        uuid_factory=dependencies.uuid_factory,
        alias_salt=alias_salt,
    )
    trials = tuple(
        run_persisted_experiment_trial(context, question, repetition)
        for question in manifest.questions
        for repetition in range(manifest.repetitions)
    )
    status = _suite_status(trials)
    suite = PersistedFinanceExperimentSuite(
        suite_id=suite_id,
        created_at=created_at,
        status=status,
        manifest=manifest,
        trials=trials,
    )
    try:
        receipt = persist_finance_experiment_bundle(request.destination, suite)
    except (
        ArtifactSanitizationError,
        FinanceExperimentArtifactError,
        KeyboardInterrupt,
    ) as error:
        return _failed_result(
            FinanceExperimentExitCode.PERSISTENCE_FAILED,
            type(error).__name__,
            suite_id,
        )
    return FinanceExperimentRunResult(
        _exit_code(status),
        status,
        True,
        request.destination,
        suite_id,
        receipt,
        None,
    )


__all__ = [
    "FinanceExperimentDependencies",
    "FinanceExperimentExitCode",
    "FinanceExperimentRunRequest",
    "FinanceExperimentRunResult",
    "run_finance_experiment",
]
