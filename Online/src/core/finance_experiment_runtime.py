"""Runtime contracts for fixed-pack finance experiment trials."""

from dataclasses import dataclass
from enum import StrEnum, unique
from typing import Protocol
from uuid import UUID

from src.domain.finance.experiment import (
    FinanceBinaryQuestion,
    FinanceEvidenceSnapshot,
    FinanceExperimentManifest,
    FinanceExperimentTrial,
)
from src.domain.finance.memory import ResolvedDagEpisode
from src.domain.finance.retrieval import HistoricalDagQuery, HistoricalDagRetrieval


class SeedIdentityVerifier(Protocol):
    """Verify immutable DB metadata without deserializing episodes."""

    def verify_identity(self) -> None:
        """Fail closed when the configured immutable DB identity differs."""
        ...


class HistoricalRetriever(Protocol):
    """Pure historical-memory retrieval capability."""

    def retrieve_from(
        self,
        episodes: tuple[ResolvedDagEpisode, ...],
        query: HistoricalDagQuery,
    ) -> HistoricalDagRetrieval: ...


@dataclass(frozen=True, slots=True)
class FinanceExperimentRequestError(Exception):
    """Manifest-bound trial request is internally inconsistent."""

    detail: str

    def __str__(self) -> str:
        return f"invalid finance experiment trial request: {self.detail}"


@dataclass(frozen=True, slots=True)
class FinanceExperimentRetrievalError(Exception):
    """Expected retriever adapter failure without partial memory."""

    detail: str

    def __str__(self) -> str:
        return f"finance experiment retrieval unavailable: {self.detail}"


@unique
class FinanceExperimentStage(StrEnum):
    """Observable fixed-pack trial ordering."""

    METADATA_PREFLIGHT = "metadata_preflight"
    DIRECT = "direct"
    INITIAL_SEARCH = "initial_search"
    FREEZE = "freeze"
    SEARCH_ONLY = "search_only"
    MEMORY_RETRIEVAL = "memory_retrieval"
    SEARCH_DAG = "search_dag"


@dataclass(frozen=True, slots=True)
class FinanceExperimentTrialRequest:
    """One manifest-bound question repetition and stable trial identity."""

    manifest: FinanceExperimentManifest
    question: FinanceBinaryQuestion
    repetition_index: int
    trial_id: UUID


@dataclass(frozen=True, slots=True)
class FinanceExperimentTrialRun:
    """Terminal three-arm trial plus its fixed-pack execution trace."""

    trial: FinanceExperimentTrial
    stage_order: tuple[FinanceExperimentStage, ...]
    evidence_snapshot: FinanceEvidenceSnapshot | None


__all__ = [
    "FinanceExperimentRequestError",
    "FinanceExperimentRetrievalError",
    "FinanceExperimentStage",
    "FinanceExperimentTrialRequest",
    "FinanceExperimentTrialRun",
    "HistoricalRetriever",
    "SeedIdentityVerifier",
]
