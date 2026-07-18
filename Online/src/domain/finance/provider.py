"""Frozen Search provider DTOs and untrusted candidate parsing."""

from dataclasses import dataclass
from enum import StrEnum, unique
from hashlib import sha256
from typing import Annotated, ClassVar, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field

from src.domain.finance import memory, search


@unique
class SearchPassKind(StrEnum):
    INITIAL = "initial"
    GUIDED = "guided"


@unique
class SearchSourcePolicy(StrEnum):
    OFFLINE_FIXTURE = "offline_fixture"
    LIVE_SEARCH = "live_search"
    PUBLIC_DB_METADATA = "public_db_metadata"


@unique
class FinanceRunMode(StrEnum):
    CURRENT_UNRESOLVED = "current_unresolved"
    HISTORICAL_BACKTEST = "historical_backtest"


@unique
class SearchQueryIntent(StrEnum):
    DIRECT_TARGET = "direct_target"
    SOURCE_OF_RECORD = "source_of_record"
    OPEN_WORLD = "open_world"
    RESOLUTION_RULE = "resolution_rule"
    HISTORICAL_GAP = "historical_gap"
    COUNTEREVIDENCE = "counterevidence"


@unique
class SearchPassFailureReason(StrEnum):
    PROVIDER_ERROR = "provider_error"
    MALFORMED_PROVIDER_OUTPUT = "malformed_provider_output"


@unique
class EvidenceAuditReason(StrEnum):
    ADMITTED = "admitted"
    MALFORMED_CANDIDATE = "malformed_candidate"
    MISSING_CORE_FIELD = "missing_core_field"
    MISSING_CANONICAL_SOURCE = "missing_canonical_source"
    MISSING_BODY_SNAPSHOT = "missing_body_snapshot"
    MISSING_SNAPSHOT_PROVENANCE = "missing_snapshot_provenance"
    INVALID_BODY_SNAPSHOT = "invalid_body_snapshot"
    MISSING_CONTENT_HASH = "missing_content_hash"
    INVALID_CONTENT_HASH = "invalid_content_hash"
    CONTENT_HASH_MISMATCH = "content_hash_mismatch"
    MISSING_EXACT_BODY = "missing_exact_body"
    MISSING_SOURCE_VERSION = "missing_source_version"
    MALFORMED_TIMESTAMP = "malformed_timestamp"
    RETRIEVAL_PRECEDES_AVAILABILITY = "retrieval_precedes_availability"
    NOT_STRICTLY_BEFORE_CUTOFF = "not_strictly_before_cutoff"
    MALFORMED_SNAPSHOT_TIMESTAMP = "malformed_snapshot_timestamp"
    SNAPSHOT_AVAILABILITY_PRECEDES_CAPTURE = "snapshot_availability_precedes_capture"
    SNAPSHOT_NOT_STRICTLY_BEFORE_CUTOFF = "snapshot_not_strictly_before_cutoff"
    DUPLICATE = "duplicate"


class _ProviderModel(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(
        frozen=True,
        extra="forbid",
        str_strip_whitespace=True,
    )


class HistoricalSearchGuidance(_ProviderModel):
    reference: memory.HistoricalDagReference
    matched_terms: tuple[str, ...]
    mechanism_hints: tuple[str, ...]


class InitialSearchRequest(_ProviderModel):
    pass_kind: Literal[SearchPassKind.INITIAL] = SearchPassKind.INITIAL
    run_mode: FinanceRunMode = FinanceRunMode.CURRENT_UNRESOLVED
    target_profile: search.TargetProfile
    source_policy: SearchSourcePolicy
    query_intents: tuple[SearchQueryIntent, ...] = Field(min_length=1)


class GuidedSearchRequest(_ProviderModel):
    pass_kind: Literal[SearchPassKind.GUIDED] = SearchPassKind.GUIDED
    run_mode: FinanceRunMode = FinanceRunMode.CURRENT_UNRESOLVED
    target_profile: search.TargetProfile
    source_policy: SearchSourcePolicy
    query_intents: tuple[SearchQueryIntent, ...] = Field(min_length=1)
    historical_guidance: tuple[HistoricalSearchGuidance, ...] = Field(min_length=1)


SearchRequest: TypeAlias = Annotated[
    InitialSearchRequest | GuidedSearchRequest,
    Field(discriminator="pass_kind"),
]


class RawSearchCandidate(_ProviderModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(
        frozen=True,
        extra="forbid",
        str_strip_whitespace=False,
    )

    candidate_id: str | None = None
    claim: str | None = None
    citation: str | None = None
    canonical_source_id: str | None = None
    source_version_id: str | None = None
    exact_body: str | None = None
    body_snapshot_id: str | None = None
    snapshot_captured_at: str | None = None
    snapshot_available_at: str | None = None
    available_at: str | None = None
    retrieved_at: str | None = None
    content_hash: str | None = None
    direction: str | None = None
    context_slot: str | None = None


class SearchProviderEnvelope(_ProviderModel):
    candidate_payloads: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CandidateAccepted:
    evidence: search.EvidenceItem
    canonical_source_id: str
    source_version_id: str
    body_snapshot_id: str | None
    content_hash: str


@dataclass(frozen=True, slots=True)
class CandidateRejected:
    candidate_key: str
    reason: EvidenceAuditReason
    canonical_source_id: str | None = None
    source_version_id: str | None = None
    body_snapshot_id: str | None = None
    content_hash: str | None = None


CandidateParseResult: TypeAlias = CandidateAccepted | CandidateRejected


def hash_exact_utf8_body(exact_body: str) -> str:
    """Hash exact UTF-8 bytes without normalization or whitespace rewriting."""
    return f"sha256:{sha256(exact_body.encode('utf-8')).hexdigest()}"


def canonical_snapshot_id(
    canonical_source_id: str,
    source_version_id: str,
    content_hash: str,
) -> str:
    """Bind an archived snapshot identity to source, version, and exact body."""
    material = f"{canonical_source_id}\0{source_version_id}\0{content_hash}"
    return f"snapshot:sha256:{sha256(material.encode('utf-8')).hexdigest()}"
