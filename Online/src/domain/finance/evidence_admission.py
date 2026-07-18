"""Pure exact-body and run-mode evidence admission."""

import re
from datetime import datetime
from typing import Final, assert_never

from pydantic import TypeAdapter, ValidationError

from src.domain.finance import memory, search
from src.domain.finance.provider import (
    CandidateAccepted,
    CandidateParseResult,
    CandidateRejected,
    EvidenceAuditReason,
    FinanceRunMode,
    RawSearchCandidate,
    SearchSourcePolicy,
    canonical_snapshot_id,
    hash_exact_utf8_body,
)

_DIRECTION = TypeAdapter(search.EvidenceDirection)
_SHA256_PATTERN: Final = re.compile(r"sha256:[0-9a-f]{64}\Z")


def _aware_timestamp(raw: str | None) -> datetime | None:
    if raw is None:
        return None
    try:
        value = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if value.tzinfo is None or value.utcoffset() is None:
        return None
    return value


def _rejected(
    raw: RawSearchCandidate,
    key: str,
    reason: EvidenceAuditReason,
) -> CandidateRejected:
    return CandidateRejected(
        key,
        reason,
        raw.canonical_source_id,
        raw.source_version_id,
        raw.body_snapshot_id,
        raw.content_hash,
    )


def _historical_snapshot(
    raw: RawSearchCandidate,
    key: str,
    cutoff: datetime,
    source_policy: SearchSourcePolicy | None,
) -> CandidateRejected | str:
    if not raw.body_snapshot_id:
        return _rejected(raw, key, EvidenceAuditReason.MISSING_BODY_SNAPSHOT)
    if raw.snapshot_captured_at is None or raw.snapshot_available_at is None:
        return _rejected(raw, key, EvidenceAuditReason.MISSING_SNAPSHOT_PROVENANCE)
    captured_at = _aware_timestamp(raw.snapshot_captured_at)
    available_at = _aware_timestamp(raw.snapshot_available_at)
    if captured_at is None or available_at is None:
        return _rejected(raw, key, EvidenceAuditReason.MALFORMED_SNAPSHOT_TIMESTAMP)
    if raw.canonical_source_id is None or raw.source_version_id is None:
        return _rejected(raw, key, EvidenceAuditReason.MISSING_SNAPSHOT_PROVENANCE)
    if raw.content_hash is None:
        return _rejected(raw, key, EvidenceAuditReason.MISSING_CONTENT_HASH)
    expected = canonical_snapshot_id(
        raw.canonical_source_id,
        raw.source_version_id,
        raw.content_hash,
    )
    if raw.body_snapshot_id != expected:
        return _rejected(raw, key, EvidenceAuditReason.INVALID_BODY_SNAPSHOT)
    if available_at < captured_at:
        return _rejected(
            raw,
            key,
            EvidenceAuditReason.SNAPSHOT_AVAILABILITY_PRECEDES_CAPTURE,
        )
    if source_policy is not SearchSourcePolicy.PUBLIC_DB_METADATA and (
        captured_at >= cutoff or available_at >= cutoff
    ):
        return _rejected(
            raw,
            key,
            EvidenceAuditReason.SNAPSHOT_NOT_STRICTLY_BEFORE_CUTOFF,
        )
    return raw.body_snapshot_id


def parse_search_candidate(
    payload: str,
    cutoff: datetime,
    run_mode: FinanceRunMode,
    source_policy: SearchSourcePolicy | None = None,
) -> CandidateParseResult:
    """Parse a candidate and prove its exact body and temporal eligibility."""
    try:
        raw = RawSearchCandidate.model_validate_json(payload)
    except ValidationError:
        return CandidateRejected(
            "malformed-provider-candidate",
            EvidenceAuditReason.MALFORMED_CANDIDATE,
        )
    key = raw.candidate_id or "missing-candidate-id"
    if (
        not raw.candidate_id
        or not raw.claim
        or not raw.citation
        or not raw.direction
        or not raw.context_slot
    ):
        return CandidateRejected(key, EvidenceAuditReason.MISSING_CORE_FIELD)
    if not raw.canonical_source_id or not raw.canonical_source_id.strip():
        return CandidateRejected(key, EvidenceAuditReason.MISSING_CANONICAL_SOURCE)
    if not raw.source_version_id or not raw.source_version_id.strip():
        return _rejected(raw, key, EvidenceAuditReason.MISSING_SOURCE_VERSION)
    if not raw.exact_body:
        return _rejected(raw, key, EvidenceAuditReason.MISSING_EXACT_BODY)
    if not raw.content_hash:
        return _rejected(raw, key, EvidenceAuditReason.MISSING_CONTENT_HASH)
    if _SHA256_PATTERN.fullmatch(raw.content_hash) is None:
        return _rejected(raw, key, EvidenceAuditReason.INVALID_CONTENT_HASH)
    if raw.content_hash != hash_exact_utf8_body(raw.exact_body):
        return _rejected(raw, key, EvidenceAuditReason.CONTENT_HASH_MISMATCH)
    available_at = _aware_timestamp(raw.available_at)
    retrieved_at = _aware_timestamp(raw.retrieved_at)
    if available_at is None or retrieved_at is None:
        return _rejected(raw, key, EvidenceAuditReason.MALFORMED_TIMESTAMP)
    try:
        direction = _DIRECTION.validate_python(raw.direction)
    except ValidationError:
        return _rejected(raw, key, EvidenceAuditReason.MALFORMED_CANDIDATE)
    if available_at >= cutoff:
        return _rejected(raw, key, EvidenceAuditReason.NOT_STRICTLY_BEFORE_CUTOFF)
    if retrieved_at < available_at:
        return _rejected(
            raw,
            key,
            EvidenceAuditReason.RETRIEVAL_PRECEDES_AVAILABILITY,
        )
    match run_mode:
        case FinanceRunMode.CURRENT_UNRESOLVED:
            snapshot_id: str | None = None
        case FinanceRunMode.HISTORICAL_BACKTEST:
            snapshot = _historical_snapshot(raw, key, cutoff, source_policy)
            match snapshot:
                case CandidateRejected():
                    return snapshot
                case str():
                    snapshot_id = snapshot
                case unreachable:
                    assert_never(unreachable)
        case unreachable:
            assert_never(unreachable)
    return CandidateAccepted(
        evidence=search.EvidenceItem(
            evidence_id=memory.EvidenceId(raw.candidate_id),
            claim=raw.claim,
            citation=raw.citation,
            available_at=available_at,
            retrieved_at=retrieved_at,
            content_hash=raw.content_hash,
            direction=direction,
            context_slot=raw.context_slot,
        ),
        canonical_source_id=raw.canonical_source_id,
        source_version_id=raw.source_version_id,
        body_snapshot_id=snapshot_id,
        content_hash=raw.content_hash,
    )
