"""Two-pass finance Searcher with an isolated synchronous provider port."""

from dataclasses import dataclass
from typing import Protocol, assert_never

from pydantic import ValidationError
from typing_extensions import override

from src.domain.finance.evidence_admission import parse_search_candidate
from src.domain.finance.memory import EvidenceId, HistoricalDagReference
from src.domain.finance.pipeline import (
    EvidenceAuditRecord,
    EvidenceDecision,
    EvidenceProvenance,
    QueryPassCompleted,
    QueryPassFailed,
    SearchCompletion,
    SearchMergeInput,
    SearchPassCompleted,
    SearchPassFailed,
    SearchPassResult,
    build_historical_guidance,
)
from src.domain.finance.provider import (
    CandidateAccepted,
    CandidateRejected,
    EvidenceAuditReason,
    GuidedSearchRequest,
    HistoricalSearchGuidance,
    InitialSearchRequest,
    SearchPassFailureReason,
    SearchPassKind,
    SearchProviderEnvelope,
    SearchRequest,
)
from src.domain.finance.retrieval import RankedHistoricalDag
from src.domain.finance.search import (
    EvidencePack,
    SearcherResult,
)


class SearchProvider(Protocol):
    """Only external capability available to the finance Searcher."""

    def search(self, request: SearchRequest) -> str:
        """Return one serialized provider envelope."""
        ...


@dataclass(frozen=True, slots=True)
class SearchProviderError(Exception):
    code: str
    detail: str

    @override
    def __str__(self) -> str:
        return f"search provider {self.code}: {self.detail}"


@dataclass(frozen=True, slots=True)
class FinanceSearchAgent:
    """Run providers and admit evidence without forecasting or graph building."""

    provider: SearchProvider

    @staticmethod
    def build_guidance(
        selected: tuple[RankedHistoricalDag, ...],
    ) -> tuple[HistoricalSearchGuidance, ...]:
        """Reduce full selected episodes to outcome-free provider guidance."""
        return build_historical_guidance(selected)

    def search(self, request: SearchRequest) -> SearchPassResult:
        """Run exactly one typed Searcher pass."""
        match request:
            case InitialSearchRequest():
                references: tuple[HistoricalDagReference, ...] = ()
                matched_terms: tuple[str, ...] = ()
                mechanism_hints: tuple[str, ...] = ()
            case GuidedSearchRequest(historical_guidance=guidance):
                references = tuple(item.reference for item in guidance)
                matched_terms = tuple(
                    sorted({term for item in guidance for term in item.matched_terms})
                )
                mechanism_hints = tuple(
                    sorted({hint for item in guidance for hint in item.mechanism_hints})
                )
            case unreachable:
                assert_never(unreachable)
        try:
            serialized = self.provider.search(request)
        except SearchProviderError:
            return SearchPassFailed(
                request=request,
                trace=QueryPassFailed(
                    pass_kind=request.pass_kind,
                    run_mode=request.run_mode,
                    source_policy=request.source_policy,
                    query_intents=request.query_intents,
                    historical_dag_references=references,
                    matched_terms=matched_terms,
                    mechanism_hints=mechanism_hints,
                    failure_reason=SearchPassFailureReason.PROVIDER_ERROR,
                ),
            )
        try:
            envelope = SearchProviderEnvelope.model_validate_json(serialized)
        except ValidationError:
            return SearchPassFailed(
                request=request,
                trace=QueryPassFailed(
                    pass_kind=request.pass_kind,
                    run_mode=request.run_mode,
                    source_policy=request.source_policy,
                    query_intents=request.query_intents,
                    historical_dag_references=references,
                    matched_terms=matched_terms,
                    mechanism_hints=mechanism_hints,
                    failure_reason=SearchPassFailureReason.MALFORMED_PROVIDER_OUTPUT,
                ),
            )
        return SearchPassCompleted(
            request=request,
            candidate_payloads=envelope.candidate_payloads,
            trace=QueryPassCompleted(
                pass_kind=request.pass_kind,
                run_mode=request.run_mode,
                source_policy=request.source_policy,
                query_intents=request.query_intents,
                historical_dag_references=references,
                matched_terms=matched_terms,
                mechanism_hints=mechanism_hints,
                candidate_count=len(envelope.candidate_payloads),
            ),
        )

    def merge_and_admit(self, merge_input: SearchMergeInput) -> SearchCompletion:
        """Merge both passes and apply the strict exact-body temporal gate."""
        accepted: list[CandidateAccepted] = []
        audit: list[EvidenceAuditRecord] = []
        seen_hashes: dict[str, EvidenceId] = {}
        seen_identities: dict[tuple[str, str], EvidenceId] = {}
        seen_ids: dict[EvidenceId, EvidenceId] = {}
        source_passes: dict[EvidenceId, list[SearchPassKind]] = {}
        for search_pass in merge_input.passes:
            for payload in search_pass.candidate_payloads:
                parsed = parse_search_candidate(
                    payload,
                    merge_input.target_profile.cutoff,
                    search_pass.request.run_mode,
                    search_pass.request.source_policy,
                )
                match parsed:
                    case CandidateRejected():
                        audit.append(
                            EvidenceAuditRecord(
                                sequence=len(audit),
                                candidate_key=parsed.candidate_key,
                                source_pass=search_pass.request.pass_kind,
                                decision=EvidenceDecision.EXCLUDED,
                                reason=parsed.reason,
                                canonical_source_id=parsed.canonical_source_id,
                                source_version_id=parsed.source_version_id,
                                body_snapshot_id=parsed.body_snapshot_id,
                                content_hash=parsed.content_hash,
                            )
                        )
                    case CandidateAccepted():
                        evidence_id = parsed.evidence.evidence_id
                        identity = (
                            parsed.canonical_source_id,
                            parsed.source_version_id,
                        )
                        duplicate_of = (
                            seen_ids.get(evidence_id)
                            or seen_hashes.get(parsed.content_hash)
                            or seen_identities.get(identity)
                        )
                        if duplicate_of is not None:
                            passes = source_passes[duplicate_of]
                            if search_pass.request.pass_kind not in passes:
                                passes.append(search_pass.request.pass_kind)
                            audit.append(
                                EvidenceAuditRecord(
                                    sequence=len(audit),
                                    candidate_key=str(evidence_id),
                                    source_pass=search_pass.request.pass_kind,
                                    decision=EvidenceDecision.EXCLUDED,
                                    reason=EvidenceAuditReason.DUPLICATE,
                                    duplicate_of=duplicate_of,
                                    canonical_source_id=parsed.canonical_source_id,
                                    source_version_id=parsed.source_version_id,
                                    body_snapshot_id=parsed.body_snapshot_id,
                                    content_hash=parsed.content_hash,
                                )
                            )
                            continue
                        accepted.append(parsed)
                        seen_ids[evidence_id] = evidence_id
                        seen_hashes[parsed.content_hash] = evidence_id
                        seen_identities[identity] = evidence_id
                        source_passes[evidence_id] = [search_pass.request.pass_kind]
                        audit.append(
                            EvidenceAuditRecord(
                                sequence=len(audit),
                                candidate_key=str(evidence_id),
                                source_pass=search_pass.request.pass_kind,
                                decision=EvidenceDecision.ADMITTED,
                                reason=EvidenceAuditReason.ADMITTED,
                                canonical_source_id=parsed.canonical_source_id,
                                source_version_id=parsed.source_version_id,
                                body_snapshot_id=parsed.body_snapshot_id,
                                content_hash=parsed.content_hash,
                            )
                        )
                    case unreachable:
                        assert_never(unreachable)
        pack = EvidencePack(
            items=tuple(item.evidence for item in accepted),
            historical_dag_references=merge_input.historical_dag_references,
        )
        return SearchCompletion(
            searcher_result=SearcherResult(
                target_profile=merge_input.target_profile,
                evidence_pack=pack,
                historical_dag_references=merge_input.historical_dag_references,
            ),
            evidence_audit=tuple(audit),
            evidence_provenance=tuple(
                EvidenceProvenance(
                    evidence_id=item.evidence.evidence_id,
                    source_passes=tuple(source_passes[item.evidence.evidence_id]),
                    canonical_source_id=item.canonical_source_id,
                    source_version_id=item.source_version_id,
                    body_snapshot_id=item.body_snapshot_id,
                    content_hash=item.content_hash,
                )
                for item in accepted
            ),
        )
