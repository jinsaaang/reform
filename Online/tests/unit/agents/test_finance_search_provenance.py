"""Gate regressions for exact-body and archived-snapshot evidence proof."""

import socket
from dataclasses import replace
from typing import assert_never
from warnings import catch_warnings, simplefilter

from src.agents.finance_search_agent import FinanceSearchAgent, SearchProviderError
from src.domain.finance.pipeline import (
    SearchCompletion,
    SearchMergeInput,
    SearchPassCompleted,
    SearchPassFailed,
    SearchPassResult,
)
from src.domain.finance.provider import (
    EvidenceAuditReason,
    FinanceRunMode,
    InitialSearchRequest,
    SearchPassFailureReason,
    SearchRequest,
    SearchQueryIntent,
    SearchSourcePolicy,
)
from tests.fixtures.finance_pipeline import (
    BASE_CANDIDATE,
    FixtureSearchProvider,
    make_target,
    provider_envelope,
)


def _completed(result: SearchPassResult) -> SearchPassCompleted:
    match result:
        case SearchPassCompleted():
            return result
        case SearchPassFailed():
            raise AssertionError(result.trace.failure_reason)
        case unreachable:
            assert_never(unreachable)


def _admit(
    payloads: tuple[str, ...],
    run_mode: FinanceRunMode = FinanceRunMode.CURRENT_UNRESOLVED,
) -> SearchCompletion:
    provider = FixtureSearchProvider((provider_envelope(payloads),))
    agent = FinanceSearchAgent(provider)
    search_pass = _completed(
        agent.search(
            InitialSearchRequest(
                run_mode=run_mode,
                target_profile=make_target(),
                source_policy=SearchSourcePolicy.OFFLINE_FIXTURE,
                query_intents=(SearchQueryIntent.DIRECT_TARGET,),
            )
        )
    )
    return agent.merge_and_admit(SearchMergeInput(make_target(), (), (search_pass,)))


class _SocketAttemptSearchProvider:
    def search(self, request: SearchRequest) -> str:
        del request
        try:
            opened = socket.socket()
        except RuntimeError as error:
            raise SearchProviderError("socket_blocked", str(error)) from error
        opened.close()
        raise AssertionError("socket policy did not block provider access")


class TestSearchProviderCapability:
    def test_should_surface_blocked_socket_as_typed_provider_failure(
        self,
        monkeypatch,
    ) -> None:
        # Given: a provider adapter whose external capability is deterministically blocked
        def blocked_socket(*args, **kwargs):
            del args, kwargs
            raise RuntimeError("socket disabled by test")

        monkeypatch.setattr(socket, "socket", blocked_socket)
        agent = FinanceSearchAgent(_SocketAttemptSearchProvider())

        # When: the provider attempts the blocked capability
        with catch_warnings():
            simplefilter("ignore", UserWarning)
            result = agent.search(
                InitialSearchRequest(
                    target_profile=make_target(),
                    source_policy=SearchSourcePolicy.OFFLINE_FIXTURE,
                    query_intents=(SearchQueryIntent.DIRECT_TARGET,),
                )
            )

        # Then: the adapter converts it to the Searcher's typed failure surface
        match result:
            case SearchPassFailed():
                assert (
                    result.trace.failure_reason
                    is SearchPassFailureReason.PROVIDER_ERROR
                )
            case SearchPassCompleted():
                raise AssertionError("blocked socket produced search success")
            case unreachable:
                assert_never(unreachable)


class TestExactBodyProof:
    def test_should_reject_invalid_and_nonmatching_body_hashes(self) -> None:
        # Given: one malformed digest and one canonical-looking false digest
        invalid = replace(
            BASE_CANDIDATE,
            candidate_id="invalid-hash",
            content_hash="sha256:not-a-digest",
        )
        mismatch = replace(
            BASE_CANDIDATE,
            candidate_id="mismatched-hash",
            content_hash=f"sha256:{'0' * 64}",
            body_snapshot_id="snapshot:false-hash",
        )
        missing_body = replace(
            BASE_CANDIDATE,
            candidate_id="missing-body",
            exact_body="",
        )

        # When: asserted provider metadata crosses the admission gate
        completion = _admit(
            (invalid.payload(), mismatch.payload(), missing_body.payload())
        )

        # Then: neither unverified exact body is admitted
        assert completion.searcher_result.evidence_pack.items == ()
        assert tuple(item.reason for item in completion.evidence_audit) == (
            EvidenceAuditReason.INVALID_CONTENT_HASH,
            EvidenceAuditReason.CONTENT_HASH_MISMATCH,
            EvidenceAuditReason.MISSING_EXACT_BODY,
        )

    def test_should_reject_asserted_only_snapshot_identity(self) -> None:
        # Given: a snapshot identifier with no verifiable archived provenance
        asserted = replace(
            BASE_CANDIDATE,
            candidate_id="asserted-snapshot",
            body_snapshot_id="snapshot:does-not-exist",
        )

        # When: the candidate crosses the admission gate
        completion = _admit(
            (asserted.payload(),),
            FinanceRunMode.HISTORICAL_BACKTEST,
        )

        # Then: an asserted string alone is not archived-snapshot proof
        assert completion.searcher_result.evidence_pack.items == ()
        assert (
            completion.evidence_audit[0].reason
            is EvidenceAuditReason.INVALID_BODY_SNAPSHOT
        )
