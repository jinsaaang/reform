"""Behavioral tests for the isolated two-pass finance Searcher."""

from dataclasses import replace
from typing import assert_never

from src.agents.finance_search_agent import FinanceSearchAgent
from src.domain.finance.pipeline import (
    SearchMergeInput,
    SearchPassCompleted,
    SearchPassFailed,
    SearchPassResult,
)
from src.domain.finance.provider import (
    EvidenceAuditReason,
    FinanceRunMode,
    GuidedSearchRequest,
    HistoricalSearchGuidance,
    InitialSearchRequest,
    SearchPassFailureReason,
    SearchQueryIntent,
    SearchSourcePolicy,
    hash_exact_utf8_body,
)
from tests.fixtures.finance_pipeline import (
    BASE_CANDIDATE,
    FixtureSearchProvider,
    make_target,
    provider_envelope,
)
from tests.unit.domain.finance._factories import make_episode

_INITIAL_INTENTS = (
    SearchQueryIntent.DIRECT_TARGET,
    SearchQueryIntent.SOURCE_OF_RECORD,
    SearchQueryIntent.OPEN_WORLD,
    SearchQueryIntent.RESOLUTION_RULE,
)
_GUIDED_INTENTS = (
    SearchQueryIntent.HISTORICAL_GAP,
    SearchQueryIntent.COUNTEREVIDENCE,
    SearchQueryIntent.OPEN_WORLD,
    SearchQueryIntent.SOURCE_OF_RECORD,
)


def _initial_request(
    run_mode: FinanceRunMode = FinanceRunMode.CURRENT_UNRESOLVED,
) -> InitialSearchRequest:
    return InitialSearchRequest(
        run_mode=run_mode,
        target_profile=make_target(),
        source_policy=SearchSourcePolicy.OFFLINE_FIXTURE,
        query_intents=_INITIAL_INTENTS,
    )


def _guided_request(
    run_mode: FinanceRunMode = FinanceRunMode.CURRENT_UNRESOLVED,
) -> GuidedSearchRequest:
    return GuidedSearchRequest(
        run_mode=run_mode,
        target_profile=make_target(),
        source_policy=SearchSourcePolicy.OFFLINE_FIXTURE,
        query_intents=_GUIDED_INTENTS,
        historical_guidance=(
            HistoricalSearchGuidance(
                reference=make_episode().reference,
                matched_terms=("gpu", "revenue"),
                mechanism_hints=("demand",),
            ),
        ),
    )


def _completed(result: SearchPassResult) -> SearchPassCompleted:
    match result:
        case SearchPassCompleted():
            return result
        case SearchPassFailed():
            raise AssertionError(result.trace.failure_reason)
        case unreachable:
            assert_never(unreachable)


def _run_two_passes(
    initial_payloads: tuple[str, ...],
    guided_payloads: tuple[str, ...],
    run_mode: FinanceRunMode = FinanceRunMode.CURRENT_UNRESOLVED,
) -> tuple[FinanceSearchAgent, SearchPassCompleted, SearchPassCompleted]:
    provider = FixtureSearchProvider(
        (provider_envelope(initial_payloads), provider_envelope(guided_payloads))
    )
    agent = FinanceSearchAgent(provider)
    initial = _completed(agent.search(_initial_request(run_mode)))
    guided = _completed(agent.search(_guided_request(run_mode)))
    return agent, initial, guided


class TestSearchProviderBoundary:
    def test_should_keep_initial_provider_request_dag_independent(self) -> None:
        # Given: a first-pass search through the real Searcher provider seam
        provider = FixtureSearchProvider((provider_envelope(()),))
        agent = FinanceSearchAgent(provider)

        # When: the initial pass invokes the provider
        _ = _completed(agent.search(_initial_request()))

        # Then: the request shape cannot carry any historical guidance
        request = provider.requests[0]
        assert set(type(request).model_fields) == {
            "pass_kind",
            "run_mode",
            "target_profile",
            "source_policy",
            "query_intents",
        }
        serialized = request.model_dump_json()
        assert "historical" not in serialized and "dag_id" not in serialized

    def test_should_send_only_outcome_free_guidance_on_second_pass(self) -> None:
        # Given: identity and mechanism guidance for one historical DAG
        provider = FixtureSearchProvider((provider_envelope(()),))
        agent = FinanceSearchAgent(provider)

        # When: the guided pass invokes the provider
        _ = _completed(agent.search(_guided_request()))

        # Then: no full episode, graph body, or historical outcome is serialized
        serialized = provider.requests[0].model_dump_json()
        assert all(
            forbidden not in serialized
            for forbidden in ("historical_outcome", "nodes", "edges", "impacts")
        )

    def test_should_fail_closed_on_malformed_provider_envelope(self) -> None:
        # Given: a Search provider response that is not a serialized envelope
        agent = FinanceSearchAgent(FixtureSearchProvider(("not-json",)))

        # When: the response crosses the Searcher boundary
        result = agent.search(_initial_request())

        # Then: a typed pass failure is returned without candidates
        match result:
            case SearchPassFailed():
                assert (
                    result.trace.failure_reason
                    is SearchPassFailureReason.MALFORMED_PROVIDER_OUTPUT
                )
            case SearchPassCompleted():
                raise AssertionError("malformed envelope produced success")
            case unreachable:
                assert_never(unreachable)


class TestEvidenceAdmission:
    def test_should_require_available_at_strictly_before_cutoff(self) -> None:
        # Given: pre-cutoff, equal-cutoff, and post-cutoff exact body candidates
        cutoff = make_target().cutoff.isoformat()
        equal = replace(BASE_CANDIDATE, candidate_id="equal", available_at=cutoff)
        post = replace(
            BASE_CANDIDATE,
            candidate_id="post",
            available_at="2026-06-02T00:00:00+00:00",
        )
        agent, initial, guided = _run_two_passes(
            (BASE_CANDIDATE.payload(), equal.payload(), post.payload()), ()
        )

        # When: both passes enter the exact-body admission gate
        completion = agent.merge_and_admit(
            SearchMergeInput(
                make_target(),
                (make_episode().reference,),
                (initial, guided),
            )
        )

        # Then: only the strictly pre-cutoff body is admitted
        assert tuple(
            str(item.evidence_id)
            for item in completion.searcher_result.evidence_pack.items
        ) == ("evidence-1",)
        assert (
            tuple(record.reason for record in completion.evidence_audit).count(
                EvidenceAuditReason.NOT_STRICTLY_BEFORE_CUTOFF
            )
            == 2
        )

    def test_should_reject_malformed_time_and_missing_body_identity(self) -> None:
        # Given: naive time, missing archived snapshot, and missing hash candidates
        naive = replace(BASE_CANDIDATE, candidate_id="naive", available_at="2024-12-01")
        missing_snapshot = replace(
            BASE_CANDIDATE,
            candidate_id="missing-snapshot",
            body_snapshot_id="",
        )
        missing_hash = replace(
            BASE_CANDIDATE, candidate_id="missing-hash", content_hash=""
        )
        agent, initial, guided = _run_two_passes(
            (
                naive.payload(),
                missing_snapshot.payload(),
                missing_hash.payload(),
            ),
            (),
            FinanceRunMode.HISTORICAL_BACKTEST,
        )

        # When: malformed candidates enter admission
        completion = agent.merge_and_admit(
            SearchMergeInput(make_target(), (), (initial, guided))
        )

        # Then: each failure remains a distinct typed audit reason
        assert tuple(record.reason for record in completion.evidence_audit) == (
            EvidenceAuditReason.MALFORMED_TIMESTAMP,
            EvidenceAuditReason.MISSING_BODY_SNAPSHOT,
            EvidenceAuditReason.MISSING_CONTENT_HASH,
        )

    def test_should_deduplicate_hash_and_canonical_identity_with_pass_audit(
        self,
    ) -> None:
        # Given: pass-two copies by hash and by canonical body identity
        duplicate_hash = replace(
            BASE_CANDIDATE,
            candidate_id="duplicate-hash",
            canonical_source_id="other-source",
            source_version_id="other-version",
        )
        other_body = "Different exact source body.\n"
        duplicate_identity = replace(
            BASE_CANDIDATE,
            candidate_id="duplicate-identity",
            exact_body=other_body,
            content_hash=hash_exact_utf8_body(other_body),
        )
        agent, initial, guided = _run_two_passes(
            (BASE_CANDIDATE.payload(),),
            (duplicate_hash.payload(), duplicate_identity.payload()),
        )

        # When: both passes are merged deterministically
        completion = agent.merge_and_admit(
            SearchMergeInput(make_target(), (), (initial, guided))
        )

        # Then: one item remains and both duplicate exclusions preserve provenance
        assert len(completion.searcher_result.evidence_pack.items) == 1
        assert (
            tuple(record.reason for record in completion.evidence_audit).count(
                EvidenceAuditReason.DUPLICATE
            )
            == 2
        )
        assert completion.evidence_provenance[0].source_passes == (
            initial.request.pass_kind,
            guided.request.pass_kind,
        )
