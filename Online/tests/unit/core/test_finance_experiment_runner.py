"""Behavioral tests for the fixed-pack three-arm finance runner."""

from dataclasses import replace
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.core.finance_experiment_runner import FinanceExperimentStage
from src.core.finance_seed_repository import FinanceSeedRepository
from src.domain.finance.experiment import (
    ArmFailed,
    ArmSucceeded,
    ArmUnavailable,
    ArmUnavailableReason,
    FinanceSuiteStatus,
    ForecastArm,
)
from src.domain.finance.provider import SearchPassKind
from src.domain.finance.retrieval import (
    HistoricalDagQuery,
    SeedAssetMismatchError,
    TopK,
)
from tests.fixtures.finance_experiment_runner import (
    FailingIdentityVerifier,
    FailingRepository,
    FailingRetriever,
    RecordingIdentityVerifier,
    RecordingRepository,
    RecordingRetriever,
    StaticRepository,
    StaticRetriever,
    search_envelope,
)
from tests.fixtures.finance_experiment_trial import (
    build_runner,
    runner_target,
    succeeded_arm,
    suite_from_trial,
    trial_request,
    unavailable_arm,
)
from tests.unit.domain.finance._factories import make_episode

_DB_PATH = Path("data/releases/worldreasoner/v1.0.0/worldreasoner_public.db")
_EXPECTED_STAGES = tuple(FinanceExperimentStage)


def test_runs_exact_fixed_pack_sequence_with_real_immutable_memory() -> None:
    # Given
    events: list[str] = []
    canonical = FinanceSeedRepository(_DB_PATH)
    repository = RecordingRepository(canonical, events)
    retriever = RecordingRetriever(repository, events)
    runner, search, forecast = build_runner(events, repository, retriever)

    # When
    completed = runner.run_trial(trial_request())

    # Then
    assert completed.stage_order == _EXPECTED_STAGES
    assert events == [
        "metadata_preflight",
        "forecast:direct",
        "search:initial",
        "forecast:search_only",
        "repository_load",
        "historical_retrieval",
        "forecast:search_dag",
    ]
    assert search.call_count == 1 and forecast.call_count == 3
    assert repository.call_count == 1 and retriever.call_count == 1
    direct, search_only, search_dag = tuple(
        succeeded_arm(record) for record in completed.trial.arms
    )
    assert direct.arm is ForecastArm.DIRECT
    assert not forecast.inputs[0].evidence_pack.items
    assert not forecast.inputs[0].historical_memory
    assert tuple(
        str(item.evidence_id) for item in forecast.inputs[1].evidence_pack.items
    ) == ("evidence-0", "evidence-1")
    assert forecast.inputs[1].evidence_pack == forecast.inputs[2].evidence_pack
    assert (
        forecast.inputs[1].evidence_pack.model_dump_json()
        == forecast.inputs[2].evidence_pack.model_dump_json()
    )
    assert completed.evidence_snapshot is not None
    assert search_only.treatment.evidence_snapshot_digest == (
        completed.evidence_snapshot.digest
    )
    assert search_dag.treatment.evidence_snapshot_digest == (
        completed.evidence_snapshot.digest
    )
    assert forecast.inputs[2].historical_memory == tuple(
        item.episode for item in retriever.results[0].selected
    )
    assert forecast.inputs[2].historical_memory[0].nodes
    assert forecast.inputs[2].historical_memory[0].edges
    assert forecast.inputs[2].historical_memory[0].historical_outcome.value
    assert "current_dag" not in forecast.inputs[2].model_dump_json()
    assert search.requests[0].pass_kind is SearchPassKind.INITIAL


def test_malformed_manifest_fails_before_identity_or_provider_call() -> None:
    # Given
    events: list[str] = []
    request = trial_request()
    invalid_manifest = request.manifest.model_copy(update={"arm_order": ()})
    invalid_request = replace(request, manifest=invalid_manifest)
    repository = StaticRepository((make_episode(),), events)
    retriever = RecordingRetriever(repository, events)
    identity = RecordingIdentityVerifier(events)
    runner, search, forecast = build_runner(
        events,
        repository,
        retriever,
        identity_verifier=identity,
    )

    # When / Then
    with pytest.raises(ValidationError):
        runner.run_trial(invalid_request)
    assert identity.call_count == 0
    assert search.call_count == 0 and forecast.call_count == 0
    assert repository.call_count == 0 and retriever.call_count == 0


def test_identity_mismatch_fails_before_any_provider_or_repository_call() -> None:
    # Given
    events: list[str] = []
    repository = StaticRepository((make_episode(),), events)
    retriever = RecordingRetriever(repository, events)
    runner, search, forecast = build_runner(
        events,
        repository,
        retriever,
        identity_verifier=FailingIdentityVerifier(),
    )

    # When / Then
    with pytest.raises(SeedAssetMismatchError):
        runner.run_trial(trial_request())
    assert search.call_count == 0 and forecast.call_count == 0
    assert repository.call_count == 0 and retriever.call_count == 0
    assert events == []


def test_empty_a0_preserves_direct_and_never_loads_repository() -> None:
    # Given
    events: list[str] = []
    repository = StaticRepository((make_episode(),), events)
    retriever = RecordingRetriever(repository, events)
    runner, search, forecast = build_runner(
        events,
        repository,
        retriever,
        search_response=search_envelope(()),
    )

    # When
    completed = runner.run_trial(trial_request())

    # Then
    assert isinstance(completed.trial.arms[0], ArmSucceeded)
    reasons = tuple(
        unavailable_arm(record).reason for record in completed.trial.arms[1:]
    )
    assert reasons == (
        ArmUnavailableReason.NO_ADMITTED_EVIDENCE,
        ArmUnavailableReason.NO_ADMITTED_EVIDENCE,
    )
    assert search.call_count == 1 and forecast.call_count == 1
    assert repository.call_count == 0 and retriever.call_count == 0
    assert completed.stage_order[-1] is FinanceExperimentStage.FREEZE


def test_failed_initial_search_preserves_direct_without_freezing_or_loading() -> None:
    # Given
    events: list[str] = []
    repository = StaticRepository((make_episode(),), events)
    retriever = RecordingRetriever(repository, events)
    runner, search, forecast = build_runner(
        events,
        repository,
        retriever,
        search_response="not-json",
    )

    # When
    completed = runner.run_trial(trial_request())

    # Then
    assert completed.trial.arms[0].status == "succeeded"
    reasons = tuple(
        unavailable_arm(record).reason for record in completed.trial.arms[1:]
    )
    assert reasons == (
        ArmUnavailableReason.SEARCH_UNAVAILABLE,
        ArmUnavailableReason.SEARCH_UNAVAILABLE,
    )
    assert completed.evidence_snapshot is None
    assert completed.stage_order[-1] is FinanceExperimentStage.INITIAL_SEARCH
    assert search.call_count == 1 and forecast.call_count == 1
    assert repository.call_count == 0 and retriever.call_count == 0


def test_repository_failure_preserves_direct_and_search_only() -> None:
    # Given
    events: list[str] = []
    repository = FailingRepository(events)
    unused_source = StaticRepository((), events)
    retriever = RecordingRetriever(unused_source, events)
    runner, _, forecast = build_runner(events, repository, retriever)

    # When
    completed = runner.run_trial(trial_request())

    # Then
    assert all(isinstance(record, ArmSucceeded) for record in completed.trial.arms[:2])
    unavailable = completed.trial.arms[2]
    assert isinstance(unavailable, ArmUnavailable)
    assert unavailable.reason is ArmUnavailableReason.MEMORY_UNAVAILABLE
    assert forecast.call_count == 2
    assert repository.call_count == 1 and retriever.call_count == 0


def test_retriever_failure_preserves_direct_and_search_only() -> None:
    # Given
    events: list[str] = []
    repository = StaticRepository((make_episode(),), events)
    retriever = FailingRetriever(events)
    runner, _, forecast = build_runner(events, repository, retriever)

    # When
    completed = runner.run_trial(trial_request())

    # Then
    assert all(isinstance(record, ArmSucceeded) for record in completed.trial.arms[:2])
    assert isinstance(completed.trial.arms[2], ArmUnavailable)
    assert forecast.call_count == 2
    assert repository.call_count == 1 and retriever.call_count == 1


def test_ineligible_memory_preserves_direct_and_search_only() -> None:
    # Given
    events: list[str] = []
    repository = StaticRepository((make_episode(resolution_year=2027),), events)
    retriever = RecordingRetriever(repository, events)
    runner, _, forecast = build_runner(events, repository, retriever)

    # When
    completed = runner.run_trial(trial_request())

    # Then
    unavailable = completed.trial.arms[2]
    assert isinstance(unavailable, ArmUnavailable)
    assert unavailable.reason is ArmUnavailableReason.MEMORY_UNAVAILABLE
    assert forecast.call_count == 2


def test_tampered_retrieval_preserves_search_only_and_blocks_c() -> None:
    # Given
    events: list[str] = []
    canonical = make_episode()
    repository = StaticRepository((canonical,), events)
    query = HistoricalDagQuery(
        target_profile=runner_target(),
        policy=trial_request().manifest.retrieval.eligibility_policy,
        top_k=TopK(trial_request().manifest.retrieval.top_k),
    )
    checker = RecordingRetriever(repository, [])
    valid = checker.delegate.retrieve_from((canonical,), query)
    tampered_episode = replace(
        canonical,
        historical_outcome=replace(canonical.historical_outcome, value="No"),
    )
    ranked = replace(valid.ranked_candidates[0], episode=tampered_episode)
    tampered = replace(valid, selected=(ranked,), ranked_candidates=(ranked,))
    retriever = StaticRetriever(tampered, events)
    runner, _, forecast = build_runner(events, repository, retriever)

    # When
    completed = runner.run_trial(trial_request())

    # Then
    assert isinstance(completed.trial.arms[1], ArmSucceeded)
    assert isinstance(completed.trial.arms[2], ArmUnavailable)
    assert forecast.call_count == 2


@pytest.mark.parametrize("malformed_arm", tuple(ForecastArm))
def test_one_malformed_arm_does_not_erase_successful_siblings(
    malformed_arm: ForecastArm,
) -> None:
    # Given
    events: list[str] = []
    repository = StaticRepository((make_episode(),), events)
    retriever = RecordingRetriever(repository, events)
    runner, _, forecast = build_runner(
        events,
        repository,
        retriever,
        malformed_arm=malformed_arm,
    )

    # When
    completed = runner.run_trial(trial_request())

    # Then
    failed = tuple(
        record for record in completed.trial.arms if isinstance(record, ArmFailed)
    )
    statuses = tuple(record.status for record in completed.trial.arms)
    assert len(failed) == 1 and failed[0].arm is malformed_arm
    assert statuses.count("succeeded") == 2 and forecast.call_count == 3
    with pytest.raises(ValidationError, match="suite_status_mismatch"):
        suite_from_trial(completed.trial, FinanceSuiteStatus.COMPLETE)
    partial = suite_from_trial(completed.trial, FinanceSuiteStatus.PARTIAL)
    assert partial.status is FinanceSuiteStatus.PARTIAL
