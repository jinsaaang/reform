"""Typed builders shared by fixed-pack runner scenarios."""

from datetime import UTC, datetime
from uuid import UUID

from src.agents.finance_forecast_agent import FinanceForecastAgent
from src.agents.finance_search_agent import FinanceSearchAgent
from src.core.finance_experiment_runner import (
    FinanceExperimentRunner,
    FinanceExperimentTrialRequest,
)
from src.core.finance_experiment_runtime import SeedIdentityVerifier
from src.domain.finance.experiment import (
    ArmFailed,
    ArmSucceeded,
    ArmUnavailable,
    FinanceExperimentSuite,
    FinanceExperimentTrial,
    FinanceSuiteStatus,
    ForecastArm,
)
from src.domain.finance.search import TargetProfile
from tests.fixtures.finance_experiment_runner import (
    FailingRepository,
    FailingRetriever,
    MemoryAwareForecastProvider,
    RecordingIdentityVerifier,
    RecordingRepository,
    RecordingRetriever,
    StaticRepository,
    StaticRetriever,
    search_envelope,
)
from tests.fixtures.finance_pipeline import FixtureSearchProvider
from tests.unit.domain.finance._experiment_factories import make_manifest, make_trial

_TRIAL_ID = UUID("00000000-0000-0000-0000-000000000002")


def trial_request() -> FinanceExperimentTrialRequest:
    """Build one valid manifest-bound trial request."""
    manifest = make_manifest()
    return FinanceExperimentTrialRequest(
        manifest=manifest,
        question=manifest.questions[0],
        repetition_index=0,
        trial_id=_TRIAL_ID,
    )


def runner_target() -> TargetProfile:
    """Build the outcome-free target corresponding to the trial request."""
    request = trial_request()
    question = request.question
    return TargetProfile(
        question_id=question.question_id,
        question_text=question.question_text,
        question_type=question.question_type,
        domain=question.domain,
        context=question.context,
        cutoff=request.manifest.cutoff,
        outcome_space=question.outcome_space,
        resolution_rule=question.resolution_rule,
    )


def build_runner(
    events: list[str],
    repository: RecordingRepository | StaticRepository | FailingRepository,
    retriever: RecordingRetriever | StaticRetriever | FailingRetriever,
    *,
    search_response: str | None = None,
    malformed_arm: ForecastArm | None = None,
    identity_verifier: SeedIdentityVerifier | None = None,
) -> tuple[
    FinanceExperimentRunner,
    FixtureSearchProvider,
    MemoryAwareForecastProvider,
]:
    """Compose the runner with typed deterministic spies."""
    search = FixtureSearchProvider((search_response or search_envelope(),), events)
    forecast = MemoryAwareForecastProvider(events, malformed_arm)
    identity = identity_verifier or RecordingIdentityVerifier(events)
    return (
        FinanceExperimentRunner(
            searcher=FinanceSearchAgent(search),
            forecaster=FinanceForecastAgent(forecast),
            repository=repository,
            retriever=retriever,
            identity_verifier=identity,
        ),
        search,
        forecast,
    )


def succeeded_arm(
    record: ArmSucceeded | ArmUnavailable | ArmFailed,
) -> ArmSucceeded:
    """Narrow a terminal fixture arm to success."""
    if isinstance(record, ArmSucceeded):
        return record
    raise AssertionError(record.status)


def unavailable_arm(
    record: ArmSucceeded | ArmUnavailable | ArmFailed,
) -> ArmUnavailable:
    """Narrow a terminal fixture arm to unavailable."""
    if isinstance(record, ArmUnavailable):
        return record
    raise AssertionError(record.status)


def suite_from_trial(
    first_trial: FinanceExperimentTrial,
    status: FinanceSuiteStatus,
) -> FinanceExperimentSuite:
    """Complete the two-question fixture schedule around one runner trial."""
    manifest = trial_request().manifest
    second_trial = make_trial(manifest.questions[1].question_id, 3)
    return FinanceExperimentSuite(
        schema_version="finance-experiment-suite/v1",
        suite_id=UUID("20000000-0000-0000-0000-000000000001"),
        created_at=datetime(2026, 7, 18, 2, tzinfo=UTC),
        status=status,
        manifest=manifest,
        trials=(first_trial, second_trial),
    )


__all__ = [
    "build_runner",
    "runner_target",
    "succeeded_arm",
    "suite_from_trial",
    "trial_request",
    "unavailable_arm",
]
