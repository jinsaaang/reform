"""Deterministic real-value factories for finance experiment contracts."""

from datetime import UTC, datetime
from uuid import UUID

from src.domain.finance.experiment import (
    ArmOrderPolicy,
    ArmSucceeded,
    FinanceBinaryQuestion,
    FinanceExperimentManifest,
    FinanceExperimentSuite,
    FinanceExperimentTrial,
    FinanceResolutionEntry,
    FinanceResolutionManifest,
    FinanceSearchBackend,
    FinanceSuiteStatus,
    ForecastArm,
    JudgeMember,
    ProtocolVersions,
    ProviderAttemptStatus,
    ProviderAttemptTelemetry,
    ProviderSeedUnsupported,
    ProviderUsageUnavailable,
    ReasoningEffort,
    RetrievalSettings,
    SafeCompletionSettings,
    SearchSettings,
    TreatmentAudit,
)
from src.domain.finance.forecast import (
    ForecastResult,
    OutcomeProbability,
    Scenario,
)
from src.domain.finance.memory import OutcomeLabel, QuestionId, QuestionKind, ScenarioId
from src.domain.finance.provider import FinanceRunMode, SearchSourcePolicy
from src.domain.finance.retrieval import EligibilityPolicy


def make_completion_settings() -> SafeCompletionSettings:
    return SafeCompletionSettings(
        model="openrouter/openai/gpt-5.6-sol",
        reasoning_effort=ReasoningEffort.HIGH,
        temperature=0.2,
        max_output_tokens=12_000,
        timeout_seconds=180.0,
        requested_seed=None,
        provider_retry_budget=0,
        schema_retry_budget=0,
    )


def make_question(identifier: str) -> FinanceBinaryQuestion:
    return FinanceBinaryQuestion(
        question_id=QuestionId(identifier),
        question_text=f"Will {identifier} resolve positively?",
        question_type=QuestionKind.BINARY,
        domain="finance",
        context=(f"Official context for {identifier}.",),
        outcome_space=(OutcomeLabel("No"), OutcomeLabel("Yes")),
        positive_label=OutcomeLabel("Yes"),
        resolution_rule=f"Resolve {identifier} from its official source.",
    )


def make_manifest() -> FinanceExperimentManifest:
    completion = make_completion_settings()
    return FinanceExperimentManifest(
        schema_version="finance-experiment-manifest/v1",
        manifest_id="finance-fixture-two-question",
        cutoff=datetime(2026, 7, 18, tzinfo=UTC),
        run_mode=FinanceRunMode.CURRENT_UNRESOLVED,
        temporal_policy_version="live_publication_filtered/v1",
        repetitions=1,
        root_seed=20_260_718,
        arm_order_policy=ArmOrderPolicy.FIXED,
        arm_order=(
            ForecastArm.DIRECT,
            ForecastArm.SEARCH_ONLY,
            ForecastArm.SEARCH_DAG,
        ),
        search=SearchSettings(
            source_policy=SearchSourcePolicy.LIVE_SEARCH,
            backend=FinanceSearchBackend.BING_NEWS_RSS_V1,
            result_limit=5,
            body_fetch_timeout_seconds=20.0,
            provider_retry_budget=0,
        ),
        retrieval=RetrievalSettings(
            evidence_admission_version="evidence-admission/v1",
            eligibility_version="public-db-bootstrap/v1",
            retriever_version="lexical-retriever/v1",
            eligibility_policy=EligibilityPolicy.PUBLIC_DB_BOOTSTRAP,
            top_k=3,
        ),
        forecast=completion,
        judges=tuple(
            JudgeMember(member_id=f"judge-{index}", settings=completion)
            for index in range(1, 4)
        ),
        protocol_versions=ProtocolVersions(
            forecast_prompt="forecast-prompt/v1",
            judge_prompt="judge-prompt/v1",
            forecast_result="forecast-result/v1",
            judge_result="judge-result/v1",
            suite_output="finance-experiment-suite/v1",
        ),
        questions=(make_question("question-alpha"), make_question("question-beta")),
    )


def make_forecast() -> ForecastResult:
    outcomes = (
        OutcomeProbability(label=OutcomeLabel("No"), probability=0.4),
        OutcomeProbability(label=OutcomeLabel("Yes"), probability=0.6),
    )
    return ForecastResult(
        scenarios=(
            Scenario(
                scenario_id=ScenarioId("scenario-base"),
                name="Base case",
                reasoning_steps=("A public indicator moves before resolution.",),
                probability=1.0,
                conditional_outcomes=outcomes,
                evidence_ids=(),
                historical_dag_references=(),
                assumptions=("The official series remains available.",),
                triggers=("The indicator crosses its threshold.",),
                disconfirmers=("The indicator reverses.",),
                uncertainty="Publication timing remains uncertain.",
            ),
        ),
        outcome_probabilities=outcomes,
        explanation="The base case receives the full scenario weight.",
    )


def make_attempt() -> ProviderAttemptTelemetry:
    return ProviderAttemptTelemetry(
        attempt_index=0,
        provider_name="offline-completion-fixture",
        status=ProviderAttemptStatus.SUCCEEDED,
        latency_ms=7,
        input_bytes=120,
        output_bytes=64,
        seed=ProviderSeedUnsupported(requested_seed=42),
        usage=ProviderUsageUnavailable(),
    )


def make_trial(question_id: QuestionId, suffix: int) -> FinanceExperimentTrial:
    digest = "a" * 64
    direct = TreatmentAudit(
        arm=ForecastArm.DIRECT,
        evidence_snapshot_digest=None,
        evidence_snapshot_bytes=0,
        evidence_item_count=0,
        historical_memory_episode_count=0,
    )
    search_only = TreatmentAudit(
        arm=ForecastArm.SEARCH_ONLY,
        evidence_snapshot_digest=digest,
        evidence_snapshot_bytes=512,
        evidence_item_count=2,
        historical_memory_episode_count=0,
    )
    search_dag = TreatmentAudit(
        arm=ForecastArm.SEARCH_DAG,
        evidence_snapshot_digest=digest,
        evidence_snapshot_bytes=512,
        evidence_item_count=2,
        historical_memory_episode_count=3,
    )
    forecast = make_forecast()
    attempt = make_attempt()
    return FinanceExperimentTrial(
        trial_id=UUID(f"00000000-0000-0000-0000-{suffix:012d}"),
        question_id=question_id,
        repetition_index=0,
        preparation_attempts=(),
        arms=tuple(
            ArmSucceeded(
                arm=audit.arm,
                attempts=(attempt,),
                treatment=audit,
                forecast=forecast,
            )
            for audit in (direct, search_only, search_dag)
        ),
    )


def make_suite() -> FinanceExperimentSuite:
    manifest = make_manifest()
    return FinanceExperimentSuite(
        schema_version="finance-experiment-suite/v1",
        suite_id=UUID("10000000-0000-0000-0000-000000000001"),
        created_at=datetime(2026, 7, 18, 1, tzinfo=UTC),
        status=FinanceSuiteStatus.COMPLETE,
        manifest=manifest,
        trials=tuple(
            make_trial(question.question_id, index)
            for index, question in enumerate(manifest.questions, start=1)
        ),
    )


def make_resolution() -> FinanceResolutionManifest:
    suite = make_suite()
    return FinanceResolutionManifest(
        schema_version="finance-resolution-manifest/v1",
        suite_id=suite.suite_id,
        experiment_manifest_id=suite.manifest.manifest_id,
        suite_sha256="b" * 64,
        entries=(
            FinanceResolutionEntry(
                question_id=suite.manifest.questions[0].question_id,
                outcome_label=OutcomeLabel("Yes"),
                resolved_at=datetime(2027, 1, 2, tzinfo=UTC),
                resolution_source="Official source publication.",
            ),
        ),
    )
