"""Strict, credential-free finance experiment manifest contracts."""

from enum import StrEnum, unique
from typing import Annotated, ClassVar, Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator
from pydantic_core import PydanticCustomError

from src.domain.finance.artifact import ForecastArm
from src.domain.finance.memory import OutcomeLabel, QuestionId, QuestionKind
from src.domain.finance.provider import FinanceRunMode, SearchSourcePolicy
from src.domain.finance.retrieval import EligibilityPolicy

Seed = Annotated[int, Field(ge=0, le=(2**63) - 1)]


class _ExperimentModel(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=True,
        str_strip_whitespace=True,
    )


@unique
class FinanceSearchBackend(StrEnum):
    """Closed search backends that an experiment may pin."""

    BING_NEWS_RSS_V1 = "BING_NEWS_RSS_V1"
    OFFLINE_FIXTURE_V1 = "OFFLINE_FIXTURE_V1"
    PUBLIC_DB_METADATA_V1 = "PUBLIC_DB_METADATA_V1"


@unique
class ArmOrderPolicy(StrEnum):
    """Closed arm scheduling policies."""

    FIXED = "fixed"


@unique
class ReasoningEffort(StrEnum):
    """Safe reasoning-effort values forwarded to completion providers."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    XHIGH = "xhigh"


class SafeCompletionSettings(_ExperimentModel):
    """Credential-free completion settings and explicit retry budgets."""

    model: str = Field(min_length=1)
    reasoning_effort: ReasoningEffort
    temperature: Annotated[float, Field(ge=0.0, le=2.0)]
    max_output_tokens: Annotated[int, Field(ge=1)]
    timeout_seconds: Annotated[float, Field(gt=0.0)]
    requested_seed: Seed | None
    provider_retry_budget: Annotated[int, Field(ge=0)]
    schema_retry_budget: Annotated[int, Field(ge=0)]


class JudgeMember(_ExperimentModel):
    """One independently identified judge and its exact safe settings."""

    member_id: str = Field(min_length=1)
    settings: SafeCompletionSettings


class SearchSettings(_ExperimentModel):
    """Pinned first-pass search policy and bounded live retrieval settings."""

    source_policy: SearchSourcePolicy
    backend: FinanceSearchBackend
    result_limit: Annotated[int, Field(ge=1)]
    body_fetch_timeout_seconds: Annotated[float, Field(gt=0.0)]
    provider_retry_budget: Annotated[int, Field(ge=0)]

    @model_validator(mode="after")
    def validate_backend_matches_source_policy(self) -> "SearchSettings":
        """Reject an environment-selectable backend combination."""
        expected_by_policy = {
            SearchSourcePolicy.LIVE_SEARCH: FinanceSearchBackend.BING_NEWS_RSS_V1,
            SearchSourcePolicy.OFFLINE_FIXTURE: (
                FinanceSearchBackend.OFFLINE_FIXTURE_V1
            ),
            SearchSourcePolicy.PUBLIC_DB_METADATA: (
                FinanceSearchBackend.PUBLIC_DB_METADATA_V1
            ),
        }
        expected = expected_by_policy[self.source_policy]
        if self.backend is not expected:
            raise PydanticCustomError(
                "search_backend_mismatch",
                "search backend must match the pinned source policy",
            )
        return self


class RetrievalSettings(_ExperimentModel):
    """Versioned admission, eligibility, and historical retrieval policy."""

    evidence_admission_version: Literal["evidence-admission/v1"]
    eligibility_version: Literal["public-db-bootstrap/v1"]
    retriever_version: Literal["lexical-retriever/v1"]
    eligibility_policy: EligibilityPolicy
    top_k: Annotated[int, Field(ge=1)]


class ProtocolVersions(_ExperimentModel):
    """Prompt and structured-output protocol versions fixed ex ante."""

    forecast_prompt: Literal["forecast-prompt/v1"]
    judge_prompt: Literal["judge-prompt/v1"]
    forecast_result: Literal["forecast-result/v1"]
    judge_result: Literal["judge-result/v1"]
    suite_output: Literal["finance-experiment-suite/v1"]


class FinanceBinaryQuestion(_ExperimentModel):
    """One unresolved binary target without any realized outcome."""

    question_id: Annotated[QuestionId, Field(min_length=1)]
    question_text: str = Field(min_length=1)
    question_type: Literal[QuestionKind.BINARY] = QuestionKind.BINARY
    domain: str = Field(min_length=1)
    context: tuple[str, ...] = Field(min_length=1)
    outcome_space: tuple[Annotated[OutcomeLabel, Field(min_length=1)], ...] = Field(
        min_length=2,
        max_length=2,
    )
    positive_label: Annotated[OutcomeLabel, Field(min_length=1)]
    resolution_rule: str = Field(min_length=1)
    forecast_cutoff: AwareDatetime | None = None

    @model_validator(mode="after")
    def validate_binary_outcomes(self) -> "FinanceBinaryQuestion":
        """Require two unique outcomes and one declared positive label."""
        if len(set(self.outcome_space)) != 2:
            raise PydanticCustomError(
                "duplicate_binary_outcome",
                "binary outcome labels must be unique",
            )
        if self.positive_label not in self.outcome_space:
            raise PydanticCustomError(
                "invalid_positive_label",
                "positive label must belong to the binary outcome space",
            )
        return self


class FinanceExperimentManifest(_ExperimentModel):
    """Complete ordered input contract for a three-arm finance suite."""

    schema_version: Literal["finance-experiment-manifest/v1"]
    manifest_id: str = Field(min_length=1)
    cutoff: AwareDatetime
    run_mode: FinanceRunMode
    temporal_policy_version: Literal[
        "live_publication_filtered/v1",
        "public_db_metadata_proxy/v1",
    ]
    repetitions: Annotated[int, Field(ge=1)]
    root_seed: Seed
    arm_order_policy: ArmOrderPolicy
    arm_order: tuple[ForecastArm, ...] = Field(min_length=3, max_length=3)
    search: SearchSettings
    retrieval: RetrievalSettings
    forecast: SafeCompletionSettings
    judges: tuple[JudgeMember, ...] = Field(min_length=3, max_length=3)
    protocol_versions: ProtocolVersions
    questions: tuple[FinanceBinaryQuestion, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_ordered_manifest(self) -> "FinanceExperimentManifest":
        """Reject ambiguous targets, judges, and three-arm schedules."""
        expected_arms = (
            ForecastArm.DIRECT,
            ForecastArm.SEARCH_ONLY,
            ForecastArm.SEARCH_DAG,
        )
        if self.arm_order != expected_arms:
            raise PydanticCustomError(
                "invalid_arm_order",
                "fixed arm order must be direct, search_only, search_dag",
            )
        question_ids = tuple(question.question_id for question in self.questions)
        if len(question_ids) != len(set(question_ids)):
            raise PydanticCustomError(
                "duplicate_question_id",
                "experiment question IDs must be unique",
            )
        member_ids = tuple(member.member_id for member in self.judges)
        if len(member_ids) != len(set(member_ids)):
            raise PydanticCustomError(
                "duplicate_judge_member",
                "exactly three unique judge members are required",
            )
        if self.search.source_policy is SearchSourcePolicy.PUBLIC_DB_METADATA:
            if (
                self.run_mode is not FinanceRunMode.HISTORICAL_BACKTEST
                or self.temporal_policy_version != "public_db_metadata_proxy/v1"
            ):
                raise PydanticCustomError(
                    "invalid_backtest_search_policy",
                    "public DB metadata search requires historical backtest mode",
                )
            cutoffs = tuple(question.forecast_cutoff for question in self.questions)
            if any(cutoff is None for cutoff in cutoffs):
                raise PydanticCustomError(
                    "missing_backtest_cutoff",
                    "every historical target requires its own forecast cutoff",
                )
            if self.cutoff != max(cutoff for cutoff in cutoffs if cutoff is not None):
                raise PydanticCustomError(
                    "invalid_backtest_manifest_cutoff",
                    "manifest cutoff must equal the latest target cutoff",
                )
        if (
            self.run_mode is FinanceRunMode.CURRENT_UNRESOLVED
            and self.temporal_policy_version != "live_publication_filtered/v1"
        ):
            raise PydanticCustomError(
                "invalid_current_temporal_policy",
                "current experiments require the live publication policy",
            )
        return self


__all__ = [
    "ArmOrderPolicy",
    "FinanceBinaryQuestion",
    "FinanceExperimentManifest",
    "FinanceSearchBackend",
    "JudgeMember",
    "ProtocolVersions",
    "ReasoningEffort",
    "RetrievalSettings",
    "SafeCompletionSettings",
    "SearchSettings",
    "Seed",
]
