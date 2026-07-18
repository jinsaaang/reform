"""Sanitized persisted-suite contracts for finance experiments."""

from decimal import Decimal
from typing import Annotated, ClassVar, Literal, assert_never
from uuid import UUID

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator
from pydantic_core import PydanticCustomError

from src.domain.finance.artifact import ForecastArm
from src.domain.finance.experiment_artifact_records import (
    PairUnavailableReason as PairUnavailableReason,
    PersistedArmFailed,
    PersistedArmResult,
    PersistedArmSucceeded,
    PersistedArmUnavailable,
    PersistedPairCompleted,
    PersistedPairResult,
    PersistedPairUnavailable,
    ProviderExchangeDigest,
    ProviderExchangeKind,
    SanitizedArmReasoning as SanitizedArmReasoning,
)
from src.domain.finance.experiment_manifest import FinanceExperimentManifest
from src.domain.finance.experiment_metrics import ForecastPair, PAIR_ARMS
from src.domain.finance.experiment_results import FinanceSuiteStatus
from src.domain.finance.experiment_variant_guard import (
    runtime_persisted_variant,
    unexpected_persisted_variant,
)
from src.domain.finance.memory import QuestionId


class _PersistedSuiteModel(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=True,
        str_strip_whitespace=True,
    )


def _successful_arm(
    result: PersistedArmResult,
) -> PersistedArmSucceeded | None:
    candidate = runtime_persisted_variant(result)
    match candidate:
        case PersistedArmSucceeded() as successful:
            return successful
        case PersistedArmUnavailable() | PersistedArmFailed():
            return None
        case _:
            assert_never(unexpected_persisted_variant(candidate))


def _completed_pair(
    result: PersistedPairResult,
) -> PersistedPairCompleted | None:
    candidate = runtime_persisted_variant(result)
    match candidate:
        case PersistedPairCompleted() as completed:
            return completed
        case PersistedPairUnavailable():
            return None
        case _:
            assert_never(unexpected_persisted_variant(candidate))


def _pair_identity(result: PersistedPairResult) -> ForecastPair:
    candidate = runtime_persisted_variant(result)
    match candidate:
        case PersistedPairCompleted() as completed:
            return completed.comparison.pair
        case PersistedPairUnavailable() as unavailable:
            return unavailable.pair
        case _:
            assert_never(unexpected_persisted_variant(candidate))


class PersistedExperimentTrial(_PersistedSuiteModel):
    trial_id: UUID
    question_id: QuestionId
    repetition_index: Annotated[int, Field(ge=0)]
    preparation_exchanges: tuple[ProviderExchangeDigest, ...]
    arms: tuple[PersistedArmResult, ...] = Field(min_length=3, max_length=3)
    pairs: tuple[PersistedPairResult, ...] = Field(min_length=3, max_length=3)

    @model_validator(mode="after")
    def validate_ordered_terminals(self) -> "PersistedExperimentTrial":
        if any(
            item.kind is not ProviderExchangeKind.SEARCH
            for item in self.preparation_exchanges
        ):
            raise PydanticCustomError(
                "persisted_preparation_exchange_kind",
                "preparation exchanges must be search provider interactions",
            )
        if tuple(item.arm for item in self.arms) != tuple(ForecastArm):
            raise PydanticCustomError(
                "persisted_arm_order",
                "persisted arms must follow direct, search_only, search_dag order",
            )
        actual_pairs = tuple(_pair_identity(item) for item in self.pairs)
        if actual_pairs != tuple(ForecastPair):
            raise PydanticCustomError(
                "persisted_pair_order",
                "persisted pairs must follow A-B, B-C, A-C order",
            )
        successful_arms = tuple(
            successful
            for item in self.arms
            if (successful := _successful_arm(item)) is not None
        )
        succeeded = {item.arm: item for item in successful_arms}
        for item in self.pairs:
            completed = _completed_pair(item)
            if completed is None:
                continue
            comparison = completed.comparison
            first_arm, second_arm = PAIR_ARMS[comparison.pair]
            first = succeeded.get(first_arm)
            second = succeeded.get(second_arm)
            if (
                first is None
                or second is None
                or comparison.first_positive_probability != first.positive_probability
                or comparison.second_positive_probability != second.positive_probability
            ):
                raise PydanticCustomError(
                    "persisted_pair_probability_mismatch",
                    "pair comparison must use its successful arm probabilities",
                )
        search_only = self.arms[1].treatment
        search_dag = self.arms[2].treatment
        if (
            search_only is not None
            and search_dag is not None
            and (
                search_only.evidence_snapshot_digest
                != search_dag.evidence_snapshot_digest
                or search_only.evidence_snapshot_bytes
                != search_dag.evidence_snapshot_bytes
            )
        ):
            raise PydanticCustomError(
                "persisted_fixed_snapshot_mismatch",
                "search_only and search_dag must share snapshot digest and bytes",
            )
        return self


class PersistedFinanceExperimentSuite(_PersistedSuiteModel):
    schema_version: Literal["finance-experiment-suite-artifact/v1"] = (
        "finance-experiment-suite-artifact/v1"
    )
    suite_id: UUID
    created_at: AwareDatetime
    status: FinanceSuiteStatus
    manifest: FinanceExperimentManifest
    trials: tuple[PersistedExperimentTrial, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_schedule_and_status(self) -> "PersistedFinanceExperimentSuite":
        expected = tuple(
            (question.question_id, repetition)
            for question in self.manifest.questions
            for repetition in range(self.manifest.repetitions)
        )
        actual = tuple(
            (trial.question_id, trial.repetition_index) for trial in self.trials
        )
        if actual != expected:
            raise PydanticCustomError(
                "persisted_suite_schedule",
                "persisted trials must match the manifest schedule",
            )
        targets = {
            question.question_id: question for question in self.manifest.questions
        }
        for trial in self.trials:
            target = targets[trial.question_id]
            for arm in trial.arms:
                successful = _successful_arm(arm)
                if successful is None:
                    continue
                reasoning = successful.reasoning
                target_view = reasoning.target
                target_matches = (
                    target_view.question_type is target.question_type
                    and target_view.domain == target.domain
                    and target_view.cutoff
                    == (target.forecast_cutoff or self.manifest.cutoff)
                    and target_view.outcome_space == target.outcome_space
                )
                positives = tuple(
                    item
                    for item in reasoning.candidate.outcome_probabilities
                    if item.label == target.positive_label
                )
                reasoning_matches = (
                    len(positives) == 1
                    and Decimal(str(positives[0].probability))
                    == successful.positive_probability
                    and len(reasoning.evidence)
                    == successful.treatment.evidence_item_count
                    and len(reasoning.memory)
                    == successful.treatment.historical_memory_episode_count
                )
                if not target_matches or not reasoning_matches:
                    raise PydanticCustomError(
                        "persisted_reasoning_mismatch",
                        "sanitized reasoning must match manifest and arm audit",
                    )
        success_count = sum(
            _successful_arm(item) is not None
            for trial in self.trials
            for item in trial.arms
        )
        pair_unavailable = any(
            _completed_pair(pair) is None
            for trial in self.trials
            for pair in trial.pairs
        )
        terminal_count = sum(len(trial.arms) for trial in self.trials)
        expected_status = (
            FinanceSuiteStatus.FAILED
            if success_count == 0
            else FinanceSuiteStatus.PARTIAL
            if success_count < terminal_count or pair_unavailable
            else FinanceSuiteStatus.COMPLETE
        )
        if self.status is not expected_status:
            raise PydanticCustomError(
                "persisted_suite_status",
                "suite status must match arm and pair terminal records",
            )
        return self
