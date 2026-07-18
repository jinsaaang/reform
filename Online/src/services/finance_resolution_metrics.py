"""Pure binary Brier analysis over sanitized persisted experiment suites."""

from decimal import Decimal
from hashlib import sha256
from typing import assert_never

from src.domain.finance.experiment_artifact import (
    PersistedArmFailed,
    PersistedArmResult,
    PersistedArmSucceeded,
    PersistedArmUnavailable,
    PersistedExperimentTrial,
    PersistedFinanceExperimentSuite,
)
from src.domain.finance.experiment_metrics import (
    ForecastPair,
    METRIC_TOLERANCE,
    PAIR_ARMS,
)
from src.domain.finance.experiment_resolution import FinanceResolutionManifest
from src.domain.finance.experiment_resolution_metrics import (
    ArmBrierScore,
    BrierDirection,
    FinanceResolutionAnalysis,
    PairBrierScore,
    PairResolutionAggregate,
    QuestionPairResolutionAggregate,
    ResolutionAuditEntry,
    ResolutionManifestAudit,
    ResolutionTrialAnalysis,
)
from src.domain.finance.experiment_variant_guard import (
    runtime_persisted_variant,
    unexpected_persisted_variant,
)
from src.domain.finance.memory import QuestionId


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


def binary_brier(probability: Decimal | float, *, resolved_positive: bool) -> Decimal:
    """Return the exact one-class binary Brier score `(p_positive - y)^2`."""
    declared = Decimal(str(probability))
    target = Decimal(1) if resolved_positive else Decimal(0)
    return (declared - target) ** 2


def binary_accuracy(
    probability: Decimal | float,
    *,
    resolved_positive: bool,
) -> bool:
    """Classify at p(positive) >= 0.5 and compare with the resolved label."""
    predicted_positive = Decimal(str(probability)) >= Decimal("0.5")
    return predicted_positive is resolved_positive


def brier_direction(first: Decimal, second: Decimal) -> BrierDirection:
    """Classify the second arm against the first using the fixed tolerance."""
    delta = second - first
    if abs(delta) <= METRIC_TOLERANCE:
        return BrierDirection.UNCHANGED
    return BrierDirection.BETTER if delta < 0 else BrierDirection.WORSE


def _trial_analysis(
    trial: PersistedExperimentTrial,
    resolved_positive: bool,
) -> ResolutionTrialAnalysis:
    scores: list[ArmBrierScore] = []
    for arm in trial.arms:
        successful = _successful_arm(arm)
        if successful is None:
            continue
        scores.append(
            ArmBrierScore(
                arm=successful.arm,
                positive_probability=successful.positive_probability,
                brier=binary_brier(
                    successful.positive_probability,
                    resolved_positive=resolved_positive,
                ),
                correct=binary_accuracy(
                    successful.positive_probability,
                    resolved_positive=resolved_positive,
                ),
            )
        )
    arm_scores = tuple(scores)
    by_arm = {score.arm: score for score in arm_scores}
    pair_scores: list[PairBrierScore] = []
    for pair, (first_arm, second_arm) in PAIR_ARMS.items():
        first = by_arm.get(first_arm)
        second = by_arm.get(second_arm)
        if first is None or second is None:
            continue
        pair_scores.append(
            PairBrierScore(
                pair=pair,
                first_brier=first.brier,
                second_brier=second.brier,
                first_correct=first.correct,
                second_correct=second.correct,
                direction=brier_direction(first.brier, second.brier),
            )
        )
    return ResolutionTrialAnalysis(
        trial_id=trial.trial_id,
        question_id=trial.question_id,
        repetition_index=trial.repetition_index,
        resolved_positive=resolved_positive,
        arm_scores=arm_scores,
        pair_scores=tuple(pair_scores),
    )


def _question_pair_aggregates(
    trials: tuple[ResolutionTrialAnalysis, ...],
    resolved_question_ids: tuple[QuestionId, ...],
) -> tuple[QuestionPairResolutionAggregate, ...]:
    aggregates: list[QuestionPairResolutionAggregate] = []
    for question_id in resolved_question_ids:
        question_trials = tuple(
            trial for trial in trials if trial.question_id == question_id
        )
        for pair in ForecastPair:
            scores = tuple(
                score
                for trial in question_trials
                for score in trial.pair_scores
                if score.pair is pair
            )
            if not scores:
                continue
            mean_first = sum(
                (score.first_brier for score in scores),
                start=Decimal(0),
            ) / Decimal(len(scores))
            mean_second = sum(
                (score.second_brier for score in scores),
                start=Decimal(0),
            ) / Decimal(len(scores))
            mean_first_accuracy = sum(
                (Decimal(int(score.first_correct)) for score in scores),
                start=Decimal(0),
            ) / Decimal(len(scores))
            mean_second_accuracy = sum(
                (Decimal(int(score.second_correct)) for score in scores),
                start=Decimal(0),
            ) / Decimal(len(scores))
            aggregates.append(
                QuestionPairResolutionAggregate(
                    question_id=question_id,
                    pair=pair,
                    successful_trial_count=len(scores),
                    eligible_trial_count=len(question_trials),
                    mean_first_brier=mean_first,
                    mean_second_brier=mean_second,
                    mean_first_accuracy=mean_first_accuracy,
                    mean_second_accuracy=mean_second_accuracy,
                    direction=brier_direction(mean_first, mean_second),
                )
            )
    return tuple(aggregates)


def _pair_aggregates(
    suite: PersistedFinanceExperimentSuite,
    resolution: FinanceResolutionManifest,
    question_aggregates: tuple[QuestionPairResolutionAggregate, ...],
) -> tuple[PairResolutionAggregate, ...]:
    resolved_ids = {entry.question_id for entry in resolution.entries}
    eligible_trial_count = sum(
        trial.question_id in resolved_ids for trial in suite.trials
    )
    results: list[PairResolutionAggregate] = []
    for pair in ForecastPair:
        questions = tuple(item for item in question_aggregates if item.pair is pair)
        if questions:
            macro_first = sum(
                (item.mean_first_brier for item in questions),
                start=Decimal(0),
            ) / Decimal(len(questions))
            macro_second = sum(
                (item.mean_second_brier for item in questions),
                start=Decimal(0),
            ) / Decimal(len(questions))
            macro_first_accuracy = sum(
                (item.mean_first_accuracy for item in questions),
                start=Decimal(0),
            ) / Decimal(len(questions))
            macro_second_accuracy = sum(
                (item.mean_second_accuracy for item in questions),
                start=Decimal(0),
            ) / Decimal(len(questions))
            direction = brier_direction(macro_first, macro_second)
        else:
            macro_first = None
            macro_second = None
            macro_first_accuracy = None
            macro_second_accuracy = None
            direction = None
        results.append(
            PairResolutionAggregate(
                pair=pair,
                resolved_question_count=len(resolution.entries),
                suite_question_count=len(suite.manifest.questions),
                successful_question_count=len(questions),
                eligible_question_count=len(resolution.entries),
                successful_trial_count=sum(
                    item.successful_trial_count for item in questions
                ),
                eligible_trial_count=eligible_trial_count,
                macro_first_brier=macro_first,
                macro_second_brier=macro_second,
                macro_first_accuracy=macro_first_accuracy,
                macro_second_accuracy=macro_second_accuracy,
                direction=direction,
            )
        )
    return tuple(results)


def build_finance_resolution_analysis(
    suite: PersistedFinanceExperimentSuite,
    source_suite_sha256: str,
    resolution: FinanceResolutionManifest,
) -> FinanceResolutionAnalysis:
    """Score resolved trials, then question means, then equal-weight macro means."""
    questions = {
        question.question_id: question for question in suite.manifest.questions
    }
    resolution_by_question = {entry.question_id: entry for entry in resolution.entries}
    trials = tuple(
        _trial_analysis(
            trial,
            resolution_by_question[trial.question_id].outcome_label
            == questions[trial.question_id].positive_label,
        )
        for trial in suite.trials
        if trial.question_id in resolution_by_question
    )
    resolved_question_ids = tuple(entry.question_id for entry in resolution.entries)
    question_aggregates = _question_pair_aggregates(
        trials,
        resolved_question_ids,
    )
    resolution_audit = ResolutionManifestAudit(
        suite_id=resolution.suite_id,
        experiment_manifest_id=resolution.experiment_manifest_id,
        suite_sha256=resolution.suite_sha256,
        entries=tuple(
            ResolutionAuditEntry(
                question_id=entry.question_id,
                outcome_label=entry.outcome_label,
                resolved_at=entry.resolved_at,
                resolution_source_sha256=sha256(
                    entry.resolution_source.encode("utf-8")
                ).hexdigest(),
            )
            for entry in resolution.entries
        ),
    )
    return FinanceResolutionAnalysis(
        source_suite_id=suite.suite_id,
        source_manifest_id=suite.manifest.manifest_id,
        source_suite_sha256=source_suite_sha256,
        resolution=resolution_audit,
        resolved_question_count=len(resolution.entries),
        suite_question_count=len(suite.manifest.questions),
        trials=trials,
        question_pair_aggregates=question_aggregates,
        pair_aggregates=_pair_aggregates(suite, resolution, question_aggregates),
    )


__all__ = [
    "binary_accuracy",
    "binary_brier",
    "brier_direction",
    "build_finance_resolution_analysis",
]
