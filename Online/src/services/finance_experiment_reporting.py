"""Deterministic, truth-free Markdown reporting for ex-ante suites."""

from decimal import ROUND_HALF_EVEN, Decimal, localcontext
from typing import assert_never

from src.domain.finance.experiment_artifact import (
    PersistedFinanceExperimentSuite,
    PersistedPairCompleted,
    PersistedPairResult,
    PersistedPairUnavailable,
)
from src.domain.finance.experiment_metrics import (
    ExAntePairSummary,
    ExAnteTieReason,
    ForecastPair,
)
from src.domain.finance.experiment_variant_guard import (
    runtime_persisted_variant,
    unexpected_persisted_variant,
)


def _completed_pair(result: PersistedPairResult) -> PersistedPairCompleted | None:
    candidate = runtime_persisted_variant(result)
    match candidate:
        case PersistedPairCompleted() as completed:
            return completed
        case PersistedPairUnavailable():
            return None
        case _:
            assert_never(unexpected_persisted_variant(candidate))


def _mean(values: tuple[Decimal, ...]) -> Decimal | None:
    if not values:
        return None
    with localcontext() as context:
        context.prec = 80
        return sum(values, start=Decimal(0)) / Decimal(len(values))


def format_percentage(value: Decimal | float) -> str:
    """Render one probability with Decimal round-half-even at two places."""
    percentage = (Decimal(str(value)) * Decimal(100)).quantize(
        Decimal("0.01"),
        rounding=ROUND_HALF_EVEN,
    )
    if percentage == 0:
        percentage = Decimal(0)
    return f"{percentage:.2f}%"


def format_percentage_points(value: Decimal | float) -> str:
    """Render a signed probability delta as percentage points."""
    percentage = (Decimal(str(value)) * Decimal(100)).quantize(
        Decimal("0.01"),
        rounding=ROUND_HALF_EVEN,
    )
    return f"{percentage:+.2f}%p"


def build_ex_ante_pair_summaries(
    suite: PersistedFinanceExperimentSuite,
) -> tuple[ExAntePairSummary, ...]:
    """Aggregate explicit panel, member, agreement, and order denominators."""
    summaries: list[ExAntePairSummary] = []
    for pair in ForecastPair:
        completed_list: list[PersistedPairCompleted] = []
        for trial in suite.trials:
            for result in trial.pairs:
                observed = _completed_pair(result)
                if observed is not None and observed.comparison.pair is pair:
                    completed_list.append(observed)
        completed = tuple(completed_list)
        panels = tuple(result.panel for result in completed)
        agreements = tuple(
            panel.agreement for panel in panels if panel.agreement is not None
        )
        order_values = tuple(
            panel.order_consistency
            for panel in panels
            if panel.order_consistency is not None
        )
        summaries.append(
            ExAntePairSummary(
                pair=pair,
                scheduled_trial_count=len(suite.trials),
                completed_panel_count=len(panels),
                preference_eligible_panel_count=sum(
                    panel.preference_eligible for panel in panels
                ),
                no_quorum_panel_count=sum(
                    panel.tie_reason is ExAnteTieReason.NO_QUORUM for panel in panels
                ),
                judge_member_count=sum(
                    panel.first_votes
                    + panel.second_votes
                    + panel.tie_votes
                    + panel.inconsistent_count
                    + panel.invalid_count
                    for panel in panels
                ),
                invalid_member_count=sum(panel.invalid_count for panel in panels),
                inconsistent_member_count=sum(
                    panel.inconsistent_count for panel in panels
                ),
                agreement_observation_count=len(agreements),
                order_consistency_observation_count=len(order_values),
                agreement_mean=_mean(agreements),
                order_consistency_mean=_mean(order_values),
            )
        )
    return tuple(summaries)


def _optional_percentage(value: Decimal | None) -> str:
    return "n/a" if value is None else format_percentage(value)


def render_ex_ante_report(suite: PersistedFinanceExperimentSuite) -> str:
    """Render byte-stable Markdown containing no truth or resolution score."""
    lines = [
        "# Finance Experiment Ex-Ante Report",
        "",
        f"- suite id: `{suite.suite_id}`",
        f"- manifest id: `{suite.manifest.manifest_id}`",
        f"- created at: `{suite.created_at.isoformat()}`",
        f"- status: `{suite.status.value}`",
        f"- scheduled trials: {len(suite.trials)}",
        "",
        "## Pairwise positive-label comparisons",
        "",
        "| Trial | Pair | First | Second | Delta | First argmax | "
        "Second argmax | Majority flip |",
        "|---|---|---:|---:|---:|---|---|---|",
    ]
    for trial in suite.trials:
        for result in trial.pairs:
            completed = _completed_pair(result)
            if completed is None:
                continue
            comparison = completed.comparison
            lines.append(
                "| "
                f"{trial.question_id}/{trial.repetition_index} | "
                f"{comparison.pair.value} | "
                f"{format_percentage(comparison.first_positive_probability)} | "
                f"{format_percentage(comparison.second_positive_probability)} | "
                f"{format_percentage_points(comparison.delta)} | "
                f"{comparison.first_argmax.value} | "
                f"{comparison.second_argmax.value} | "
                f"{comparison.majority_flip.value} |"
            )
    lines.extend(["", "## Panel diagnostics", ""])
    for summary in build_ex_ante_pair_summaries(suite):
        lines.extend(
            [
                f"### {summary.pair.value}",
                "",
                "- completed panels: "
                f"{summary.completed_panel_count}/{summary.scheduled_trial_count}",
                "- preference-eligible panels: "
                f"{summary.preference_eligible_panel_count}/"
                f"{summary.completed_panel_count}",
                "- invalid members: "
                f"{summary.invalid_member_count}/{summary.judge_member_count}",
                "- inconsistent members: "
                f"{summary.inconsistent_member_count}/{summary.judge_member_count}",
                "- no-quorum panels: "
                f"{summary.no_quorum_panel_count}/"
                f"{summary.completed_panel_count}",
                "- agreement observations: "
                f"{summary.agreement_observation_count}/"
                f"{summary.completed_panel_count}; mean "
                f"{_optional_percentage(summary.agreement_mean)}",
                "- order-consistency observations: "
                f"{summary.order_consistency_observation_count}/"
                f"{summary.completed_panel_count}; mean "
                f"{_optional_percentage(summary.order_consistency_mean)}",
                "",
            ]
        )
    return "\n".join(lines)


__all__ = [
    "build_ex_ante_pair_summaries",
    "format_percentage",
    "format_percentage_points",
    "render_ex_ante_report",
]
