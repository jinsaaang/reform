"""Deterministic Markdown rendering for derived resolution analysis."""

from decimal import ROUND_HALF_EVEN, Decimal

from src.domain.finance.experiment_resolution_metrics import FinanceResolutionAnalysis


def format_brier(value: Decimal) -> str:
    """Render one binary Brier score with round-half-even."""
    rounded = value.quantize(Decimal("0.0001"), rounding=ROUND_HALF_EVEN)
    return f"{rounded:.4f}"


def render_resolution_report(analysis: FinanceResolutionAnalysis) -> str:
    """Render a deterministic report distinct from the ex-ante bundle."""
    outcome_by_question = {
        entry.question_id: entry.outcome_label for entry in analysis.resolution.entries
    }
    lines = [
        "# Finance Experiment Resolution Analysis",
        "",
        f"- source suite id: `{analysis.source_suite_id}`",
        f"- source manifest id: `{analysis.source_manifest_id}`",
        f"- source suite SHA-256: `{analysis.source_suite_sha256}`",
        "- resolved questions: "
        f"{analysis.resolved_question_count}/{analysis.suite_question_count}",
        "",
        "## Trial binary Brier scores",
        "",
        "| Trial | Resolved answer | Arm | p(positive) | Brier | Correct |",
        "|---|---|---|---:|---:|:---:|",
    ]
    for trial in analysis.trials:
        for score in trial.arm_scores:
            lines.append(
                f"| {trial.question_id}/{trial.repetition_index} | "
                f"{outcome_by_question[trial.question_id]} | "
                f"{score.arm.value} | {score.positive_probability} | "
                f"{format_brier(score.brier)} | "
                f"{'yes' if score.correct else 'no'} |"
            )
    lines.extend(
        [
            "",
            "## Question-level pair outcome direction",
            "",
            "| Question | Pair | First Brier | Second Brier | "
            "First Accuracy | Second Accuracy | Direction |",
            "|---|---|---:|---:|---:|---:|---|",
        ]
    )
    for aggregate in analysis.question_pair_aggregates:
        lines.append(
            f"| {aggregate.question_id} | {aggregate.pair.value} | "
            f"{format_brier(aggregate.mean_first_brier)} | "
            f"{format_brier(aggregate.mean_second_brier)} | "
            f"{format_brier(aggregate.mean_first_accuracy)} | "
            f"{format_brier(aggregate.mean_second_accuracy)} | "
            f"{aggregate.direction.value} |"
        )
    lines.extend(
        [
            "",
            "## Pair macro aggregation",
            "",
            "| Pair | Questions | Trials | First Brier | Second Brier | "
            "First Accuracy | Second Accuracy | Direction |",
            "|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for aggregate in analysis.pair_aggregates:
        first = (
            "n/a"
            if aggregate.macro_first_brier is None
            else format_brier(aggregate.macro_first_brier)
        )
        second = (
            "n/a"
            if aggregate.macro_second_brier is None
            else format_brier(aggregate.macro_second_brier)
        )
        first_accuracy = (
            "n/a"
            if aggregate.macro_first_accuracy is None
            else format_brier(aggregate.macro_first_accuracy)
        )
        second_accuracy = (
            "n/a"
            if aggregate.macro_second_accuracy is None
            else format_brier(aggregate.macro_second_accuracy)
        )
        direction = "n/a" if aggregate.direction is None else aggregate.direction.value
        lines.append(
            f"| {aggregate.pair.value} | "
            f"{aggregate.successful_question_count}/"
            f"{aggregate.eligible_question_count} "
            f"(resolved {aggregate.resolved_question_count}/"
            f"{aggregate.suite_question_count}) | "
            f"{aggregate.successful_trial_count}/"
            f"{aggregate.eligible_trial_count} | "
            f"{first} | {second} | {first_accuracy} | "
            f"{second_accuracy} | {direction} |"
        )
    return "\n".join(lines) + "\n"


__all__ = ["format_brier", "render_resolution_report"]
