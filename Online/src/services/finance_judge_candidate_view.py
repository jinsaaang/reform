"""Sanitized target, evidence, and candidate forecast projections."""

from src.domain.finance.forecast import OutcomeProbability
from src.domain.finance.judge_source import CanonicalJudgeCandidate
from src.domain.finance.judge_views import (
    SanitizedCandidateView,
    SanitizedEvidenceView,
    SanitizedScenarioView,
    SanitizedTargetView,
)
from src.domain.finance.memory import OutcomeLabel
from src.domain.finance.search import EvidenceItem, TargetProfile
from src.services.finance_aliases import RunAliasTable
from src.services.finance_judge_alias_sources import scenario_alias_identity


def _probabilities(
    values: tuple[OutcomeProbability, ...],
    aliases: RunAliasTable,
) -> tuple[OutcomeProbability, ...]:
    return tuple(
        OutcomeProbability(
            label=OutcomeLabel(aliases.replace_text(str(item.label))),
            probability=item.probability,
        )
        for item in values
    )


def build_sanitized_target_view(
    target: TargetProfile,
    aliases: RunAliasTable,
) -> SanitizedTargetView:
    return SanitizedTargetView(
        question_text=aliases.replace_text(target.question_text),
        question_type=target.question_type,
        domain=aliases.replace_text(target.domain),
        context=tuple(aliases.replace_text(item) for item in target.context),
        cutoff=target.cutoff,
        outcome_space=tuple(
            aliases.replace_text(item) for item in target.outcome_space
        ),
        resolution_rule=(
            None
            if target.resolution_rule is None
            else aliases.replace_text(target.resolution_rule)
        ),
    )


def build_sanitized_evidence_views(
    evidence: tuple[EvidenceItem, ...],
    aliases: RunAliasTable,
) -> tuple[SanitizedEvidenceView, ...]:
    return tuple(
        SanitizedEvidenceView(
            alias=aliases.alias_for(str(item.evidence_id)),
            claim=aliases.replace_text(item.claim),
            citation=aliases.replace_text(item.citation),
            available_at=item.available_at,
            retrieved_at=item.retrieved_at,
            direction=item.direction,
            context_slot=aliases.replace_text(item.context_slot),
        )
        for item in evidence
    )


def build_sanitized_candidate_view(
    candidate: CanonicalJudgeCandidate,
    aliases: RunAliasTable,
) -> SanitizedCandidateView:
    forecast = candidate.arm_result.forecast
    return SanitizedCandidateView(
        alias=candidate.alias,
        scenarios=tuple(
            SanitizedScenarioView(
                alias=aliases.alias_for(
                    scenario_alias_identity(
                        candidate.alias,
                        str(scenario.scenario_id),
                    )
                ),
                name=aliases.replace_text(scenario.name),
                reasoning_steps=tuple(
                    aliases.replace_text(step) for step in scenario.reasoning_steps
                ),
                probability=scenario.probability,
                conditional_outcomes=_probabilities(
                    scenario.conditional_outcomes,
                    aliases,
                ),
                evidence_aliases=tuple(
                    aliases.alias_for(str(item)) for item in scenario.evidence_ids
                ),
                memory_aliases=tuple(
                    aliases.alias_for(str(item.dag_id))
                    for item in scenario.historical_dag_references
                ),
                assumptions=tuple(
                    aliases.replace_text(item) for item in scenario.assumptions
                ),
                triggers=tuple(
                    aliases.replace_text(item) for item in scenario.triggers
                ),
                disconfirmers=tuple(
                    aliases.replace_text(item) for item in scenario.disconfirmers
                ),
                uncertainty=aliases.replace_text(scenario.uncertainty),
            )
            for scenario in forecast.scenarios
        ),
        outcome_probabilities=_probabilities(
            forecast.outcome_probabilities,
            aliases,
        ),
        explanation=aliases.replace_text(forecast.explanation),
    )


__all__ = [
    "build_sanitized_candidate_view",
    "build_sanitized_evidence_views",
    "build_sanitized_target_view",
]
