"""Bounded, non-dangling historical memory views for finance judging."""

from src.domain.finance.judge_views import (
    SanitizedHistoricalQuestionView,
    SanitizedMemoryEdgeView,
    SanitizedMemoryImpactView,
    SanitizedMemoryNodeView,
    SanitizedMemoryView,
)
from src.domain.finance.memory import HistoricalOutcomeValue, ResolvedDagEpisode
from src.services.finance_aliases import RunAliasTable


def _text(value: str | None, aliases: RunAliasTable) -> str | None:
    return None if value is None else aliases.replace_text(value)


def _outcome(
    value: HistoricalOutcomeValue,
    aliases: RunAliasTable,
) -> HistoricalOutcomeValue:
    if isinstance(value, str):
        return aliases.replace_text(value)
    return value


def _memory_view(
    episode: ResolvedDagEpisode,
    aliases: RunAliasTable,
) -> SanitizedMemoryView:
    retained_nodes = episode.nodes[:12]
    retained_node_ids = {node.event_id for node in retained_nodes}
    retained_edges = tuple(
        edge
        for edge in episode.edges
        if edge.source_event_id in retained_node_ids
        and edge.target_event_id in retained_node_ids
    )[:16]
    retained_edge_ids = {edge.edge_id for edge in retained_edges}
    retained_impacts = tuple(
        impact
        for impact in episode.impacts
        if impact.event_id in retained_node_ids
        and impact.outcome_event_id in retained_node_ids
        and set(impact.causal_edge_ids).issubset(retained_edge_ids)
    )[:8]
    profile = episode.source_profile
    return SanitizedMemoryView(
        alias=aliases.alias_for(str(episode.dag_id)),
        question=SanitizedHistoricalQuestionView(
            question_text=aliases.replace_text(profile.question_text),
            question_type=profile.question_type,
            domain=aliases.replace_text(profile.domain),
            source=aliases.replace_text(profile.source),
            context=_text(profile.context, aliases),
            outcome_space=tuple(
                aliases.replace_text(label) for label in profile.outcome_space
            ),
            resolution_rule=_text(profile.resolution_rule, aliases),
            quantity_unit=_text(profile.quantity_unit, aliases),
        ),
        resolution_date=episode.historical_resolution.resolution_date_proxy,
        resolved_historical_outcome=_outcome(
            episode.historical_outcome.value,
            aliases,
        ),
        nodes=tuple(
            SanitizedMemoryNodeView(
                alias=aliases.alias_for(str(node.event_id)),
                title=aliases.replace_text(node.title),
                description=aliases.replace_text(node.description),
                event_type=aliases.replace_text(node.event_type),
                domain=aliases.replace_text(node.domain),
                tags=tuple(aliases.replace_text(tag) for tag in node.tags),
                occurred_at=node.occurred_at,
                predicted_at=node.predicted_at,
                source_citation_aliases=tuple(
                    aliases.alias_for(str(item)) for item in node.source_article_ids
                ),
                is_outcome=node.is_outcome,
                outcome_scenario=_text(node.outcome_scenario, aliases),
                is_actual_outcome=node.is_actual_outcome,
            )
            for node in retained_nodes
        ),
        edges=tuple(
            SanitizedMemoryEdgeView(
                alias=aliases.alias_for(str(edge.edge_id)),
                source_node_alias=aliases.alias_for(str(edge.source_event_id)),
                target_node_alias=aliases.alias_for(str(edge.target_event_id)),
                relation_type=_text(edge.relation_type, aliases),
                strength=edge.strength,
                confidence=edge.confidence,
                time_lag_hours=edge.time_lag_hours,
                reasoning=aliases.replace_text(edge.reasoning),
                evidence_citation_aliases=tuple(
                    aliases.alias_for(str(item)) for item in edge.evidence_article_ids
                ),
            )
            for edge in retained_edges
        ),
        impacts=tuple(
            SanitizedMemoryImpactView(
                alias=aliases.alias_for(str(impact.impact_id)),
                event_alias=aliases.alias_for(str(impact.event_id)),
                outcome_event_alias=aliases.alias_for(str(impact.outcome_event_id)),
                direction=aliases.replace_text(impact.direction),
                magnitude=impact.magnitude,
                confidence=impact.confidence,
                reasoning=aliases.replace_text(impact.reasoning),
                evidence_citation_aliases=tuple(
                    aliases.alias_for(str(item)) for item in impact.evidence_article_ids
                ),
                causal_edge_aliases=tuple(
                    aliases.alias_for(str(item)) for item in impact.causal_edge_ids
                ),
            )
            for impact in retained_impacts
        ),
        omitted_node_count=len(episode.nodes) - len(retained_nodes),
        omitted_edge_count=len(episode.edges) - len(retained_edges),
        omitted_impact_count=len(episode.impacts) - len(retained_impacts),
    )


def build_sanitized_memory_views(
    episodes: tuple[ResolvedDagEpisode, ...],
    aliases: RunAliasTable,
) -> tuple[SanitizedMemoryView, ...]:
    """Preserve retriever rank while bounding each episode independently."""
    return tuple(_memory_view(episode, aliases) for episode in episodes)


__all__ = ["build_sanitized_memory_views"]
