"""Deterministic alias assignment before finance judge-view truncation."""

from dataclasses import dataclass

from src.domain.finance.judge_source import CanonicalJudgeCandidate
from src.domain.finance.judge_views import NeutralCandidate
from src.domain.finance.memory import ResolvedDagEpisode
from src.domain.finance.sanitized_artifact import AliasKind
from src.domain.finance.search import EvidenceItem
from src.services.finance_aliases import AliasSource


@dataclass(frozen=True, slots=True)
class JudgeAliasInputs:
    candidates: tuple[CanonicalJudgeCandidate, CanonicalJudgeCandidate]
    evidence: tuple[EvidenceItem, ...]
    memory: tuple[ResolvedDagEpisode, ...]
    transient_identities: tuple[str, ...]


def scenario_alias_identity(candidate: NeutralCandidate, scenario_id: str) -> str:
    """Namespace provider-local scenario IDs by their neutral candidate."""
    return f"scenario:{candidate.value}:{scenario_id}"


def _memory_aliases(
    memory: tuple[ResolvedDagEpisode, ...],
) -> tuple[AliasSource, ...]:
    sources: list[AliasSource] = []
    article_ids: list[str] = []
    for memory_index, episode in enumerate(memory, start=1):
        base = f"memory_{memory_index:03d}"
        sources.extend(
            (
                AliasSource(AliasKind.MEMORY, str(episode.dag_id), base),
                AliasSource(
                    AliasKind.MEMORY,
                    str(episode.episode_id),
                    f"{base}_episode",
                ),
                AliasSource(
                    AliasKind.MEMORY,
                    str(episode.question_id),
                    f"{base}_question",
                ),
            )
        )
        for node_index, node in enumerate(episode.nodes, start=1):
            sources.append(
                AliasSource(
                    AliasKind.EVENT,
                    str(node.event_id),
                    f"{base}_node_{node_index:03d}",
                )
            )
            article_ids.extend(str(item) for item in node.source_article_ids)
        for edge_index, edge in enumerate(episode.edges, start=1):
            sources.append(
                AliasSource(
                    AliasKind.EDGE,
                    str(edge.edge_id),
                    f"{base}_edge_{edge_index:03d}",
                )
            )
            article_ids.extend(str(item) for item in edge.evidence_article_ids)
        for impact_index, impact in enumerate(episode.impacts, start=1):
            sources.append(
                AliasSource(
                    AliasKind.IMPACT,
                    str(impact.impact_id),
                    f"{base}_impact_{impact_index:03d}",
                )
            )
            article_ids.extend(str(item) for item in impact.evidence_article_ids)
    seen_articles: set[str] = set()
    for article_id in article_ids:
        if article_id in seen_articles:
            continue
        seen_articles.add(article_id)
        sources.append(
            AliasSource(
                AliasKind.ARTICLE,
                article_id,
                f"citation_{len(seen_articles):03d}",
            )
        )
    return tuple(sources)


def collect_judge_alias_sources(inputs: JudgeAliasInputs) -> tuple[AliasSource, ...]:
    """Assign every known identity before any bounded view is selected."""
    target = inputs.candidates[0].forecast_input.target_profile
    sources: list[AliasSource] = [
        AliasSource(AliasKind.TARGET, str(target.question_id), "target")
    ]
    for index, evidence in enumerate(inputs.evidence, start=1):
        sources.extend(
            (
                AliasSource(
                    AliasKind.EVIDENCE,
                    str(evidence.evidence_id),
                    f"evidence_{index:03d}",
                ),
                AliasSource(
                    AliasKind.IDENTITY,
                    evidence.content_hash,
                    f"redacted_hash_{index:03d}",
                ),
            )
        )
    sources.extend(_memory_aliases(inputs.memory))
    for candidate in inputs.candidates:
        base = candidate.alias.value
        arm = candidate.arm_result.arm
        sources.extend(
            (
                AliasSource(
                    AliasKind.IDENTITY,
                    arm.value,
                    f"{base}_arm_value",
                    forbid_original=False,
                ),
                AliasSource(
                    AliasKind.IDENTITY,
                    arm.name,
                    f"{base}_arm_name",
                    forbid_original=False,
                ),
            )
        )
        for scenario_index, scenario in enumerate(
            candidate.arm_result.forecast.scenarios,
            start=1,
        ):
            sources.append(
                AliasSource(
                    AliasKind.SCENARIO,
                    scenario_alias_identity(
                        candidate.alias,
                        str(scenario.scenario_id),
                    ),
                    f"{base}_scenario_{scenario_index:03d}",
                    forbid_original=False,
                )
            )
    for index, identity in enumerate(
        sorted(set(inputs.transient_identities)),
        start=1,
    ):
        sources.append(
            AliasSource(
                AliasKind.IDENTITY,
                identity,
                f"redacted_identity_{index:03d}",
            )
        )
    return tuple(sources)


__all__ = [
    "JudgeAliasInputs",
    "collect_judge_alias_sources",
    "scenario_alias_identity",
]
