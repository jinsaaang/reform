"""Deterministic lexical retrieval over eligible historical finance DAGs."""

import re
import unicodedata
from dataclasses import dataclass
from typing import Final

from src.domain.finance.memory import ResolvedDagEpisode
from src.domain.finance.retrieval import (
    ExcludedHistoricalDag,
    HistoricalDagQuery,
    HistoricalDagRetrieval,
    HistoricalEpisodeSource,
    RankedHistoricalDag,
    RetrievalFeature,
    ScoreComponent,
    assess_eligibility,
)
from src.domain.finance.search import TargetProfile

_TOKEN_PATTERN: Final = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")
_FEATURE_WEIGHTS: Final = (
    (RetrievalFeature.QUESTION, 4),
    (RetrievalFeature.CONTEXT, 3),
    (RetrievalFeature.DOMAIN, 2),
    (RetrievalFeature.QUESTION_TYPE, 1),
)


def _tokenize(text: str) -> frozenset[str]:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return frozenset(_TOKEN_PATTERN.findall(normalized))


def _candidate_tokens(episode: ResolvedDagEpisode) -> frozenset[str]:
    profile = episode.source_profile
    parts = [
        profile.question_text,
        profile.context or "",
        profile.domain,
        profile.question_type.value,
        profile.resolution_rule or "",
        " ".join(profile.outcome_space),
    ]
    for node in episode.nodes:
        parts.extend(
            (
                node.title,
                node.description,
                node.event_type,
                node.domain,
                " ".join(node.tags),
            )
        )
    for edge in episode.edges:
        parts.extend((edge.relation_type or "", edge.reasoning))
    for impact in episode.impacts:
        parts.extend((impact.direction, impact.reasoning))
    return _tokenize(" ".join(parts))


def _target_features(
    target: TargetProfile,
) -> tuple[tuple[RetrievalFeature, int, frozenset[str]], ...]:
    text_by_feature = (
        (RetrievalFeature.QUESTION, target.question_text),
        (RetrievalFeature.CONTEXT, " ".join(target.context)),
        (RetrievalFeature.DOMAIN, target.domain),
        (RetrievalFeature.QUESTION_TYPE, target.question_type.value),
    )
    weights = dict(_FEATURE_WEIGHTS)
    return tuple(
        (feature, weights[feature], _tokenize(text))
        for feature, text in text_by_feature
    )


def _score_episode(
    episode: ResolvedDagEpisode,
    target: TargetProfile,
) -> RankedHistoricalDag:
    candidate_terms = _candidate_tokens(episode)
    components = tuple(
        ScoreComponent(
            feature=feature,
            weight=weight,
            matched_terms=tuple(sorted(query_terms & candidate_terms)),
        )
        for feature, weight, query_terms in _target_features(target)
    )
    matched_terms = tuple(
        sorted({term for component in components for term in component.matched_terms})
    )
    return RankedHistoricalDag(
        episode=episode,
        score=sum(component.contribution for component in components),
        score_components=components,
        matched_terms=matched_terms,
        audit_markers=(),
    )


@dataclass(frozen=True, slots=True)
class HistoricalDagRetriever:
    """Apply hard eligibility, then a stable question/context lexical ranker."""

    episode_source: HistoricalEpisodeSource

    def retrieve(self, query: HistoricalDagQuery) -> HistoricalDagRetrieval:
        """Load the configured source once and delegate to the pure path."""
        return self.retrieve_from(self.episode_source.load_episodes(), query)

    def retrieve_from(
        self,
        episodes: tuple[ResolvedDagEpisode, ...],
        query: HistoricalDagQuery,
    ) -> HistoricalDagRetrieval:
        """Rank a caller-supplied canonical immutable episode tuple."""
        ranked: list[RankedHistoricalDag] = []
        excluded: list[ExcludedHistoricalDag] = []
        ordered_episodes = sorted(
            episodes,
            key=lambda episode: str(episode.episode_id),
        )
        for episode in ordered_episodes:
            decision = assess_eligibility(
                episode,
                query.target_profile,
                query.policy,
            )
            if decision.eligible:
                scored = _score_episode(episode, query.target_profile)
                ranked.append(
                    RankedHistoricalDag(
                        episode=scored.episode,
                        score=scored.score,
                        score_components=scored.score_components,
                        matched_terms=scored.matched_terms,
                        audit_markers=decision.audit_markers,
                    )
                )
            else:
                excluded.append(
                    ExcludedHistoricalDag(episode=episode, decision=decision)
                )
        ranked.sort(key=lambda item: (-item.score, str(item.episode.episode_id)))
        ranked_candidates = tuple(ranked)
        return HistoricalDagRetrieval(
            selected=ranked_candidates[: query.top_k.value],
            ranked_candidates=ranked_candidates,
            excluded=tuple(excluded),
        )
