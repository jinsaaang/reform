"""Pure partition regressions for canonical historical retrieval audits."""

from dataclasses import dataclass, replace

from src.domain.finance.memory import ResolvedDagEpisode
from src.domain.finance.pipeline import retrieval_matches_canonical_memory
from src.domain.finance.retrieval import (
    EligibilityDecision,
    EligibilityPolicy,
    ExcludedHistoricalDag,
    ExclusionReason,
    HistoricalDagQuery,
    HistoricalDagRetrieval,
    TopK,
)
from src.services.historical_dag_retriever import HistoricalDagRetriever
from tests.fixtures.finance_pipeline import make_target
from tests.unit.domain.finance._factories import make_episode


@dataclass(frozen=True, slots=True)
class _EpisodeSource:
    episodes: tuple[ResolvedDagEpisode, ...]

    def load_episodes(self) -> tuple[ResolvedDagEpisode, ...]:
        return self.episodes


def _eligible_episodes() -> tuple[ResolvedDagEpisode, ...]:
    return tuple(make_episode(episode_id=name) for name in ("alpha", "beta", "gamma"))


def _retrieve(
    episodes: tuple[ResolvedDagEpisode, ...],
    top_k: int = 2,
) -> HistoricalDagRetrieval:
    query = HistoricalDagQuery(
        target_profile=make_target(),
        policy=EligibilityPolicy.PUBLIC_DB_BOOTSTRAP,
        top_k=TopK(top_k),
    )
    return HistoricalDagRetriever(_EpisodeSource(episodes)).retrieve_from(
        episodes,
        query,
    )


class TestCanonicalRetrievalPartition:
    def test_should_reject_duplicate_ranked_episode_id(self) -> None:
        # Given: the complete canonical set has one ranked entry duplicated
        episodes = _eligible_episodes()
        result = _retrieve(episodes)
        malformed = replace(
            result,
            ranked_candidates=(
                result.ranked_candidates[0],
                result.ranked_candidates[0],
                result.ranked_candidates[2],
            ),
        )

        # When: the retrieval partition crosses the canonical boundary
        matches = retrieval_matches_canonical_memory(episodes, malformed, TopK(2))

        # Then: ranked IDs are not accepted as a set-like approximation
        assert not matches

    def test_should_reject_duplicate_excluded_episode_id(self) -> None:
        # Given: one post-cutoff canonical episode is excluded twice
        episodes = (
            make_episode(episode_id="past"),
            make_episode(episode_id="future", resolution_year=2027),
        )
        result = _retrieve(episodes)
        malformed = replace(
            result,
            excluded=(result.excluded[0], result.excluded[0]),
        )

        # When: the retrieval partition crosses the canonical boundary
        matches = retrieval_matches_canonical_memory(episodes, malformed, TopK(2))

        # Then: excluded IDs must also be unique
        assert not matches

    def test_should_reject_ranked_and_excluded_overlap(self) -> None:
        # Given: one eligible ranked episode is also labeled excluded
        episodes = _eligible_episodes()
        result = _retrieve(episodes)
        overlap = ExcludedHistoricalDag(
            episode=result.ranked_candidates[0].episode,
            decision=EligibilityDecision(
                exclusion_reasons=(ExclusionReason.NEAR_DUPLICATE,),
                audit_markers=(),
            ),
        )
        malformed = replace(result, excluded=(overlap,))

        # When: the retrieval partition crosses the canonical boundary
        matches = retrieval_matches_canonical_memory(episodes, malformed, TopK(2))

        # Then: eligible and excluded collections are disjoint
        assert not matches

    def test_should_reject_missing_canonical_episode(self) -> None:
        # Given: one non-selected canonical ranked entry is omitted
        episodes = _eligible_episodes()
        result = _retrieve(episodes)
        malformed = replace(
            result,
            ranked_candidates=result.ranked_candidates[:-1],
        )

        # When: the retrieval partition crosses the canonical boundary
        matches = retrieval_matches_canonical_memory(episodes, malformed, TopK(2))

        # Then: the audit must cover every canonical episode exactly once
        assert not matches

    def test_should_reject_extra_noncanonical_episode(self) -> None:
        # Given: ranking includes a fully typed episode absent from the source load
        episodes = _eligible_episodes()
        result = _retrieve(episodes)
        extra = replace(
            result.ranked_candidates[0],
            episode=make_episode(episode_id="extra"),
        )
        malformed = replace(
            result,
            ranked_candidates=(*result.ranked_candidates, extra),
        )

        # When: the retrieval partition crosses the canonical boundary
        matches = retrieval_matches_canonical_memory(episodes, malformed, TopK(2))

        # Then: the audit cannot introduce noncanonical memory
        assert not matches

    def test_should_reject_selected_nonprefix_ranked_entry(self) -> None:
        # Given: selection skips the leading ranked candidate
        episodes = _eligible_episodes()
        result = _retrieve(episodes)
        malformed = replace(result, selected=(result.ranked_candidates[1],))

        # When: the retrieval partition crosses the canonical boundary
        matches = retrieval_matches_canonical_memory(episodes, malformed, TopK(2))

        # Then: selected values must be the exact leading ranked prefix
        assert not matches

    def test_should_reject_selected_prefix_larger_than_requested_top_k(self) -> None:
        # Given: retrieval selects a valid three-entry prefix for a top-two request
        episodes = _eligible_episodes()
        result = _retrieve(episodes, top_k=3)

        # When: the result crosses the boundary configured for two selections
        matches = retrieval_matches_canonical_memory(episodes, result, TopK(2))

        # Then: a structurally valid prefix still respects the requested bound
        assert not matches
