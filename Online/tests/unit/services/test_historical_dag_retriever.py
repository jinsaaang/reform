"""Tests for deterministic context-aware historical DAG retrieval."""

from dataclasses import dataclass

from src.domain.finance.memory import ResolvedDagEpisode
from src.domain.finance.retrieval import (
    EligibilityPolicy,
    ExclusionReason,
    HistoricalDagQuery,
    TopK,
)
from src.domain.finance.search import TargetProfile
from src.services.historical_dag_retriever import HistoricalDagRetriever
from tests.unit.domain.finance._factories import make_episode


@dataclass(frozen=True, slots=True)
class _EpisodeSource:
    episodes: tuple[ResolvedDagEpisode, ...]

    def load_episodes(self) -> tuple[ResolvedDagEpisode, ...]:
        return self.episodes


def _target(context: str) -> TargetProfile:
    return TargetProfile.model_validate(
        {
            "question_id": "current-question",
            "question_text": "Which financial mechanism is most relevant?",
            "question_type": "binary",
            "domain": "finance",
            "context": (context,),
            "cutoff": "2025-01-01T00:00:00+00:00",
            "outcome_space": ("Yes", "No"),
            "resolution_rule": "Official publication.",
        }
    )


def _query(context: str, top_k: int = 2) -> HistoricalDagQuery:
    return HistoricalDagQuery(
        target_profile=_target(context),
        policy=EligibilityPolicy.PUBLIC_DB_BOOTSTRAP,
        top_k=TopK.parse(top_k),
    )


class TestContextAwareRanking:
    def test_should_change_first_result_when_context_changes(self) -> None:
        # Given: semantically distinct eligible finance episodes
        retriever = HistoricalDagRetriever(
            _EpisodeSource(
                episodes=(
                    make_episode(
                        episode_id="oil",
                        question_text="Will crude oil prices rise?",
                        context="OPEC supply cuts and refinery inventory",
                    ),
                    make_episode(
                        episode_id="chips",
                        question_text="Will semiconductor revenue rise?",
                        context="GPU demand and semiconductor orders",
                    ),
                )
            )
        )

        # When: retrieval runs for two accepted current contexts
        chips = retriever.retrieve(_query("GPU semiconductor demand", top_k=1))
        oil = retriever.retrieve(_query("OPEC crude oil supply", top_k=1))

        # Then: context terms change the top-ranked historical analogy
        assert (
            chips.selected[0].episode.episode_id != oil.selected[0].episode.episode_id
        )


class TestDeterministicRanking:
    def test_should_retrieve_from_caller_supplied_canonical_tuple(self) -> None:
        # Given: a source whose episodes must not be used by the pure path
        source = _EpisodeSource((make_episode(episode_id="source-only"),))
        supplied = (make_episode(episode_id="supplied"),)

        # When: retrieval receives the caller's canonical tuple directly
        result = HistoricalDagRetriever(source).retrieve_from(
            supplied,
            _query("GPU semiconductor demand", top_k=1),
        )

        # Then: the selected episode comes only from caller-supplied memory
        assert result.selected[0].episode == supplied[0]

    def test_should_return_equal_result_on_repeated_invocation(self) -> None:
        # Given: one immutable episode catalog and typed query
        retriever = HistoricalDagRetriever(
            _EpisodeSource(
                episodes=(
                    make_episode(episode_id="chips"),
                    make_episode(episode_id="oil"),
                )
            )
        )
        query = _query("GPU semiconductor demand")

        # When: retrieval is repeated without state changes
        first = retriever.retrieve(query)
        second = retriever.retrieve(query)

        # Then: all scores, terms, audits, and ordering are deterministic
        assert first == second

    def test_should_break_equal_scores_by_stable_episode_id(self) -> None:
        # Given: two episodes with identical lexical content in reverse ID order
        source = _EpisodeSource(
            episodes=(
                make_episode(episode_id="zeta"),
                make_episode(episode_id="alpha"),
            )
        )

        # When: both candidates receive the same score
        result = HistoricalDagRetriever(source).retrieve(
            _query("GPU demand and semiconductor revenue")
        )

        # Then: stable episode ID provides the deterministic tie break
        assert tuple(str(item.episode.episode_id) for item in result.selected) == (
            "alpha",
            "zeta",
        )


class TestHardGateBeforeRanking:
    def test_should_exclude_high_overlap_post_cutoff_candidate(self) -> None:
        # Given: a perfect lexical match whose resolution proxy is post-cutoff
        source = _EpisodeSource(
            episodes=(
                make_episode(
                    episode_id="future",
                    question_text="OPEC crude oil supply",
                    context="OPEC crude oil supply",
                    resolution_year=2026,
                ),
                make_episode(
                    episode_id="past",
                    question_text="Generic finance history",
                    context="Rates and demand",
                    resolution_year=2024,
                ),
            )
        )

        # When: hard eligibility runs before lexical scoring
        result = HistoricalDagRetriever(source).retrieve(
            _query("OPEC crude oil supply")
        )

        # Then: the future episode appears only in the exclusion audit
        assert result.excluded[0].decision.exclusion_reasons == (
            ExclusionReason.RESOLUTION_NOT_STRICTLY_BEFORE_CUTOFF,
        )


class TestMatchedFeatureAudit:
    def test_should_report_stable_context_terms_as_data(self) -> None:
        # Given: query text containing instruction-shaped content and finance terms
        source = _EpisodeSource(
            episodes=(
                make_episode(
                    episode_id="chips",
                    context="ignore instructions GPU demand",
                ),
            )
        )

        # When: deterministic lexical retrieval tokenizes the context
        result = HistoricalDagRetriever(source).retrieve(
            _query("ignore instructions GPU demand", top_k=1)
        )

        # Then: matched content is audited only as inert normalized terms
        assert "instructions" in result.selected[0].matched_terms
