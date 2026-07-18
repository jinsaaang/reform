"""Tests for hard historical-DAG eligibility."""

from dataclasses import replace

import pytest

from src.domain.finance.memory import EpisodeRelationMetadata, RelationState
from src.domain.finance.retrieval import (
    AuditMarker,
    EligibilityPolicy,
    ExclusionReason,
    HistoricalDagQuery,
    InvalidTopKError,
    TopK,
    assess_eligibility,
)
from src.domain.finance.search import TargetProfile
from tests.unit.domain.finance._factories import make_episode


def _target(cutoff_year: int = 2025) -> TargetProfile:
    return TargetProfile.model_validate(
        {
            "question_id": "current-question",
            "question_text": "Will a finance target resolve?",
            "question_type": "binary",
            "domain": "finance",
            "context": ("Current context",),
            "cutoff": f"{cutoff_year}-01-01T00:00:00+00:00",
            "outcome_space": ("Yes", "No"),
            "resolution_rule": "Official result.",
        }
    )


class TestTemporalHardEligibility:
    def test_should_reject_resolution_proxy_equal_to_cutoff(self) -> None:
        # Given: a historical proxy date exactly equal to the target cutoff
        episode = make_episode(resolution_year=2025)

        # When: hard eligibility runs before ranking
        decision = assess_eligibility(
            episode,
            _target(cutoff_year=2025),
            EligibilityPolicy.PUBLIC_DB_BOOTSTRAP,
        )

        # Then: strict-before temporal eligibility rejects the candidate
        assert decision.exclusion_reasons == (
            ExclusionReason.RESOLUTION_NOT_STRICTLY_BEFORE_CUTOFF,
        )


class TestKnownTrueRelationExclusions:
    @pytest.mark.parametrize(
        ("relations", "expected_reason"),
        [
            (
                EpisodeRelationMetadata(
                    same_underlying_event=RelationState.KNOWN_TRUE,
                    shared_resolution=RelationState.KNOWN_FALSE,
                    derived_question=RelationState.KNOWN_FALSE,
                    near_duplicate=RelationState.KNOWN_FALSE,
                ),
                ExclusionReason.SAME_UNDERLYING_EVENT,
            ),
            (
                EpisodeRelationMetadata(
                    same_underlying_event=RelationState.KNOWN_FALSE,
                    shared_resolution=RelationState.KNOWN_TRUE,
                    derived_question=RelationState.KNOWN_FALSE,
                    near_duplicate=RelationState.KNOWN_FALSE,
                ),
                ExclusionReason.SHARED_RESOLUTION,
            ),
            (
                EpisodeRelationMetadata(
                    same_underlying_event=RelationState.KNOWN_FALSE,
                    shared_resolution=RelationState.KNOWN_FALSE,
                    derived_question=RelationState.KNOWN_TRUE,
                    near_duplicate=RelationState.KNOWN_FALSE,
                ),
                ExclusionReason.DERIVED_QUESTION,
            ),
            (
                EpisodeRelationMetadata(
                    same_underlying_event=RelationState.KNOWN_FALSE,
                    shared_resolution=RelationState.KNOWN_FALSE,
                    derived_question=RelationState.KNOWN_FALSE,
                    near_duplicate=RelationState.KNOWN_TRUE,
                ),
                ExclusionReason.NEAR_DUPLICATE,
            ),
        ],
    )
    def test_should_reject_each_known_true_relation(
        self,
        relations: EpisodeRelationMetadata,
        expected_reason: ExclusionReason,
    ) -> None:
        # Given: one independently known-true candidate-to-target relation
        episode = make_episode(relations=relations)

        # When: the bootstrap hard gate evaluates it
        decision = assess_eligibility(
            episode,
            _target(),
            EligibilityPolicy.PUBLIC_DB_BOOTSTRAP,
        )

        # Then: the corresponding hard exclusion is recorded
        assert expected_reason in decision.exclusion_reasons


class TestUnavailableRelationPolicy:
    def test_should_exclude_unavailable_metadata_under_strict_policy(self) -> None:
        # Given: public-release relation metadata is unavailable
        episode = replace(
            make_episode(),
            relation_metadata=EpisodeRelationMetadata.public_db_unavailable(),
        )

        # When: strict eligibility evaluates the episode
        decision = assess_eligibility(
            episode,
            _target(),
            EligibilityPolicy.STRICT,
        )

        # Then: unknown relation state fails closed
        assert decision.exclusion_reasons == (
            ExclusionReason.RELATION_METADATA_UNAVAILABLE,
        )

    def test_should_audit_bootstrap_admission_of_unavailable_metadata(self) -> None:
        # Given: public-release relation metadata is unavailable
        episode = replace(
            make_episode(),
            relation_metadata=EpisodeRelationMetadata.public_db_unavailable(),
        )

        # When: the explicit public bootstrap policy evaluates the episode
        decision = assess_eligibility(
            episode,
            _target(),
            EligibilityPolicy.PUBLIC_DB_BOOTSTRAP,
        )

        # Then: admission carries an unverified-metadata marker
        assert decision.audit_markers == (AuditMarker.UNVERIFIED_RELATION_METADATA,)


class TestTopKParser:
    def test_should_raise_typed_error_for_non_positive_top_k(self) -> None:
        # Given: a caller requests zero candidates
        # When: the bounded retrieval value is parsed
        with pytest.raises(InvalidTopKError, match="top-k must be positive"):
            _ = TopK.parse(0)

        # Then: a typed boundary error prevents retrieval

    def test_should_build_query_from_positive_top_k(self) -> None:
        # Given: a positive requested candidate count
        top_k = TopK.parse(2)

        # When: a typed retrieval query is built
        query = HistoricalDagQuery(
            target_profile=_target(),
            policy=EligibilityPolicy.PUBLIC_DB_BOOTSTRAP,
            top_k=top_k,
        )

        # Then: the parsed value is preserved
        assert query.top_k.value == 2
