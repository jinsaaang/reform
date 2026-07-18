"""Tests for immutable historical finance memory."""

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from src.core.finance_seed_repository import FinanceSeedRepository
from src.domain.finance.memory import (
    EpisodeRelationMetadata,
    RelationState,
    ResolutionProvenance,
)
from tests.unit.domain.finance._factories import make_episode

_REAL_DB = Path("data/releases/worldreasoner/v1.0.0/worldreasoner_public.db")


class TestHistoricalResolutionNamespace:
    def test_should_label_resolution_date_as_bootstrap_proxy(self) -> None:
        # Given: the pinned immutable public seed repository
        repository = FinanceSeedRepository(_REAL_DB)

        # When: all real historical episodes are reconstructed
        episodes = repository.load_episodes()
        provenances = {episode.historical_resolution.provenance for episode in episodes}

        # Then: every date is labeled only as the bootstrap proxy
        assert provenances == {
            ResolutionProvenance.BOOTSTRAP_RESOLUTION_DATE_PROXY,
        }


class TestImmutableHistoricalEpisode:
    def test_should_reject_episode_identity_mutation(self) -> None:
        # Given: a fully constructed resolved historical episode
        episode = make_episode()

        # When: a caller attempts to replace its stable ID
        with pytest.raises(FrozenInstanceError):
            setattr(episode, "episode_id", episode.episode_id)

        # Then: the frozen dataclass raises before state can change


class TestUnavailableRelationState:
    def test_should_preserve_unavailable_as_distinct_from_false(self) -> None:
        # Given: relation metadata absent from the public release
        relations = EpisodeRelationMetadata.public_db_unavailable()

        # When: the same-event relation state is read
        state = relations.same_underlying_event

        # Then: absence remains unavailable rather than known false
        assert state is RelationState.UNAVAILABLE
