"""Tests for the immutable public-finance SQLite read adapter."""

import hashlib
import sqlite3
from pathlib import Path

import pytest

from src.core.finance_seed_repository import (
    DuplicateEpisodeIdError,
    FinanceSeedAssetSpec,
    FinanceSeedRepository,
    SeedAssetMismatchError,
    SeedDataError,
    SeedSchemaError,
    UnsafeSeedPathError,
    WritableSeedAssetError,
)
from src.domain.finance.memory import ResolvedDagEpisode

_REAL_DB = Path("data/releases/worldreasoner/v1.0.0/worldreasoner_public.db")


def _asset_spec(path: Path, episode_count: int) -> FinanceSeedAssetSpec:
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return FinanceSeedAssetSpec(
        byte_size=path.stat().st_size,
        sha256=digest,
        schema_version=8,
        episode_count=episode_count,
    )


def _write_fixture(
    path: Path,
    question_ids: tuple[str, ...] = ("question-1",),
    event_tags: str = "[]",
) -> None:
    with sqlite3.connect(path) as connection:
        _ = connection.executescript(
            """
            PRAGMA user_version=8;
            CREATE TABLE questions (
                id TEXT, question_text TEXT, question_type TEXT, domain TEXT,
                source TEXT, context TEXT, resolution_date TEXT, ground_truth TEXT,
                resolution_criteria TEXT, options TEXT, quantity_unit TEXT,
                outcome_event_ids TEXT, graph_built INTEGER
            );
            CREATE TABLE events (
                id TEXT, title TEXT, description TEXT, event_type TEXT,
                domain TEXT, tags TEXT, occurred_date TEXT, predicted_date TEXT,
                article_ids TEXT, extracted_for_question_id TEXT,
                is_outcome INTEGER, outcome_scenario TEXT,
                is_actual_outcome INTEGER
            );
            CREATE TABLE causal_hypotheses (
                id TEXT, source_event_id TEXT, target_event_id TEXT,
                relation_type TEXT, strength REAL, confidence REAL,
                time_lag_hours REAL, reasoning TEXT,
                evidence_article_ids TEXT, discovered_by_question_ids TEXT
            );
            CREATE TABLE event_outcome_impacts (
                id TEXT, event_id TEXT, outcome_event_id TEXT,
                question_id TEXT, impact_direction TEXT,
                impact_magnitude REAL, confidence REAL, reasoning TEXT,
                evidence_article_ids TEXT, causal_chain_hypothesis_ids TEXT
            );
            """
        )
        for question_id in question_ids:
            _ = connection.execute(
                "INSERT INTO questions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    question_id,
                    "Will the fixture resolve yes?",
                    "binary",
                    "finance",
                    "fixture",
                    "Fixture context",
                    "2024-01-01T00:00:00+00:00",
                    '"Yes"',
                    "Official fixture result.",
                    '["Yes", "No"]',
                    None,
                    '["outcome-1"]',
                    1,
                ),
            )
        _ = connection.execute(
            "INSERT INTO events VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "outcome-1",
                "Fixture outcome",
                "The fixture resolved.",
                "outcome",
                "finance",
                event_tags,
                "2024-01-01T00:00:00+00:00",
                None,
                "[]",
                "question-1",
                1,
                "Yes",
                1,
            ),
        )
        _ = connection.execute("PRAGMA schema_version=8")
    path.chmod(0o444)


class TestRealFinanceSeed:
    def test_should_load_all_37_typed_episodes(self) -> None:
        # Given: the pinned immutable public database
        repository = FinanceSeedRepository(_REAL_DB)

        # When: the canonical finance memory is reconstructed
        episodes = repository.load_episodes()

        # Then: all canonical rows cross into typed episodes
        assert len(episodes) == 37 and all(
            isinstance(episode, ResolvedDagEpisode) for episode in episodes
        )

    def test_should_reconstruct_raw_graph_reference_counts(self) -> None:
        # Given: the pinned immutable public database
        repository = FinanceSeedRepository(_REAL_DB)

        # When: all historical graph references are reconstructed
        episodes = repository.load_episodes()
        counts = (
            sum(len(episode.nodes) for episode in episodes),
            sum(len(episode.edges) for episode in episodes),
            sum(len(episode.impacts) for episode in episodes),
        )

        # Then: the adapter preserves the independently characterized graph rows
        assert counts == (950, 884, 888)

    def test_should_preserve_hash_and_emit_no_sqlite_sidecars(self) -> None:
        # Given: the pinned immutable public database and its hash
        before = hashlib.sha256(_REAL_DB.read_bytes()).hexdigest()

        # When: repository reconstruction completes
        _ = FinanceSeedRepository(_REAL_DB).load_episodes()
        after = hashlib.sha256(_REAL_DB.read_bytes()).hexdigest()
        sidecars = tuple(
            path.name
            for path in _REAL_DB.parent.iterdir()
            if path.name.startswith(f"{_REAL_DB.name}-")
        )

        # Then: the official asset has no mutation or SQLite side effects
        assert (after, sidecars) == (before, ())


class TestSeedSchemaValidation:
    def test_should_raise_typed_error_for_missing_schema(self, tmp_path: Path) -> None:
        # Given: a read-only SQLite file without the pinned tables
        path = tmp_path / "missing-schema.db"
        with sqlite3.connect(path):
            pass
        path.chmod(0o444)

        # When: repository schema validation runs
        with pytest.raises(SeedSchemaError, match="schema"):
            _ = FinanceSeedRepository(path, _asset_spec(path, 0)).load_episodes()

        # Then: a typed schema diagnostic is exposed


class TestSeedRowValidation:
    def test_should_raise_typed_error_for_blank_question_id(
        self,
        tmp_path: Path,
    ) -> None:
        # Given: a schema-compatible fixture with an invalid blank stable ID
        path = tmp_path / "blank-id.db"
        _write_fixture(path, question_ids=("",))

        # When: boundary row parsing runs
        with pytest.raises(SeedDataError, match="questions"):
            _ = FinanceSeedRepository(path, _asset_spec(path, 1)).load_episodes()

        # Then: malformed IDs never become branded domain identities

    def test_should_raise_typed_error_for_malformed_json(self, tmp_path: Path) -> None:
        # Given: a structurally valid fixture containing malformed event tags
        path = tmp_path / "malformed-json.db"
        _write_fixture(path, event_tags="not-json")

        # When: boundary row parsing runs
        with pytest.raises(SeedDataError, match="events"):
            _ = FinanceSeedRepository(path, _asset_spec(path, 1)).load_episodes()

        # Then: malformed DB JSON never enters the domain

    def test_should_raise_typed_error_for_duplicate_ids(self, tmp_path: Path) -> None:
        # Given: a schema-compatible fixture with duplicate canonical IDs
        path = tmp_path / "duplicate.db"
        _write_fixture(path, question_ids=("question-1", "question-1"))

        # When: canonical rows are reconstructed
        with pytest.raises(DuplicateEpisodeIdError, match="question-1"):
            _ = FinanceSeedRepository(path, _asset_spec(path, 2)).load_episodes()

        # Then: duplicate stable identities fail closed


class TestSeedAssetValidation:
    def test_should_raise_typed_error_for_digest_mismatch(self, tmp_path: Path) -> None:
        # Given: a valid fixture and an incorrect expected digest
        path = tmp_path / "digest.db"
        _write_fixture(path)
        spec = _asset_spec(path, 1)
        wrong_spec = FinanceSeedAssetSpec(
            byte_size=spec.byte_size,
            sha256="0" * 64,
            schema_version=spec.schema_version,
            episode_count=spec.episode_count,
        )

        # When: immutable asset verification runs
        with pytest.raises(SeedAssetMismatchError, match="sha256"):
            _ = FinanceSeedRepository(path, wrong_spec).load_episodes()

        # Then: stale or tampered state is rejected before SQLite access

    def test_should_reject_writable_asset(self, tmp_path: Path) -> None:
        # Given: a schema-compatible DB with write permission restored
        path = tmp_path / "writable.db"
        _write_fixture(path)
        path.chmod(0o644)

        # When: immutable asset verification runs
        with pytest.raises(WritableSeedAssetError, match="read-only"):
            _ = FinanceSeedRepository(path, _asset_spec(path, 1)).load_episodes()

        # Then: a mutable seed cannot cross the repository boundary

    def test_should_reject_caller_supplied_sqlite_uri(self) -> None:
        # Given: a caller attempts to select writable SQLite URI semantics
        unsafe_path = Path("file:/tmp/seed.db?mode=rw")

        # When: repository location parsing runs
        with pytest.raises(UnsafeSeedPathError, match="filesystem path"):
            _ = FinanceSeedRepository(unsafe_path)

        # Then: only the adapter can construct the immutable URI
