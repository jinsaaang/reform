"""Atomic-bundle cases re-exported by the Todo 5 acceptance module."""

import json
import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.domain.finance.experiment_artifact import (
    PersistedArmSucceeded,
    PersistedFinanceExperimentSuite,
)
from src.domain.finance.sanitized_artifact import (
    ArtifactSanitizationError,
    TransientForbiddenValueRegistry,
)
from src.services.finance_artifact_sanitizer import validate_persisted_artifact
from src.services.finance_experiment_artifact import (
    FinanceExperimentArtifactError,
    FinanceExperimentArtifactFailure,
    ensure_bundle_destination_available,
    load_finance_experiment_bundle,
    persist_finance_experiment_bundle,
)
from tests.fixtures.finance_experiment_artifact import make_persisted_suite


def test_bundle_writer_refuses_an_existing_destination(tmp_path: Path) -> None:
    destination = tmp_path / "existing"
    destination.mkdir()

    with pytest.raises(FinanceExperimentArtifactError) as caught:
        ensure_bundle_destination_available(destination)

    assert caught.value.reason.value == "destination_exists"


def test_atomic_bundle_round_trip_preserves_reasoning_hashes_and_report(
    tmp_path: Path,
) -> None:
    suite = make_persisted_suite()
    destination = tmp_path / "bundle"

    receipt = persist_finance_experiment_bundle(destination, suite)
    verified = load_finance_experiment_bundle(destination)

    assert verified.suite == suite
    assert verified.receipt == receipt
    assert {path.name for path in destination.iterdir()} == {
        "suite.json",
        "report.md",
        "SHA256SUMS",
    }
    direct = verified.suite.trials[0].arms[0]
    assert isinstance(direct, PersistedArmSucceeded)
    assert len(direct.reasoning.candidate.scenarios) == 1
    assert direct.exchanges[0].input_sha256 == "1" * 64
    checksums = (destination / "SHA256SUMS").read_text(encoding="utf-8")
    assert "suite.json" in checksums
    assert "report.md" in checksums
    assert "SHA256SUMS" not in checksums


def test_report_is_byte_stable_across_two_new_destinations(tmp_path: Path) -> None:
    suite = make_persisted_suite()
    first = tmp_path / "first"
    second = tmp_path / "second"

    persist_finance_experiment_bundle(first, suite)
    persist_finance_experiment_bundle(second, suite)

    assert (first / "report.md").read_bytes() == (second / "report.md").read_bytes()


def test_loader_rejects_one_byte_tamper(tmp_path: Path) -> None:
    destination = tmp_path / "bundle"
    persist_finance_experiment_bundle(destination, make_persisted_suite())
    suite_path = destination / "suite.json"
    payload = suite_path.read_bytes()
    suite_path.write_bytes(payload.replace(b"Base case", b"Bose case", 1))

    with pytest.raises(FinanceExperimentArtifactError) as caught:
        load_finance_experiment_bundle(destination)

    assert caught.value.reason is FinanceExperimentArtifactFailure.HASH_MISMATCH


def test_unwritable_destination_fails_without_partial_residue(tmp_path: Path) -> None:
    parent_file = tmp_path / "not-a-directory"
    parent_file.write_text("occupied", encoding="utf-8")
    destination = parent_file / "bundle"

    with pytest.raises(FinanceExperimentArtifactError) as caught:
        persist_finance_experiment_bundle(destination, make_persisted_suite())

    assert caught.value.reason is FinanceExperimentArtifactFailure.PARENT_UNAVAILABLE
    assert not destination.exists()
    assert tuple(tmp_path.glob(".bundle.tmp-*")) == ()


@pytest.mark.parametrize("interrupted_write", (1, 2))
def test_interrupted_bundle_write_cleans_every_partial_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    interrupted_write: int,
) -> None:
    destination = tmp_path / "bundle"
    call_count = 0

    def interrupt_selected_write(path: Path, content: bytes) -> None:
        nonlocal call_count
        call_count += 1
        if call_count == interrupted_write:
            raise KeyboardInterrupt
        with path.open("xb") as handle:
            _ = handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())

    monkeypatch.setattr(
        "src.services.finance_hashed_bundle._write_fsync",
        interrupt_selected_write,
    )

    with pytest.raises(KeyboardInterrupt):
        persist_finance_experiment_bundle(destination, make_persisted_suite())

    assert not destination.exists()
    assert tuple(tmp_path.glob(".bundle.tmp-*")) == ()


def test_parent_fsync_failure_rolls_back_published_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "bundle"
    call_count = 0

    def fail_parent_fsync(directory: Path) -> None:
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise OSError("forced parent fsync failure")

    monkeypatch.setattr(
        "src.services.finance_hashed_bundle._fsync_directory",
        fail_parent_fsync,
    )

    with pytest.raises(FinanceExperimentArtifactError) as caught:
        persist_finance_experiment_bundle(destination, make_persisted_suite())

    assert call_count == 2
    assert caught.value.reason is FinanceExperimentArtifactFailure.WRITE_FAILED
    assert not destination.exists()
    assert tuple(tmp_path.glob(".bundle.tmp-*")) == ()


@pytest.mark.parametrize(
    ("key", "value"),
    (
        ("apiKey", "redacted"),
        ("requestBody", "redacted"),
        ("rawResponse", "redacted"),
        ("chainOfThought", "redacted"),
        ("safe", "ORIGINAL_EVIDENCE_SENTINEL_7f9c"),
        ("safe", "ORIGINAL_DAG_SENTINEL_4a2e"),
        ("safe", "RAW_BODY_SENTINEL_91bd"),
        ("safe", "sk-" + "or-v1-" + "A" * 16),
    ),
)
def test_persisted_fixture_recursively_rejects_forbidden_keys_and_values(
    key: str,
    value: str,
) -> None:
    payload = json.dumps({"safe": {key: value}}, separators=(",", ":"))

    with pytest.raises(ArtifactSanitizationError):
        validate_persisted_artifact(payload)

    serialized = make_persisted_suite().model_dump_json()
    assert all(
        forbidden not in serialized
        for forbidden in (
            "ORIGINAL_EVIDENCE_SENTINEL_7f9c",
            "ORIGINAL_DAG_SENTINEL_4a2e",
            "RAW_BODY_SENTINEL_91bd",
            "sk-" + "or-v1-" + "Q" * 16,
        )
    )


def test_persistence_registry_rejects_exact_transient_body_and_cleans_temp(
    tmp_path: Path,
) -> None:
    suite = make_persisted_suite()
    direct = suite.trials[0].arms[0]
    assert isinstance(direct, PersistedArmSucceeded)
    exact_body = direct.reasoning.candidate.explanation
    registry = TransientForbiddenValueRegistry(raw_values=(exact_body,))
    destination = tmp_path / "bundle"

    with pytest.raises(ArtifactSanitizationError):
        persist_finance_experiment_bundle(destination, suite, registry=registry)

    assert not destination.exists()
    assert tuple(tmp_path.glob(".bundle.tmp-*")) == ()


def test_strict_suite_schema_rejects_camel_case_secret_field() -> None:
    payload = (
        make_persisted_suite()
        .model_dump_json()
        .replace(
            '"suite_id":',
            '"requestBody":"forbidden","suite_id":',
            1,
        )
    )

    with pytest.raises(ValidationError):
        PersistedFinanceExperimentSuite.model_validate_json(payload)

    assert "requestBody" in payload


__all__ = [
    "test_atomic_bundle_round_trip_preserves_reasoning_hashes_and_report",
    "test_bundle_writer_refuses_an_existing_destination",
    "test_interrupted_bundle_write_cleans_every_partial_directory",
    "test_loader_rejects_one_byte_tamper",
    "test_parent_fsync_failure_rolls_back_published_directory",
    "test_persisted_fixture_recursively_rejects_forbidden_keys_and_values",
    "test_persistence_registry_rejects_exact_transient_body_and_cleans_temp",
    "test_report_is_byte_stable_across_two_new_destinations",
    "test_strict_suite_schema_rejects_camel_case_secret_field",
    "test_unwritable_destination_fails_without_partial_residue",
]
