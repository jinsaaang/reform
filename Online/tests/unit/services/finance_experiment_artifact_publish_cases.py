"""Exclusive-publication cases re-exported by the Todo 5 acceptance module."""

import errno
from pathlib import Path

import pytest

from src.services.finance_atomic_publish import RenamexResult, invoke_darwin_renamex
from src.services.finance_experiment_artifact import (
    FinanceExperimentArtifactError,
    FinanceExperimentArtifactFailure,
    persist_finance_experiment_bundle,
)
from tests.fixtures.finance_experiment_artifact import make_persisted_suite


def test_racing_empty_destination_is_preserved_without_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "bundle"
    raced_inode: int | None = None

    def create_destination_before_rename(
        source: Path,
        target: Path,
    ) -> RenamexResult:
        nonlocal raced_inode
        target.mkdir()
        raced_inode = target.stat().st_ino
        return invoke_darwin_renamex(source, target)

    monkeypatch.setattr(
        "src.services.finance_atomic_publish.invoke_darwin_renamex",
        create_destination_before_rename,
    )

    with pytest.raises(FinanceExperimentArtifactError) as caught:
        persist_finance_experiment_bundle(destination, make_persisted_suite())

    assert caught.value.reason is FinanceExperimentArtifactFailure.DESTINATION_EXISTS
    assert raced_inode is not None
    assert destination.stat().st_ino == raced_inode
    assert tuple(destination.iterdir()) == ()
    assert tuple(tmp_path.glob(".bundle.tmp-*")) == ()


def test_unsupported_atomic_publication_fails_closed_without_residue(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "bundle"
    monkeypatch.setattr(
        "src.services.finance_atomic_publish._RUNTIME_PLATFORM",
        "unsupported-test-platform",
    )

    with pytest.raises(FinanceExperimentArtifactError) as caught:
        persist_finance_experiment_bundle(destination, make_persisted_suite())

    assert caught.value.reason is FinanceExperimentArtifactFailure.WRITE_FAILED
    assert caught.value.failure_class == "unsupported-test-platform"
    assert not destination.exists()
    assert tuple(tmp_path.glob(".bundle.tmp-*")) == ()


def test_atomic_publication_system_error_fails_closed_without_residue(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "bundle"

    def fail_exclusive_publish(
        source: Path,
        target: Path,
    ) -> RenamexResult:
        assert source.parent == target.parent
        return RenamexResult(status=-1, error_number=errno.EIO)

    monkeypatch.setattr(
        "src.services.finance_atomic_publish.invoke_darwin_renamex",
        fail_exclusive_publish,
    )

    with pytest.raises(FinanceExperimentArtifactError) as caught:
        persist_finance_experiment_bundle(destination, make_persisted_suite())

    assert caught.value.reason is FinanceExperimentArtifactFailure.WRITE_FAILED
    assert caught.value.failure_class == "EIO"
    assert not destination.exists()
    assert tuple(tmp_path.glob(".bundle.tmp-*")) == ()


__all__ = [
    "test_atomic_publication_system_error_fails_closed_without_residue",
    "test_racing_empty_destination_is_preserved_without_publication",
    "test_unsupported_atomic_publication_fails_closed_without_residue",
]
