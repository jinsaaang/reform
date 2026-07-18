"""Identity-only preflight tests for the immutable finance seed."""

import hashlib
from dataclasses import replace
from pathlib import Path

import pytest

from src.domain.finance.retrieval import (
    FinanceSeedAssetSpec,
    SeedAssetMismatchError,
    WritableSeedAssetError,
)
from src.services.finance_seed_identity import (
    FinanceSeedIdentityVerifier,
    FinanceSeedSidecarError,
)

_DB_PATH = Path("data/releases/worldreasoner/v1.0.0/worldreasoner_public.db")


def _fixture_spec(path: Path) -> FinanceSeedAssetSpec:
    return FinanceSeedAssetSpec(
        byte_size=path.stat().st_size,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        schema_version=8,
        episode_count=1,
    )


def test_real_seed_identity_preflight_has_no_sqlite_side_effect() -> None:
    # Given
    before = hashlib.sha256(_DB_PATH.read_bytes()).hexdigest()

    # When
    FinanceSeedIdentityVerifier(_DB_PATH).verify_identity()

    # Then
    assert hashlib.sha256(_DB_PATH.read_bytes()).hexdigest() == before
    assert not tuple(_DB_PATH.parent.glob(f"{_DB_PATH.name}-*"))


def test_identity_preflight_rejects_digest_mismatch() -> None:
    # Given
    verifier = FinanceSeedIdentityVerifier(
        _DB_PATH,
        replace(_fixture_spec(_DB_PATH), sha256="0" * 64),
    )

    # When / Then
    with pytest.raises(SeedAssetMismatchError, match="sha256"):
        verifier.verify_identity()


def test_identity_preflight_rejects_writable_file(tmp_path: Path) -> None:
    # Given
    db_path = tmp_path / "writable.db"
    db_path.write_bytes(b"fixture")
    verifier = FinanceSeedIdentityVerifier(db_path, _fixture_spec(db_path))

    # When / Then
    with pytest.raises(WritableSeedAssetError):
        verifier.verify_identity()


def test_identity_preflight_rejects_sqlite_sidecar(tmp_path: Path) -> None:
    # Given
    db_path = tmp_path / "immutable.db"
    db_path.write_bytes(b"fixture")
    spec = _fixture_spec(db_path)
    db_path.chmod(0o444)
    (tmp_path / "immutable.db-wal").write_bytes(b"stale")

    # When / Then
    with pytest.raises(FinanceSeedSidecarError):
        FinanceSeedIdentityVerifier(db_path, spec).verify_identity()
