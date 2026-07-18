"""Identity-only verification for an immutable finance seed asset."""

import hashlib
from dataclasses import dataclass
from pathlib import Path

from src.domain.finance.retrieval import (
    FINANCE_SEED_V1_SPEC,
    FinanceSeedAssetSpec,
    SeedAssetMismatchError,
    SeedDatabaseReadError,
    WritableSeedAssetError,
)


@dataclass(frozen=True, slots=True)
class FinanceSeedSidecarError(Exception):
    """SQLite sidecars make an otherwise immutable identity ambiguous."""

    path: Path
    sidecars: tuple[str, ...]

    def __str__(self) -> str:
        return f"immutable finance seed has SQLite sidecars: {self.sidecars}"


@dataclass(frozen=True, slots=True)
class FinanceSeedIdentityVerifier:
    """Verify path, permissions, size, and digest without opening SQLite."""

    db_path: Path
    asset_spec: FinanceSeedAssetSpec = FINANCE_SEED_V1_SPEC

    def verify_identity(self) -> None:
        """Fail closed before providers when immutable metadata has drifted."""
        if not self.db_path.is_file():
            raise SeedAssetMismatchError(
                self.db_path,
                "path",
                "file",
                "missing",
            )
        if self.db_path.stat().st_mode & 0o222:
            raise WritableSeedAssetError(self.db_path)
        sidecars = tuple(
            sorted(
                path.name for path in self.db_path.parent.glob(f"{self.db_path.name}-*")
            )
        )
        if sidecars:
            raise FinanceSeedSidecarError(self.db_path, sidecars)
        actual_size = self.db_path.stat().st_size
        if actual_size != self.asset_spec.byte_size:
            raise SeedAssetMismatchError(
                self.db_path,
                "size",
                self.asset_spec.byte_size,
                actual_size,
            )
        try:
            with self.db_path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
        except OSError as error:
            raise SeedDatabaseReadError(
                self.db_path,
                type(error).__name__,
            ) from error
        if digest != self.asset_spec.sha256:
            raise SeedAssetMismatchError(
                self.db_path,
                "sha256",
                self.asset_spec.sha256,
                digest,
            )


__all__ = ["FinanceSeedIdentityVerifier", "FinanceSeedSidecarError"]
