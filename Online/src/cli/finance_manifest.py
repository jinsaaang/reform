"""Typed audit boundary for the pinned WorldReasoner finance seed manifest."""

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing_extensions import override

from pydantic import ValidationError

from src.cli.finance_manifest_models import (
    FinanceSeedManifest,
    SeedAssetAudit,
    SeedAuditSummary,
    SeedCanonicalAudit,
    SeedTypeCounts,
)
from src.cli.finance_manifest_spec import (
    CANONICAL_SQL,
    EXPECTED_MISSING_FIELDS,
    EXPECTED_ROW_COUNTS,
    EXPECTED_TABLES,
    LIMITATIONS,
    MANIFEST_SHA256,
    MANIFEST_SIZE_BYTES,
    REQUIRED_COLUMNS,
    SOURCE_URL,
)
from src.core.finance_seed_repository import (
    DuplicateEpisodeIdError,
    FinanceSeedRepository,
    SeedAssetMismatchError,
    SeedDataError,
    SeedSchemaError,
    UnsafeSeedPathError,
    WritableSeedAssetError,
)
from src.domain.finance.memory import QuestionKind
from src.domain.finance.retrieval import FINANCE_SEED_V1_SPEC, SeedDatabaseReadError


@dataclass(frozen=True, slots=True)
class FinanceManifestError(Exception):
    """Actionable fail-closed manifest or asset mismatch."""

    path: Path
    detail: str

    @override
    def __str__(self) -> str:
        return f"finance seed audit failed for {self.path}: {self.detail}"


def _parse_manifest(path: Path) -> tuple[FinanceSeedManifest, str, int]:
    if not path.is_file():
        raise FinanceManifestError(path, "manifest path does not name a file")
    try:
        data = path.read_bytes()
    except OSError as error:
        raise FinanceManifestError(path, f"malformed manifest: {error}") from error
    try:
        manifest = FinanceSeedManifest.model_validate_json(data)
    except ValidationError as error:
        raise FinanceManifestError(path, f"malformed manifest: {error}") from error
    source = manifest.source
    if (
        source.url != SOURCE_URL
        or source.sha256 != FINANCE_SEED_V1_SPEC.sha256
        or source.size_bytes != FINANCE_SEED_V1_SPEC.byte_size
    ):
        raise FinanceManifestError(path, "source asset identity does not match v1.0.0")
    size = len(data)
    digest = hashlib.sha256(data).hexdigest()
    if size != MANIFEST_SIZE_BYTES:
        raise FinanceManifestError(
            path,
            f"manifest size mismatch: expected {MANIFEST_SIZE_BYTES}, got {size}",
        )
    if digest != MANIFEST_SHA256:
        raise FinanceManifestError(
            path,
            f"manifest sha256 mismatch: expected {MANIFEST_SHA256}, got {digest}",
        )
    return manifest, digest, size


def _validate_manifest(manifest: FinanceSeedManifest, path: Path) -> None:
    source = manifest.source
    if (
        source.url != SOURCE_URL
        or source.sha256 != FINANCE_SEED_V1_SPEC.sha256
        or source.size_bytes != FINANCE_SEED_V1_SPEC.byte_size
    ):
        raise FinanceManifestError(path, "source asset identity does not match v1.0.0")
    selection = manifest.canonical_selection
    expected_types = {
        QuestionKind.BINARY.value: 23,
        QuestionKind.MCQ.value: 1,
        QuestionKind.QUANTITY.value: 8,
        QuestionKind.TIMEFRAME.value: 5,
    }
    if selection.canonical_sql != CANONICAL_SQL:
        raise FinanceManifestError(path, "canonical selection SQL mismatch")
    if (
        selection.total_rows != 37
        or selection.unique_ids != 37
        or selection.type_counts != expected_types
        or selection.missing_field_counts != EXPECTED_MISSING_FIELDS
    ):
        raise FinanceManifestError(path, "canonical selection counts mismatch")
    if (
        manifest.limitations != LIMITATIONS
        or manifest.absent_schema_features
        or manifest.graph_linkage.canonical_questions != 37
    ):
        raise FinanceManifestError(path, "public-DB limitation declaration mismatch")
    schema_tables = {table.name: table for table in manifest.schema_inventory.tables}
    if (
        manifest.schema_inventory.schema_version != FINANCE_SEED_V1_SPEC.schema_version
        or frozenset(schema_tables) != EXPECTED_TABLES
        or len(schema_tables) != len(manifest.schema_inventory.tables)
    ):
        raise FinanceManifestError(path, "schema inventory mismatch")
    if {
        table.name: table.row_count for table in manifest.schema_inventory.tables
    } != EXPECTED_ROW_COUNTS:
        raise FinanceManifestError(path, "schema row counts mismatch")
    for table_name, required in REQUIRED_COLUMNS.items():
        actual = {column.name for column in schema_tables[table_name].columns}
        if not required.issubset(actual):
            raise FinanceManifestError(
                path, f"schema inventory missing {table_name} columns"
            )


def audit_seed(db_path: Path, manifest_path: Path) -> SeedAuditSummary:
    """Audit the immutable DB through FinanceSeedRepository and the manifest."""
    manifest, manifest_digest, manifest_size = _parse_manifest(manifest_path)
    _validate_manifest(manifest, manifest_path)
    sidecars = tuple(
        sorted(path.name for path in db_path.parent.glob(f"{db_path.name}-*"))
    )
    if sidecars:
        raise FinanceManifestError(db_path, f"SQLite sidecars present: {sidecars}")
    try:
        episodes = FinanceSeedRepository(db_path).load_episodes()
    except (
        DuplicateEpisodeIdError,
        SeedAssetMismatchError,
        SeedDataError,
        SeedDatabaseReadError,
        SeedSchemaError,
        UnsafeSeedPathError,
        WritableSeedAssetError,
    ) as error:
        raise FinanceManifestError(db_path, str(error)) from error
    with db_path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    ordered_ids = tuple(str(episode.question_id) for episode in episodes)
    counts = {
        kind.value: sum(
            episode.source_profile.question_type is kind for episode in episodes
        )
        for kind in QuestionKind
    }
    expected = manifest.canonical_selection
    if (
        digest != manifest.source.sha256
        or db_path.stat().st_size != manifest.source.size_bytes
        or ordered_ids != expected.ordered_ids
        or counts != expected.type_counts
    ):
        raise FinanceManifestError(
            db_path,
            "database rows, digest, or type counts mismatch manifest",
        )
    return SeedAuditSummary(
        manifest_sha256=manifest_digest,
        manifest_size_bytes=manifest_size,
        source=manifest.source,
        asset=SeedAssetAudit(
            size_bytes=db_path.stat().st_size,
            sha256=digest,
            schema_version=manifest.schema_inventory.schema_version,
            read_only=True,
            sqlite_sidecars=sidecars,
        ),
        canonical_selection=SeedCanonicalAudit(
            canonical_sql=expected.canonical_sql,
            total_rows=len(episodes),
            unique_ids=len(set(ordered_ids)),
            ordered_ids=ordered_ids,
            type_counts=SeedTypeCounts(**counts),
        ),
        limitations=manifest.limitations,
    )
