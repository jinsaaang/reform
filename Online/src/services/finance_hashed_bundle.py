"""Atomic directory bundles with deterministic SHA-256 manifests."""

import os
import shutil
import tempfile
from dataclasses import dataclass
from enum import StrEnum, unique
from hashlib import sha256
from pathlib import Path

from typing_extensions import override

from src.services.finance_atomic_publish import (
    AtomicPublishError,
    AtomicPublishFailure,
    publish_directory_exclusive,
)


@unique
class HashedBundleFailure(StrEnum):
    DESTINATION_EXISTS = "destination_exists"
    PARENT_UNAVAILABLE = "parent_unavailable"
    WRITE_FAILED = "write_failed"
    READ_FAILED = "read_failed"
    INVALID_LAYOUT = "invalid_layout"
    HASH_MISMATCH = "hash_mismatch"
    CLEANUP_FAILED = "cleanup_failed"


class HashedBundleError(Exception):
    __slots__ = ("failure_class", "path", "reason")

    def __init__(
        self,
        reason: HashedBundleFailure,
        path: Path,
        failure_class: str | None = None,
    ) -> None:
        super().__init__(reason, path, failure_class)
        self.reason = reason
        self.path = path
        self.failure_class = failure_class

    @override
    def __str__(self) -> str:
        suffix = f" ({self.failure_class})" if self.failure_class else ""
        return f"hashed bundle failed: {self.reason.value} at {self.path}{suffix}"


@dataclass(frozen=True, slots=True)
class HashedBundleContent:
    data_filename: str
    data_bytes: bytes
    report_bytes: bytes


@dataclass(frozen=True, slots=True)
class HashedBundleReceipt:
    directory: Path
    data_filename: str
    data_sha256: str
    report_sha256: str


@dataclass(frozen=True, slots=True)
class VerifiedHashedBundle:
    receipt: HashedBundleReceipt
    data_bytes: bytes
    report_bytes: bytes


def ensure_new_bundle_destination(destination: Path) -> None:
    """Require an existing directory parent and a nonexistent destination."""
    if os.path.lexists(destination):
        raise HashedBundleError(
            reason=HashedBundleFailure.DESTINATION_EXISTS,
            path=destination,
        )
    if not destination.parent.is_dir():
        raise HashedBundleError(
            reason=HashedBundleFailure.PARENT_UNAVAILABLE,
            path=destination.parent,
        )


def _manifest_bytes(content: HashedBundleContent) -> bytes:
    data_digest = sha256(content.data_bytes).hexdigest()
    report_digest = sha256(content.report_bytes).hexdigest()
    return (
        f"{data_digest}  {content.data_filename}\n{report_digest}  report.md\n"
    ).encode("ascii")


def _write_fsync(path: Path, content: bytes) -> None:
    with path.open("xb") as handle:
        _ = handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _cleanup_temporary(temporary: Path) -> None:
    try:
        shutil.rmtree(temporary)
    except OSError as error:
        raise HashedBundleError(
            reason=HashedBundleFailure.CLEANUP_FAILED,
            path=temporary,
            failure_class=type(error).__name__,
        ) from error


def persist_hashed_bundle(
    destination: Path,
    content: HashedBundleContent,
) -> HashedBundleReceipt:
    """Fsync files and exclusively publish one unique sibling directory."""
    ensure_new_bundle_destination(destination)
    temporary: Path | None = None
    published = False
    committed = False
    try:
        temporary = Path(
            tempfile.mkdtemp(
                prefix=f".{destination.name}.tmp-",
                dir=destination.parent,
            )
        )
        _write_fsync(temporary / content.data_filename, content.data_bytes)
        _write_fsync(temporary / "report.md", content.report_bytes)
        _write_fsync(temporary / "SHA256SUMS", _manifest_bytes(content))
        _fsync_directory(temporary)
        ensure_new_bundle_destination(destination)
        publish_directory_exclusive(temporary, destination)
        published = True
        _fsync_directory(destination.parent)
        committed = True
        temporary = None
    except AtomicPublishError as error:
        reason = (
            HashedBundleFailure.DESTINATION_EXISTS
            if error.reason is AtomicPublishFailure.DESTINATION_EXISTS
            else HashedBundleFailure.WRITE_FAILED
        )
        raise HashedBundleError(
            reason=reason,
            path=destination,
            failure_class=error.failure_class,
        ) from error
    except HashedBundleError:
        raise
    except OSError as error:
        raise HashedBundleError(
            reason=HashedBundleFailure.WRITE_FAILED,
            path=destination,
            failure_class=type(error).__name__,
        ) from error
    finally:
        if published and not committed and destination.exists():
            _cleanup_temporary(destination)
        if temporary is not None and temporary.exists():
            _cleanup_temporary(temporary)
    return HashedBundleReceipt(
        directory=destination,
        data_filename=content.data_filename,
        data_sha256=sha256(content.data_bytes).hexdigest(),
        report_sha256=sha256(content.report_bytes).hexdigest(),
    )


def load_hashed_bundle(directory: Path, data_filename: str) -> VerifiedHashedBundle:
    """Verify exact layout and both content hashes before returning bytes."""
    expected_names = {data_filename, "report.md", "SHA256SUMS"}
    try:
        children = tuple(directory.iterdir())
    except OSError as error:
        raise HashedBundleError(
            reason=HashedBundleFailure.READ_FAILED,
            path=directory,
            failure_class=type(error).__name__,
        ) from error
    if {path.name for path in children} != expected_names or any(
        not path.is_file() or path.is_symlink() for path in children
    ):
        raise HashedBundleError(
            reason=HashedBundleFailure.INVALID_LAYOUT,
            path=directory,
        )
    try:
        data_bytes = (directory / data_filename).read_bytes()
        report_bytes = (directory / "report.md").read_bytes()
        manifest_bytes = (directory / "SHA256SUMS").read_bytes()
    except OSError as error:
        raise HashedBundleError(
            reason=HashedBundleFailure.READ_FAILED,
            path=directory,
            failure_class=type(error).__name__,
        ) from error
    content = HashedBundleContent(
        data_filename=data_filename,
        data_bytes=data_bytes,
        report_bytes=report_bytes,
    )
    if manifest_bytes != _manifest_bytes(content):
        raise HashedBundleError(
            reason=HashedBundleFailure.HASH_MISMATCH,
            path=directory,
        )
    return VerifiedHashedBundle(
        receipt=HashedBundleReceipt(
            directory=directory,
            data_filename=data_filename,
            data_sha256=sha256(data_bytes).hexdigest(),
            report_sha256=sha256(report_bytes).hexdigest(),
        ),
        data_bytes=data_bytes,
        report_bytes=report_bytes,
    )


__all__ = [
    "HashedBundleContent",
    "HashedBundleError",
    "HashedBundleFailure",
    "HashedBundleReceipt",
    "VerifiedHashedBundle",
    "ensure_new_bundle_destination",
    "load_hashed_bundle",
    "persist_hashed_bundle",
]
