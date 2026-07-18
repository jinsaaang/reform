"""Fail-closed atomic directory publication at the platform boundary."""

import ctypes
import errno
import os
import sys
from dataclasses import dataclass
from enum import StrEnum, unique
from pathlib import Path
from typing import Final

from typing_extensions import override


_RENAME_EXCL: Final = 0x00000004
_RUNTIME_PLATFORM: Final = sys.platform
_RenamexPrototype = ctypes.CFUNCTYPE(
    ctypes.c_int,
    ctypes.c_char_p,
    ctypes.c_char_p,
    ctypes.c_uint,
    use_errno=True,
)


@unique
class AtomicPublishFailure(StrEnum):
    DESTINATION_EXISTS = "destination_exists"
    UNSUPPORTED = "unsupported"
    SYSTEM_ERROR = "system_error"


class AtomicPublishError(Exception):
    __slots__ = ("failure_class", "reason")

    def __init__(
        self,
        reason: AtomicPublishFailure,
        failure_class: str,
    ) -> None:
        super().__init__(reason, failure_class)
        self.reason = reason
        self.failure_class = failure_class

    @override
    def __str__(self) -> str:
        return f"atomic publish failed: {self.reason.value} ({self.failure_class})"


@dataclass(frozen=True, slots=True)
class RenamexResult:
    status: int
    error_number: int


def invoke_darwin_renamex(source: Path, destination: Path) -> RenamexResult:
    """Invoke Darwin renamex_np with RENAME_EXCL and capture errno immediately."""
    library = ctypes.CDLL(None, use_errno=True)
    renamex = _RenamexPrototype(("renamex_np", library))
    ctypes.set_errno(0)
    status = renamex(
        os.fsencode(source),
        os.fsencode(destination),
        _RENAME_EXCL,
    )
    return RenamexResult(status=status, error_number=ctypes.get_errno())


def publish_directory_exclusive(source: Path, destination: Path) -> None:
    """Atomically publish without replacing any destination, or fail closed."""
    if _RUNTIME_PLATFORM != "darwin":
        raise AtomicPublishError(
            reason=AtomicPublishFailure.UNSUPPORTED,
            failure_class=_RUNTIME_PLATFORM,
        )
    try:
        result = invoke_darwin_renamex(source, destination)
    except (AttributeError, OSError) as error:
        raise AtomicPublishError(
            reason=AtomicPublishFailure.UNSUPPORTED,
            failure_class=type(error).__name__,
        ) from error
    if result.status == 0:
        return
    failure_class = errno.errorcode.get(
        result.error_number,
        f"errno_{result.error_number}",
    )
    reason = (
        AtomicPublishFailure.DESTINATION_EXISTS
        if result.error_number == errno.EEXIST
        else AtomicPublishFailure.SYSTEM_ERROR
    )
    raise AtomicPublishError(reason=reason, failure_class=failure_class)


__all__ = [
    "AtomicPublishError",
    "AtomicPublishFailure",
    "RenamexResult",
    "invoke_darwin_renamex",
    "publish_directory_exclusive",
]
