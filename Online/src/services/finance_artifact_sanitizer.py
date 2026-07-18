"""Fail-closed recursive finance artifact sanitization."""

import re
from typing import Final

from pydantic import ValidationError

from src.domain.finance.sanitized_artifact import (
    ArtifactSanitizationError,
    ArtifactSanitizationFailureReason,
    JsonValue,
    PersistedArtifactTree,
    TransientForbiddenValueRegistry,
)

_ACRONYM_BOUNDARY: Final = re.compile(r"([A-Z]+)([A-Z][a-z])")
_CAMEL_BOUNDARY: Final = re.compile(r"([a-z0-9])([A-Z])")
_NON_ALPHANUMERIC: Final = re.compile(r"[^A-Za-z0-9]+")
_CREDENTIAL_PATTERNS: Final[tuple[re.Pattern[str], ...]] = (
    re.compile(r"\bsk-(?:or-v1-)?[A-Za-z0-9_-]{8,}\b", re.IGNORECASE),
    re.compile(r"Bearer\s+[A-Za-z0-9._~-]{20,}"),
    re.compile(r"-----BEGIN(?: [A-Z0-9]+)? PRIVATE KEY-----"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
)
_FIXED_SENTINELS: Final = (
    "ORIGINAL_EVIDENCE_SENTINEL_7f9c",
    "ORIGINAL_DAG_SENTINEL_4a2e",
    "RAW_BODY_SENTINEL_91bd",
)
FORBIDDEN_ARTIFACT_KEYS: Final[frozenset[str]] = frozenset(
    {
        "api_key",
        "authorization",
        "proxy_authorization",
        "cookie",
        "set_cookie",
        "headers",
        "environment",
        "env",
        "request_body",
        "response_body",
        "raw_request",
        "raw_response",
        "provider_payload",
        "raw_provider_payload",
        "chain_of_thought",
        "current_target_outcome",
        "realized_outcome",
    }
)


def normalize_artifact_key(key: str) -> str:
    """Normalize acronym/camel boundaries into a closed ASCII key form."""
    acronym_split = _ACRONYM_BOUNDARY.sub(r"\1_\2", key)
    camel_split = _CAMEL_BOUNDARY.sub(r"\1_\2", acronym_split)
    collapsed = _NON_ALPHANUMERIC.sub("_", camel_split)
    return collapsed.strip("_").lower()


def _scan_string(
    value: str,
    registry: TransientForbiddenValueRegistry,
    location: str,
) -> None:
    if any(pattern.search(value) is not None for pattern in _CREDENTIAL_PATTERNS):
        raise ArtifactSanitizationError(
            reason=ArtifactSanitizationFailureReason.CREDENTIAL_PATTERN,
            location=location,
        )
    forbidden_values = (
        *_FIXED_SENTINELS,
        *registry.original_identifiers,
        *registry.raw_values,
    )
    if any(forbidden in value for forbidden in forbidden_values):
        raise ArtifactSanitizationError(
            reason=ArtifactSanitizationFailureReason.FORBIDDEN_VALUE,
            location=location,
        )


def _scan_tree(
    value: JsonValue,
    registry: TransientForbiddenValueRegistry,
    location: str,
) -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            normalized = normalize_artifact_key(key)
            if normalized in FORBIDDEN_ARTIFACT_KEYS:
                raise ArtifactSanitizationError(
                    reason=ArtifactSanitizationFailureReason.FORBIDDEN_KEY,
                    location=f"{location}.<key>",
                )
            _scan_string(key, registry, f"{location}.<key>")
            _scan_tree(nested, registry, f"{location}.{normalized}")
        return
    if isinstance(value, list):
        for index, nested in enumerate(value):
            _scan_tree(nested, registry, f"{location}[{index}]")
        return
    if isinstance(value, str):
        _scan_string(value, registry, location)


def validate_persisted_artifact(
    serialized_json: str,
    registry: TransientForbiddenValueRegistry | None = None,
) -> PersistedArtifactTree:
    """Parse and recursively reject forbidden keys, credentials, and values."""
    try:
        tree = PersistedArtifactTree.model_validate_json(serialized_json)
    except ValidationError as error:
        raise ArtifactSanitizationError(
            reason=ArtifactSanitizationFailureReason.MALFORMED_JSON,
            location="$",
        ) from error
    _scan_tree(
        tree.root,
        registry or TransientForbiddenValueRegistry(),
        "$",
    )
    return tree


__all__ = [
    "FORBIDDEN_ARTIFACT_KEYS",
    "normalize_artifact_key",
    "validate_persisted_artifact",
]
