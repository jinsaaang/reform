"""Persisted finance artifact sanitizer tests."""

import hashlib
import hmac
import json
from importlib.util import find_spec

import pytest

from src.domain.finance.sanitized_artifact import (
    AliasKind,
    ArtifactSanitizationError,
    ArtifactSanitizationFailureReason,
    TransientForbiddenValueRegistry,
)
from src.services.finance_aliases import AliasSource, build_run_alias_table
from src.services.finance_artifact_sanitizer import (
    normalize_artifact_key,
    validate_persisted_artifact,
)


def test_finance_artifact_sanitizer_service_exists() -> None:
    # Given / When
    specification = find_spec("src.services.finance_artifact_sanitizer")

    # Then
    assert specification is not None


def test_normalizes_camel_case_and_rejects_forbidden_keys_and_values() -> None:
    # Given
    assert tuple(
        normalize_artifact_key(key) for key in ("apiKey", "requestBody", "rawResponse")
    ) == ("api_key", "request_body", "raw_response")
    registry = TransientForbiddenValueRegistry(
        original_identifiers=("ORIGINAL_EVIDENCE_SENTINEL_7f9c",),
        raw_values=("RAW_BODY_SENTINEL_91bd",),
    )
    for serialized in (
        '{"outer":{"apiKey":"hidden"}}',
        '{"safe":"ORIGINAL_EVIDENCE_SENTINEL_7f9c"}',
        '{"safe":"prefix RAW_BODY_SENTINEL_91bd suffix"}',
    ):
        with pytest.raises(ArtifactSanitizationError):
            validate_persisted_artifact(serialized, registry)


@pytest.mark.parametrize(
    "key",
    ("APIKey", "proxyAuthorization", "set-cookie", "raw_provider_payload"),
)
def test_recursive_key_denylist_is_closed_over_normalization(key: str) -> None:
    # Given
    serialized = f'{{"outer":[{{"{key}":"value"}}]}}'

    # When
    with pytest.raises(ArtifactSanitizationError) as captured:
        validate_persisted_artifact(serialized)

    # Then
    assert captured.value.reason is ArtifactSanitizationFailureReason.FORBIDDEN_KEY


def test_alias_audit_uses_exact_namespaced_hmac_and_hides_original() -> None:
    # Given
    salt = b"f" * 32
    original = "ORIGINAL_DAG_SENTINEL_4a2e"
    material = f"finance-alias-audit/v1\0memory\0{original}".encode()
    expected = hmac.new(salt, material, hashlib.sha256).hexdigest()

    # When
    aliases = build_run_alias_table(
        salt,
        (AliasSource(AliasKind.MEMORY, original, "memory_001"),),
    )
    serialized_audit = aliases.audit.model_dump_json()

    # Then
    assert aliases.audit.mappings[0].digest == expected
    assert original not in serialized_audit


def test_alias_replacement_is_longest_first() -> None:
    # Given
    aliases = build_run_alias_table(
        b"a" * 32,
        (
            AliasSource(AliasKind.MEMORY, "dag-alpha", "memory_001"),
            AliasSource(
                AliasKind.EVENT,
                "dag-alpha-node",
                "node_001",
            ),
        ),
    )

    # When
    sanitized = aliases.replace_text("dag-alpha-node follows dag-alpha")

    # Then
    assert sanitized == "node_001 follows memory_001"


def test_artifact_scan_rejects_credential_without_retaining_value() -> None:
    # Given
    credential = "sk-" + "or-v1-" + "abcdef1234567890"
    serialized = json.dumps({"safe": credential})

    # When
    with pytest.raises(ArtifactSanitizationError) as captured:
        validate_persisted_artifact(serialized)

    # Then
    assert captured.value.reason is (
        ArtifactSanitizationFailureReason.CREDENTIAL_PATTERN
    )
    assert credential not in str(captured.value)


@pytest.mark.parametrize(
    "credential",
    (
        "Bearer " + "A" * 20,
        "Bearer " + "A" * 25,
        "prefix Bearer\t" + "Az09._~-" * 3 + " suffix",
    ),
)
def test_artifact_scan_rejects_bearer_credentials_in_safe_string_leaves(
    credential: str,
) -> None:
    # Given
    serialized = json.dumps({"safe": {"nested": [credential]}})

    # When
    with pytest.raises(ArtifactSanitizationError) as captured:
        validate_persisted_artifact(serialized)

    # Then
    assert captured.value.reason is (
        ArtifactSanitizationFailureReason.CREDENTIAL_PATTERN
    )
    assert credential not in str(captured.value)


@pytest.mark.parametrize(
    "safe_value",
    (
        "Bearer " + "A" * 19,
        "Bearer " + "A" * 10 + "/" + "B" * 10,
    ),
)
def test_bearer_scan_respects_minimum_length_and_allowed_charset(
    safe_value: str,
) -> None:
    # Given
    serialized = json.dumps({"safe": safe_value})

    # When
    tree = validate_persisted_artifact(serialized)

    # Then
    assert tree.root == {"safe": safe_value}
