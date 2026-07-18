"""Closed persisted-artifact and alias-audit contracts."""

from dataclasses import dataclass
from enum import StrEnum, unique
from typing import Annotated, ClassVar, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, RootModel, model_validator
from pydantic_core import PydanticCustomError
from typing_extensions import TypeAliasType, override

JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue = TypeAliasType(
    "JsonValue",
    JsonScalar | list["JsonValue"] | dict[str, "JsonValue"],
)


class _SanitizedModel(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=True,
        str_strip_whitespace=True,
    )


@unique
class AliasKind(StrEnum):
    """Identity classes that may enter sanitized reasoning."""

    TARGET = "target"
    EVIDENCE = "evidence"
    MEMORY = "memory"
    EVENT = "event"
    EDGE = "edge"
    IMPACT = "impact"
    ARTICLE = "article"
    SCENARIO = "scenario"
    IDENTITY = "identity"


class AliasAuditEntry(_SanitizedModel):
    """One non-reversible alias-to-HMAC audit entry."""

    kind: AliasKind
    alias: str = Field(min_length=1)
    digest: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class FinanceAliasAudit(_SanitizedModel):
    """Run-local salt and namespaced HMAC mappings without original IDs."""

    schema_version: Literal["finance-alias-audit/v1"] = "finance-alias-audit/v1"
    salt_base64: Annotated[str, Field(pattern=r"^[A-Za-z0-9+/]{43}=$")]
    mappings: tuple[AliasAuditEntry, ...]

    @model_validator(mode="after")
    def validate_unique_mappings(self) -> "FinanceAliasAudit":
        """Reject ambiguous aliases and duplicate audit coordinates."""
        coordinates = tuple((item.kind, item.alias) for item in self.mappings)
        if len(coordinates) != len(set(coordinates)):
            raise PydanticCustomError(
                "duplicate_alias_audit_mapping",
                "alias audit coordinates must be unique",
            )
        return self


class TransientForbiddenValueRegistry(_SanitizedModel):
    """Non-persisted originals and exact bodies that must never survive."""

    original_identifiers: tuple[str, ...] = ()
    raw_values: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_values(self) -> "TransientForbiddenValueRegistry":
        """Require useful, unambiguous transient scan values."""
        if any(not value for value in self.original_identifiers):
            raise PydanticCustomError(
                "empty_original_identifier",
                "original identifiers must be nonempty",
            )
        if any(len(value) < 8 for value in self.raw_values):
            raise PydanticCustomError(
                "short_transient_forbidden_value",
                "transient raw values must contain at least eight characters",
            )
        return self


class PersistedArtifactTree(RootModel[JsonValue]):
    """Typed recursive JSON tree admitted at the persistence boundary."""

    model_config = ConfigDict(frozen=True, strict=True)


@unique
class ArtifactSanitizationFailureReason(StrEnum):
    """Closed diagnostics safe to retain after sanitizer rejection."""

    FORBIDDEN_KEY = "forbidden_key"
    FORBIDDEN_VALUE = "forbidden_value"
    CREDENTIAL_PATTERN = "credential_pattern"
    MALFORMED_JSON = "malformed_json"
    INVALID_ALIAS_SALT = "invalid_alias_salt"
    AMBIGUOUS_IDENTIFIER = "ambiguous_identifier"
    UNKNOWN_IDENTIFIER = "unknown_identifier"


@dataclass(frozen=True, slots=True)
class ArtifactSanitizationError(Exception):
    """Fail-closed sanitizer error without the rejected secret or body."""

    reason: ArtifactSanitizationFailureReason
    location: str

    @override
    def __str__(self) -> str:
        return f"artifact sanitization failed: {self.reason.value} at {self.location}"


__all__ = [
    "AliasAuditEntry",
    "AliasKind",
    "ArtifactSanitizationError",
    "ArtifactSanitizationFailureReason",
    "FinanceAliasAudit",
    "JsonScalar",
    "JsonValue",
    "PersistedArtifactTree",
    "TransientForbiddenValueRegistry",
]
