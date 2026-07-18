"""Run-local deterministic aliases with namespaced HMAC audit."""

import base64
import hashlib
import hmac
from dataclasses import dataclass

from src.domain.finance.sanitized_artifact import (
    AliasAuditEntry,
    AliasKind,
    ArtifactSanitizationError,
    ArtifactSanitizationFailureReason,
    FinanceAliasAudit,
    TransientForbiddenValueRegistry,
)


@dataclass(frozen=True, slots=True)
class AliasSource:
    """One transient original-to-neutral alias assignment."""

    kind: AliasKind
    original_id: str
    alias: str
    forbid_original: bool = True


@dataclass(frozen=True, slots=True)
class _AliasMapping:
    kind: AliasKind
    original_id: str
    alias: str
    forbid_original: bool


@dataclass(frozen=True, slots=True)
class RunAliasTable:
    """Transient lookup plus its persistence-safe audit projection."""

    mappings: tuple[_AliasMapping, ...]
    audit: FinanceAliasAudit

    def alias_for(self, original_id: str) -> str:
        """Resolve one known identifier without exposing it on failure."""
        for mapping in self.mappings:
            if mapping.original_id == original_id:
                return mapping.alias
        raise ArtifactSanitizationError(
            reason=ArtifactSanitizationFailureReason.UNKNOWN_IDENTIFIER,
            location="alias_lookup",
        )

    def replace_text(self, text: str) -> str:
        """Replace every known original longest-first to handle overlaps."""
        replaced = text
        ordered = sorted(
            self.mappings,
            key=lambda item: (-len(item.original_id), item.original_id),
        )
        for mapping in ordered:
            replaced = replaced.replace(mapping.original_id, mapping.alias)
        return replaced

    def forbidden_registry(
        self,
        transient: TransientForbiddenValueRegistry,
    ) -> TransientForbiddenValueRegistry:
        """Combine alias originals with caller-held exact forbidden values."""
        originals = tuple(
            mapping.original_id for mapping in self.mappings if mapping.forbid_original
        )
        return TransientForbiddenValueRegistry(
            original_identifiers=(*originals, *transient.original_identifiers),
            raw_values=transient.raw_values,
        )


def build_run_alias_table(
    salt: bytes,
    sources: tuple[AliasSource, ...],
) -> RunAliasTable:
    """Build aliases and exact finance-alias-audit/v1 HMAC digests."""
    if len(salt) != 32:
        raise ArtifactSanitizationError(
            reason=ArtifactSanitizationFailureReason.INVALID_ALIAS_SALT,
            location="alias_salt",
        )
    seen_originals: dict[str, str] = {}
    seen_coordinates: set[tuple[AliasKind, str]] = set()
    mappings: list[_AliasMapping] = []
    audit_entries: list[AliasAuditEntry] = []
    for source in sources:
        known_alias = seen_originals.get(source.original_id)
        if (
            not source.original_id
            or not source.alias
            or source.original_id == source.alias
            or (known_alias is not None and known_alias != source.alias)
        ):
            raise ArtifactSanitizationError(
                reason=ArtifactSanitizationFailureReason.AMBIGUOUS_IDENTIFIER,
                location="alias_source",
            )
        coordinate = (source.kind, source.original_id)
        if coordinate in seen_coordinates:
            continue
        seen_originals[source.original_id] = source.alias
        seen_coordinates.add(coordinate)
        mappings.append(
            _AliasMapping(
                kind=source.kind,
                original_id=source.original_id,
                alias=source.alias,
                forbid_original=source.forbid_original,
            )
        )
        material = (
            f"finance-alias-audit/v1\0{source.kind.value}\0{source.original_id}"
        ).encode("utf-8")
        audit_entries.append(
            AliasAuditEntry(
                kind=source.kind,
                alias=source.alias,
                digest=hmac.new(salt, material, hashlib.sha256).hexdigest(),
            )
        )
    return RunAliasTable(
        mappings=tuple(mappings),
        audit=FinanceAliasAudit(
            salt_base64=base64.b64encode(salt).decode("ascii"),
            mappings=tuple(audit_entries),
        ),
    )


__all__ = ["AliasSource", "RunAliasTable", "build_run_alias_table"]
