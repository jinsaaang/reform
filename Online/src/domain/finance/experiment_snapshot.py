"""Canonical fixed-pack evidence snapshots for finance experiments."""

import json
from datetime import UTC
from hashlib import sha256
from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, field_validator
from pydantic_core import PydanticCustomError

from src.domain.finance.search import EvidenceItem, EvidencePack


class FinanceEvidenceSnapshot(BaseModel):
    """DAG-independent evidence bytes replayed identically to arms B and C."""

    model_config: ClassVar[ConfigDict] = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=True,
    )

    schema_version: Literal["finance-evidence-snapshot/v1"] = (
        "finance-evidence-snapshot/v1"
    )
    items: tuple[EvidenceItem, ...]

    @field_validator("items", mode="after")
    @classmethod
    def canonicalize_items(
        cls,
        items: tuple[EvidenceItem, ...],
    ) -> tuple[EvidenceItem, ...]:
        """Reject duplicate IDs, normalize UTC, and impose canonical order."""
        evidence_ids = tuple(item.evidence_id for item in items)
        if len(evidence_ids) != len(set(evidence_ids)):
            raise PydanticCustomError(
                "duplicate_snapshot_evidence",
                "snapshot evidence IDs must be unique",
            )
        normalized = tuple(
            item.model_copy(
                update={
                    "available_at": item.available_at.astimezone(UTC),
                    "retrieved_at": item.retrieved_at.astimezone(UTC),
                }
            )
            for item in items
        )
        return tuple(
            sorted(
                normalized,
                key=lambda item: (
                    item.available_at,
                    item.citation,
                    item.content_hash,
                    item.evidence_id,
                ),
            )
        )

    @classmethod
    def from_evidence_pack(cls, pack: EvidencePack) -> "FinanceEvidenceSnapshot":
        """Freeze only a DAG-independent pack and exclude historical references."""
        if pack.historical_dag_references:
            raise PydanticCustomError(
                "historical_reference_in_snapshot",
                "fixed evidence snapshots cannot contain historical references",
            )
        return cls(items=pack.items)

    def canonical_bytes(self) -> bytes:
        """Return compact sorted-key UTF-8 JSON without a trailing newline."""
        return json.dumps(
            self.model_dump(mode="json"),
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")

    @property
    def digest(self) -> str:
        """Return the lowercase SHA-256 digest of the canonical bytes."""
        return sha256(self.canonical_bytes()).hexdigest()

    @property
    def byte_length(self) -> int:
        """Return the exact canonical byte count embedded in arm audits."""
        return len(self.canonical_bytes())


__all__ = ["FinanceEvidenceSnapshot"]
