"""Frozen Pydantic models for the tracked finance seed manifest."""

from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict


class _ManifestModel(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=True,
    )


class ManifestSource(_ManifestModel):
    asset: Literal["worldreasoner_public.db"]
    release: Literal["v1.0.0"]
    sha256: str
    size_bytes: int
    tag: Literal["v1.0.0"]
    url: str


class ManifestColumn(_ManifestModel):
    default: str | None
    name: str
    not_null: bool
    primary_key: bool
    type: str


class ManifestTable(_ManifestModel):
    columns: tuple[ManifestColumn, ...]
    name: str
    row_count: int


class ManifestIndex(_ManifestModel):
    name: str
    sql: str | None
    tbl_name: str
    type: Literal["index", "view", "trigger"]


class ManifestSchema(_ManifestModel):
    indexes_views_triggers: tuple[ManifestIndex, ...]
    schema_version: int
    tables: tuple[ManifestTable, ...]


class ManifestReferenceStats(_ManifestModel):
    distinct_ids: int
    distinct_questions: int
    rows: int


class ManifestReferenceCounts(_ManifestModel):
    references_distinct: int
    references_total: int
    resolvable: int
    unresolved: int


class ManifestEventReferences(_ManifestModel):
    outcome_event_ids: ManifestReferenceCounts
    related_event_ids: ManifestReferenceCounts
    target_event_id: ManifestReferenceCounts


class ManifestGraphLinkage(_ManifestModel):
    canonical_questions: int
    causal_hypotheses_by_discovered_by_question_ids: ManifestReferenceStats
    event_outcome_impacts_by_question_id: ManifestReferenceStats
    events_by_extracted_for_question_id: ManifestReferenceStats
    forecasts: ManifestReferenceStats
    graph_rows_readable: Literal[True]
    referenced_article_ids: ManifestReferenceCounts
    referenced_event_ids: ManifestEventReferences


class ManifestCanonicalSelection(_ManifestModel):
    canonical_sql: str
    missing_field_counts: dict[str, int]
    ordered_ids: tuple[str, ...]
    total_rows: int
    type_counts: dict[str, int]
    unique_ids: int


class FinanceSeedManifest(_ManifestModel):
    absent_schema_features: tuple[str, ...]
    canonical_selection: ManifestCanonicalSelection
    graph_linkage: ManifestGraphLinkage
    limitations: tuple[str, ...]
    schema_inventory: ManifestSchema
    source: ManifestSource


class SeedTypeCounts(_ManifestModel):
    binary: int
    mcq: int
    quantity: int
    timeframe: int


class SeedCanonicalAudit(_ManifestModel):
    canonical_sql: str
    total_rows: int
    unique_ids: int
    ordered_ids: tuple[str, ...]
    type_counts: SeedTypeCounts


class SeedAssetAudit(_ManifestModel):
    size_bytes: int
    sha256: str
    schema_version: int
    read_only: Literal[True]
    sqlite_sidecars: tuple[str, ...]


class SeedAuditSummary(_ManifestModel):
    status: Literal["ok"] = "ok"
    manifest_sha256: str
    manifest_size_bytes: int
    source: ManifestSource
    asset: SeedAssetAudit
    canonical_selection: SeedCanonicalAudit
    limitations: tuple[str, ...]


__all__ = [
    "FinanceSeedManifest",
    "ManifestCanonicalSelection",
    "ManifestSource",
    "SeedAuditSummary",
    "SeedTypeCounts",
]
