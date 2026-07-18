"""Read-only adapter for the immutable WorldReasoner v1 finance seed."""

import hashlib
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Generic, TypeVar, assert_never

from pydantic import TypeAdapter, ValidationError

from src.domain.finance.memory import (
    ArticleId,
    CausalEdgeId,
    DagId,
    EpisodeId,
    EpisodeRelationMetadata,
    EventId,
    HistoricalCausalEdge,
    HistoricalEventNode,
    HistoricalImpact,
    HistoricalOutcomeMetadata,
    HistoricalQuestionProfile,
    HistoricalResolutionMetadata,
    ImpactId,
    QuestionId,
    QuestionKind,
    ResolutionProvenance,
    ResolvedDagEpisode,
    SeedBoundaryRow,
    SeedCausalEdgeRow,
    SeedEventRow,
    SeedImpactRow,
    SeedQuestionRow,
    SeedSchemaNameRow,
    SeedSchemaVersionRow,
)
from src.domain.finance.retrieval import (
    FINANCE_SEED_V1_SPEC,
    DuplicateEpisodeIdError,
    FinanceSeedAssetSpec,
    SeedAssetMismatchError,
    SeedDatabaseReadError,
    SeedDataError,
    SeedSchemaError,
    UnsafeSeedPathError,
    WritableSeedAssetError,
)

# fmt: off
__all__ = ("DuplicateEpisodeIdError", "FinanceSeedAssetSpec", "FinanceSeedRepository", "SeedAssetMismatchError", "SeedDataError", "SeedSchemaError", "UnsafeSeedPathError", "WritableSeedAssetError")
# fmt: on


_RowT = TypeVar("_RowT", bound=SeedBoundaryRow)
_SQLITE_JSON_ROWS: Final = TypeAdapter(tuple[tuple[str], ...])


@dataclass(frozen=True, slots=True)
class _RowQuery(Generic[_RowT]):
    label: str
    sql: str
    row_type: type[_RowT]


@dataclass(frozen=True, slots=True)
class _SeedRows:
    questions: tuple[SeedQuestionRow, ...]
    events: tuple[SeedEventRow, ...]
    edges: tuple[SeedCausalEdgeRow, ...]
    impacts: tuple[SeedImpactRow, ...]


_QUESTION_SQL: Final = """
SELECT json_object(
 'question_id',id,'question_text',question_text,'question_type',question_type,
 'domain',domain,'source',source,'context',context,'resolution_date',resolution_date,
 'ground_truth',ground_truth,'resolution_rule',resolution_criteria,'options',options,
 'quantity_unit',quantity_unit,'outcome_event_ids',outcome_event_ids)
FROM questions WHERE lower(domain)='finance' AND graph_built=1 ORDER BY id
"""
_EVENT_SQL: Final = """
WITH fq AS (SELECT id,outcome_event_ids FROM questions WHERE lower(domain)='finance' AND graph_built=1),
qe AS (SELECT fq.id question_id,e.* FROM fq JOIN events e ON e.extracted_for_question_id=fq.id
UNION SELECT fq.id,e.* FROM fq JOIN json_each(fq.outcome_event_ids) j JOIN events e ON e.id=j.value)
SELECT json_object('question_id',question_id,'event_id',id,'title',title,'description',description,
'event_type',event_type,'domain',domain,'tags',tags,'occurred_at',occurred_date,
'predicted_at',predicted_date,'article_ids',article_ids,'is_outcome',is_outcome,
'outcome_scenario',outcome_scenario,'is_actual_outcome',is_actual_outcome) FROM qe ORDER BY question_id,id
"""
_EDGE_SQL: Final = """
SELECT json_object('question_id',j.value,'edge_id',c.id,'source_event_id',c.source_event_id,
'target_event_id',c.target_event_id,'relation_type',c.relation_type,'strength',c.strength,
'confidence',c.confidence,'time_lag_hours',c.time_lag_hours,'reasoning',c.reasoning,
'evidence_article_ids',c.evidence_article_ids) FROM causal_hypotheses c
JOIN json_each(c.discovered_by_question_ids) j JOIN questions q ON q.id=j.value
WHERE lower(q.domain)='finance' AND q.graph_built=1 ORDER BY j.value,c.id
"""
_IMPACT_SQL: Final = """
SELECT json_object('question_id',i.question_id,'impact_id',i.id,'event_id',i.event_id,
'outcome_event_id',i.outcome_event_id,'direction',i.impact_direction,
'magnitude',i.impact_magnitude,'confidence',i.confidence,'reasoning',i.reasoning,
'evidence_article_ids',i.evidence_article_ids,'causal_edge_ids',i.causal_chain_hypothesis_ids)
FROM event_outcome_impacts i JOIN questions q ON q.id=i.question_id
WHERE lower(q.domain)='finance' AND q.graph_built=1 ORDER BY i.question_id,i.id
"""

# This compact immutable table mirrors the four SELECT boundaries above.
# fmt: off
_REQUIRED_COLUMNS: Final = (
    ("questions", ("id", "question_text", "question_type", "domain", "source", "context", "resolution_date", "ground_truth", "resolution_criteria", "options", "quantity_unit", "outcome_event_ids", "graph_built")),
    ("events", ("id", "title", "description", "event_type", "domain", "tags", "occurred_date", "predicted_date", "article_ids", "extracted_for_question_id", "is_outcome", "outcome_scenario", "is_actual_outcome")),
    ("causal_hypotheses", ("id", "source_event_id", "target_event_id", "relation_type", "strength", "confidence", "time_lag_hours", "reasoning", "evidence_article_ids", "discovered_by_question_ids")),
    ("event_outcome_impacts", ("id", "event_id", "outcome_event_id", "question_id", "impact_direction", "impact_magnitude", "confidence", "reasoning", "evidence_article_ids", "causal_chain_hypothesis_ids")),
)
# fmt: on


def _read_rows(
    connection: sqlite3.Connection, query: _RowQuery[_RowT]
) -> tuple[_RowT, ...]:
    parsed: list[_RowT] = []
    try:
        payload_rows = _SQLITE_JSON_ROWS.validate_python(
            connection.execute(query.sql).fetchall()
        )
    except ValidationError as error:
        raise SeedDataError(query.label, str(error)) from error
    for (payload,) in payload_rows:
        try:
            parsed.append(query.row_type.model_validate_json(payload))
        except ValidationError as error:
            raise SeedDataError(query.label, str(error)) from error
    return tuple(parsed)


@dataclass(frozen=True, slots=True)
class FinanceSeedRepository:
    """Verify and reconstruct canonical finance episodes without SQLite writes."""

    db_path: Path
    asset_spec: FinanceSeedAssetSpec = FINANCE_SEED_V1_SPEC

    def __post_init__(self) -> None:
        raw_path = str(self.db_path)
        if raw_path.startswith("file:") or "?mode=" in raw_path:
            raise UnsafeSeedPathError(self.db_path)

    def load_episodes(self) -> tuple[ResolvedDagEpisode, ...]:
        """Load all canonical finance graph-built episodes in stable ID order."""
        self._verify_asset()
        uri = f"{self.db_path.resolve().as_uri()}?mode=ro&immutable=1"
        try:
            with closing(sqlite3.connect(uri, uri=True)) as connection:
                self._validate_schema(connection)
                rows = _SeedRows(
                    questions=_read_rows(
                        connection,
                        _RowQuery("questions", _QUESTION_SQL, SeedQuestionRow),
                    ),
                    events=_read_rows(
                        connection, _RowQuery("events", _EVENT_SQL, SeedEventRow)
                    ),
                    edges=_read_rows(
                        connection,
                        _RowQuery("causal_hypotheses", _EDGE_SQL, SeedCausalEdgeRow),
                    ),
                    impacts=_read_rows(
                        connection,
                        _RowQuery("event_outcome_impacts", _IMPACT_SQL, SeedImpactRow),
                    ),
                )
        except sqlite3.Error as error:
            raise SeedDatabaseReadError(self.db_path, str(error)) from error
        return self._assemble(rows)

    def _verify_asset(self) -> None:
        if not self.db_path.is_file():
            raise SeedAssetMismatchError(self.db_path, "path", "file", "missing")
        if self.db_path.stat().st_mode & 0o222:
            raise WritableSeedAssetError(self.db_path)
        actual_size = self.db_path.stat().st_size
        if actual_size != self.asset_spec.byte_size:
            raise SeedAssetMismatchError(
                self.db_path, "size", self.asset_spec.byte_size, actual_size
            )
        with self.db_path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != self.asset_spec.sha256:
            raise SeedAssetMismatchError(
                self.db_path, "sha256", self.asset_spec.sha256, digest
            )

    def _validate_schema(self, connection: sqlite3.Connection) -> None:
        versions = _read_rows(
            connection,
            _RowQuery(
                "schema",
                "SELECT json_object('version',schema_version) FROM pragma_schema_version",
                SeedSchemaVersionRow,
            ),
        )
        if len(versions) != 1 or versions[0].version != self.asset_spec.schema_version:
            actual = versions[0].version if versions else "missing"
            raise SeedSchemaError(
                self.db_path,
                f"schema version expected {self.asset_spec.schema_version}, got {actual}",
            )
        for table, required in _REQUIRED_COLUMNS:
            sql = f"SELECT json_object('name',name) FROM pragma_table_info('{table}')"
            columns = {
                row.name
                for row in _read_rows(
                    connection, _RowQuery(table, sql, SeedSchemaNameRow)
                )
            }
            missing = tuple(name for name in required if name not in columns)
            if missing:
                raise SeedSchemaError(
                    self.db_path, f"{table} missing columns {missing}"
                )

    def _assemble(self, rows: _SeedRows) -> tuple[ResolvedDagEpisode, ...]:
        question_ids = tuple(QuestionId(row.question_id) for row in rows.questions)
        if len(question_ids) != len(set(question_ids)):
            duplicate = next(
                identifier
                for identifier in question_ids
                if question_ids.count(identifier) > 1
            )
            raise DuplicateEpisodeIdError(duplicate)
        if len(rows.questions) != self.asset_spec.episode_count:
            raise SeedDataError(
                "questions",
                f"expected {self.asset_spec.episode_count} episodes, got {len(rows.questions)}",
            )
        episodes: list[ResolvedDagEpisode] = []
        for question in rows.questions:
            episodes.append(self._assemble_episode(question, rows))
        return tuple(episodes)

    def _assemble_episode(
        self,
        question: SeedQuestionRow,
        rows: _SeedRows,
    ) -> ResolvedDagEpisode:
        question_id = QuestionId(question.question_id)
        if question.options:
            outcome_space = question.options
        else:
            match question.question_type:
                case QuestionKind.BINARY:
                    outcome_space = ("Yes", "No")
                case QuestionKind.MCQ | QuestionKind.QUANTITY | QuestionKind.TIMEFRAME:
                    outcome_space = ()
                case _ as unreachable:
                    assert_never(unreachable)
        # Positional conversions mirror the ordered frozen dataclass schemas.
        # fmt: off
        return ResolvedDagEpisode(
            episode_id=EpisodeId(f"historical:{question.question_id}:episode"),
            question_id=question_id,
            dag_id=DagId(f"historical:{question.question_id}:dag"),
            source_profile=HistoricalQuestionProfile(question.question_text, question.question_type, question.domain, question.source, question.context, outcome_space, question.resolution_rule, question.quantity_unit),
            historical_resolution=HistoricalResolutionMetadata(question.resolution_date, ResolutionProvenance.BOOTSTRAP_RESOLUTION_DATE_PROXY),
            historical_outcome=HistoricalOutcomeMetadata(question.ground_truth, tuple(EventId(value) for value in question.outcome_event_ids)),
            nodes=tuple(HistoricalEventNode(EventId(row.event_id), row.title, row.description, row.event_type, row.domain, row.tags or (), row.occurred_at, row.predicted_at, tuple(ArticleId(value) for value in (row.article_ids or ())), row.is_outcome, row.outcome_scenario, row.is_actual_outcome) for row in rows.events if row.question_id == question.question_id),
            edges=tuple(HistoricalCausalEdge(CausalEdgeId(row.edge_id), EventId(row.source_event_id), EventId(row.target_event_id), row.relation_type, row.strength, row.confidence, row.time_lag_hours, row.reasoning, tuple(ArticleId(value) for value in (row.evidence_article_ids or ()))) for row in rows.edges if row.question_id == question.question_id),
            impacts=tuple(HistoricalImpact(ImpactId(row.impact_id), EventId(row.event_id), EventId(row.outcome_event_id), row.direction, row.magnitude, row.confidence, row.reasoning, tuple(ArticleId(value) for value in (row.evidence_article_ids or ())), tuple(CausalEdgeId(value) for value in (row.causal_edge_ids or ()))) for row in rows.impacts if row.question_id == question.question_id),
            relation_metadata=EpisodeRelationMetadata.public_db_unavailable(),
        )
        # fmt: on
