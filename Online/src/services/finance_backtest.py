"""Resolved finance backtest preparation and post-run outcome binding."""

import json
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from pydantic import ValidationError
from typing_extensions import override

from src.domain.finance.experiment import (
    FinanceBinaryQuestion,
    FinanceExperimentManifest,
    FinanceResolutionEntry,
    FinanceResolutionManifest,
    FinanceSearchBackend,
)
from src.domain.finance.memory import OutcomeLabel, QuestionId, QuestionKind
from src.domain.finance.provider import FinanceRunMode, SearchSourcePolicy
from src.domain.finance.sanitized_artifact import ArtifactSanitizationError
from src.services.finance_experiment_artifact import (
    FinanceExperimentArtifactError,
    load_finance_experiment_bundle,
)
from src.services.finance_hashed_bundle import HashedBundleReceipt
from src.services.finance_resolution_analysis import (
    analyze_finance_resolution_manifest,
)
from src.services.finance_seed_identity import FinanceSeedIdentityVerifier


@dataclass(frozen=True, slots=True)
class FinanceBacktestError(Exception):
    """A retrospective DB boundary or backtest contract was rejected."""

    detail: str

    @override
    def __str__(self) -> str:
        return f"finance backtest failed: {self.detail}"


@dataclass(frozen=True, slots=True)
class _TargetRow:
    question_id: str
    question_text: str
    domain: str
    context: str | None
    resolution_rule: str | None
    forecast_cutoff: datetime


def _aware_datetime(raw: str, field: str) -> datetime:
    try:
        value = datetime.fromisoformat(raw)
    except ValueError as error:
        raise FinanceBacktestError(f"malformed {field}") from error
    if value.tzinfo is None or value.utcoffset() is None:
        raise FinanceBacktestError(f"naive {field}")
    return value


def _readonly_connection(db_path: Path) -> sqlite3.Connection:
    uri = f"{db_path.resolve().as_uri()}?mode=ro&immutable=1"
    try:
        return sqlite3.connect(uri, uri=True)
    except sqlite3.Error as error:
        raise FinanceBacktestError("unable to open immutable DB") from error


def _load_target_rows(db_path: Path) -> tuple[_TargetRow, ...]:
    """Load target definitions without selecting the ground-truth column."""
    sql = """
    SELECT id,question_text,domain,context,resolution_criteria,estimated_start_time
    FROM questions
    WHERE lower(domain)='finance'
      AND lower(question_type)='binary'
      AND graph_built=1
      AND ground_truth IS NOT NULL
      AND estimated_start_time IS NOT NULL
    ORDER BY estimated_start_time,id
    """
    try:
        with closing(_readonly_connection(db_path)) as connection:
            rows = connection.execute(sql).fetchall()
    except sqlite3.Error as error:
        raise FinanceBacktestError("unable to load resolved targets") from error
    targets = tuple(
        _TargetRow(
            question_id=row[0],
            question_text=row[1],
            domain=row[2],
            context=row[3],
            resolution_rule=row[4],
            forecast_cutoff=_aware_datetime(row[5], "estimated_start_time"),
        )
        for row in rows
    )
    if not targets:
        raise FinanceBacktestError("no eligible resolved binary targets")
    return targets


def _question(row: _TargetRow) -> FinanceBinaryQuestion:
    context = row.context.strip() if row.context else ""
    rule = row.resolution_rule.strip() if row.resolution_rule else ""
    if not rule:
        raise FinanceBacktestError(f"missing resolution rule for {row.question_id}")
    return FinanceBinaryQuestion(
        question_id=QuestionId(row.question_id),
        question_text=row.question_text,
        question_type=QuestionKind.BINARY,
        domain=row.domain,
        context=(context or "No additional public question context.",),
        outcome_space=(OutcomeLabel("No"), OutcomeLabel("Yes")),
        positive_label=OutcomeLabel("Yes"),
        resolution_rule=rule,
        forecast_cutoff=row.forecast_cutoff,
    )


def build_finance_backtest_manifest(
    db_path: Path,
    template: FinanceExperimentManifest,
    manifest_id: str,
    limit: int | None = None,
    model: str | None = None,
    exclude_question_ids: frozenset[str] = frozenset(),
) -> FinanceExperimentManifest:
    """Build a ground-truth-free historical manifest from resolved DB targets."""
    FinanceSeedIdentityVerifier(db_path).verify_identity()
    if not manifest_id.strip():
        raise FinanceBacktestError("manifest id must be nonempty")
    if limit is not None and limit <= 0:
        raise FinanceBacktestError("limit must be positive")
    rows = tuple(
        row
        for row in _load_target_rows(db_path)
        if row.question_id not in exclude_question_ids
    )
    if not rows:
        raise FinanceBacktestError("all eligible targets were excluded")
    selected = rows if limit is None else rows[:limit]
    questions = tuple(_question(row) for row in selected)
    latest_cutoff = max(row.forecast_cutoff for row in selected)
    search = template.search.model_copy(
        update={
            "source_policy": SearchSourcePolicy.PUBLIC_DB_METADATA,
            "backend": FinanceSearchBackend.PUBLIC_DB_METADATA_V1,
        }
    )
    forecast = template.forecast.model_copy(
        update={"model": model or template.forecast.model}
    )
    judges = tuple(
        member.model_copy(
            update={
                "settings": member.settings.model_copy(
                    update={"model": model or member.settings.model}
                )
            }
        )
        for member in template.judges
    )
    payload = template.model_dump()
    payload.update(
        {
            "manifest_id": manifest_id,
            "cutoff": latest_cutoff,
            "run_mode": FinanceRunMode.HISTORICAL_BACKTEST,
            "temporal_policy_version": "public_db_metadata_proxy/v1",
            "search": search,
            "forecast": forecast,
            "judges": judges,
            "questions": questions,
        }
    )
    try:
        return FinanceExperimentManifest.model_validate(payload)
    except ValidationError as error:
        raise FinanceBacktestError("generated manifest is invalid") from error


def write_finance_backtest_manifest(
    destination: Path,
    manifest: FinanceExperimentManifest,
) -> None:
    """Write a new manifest without replacing any existing experiment input."""
    if not destination.parent.is_dir():
        raise FinanceBacktestError("manifest parent directory does not exist")
    try:
        with destination.open("x", encoding="utf-8") as stream:
            _ = stream.write(manifest.model_dump_json(indent=2) + "\n")
    except FileExistsError as error:
        raise FinanceBacktestError("manifest destination already exists") from error
    except OSError as error:
        raise FinanceBacktestError("unable to write backtest manifest") from error


def _outcome_label(raw: str) -> OutcomeLabel:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise FinanceBacktestError("malformed ground truth") from error
    if isinstance(value, bool):
        return OutcomeLabel("Yes" if value else "No")
    if isinstance(value, str) and value.casefold() in {"yes", "no"}:
        return OutcomeLabel(value.title())
    raise FinanceBacktestError("unsupported binary ground truth")


def _resolution_manifest(
    source_bundle: Path,
    db_path: Path,
) -> FinanceResolutionManifest:
    """Load outcomes only after the immutable forecast suite is available."""
    FinanceSeedIdentityVerifier(db_path).verify_identity()
    try:
        verified = load_finance_experiment_bundle(source_bundle)
    except (FinanceExperimentArtifactError, ArtifactSanitizationError) as error:
        raise FinanceBacktestError("source suite is invalid") from error
    suite = verified.suite
    if (
        suite.manifest.run_mode is not FinanceRunMode.HISTORICAL_BACKTEST
        or suite.manifest.search.source_policy
        is not SearchSourcePolicy.PUBLIC_DB_METADATA
    ):
        raise FinanceBacktestError("suite is not a resolved historical backtest")
    question_ids = tuple(
        str(question.question_id) for question in suite.manifest.questions
    )
    placeholders = ",".join("?" for _ in question_ids)
    sql = f"""
    SELECT id,ground_truth,resolution_date FROM questions
    WHERE id IN ({placeholders}) ORDER BY id
    """
    try:
        with closing(_readonly_connection(db_path)) as connection:
            rows = connection.execute(sql, question_ids).fetchall()
    except sqlite3.Error as error:
        raise FinanceBacktestError("unable to load resolved outcomes") from error
    by_id = {row[0]: row for row in rows}
    if set(by_id) != set(question_ids):
        raise FinanceBacktestError("suite target is absent from the pinned DB")
    entries = tuple(
        FinanceResolutionEntry(
            question_id=question.question_id,
            outcome_label=_outcome_label(by_id[str(question.question_id)][1]),
            resolved_at=_aware_datetime(
                by_id[str(question.question_id)][2],
                "resolution_date",
            ),
            resolution_source="WorldReasoner public DB v1.0.0 questions.ground_truth",
        )
        for question in suite.manifest.questions
    )
    return FinanceResolutionManifest(
        schema_version="finance-resolution-manifest/v1",
        suite_id=suite.suite_id,
        experiment_manifest_id=suite.manifest.manifest_id,
        suite_sha256=verified.receipt.data_sha256,
        entries=entries,
    )


def analyze_finance_backtest(
    source_bundle: Path,
    db_path: Path,
    destination: Path,
) -> HashedBundleReceipt:
    """Bind pinned outcomes after forecasting and persist resolved metrics."""
    resolution = _resolution_manifest(source_bundle, db_path)
    return analyze_finance_resolution_manifest(
        source_bundle,
        resolution,
        destination,
    )


__all__ = [
    "FinanceBacktestError",
    "analyze_finance_backtest",
    "build_finance_backtest_manifest",
    "write_finance_backtest_manifest",
]
