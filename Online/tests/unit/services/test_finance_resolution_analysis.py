"""Behavioral contracts for separately invoked resolution analysis."""

from datetime import UTC, datetime
from decimal import Decimal
from hashlib import sha256
from pathlib import Path
from uuid import UUID

import pytest

from src.domain.finance.experiment import (
    FinanceResolutionEntry,
    FinanceResolutionManifest,
)
from src.domain.finance.experiment_resolution_metrics import BrierDirection
from src.services.finance_experiment_artifact import persist_finance_experiment_bundle
from src.services.finance_resolution_analysis import (
    FinanceResolutionAnalysisError,
    analyze_finance_resolution,
    binary_brier,
    load_finance_resolution_analysis_bundle,
)
from tests.fixtures.finance_experiment_artifact import make_persisted_suite


def test_binary_brier_is_one_class_squared_error() -> None:
    # Given
    probability = Decimal("0.7")

    # When
    score = binary_brier(probability, resolved_positive=True)

    # Then
    assert score == Decimal("0.09")


def _write_resolution(path: Path, suite_hash: str, suite_id: UUID) -> None:
    suite = make_persisted_suite()
    manifest = FinanceResolutionManifest(
        schema_version="finance-resolution-manifest/v1",
        suite_id=suite_id,
        experiment_manifest_id=suite.manifest.manifest_id,
        suite_sha256=suite_hash,
        entries=(
            FinanceResolutionEntry(
                question_id=suite.manifest.questions[0].question_id,
                outcome_label=suite.manifest.questions[0].positive_label,
                resolved_at=datetime(2027, 1, 2, tzinfo=UTC),
                resolution_source="RAW_BODY_SENTINEL_91bd",
            ),
        ),
    )
    path.write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")


def test_matching_resolution_writes_distinct_exact_brier_bundle(tmp_path: Path) -> None:
    # Given
    source = tmp_path / "ex-ante"
    destination = tmp_path / "resolution"
    suite = make_persisted_suite()
    receipt = persist_finance_experiment_bundle(source, suite)
    resolution_path = tmp_path / "resolution.json"
    _write_resolution(resolution_path, receipt.data_sha256, suite.suite_id)
    before_hash = sha256((source / "suite.json").read_bytes()).hexdigest()

    # When
    analyze_finance_resolution(source, resolution_path, destination)
    verified = load_finance_resolution_analysis_bundle(destination)

    # Then
    assert source != destination
    assert sha256((source / "suite.json").read_bytes()).hexdigest() == before_hash
    first_trial = verified.analysis.trials[0]
    assert tuple(score.brier for score in first_trial.arm_scores) == (
        Decimal("0.16"),
        Decimal("0.36"),
        Decimal("0.25"),
    )
    assert tuple(item.direction for item in first_trial.pair_scores) == (
        BrierDirection.WORSE,
        BrierDirection.BETTER,
        BrierDirection.WORSE,
    )
    assert verified.analysis.pair_aggregates[0].successful_trial_count == 2
    assert verified.analysis.pair_aggregates[0].eligible_trial_count == 2
    assert verified.analysis.pair_aggregates[0].macro_first_brier == Decimal("0.10")
    assert verified.analysis.pair_aggregates[0].macro_second_brier == Decimal("0.425")
    assert {path.name for path in destination.iterdir()} == {
        "analysis.json",
        "report.md",
        "SHA256SUMS",
    }
    serialized = (destination / "analysis.json").read_text(encoding="utf-8")
    assert "RAW_BODY_SENTINEL_91bd" not in serialized
    assert '"resolution_source"' not in serialized
    assert '"resolution_source_sha256"' in serialized


def test_mismatched_resolution_fails_before_output(tmp_path: Path) -> None:
    # Given
    source = tmp_path / "ex-ante"
    destination = tmp_path / "resolution"
    suite = make_persisted_suite()
    receipt = persist_finance_experiment_bundle(source, suite)
    resolution_path = tmp_path / "resolution.json"
    _write_resolution(
        resolution_path,
        receipt.data_sha256,
        UUID("90000000-0000-0000-0000-000000000009"),
    )

    # When
    with pytest.raises(FinanceResolutionAnalysisError):
        analyze_finance_resolution(source, resolution_path, destination)

    # Then
    assert not destination.exists()
    assert tuple(tmp_path.glob(".resolution.tmp-*")) == ()
