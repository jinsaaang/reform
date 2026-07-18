"""Contract tests for versioned finance experiments."""

from datetime import datetime, timedelta, timezone
from importlib.util import find_spec

import pytest
from pydantic import ValidationError

from src.domain.finance.experiment import (
    FinanceEvidenceSnapshot,
    FinanceExperimentManifest,
    FinanceExperimentTrial,
    ForecastArm,
    ProviderSeedUnsupported,
    ProviderUsageUnavailable,
    TreatmentAudit,
)
from src.domain.finance.memory import EvidenceId, OutcomeLabel, QuestionId
from src.domain.finance.search import EvidenceDirection, EvidenceItem
from tests.unit.domain.finance._experiment_factories import (
    make_manifest,
    make_resolution,
    make_suite,
)


def _evidence(
    identifier: str,
    hour: int,
    offset_hours: int = 0,
) -> EvidenceItem:
    zone = timezone(timedelta(hours=offset_hours))
    return EvidenceItem(
        evidence_id=EvidenceId(identifier),
        claim=f"Claim {identifier}",
        citation=f"https://example.test/{identifier}",
        available_at=datetime(2026, 7, 17, hour, tzinfo=zone),
        retrieved_at=datetime(2026, 7, 17, hour + 1, tzinfo=zone),
        content_hash=f"sha256:{identifier:0<64}"[:71],
        direction=EvidenceDirection.SUPPORTS,
        context_slot="current_state",
    )


def test_three_arm_manifest_contract_exists() -> None:
    # Given / When
    specification = find_spec("src.domain.finance.experiment")

    # Then
    assert specification is not None


def test_manifest_round_trip_preserves_two_question_order() -> None:
    # Given
    payload = make_manifest().model_dump_json()

    # When
    first = FinanceExperimentManifest.model_validate_json(payload)
    second = FinanceExperimentManifest.model_validate_json(payload)

    # Then
    assert first == second
    assert tuple(question.question_id for question in first.questions) == (
        QuestionId("question-alpha"),
        QuestionId("question-beta"),
    )
    assert first.arm_order == tuple(ForecastArm)


def _malformed_manifest_payloads() -> tuple[str, ...]:
    manifest = make_manifest()
    first_question = manifest.questions[0]
    duplicate_questions = manifest.model_copy(
        update={"questions": (first_question, first_question)}
    )
    duplicate_judges = manifest.model_copy(
        update={"judges": (manifest.judges[0], manifest.judges[1], manifest.judges[1])}
    )
    missing_arm = manifest.model_copy(update={"arm_order": tuple(ForecastArm)[:2]})
    naive_cutoff = manifest.model_copy(update={"cutoff": datetime(2026, 7, 18)})
    invalid_question = first_question.model_copy(
        update={"positive_label": OutcomeLabel("Maybe")}
    )
    invalid_positive = manifest.model_copy(
        update={"questions": (invalid_question, manifest.questions[1])}
    )
    extra_secret = manifest.model_dump_json().replace(
        '"manifest_id":',
        '"api_key":"forbidden-sentinel","manifest_id":',
        1,
    )
    return (
        duplicate_questions.model_dump_json(),
        duplicate_judges.model_dump_json(),
        missing_arm.model_dump_json(),
        naive_cutoff.model_dump_json(),
        invalid_positive.model_dump_json(),
        extra_secret,
    )


@pytest.mark.parametrize("payload", _malformed_manifest_payloads())
def test_manifest_rejects_malformed_or_secret_boundaries(payload: str) -> None:
    # Given / When / Then
    with pytest.raises(ValidationError):
        FinanceExperimentManifest.model_validate_json(payload)


def test_snapshot_bytes_are_ordered_utc_stable_and_history_free() -> None:
    # Given
    later = _evidence("later", 12)
    earlier_offset = _evidence("earlier", 14, offset_hours=3)

    # When
    first = FinanceEvidenceSnapshot(items=(later, earlier_offset))
    second = FinanceEvidenceSnapshot(items=(earlier_offset, later))
    canonical = first.canonical_bytes()

    # Then
    assert canonical == second.canonical_bytes()
    assert canonical.endswith(b"}")
    assert not canonical.endswith(b"\n")
    assert b'"schema_version":"finance-evidence-snapshot/v1"' in canonical
    assert b'"available_at":"2026-07-17T11:00:00Z"' in canonical
    assert b"historical_dag_references" not in canonical
    assert first.digest == second.digest
    assert first.byte_length == len(canonical)


def test_snapshot_rejects_duplicate_evidence_ids() -> None:
    # Given
    evidence = _evidence("duplicate", 8)

    # When / Then
    with pytest.raises(ValidationError):
        FinanceEvidenceSnapshot(items=(evidence, evidence))


def test_suite_records_fixed_snapshot_and_explicit_unsupported_telemetry() -> None:
    # Given / When
    suite = make_suite()
    search_only = suite.trials[0].arms[1]
    search_dag = suite.trials[0].arms[2]
    attempt = search_only.attempts[0]

    # Then
    assert search_only.treatment is not None
    assert search_dag.treatment is not None
    assert (
        search_only.treatment.evidence_snapshot_digest
        == search_dag.treatment.evidence_snapshot_digest
    )
    assert (
        search_only.treatment.evidence_snapshot_bytes
        == search_dag.treatment.evidence_snapshot_bytes
    )
    assert isinstance(attempt.seed, ProviderSeedUnsupported)
    assert isinstance(attempt.usage, ProviderUsageUnavailable)


def test_trial_rejects_search_snapshot_mismatch() -> None:
    # Given
    trial = make_suite().trials[0]
    search_dag = trial.arms[2]
    assert search_dag.treatment is not None
    mismatched_treatment = search_dag.treatment.model_copy(
        update={"evidence_snapshot_digest": "c" * 64}
    )
    mismatched_arm = search_dag.model_copy(update={"treatment": mismatched_treatment})
    invalid = trial.model_copy(update={"arms": (*trial.arms[:2], mismatched_arm)})

    # When / Then
    with pytest.raises(ValidationError):
        FinanceExperimentTrial.model_validate_json(invalid.model_dump_json())


def test_treatment_rejects_direct_arm_with_evidence() -> None:
    # Given / When / Then
    with pytest.raises(ValidationError):
        TreatmentAudit(
            arm=ForecastArm.DIRECT,
            evidence_snapshot_digest="d" * 64,
            evidence_snapshot_bytes=1,
            evidence_item_count=1,
            historical_memory_episode_count=0,
        )


def test_resolution_manifest_rejects_duplicate_question() -> None:
    # Given
    resolution = make_resolution()
    duplicate = resolution.model_copy(
        update={"entries": (resolution.entries[0], resolution.entries[0])}
    )

    # When / Then
    with pytest.raises(ValidationError):
        type(resolution).model_validate_json(duplicate.model_dump_json())
