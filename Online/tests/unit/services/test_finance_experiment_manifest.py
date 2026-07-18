"""Service-boundary tests for finance experiment and resolution manifests."""

from pathlib import Path
from uuid import UUID

import pytest

from src.domain.finance.experiment import (
    FinanceResolutionEntry,
    FinanceResolutionManifest,
)
from src.domain.finance.memory import OutcomeLabel, QuestionId
from src.services.finance_experiment_manifest import (
    FinanceManifestValidationError,
    ResolutionBindingError,
    ResolutionBindingReason,
    bind_resolution_manifest,
    load_finance_experiment_manifest,
    load_finance_resolution_manifest,
)
from tests.unit.domain.finance._experiment_factories import (
    make_manifest,
    make_resolution,
    make_suite,
)


def test_loader_reads_valid_manifest_twice_without_side_effects(tmp_path: Path) -> None:
    # Given
    manifest_path = tmp_path / "experiment.json"
    manifest_path.write_text(make_manifest().model_dump_json(), encoding="utf-8")

    # When
    first = load_finance_experiment_manifest(manifest_path)
    second = load_finance_experiment_manifest(manifest_path)

    # Then
    assert first == second
    assert tuple(tmp_path.iterdir()) == (manifest_path,)


def test_loader_rejects_secret_extra_without_writing_artifact(tmp_path: Path) -> None:
    # Given
    manifest_path = tmp_path / "invalid.json"
    payload = (
        make_manifest()
        .model_dump_json()
        .replace(
            '"manifest_id":',
            '"api_key":"forbidden-sentinel","manifest_id":',
            1,
        )
    )
    manifest_path.write_text(payload, encoding="utf-8")

    # When / Then
    with pytest.raises(FinanceManifestValidationError):
        load_finance_experiment_manifest(manifest_path)
    assert tuple(tmp_path.iterdir()) == (manifest_path,)


def test_resolution_loader_accepts_partial_outcome_set(tmp_path: Path) -> None:
    # Given
    resolution_path = tmp_path / "resolution.json"
    expected = make_resolution()
    resolution_path.write_text(expected.model_dump_json(), encoding="utf-8")

    # When
    actual = load_finance_resolution_manifest(resolution_path)

    # Then
    assert actual == expected
    assert len(actual.entries) == 1


def test_resolution_binding_accepts_exact_suite_identity_and_hash() -> None:
    # Given
    suite = make_suite()
    resolution = make_resolution()

    # When
    bound = bind_resolution_manifest(resolution, suite, "b" * 64)

    # Then
    assert bound is resolution


def _resolution_with_entry(entry: FinanceResolutionEntry) -> FinanceResolutionManifest:
    return make_resolution().model_copy(update={"entries": (entry,)})


def test_resolution_binding_rejects_unknown_question() -> None:
    # Given
    resolution = make_resolution()
    unknown = resolution.entries[0].model_copy(
        update={"question_id": QuestionId("unknown-question")}
    )

    # When / Then
    with pytest.raises(ResolutionBindingError) as captured:
        bind_resolution_manifest(
            _resolution_with_entry(unknown),
            make_suite(),
            "b" * 64,
        )
    assert captured.value.reason is ResolutionBindingReason.UNKNOWN_QUESTION


def test_resolution_binding_rejects_label_outside_target_space() -> None:
    # Given
    resolution = make_resolution()
    invalid_label = resolution.entries[0].model_copy(
        update={"outcome_label": OutcomeLabel("Maybe")}
    )

    # When / Then
    with pytest.raises(ResolutionBindingError) as captured:
        bind_resolution_manifest(
            _resolution_with_entry(invalid_label),
            make_suite(),
            "b" * 64,
        )
    assert captured.value.reason is ResolutionBindingReason.INVALID_OUTCOME_LABEL


def test_resolution_binding_rejects_suite_identity_mismatch() -> None:
    # Given
    resolution = make_resolution().model_copy(
        update={"suite_id": UUID("20000000-0000-0000-0000-000000000002")}
    )

    # When / Then
    with pytest.raises(ResolutionBindingError) as captured:
        bind_resolution_manifest(resolution, make_suite(), "b" * 64)
    assert captured.value.reason is ResolutionBindingReason.SUITE_ID_MISMATCH


def test_resolution_binding_rejects_verified_hash_mismatch() -> None:
    # Given
    resolution = make_resolution()

    # When / Then
    with pytest.raises(ResolutionBindingError) as captured:
        bind_resolution_manifest(resolution, make_suite(), "c" * 64)
    assert captured.value.reason is ResolutionBindingReason.SUITE_HASH_MISMATCH
