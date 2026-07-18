"""Schema cases plus re-exported atomic bundle cases for Todo 5 acceptance."""

from decimal import Decimal
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.domain.finance.experiment import (
    ArmFailureReason,
    ArmUnavailableReason,
    FinanceSuiteStatus,
)
from src.domain.finance.experiment_artifact import (
    PairUnavailableReason,
    PersistedArmFailed,
    PersistedArmSucceeded,
    PersistedArmUnavailable,
    PersistedExperimentTrial,
    PersistedFinanceExperimentSuite,
    PersistedPairCompleted,
    PersistedPairUnavailable,
)
from src.domain.finance.experiment_metrics import (
    ExAnteTieReason,
    ForecastPair,
    JudgePanelMetrics,
    PanelPreference,
)
from src.services.finance_experiment_artifact import (
    load_finance_experiment_bundle,
    persist_finance_experiment_bundle,
)
from tests.fixtures.finance_experiment_artifact import make_persisted_suite
from tests.unit.services.finance_experiment_artifact_bundle_cases import (
    test_atomic_bundle_round_trip_preserves_reasoning_hashes_and_report as test_atomic_bundle_round_trip_preserves_reasoning_hashes_and_report,
    test_bundle_writer_refuses_an_existing_destination as test_bundle_writer_refuses_an_existing_destination,
    test_interrupted_bundle_write_cleans_every_partial_directory as test_interrupted_bundle_write_cleans_every_partial_directory,
    test_loader_rejects_one_byte_tamper as test_loader_rejects_one_byte_tamper,
    test_parent_fsync_failure_rolls_back_published_directory as test_parent_fsync_failure_rolls_back_published_directory,
    test_persisted_fixture_recursively_rejects_forbidden_keys_and_values as test_persisted_fixture_recursively_rejects_forbidden_keys_and_values,
    test_persistence_registry_rejects_exact_transient_body_and_cleans_temp as test_persistence_registry_rejects_exact_transient_body_and_cleans_temp,
    test_report_is_byte_stable_across_two_new_destinations as test_report_is_byte_stable_across_two_new_destinations,
    test_strict_suite_schema_rejects_camel_case_secret_field as test_strict_suite_schema_rejects_camel_case_secret_field,
    test_unwritable_destination_fails_without_partial_residue as test_unwritable_destination_fails_without_partial_residue,
)
from tests.unit.services.finance_experiment_artifact_publish_cases import (
    test_atomic_publication_system_error_fails_closed_without_residue as test_atomic_publication_system_error_fails_closed_without_residue,
    test_racing_empty_destination_is_preserved_without_publication as test_racing_empty_destination_is_preserved_without_publication,
    test_unsupported_atomic_publication_fails_closed_without_residue as test_unsupported_atomic_publication_fails_closed_without_residue,
)


def test_partial_suite_persists_unavailable_arm_and_pair_terminals(
    tmp_path: Path,
) -> None:
    source = make_persisted_suite()
    trials: list[PersistedExperimentTrial] = []
    for trial in source.trials:
        search_dag = trial.arms[2]
        unavailable_arm = PersistedArmUnavailable(
            arm=search_dag.arm,
            exchanges=(),
            treatment=search_dag.treatment,
            reason=ArmUnavailableReason.MEMORY_UNAVAILABLE,
        )
        unavailable_pairs = (
            trial.pairs[0],
            PersistedPairUnavailable(
                pair=ForecastPair.B_C,
                reason=PairUnavailableReason.ARM_UNAVAILABLE,
            ),
            PersistedPairUnavailable(
                pair=ForecastPair.A_C,
                reason=PairUnavailableReason.ARM_UNAVAILABLE,
            ),
        )
        trials.append(
            PersistedExperimentTrial(
                trial_id=trial.trial_id,
                question_id=trial.question_id,
                repetition_index=trial.repetition_index,
                preparation_exchanges=trial.preparation_exchanges,
                arms=(trial.arms[0], trial.arms[1], unavailable_arm),
                pairs=unavailable_pairs,
            )
        )
    partial = PersistedFinanceExperimentSuite(
        suite_id=source.suite_id,
        created_at=source.created_at,
        status=FinanceSuiteStatus.PARTIAL,
        manifest=source.manifest,
        trials=tuple(trials),
    )
    destination = tmp_path / "partial"

    persist_finance_experiment_bundle(destination, partial)
    loaded = load_finance_experiment_bundle(destination).suite

    assert loaded.status is FinanceSuiteStatus.PARTIAL
    assert loaded.trials[0].arms[2].status == "unavailable"
    assert loaded.trials[0].pairs[1].status == "unavailable"


def test_attempted_no_quorum_panel_remains_complete_and_counted() -> None:
    source = make_persisted_suite()
    first_trial = source.trials[0]
    first_pair = first_trial.pairs[0]
    assert first_pair.status == "completed"
    no_quorum = JudgePanelMetrics(
        overall_preference=PanelPreference.TIE,
        preference_eligible=False,
        tie_reason=ExAnteTieReason.NO_QUORUM,
        first_votes=1,
        second_votes=0,
        tie_votes=0,
        inconsistent_count=0,
        invalid_count=2,
        evaluable_count=1,
        two_parse_valid_count=1,
        attempted_call_count=6,
        invalid_rate=Decimal(2) / Decimal(3),
        inconsistent_rate=Decimal(0),
        agreement=Decimal(1),
        order_consistency=Decimal(1),
    )
    replaced_pair = first_pair.model_copy(update={"panel": no_quorum})
    replaced_trial = PersistedExperimentTrial(
        trial_id=first_trial.trial_id,
        question_id=first_trial.question_id,
        repetition_index=first_trial.repetition_index,
        preparation_exchanges=first_trial.preparation_exchanges,
        arms=first_trial.arms,
        pairs=(replaced_pair, *first_trial.pairs[1:]),
    )

    suite = PersistedFinanceExperimentSuite(
        suite_id=source.suite_id,
        created_at=source.created_at,
        status=FinanceSuiteStatus.COMPLETE,
        manifest=source.manifest,
        trials=(replaced_trial, *source.trials[1:]),
    )

    assert suite.status is FinanceSuiteStatus.COMPLETE
    observed = suite.trials[0].pairs[0]
    assert isinstance(observed, PersistedPairCompleted)
    assert observed.panel.tie_reason is ExAnteTieReason.NO_QUORUM


def test_suite_rejects_drift_between_probability_and_sanitized_reasoning() -> None:
    source = make_persisted_suite()
    first_trial = source.trials[0]
    direct = first_trial.arms[0]
    assert isinstance(direct, PersistedArmSucceeded)
    candidate = direct.reasoning.candidate.model_copy(
        update={
            "outcome_probabilities": (
                direct.reasoning.candidate.outcome_probabilities[0].model_copy(
                    update={"probability": 0.9}
                ),
                direct.reasoning.candidate.outcome_probabilities[1].model_copy(
                    update={"probability": 0.1}
                ),
            )
        }
    )
    drifted = direct.model_copy(
        update={
            "reasoning": direct.reasoning.model_copy(update={"candidate": candidate})
        }
    )
    trial = PersistedExperimentTrial(
        trial_id=first_trial.trial_id,
        question_id=first_trial.question_id,
        repetition_index=first_trial.repetition_index,
        preparation_exchanges=first_trial.preparation_exchanges,
        arms=(drifted, *first_trial.arms[1:]),
        pairs=first_trial.pairs,
    )

    with pytest.raises(ValidationError):
        PersistedFinanceExperimentSuite(
            suite_id=source.suite_id,
            created_at=source.created_at,
            status=source.status,
            manifest=source.manifest,
            trials=(trial, *source.trials[1:]),
        )

    assert direct.positive_probability == Decimal("0.6")


@pytest.mark.parametrize("terminal", ("unavailable", "failed"))
def test_non_success_arm_rejects_treatment_from_another_arm(terminal: str) -> None:
    source = make_persisted_suite().trials[0]
    direct = source.arms[0]
    search_dag = source.arms[2]
    assert isinstance(direct, PersistedArmSucceeded)
    assert isinstance(search_dag, PersistedArmSucceeded)

    with pytest.raises(ValidationError):
        if terminal == "unavailable":
            PersistedArmUnavailable(
                arm=search_dag.arm,
                exchanges=(),
                treatment=direct.treatment,
                reason=ArmUnavailableReason.MEMORY_UNAVAILABLE,
            )
        else:
            PersistedArmFailed(
                arm=search_dag.arm,
                exchanges=search_dag.exchanges,
                treatment=direct.treatment,
                reason=ArmFailureReason.PROVIDER_ERROR,
            )

    assert direct.arm is not search_dag.arm


def test_trial_rejects_non_search_preparation_and_snapshot_drift() -> None:
    source = make_persisted_suite().trials[0]
    search_only = source.arms[1]
    assert isinstance(search_only, PersistedArmSucceeded)
    drifted = search_only.model_copy(
        update={
            "treatment": search_only.treatment.model_copy(
                update={"evidence_snapshot_digest": "f" * 64}
            )
        }
    )

    with pytest.raises(ValidationError):
        PersistedExperimentTrial(
            trial_id=source.trial_id,
            question_id=source.question_id,
            repetition_index=source.repetition_index,
            preparation_exchanges=(source.arms[0].exchanges[0],),
            arms=(source.arms[0], drifted, source.arms[2]),
            pairs=source.pairs,
        )

    assert drifted.treatment.evidence_snapshot_digest == "f" * 64


def test_unavailable_pair_rejects_forecast_exchange_digest() -> None:
    exchange = make_persisted_suite().trials[0].arms[0].exchanges[0]

    with pytest.raises(ValidationError):
        PersistedPairUnavailable(
            pair=ForecastPair.A_B,
            reason=PairUnavailableReason.PANEL_UNAVAILABLE,
            judge_exchanges=(exchange,),
        )

    assert exchange.kind.value == "forecast"
