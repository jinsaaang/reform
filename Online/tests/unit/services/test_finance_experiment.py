"""Todo 6 batch-service and deterministic seed contracts."""

from dataclasses import replace
from pathlib import Path

import pytest

from src.domain.finance import experiment_artifact as artifact_models
from src.domain.finance.experiment import FinanceSuiteStatus
from src.domain.finance.experiment_artifact import PersistedPairCompleted
from src.domain.finance.experiment_metrics import ForecastPair
from src.domain.finance.experiment_telemetry import ProviderSeedEffective
from src.domain.finance.memory import QuestionId
from src.services.finance_experiment import (
    FinanceExperimentExitCode,
    FinanceExperimentRunRequest,
    FinanceTrialSeedInputs,
    derive_forecast_seed,
    derive_judge_seed,
    derive_panel_seed,
    run_finance_experiment,
)
from src.services.finance_experiment_artifact import load_finance_experiment_bundle
from src.services.finance_experiment_manifest import (
    load_finance_experiment_manifest,
)
from src.services import finance_hashed_bundle
from tests.fixtures.finance_experiment_manifest import (
    EXPECTED_FINANCE_QUESTION_ROWS,
)
from tests.fixtures.finance_experiment_runner import FailingIdentityVerifier
from tests.fixtures.finance_experiment_service import (
    OfflineExperimentState,
    OfflineProviderBuilder,
    make_offline_dependencies,
)
from tests.unit.domain.finance._experiment_factories import make_manifest

_MANIFEST_PATH = (
    Path(__file__).resolve().parents[3]
    / "configs/experiments/finance_live_10_2026-07-18.json"
)


def test_locked_seed_vectors_match_the_preregistered_derivation() -> None:
    inputs = FinanceTrialSeedInputs(
        root_seed=20_260_718,
        question_id=QuestionId("fin-fed-cut-2026"),
        repetition_index=0,
    )

    forecast_seed = derive_forecast_seed(inputs)
    panel_seed = derive_panel_seed(inputs, ForecastPair.A_B)
    judge_seed = derive_judge_seed(inputs, ForecastPair.A_B, "judge-1")

    assert forecast_seed == 2_726_063_054_686_746_225
    assert panel_seed == 8_601_553_002_256_258_062
    assert judge_seed == 4_316_865_573_165_661_247


def test_checked_in_manifest_matches_all_locked_question_rows() -> None:
    manifest = load_finance_experiment_manifest(_MANIFEST_PATH)

    actual_rows = tuple(
        (
            str(question.question_id),
            question.question_text,
            question.context[0],
            question.resolution_rule,
        )
        for question in manifest.questions
    )

    assert actual_rows == EXPECTED_FINANCE_QUESTION_ROWS
    assert manifest.cutoff.isoformat() == "2026-07-18T00:00:00+00:00"
    assert all(
        question.context == (row[2],)
        for question, row in zip(
            manifest.questions,
            EXPECTED_FINANCE_QUESTION_ROWS,
            strict=True,
        )
    )
    assert all(str(question.positive_label) == "Yes" for question in manifest.questions)
    serialized = manifest.model_dump_json()
    assert "realized_outcome" not in serialized
    assert "current_target_outcome" not in serialized


def test_two_questions_two_repetitions_write_one_complete_bundle(
    tmp_path: Path,
) -> None:
    state = OfflineExperimentState()
    dependencies = make_offline_dependencies(state)
    manifest = make_manifest().model_copy(update={"repetitions": 2})
    destination = tmp_path / "complete"

    result = run_finance_experiment(
        FinanceExperimentRunRequest(manifest, destination),
        dependencies,
    )
    suite = load_finance_experiment_bundle(destination).suite

    assert result.exit_code is FinanceExperimentExitCode.SUCCESS
    assert result.recorded and result.suite_status is FinanceSuiteStatus.COMPLETE
    assert len(suite.trials) == 4
    assert sum(len(trial.arms) for trial in suite.trials) == 12
    assert sum(len(trial.pairs) for trial in suite.trials) == 12
    assert len(state.search_requests) == 4
    assert len(state.forecast_inputs) == 12
    assert len(state.judge_payloads) == 72
    assert len(state.forecast_settings) == 4
    assert len(state.judge_settings) == 36
    assert all(
        settings.model_copy(update={"requested_seed": None}) == manifest.forecast
        for settings in state.forecast_settings
    )
    assert all(
        any(
            settings.model_copy(update={"requested_seed": None}) == member.settings
            for member in manifest.judges
        )
        for settings in state.judge_settings
    )
    for trial in suite.trials:
        forecast_seeds = tuple(
            exchange.attempt.seed for arm in trial.arms for exchange in arm.exchanges
        )
        assert len(forecast_seeds) == 3
        assert all(isinstance(seed, ProviderSeedEffective) for seed in forecast_seeds)
        effective_forecast_seeds = tuple(
            seed for seed in forecast_seeds if isinstance(seed, ProviderSeedEffective)
        )
        assert len(effective_forecast_seeds) == 3
        assert len({seed.requested_seed for seed in effective_forecast_seeds}) == 1
        for pair in trial.pairs:
            assert isinstance(pair, PersistedPairCompleted)
            seeds = tuple(exchange.attempt.seed for exchange in pair.judge_exchanges)
            assert all(isinstance(seed, ProviderSeedEffective) for seed in seeds)
            effective_judge_seeds = tuple(
                seed for seed in seeds if isinstance(seed, ProviderSeedEffective)
            )
            assert len(effective_judge_seeds) == 6
            assert all(
                effective_judge_seeds[index].requested_seed
                == effective_judge_seeds[index + 1].requested_seed
                for index in range(0, 6, 2)
            )


def test_identity_preflight_failure_constructs_no_provider_and_writes_no_bundle(
    tmp_path: Path,
) -> None:
    state = OfflineExperimentState()
    dependencies = replace(
        make_offline_dependencies(state),
        identity_verifier=FailingIdentityVerifier(),
    )
    destination = tmp_path / "rejected"

    result = run_finance_experiment(
        FinanceExperimentRunRequest(make_manifest(), destination),
        dependencies,
    )

    assert result.exit_code is FinanceExperimentExitCode.PREFLIGHT_FAILED
    assert not result.recorded and result.suite_status is None
    assert state.provider_build_count == []
    assert not destination.exists()


def test_invalid_manifest_copy_constructs_no_provider_and_writes_no_bundle(
    tmp_path: Path,
) -> None:
    state = OfflineExperimentState()
    invalid = make_manifest().model_copy(update={"arm_order": ()})
    destination = tmp_path / "invalid-manifest"

    result = run_finance_experiment(
        FinanceExperimentRunRequest(invalid, destination),
        make_offline_dependencies(state),
    )

    assert result.exit_code is FinanceExperimentExitCode.PREFLIGHT_FAILED
    assert state.provider_build_count == []
    assert state.search_requests == []
    assert not destination.exists()


def test_empty_preparation_persists_direct_and_explicit_partial_terminals(
    tmp_path: Path,
) -> None:
    state = OfflineExperimentState()
    dependencies = make_offline_dependencies(state)
    dependencies = replace(
        dependencies,
        provider_builder=OfflineProviderBuilder(
            state,
            search_response='{"candidate_payloads":[]}',
        ),
    )
    destination = tmp_path / "partial"

    result = run_finance_experiment(
        FinanceExperimentRunRequest(make_manifest(), destination),
        dependencies,
    )
    suite = load_finance_experiment_bundle(destination).suite

    assert result.exit_code is FinanceExperimentExitCode.PARTIAL_RECORDED
    assert result.recorded and result.suite_status is FinanceSuiteStatus.PARTIAL
    assert all(trial.arms[0].status == "succeeded" for trial in suite.trials)
    assert all(
        arm.status == "unavailable" for trial in suite.trials for arm in trial.arms[1:]
    )
    assert all(
        pair.status == "unavailable" for trial in suite.trials for pair in trial.pairs
    )


def test_malformed_forecasts_persist_failed_terminals_and_nonzero_result(
    tmp_path: Path,
) -> None:
    state = OfflineExperimentState(malformed_forecasts=True)
    destination = tmp_path / "failed"

    result = run_finance_experiment(
        FinanceExperimentRunRequest(make_manifest(), destination),
        make_offline_dependencies(state),
    )
    suite = load_finance_experiment_bundle(destination).suite

    assert result.exit_code is FinanceExperimentExitCode.FAILED_RECORDED
    assert result.recorded and result.suite_status is FinanceSuiteStatus.FAILED
    assert all(
        isinstance(arm, artifact_models.PersistedArmFailed)
        for trial in suite.trials
        for arm in trial.arms
    )
    assert state.judge_payloads == []


def test_interrupted_bundle_write_cleans_temporary_and_records_nothing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = OfflineExperimentState()
    destination = tmp_path / "interrupted"

    def interrupt_write(path: Path, content: bytes) -> None:
        del path, content
        raise KeyboardInterrupt

    monkeypatch.setattr(finance_hashed_bundle, "_write_fsync", interrupt_write)

    result = run_finance_experiment(
        FinanceExperimentRunRequest(make_manifest(), destination),
        make_offline_dependencies(state),
    )

    assert result.exit_code is FinanceExperimentExitCode.PERSISTENCE_FAILED
    assert not result.recorded and result.destination is None
    assert not destination.exists()
    assert tuple(tmp_path.glob(".interrupted.tmp-*")) == ()
