"""Leakage-boundary smoke test for the resolved finance backtest."""

from dataclasses import replace
from pathlib import Path

from src.core.finance_seed_repository import FinanceSeedRepository
from src.domain.finance.experiment import FinanceExperimentManifest
from src.integrations.finance_public_db_search import PublicDbMetadataSearchProvider
from src.services.finance_backtest import (
    analyze_finance_backtest,
    build_finance_backtest_manifest,
)
from src.services.finance_experiment import (
    FinanceExperimentProviders,
    FinanceExperimentRunRequest,
    run_finance_experiment,
)
from src.services.finance_experiment_manifest import (
    load_finance_experiment_manifest,
)
from src.services.finance_resolution_analysis import (
    load_finance_resolution_analysis_bundle,
)
from src.services.finance_seed_identity import FinanceSeedIdentityVerifier
from src.services.historical_dag_retriever import HistoricalDagRetriever
from tests.fixtures.finance_experiment_service import (
    OfflineExperimentState,
    OfflineForecastFactory,
    OfflineJudgeFactory,
    make_offline_dependencies,
)

_ROOT = Path(__file__).resolve().parents[2]
_DB_PATH = _ROOT / "data/releases/worldreasoner/v1.0.0/worldreasoner_public.db"
_TEMPLATE_PATH = _ROOT / "configs/experiments/finance_live_10_2026-07-18.json"


def test_resolved_backtest_hides_outcomes_until_post_run_analysis(
    tmp_path: Path,
) -> None:
    template = load_finance_experiment_manifest(_TEMPLATE_PATH)
    manifest = build_finance_backtest_manifest(
        _DB_PATH,
        template,
        "resolved-backtest-smoke",
        limit=1,
    )
    serialized_manifest = manifest.model_dump_json()
    assert "ground_truth" not in serialized_manifest
    assert "realized_outcome" not in serialized_manifest

    state = OfflineExperimentState()
    repository = FinanceSeedRepository(_DB_PATH)

    def provider_builder(
        current_manifest: FinanceExperimentManifest,
    ) -> FinanceExperimentProviders:
        return FinanceExperimentProviders(
            search_provider=PublicDbMetadataSearchProvider(
                _DB_PATH,
                current_manifest.search.result_limit,
            ),
            forecast_provider_factory=OfflineForecastFactory(state),
            judge_provider_factory=OfflineJudgeFactory(state),
        )

    dependencies = replace(
        make_offline_dependencies(state),
        provider_builder=provider_builder,
        identity_verifier=FinanceSeedIdentityVerifier(_DB_PATH),
        repository=repository,
        retriever=HistoricalDagRetriever(repository),
    )
    suite_dir = tmp_path / "suite"
    result = run_finance_experiment(
        FinanceExperimentRunRequest(manifest, suite_dir),
        dependencies,
    )

    assert result.recorded
    suite_json = (suite_dir / "suite.json").read_text(encoding="utf-8")
    assert "ground_truth" not in suite_json
    assert "realized_outcome" not in suite_json
    assert len(state.forecast_inputs) == 3
    assert len(state.judge_payloads) == 18
    assert len(state.forecast_inputs[1].evidence_pack.items) == 5
    assert all(
        item.available_at < manifest.questions[0].forecast_cutoff
        for item in state.forecast_inputs[1].evidence_pack.items
    )

    analysis_dir = tmp_path / "analysis"
    _ = analyze_finance_backtest(suite_dir, _DB_PATH, analysis_dir)
    analysis = load_finance_resolution_analysis_bundle(analysis_dir).analysis

    assert analysis.resolved_question_count == 1
    assert tuple(score.correct for score in analysis.trials[0].arm_scores) == (
        True,
        False,
        False,
    )
    aggregate = analysis.pair_aggregates[0]
    assert aggregate.macro_first_accuracy is not None
    assert aggregate.macro_second_accuracy is not None
