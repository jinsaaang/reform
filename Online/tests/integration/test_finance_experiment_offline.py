"""Real immutable seed DB with deterministic offline experiment providers."""

import hashlib
from dataclasses import replace
from pathlib import Path

from src.core.finance_seed_repository import FinanceSeedRepository
from src.services.finance_experiment import (
    FinanceExperimentExitCode,
    FinanceExperimentRunRequest,
    FinanceExperimentRunResult,
    run_finance_experiment,
)
from src.services.finance_experiment_artifact import load_finance_experiment_bundle
from src.services.finance_experiment_manifest import (
    load_finance_experiment_manifest,
)
from src.services.finance_seed_identity import FinanceSeedIdentityVerifier
from src.services.historical_dag_retriever import HistoricalDagRetriever
from tests.fixtures.finance_experiment_service import (
    OfflineExperimentState,
    make_offline_dependencies,
)

_ROOT = Path(__file__).resolve().parents[2]
_DB_PATH = _ROOT / "data/releases/worldreasoner/v1.0.0/worldreasoner_public.db"
_MANIFEST_PATH = _ROOT / "configs/experiments/finance_live_10_2026-07-18.json"


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _run_offline_bundle(
    destination: Path,
) -> tuple[FinanceExperimentRunResult, OfflineExperimentState]:
    state = OfflineExperimentState()
    repository = FinanceSeedRepository(_DB_PATH)
    dependencies = replace(
        make_offline_dependencies(state),
        identity_verifier=FinanceSeedIdentityVerifier(_DB_PATH),
        repository=repository,
        retriever=HistoricalDagRetriever(repository),
    )
    locked = load_finance_experiment_manifest(_MANIFEST_PATH)
    manifest = locked.model_copy(
        update={"questions": locked.questions[:1], "repetitions": 1}
    )
    result = run_finance_experiment(
        FinanceExperimentRunRequest(manifest, destination),
        dependencies,
    )
    return result, state


def write_offline_evidence_bundle(destination: Path) -> FinanceExperimentRunResult:
    """Write the same real-DB offline artifact used by the integration test."""
    result, _ = _run_offline_bundle(destination)
    return result


def test_real_immutable_db_runs_offline_without_sidecars_or_mutation(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "offline-bundle"
    before = (
        _sha256(_DB_PATH),
        _DB_PATH.stat().st_mode,
        tuple(_DB_PATH.parent.glob(f"{_DB_PATH.name}-*")),
    )

    result, state = _run_offline_bundle(destination)
    verified = load_finance_experiment_bundle(destination)

    assert result.exit_code is FinanceExperimentExitCode.SUCCESS
    assert result.recorded and result.receipt == verified.receipt
    assert len(verified.suite.trials) == 1
    assert len(state.search_requests) == 1
    assert len(state.forecast_inputs) == 3
    assert len(state.judge_payloads) == 18
    assert state.provider_build_count == [1]
    assert tuple(sorted(path.name for path in destination.iterdir())) == (
        "SHA256SUMS",
        "report.md",
        "suite.json",
    )
    after = (
        _sha256(_DB_PATH),
        _DB_PATH.stat().st_mode,
        tuple(_DB_PATH.parent.glob(f"{_DB_PATH.name}-*")),
    )
    assert after == before


__all__ = ["write_offline_evidence_bundle"]
