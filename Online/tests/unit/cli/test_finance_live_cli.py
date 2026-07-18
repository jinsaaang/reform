"""Focused offline tests for the current-only finance live command."""

import json
from pathlib import Path

from typer.testing import CliRunner

import src.cli.commands.finance as finance
from src.cli.finance_offline import (
    OfflineFixtureForecastProvider,
    OfflineFixtureSearchProvider,
)
from src.cli.main import app
from src.services.finance_reasoning_artifact import load_finance_reasoning_artifact

_DB = Path("data/releases/worldreasoner/v1.0.0/worldreasoner_public.db")
_MANIFEST = Path("docs/research/finance_seed_v1_manifest.json")


def test_pipeline_run_wires_live_policy_with_fake_providers(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Given valid immutable inputs, current live wiring reaches both providers."""
    monkeypatch.setattr(
        finance,
        "LiveSearchProvider",
        lambda result_limit: OfflineFixtureSearchProvider(),
    )
    monkeypatch.setattr(
        finance,
        "LiveForecastProvider",
        lambda: OfflineFixtureForecastProvider(),
    )

    result = CliRunner().invoke(
        app,
        [
            "finance",
            "pipeline-run",
            "--db",
            str(_DB),
            "--manifest",
            str(_MANIFEST),
            "--question",
            "Will NVIDIA revenue exceed analyst expectations?",
            "--cutoff",
            "2026-06-01T00:00:00+00:00",
            "--context",
            "GPU demand and semiconductor quarterly revenue",
            "--top-k",
            "3",
            "--search-result-limit",
            "2",
            "--artifact-dir",
            str(tmp_path),
            "--json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["status"] == "succeeded"
    assert payload["run_mode"] == "current_unresolved"
    assert payload["query_passes"][0]["source_policy"] == "live_search"

    artifact_paths = tuple(tmp_path.glob("*.json"))
    assert len(artifact_paths) == 1
    artifact = json.loads(artifact_paths[0].read_text(encoding="utf-8"))
    assert artifact["schema_version"] == "finance-reasoning-run/v1"
    assert artifact["metadata"]["arm"] == "search_dag"
    assert artifact["metadata"]["forecast_model"]
    assert artifact["metadata"]["reasoning_effort"] == "high"
    assert artifact["pipeline_result"] == payload
    restored = load_finance_reasoning_artifact(artifact_paths[0])
    assert restored.pipeline_result.model_dump(mode="json") == payload

    scenario = artifact["pipeline_result"]["forecast_result"]["scenarios"][0]
    assert scenario["reasoning_steps"]
    assert scenario["assumptions"]
    assert scenario["triggers"]
    assert scenario["disconfirmers"]
    assert scenario["uncertainty"]
    assert scenario["evidence_ids"]
    assert scenario["historical_dag_references"]

    serialized_artifact = artifact_paths[0].read_text(encoding="utf-8")
    assert "ground_truth" not in serialized_artifact
    assert "current_dag" not in serialized_artifact
    assert "exact_body" not in serialized_artifact


def test_pipeline_run_rejects_historical_before_provider_construction(
    monkeypatch,
) -> None:
    """Given historical mode, live CLI fails before any provider can run."""
    calls: list[str] = []
    monkeypatch.setattr(
        finance,
        "LiveSearchProvider",
        lambda result_limit: calls.append("search") or OfflineFixtureSearchProvider(),
    )

    result = CliRunner().invoke(
        app,
        [
            "finance",
            "pipeline-run",
            "--db",
            str(_DB),
            "--manifest",
            str(_MANIFEST),
            "--question",
            "Will NVIDIA revenue exceed analyst expectations?",
            "--cutoff",
            "2026-06-01T00:00:00+00:00",
            "--context",
            "GPU demand and semiconductor quarterly revenue",
            "--mode",
            "historical",
        ],
    )

    assert result.exit_code == 1
    assert "current mode only" in result.output
    assert calls == []
