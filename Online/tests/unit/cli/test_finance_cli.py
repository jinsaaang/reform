"""Behavioral tests for the deterministic finance CLI surface."""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from src.cli.main import app

_DB = Path("data/releases/worldreasoner/v1.0.0/worldreasoner_public.db")
_MANIFEST = Path("docs/research/finance_seed_v1_manifest.json")
_QUESTION = "Will NVIDIA revenue exceed analyst expectations?"
_CUTOFF = "2026-06-01T00:00:00+00:00"
_CONTEXT = "GPU demand and semiconductor quarterly revenue"


def test_finance_group_is_registered() -> None:
    """Given the root CLI, finance help is an available command."""
    runner = CliRunner()

    result = runner.invoke(app, ["finance", "--help"])

    assert result.exit_code == 0
    assert "seed-audit" in result.stdout
    assert "pipeline-smoke" in result.stdout


def test_finance_commands_accept_explicit_paths_without_default_database(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given an isolated cwd, finance commands only use explicit paths."""
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "finance",
            "seed-audit",
            "--db",
            str(tmp_path / "missing.db"),
            "--manifest",
            str(tmp_path / "missing.json"),
            "--json",
        ],
    )

    assert result.exit_code != 0
    assert not (tmp_path / "worldreasoner.db").exists()


def test_finance_help_does_not_initialize_default_database(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given an isolated cwd, finance help remains side-effect free."""
    monkeypatch.chdir(tmp_path)

    result = CliRunner().invoke(app, ["finance", "--help"])

    assert result.exit_code == 0
    assert "seed-audit" in result.stdout
    assert not (tmp_path / "worldreasoner.db").exists()


def test_seed_audit_reports_pinned_counts_deterministically() -> None:
    """Given the pinned DB/manifest, audit JSON is stable and complete."""
    runner = CliRunner()
    args = [
        "finance",
        "seed-audit",
        "--db",
        str(_DB),
        "--manifest",
        str(_MANIFEST),
        "--json",
    ]

    first = runner.invoke(app, args)
    second = runner.invoke(app, args)

    assert first.exit_code == second.exit_code == 0
    assert first.stdout == second.stdout
    payload = json.loads(first.stdout)
    assert payload["canonical_selection"]["total_rows"] == 37
    assert payload["canonical_selection"]["type_counts"] == {
        "binary": 23,
        "mcq": 1,
        "quantity": 8,
        "timeframe": 5,
    }
    assert payload["asset"]["read_only"] is True
    assert payload["asset"]["sqlite_sidecars"] == []
    assert payload["manifest_sha256"] == (
        "8781718ed4bad08cabaecf820fe5b8ffe9b7d19986ad990b4f0fab359feda269"
    )
    assert payload["manifest_size_bytes"] == 30743


def test_seed_audit_rejects_tampered_manifest(tmp_path: Path) -> None:
    """Given a changed source digest, audit fails with actionable stderr."""
    manifest = tmp_path / "tampered.json"
    content = _MANIFEST.read_text(encoding="utf-8").replace(
        "94ffd8cca51906edec0b05f7e94e78de80d26f268f082521bded80d0aed06fab",
        "0" * 64,
        1,
    )
    manifest.write_text(content, encoding="utf-8")

    result = CliRunner().invoke(
        app,
        [
            "finance",
            "seed-audit",
            "--db",
            str(_DB),
            "--manifest",
            str(manifest),
            "--json",
        ],
    )

    assert result.exit_code == 1
    assert "source asset identity" in result.output


@pytest.mark.parametrize("mode", ["current", "historical"])
def test_pipeline_smoke_runs_real_two_pass_path_deterministically(mode: str) -> None:
    """Given a valid target, both modes expose the real pipeline trace."""
    args = [
        "finance",
        "pipeline-smoke",
        "--db",
        str(_DB),
        "--question",
        _QUESTION,
        "--cutoff",
        _CUTOFF,
        "--context",
        _CONTEXT,
        "--mode",
        mode,
        "--json",
    ]
    runner = CliRunner()

    first = runner.invoke(app, args)
    second = runner.invoke(app, args)

    assert first.exit_code == second.exit_code == 0
    assert first.stdout == second.stdout
    payload = json.loads(first.stdout)
    assert payload["run_mode"] in {"current_unresolved", "historical_backtest"}
    assert payload["stage_order"] == [
        "initial_search",
        "immutable_episode_load",
        "historical_retrieval",
        "guided_search",
        "evidence_admission",
        "forecaster",
    ]
    assert len(payload["selected_historical_dags"]) == 3
    assert len(payload["query_passes"]) == 2
    assert [item["decision"] for item in payload["evidence_audit"]].count(
        "admitted"
    ) == 2
    assert payload["forecast_result"]["outcome_probabilities"] == [
        {"label": "Yes", "probability": 0.5},
        {"label": "No", "probability": 0.5},
    ]
    raw = first.stdout
    assert all(
        marker not in raw
        for marker in ("ground_truth", "current_dag", "historical_outcome")
    )


def test_pipeline_smoke_rejects_invalid_cutoff_and_mode() -> None:
    """Given malformed mode or cutoff, CLI exits nonzero before providers run."""
    base = [
        "finance",
        "pipeline-smoke",
        "--db",
        str(_DB),
        "--question",
        _QUESTION,
        "--context",
        _CONTEXT,
        "--json",
    ]
    runner = CliRunner()

    invalid_cutoff = runner.invoke(app, [*base, "--cutoff", "not-a-date"])
    invalid_mode = runner.invoke(
        app,
        [*base, "--cutoff", _CUTOFF, "--mode", "live"],
    )

    assert invalid_cutoff.exit_code == invalid_mode.exit_code == 1
    assert "ISO-8601" in invalid_cutoff.output
    assert "current or historical" in invalid_mode.output


def test_pipeline_smoke_abstains_when_fixture_evidence_is_after_cutoff() -> None:
    """Given a cutoff before every offline body, the pipeline exits nonzero safely."""
    result = CliRunner().invoke(
        app,
        [
            "finance",
            "pipeline-smoke",
            "--db",
            str(_DB),
            "--question",
            _QUESTION,
            "--cutoff",
            "2024-01-01T00:00:00+00:00",
            "--context",
            _CONTEXT,
            "--json",
        ],
    )

    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["status"] == "abstained"
    assert payload["reason"] == "no_admitted_evidence"
    assert "ground_truth" not in result.stdout
