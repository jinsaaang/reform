"""CLI boundary tests for the finance experiment runner."""

from pathlib import Path

from typer.testing import CliRunner

from src.cli.main import app


def test_finance_experiment_help_exposes_run_analyze_and_backtest() -> None:
    """Given the finance group, experiment and backtest adapters are visible."""
    result = CliRunner().invoke(app, ["finance", "--help"])

    assert result.exit_code == 0
    assert "experiment-run" in result.stdout
    assert "experiment-analyze" in result.stdout
    assert "backtest-prepare" in result.stdout
    assert "backtest-analyze" in result.stdout


def test_experiment_run_help_requires_explicit_inputs() -> None:
    """Given the run adapter, help documents explicit filesystem boundaries."""
    result = CliRunner().invoke(app, ["finance", "experiment-run", "--help"])

    assert result.exit_code == 0
    assert all(
        flag in result.stdout
        for flag in ("--db", "--seed-manifest", "--experiment-manifest", "--output-dir")
    )


def test_experiment_run_json_is_machine_readable_on_missing_inputs(
    tmp_path: Path,
) -> None:
    """Given absent explicit inputs, the adapter emits one actionable failure."""
    result = CliRunner().invoke(
        app,
        [
            "finance",
            "experiment-run",
            "--db",
            str(tmp_path / "missing.db"),
            "--seed-manifest",
            str(tmp_path / "missing-seed.json"),
            "--experiment-manifest",
            str(tmp_path / "missing-experiment.json"),
            "--output-dir",
            str(tmp_path / "bundle"),
            "--json",
        ],
    )

    assert result.exit_code == 1
    assert "Error:" in result.output
    assert not (tmp_path / "bundle").exists()


def test_root_help_does_not_initialize_a_default_database(
    tmp_path: Path, monkeypatch
) -> None:
    """Given an isolated cwd, root help remains non-mutating."""
    monkeypatch.chdir(tmp_path)

    result = CliRunner().invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "finance" in result.stdout
    assert not (tmp_path / "worldreasoner.db").exists()
