"""CLI boundary tests for separate finance resolution analysis."""

from pathlib import Path

from typer.testing import CliRunner

from src.cli.main import app


def test_experiment_analyze_help_requires_verified_suite_and_resolution() -> None:
    """Given the analysis adapter, its two source boundaries are explicit."""
    result = CliRunner().invoke(app, ["finance", "experiment-analyze", "--help"])

    assert result.exit_code == 0
    assert all(
        flag in result.stdout
        for flag in ("--suite", "--resolution-manifest", "--output-dir")
    )


def test_experiment_analyze_rejects_missing_source_before_output(
    tmp_path: Path,
) -> None:
    """Given missing source files, no derived bundle is created."""
    result = CliRunner().invoke(
        app,
        [
            "finance",
            "experiment-analyze",
            "--suite",
            str(tmp_path / "missing-suite"),
            "--resolution-manifest",
            str(tmp_path / "missing-resolution.json"),
            "--output-dir",
            str(tmp_path / "analysis"),
        ],
    )

    assert result.exit_code == 1
    assert "Error:" in result.output
    assert not (tmp_path / "analysis").exists()
