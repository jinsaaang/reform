from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_hgf_portable_launcher_builds_canonical_command(tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/run_hgf.py"),
            "--dataset-root",
            str(tmp_path / "benchmark"),
            "--model",
            "example/model",
            "--provider",
            "example-provider",
            "--output-dir",
            str(tmp_path / "run"),
            "--limit",
            "2",
            "--dry-run",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "hgf_e2e_topology_provider_pinned.run" in completed.stdout
    assert "--blueprint-root" in completed.stdout
    assert "--exemplar-root" in completed.stdout
    assert not (tmp_path / "run").exists()


def test_hgf_launcher_rejects_partially_frozen_inputs(tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/run_hgf.py"),
            "--dataset-root",
            str(tmp_path / "benchmark"),
            "--model",
            "example/model",
            "--provider",
            "example-provider",
            "--output-dir",
            str(tmp_path / "run"),
            "--evidence-selection-manifest",
            str(tmp_path / "evidence.json"),
            "--dry-run",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "requires both" in completed.stderr


def test_public_baseline_choices_exclude_obsolete_hgf() -> None:
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts/run_baselines.py"), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "procedural_topology_hgf" not in completed.stdout
    assert "{search_only,prospective_dag,direct_dag,factor_memory,case_memory,text_memory}" in completed.stdout

