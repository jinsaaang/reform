from __future__ import annotations

import sys
from pathlib import Path

import pytest

from hgf.baselines import _parse_args as _parse_baseline_args
from hgf.memory_bank import (
    load_factor_blueprint_bank,
    load_hgf_blueprint_bank,
)
from hgf.package import PACKAGE_ROOT


def test_hgf_and_factor_memory_load_from_separate_artifact_banks() -> None:
    hgf_bank = load_hgf_blueprint_bank(
        PACKAGE_ROOT / "artifacts" / "hgf" / "blueprints"
    )
    factor_bank = load_factor_blueprint_bank(
        PACKAGE_ROOT / "artifacts" / "baselines" / "factor_memory"
    )

    assert set(hgf_bank) == set(factor_bank)
    assert len(hgf_bank) == 200
    assert all(
        payload["schema_version"] == "hgf_blueprint_topology_v2"
        for payload in hgf_bank.values()
    )
    assert all(
        payload.get("schema_version") != "hgf_blueprint_topology_v2"
        for payload in factor_bank.values()
    )


def test_baseline_cli_accepts_only_complete_hgf_artifact_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["hgf.baselines", "--hgf-artifact-root", str(tmp_path)],
    )

    args = _parse_baseline_args()

    assert args.hgf_artifact_root == tmp_path
    assert not hasattr(args, "blueprint_override_dir")
    assert not hasattr(args, "semantic_cache_dir")
    assert not hasattr(args, "hgf_safety_gates")


def test_baseline_cli_rejects_partial_blueprint_override(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["hgf.baselines", "--blueprint-override-dir", str(tmp_path)],
    )

    with pytest.raises(SystemExit):
        _parse_baseline_args()
