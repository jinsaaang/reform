from __future__ import annotations

from pathlib import Path

from hgf.manifest import _included_files
from hgf.verify import source_dependency_violations


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src" / "hgf"


def test_public_runtime_has_no_external_runner_imports() -> None:
    for path in SOURCE.glob("*.py"):
        assert source_dependency_violations(path) == []


def test_public_runtime_uses_fixed_exemplars() -> None:
    runner = (SOURCE / "runner.py").read_text(encoding="utf-8")
    assert 'exemplar_case["retrieved_memory_question_id"]' in runner
    assert 'exemplar_case["worked_exemplar"]' in runner
    assert "_distill_exemplar(" not in runner


def test_main_table_hgf_uses_the_same_fixed_contract() -> None:
    baselines = (SOURCE / "baselines.py").read_text(encoding="utf-8")
    assert 'fixed_case["retrieved_memory_question_id"]' in baselines
    assert 'fixed_case["worked_exemplar"]' in baselines
    assert "_distill_exemplar(" not in baselines
    assert 'seed_role="boundary_mapping"' in baselines


def test_only_one_readme_is_published() -> None:
    readmes = [path for path in _included_files() if path.name == "README.md"]
    assert readmes == [ROOT / "README.md"]
