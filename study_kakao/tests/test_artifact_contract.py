from __future__ import annotations

import json
import sys
from pathlib import Path

from hgf import manifest as artifact_manifest
from hgf.baselines import METHODS, _parse_args
from hgf.runner import _load_source_cases


ROOT = Path(__file__).resolve().parents[1]


def test_artifact_manifest_prunes_local_runtime_directories(
    tmp_path,
    monkeypatch,
) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "keep.py").write_text("kept", encoding="utf-8")
    for excluded in ("runs", ".venv", "__pycache__"):
        directory = tmp_path / excluded
        directory.mkdir()
        (directory / "ignored.txt").write_text("ignored", encoding="utf-8")
    (tmp_path / "data.sqlite-wal").write_text("transient", encoding="utf-8")
    (tmp_path / "data.sqlite-shm").write_text("transient", encoding="utf-8")
    monkeypatch.setattr(artifact_manifest, "PACKAGE_ROOT", tmp_path)
    included = {
        path.relative_to(tmp_path).as_posix()
        for path in artifact_manifest._included_files()
    }
    assert included == {"src/keep.py"}


def test_artifact_manifest_normalizes_text_line_endings(tmp_path) -> None:
    text_path = tmp_path / "payload.json"
    text_path.write_bytes(b"{\r\n  \"value\": 1\r\n}\r\n")
    windows_record = artifact_manifest._file_record(text_path)
    text_path.write_bytes(b"{\n  \"value\": 1\n}\n")
    assert artifact_manifest._file_record(text_path) == windows_record

    database_path = tmp_path / "payload.sqlite"
    database_path.write_bytes(b"binary\r\n")
    binary_windows_record = artifact_manifest._file_record(database_path)
    database_path.write_bytes(b"binary\n")
    assert artifact_manifest._file_record(database_path) != binary_windows_record


def test_extension_verifier_uses_manifest_record() -> None:
    source = (
        ROOT / "experiments" / "verify_experiment_extensions.py"
    ).read_text(encoding="utf-8")
    assert "actual = _file_record(path)" in source
    assert "path.stat().st_size" not in source


def test_fixed_exemplars_cover_all_100_questions() -> None:
    selected = json.loads(
        (ROOT / "data" / "questions" / "selection.json").read_text(
            encoding="utf-8"
        )
    )["question_ids"]
    cases = _load_source_cases(ROOT / "artifacts" / "hgf" / "exemplars")
    assert len(selected) == len(set(selected)) == 100
    assert set(cases) == set(selected)
    for case in cases.values():
        assert case["retrieved_memory_question_id"]
        assert case["worked_exemplar"]


def test_final_memory_bank_has_200_entries() -> None:
    manifest = json.loads(
        (ROOT / "data" / "memory_bank" / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["memory_question_count"] == 200
    assert manifest["test_question_count"] == 100
    assert manifest["total_validated_count"] == 200
    assert len(manifest["entries"]) == 200
    for entry in manifest["entries"]:
        assert "guidance_path" not in entry
        assert (ROOT / entry["graph_path"]).is_file()
        for key in ("audit_path", "evidence_path"):
            if entry.get(key):
                assert (ROOT / entry[key]).is_file()


def test_reproduction_defaults_match_public_layout(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["hgf-replay"])
    args = _parse_args(
        default_methods=("hgf",),
        default_output_dir=Path("runs/hgf"),
    )
    assert args.questions_dir == Path("data/questions")
    assert args.evidence_dir == Path("data/evidence")
    assert args.memory_bank_manifest == Path(
        "data/memory_bank/manifest.json"
    )
    assert args.selection_file == Path("data/questions/selection.json")
    assert args.hgf_artifact_root == Path("artifacts/hgf")
    assert not hasattr(args, "exemplar_dir")
    assert not hasattr(args, "semantic_cache_dir")
    assert args.output_dir == Path("runs/hgf")
    assert args.methods == ("hgf",)
    assert args.limit == 100
    assert args.workers == 4


def test_all_paper_methods_are_registered() -> None:
    assert METHODS == (
        "search_only",
        "factor_memory",
        "case_memory",
        "text_memory",
        "direct_dag",
        "prospective_dag",
        "hgf",
    )
