from __future__ import annotations

import json
import sys
from pathlib import Path

from hgf.baselines import METHODS
from hgf.runner import _parse_args, _load_source_cases


ROOT = Path(__file__).resolve().parents[1]


def test_fixed_exemplars_cover_all_100_questions() -> None:
    selected = json.loads(
        (ROOT / "data" / "questions" / "selection.json").read_text(
            encoding="utf-8"
        )
    )["question_ids"]
    cases = _load_source_cases(ROOT / "artifacts" / "exemplars")
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
        path_keys = ["graph_path"]
        if entry.get("guidance_path"):
            path_keys.append("guidance_path")
        else:
            path_keys.extend(("audit_path", "evidence_path"))
        assert all((ROOT / entry[key]).is_file() for key in path_keys)


def test_reproduction_defaults_match_public_layout(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["hgf-replay"])
    args = _parse_args()
    assert args.questions_dir == Path("data/questions")
    assert args.evidence_dir == Path("data/evidence")
    assert args.memory_bank_manifest == Path(
        "data/memory_bank/manifest.json"
    )
    assert args.selection_file == Path("data/questions/selection.json")
    assert args.exemplar_dir == Path("artifacts/exemplars")
    assert args.semantic_cache_dir == Path("artifacts/semantic_lessons")
    assert args.output_dir == Path("runs/hgf")
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
