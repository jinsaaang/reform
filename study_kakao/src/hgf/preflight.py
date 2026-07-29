"""Offline validation for every published forecasting method."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .baselines import METHODS, _condition_evidence
from .forecast_core import _resolve_evidence
from .memory_bank import load_final_memory_bank
from .package import PACKAGE_ROOT
from .question_io import read_questions, resolve_forecast_cutoff
from .runner import _load_source_cases


def run_preflight(*, validate_evidence: bool = True) -> dict[str, Any]:
    questions_dir = PACKAGE_ROOT / "data" / "questions"
    evidence_dir = PACKAGE_ROOT / "data" / "evidence"
    test_questions = read_questions(questions_dir / "test_questions.jsonl")
    memory_questions_list = read_questions(
        questions_dir / "memory_questions.jsonl"
    )
    memory_questions = {
        str(question.id): question for question in memory_questions_list
    }
    selection = json.loads(
        (questions_dir / "selection.json").read_text(encoding="utf-8")
    )["question_ids"]
    exemplars = _load_source_cases(PACKAGE_ROOT / "artifacts" / "exemplars")
    _, blueprints = load_final_memory_bank(
        PACKAGE_ROOT / "data" / "memory_bank" / "manifest.json",
        memory_questions,
    )

    test_ids = [str(question.id) for question in test_questions]
    errors: list[str] = []
    if len(test_ids) != 100 or len(set(test_ids)) != 100:
        errors.append("test question set must contain 100 unique IDs")
    if len(memory_questions) != 200:
        errors.append("memory question set must contain 200 unique IDs")
    if len(blueprints) != 200:
        errors.append("memory bank must contain 200 blueprints")
    if set(selection) != set(test_ids):
        errors.append("selection does not match the test question set")
    if set(exemplars) != set(test_ids):
        errors.append("fixed exemplar cases do not match test questions")

    memory_ids = {
        str(case["retrieved_memory_question_id"])
        for case in exemplars.values()
    }
    cache_ids = {
        path.stem
        for path in (
            PACKAGE_ROOT / "artifacts" / "semantic_lessons"
        ).glob("*.json")
    }
    if cache_ids != memory_ids:
        errors.append("semantic lesson cache does not match exemplar memories")

    e0_count = len(list((evidence_dir / "e0").glob("*.sqlite")))
    e1_count = len(list((evidence_dir / "e1").glob("*.sqlite")))
    if e0_count != 100 or e1_count != 100:
        errors.append(
            f"expected 100 E0 and 100 E1 databases, got {e0_count}/{e1_count}"
        )

    evidence_rows = {"e0": 0, "e1": 0}
    if validate_evidence:
        for question in test_questions:
            cutoff, _ = resolve_forecast_cutoff(question)
            _, e0 = _condition_evidence(
                evidence_dir,
                question,
                cutoff,
                guided=False,
                limit=80,
            )
            _, e1 = _resolve_evidence(
                evidence_dir,
                question,
                cutoff,
                80,
            )
            if not e0:
                errors.append(f"E0 evidence is empty for {question.id}")
            if not e1:
                errors.append(f"E1 evidence is empty for {question.id}")
            evidence_rows["e0"] += len(e0)
            evidence_rows["e1"] += len(e1)

    return {
        "status": "pass" if not errors else "fail",
        "errors": errors,
        "methods": list(METHODS),
        "method_count": len(METHODS),
        "test_questions": len(test_ids),
        "memory_questions": len(memory_questions),
        "memory_blueprints": len(blueprints),
        "fixed_exemplars": len(exemplars),
        "semantic_lessons": len(cache_ids),
        "e0_databases": e0_count,
        "e1_databases": e1_count,
        "evidence_rows": evidence_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-evidence-content",
        action="store_true",
        help="Check evidence coverage without reading every database.",
    )
    args = parser.parse_args()
    report = run_preflight(
        validate_evidence=not args.skip_evidence_content
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
