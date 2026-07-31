"""Offline validation for every published forecasting method."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .baselines import METHODS, _condition_evidence
from .forecast_core import _resolve_evidence
from .exemplar_selection import load_fixed_exemplar_bank
from .memory_bank import (
    load_factor_blueprint_bank,
    load_graph_bank,
    load_hgf_blueprint_bank,
)
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
    hgf_root = PACKAGE_ROOT / "artifacts" / "hgf"
    exemplar_root = hgf_root / "exemplars"
    exemplars = _load_source_cases(exemplar_root)
    exemplar_bank = load_fixed_exemplar_bank(
        [exemplar_root]
    )
    graphs = load_graph_bank(
        PACKAGE_ROOT / "data" / "memory_bank" / "manifest.json",
        memory_questions,
    )
    hgf_blueprints = load_hgf_blueprint_bank(
        hgf_root / "blueprints",
        expected_ids=set(memory_questions),
    )
    factor_blueprints = load_factor_blueprint_bank(
        PACKAGE_ROOT / "artifacts" / "baselines" / "factor_memory",
        expected_ids=set(memory_questions),
    )

    test_ids = [str(question.id) for question in test_questions]
    errors: list[str] = []
    if len(test_ids) != 100 or len(set(test_ids)) != 100:
        errors.append("test question set must contain 100 unique IDs")
    if len(memory_questions) != 200:
        errors.append("memory question set must contain 200 unique IDs")
    if len(graphs) != 200:
        errors.append("shared graph bank must contain 200 DAGs")
    if len(hgf_blueprints) != 200:
        errors.append("canonical HGF bank must contain 200 Blueprints")
    if len(factor_blueprints) != 200:
        errors.append("frozen Factor-Memory bank must contain 200 cards")
    if set(selection) != set(test_ids):
        errors.append("selection does not match the test question set")
    if set(exemplars) != set(test_ids):
        errors.append("fixed exemplar cases do not match test questions")
    if set(exemplar_bank) != set(memory_questions):
        errors.append(
            "complete exemplar bank does not match memory questions"
        )

    if any(
        payload.get("schema_version") != "hgf_blueprint_topology_v2"
        for payload in hgf_blueprints.values()
    ):
        errors.append("canonical HGF bank contains a non-HGF Blueprint")
    if any(
        payload.get("schema_version") == "hgf_blueprint_topology_v2"
        for payload in factor_blueprints.values()
    ):
        errors.append("Factor-Memory bank contains a canonical HGF Blueprint")
    blueprint_manifest = json.loads(
        (hgf_root / "blueprints" / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    aggregate = blueprint_manifest.get("aggregate_validation", {})
    for field, expected in (
        ("minimum_edge_coverage", 1.0),
        ("minimum_path_precision", 1.0),
        ("outcome_event_leak_count", 0),
        ("outcome_text_leak_count", 0),
        ("realized_value_count", 0),
        ("absolute_period_count", 0),
    ):
        if aggregate.get(field) != expected:
            errors.append(
                f"Blueprint aggregate validation {field} "
                f"is {aggregate.get(field)!r}, expected {expected!r}"
            )

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
        "shared_graphs": len(graphs),
        "hgf_blueprints": len(hgf_blueprints),
        "factor_memory_cards": len(factor_blueprints),
        "fixed_exemplars": len(exemplars),
        "unique_memory_exemplars": len(exemplar_bank),
        "blueprint_validation": aggregate,
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
