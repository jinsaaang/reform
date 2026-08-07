#!/usr/bin/env python3
"""Validate the portable benchmark files required by canonical HGF."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "hgf" / "shared"))

from hgf.exemplar_selection import load_fixed_exemplar_bank
from hgf.memory_bank import load_hgf_blueprint_bank
from hgf.question_io import read_questions


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--selection-file", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _args()
    root = args.dataset_root.expanduser().resolve()
    questions_root = root / "data/questions"
    test_path = questions_root / "test_questions.jsonl"
    memory_path = questions_root / "memory_questions.jsonl"
    selection_path = (
        args.selection_file or questions_root / "selection.json"
    ).resolve()
    required = [test_path, memory_path, selection_path]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing required files: {missing}")

    tests = {str(row.id): row for row in read_questions(test_path)}
    memories = {str(row.id): row for row in read_questions(memory_path)}
    selected = [
        str(value)
        for value in json.loads(selection_path.read_text(encoding="utf-8"))[
            "question_ids"
        ]
    ]
    unknown = sorted(set(selected) - set(tests))
    if unknown:
        raise ValueError(f"selection contains unknown target IDs: {unknown}")

    blueprint_root = root / "artifacts/hgf/blueprints"
    exemplar_root = root / "artifacts/hgf/exemplars"
    blueprints = load_hgf_blueprint_bank(blueprint_root, expected_ids=set(memories))
    exemplars = load_fixed_exemplar_bank([exemplar_root])
    missing_exemplars = sorted(set(memories) - set(exemplars))
    if missing_exemplars:
        raise ValueError(f"missing answer-free exemplars: {missing_exemplars}")

    bad_databases: list[str] = []
    for question_id in selected:
        path = root / "data/evidence/e1" / f"{question_id}.sqlite"
        if not path.is_file():
            bad_databases.append(f"{question_id}: missing {path}")
            continue
        with sqlite3.connect(path) as connection:
            columns = {
                row[1] for row in connection.execute("PRAGMA table_info(articles)")
            }
        required_columns = {
            "id",
            "title",
            "source",
            "published_date",
            "content",
            "collected_for_question_id",
        }
        if not required_columns.issubset(columns):
            bad_databases.append(
                f"{question_id}: articles table lacks "
                f"{sorted(required_columns - columns)}"
            )
    if bad_databases:
        raise ValueError("invalid E1 evidence databases: " + "; ".join(bad_databases))

    print(
        json.dumps(
            {
                "status": "passed",
                "targets": len(tests),
                "selected_targets": len(selected),
                "historical_events": len(memories),
                "blueprints": len(blueprints),
                "answer_free_exemplars": len(exemplars),
                "e1_databases_checked": len(selected),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
