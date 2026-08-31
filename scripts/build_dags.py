"""Generate a fresh WorldReasoner DAG bank from resolved questions."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import run_fresh_pipeline as core


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--memory-questions", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model", default="google/gemini-2.5-flash")
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--question-id", action="append", default=[])
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--min-evidence-articles", type=int, default=10)
    parser.add_argument("--min-graph-events", type=int, default=8)
    parser.add_argument("--min-graph-depth", type=int, default=3)
    parser.add_argument("--search-query-budget", type=int, default=10)
    parser.add_argument("--max-evidence-rounds", type=int, default=3)
    parser.add_argument(
        "--search-provider",
        choices=("auto", "google_news", "gdelt", "ddgs", "smolagents"),
        default="auto",
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _args()
    output = args.output_root.expanduser().resolve()
    if output.exists() and not args.resume:
        raise FileExistsError(f"output root already exists: {output}")
    output.mkdir(parents=True, exist_ok=args.resume)
    rows = core._subset(
        core._read_jsonl(args.memory_questions.expanduser().resolve()),
        args.question_id,
        "memory question",
    )
    unresolved = [str(row["id"]) for row in rows if row.get("ground_truth") is None]
    if unresolved:
        raise ValueError("DAG inputs must be resolved: " + ", ".join(unresolved))
    core._write_jsonl(output / "data/questions/memory_questions.jsonl", rows)
    core._write_jsonl(output / "data/questions/test_questions.jsonl", [])
    core._write_json(
        output / "data/questions/selection.json",
        {"selection_rule": "empty until forecast", "question_ids": []},
    )
    core._write_json(
        output / "data/memory_bank/fixed_exemplar_selection.json",
        {
            "schema_version": "hgf_fixed_memory_selection_v1",
            "question_count": 0,
            "entries": [],
        },
    )
    core._write_json(
        output / "configs/reproduction.json",
        {
            "schema_version": "fresh_pipeline_root_v1",
            "implementation": "reform_artifact_generator",
        },
    )
    core._write_json(
        output / "fresh_pipeline.json",
        {
            "schema_version": "reform_fresh_pipeline_v1",
            "model": args.model,
            "memory_question_ids": [str(row["id"]) for row in rows],
            "test_question_ids": [],
            "uses_frozen_dags": False,
            "uses_frozen_blueprints": False,
            "uses_frozen_exemplars": False,
            "uses_frozen_evidence_databases": False,
        },
    )
    args.python = Path(os.path.abspath(os.path.expanduser(str(args.python))))
    args.dag_workers = args.workers
    args.dag_model = args.model
    core._build_dags(args, output, len(rows))
    core._materialize_graph_bank(output, rows, 0)
    print(f"DAGs: {len(rows)}/{len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
