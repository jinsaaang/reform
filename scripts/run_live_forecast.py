"""Forecast new target questions from fresh artifacts and live evidence."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--test-questions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="google/gemini-2.5-flash-lite")
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--provider")
    parser.add_argument("--question-id", action="append", default=[])
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--reasoning-effort", default="medium")
    parser.add_argument("--max-output-tokens", type=int, default=16000)
    parser.add_argument("--run-seed", type=int, default=0)
    parser.add_argument(
        "--search-provider",
        choices=("auto", "google_news", "gdelt", "ddgs", "smolagents"),
        default="auto",
    )
    parser.add_argument("--live-query-limit", type=int, default=6)
    parser.add_argument("--live-fetch-limit", type=int, default=12)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _rows(path: Path) -> list[dict]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return rows


def main() -> int:
    args = _args()
    root = args.artifact_root.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    tests = _rows(args.test_questions.expanduser().resolve())
    if args.question_id:
        requested = set(args.question_id)
        tests = [row for row in tests if str(row.get("id")) in requested]
        found = {str(row.get("id")) for row in tests}
        if found != requested:
            raise ValueError(f"unknown target IDs: {sorted(requested - found)}")
    if not tests:
        raise ValueError("no target questions selected")

    input_root = root / "target_inputs" / output.name
    questions = input_root / "questions"
    questions.mkdir(parents=True, exist_ok=True)
    shutil.copy2(root / "data/questions/memory_questions.jsonl", questions)
    (questions / "test_questions.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in tests),
        encoding="utf-8",
    )
    selection = input_root / "selection.json"
    selection.write_text(
        json.dumps(
            {
                "selection_rule": "live forecast input order",
                "question_ids": [str(row["id"]) for row in tests],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    python = Path(os.path.abspath(os.path.expanduser(str(args.python))))
    command = [
        str(python),
        str(ROOT / "scripts/run_hgf.py"),
        "--dataset-root",
        str(root),
        "--questions-dir",
        str(questions),
        "--selection-file",
        str(selection),
        "--blueprint-root",
        str(root / "artifacts/hgf/blueprints"),
        "--exemplar-root",
        str(root / "artifacts/hgf/exemplars"),
        "--model",
        args.model,
        "--output-dir",
        str(output),
        "--evidence-mode",
        "live",
        "--live-search-provider",
        args.search_provider,
        "--live-query-limit",
        str(args.live_query_limit),
        "--live-fetch-limit",
        str(args.live_fetch_limit),
        "--workers",
        str(args.workers),
        "--limit",
        str(len(tests)),
        "--reasoning-effort",
        args.reasoning_effort,
        "--max-output-tokens",
        str(args.max_output_tokens),
        "--run-seed",
        str(args.run_seed),
    ]
    if args.provider:
        command.extend(["--provider", args.provider])
    if args.resume:
        command.append("--resume")
    return subprocess.run(command, cwd=root, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
