#!/usr/bin/env python3
"""Run the six controlled baselines from the frozen comparison implementation."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE_SRC = ROOT / "hgf" / "shared"
METHODS = (
    "search_only",
    "prospective_dag",
    "direct_dag",
    "factor_memory",
    "case_memory",
    "text_memory",
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--selection-file", type=Path)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--reasoning-effort", default="medium")
    parser.add_argument("--max-output-tokens", type=int, default=16000)
    parser.add_argument("--run-seed", type=int, default=0)
    parser.add_argument("--question-ids", nargs="*")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _args()
    dataset = args.dataset_root.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if output.exists() and not args.dry_run:
        raise FileExistsError(f"fresh output directory required: {output}")
    selection = (
        args.selection_file or dataset / "data/questions/selection.json"
    ).resolve()
    command = [
        sys.executable,
        "-m",
        "hgf.baselines",
        "--questions-dir",
        str(dataset / "data/questions"),
        "--evidence-dir",
        str(dataset / "data/evidence"),
        "--memory-bank-manifest",
        str(dataset / "data/memory_bank/manifest.json"),
        "--selection-file",
        str(selection),
        "--output-dir",
        str(output),
        "--model",
        args.model,
        "--limit",
        str(args.limit),
        "--workers",
        str(args.workers),
        "--reasoning-effort",
        args.reasoning_effort,
        "--max-output-tokens",
        str(args.max_output_tokens),
        "--run-seed",
        str(args.run_seed),
        "--methods",
        *args.methods,
        *(["--question-ids", *args.question_ids] if args.question_ids else []),
    ]
    env = os.environ.copy()
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = os.pathsep.join(
        [str(BASE_SRC), *([existing] if existing else [])]
    )
    if args.dry_run:
        print(" ".join(command))
        return 0
    return subprocess.run(command, cwd=dataset, env=env, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
