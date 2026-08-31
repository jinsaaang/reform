"""Compile fresh DAGs into 1.7.0 Blueprints and worked exemplars."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import run_fresh_pipeline as core


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--model", default="google/gemini-2.5-flash-lite")
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--reasoning-effort", default="medium")
    parser.add_argument("--blueprints-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _args()
    root = args.artifact_root.expanduser().resolve()
    required = (
        root / "data/memory_bank/manifest.json",
        root / "data/questions/memory_questions.jsonl",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing fresh DAG inputs: " + ", ".join(missing))
    args.python = Path(os.path.abspath(os.path.expanduser(str(args.python))))
    args.exemplar_workers = args.workers
    core._build_blueprints(args, root)
    print("Blueprint compilation complete")
    if not args.blueprints_only:
        core._build_exemplars(args, root)
        print("Exemplar generation complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

