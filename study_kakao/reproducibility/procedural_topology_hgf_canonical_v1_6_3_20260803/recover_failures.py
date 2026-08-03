#!/usr/bin/env python3
"""Retry only failed cases from a recorded canonical suite.

The retry criterion is execution validity, never forecast score. Each model is
re-run in a fresh isolated suite and retains its original frozen evidence and
retrieval manifests through ``run.py``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


BUNDLE = Path(__file__).resolve().parent
ROOT = BUNDLE.parents[1]


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--selection-file", type=Path, required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--workers-per-model", type=int, default=4)
    parser.add_argument("--max-parallel-models", type=int, default=3)
    parser.add_argument(
        "--provider-overrides", nargs="*", default=[], metavar="MODEL=PROVIDER"
    )
    return parser.parse_args()


def _slug(value: str) -> str:
    return value.replace("/", "_").replace(".", "_")


def _failed_ids(source_root: Path, model: str) -> list[str]:
    path = source_root / _slug(model) / "results.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [
        str(row["question_id"])
        for row in payload["results"]
        if row.get("status") != "success"
    ]


def main() -> int:
    args = _args()
    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    selection = args.selection_file.resolve()
    if output_root.exists():
        raise FileExistsError(f"fresh output root required: {output_root}")
    output_root.mkdir(parents=True)

    jobs: list[tuple[str, list[str], Path]] = []
    manifest: dict[str, object] = {
        "schema_version": "canonical_failure_recovery_v1",
        "implementation_revision": "canonical_v1_6_3",
        "source_root": str(source_root),
        "selection_file": str(selection),
        "selection_rule": "all and only non-success rows; no metric-based selection",
        "models": {},
    }
    for model in args.models:
        question_ids = _failed_ids(source_root, model)
        suite_root = output_root / f"{_slug(model)}_suite"
        model_provider_overrides = [
            item
            for item in args.provider_overrides
            if item.partition("=")[0] == model
        ]
        manifest["models"][model] = {
            "failed_question_ids": question_ids,
            "count": len(question_ids),
            "suite_root": str(suite_root),
            "provider_overrides": model_provider_overrides,
        }
        if not question_ids:
            continue
        command = [
            sys.executable,
            str(BUNDLE / "run.py"),
            "--models",
            model,
            "--selection-file",
            str(selection),
            "--limit",
            "100",
            "--question-ids",
            *question_ids,
            "--workers-per-model",
            str(min(args.workers_per_model, len(question_ids))),
            "--max-parallel-models",
            "1",
            "--output-root",
            str(suite_root),
            *(
                ["--provider-overrides", *model_provider_overrides]
                if model_provider_overrides
                else []
            ),
        ]
        jobs.append((model, command, suite_root))

    (output_root / "recovery_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    records: list[dict[str, object]] = []
    with ThreadPoolExecutor(
        max_workers=min(args.max_parallel_models, len(jobs) or 1)
    ) as executor:
        futures = {
            executor.submit(
                subprocess.run,
                command,
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            ): (model, command, suite_root)
            for model, command, suite_root in jobs
        }
        for future in as_completed(futures):
            model, command, suite_root = futures[future]
            completed = future.result()
            record = {
                "model": model,
                "returncode": completed.returncode,
                "suite_root": str(suite_root),
                "command": command,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
            }
            records.append(record)
            print(json.dumps(record, ensure_ascii=False), flush=True)
    (output_root / "recovery_status.json").write_text(
        json.dumps({"records": records}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0 if all(row["returncode"] == 0 for row in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
