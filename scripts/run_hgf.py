#!/usr/bin/env python3
"""Run the canonical Procedural Topology HGF on a compatible benchmark."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOTS = (
    ROOT / "hgf" / "method",
    ROOT / "hgf" / "shared",
    ROOT / "hgf" / "input_adapter",
    ROOT / "hgf" / "execution",
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--provider", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--questions-dir", type=Path)
    parser.add_argument("--evidence-dir", type=Path)
    parser.add_argument("--selection-file", type=Path)
    parser.add_argument("--blueprint-root", type=Path)
    parser.add_argument("--exemplar-root", type=Path)
    parser.add_argument("--evidence-selection-manifest", type=Path)
    parser.add_argument("--retrieval-manifest", type=Path)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--reasoning-effort", default="medium")
    parser.add_argument("--max-output-tokens", type=int, default=16000)
    parser.add_argument("--run-seed", type=int, default=0)
    parser.add_argument("--question-ids", nargs="*")
    parser.add_argument("--disable-native-reasoning", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _resolve(value: Path | None, default: Path) -> Path:
    return (value or default).expanduser().resolve()


def main() -> int:
    args = _args()
    dataset = args.dataset_root.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if output.exists() and not args.dry_run:
        raise FileExistsError(f"fresh output directory required: {output}")

    evidence_manifest = args.evidence_selection_manifest
    retrieval_manifest = args.retrieval_manifest
    if bool(evidence_manifest) != bool(retrieval_manifest):
        raise ValueError(
            "frozen replay requires both --evidence-selection-manifest and "
            "--retrieval-manifest"
        )

    questions = _resolve(args.questions_dir, dataset / "data" / "questions")
    evidence = _resolve(args.evidence_dir, dataset / "data" / "evidence")
    selection = _resolve(args.selection_file, questions / "selection.json")
    blueprints = _resolve(
        args.blueprint_root, dataset / "artifacts" / "hgf" / "blueprints"
    )
    exemplars = _resolve(
        args.exemplar_root, dataset / "artifacts" / "hgf" / "exemplars"
    )

    module = (
        "hgf_original_input_adapter.run"
        if evidence_manifest
        else "hgf_e2e_topology_provider_pinned.run"
    )
    command = [
        sys.executable,
        "-m",
        module,
        "--provider-only",
        args.provider,
        *(["--disable-native-reasoning"] if args.disable_native_reasoning else []),
        "--model",
        args.model,
        "--questions-dir",
        str(questions),
        "--evidence-dir",
        str(evidence),
        "--selection-file",
        str(selection),
        "--blueprint-root",
        str(blueprints),
        "--exemplar-root",
        str(exemplars),
        "--output-dir",
        str(output),
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
        *(["--question-ids", *args.question_ids] if args.question_ids else []),
        *(
            [
                "--evidence-selection-manifest",
                str(evidence_manifest.expanduser().resolve()),
                "--retrieval-manifest",
                str(retrieval_manifest.expanduser().resolve()),
            ]
            if evidence_manifest and retrieval_manifest
            else []
        ),
    ]
    env = os.environ.copy()
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = os.pathsep.join(
        [*(str(path) for path in SOURCE_ROOTS), *([existing] if existing else [])]
    )
    launch = {
        "schema_version": "procedural_topology_hgf_portable_launch_v1",
        "command": command,
        "dataset_root": str(dataset),
        "frozen_inputs": bool(evidence_manifest),
        "source_roots": [str(path) for path in SOURCE_ROOTS],
    }
    if args.dry_run:
        print(json.dumps(launch, ensure_ascii=False, indent=2))
        print(shlex.join(command))
        return 0
    output.mkdir(parents=True, exist_ok=False)
    (output / "portable_launch.json").write_text(
        json.dumps(launch, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    completed = subprocess.run(command, cwd=dataset, env=env, check=False)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
