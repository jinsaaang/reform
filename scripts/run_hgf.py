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
SOURCE_ROOT = ROOT / "hgf"


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--provider",
        help="Optional OpenRouter provider tag. Omit to use automatic routing.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--questions-dir", type=Path)
    parser.add_argument("--evidence-dir", type=Path)
    parser.add_argument(
        "--evidence-mode", choices=("frozen", "live"), default="frozen"
    )
    parser.add_argument(
        "--live-search-provider",
        choices=("auto", "google_news", "gdelt", "ddgs", "smolagents"),
        default="auto",
    )
    parser.add_argument("--live-query-limit", type=int, default=6)
    parser.add_argument("--live-fetch-limit", type=int, default=12)
    parser.add_argument("--live-fetch-workers", type=int, default=2)
    parser.add_argument("--selection-file", type=Path)
    parser.add_argument("--blueprint-root", type=Path)
    parser.add_argument("--exemplar-root", type=Path)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--reasoning-effort", default="medium")
    parser.add_argument("--max-output-tokens", type=int, default=16000)
    parser.add_argument("--run-seed", type=int, default=0)
    parser.add_argument("--question-ids", nargs="*")
    parser.add_argument("--disable-native-reasoning", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Continue an existing --output-dir instead of requiring a "
            "fresh one. Cases already recorded as successful are returned "
            "from their case file and not re-requested; only missing and "
            "previously failed cases run."
        ),
    )
    return parser.parse_args()


def _resolve(value: Path | None, default: Path) -> Path:
    return (value or default).expanduser().resolve()


def main() -> int:
    args = _args()
    if args.evidence_mode == "frozen" and not args.provider:
        raise ValueError("frozen runs require --provider")
    if args.evidence_mode == "live" and args.provider:
        raise ValueError("live runs use automatic routing; omit --provider")
    if args.evidence_mode == "live" and args.disable_native_reasoning:
        raise ValueError("--disable-native-reasoning is unavailable in live mode")
    dataset = args.dataset_root.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if output.exists() and not (args.dry_run or args.resume):
        raise FileExistsError(
            f"output directory already exists: {output}; "
            "pass --resume to continue it"
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
        "hgf_e2e_topology_live.run"
        if args.evidence_mode == "live"
        else "hgf_e2e_topology_provider_pinned.run"
    )
    command = [
        sys.executable,
        "-m",
        module,
        *(["--provider-only", args.provider] if args.provider else []),
        *(["--disable-native-reasoning"] if args.disable_native_reasoning else []),
        "--model",
        args.model,
        "--questions-dir",
        str(questions),
        "--evidence-dir",
        str(evidence),
        *(
            [
                "--evidence-mode",
                "live",
                "--live-search-provider",
                args.live_search_provider,
                "--live-query-limit",
                str(args.live_query_limit),
                "--live-fetch-limit",
                str(args.live_fetch_limit),
                "--live-fetch-workers",
                str(args.live_fetch_workers),
            ]
            if args.evidence_mode == "live"
            else []
        ),
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
    ]
    env = os.environ.copy()
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = os.pathsep.join(
        [str(SOURCE_ROOT), *([existing] if existing else [])]
    )
    launch = {
        "schema_version": "procedural_topology_hgf_portable_launch_v1",
        "command": command,
        "dataset_root": str(dataset),
        "source_root": str(SOURCE_ROOT),
    }
    if args.dry_run:
        print(json.dumps(launch, ensure_ascii=False, indent=2))
        print(shlex.join(command))
        return 0
    output.mkdir(parents=True, exist_ok=args.resume)
    (output / "portable_launch.json").write_text(
        json.dumps(launch, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    completed = subprocess.run(command, cwd=dataset, env=env, check=False)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
