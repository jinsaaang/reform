#!/usr/bin/env python3
"""Run the archived Procedural Topology HGF without method changes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SNAPSHOT_SRC = Path(
    "/tmp/hgf-original-a3b07a0/icaif/live_hgf_snapshots/"
    "20260731_procedural_topology_hgf_full100_brier_0_203901/src"
)
DEFAULT_BASE_SRC = Path("/tmp/hgf-base-27ff13c/study_kakao/src")
ARCHIVED_COMMIT = "a3b07a06e51772bb25d2fb99b3d36a61fccc2898"
BASE_COMMIT = "27ff13cf8b2e1f20e88822e895a7b02055d9be30"

MODEL_SETTINGS: dict[str, dict[str, Any]] = {
    "google/gemini-2.5-flash-lite": {
        "provider": "google-ai-studio",
        "max_tokens": 16000,
        "evidence": "runs/paper_canonical_v2_20260801/model_evidence/google_gemini-2.5-flash-lite",
    },
    "openai/gpt-5-mini": {
        "provider": "openai",
        "max_tokens": 16000,
        "evidence": "runs/paper_canonical_v2_20260801/model_evidence/openai_gpt-5-mini",
    },
    "deepseek/deepseek-v3.2": {
        "provider": "baidu",
        "max_tokens": 16000,
        "evidence": "runs/paper_canonical_v2_20260801/model_evidence/deepseek_deepseek-v3.2",
    },
    "meta-llama/llama-4-maverick": {
        "provider": "deepinfra",
        "max_tokens": 16000,
        "disable_native_reasoning": True,
        "evidence": "runs/paper_canonical_v2_20260801/model_evidence/meta-llama_llama-4-maverick",
    },
    "minimax/minimax-m2.5": {
        "provider": "friendli",
        "max_tokens": 32768,
        "evidence": "runs/paper_canonical_v2_20260801/additional_model_evidence/minimax_minimax-m2.5_friendli",
    },
}


def _slug(value: str) -> str:
    return value.replace("/", "_")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-src", type=Path, default=DEFAULT_SNAPSHOT_SRC)
    parser.add_argument("--base-src", type=Path, default=DEFAULT_BASE_SRC)
    parser.add_argument("--models", nargs="+", choices=tuple(MODEL_SETTINGS))
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-parallel-models", type=int, default=5)
    parser.add_argument("--reasoning-effort", default="medium")
    parser.add_argument("--run-seed", type=int, default=0)
    parser.add_argument(
        "--question-ids",
        nargs="*",
        help="Optional exact recovery set, filtered in registered selection order.",
    )
    parser.add_argument(
        "--selection-file",
        type=Path,
        default=ROOT / "data/questions/selection_balanced_40.json",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _command(
    *,
    model: str,
    output_dir: Path,
    args: argparse.Namespace,
) -> list[str]:
    setting = MODEL_SETTINGS[model]
    evidence_root = ROOT / str(setting["evidence"])
    return [
        sys.executable,
        "-m",
        "hgf_original_input_adapter.run",
        "--provider-only",
        str(setting["provider"]),
        *(
            ["--disable-native-reasoning"]
            if setting.get("disable_native_reasoning")
            else []
        ),
        "--model",
        model,
        "--questions-dir",
        str(ROOT / "data/questions"),
        "--evidence-dir",
        str(ROOT / "data/evidence"),
        "--selection-file",
        str(args.selection_file.resolve()),
        "--blueprint-root",
        str(ROOT / "artifacts/hgf/blueprints"),
        "--output-dir",
        str(output_dir),
        "--limit",
        str(args.limit),
        "--workers",
        str(args.workers),
        "--reasoning-effort",
        str(args.reasoning_effort),
        "--max-output-tokens",
        str(setting["max_tokens"]),
        "--run-seed",
        str(args.run_seed),
        *(
            ["--question-ids", *args.question_ids]
            if args.question_ids
            else []
        ),
        "--evidence-selection-manifest",
        str(evidence_root / "manifest.json"),
        "--retrieval-manifest",
        str(evidence_root / "retrieval_manifest.json"),
    ]


def _run_one(
    *, model: str, output_dir: Path, args: argparse.Namespace, env: dict[str, str]
) -> dict[str, Any]:
    command = _command(model=model, output_dir=output_dir, args=args)
    output_dir.mkdir(parents=True, exist_ok=False)
    _write(output_dir / "launch_command.json", {"command": command})
    with (output_dir / "run.log").open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    result_path = output_dir / "results.json"
    summary: dict[str, Any] = {}
    if result_path.is_file():
        summary = json.loads(result_path.read_text(encoding="utf-8")).get(
            "summary", {}
        )
    return {
        "model": model,
        "returncode": completed.returncode,
        "output_dir": str(output_dir),
        "summary": summary,
    }


def main() -> None:
    args = _args()
    models = args.models or list(MODEL_SETTINGS)
    snapshot_src = args.snapshot_src.resolve()
    base_src = args.base_src.resolve()
    method_run = snapshot_src / "hgf_e2e_topology/run.py"
    if not method_run.is_file():
        raise FileNotFoundError(f"archived HGF source is missing: {method_run}")
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"fresh output root required: {output_root}")

    required_inputs: list[Path] = [
        args.selection_file.resolve(),
        method_run,
        base_src / "hgf/boundary.py",
        base_src / "hgf/baselines.py",
    ]
    for model in models:
        evidence_root = ROOT / str(MODEL_SETTINGS[model]["evidence"])
        required_inputs.extend(
            [evidence_root / "manifest.json", evidence_root / "retrieval_manifest.json"]
        )
    missing = [str(path) for path in required_inputs if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing registered inputs: {missing}")

    commands = {
        model: _command(
            model=model,
            output_dir=output_root / _slug(model),
            args=args,
        )
        for model in models
    }
    manifest = {
        "schema_version": "original_procedural_topology_suite_v1",
        "archived_commit": ARCHIVED_COMMIT,
        "historical_base_commit": BASE_COMMIT,
        "historical_base_src": str(base_src),
        "historical_base_boundary_sha256": _sha256(base_src / "hgf/boundary.py"),
        "method_source": str(method_run),
        "method_source_sha256": _sha256(method_run),
        "method_changes": "none",
        "observer_changes_forecast": False,
        "input_adapter_changes_forecast_implementation": False,
        "input_adapter_changes_registered_inputs": True,
        "registered_input_change": (
            "model-specific cutoff-safe evidence and retrieved past-event IDs "
            "are supplied from frozen manifests"
        ),
        "reasoning_effort": args.reasoning_effort,
        "selection_file": str(args.selection_file.resolve()),
        "selection_sha256": _sha256(args.selection_file.resolve()),
        "limit": args.limit,
        "question_ids": list(args.question_ids or []),
        "workers_per_model": args.workers,
        "models": models,
        "commands": commands,
    }
    if args.dry_run:
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return
    output_root.mkdir(parents=True, exist_ok=False)
    _write(output_root / "suite_manifest.json", manifest)

    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(
        value
        for value in (
            str(snapshot_src),
            str(base_src),
            str(ROOT / "src"),
            existing,
        )
        if value
    )
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(
        max_workers=max(1, min(args.max_parallel_models, len(models)))
    ) as executor:
        futures = {
            executor.submit(
                _run_one,
                model=model,
                output_dir=output_root / _slug(model),
                args=args,
                env=env,
            ): model
            for model in models
        }
        for future in as_completed(futures):
            row = future.result()
            results.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
            _write(output_root / "suite_status.json", {"results": results})
    _write(output_root / "suite_results.json", {"results": results})
    if any(int(row["returncode"]) != 0 for row in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
