#!/usr/bin/env python3
"""Replay the canonical full-100 HGF and sanitized baseline experiments."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


BUNDLE = Path(__file__).resolve().parent
ROOT = BUNDLE.parents[1]
CONFIG = json.loads((BUNDLE / "config.json").read_text(encoding="utf-8"))
EVIDENCE_SLUGS = {
    "google/gemini-2.5-flash-lite": "google_gemini-2.5-flash-lite",
    "openai/gpt-5-mini": "openai_gpt-5-mini",
    "deepseek/deepseek-v3.2": "deepseek_deepseek-v3.2",
    "meta-llama/llama-4-maverick": "meta-llama_llama-4-maverick",
    "minimax/minimax-m2.5": "minimax_minimax-m2.5",
}


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("hgf", "baselines", "both"), default="both")
    parser.add_argument("--models", nargs="*", choices=tuple(CONFIG["models"]))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers-per-model", type=int, default=8)
    parser.add_argument("--max-parallel-models", type=int, default=5)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _slug(model: str) -> str:
    return model.replace("/", "_").replace(".", "_")


def _inputs(model: str) -> tuple[Path, Path]:
    root = BUNDLE / "inputs/model_evidence" / EVIDENCE_SLUGS[model]
    return root / "manifest.json", root / "retrieval_manifest.json"


def _hgf_command(model: str, output: Path, workers: int) -> list[str]:
    setting = CONFIG["models"][model]
    evidence, retrieval = _inputs(model)
    return [
        sys.executable,
        "-m",
        "hgf_original_input_adapter.run",
        "--provider-only",
        str(setting["hgf_provider"]),
        *(["--disable-native-reasoning"] if setting.get("disable_native_reasoning") else []),
        "--model",
        model,
        "--questions-dir",
        str(ROOT / "data/questions"),
        "--evidence-dir",
        str(ROOT / "data/evidence"),
        "--selection-file",
        str(ROOT / CONFIG["selection"]),
        "--blueprint-root",
        str(ROOT / "artifacts/hgf/blueprints"),
        "--output-dir",
        str(output),
        "--limit",
        "100",
        "--workers",
        str(workers),
        "--reasoning-effort",
        str(setting["reasoning_effort"]),
        "--max-output-tokens",
        str(setting["max_output_tokens"]),
        "--run-seed",
        "0",
        "--evidence-selection-manifest",
        str(evidence),
        "--retrieval-manifest",
        str(retrieval),
    ]


def _baseline_command(model: str, output: Path, workers: int) -> list[str]:
    setting = CONFIG["models"][model]
    evidence, retrieval = _inputs(model)
    return [
        sys.executable,
        "-m",
        "hgf_baseline_sanitation_v1_2.run",
        "--model",
        model,
        "--provider-only",
        str(setting["baseline_provider"]),
        "--reasoning-effort",
        str(setting["reasoning_effort"]),
        "--max-output-tokens",
        str(setting["max_output_tokens"]),
        "--workers",
        str(workers),
        "--limit",
        "100",
        "--run-seed",
        "0",
        "--selection-file",
        str(ROOT / CONFIG["selection"]),
        "--methods",
        "case_memory",
        "direct_dag",
        "--output-dir",
        str(output),
        "--evidence-selection-manifest",
        str(evidence),
        "--retrieval-manifest",
        str(retrieval),
        "--neutral-topology-cache-dir",
        str(BUNDLE / "inputs/neutral_topology"),
        "--require-frozen-neutral-topology",
    ]


def _run(command: list[str], env: dict[str, str], log: Path, dry_run: bool) -> dict[str, Any]:
    log.parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        return {"returncode": 0, "command": command, "dry_run": True}
    with log.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    return {"returncode": completed.returncode, "command": command, "dry_run": False}


def main() -> int:
    args = _args()
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"fresh output root required: {output_root}")
    models = args.models or list(CONFIG["models"])
    suites = ("hgf", "baselines") if args.suite == "both" else (args.suite,)
    tasks: list[tuple[str, str, list[str], dict[str, str], Path]] = []
    for suite in suites:
        for model in models:
            output = output_root / suite / _slug(model)
            env = os.environ.copy()
            if suite == "hgf":
                pythonpath = [
                    BUNDLE / "hgf_method_src",
                    BUNDLE / "hgf_historical_base_src",
                    BUNDLE / "hgf_input_adapter_src",
                ]
                command = _hgf_command(model, output, args.workers_per_model)
            else:
                pythonpath = [BUNDLE / "baseline_src"]
                command = _baseline_command(model, output, args.workers_per_model)
            existing = env.get("PYTHONPATH")
            env["PYTHONPATH"] = os.pathsep.join(
                [*(str(path) for path in pythonpath), *([existing] if existing else [])]
            )
            tasks.append((suite, model, command, env, output_root / "logs" / f"{suite}_{_slug(model)}.log"))

    output_root.mkdir(parents=True, exist_ok=False)
    records: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=min(args.max_parallel_models, len(tasks))) as executor:
        futures = {
            executor.submit(_run, command, env, log, args.dry_run): (suite, model)
            for suite, model, command, env, log in tasks
        }
        for future in as_completed(futures):
            suite, model = futures[future]
            record = {"suite": suite, "model": model, **future.result()}
            records.append(record)
            print(json.dumps(record, ensure_ascii=False), flush=True)
    (output_root / "replay_status.json").write_text(
        json.dumps({"records": records}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return int(any(record["returncode"] != 0 for record in records))


if __name__ == "__main__":
    raise SystemExit(main())
