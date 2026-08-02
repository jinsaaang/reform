#!/usr/bin/env python3
"""Run Procedural Topology HGF with an answer-free worked reasoning check."""

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


BUNDLE = Path(__file__).resolve().parent
ROOT = BUNDLE.parents[1]
PARENT_BUNDLE = ROOT / "reproducibility/procedural_topology_hgf_full100_20260802"
CONFIG = json.loads((PARENT_BUNDLE / "config.json").read_text(encoding="utf-8"))
EVIDENCE_SLUGS = {
    "google/gemini-2.5-flash-lite": "google_gemini-2.5-flash-lite",
    "openai/gpt-5-mini": "openai_gpt-5-mini",
    "deepseek/deepseek-v3.2": "deepseek_deepseek-v3.2",
    "meta-llama/llama-4-maverick": "meta-llama_llama-4-maverick",
    "minimax/minimax-m2.5": "minimax_minimax-m2.5",
}


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="*", choices=tuple(CONFIG["models"]))
    parser.add_argument("--selection-file", type=Path, required=True)
    parser.add_argument("--limit", type=int, required=True)
    parser.add_argument("--question-ids", nargs="*")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers-per-model", type=int, default=8)
    parser.add_argument("--max-parallel-models", type=int, default=5)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _slug(value: str) -> str:
    return value.replace("/", "_").replace(".", "_")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _inputs(model: str) -> tuple[Path, Path]:
    root = PARENT_BUNDLE / "inputs/model_evidence" / EVIDENCE_SLUGS[model]
    return root / "manifest.json", root / "retrieval_manifest.json"


def _command(
    *,
    model: str,
    selection: Path,
    limit: int,
    output: Path,
    workers: int,
    question_ids: list[str] | None,
) -> list[str]:
    setting = CONFIG["models"][model]
    evidence, retrieval = _inputs(model)
    return [
        sys.executable,
        "-m",
        "hgf_original_input_adapter.run",
        "--provider-only",
        str(setting["hgf_provider"]),
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
        str(selection),
        "--blueprint-root",
        str(ROOT / "artifacts/hgf/blueprints"),
        "--exemplar-root",
        str(ROOT / "artifacts/hgf/exemplars"),
        "--output-dir",
        str(output),
        "--limit",
        str(limit),
        "--workers",
        str(workers),
        "--reasoning-effort",
        str(
            "medium"
            if setting.get("disable_native_reasoning")
            else setting["reasoning_effort"]
        ),
        "--max-output-tokens",
        str(setting["max_output_tokens"]),
        "--run-seed",
        "0",
        *(
            ["--question-ids", *question_ids]
            if question_ids
            else []
        ),
        "--evidence-selection-manifest",
        str(evidence),
        "--retrieval-manifest",
        str(retrieval),
    ]


def _run(command: list[str], env: dict[str, str], log: Path, dry_run: bool) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        log.write_text(" ".join(command) + "\n", encoding="utf-8")
        return 0
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
    return completed.returncode


def main() -> int:
    args = _args()
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"fresh output root required: {output_root}")
    selection = args.selection_file.resolve()
    selected = json.loads(selection.read_text(encoding="utf-8")).get("question_ids") or []
    if len(selected) < args.limit:
        raise ValueError(
            f"selection contains {len(selected)} questions but limit is {args.limit}"
        )
    models = args.models or list(CONFIG["models"])
    method_files = sorted((BUNDLE / "method_src/hgf_e2e_topology").glob("*.py"))
    manifest = {
        "schema_version": "procedural_topology_exemplar_check_suite_v1",
        "parent_registered_method": "Procedural Topology HGF",
        "parent_method_commit": "a3b07a06e51772bb25d2fb99b3d36a61fccc2898",
        "historical_dependency_commit": "27ff13cf8b2e1f20e88822e895a7b02055d9be30",
        "selection_file": str(selection),
        "selection_sha256": _sha256(selection),
        "limit": args.limit,
        "question_ids": list(args.question_ids or []),
        "seed": 0,
        "models": models,
        "workers_per_model": args.workers_per_model,
        "method_changes": [
            "fixed cutoff-safe worked traces are projected into answer-free reasoning checks",
            "checks retain reasoning order, counterevidence, uncertainty, and structural lessons",
            "historical estimates, option mappings, evidence payloads, answers, and probabilities are excluded",
            "DAG paths guide but do not confine current reasoning",
            "reasoning uses the same maxItems 10 budget as registered baselines",
            "provider-specific audit aliases and missing step labels are deterministically normalized without changing probabilities",
            "unregistered audit labels conservatively fall back to target-contract or current-evidence sources without claiming DAG use",
        ],
        "unchanged": [
            "frozen model-specific evidence",
            "frozen model-specific historical retrieval",
            "evidence ledger",
            "exact-family subgraph routing",
            "current graph instantiation",
            "boundary mapping",
            "single probability call",
            "no probability postprocessing or answer reuse",
        ],
        "method_source_hashes": {
            path.name: _sha256(path) for path in method_files
        },
    }
    output_root.mkdir(parents=True, exist_ok=False)
    _write(output_root / "suite_manifest.json", manifest)

    tasks = []
    for model in models:
        output = output_root / _slug(model)
        command = _command(
            model=model,
            selection=selection,
            limit=args.limit,
            output=output,
            workers=args.workers_per_model,
            question_ids=args.question_ids,
        )
        env = os.environ.copy()
        pythonpath = [
            BUNDLE / "method_src",
            PARENT_BUNDLE / "hgf_historical_base_src",
            BUNDLE / "input_adapter_src",
            ROOT / "src",
        ]
        existing = env.get("PYTHONPATH")
        env["PYTHONPATH"] = os.pathsep.join(
            [*(str(path) for path in pythonpath), *([existing] if existing else [])]
        )
        tasks.append(
            (
                model,
                command,
                env,
                output_root / "logs" / f"{_slug(model)}.log",
            )
        )

    records: list[dict[str, Any]] = []
    with ThreadPoolExecutor(
        max_workers=min(args.max_parallel_models, len(tasks))
    ) as executor:
        futures = {
            executor.submit(_run, command, env, log, args.dry_run): (model, log)
            for model, command, env, log in tasks
        }
        for future in as_completed(futures):
            model, log = futures[future]
            record = {
                "model": model,
                "returncode": future.result(),
                "log": str(log),
            }
            records.append(record)
            print(json.dumps(record, ensure_ascii=False), flush=True)
            _write(output_root / "run_status.json", {"records": records})
    return int(any(record["returncode"] != 0 for record in records))


if __name__ == "__main__":
    raise SystemExit(main())
