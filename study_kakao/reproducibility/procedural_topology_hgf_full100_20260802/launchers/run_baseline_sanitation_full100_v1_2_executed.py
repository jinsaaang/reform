#!/usr/bin/env python3
"""Run only the two sanitized baselines through a strict full-100 gate.

The runner uses one fresh output root.  Its five-question preflight is a
promotion gate, not a balanced-40 experiment.  The final balanced-40 numbers
are calculated solely by filtering the completed full-100 artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from statistics import fmean
from typing import Any

import run_baseline_sanitation_balanced40 as base


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
MODELS = base.MODELS
METHODS = ("case_memory", "direct_dag")
PHASE_LIMITS = {"preflight": 5, "full": 100}
SOURCE_PATHS = {
    "baseline_sanitation_v1": SRC / "hgf_baseline_sanitation_v1" / "run.py",
    "baseline_sanitation_v1_1": SRC / "hgf_baseline_sanitation_v1_1" / "run.py",
    "baseline_sanitation_v1_2": SRC / "hgf_baseline_sanitation_v1_2" / "run.py",
    "baseline_admission_audit_v1_2": SRC / "hgf_baseline_sanitation_v1_2" / "audit.py",
    "baseline_core": SRC / "hgf" / "baselines.py",
    "raw_audit": SRC / "hgf" / "raw_audit.py",
    "neutral_topology": SRC / "hgf_historical_live_structured" / "neutral_topology.py",
}


def _slug(value: str) -> str:
    return value.replace("/", "_").replace(".", "_")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("preflight", "full"), required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/baseline_sanitation_full100_v1_2_20260802"),
    )
    parser.add_argument(
        "--selection-file",
        type=Path,
        default=Path("data/questions/selection.json"),
    )
    parser.add_argument(
        "--balanced40-selection-file",
        type=Path,
        default=Path("data/questions/selection_balanced_40.json"),
    )
    parser.add_argument(
        "--neutral-topology-cache-dir",
        type=Path,
        default=Path(
            "runs/baseline_sanitation_v1_20260802_r3/"
            "frozen_outcome_neutral_topology"
        ),
    )
    parser.add_argument("--workers-per-model", type=int, default=6)
    parser.add_argument("--max-parallel-models", type=int, default=5)
    parser.add_argument("--models", nargs="*", choices=tuple(MODELS))
    return parser.parse_args()


def _environment() -> dict[str, str]:
    env = os.environ.copy()
    current = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(SRC) + (os.pathsep + current if current else "")
    return env


def _input_manifest(
    *,
    output_root: Path,
    selection_file: Path,
    balanced40_file: Path,
    topology_dir: Path,
    models: list[str],
) -> None:
    manifest = {
        "schema_version": "baseline_sanitation_full100_manifest_v1_2",
        "output_root": str(output_root),
        "methods": list(METHODS),
        "selection": {
            "path": str(selection_file),
            "sha256": _sha256(selection_file),
            "question_count": len(_read(selection_file).get("question_ids") or []),
        },
        "balanced40_reporting_subset": {
            "path": str(balanced40_file),
            "sha256": _sha256(balanced40_file),
            "question_count": len(_read(balanced40_file).get("question_ids") or []),
            "derivation": "filtered from complete full100 results only",
        },
        "frozen_topology": {
            "path": str(topology_dir / "manifest.json"),
            "sha256": _sha256(topology_dir / "manifest.json"),
        },
        "source_hashes": {name: _sha256(path) for name, path in SOURCE_PATHS.items()},
        "model_configs": {
            model: {
                "provider": MODELS[model]["provider"],
                "reasoning_effort": MODELS[model]["reasoning_effort"],
                "max_output_tokens": MODELS[model]["max_output_tokens"],
                "evidence_manifest": str(MODELS[model]["evidence"]),
                "evidence_manifest_sha256": _sha256(Path(MODELS[model]["evidence"])),
                "retrieval_manifest": str(MODELS[model]["retrieval"]),
                "retrieval_manifest_sha256": _sha256(Path(MODELS[model]["retrieval"])),
            }
            for model in models
        },
        "generation_policy": {
            "run_seed": 0,
            "provider_fallback": False,
            "baseline_fallback": False,
            "probability_pooling": False,
            "posterior_adjustment": False,
            "cross_method_prediction_reuse": False,
        },
    }
    _write(output_root / "input_manifest.json", manifest)


def _command(
    *,
    phase: str,
    model: str,
    output_root: Path,
    selection_file: Path,
    topology_dir: Path,
    workers: int,
) -> list[str]:
    setting = MODELS[model]
    run_dir = output_root / phase / _slug(model)
    return [
        sys.executable,
        "-m",
        "hgf_baseline_sanitation_v1_2.run",
        "--model",
        model,
        "--provider-only",
        str(setting["provider"]),
        "--reasoning-effort",
        str(setting["reasoning_effort"]),
        "--max-output-tokens",
        str(setting["max_output_tokens"]),
        "--workers",
        str(workers),
        "--limit",
        str(PHASE_LIMITS[phase]),
        "--run-seed",
        "0",
        "--selection-file",
        str(selection_file),
        "--methods",
        *METHODS,
        "--output-dir",
        str(run_dir),
        "--evidence-selection-manifest",
        str(setting["evidence"]),
        "--retrieval-manifest",
        str(setting["retrieval"]),
        "--neutral-topology-cache-dir",
        str(topology_dir),
        "--require-frozen-neutral-topology",
    ]


def _run_one(command: list[str], log_path: Path) -> dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        process = subprocess.run(
            command,
            cwd=ROOT,
            env=_environment(),
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    return {"returncode": process.returncode, "command": command, "log": str(log_path)}


def _gate(run_dir: Path, expected: int) -> list[str]:
    errors: list[str] = []
    audit_path = run_dir / "baseline_admission_audit.json"
    if not audit_path.is_file():
        return ["missing baseline_admission_audit.json"]
    audit = _read(audit_path)
    if audit.get("status") != "passed":
        errors.extend(audit.get("errors") or ["admission audit failed"])
    if int(audit.get("expected_runs") or 0) != expected * len(METHODS):
        errors.append("admission audit expected count differs from phase contract")
    if int(audit.get("result_count") or 0) != expected * len(METHODS):
        errors.append("admission audit result count differs from phase contract")
    return errors


def _metric(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    return {
        "n": len(rows),
        "accuracy": fmean(float(row["metrics"]["accuracy"]) for row in rows),
        "brier": fmean(float(row["metrics"]["brier"]) for row in rows),
        "nll": fmean(float(row["metrics"]["nll"]) for row in rows),
        "mean_total_tokens": fmean(
            float((row.get("audit_usage") or row.get("usage") or {}).get("total_tokens") or 0)
            for row in rows
        ),
        "mean_cost": fmean(float((row.get("audit_usage") or {}).get("cost") or 0) for row in rows),
        "mean_seconds": fmean(float(row.get("seconds") or 0) for row in rows),
    }


def _write_balanced40_summary(output_root: Path, models: list[str], balanced40_file: Path) -> None:
    subset = set(str(value) for value in _read(balanced40_file).get("question_ids") or [])
    table: dict[str, dict[str, Any]] = {}
    for model in models:
        result_path = output_root / "full" / _slug(model) / "results.json"
        payload = _read(result_path)
        rows = [
            row
            for row in payload.get("results") or []
            if str(row.get("question_id")) in subset and row.get("status") == "success"
        ]
        table[model] = {}
        for method in METHODS:
            selected = [row for row in rows if row.get("method") == method]
            if len(selected) != len(subset):
                raise RuntimeError(
                    f"{model}/{method} has {len(selected)} balanced40 rows, expected {len(subset)}"
                )
            table[model][method] = _metric(selected)
    _write(
        output_root / "full100_balanced40_summary.json",
        {
            "schema_version": "baseline_sanitation_balanced40_from_full100_v1_2",
            "source": "full100 results only; no independent balanced40 execution",
            "balanced40_selection_file": str(balanced40_file),
            "balanced40_selection_sha256": _sha256(balanced40_file),
            "models": table,
        },
    )


def main() -> None:
    args = _args()
    output_root = args.output_dir.resolve()
    selection_file = args.selection_file.resolve()
    balanced40_file = args.balanced40_selection_file.resolve()
    topology_dir = args.neutral_topology_cache_dir.resolve()
    models = args.models or list(MODELS)
    if not selection_file.is_file() or not balanced40_file.is_file():
        raise FileNotFoundError("selection input is missing")
    if not (topology_dir / "manifest.json").is_file():
        raise FileNotFoundError("frozen neutral topology manifest is missing")
    selected = _read(selection_file).get("question_ids") or []
    if args.phase == "full" and len(selected) != 100:
        raise ValueError("full phase requires data/questions/selection.json with exactly 100 IDs")
    if not set(_read(balanced40_file).get("question_ids") or []).issubset(set(selected)):
        raise ValueError("balanced40 IDs must be a subset of the full selection")
    for model in models:
        for field in ("evidence", "retrieval"):
            if not Path(MODELS[model][field]).is_file():
                raise FileNotFoundError(f"missing {field} manifest for {model}")

    _input_manifest(
        output_root=output_root,
        selection_file=selection_file,
        balanced40_file=balanced40_file,
        topology_dir=topology_dir,
        models=models,
    )
    suite = {
        "schema_version": "baseline_sanitation_full100_suite_v1_2",
        "phase": args.phase,
        "status": "running",
        "models": {},
        "started_unix": time.time(),
        "input_manifest": str(output_root / "input_manifest.json"),
        "registered_or_hgf_outputs_modified": False,
    }
    _write(output_root / f"{args.phase}_suite_status.json", suite)

    def task(model: str) -> tuple[str, dict[str, Any]]:
        run_dir = output_root / args.phase / _slug(model)
        command = _command(
            phase=args.phase,
            model=model,
            output_root=output_root,
            selection_file=selection_file,
            topology_dir=topology_dir,
            workers=max(1, args.workers_per_model),
        )
        outcome = _run_one(
            command,
            output_root / "logs" / args.phase / f"{_slug(model)}.log",
        )
        outcome["run_dir"] = str(run_dir)
        outcome["gate_errors"] = (
            _gate(run_dir, PHASE_LIMITS[args.phase])
            if outcome["returncode"] == 0
            else ["subprocess failed"]
        )
        return model, outcome

    with ThreadPoolExecutor(max_workers=min(args.max_parallel_models, len(models))) as executor:
        futures = {executor.submit(task, model): model for model in models}
        for future in as_completed(futures):
            model, outcome = future.result()
            suite["models"][model] = outcome
            _write(output_root / f"{args.phase}_suite_status.json", suite)
    failures = [
        model
        for model, outcome in suite["models"].items()
        if outcome["returncode"] != 0 or outcome["gate_errors"]
    ]
    suite["status"] = "passed" if not failures else "failed"
    suite["failure_models"] = failures
    suite["finished_unix"] = time.time()
    _write(output_root / f"{args.phase}_suite_status.json", suite)
    if failures:
        raise SystemExit(1)
    if args.phase == "full":
        _write_balanced40_summary(output_root, models, balanced40_file)


if __name__ == "__main__":
    main()
