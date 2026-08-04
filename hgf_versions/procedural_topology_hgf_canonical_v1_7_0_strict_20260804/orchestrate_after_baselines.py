#!/usr/bin/env python3
"""Finish strict baselines, then launch both HGF seeds without overlap."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


BUNDLE = Path(__file__).resolve().parent
ROOT = BUNDLE.parents[1]
BASELINE_BUNDLE = ROOT / "reproducibility/procedural_topology_hgf_multiseed_v1_20260803"
FINALIZER_PATH = BASELINE_BUNDLE / "recover_and_finalize.py"
RECOVER_MODEL = BASELINE_BUNDLE / "recover_model.py"
HGF_RUNNER = BUNDLE / "run.py"
HGF_VALIDATOR = BUNDLE / "strict_validate.py"
OUTPUT_ROOT = ROOT / "runs/procedural_topology_hgf_v1_6_0_strict_multiseed_20260803"


def _load(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


FINALIZER = _load(FINALIZER_PATH, "baseline_finalizer_for_v160")


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--baseline-attempt-start", type=int, default=92)
    parser.add_argument("--max-baseline-attempts", type=int, default=3)
    parser.add_argument("--max-parallel-models", type=int, default=5)
    return parser.parse_args()


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _active_external_locks() -> list[dict[str, Any]]:
    active: list[dict[str, Any]] = []
    lock_root = FINALIZER.EXTERNAL_RECOVERY_LOCK_ROOT
    if not lock_root.is_dir():
        return active
    for path in sorted(lock_root.glob("*.json")):
        payload = FINALIZER._read(path)
        pid = int(payload.get("pid") or 0)
        if pid <= 0:
            continue
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            continue
        active.append({"path": str(path), **payload})
    return active


def _wait_for_existing_recovery(poll: int) -> None:
    while True:
        active = _active_external_locks()
        _write(
            OUTPUT_ROOT / "BASELINE_WAIT_STATUS.json",
            {"updated_unix": time.time(), "active": active},
        )
        if not active:
            return
        time.sleep(max(5, min(60, poll)))


def _baseline_missing() -> dict[str, dict[int, dict[str, list[str]]]]:
    qids = list(FINALIZER._read(FINALIZER.SELECTION_PATH)["question_ids"])
    result: dict[str, dict[int, dict[str, list[str]]]] = {}
    methods = [method for method in FINALIZER.METHODS if method != FINALIZER.HGF]
    for model in FINALIZER.MODELS:
        by_seed: dict[int, dict[str, list[str]]] = {}
        for seed in FINALIZER.SEEDS:
            candidates = FINALIZER._candidates(model, seed)
            missing: dict[str, list[str]] = {}
            for method in methods:
                ids = [
                    qid
                    for qid in qids
                    if FINALIZER._first_valid(candidates.get((qid, method), [])) is None
                ]
                if ids:
                    missing[method] = ids
            if missing:
                by_seed[seed] = missing
        if by_seed:
            result[model] = by_seed
    return result


def _baseline_count(missing: dict[str, dict[int, dict[str, list[str]]]]) -> int:
    missing_count = sum(
        len(ids)
        for by_seed in missing.values()
        for by_method in by_seed.values()
        for ids in by_method.values()
    )
    return 6000 - missing_count


def _recover_baselines(args: argparse.Namespace) -> None:
    for offset in range(args.max_baseline_attempts + 1):
        missing = _baseline_missing()
        status = {
            "updated_unix": time.time(),
            "strict_valid": _baseline_count(missing),
            "expected": 6000,
            "missing": missing,
        }
        _write(OUTPUT_ROOT / "BASELINE_COMPLETENESS.json", status)
        if not missing:
            return
        if offset == args.max_baseline_attempts:
            raise RuntimeError("baseline completeness remained below 6000")
        attempt = args.baseline_attempt_start + offset
        commands = [
            [
                sys.executable,
                str(RECOVER_MODEL),
                "--model",
                model,
                "--attempt",
                str(attempt),
                "--baselines-only",
            ]
            for model in sorted(missing)
        ]
        records: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=min(3, len(commands))) as executor:
            futures = {
                executor.submit(
                    subprocess.run,
                    command,
                    cwd=ROOT,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    check=False,
                ): command
                for command in commands
            }
            for future in as_completed(futures):
                completed = future.result()
                records.append(
                    {
                        "command": futures[future],
                        "returncode": completed.returncode,
                        "output": completed.stdout,
                    }
                )
        _write(OUTPUT_ROOT / f"BASELINE_RECOVERY_ATTEMPT_{attempt}.json", records)


def _run_seed(seed: int, max_parallel_models: int) -> dict[str, Any]:
    root = OUTPUT_ROOT / f"seed_{seed}"
    command = [
        sys.executable,
        str(HGF_RUNNER),
        "--selection-file",
        str(ROOT / "data/questions/selection.json"),
        "--limit",
        "100",
        "--workers-per-model",
        "20",
        "--max-parallel-models",
        str(max_parallel_models),
        "--run-seed",
        str(seed),
        "--output-root",
        str(root),
    ]
    log = OUTPUT_ROOT / "logs" / f"seed_{seed}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            text=True,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return {"seed": seed, "returncode": completed.returncode, "log": str(log)}


def _validate_seed(seed: int) -> dict[str, Any]:
    root = OUTPUT_ROOT / f"seed_{seed}"
    command = [
        sys.executable,
        str(HGF_VALIDATOR),
        "--run-root",
        str(root),
        "--seed",
        str(seed),
        "--expected-per-model",
        "100",
    ]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return {
        "seed": seed,
        "returncode": completed.returncode,
        "output": completed.stdout,
    }


def main() -> int:
    args = _args()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    _wait_for_existing_recovery(args.poll_seconds)
    _recover_baselines(args)
    if any((OUTPUT_ROOT / f"seed_{seed}").exists() for seed in (1, 2)):
        raise FileExistsError("fresh HGF seed roots are required")

    run_records: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {
            executor.submit(_run_seed, seed, args.max_parallel_models): seed
            for seed in (1, 2)
        }
        for future in as_completed(futures):
            run_records.append(future.result())
            _write(OUTPUT_ROOT / "HGF_RUN_STATUS.json", run_records)

    audit_records = [_validate_seed(seed) for seed in (1, 2)]
    _write(OUTPUT_ROOT / "HGF_AUDIT_STATUS.json", audit_records)
    return int(
        any(record["returncode"] != 0 for record in run_records + audit_records)
    )


if __name__ == "__main__":
    raise SystemExit(main())
