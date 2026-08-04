#!/usr/bin/env python3
"""Run Qwen Plus on 100 cases with canonical v1.7.0 strict HGF.

Defaults are fully specified for one-command reproduction: seed 0, Alibaba,
16,000 maximum output tokens, 20 workers, and at most two fresh trials.  Each
trial is stored permanently.  Only the first independently reportable result
for each question is copied into the canonical 100-case result directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BUNDLE = Path(__file__).resolve().parent
METHOD_NAME = "procedural_topology_hgf_canonical"
METHOD_REVISION = "canonical_v1_7_0_strict"
METHOD_RESULT = f"{METHOD_NAME}.json"
METHOD_FAILURE = f"{METHOD_NAME}.failed.json"
RUN_MODULE = "hgf_e2e_topology_provider_pinned.run"
PARENT_REVISION = "canonical_v1_6_0_strict"
DEFAULT_MODEL = "qwen/qwen-plus-2025-07-28"
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT.parent
    / "results"
    / "qwen_plus_2025_07_28_v170_strict_seed0_20260804_final"
    / "full100"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--provider-only",
        default="alibaba",
        help="Pinned OpenRouter provider tag, or auto-latency.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--selection-file",
        type=Path,
        default=PROJECT_ROOT / "data" / "questions" / "selection.json",
    )
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--final-workers", type=int, default=20)
    parser.add_argument("--run-seed", type=int, default=0)
    parser.add_argument("--max-output-tokens", type=int, default=16_000)
    parser.add_argument("--reasoning-effort", default="medium")
    parser.add_argument(
        "--max-trials",
        type=int,
        default=2,
        help="Maximum fresh trials for unresolved cases; default: 2.",
    )
    parser.add_argument("--backoff-seconds", type=float, default=5.0)
    parser.add_argument("--max-backoff-seconds", type=float, default=5.0)
    parser.add_argument(
        "--attempt-timeout-seconds",
        type=float,
        default=720.0,
        help="Maximum wall time for one trial before harvesting partial successes.",
    )
    parser.add_argument(
        "--enable-native-reasoning",
        action="store_true",
        help="Forward native reasoning parameters (disabled by default for Qwen).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report current clean/pending cases without writing or calling APIs.",
    )
    args = parser.parse_args()
    if args.workers < 1 or args.final_workers < 1:
        parser.error("worker counts must be positive")
    if args.max_trials < 1 or args.max_trials > 2:
        parser.error("--max-trials must be 1 or 2")
    if args.backoff_seconds < 0 or args.max_backoff_seconds < 0:
        parser.error("backoff values must be nonnegative")
    if args.attempt_timeout_seconds <= 0:
        parser.error("--attempt-timeout-seconds must be positive")
    return args


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _method_hashes() -> dict[str, str]:
    method_root = BUNDLE / "method_src" / "hgf_e2e_topology"
    return {
        path.name: _sha256(path)
        for path in sorted(method_root.glob("*.py"))
    }


def _dependency_hashes() -> dict[str, str]:
    return {
        "hgf/boundary.py": _sha256(
            BUNDLE / "hgf_historical_base_src" / "hgf" / "boundary.py"
        ),
        "hgf/exemplar.py": _sha256(
            BUNDLE / "hgf_historical_base_src" / "hgf" / "exemplar.py"
        ),
        "execution/provider_pinned_run.py": _sha256(
            BUNDLE
            / "execution_src"
            / "hgf_e2e_topology_provider_pinned"
            / "run.py"
        ),
    }


def _selection_ids(path: Path) -> list[str]:
    payload = _read_json(path)
    if payload is None or not isinstance(payload.get("question_ids"), list):
        raise ValueError(f"invalid selection file: {path}")
    question_ids = [str(value) for value in payload["question_ids"]]
    if len(question_ids) != len(set(question_ids)):
        raise ValueError("selection contains duplicate question IDs")
    return question_ids


def _case_dir(output_dir: Path, question_id: str) -> Path:
    return output_dir / "cases" / question_id


def _case_is_reportable(output_dir: Path, question_id: str) -> bool:
    case_dir = _case_dir(output_dir, question_id)
    result = _read_json(case_dir / METHOD_RESULT)
    audit = _read_json(case_dir / "prediction_audit.json")
    return bool(
        result
        and result.get("status") == "success"
        and result.get("implementation_revision") == METHOD_REVISION
        and audit
        and (audit.get("completeness") or {}).get("reportable_case") is True
    )


def _partition(output_dir: Path, question_ids: list[str]) -> tuple[list[str], list[str]]:
    clean = [qid for qid in question_ids if _case_is_reportable(output_dir, qid)]
    clean_set = set(clean)
    pending = [qid for qid in question_ids if qid not in clean_set]
    return clean, pending


def _next_attempt_index(attempts_root: Path) -> int:
    indices = []
    if attempts_root.is_dir():
        for path in attempts_root.iterdir():
            if not path.is_dir() or not path.name.startswith("trial_"):
                continue
            try:
                indices.append(int(path.name.removeprefix("trial_")))
            except ValueError:
                continue
    return max(indices, default=0) + 1


def _existing_attempts(attempts_root: Path) -> list[tuple[int, Path]]:
    attempts = []
    if not attempts_root.is_dir():
        return attempts
    for path in attempts_root.iterdir():
        if not path.is_dir() or not path.name.startswith("trial_"):
            continue
        try:
            index = int(path.name.removeprefix("trial_"))
        except ValueError:
            continue
        attempts.append((index, path))
    return sorted(attempts)


def _environment() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    roots = [
        BUNDLE / "method_src",
        BUNDLE / "hgf_historical_base_src",
        BUNDLE / "execution_src",
    ]
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = os.pathsep.join(
        [*(str(path) for path in roots), *([existing] if existing else [])]
    )
    return env


def _run_command(
    *,
    args: argparse.Namespace,
    output_dir: Path,
    question_ids: list[str],
    workers: int,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        RUN_MODULE,
        "--provider-only",
        args.provider_only,
    ]
    if not args.enable_native_reasoning:
        command.append("--disable-native-reasoning")
    command.extend(
        [
            "--model",
            args.model,
            "--questions-dir",
            str(PROJECT_ROOT / "data" / "questions"),
            "--evidence-dir",
            str(PROJECT_ROOT / "data" / "evidence"),
            "--selection-file",
            str(args.selection_file.resolve()),
            "--blueprint-root",
            str(PROJECT_ROOT / "artifacts" / "hgf" / "blueprints"),
            "--exemplar-root",
            str(PROJECT_ROOT / "artifacts" / "hgf" / "exemplars"),
            "--output-dir",
            str(output_dir),
            "--limit",
            str(len(_selection_ids(args.selection_file))),
            "--workers",
            str(workers),
            "--reasoning-effort",
            args.reasoning_effort,
            "--max-output-tokens",
            str(args.max_output_tokens),
            "--run-seed",
            str(args.run_seed),
        ]
    )
    if question_ids:
        command.extend(["--question-ids", *question_ids])
    return command


def _execute(
    command: list[str],
    *,
    log_path: Path,
    timeout_seconds: float,
) -> tuple[int | None, bool, float]:
    started = time.monotonic()
    timed_out = False
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("COMMAND " + json.dumps(command, ensure_ascii=False) + "\n")
        log.flush()
        process = subprocess.Popen(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=_environment(),
            start_new_session=True,
            text=True,
        )
        try:
            return_code = process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGTERM)
            try:
                return_code = process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                return_code = process.wait()
            log.write(f"\nORCHESTRATOR TIMEOUT after {timeout_seconds:.1f}s\n")
        except BaseException:
            # The child starts a new session, so an interrupt delivered to this
            # orchestrator would otherwise leave the model runner behind.
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            raise
    return return_code, timed_out, time.monotonic() - started


def _archive_and_merge(
    *,
    canonical_dir: Path,
    attempt_dir: Path,
    question_ids: list[str],
    history_root: Path,
    attempt_index: int,
) -> list[str]:
    merged = []
    for question_id in question_ids:
        if not _case_is_reportable(attempt_dir, question_id):
            continue
        source = _case_dir(attempt_dir, question_id)
        destination = _case_dir(canonical_dir, question_id)
        incoming = destination.with_name(
            f".{destination.name}.incoming-trial-{attempt_index:04d}"
        )
        if incoming.exists():
            raise FileExistsError(f"stale incoming case directory: {incoming}")
        incoming.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, incoming)
        if destination.exists():
            archived = (
                history_root
                / question_id
                / f"before_trial_{attempt_index:04d}"
            )
            archived.parent.mkdir(parents=True, exist_ok=True)
            if archived.exists():
                raise FileExistsError(f"history destination exists: {archived}")
            shutil.move(str(destination), str(archived))
        os.replace(incoming, destination)
        merged.append(question_id)
    return merged


def _new_status(args: argparse.Namespace, question_ids: list[str]) -> dict[str, Any]:
    return {
        "schema_version": "v170_strict_two_trial_status_v1",
        "method_revision": METHOD_REVISION,
        "parent_method_revision": PARENT_REVISION,
        "method_source_sha256": _method_hashes(),
        "dependency_source_sha256": _dependency_hashes(),
        "model": args.model,
        "provider_only": args.provider_only,
        "run_seed": args.run_seed,
        "reasoning_effort": args.reasoning_effort,
        "max_output_tokens": args.max_output_tokens,
        "workers": args.workers,
        "selection_count": len(question_ids),
        "max_trials": args.max_trials,
        "trials": [],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def _update_counts(
    status: dict[str, Any], output_dir: Path, question_ids: list[str]
) -> tuple[list[str], list[str]]:
    clean, pending = _partition(output_dir, question_ids)
    status["updated_at"] = datetime.now(timezone.utc).isoformat()
    status["clean_count"] = len(clean)
    status["pending_count"] = len(pending)
    status["pending_question_ids"] = pending
    return clean, pending


def _print_state(clean: list[str], pending: list[str]) -> None:
    print(f"AUDIT_CLEAN {len(clean)}  PENDING {len(pending)}", flush=True)
    if pending:
        print("PENDING_IDS " + " ".join(pending), flush=True)


def _write_run_manifest(
    *,
    model_root: Path,
    output_dir: Path,
    status: dict[str, Any],
) -> None:
    overall = (status.get("final_summary") or {}).get("overall") or {}
    results_path = output_dir / "results.json"
    lines = [
        f"# Qwen Plus — canonical v1.7.0 strict, seed {status['run_seed']}",
        "",
        f"- Model: `{status['model']}`",
        f"- Method revision: `{status['method_revision']}`",
        f"- Parent revision: `{status['parent_method_revision']}`",
        f"- Provider policy: `{status['provider_only']}`",
        f"- Run seed: {status['run_seed']}",
        f"- Reasoning effort: `{status['reasoning_effort']}`",
        f"- Maximum output tokens: {status['max_output_tokens']:,}",
        f"- Workers: {status['workers']}",
        f"- Maximum trials: {status['max_trials']}",
        f"- Audit-clean cases: {status.get('clean_count', 0)}/100",
        f"- Accuracy: {overall.get('accuracy')}",
        f"- Multiclass Brier: {overall.get('brier')}",
        f"- NLL: {overall.get('nll')}",
        "",
        "## Saved artifacts",
        "",
        f"- Aggregate: `{results_path}`",
        f"- Per-case prediction/reasoning/audit/raw calls: `{output_dir / 'cases'}`",
        f"- Trial history: `{model_root / 'trials'}`",
        f"- Machine-readable status: `{model_root / 'run_status.json'}`",
    ]
    if results_path.is_file():
        lines.append(f"- `results.json` SHA-256: `{_sha256(results_path)}`")
    _atomic_text(model_root / "RUN_MANIFEST.md", "\n".join(lines) + "\n")


def main() -> int:
    args = _parse_args()
    output_dir = args.output_dir.resolve()
    selection_file = args.selection_file.resolve()
    args.selection_file = selection_file
    if not BUNDLE.is_dir():
        raise FileNotFoundError(f"frozen strict bundle not found: {BUNDLE}")
    question_ids = _selection_ids(selection_file)
    if len(question_ids) != 100:
        raise ValueError(
            f"expected the frozen 100-question selection, found {len(question_ids)}"
        )

    clean, pending = _partition(output_dir, question_ids)
    _print_state(clean, pending)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "method_revision": METHOD_REVISION,
                    "model": args.model,
                    "provider_only": args.provider_only,
                    "run_seed": args.run_seed,
                    "max_output_tokens": args.max_output_tokens,
                    "workers": args.workers,
                    "max_trials": args.max_trials,
                    "bundle": str(BUNDLE),
                    "output_dir": str(output_dir),
                    "clean_count": len(clean),
                    "pending_question_ids": pending,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    model_root = output_dir.parent
    attempts_root = model_root / "trials"
    history_root = model_root / "rejected_history"
    status_path = model_root / "run_status.json"
    existing_status = _read_json(status_path)
    status = existing_status or _new_status(args, question_ids)
    for field, expected in {
        "method_revision": METHOD_REVISION,
        "model": args.model,
        "provider_only": args.provider_only,
        "run_seed": args.run_seed,
        "max_trials": args.max_trials,
    }.items():
        if status.get(field) != expected:
            raise ValueError(
                f"retry status {field}={status.get(field)!r}, expected {expected!r}"
            )

    # Recover valid results from a trial that finished (or partly finished)
    # before a prior orchestrator was interrupted.  This makes resume real and
    # avoids paying for an already clean case again.
    recovered = []
    for attempt_index, attempt_dir in _existing_attempts(attempts_root):
        _, pending_now = _partition(output_dir, question_ids)
        available = [
            qid
            for qid in pending_now
            if _case_is_reportable(attempt_dir, qid)
        ]
        if not available:
            continue
        merged = _archive_and_merge(
            canonical_dir=output_dir,
            attempt_dir=attempt_dir,
            question_ids=available,
            history_root=history_root,
            attempt_index=attempt_index,
        )
        recovered.extend(
            {"trial_index": attempt_index, "question_id": qid}
            for qid in merged
        )
    if recovered:
        status.setdefault("recovered_from_existing_trials", []).extend(recovered)
    clean, pending = _update_counts(status, output_dir, question_ids)
    _atomic_json(status_path, status)

    completed_trials = len(status.get("trials") or [])
    while pending and completed_trials < args.max_trials:
        attempt_index = _next_attempt_index(attempts_root)
        attempt_dir = attempts_root / f"trial_{attempt_index:04d}"
        attempt_dir.mkdir(parents=True, exist_ok=False)
        attempted_ids = list(pending)
        command = _run_command(
            args=args,
            output_dir=attempt_dir,
            question_ids=attempted_ids,
            workers=args.workers,
        )
        print(
            f"TRIAL {attempt_index:04d} START cases={len(attempted_ids)} "
            f"workers={args.workers}",
            flush=True,
        )
        return_code, timed_out, elapsed = _execute(
            command,
            log_path=attempt_dir / "run.log",
            timeout_seconds=args.attempt_timeout_seconds,
        )
        reportable_in_attempt = [
            qid
            for qid in attempted_ids
            if _case_is_reportable(attempt_dir, qid)
        ]
        merged = _archive_and_merge(
            canonical_dir=output_dir,
            attempt_dir=attempt_dir,
            question_ids=reportable_in_attempt,
            history_root=history_root,
            attempt_index=attempt_index,
        )
        clean, pending = _update_counts(status, output_dir, question_ids)
        status.setdefault("trials", []).append(
            {
                "trial_index": attempt_index,
                "trial_dir": str(attempt_dir),
                "started_with_count": len(attempted_ids),
                "started_with_question_ids": attempted_ids,
                "return_code": return_code,
                "timed_out": timed_out,
                "elapsed_seconds": elapsed,
                "reportable_count": len(reportable_in_attempt),
                "merged_question_ids": merged,
                "clean_count_after_merge": len(clean),
                "pending_count_after_merge": len(pending),
            }
        )
        _atomic_json(status_path, status)
        completed_trials += 1
        print(
            f"TRIAL {attempt_index:04d} END rc={return_code} "
            f"timeout={timed_out} merged={len(merged)} "
            f"clean={len(clean)} pending={len(pending)}",
            flush=True,
        )
        if pending and completed_trials < args.max_trials:
            delay = min(
                args.max_backoff_seconds,
                args.backoff_seconds * (2 ** min(completed_trials - 1, 8)),
            )
            if delay:
                print(f"BACKOFF {delay:.1f}s", flush=True)
                time.sleep(delay)

    if pending:
        status["complete"] = False
        _atomic_json(status_path, status)
        _write_run_manifest(
            model_root=model_root,
            output_dir=output_dir,
            status=status,
        )
        _print_state(clean, pending)
        print(
            "STOPPED before 100 clean cases because --max-trials was reached",
            file=sys.stderr,
        )
        return 2

    print("REBUILD aggregate from 100 cached audit-clean cases", flush=True)
    final_command = _run_command(
        args=args,
        output_dir=output_dir,
        question_ids=[],
        workers=args.final_workers,
    )
    final_rc, final_timed_out, final_elapsed = _execute(
        final_command,
        log_path=model_root / "final_cache_rebuild.log",
        timeout_seconds=args.attempt_timeout_seconds,
    )
    clean, pending = _update_counts(status, output_dir, question_ids)
    status["final_cache_rebuild"] = {
        "return_code": final_rc,
        "timed_out": final_timed_out,
        "elapsed_seconds": final_elapsed,
        "log": str(model_root / "final_cache_rebuild.log"),
    }
    results = _read_json(output_dir / "results.json") or {}
    status["final_summary"] = results.get("summary")
    status["complete"] = final_rc == 0 and not final_timed_out and not pending
    _atomic_json(status_path, status)
    _write_run_manifest(
        model_root=model_root,
        output_dir=output_dir,
        status=status,
    )
    _print_state(clean, pending)
    print(json.dumps(results.get("summary", {}), ensure_ascii=False, indent=2))
    return 0 if status["complete"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
