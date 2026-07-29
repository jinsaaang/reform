"""Run the experiments.md main table with sequential model lanes.

This wrapper never changes the frozen forecaster.  It invokes
``hgf.baselines`` as a subprocess, retries incomplete matrices by relying on
the existing per-case checkpoints, and aggregates the three repetitions.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hgf.experiment_common import (  # noqa: E402
    PAPER_METHODS,
    PAPER_MODELS,
    complete_main_table,
    model_slug,
    provenance_snapshot,
    read_json,
    utc_now,
    write_json,
)
from hgf.experiment_stats import (  # noqa: E402
    aggregate_main_table,
    write_main_table_reports,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=list(PAPER_MODELS))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-attempts", type=int, default=12)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/paper_main_table_v27"),
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=("minimal", "low", "medium", "high", "xhigh", "max"),
    )
    parser.add_argument("--max-output-tokens", type=int)
    parser.add_argument("--bootstrap-iterations", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=27)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--only-analyze", action="store_true")
    return parser.parse_args()


def _selection_ids() -> list[str]:
    payload = read_json(ROOT / "data" / "questions" / "selection.json")
    return [str(value) for value in payload["question_ids"]]


def _command(
    *,
    model: str,
    repeat: int,
    workers: int,
    output_dir: Path,
    reasoning_effort: str | None,
    max_output_tokens: int | None,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "hgf.baselines",
        "--model",
        model,
        "--limit",
        "100",
        "--workers",
        str(workers),
        "--run-seed",
        str(repeat),
        "--output-dir",
        str(output_dir),
    ]
    if reasoning_effort is not None:
        command.extend(["--reasoning-effort", reasoning_effort])
    if max_output_tokens is not None:
        command.extend(["--max-output-tokens", str(max_output_tokens)])
    return command


def _write_status(path: Path, **values: object) -> None:
    write_json(
        path,
        {
            "schema_version": "hgf_main_table_lane_status_v1",
            "updated_at_utc": utc_now(),
            **values,
        },
    )


def _analyze(
    model_runs: dict[str, list[Path]],
    *,
    output_root: Path,
    iterations: int,
    seed: int,
) -> None:
    aggregate = aggregate_main_table(
        model_runs,
        bootstrap_iterations=iterations,
        seed=seed,
    )
    write_main_table_reports(aggregate, output_root / "reports")


def main() -> None:
    args = parse_args()
    if args.workers != 4:
        raise ValueError("experiments.md fixes the global worker count at 4")
    if args.repeats != 3:
        raise ValueError("experiments.md requires exactly three repetitions")
    output_root = (
        args.output_dir
        if args.output_dir.is_absolute()
        else (ROOT / args.output_dir)
    ).resolve()
    question_ids = _selection_ids()
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(SRC)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"

    model_runs: dict[str, list[Path]] = {}
    plan = []
    for model in args.models:
        paths = []
        for repeat in range(1, args.repeats + 1):
            run_dir = output_root / model_slug(model) / f"repeat_{repeat}"
            paths.append(run_dir / "results.json")
            plan.append(
                _command(
                    model=model,
                    repeat=repeat,
                    workers=args.workers,
                    output_dir=run_dir,
                    reasoning_effort=args.reasoning_effort,
                    max_output_tokens=args.max_output_tokens,
                )
            )
        model_runs[model] = paths

    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return
    if args.only_analyze:
        _analyze(
            model_runs,
            output_root=output_root,
            iterations=args.bootstrap_iterations,
            seed=args.bootstrap_seed,
        )
        return

    suite_started = utc_now()
    write_json(
        output_root / "suite_provenance.json",
        provenance_snapshot(
            root=ROOT,
            config_paths=(Path("configs/experiments_v27.json"),),
            extra={
                "suite": "main_table",
                "models": args.models,
                "repeats": args.repeats,
                "workers": args.workers,
                "started_at_utc": suite_started,
                "execution_policy": "models_and_repeats_sequential",
            },
        ),
    )
    for model in args.models:
        # Deliberately sequential: at most one four-worker process exists.
        for repeat, result_path in enumerate(model_runs[model], start=1):
            run_dir = result_path.parent
            status_path = run_dir.parent / f"repeat_{repeat}.status.json"
            complete, errors = complete_main_table(
                result_path,
                question_ids=question_ids,
                methods=PAPER_METHODS,
            )
            if complete:
                _write_status(
                    status_path,
                    state="complete",
                    model=model,
                    repeat=repeat,
                    attempts=0,
                    reused=True,
                )
                continue
            for attempt in range(1, args.max_attempts + 1):
                command = _command(
                    model=model,
                    repeat=repeat,
                    workers=args.workers,
                    output_dir=run_dir,
                    reasoning_effort=args.reasoning_effort,
                    max_output_tokens=args.max_output_tokens,
                )
                _write_status(
                    status_path,
                    state="running",
                    model=model,
                    repeat=repeat,
                    attempt=attempt,
                    command=command,
                    previous_validation_errors=errors,
                )
                started = time.monotonic()
                log_path = run_dir.parent / f"repeat_{repeat}.log"
                log_path.parent.mkdir(parents=True, exist_ok=True)
                with log_path.open("a", encoding="utf-8") as log:
                    process = subprocess.run(
                        command,
                        cwd=ROOT,
                        env=environment,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=False,
                    )
                complete, errors = complete_main_table(
                    result_path,
                    question_ids=question_ids,
                    methods=PAPER_METHODS,
                )
                if complete:
                    _write_status(
                        status_path,
                        state="complete",
                        model=model,
                        repeat=repeat,
                        attempt=attempt,
                        exit_code=process.returncode,
                        elapsed_seconds=time.monotonic() - started,
                        reused=False,
                    )
                    break
                _write_status(
                    status_path,
                    state="retrying",
                    model=model,
                    repeat=repeat,
                    attempt=attempt,
                    exit_code=process.returncode,
                    elapsed_seconds=time.monotonic() - started,
                    validation_errors=errors,
                )
            else:
                raise RuntimeError(
                    f"{model} repeat {repeat} remains incomplete: {errors}"
                )

    _analyze(
        model_runs,
        output_root=output_root,
        iterations=args.bootstrap_iterations,
        seed=args.bootstrap_seed,
    )
    provenance = read_json(output_root / "suite_provenance.json")
    provenance["completed_at_utc"] = utc_now()
    write_json(output_root / "suite_provenance.json", provenance)


if __name__ == "__main__":
    main()

