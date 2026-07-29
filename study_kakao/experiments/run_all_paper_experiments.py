"""Sequentially execute every runnable experiment in experiments.md.

The master process enforces a single active four-worker child.  Use
``--dry-run`` to inspect the complete command plan without making API calls.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hgf.experiment_common import PAPER_MODELS, model_slug, read_json  # noqa: E402
from hgf.experiment_judge import DEFAULT_JUDGE_MODEL  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("all", "main", "ablation", "topk", "judge", "analysis"),
        default="all",
    )
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--judge-passes", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-attempts", type=int, default=12)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/paper_experiments_v27"),
    )
    parser.add_argument(
        "--additional-exemplar-dir",
        type=Path,
        action="append",
        default=[],
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _run(command: list[str], environment: dict[str, str]) -> None:
    process = subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        check=False,
    )
    if process.returncode:
        raise SystemExit(
            f"command failed with exit code {process.returncode}: {command}"
        )


def _complete(path: Path, expected: int) -> bool:
    if not path.is_file():
        return False
    summary = read_json(path).get("summary", {})
    return (
        int(summary.get("completed_runs") or 0) == expected
        and int(summary.get("failed_runs") or 0) == 0
    )


def _complete_judge(path: Path, expected: int) -> bool:
    if not path.is_file():
        return False
    summary = read_json(path).get("summary", {})
    return (
        int(summary.get("successful_judgments") or 0) == expected
        and int(summary.get("failed_judgments") or 0) == 0
    )


def _retry_experiment(
    command: list[str],
    *,
    result_path: Path,
    expected: int,
    attempts: int,
    environment: dict[str, str],
) -> None:
    if _complete(result_path, expected):
        return
    for _ in range(attempts):
        _run(command, environment)
        if _complete(result_path, expected):
            return
    raise RuntimeError(f"incomplete result after retries: {result_path}")


def _plan(args: argparse.Namespace) -> dict[str, list[list[str]]]:
    output = args.output_dir.resolve()
    main_root = output / "main_table"
    ablation_root = output / "ablation"
    topk_root = output / "topk"
    judge_root = output / "reasoning_judge"
    reports = output / "reports"
    commands: dict[str, list[list[str]]] = {
        "main": [
            [
                sys.executable,
                str(ROOT / "experiments" / "run_paper_main_table.py"),
                "--workers",
                "4",
                "--repeats",
                "3",
                "--output-dir",
                str(main_root),
                "--max-attempts",
                str(args.max_attempts),
            ]
        ],
        "ablation": [],
        "topk": [],
        "judge": [],
        "analysis": [],
    }
    for repeat in range(1, 4):
        commands["ablation"].append(
            [
                sys.executable,
                "-m",
                "hgf.experiment_ablation",
                "--workers",
                "4",
                "--limit",
                "100",
                "--run-seed",
                str(repeat),
                "--output-dir",
                str(ablation_root / f"repeat_{repeat}"),
            ]
        )
        topk_command = [
            sys.executable,
            "-m",
            "hgf.experiment_topk",
            "--workers",
            "4",
            "--limit",
            "100",
            "--run-seed",
            str(repeat),
            "--output-dir",
            str(topk_root / f"repeat_{repeat}"),
        ]
        for path in args.additional_exemplar_dir:
            topk_command.extend(["--additional-exemplar-dir", str(path.resolve())])
        commands["topk"].append(topk_command)
    ablation_results = [
        ablation_root / f"repeat_{repeat}" / "results.json"
        for repeat in range(1, 4)
    ]
    topk_results = [
        topk_root / f"repeat_{repeat}" / "results.json"
        for repeat in range(1, 4)
    ]
    judge_command = [
        sys.executable,
        "-m",
        "hgf.experiment_judge",
        "--judge-model",
        args.judge_model,
        "--passes",
        str(args.judge_passes),
        "--workers",
        "4",
        "--output-dir",
        str(judge_root),
    ]
    for path in ablation_results:
        judge_command.extend(["--forecast-results", str(path)])
    commands["judge"].append(judge_command)
    commands["analysis"].extend(
        [
            [
                sys.executable,
                str(ROOT / "experiments" / "analyze_ablation.py"),
                *[str(path) for path in ablation_results],
                "--output-dir",
                str(reports / "ablation"),
            ],
            [
                sys.executable,
                str(ROOT / "experiments" / "analyze_topk.py"),
                *[str(path) for path in topk_results],
                "--output-dir",
                str(reports / "topk"),
            ],
        ]
    )
    commands["analysis"].append(
        [
            sys.executable,
            str(ROOT / "experiments" / "analyze_reasoning_judge.py"),
            str(judge_root / "results.json"),
            "--output-dir",
            str(reports / "reasoning_judge"),
        ]
    )
    gemini_paths = [
        main_root
        / model_slug(PAPER_MODELS[0])
        / f"repeat_{repeat}"
        / "results.json"
        for repeat in range(1, 4)
    ]
    commands["analysis"].append(
        [
            sys.executable,
            str(ROOT / "experiments" / "build_case_studies.py"),
            *[str(path) for path in gemini_paths],
            "--count",
            "3",
            "--output-dir",
            str(reports / "case_studies"),
        ]
    )
    return commands


def main() -> None:
    args = _parse_args()
    if args.workers != 4:
        raise ValueError("experiments.md fixes workers at 4")
    commands = _plan(args)
    if args.dry_run:
        selected = (
            commands
            if args.stage == "all"
            else {args.stage: commands[args.stage]}
        )
        print(json.dumps(selected, ensure_ascii=False, indent=2))
        return
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(SRC)
    environment["PYTHONDWRITEBYTECODE"] = "1"
    stages = (
        ("main", "ablation", "topk", "judge", "analysis")
        if args.stage == "all"
        else (args.stage,)
    )
    for stage in stages:
        for command in commands[stage]:
            if stage == "ablation":
                output_index = command.index("--output-dir") + 1
                result_path = Path(command[output_index]) / "results.json"
                _retry_experiment(
                    command,
                    result_path=result_path,
                    expected=500,
                    attempts=args.max_attempts,
                    environment=environment,
                )
            elif stage == "topk":
                output_index = command.index("--output-dir") + 1
                result_path = Path(command[output_index]) / "results.json"
                _retry_experiment(
                    command,
                    result_path=result_path,
                    expected=400,
                    attempts=args.max_attempts,
                    environment=environment,
                )
            elif stage == "judge":
                result_path = args.output_dir.resolve() / "reasoning_judge" / "results.json"
                expected = 300 * args.judge_passes
                if _complete_judge(result_path, expected):
                    continue
                for _ in range(args.max_attempts):
                    _run(command, environment)
                    if _complete_judge(result_path, expected):
                        break
                else:
                    raise RuntimeError(
                        f"reasoning judge remains incomplete: {result_path}"
                    )
            else:
                _run(command, environment)


if __name__ == "__main__":
    main()
