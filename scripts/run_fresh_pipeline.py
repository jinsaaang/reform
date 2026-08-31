"""Build ReFoRM artifacts and forecast from live evidence only.

The input consists of resolved historical questions and later target questions.
This launcher creates a new workspace and never reads the repository's frozen
DAG, Blueprint, exemplar, or evidence databases.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORLDREASONER_ROOT = ROOT / "worldreasoner"
HGF_ROOT = ROOT / "hgf"
HGF_170_ROOT = ROOT / "generation" / "hgf_170"
STAGES = ("dags", "blueprints", "exemplars", "forecast")


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--memory-questions", type=Path, required=True)
    parser.add_argument("--test-questions", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--model", default="google/gemini-2.5-flash-lite")
    parser.add_argument(
        "--python",
        type=Path,
        default=Path(sys.executable),
        help="Python with requirements-generation.txt installed.",
    )
    parser.add_argument("--provider", help="Optional pinned forecast provider.")
    parser.add_argument("--memory-question-id", action="append", default=[])
    parser.add_argument("--test-question-id", action="append", default=[])
    parser.add_argument("--dag-workers", type=int, default=2)
    parser.add_argument("--forecast-workers", type=int, default=1)
    parser.add_argument("--min-evidence-articles", type=int, default=10)
    parser.add_argument("--min-graph-events", type=int, default=8)
    parser.add_argument("--min-graph-depth", type=int, default=3)
    parser.add_argument("--search-query-budget", type=int, default=10)
    parser.add_argument("--max-evidence-rounds", type=int, default=3)
    parser.add_argument(
        "--search-provider",
        choices=("auto", "google_news", "gdelt", "ddgs", "smolagents"),
        default="auto",
    )
    parser.add_argument("--live-query-limit", type=int, default=6)
    parser.add_argument("--live-fetch-limit", type=int, default=12)
    parser.add_argument("--exemplar-workers", type=int, default=4)
    parser.add_argument("--reasoning-effort", default="medium")
    parser.add_argument("--max-output-tokens", type=int, default=16000)
    parser.add_argument("--run-seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after", choices=STAGES, default="forecast")
    return parser.parse_args()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    ids = [str(row.get("id") or "") for row in rows]
    if not rows or any(not value for value in ids):
        raise ValueError(f"empty input or missing question ID: {path}")
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate question IDs: {path}")
    return rows


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _subset(
    rows: list[dict[str, Any]], requested: list[str], label: str
) -> list[dict[str, Any]]:
    if not requested:
        return rows
    by_id = {str(row["id"]): row for row in rows}
    missing = sorted(set(requested) - set(by_id))
    if missing:
        raise ValueError(f"unknown {label} IDs: {missing}")
    return [by_id[value] for value in requested]


def _finance(row: dict[str, Any]) -> dict[str, Any]:
    metadata = row.get("metadata") or {}
    for namespace in ("finance", "finfactorbench", "benchmark"):
        value = metadata.get(namespace)
        if isinstance(value, dict) and value:
            return value
    return metadata if isinstance(metadata, dict) else {}


def _datetime(value: Any) -> datetime:
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


def _cutoff(row: dict[str, Any]) -> datetime:
    finance = _finance(row)
    value = finance.get("forecast_cutoff") or row.get("estimated_start_time")
    if value is None:
        raise ValueError(f"target {row['id']} has no forecast cutoff")
    return _datetime(value)


def _compatible(memory: dict[str, Any], target: dict[str, Any]) -> bool:
    left = _finance(memory)
    right = _finance(target)
    return bool(
        left.get("family_id")
        and left.get("family_id") == right.get("family_id")
        and left.get("target_metric") == right.get("target_metric")
        and _datetime(memory["resolution_date"]) < _cutoff(target)
    )


def _prepare_question_workspace(
    *,
    work_dir: Path,
    memory_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
) -> None:
    unresolved_memory = [
        str(row["id"]) for row in memory_rows if row.get("ground_truth") is None
    ]
    if unresolved_memory:
        raise ValueError(
            "historical questions must be resolved before DAG construction: "
            + ", ".join(unresolved_memory)
        )
    fixed_entries = []
    for target in test_rows:
        candidates = [row for row in memory_rows if _compatible(row, target)]
        if not candidates:
            raise ValueError(
                f"target {target['id']} has no resolved, cutoff-eligible "
                "historical question with the same family and metric"
            )
        chosen = max(candidates, key=lambda row: _datetime(row["resolution_date"]))
        fixed_entries.append(
            {
                "question_id": str(target["id"]),
                "memory_question_id": str(chosen["id"]),
            }
        )

    questions_dir = work_dir / "data" / "questions"
    _write_jsonl(questions_dir / "memory_questions.jsonl", memory_rows)
    _write_jsonl(questions_dir / "test_questions.jsonl", test_rows)
    _write_json(
        questions_dir / "selection.json",
        {
            "selection_rule": "fresh input order",
            "question_ids": [str(row["id"]) for row in test_rows],
            "categories": [str(_finance(row).get("category") or "") for row in test_rows],
        },
    )
    _write_json(
        work_dir / "data" / "memory_bank" / "fixed_exemplar_selection.json",
        {
            "schema_version": "hgf_fixed_memory_selection_v1",
            "question_count": len(test_rows),
            "entries": fixed_entries,
        },
    )
    _write_json(
        work_dir / "configs" / "reproduction.json",
        {
            "schema_version": "fresh_pipeline_root_v1",
            "implementation": "canonical_1_7_0_generation_snapshot",
        },
    )


def _environment(work_dir: Path) -> dict[str, str]:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = os.pathsep.join(
        [str(WORLDREASONER_ROOT), str(HGF_ROOT), *([existing] if existing else [])]
    )
    env["HGF_ROOT"] = str(work_dir)
    return env


def _generator_environment(work_dir: Path) -> dict[str, str]:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = os.pathsep.join(
        [str(HGF_170_ROOT), *([existing] if existing else [])]
    )
    env["HGF_ROOT"] = str(work_dir)
    return env


def _run_command(
    command: list[str], *, cwd: Path, env: dict[str, str], log_path: Path
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            cwd=cwd,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
            text=True,
        )
    if completed.returncode:
        tail = "\n".join(log_path.read_text(encoding="utf-8").splitlines()[-30:])
        raise RuntimeError(
            f"command failed with exit {completed.returncode}: {' '.join(command)}\n{tail}"
        )


def _build_dags(args: argparse.Namespace, work_dir: Path, count: int) -> None:
    workers = max(1, min(args.dag_workers, count))
    env = _environment(work_dir)

    def run_shard(index: int) -> None:
        output = work_dir / "worldreasoner_runs" / f"shard_{index:03d}"
        command = [
            str(args.python),
            str(WORLDREASONER_ROOT / "scripts" / "finance" / "run_dag_sample.py"),
            "--sample",
            str(work_dir / "data" / "questions" / "memory_questions.jsonl"),
            "--output-dir",
            str(output),
            "--model",
            args.model,
            "--limit",
            str(count),
            "--shard-index",
            str(index),
            "--shard-count",
            str(workers),
            "--agent-mode",
            "hybrid",
            "--min-evidence-articles",
            str(args.min_evidence_articles),
            "--search-query-budget",
            str(args.search_query_budget),
            "--search-provider",
            args.search_provider,
            "--browser-concurrency",
            "1",
            "--min-graph-depth",
            str(args.min_graph_depth),
            "--min-graph-events",
            str(args.min_graph_events),
            "--max-evidence-rounds",
            str(args.max_evidence_rounds),
        ]
        _run_command(
            command,
            cwd=work_dir,
            env=env,
            log_path=work_dir / "logs" / f"dag_shard_{index:03d}.log",
        )

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(run_shard, index) for index in range(workers)]
        for future in as_completed(futures):
            future.result()


def _strict_graph(payload: dict[str, Any]) -> bool:
    graph = payload.get("graph") or {}
    return bool(
        (payload.get("evidence") or {}).get("satisfied")
        and graph.get("built")
        and graph.get("satisfied")
        and (graph.get("validation") or {}).get("status") == "pass"
    )


def _materialize_graph_bank(
    work_dir: Path,
    memory_rows: list[dict[str, Any]],
    test_count: int,
) -> None:
    source_by_id: dict[str, Path] = {}
    for path in sorted((work_dir / "worldreasoner_runs").glob("shard_*/graphs/*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        question_id = str((payload.get("question") or {}).get("id") or path.stem)
        if _strict_graph(payload):
            source_by_id[question_id] = path
    expected = {str(row["id"]) for row in memory_rows}
    missing = sorted(expected - set(source_by_id))
    if missing:
        raise RuntimeError(
            f"WorldReasoner did not produce strict DAGs for {len(missing)} questions: "
            + ", ".join(missing)
        )

    entries = []
    for row in memory_rows:
        question_id = str(row["id"])
        payload = json.loads(source_by_id[question_id].read_text(encoding="utf-8"))
        destination = work_dir / "data" / "dags" / question_id / "graph.json"
        _write_json(destination, payload)
        entries.append(
            {
                "question_id": question_id,
                "category": _finance(row).get("category"),
                "validation_status": "pass",
                "graph_path": f"data/dags/{question_id}/graph.json",
            }
        )
    _write_json(
        work_dir / "data" / "memory_bank" / "manifest.json",
        {
            "schema_version": "memory_bank_manifest",
            "memory_question_count": len(memory_rows),
            "test_question_count": test_count,
            "total_validated_count": len(entries),
            "entries": entries,
        },
    )


def _run_hgf_module(
    args: argparse.Namespace,
    work_dir: Path,
    module: str,
    arguments: list[str],
    log_name: str,
) -> None:
    _run_command(
        [str(args.python), "-m", module, *arguments],
        cwd=work_dir,
        env=_generator_environment(work_dir),
        log_path=work_dir / "logs" / log_name,
    )


def _build_blueprints(args: argparse.Namespace, work_dir: Path) -> None:
    _run_hgf_module(
        args,
        work_dir,
        "hgf.build_memory",
        [
            "--memory-manifest",
            str(work_dir / "data" / "memory_bank" / "manifest.json"),
            "--memory-questions",
            str(work_dir / "data" / "questions" / "memory_questions.jsonl"),
            "--output-dir",
            str(work_dir / "artifacts" / "hgf" / "blueprints"),
        ],
        "build_blueprints.log",
    )


def _build_exemplars(args: argparse.Namespace, work_dir: Path) -> None:
    arguments = [
        "--blueprint-root",
        str(work_dir / "artifacts" / "hgf" / "blueprints"),
        "--memory-manifest",
        str(work_dir / "data" / "memory_bank" / "manifest.json"),
        "--memory-questions",
        str(work_dir / "data" / "questions" / "memory_questions.jsonl"),
        "--fixed-selection",
        str(work_dir / "data" / "memory_bank" / "fixed_exemplar_selection.json"),
        "--output-dir",
        str(work_dir / "artifacts" / "hgf" / "exemplars"),
        "--cache-dir",
        str(work_dir / "artifacts" / ".cache" / "hgf_exemplars"),
        "--model",
        args.model,
        "--workers",
        str(args.exemplar_workers),
        "--reasoning-effort",
        args.reasoning_effort,
    ]
    _run_hgf_module(
        args,
        work_dir,
        "hgf.build_exemplars",
        arguments,
        "build_exemplars.log",
    )


def _run_forecast(args: argparse.Namespace, work_dir: Path, test_count: int) -> None:
    command = [
        str(args.python),
        str(ROOT / "scripts" / "run_hgf.py"),
        "--dataset-root",
        str(work_dir),
        "--model",
        args.model,
        "--output-dir",
        str(work_dir / "forecast"),
        "--evidence-mode",
        "live",
        "--live-search-provider",
        args.search_provider,
        "--live-query-limit",
        str(args.live_query_limit),
        "--live-fetch-limit",
        str(args.live_fetch_limit),
        "--workers",
        str(args.forecast_workers),
        "--limit",
        str(test_count),
        "--reasoning-effort",
        args.reasoning_effort,
        "--max-output-tokens",
        str(args.max_output_tokens),
        "--run-seed",
        str(args.run_seed),
    ]
    if args.provider:
        command.extend(["--provider", args.provider])
    if args.resume:
        command.append("--resume")
    _run_command(
        command,
        cwd=work_dir,
        env=_environment(work_dir),
        log_path=work_dir / "logs" / "forecast.log",
    )


def main() -> int:
    args = _args()
    work_dir = args.work_dir.expanduser().resolve()
    # Keep virtual-environment launcher symlinks intact. Resolving the path can
    # silently replace ``.venv/bin/python`` with the system interpreter and
    # discard the environment's installed generation dependencies.
    args.python = Path(os.path.abspath(os.path.expanduser(str(args.python))))
    if not args.python.is_file():
        raise FileNotFoundError(f"generation Python not found: {args.python}")
    if work_dir.exists() and not args.resume:
        raise FileExistsError(f"work directory already exists: {work_dir}")
    work_dir.mkdir(parents=True, exist_ok=args.resume)

    memory_rows = _subset(
        _read_jsonl(args.memory_questions.expanduser().resolve()),
        args.memory_question_id,
        "memory question",
    )
    test_rows = _subset(
        _read_jsonl(args.test_questions.expanduser().resolve()),
        args.test_question_id,
        "test question",
    )
    _prepare_question_workspace(
        work_dir=work_dir,
        memory_rows=memory_rows,
        test_rows=test_rows,
    )
    _write_json(
        work_dir / "fresh_pipeline.json",
        {
            "schema_version": "reform_fresh_pipeline_v1",
            "model": args.model,
            "memory_question_ids": [str(row["id"]) for row in memory_rows],
            "test_question_ids": [str(row["id"]) for row in test_rows],
            "uses_frozen_dags": False,
            "uses_frozen_blueprints": False,
            "uses_frozen_exemplars": False,
            "uses_frozen_evidence_databases": False,
        },
    )

    _build_dags(args, work_dir, len(memory_rows))
    _materialize_graph_bank(work_dir, memory_rows, len(test_rows))
    print(f"DAGs: {len(memory_rows)}/{len(memory_rows)}", flush=True)
    if args.stop_after == "dags":
        return 0

    _build_blueprints(args, work_dir)
    print(f"Blueprints: {len(memory_rows)}/{len(memory_rows)}", flush=True)
    if args.stop_after == "blueprints":
        return 0

    _build_exemplars(args, work_dir)
    print(f"Exemplars: {len(memory_rows)}/{len(memory_rows)}", flush=True)
    if args.stop_after == "exemplars":
        return 0

    _run_forecast(args, work_dir, len(test_rows))
    print(f"Forecasts requested: {len(test_rows)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
