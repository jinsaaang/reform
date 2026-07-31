#!/usr/bin/env python3
"""Adapt ForecastBench Mixed 300 to the frozen HGF-v1 runtime contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from hgf.exemplar import _rerank_current_evidence  # noqa: E402
from hgf.forecast_core import _resolve_evidence  # noqa: E402
from hgf.memory_bank import load_final_memory_bank  # noqa: E402
from hgf.memory_retrieval import select_relevant_blueprints  # noqa: E402
from hgf.question_io import read_questions, resolve_forecast_cutoff  # noqa: E402


DATA_DIR = ROOT / "data" / "forecastbench_mixed_300"
DAG_DIR = ROOT / "runs" / "forecastbench_mixed_300_dags"
RUN_DIR = ROOT / "runs" / "forecastbench_hgf_v1_gemini25flashlite_medium"


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--dag-dir", type=Path, default=DAG_DIR)
    parser.add_argument("--run-dir", type=Path, default=RUN_DIR)
    parser.add_argument("--finalize-exemplars", action="store_true")
    return parser.parse_args()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _timestamp(date: str, *, end: bool = False) -> str:
    if "T" in date:
        return date
    return date + ("T23:59:59Z" if end else "T00:00:00Z")


def _safe_test_id(row: dict[str, Any]) -> str:
    source = str(row["source"]).lower().replace("-", "_")
    digest = hashlib.sha256(str(row["task_id"]).encode()).hexdigest()[:16]
    return f"fbtest_{source}_{digest}"


def _hgf_metadata(row: dict[str, Any]) -> dict[str, Any]:
    source = str(row.get("source") or "forecastbench")
    question_class = str(row.get("question_type") or "binary")
    return {
        "dataset_version": "forecastbench_mixed_300_hgf_v1",
        "family_id": f"{source}::{row.get('source_question_id')}",
        "category": source,
        "original_domain": question_class,
        "subdomain": question_class,
        "entity": source,
        "region": "global",
        "target_metric": str(
            row.get("question_template")
            or row.get("question_text")
            or "binary event resolution"
        ),
        "target_period": str(row["resolution_date"]),
        "forecast_cutoff": _timestamp(str(row["forecast_due_date"])),
        "forecast_date_options": [str(row["forecast_due_date"])],
        "resolution_available_at": _timestamp(
            str(row["resolution_date"]),
            end=True,
        ),
        "source_type": source,
        "source_series_id": str(row.get("source_question_id") or ""),
        "comparison_rule": str(row.get("resolution_criteria") or ""),
        "change_unit": "binary_probability",
        "question_class": question_class,
    }


def _context(row: dict[str, Any]) -> str:
    parts = []
    for label, key in (
        ("Source", "source_intro"),
        ("Background", "background"),
        ("Forecast-time snapshot", "freeze_datetime_value_explanation"),
    ):
        value = str(row.get(key) or "").strip()
        if value:
            parts.append(f"{label}:\n{value}")
    if row.get("freeze_datetime_value") not in (None, "", "N/A"):
        parts.append(
            "Snapshot details:\n"
            f"Date: {row.get('freeze_datetime')}\n"
            f"Value: {row.get('freeze_datetime_value')}"
        )
    market = {
        key: value
        for key, value in row.items()
        if key.startswith("market_info_") and value not in (None, "", "N/A")
    }
    if market:
        parts.append(
            "Forecast-time market information:\n"
            + json.dumps(market, ensure_ascii=False)
        )
    if row.get("url"):
        parts.append(f"Source URL: {row['url']}")
    return "\n\n".join(parts)


def _convert_memory(
    worldreasoner: list[dict[str, Any]],
    public: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    source_rows = {str(row["source_question_id"]): row for row in public}
    converted = []
    for row in worldreasoner:
        payload = dict(row)
        forecastbench = (payload.get("metadata") or {})["forecastbench"]
        source = source_rows[str(forecastbench["source_question_id"])]
        metadata = dict(payload.get("metadata") or {})
        metadata["finance"] = _hgf_metadata(source)
        payload["metadata"] = metadata
        payload["question_type"] = "binary"
        converted.append(payload)
    return converted


def _convert_test(
    public: list[dict[str, Any]],
    answers: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    converted = []
    id_map = {}
    for row in public:
        task_id = str(row["task_id"])
        question_id = _safe_test_id(row)
        id_map[task_id] = question_id
        converted.append(
            {
                "id": question_id,
                "question_text": row["question_text"],
                "question_type": "binary",
                "domain": str(row.get("source") or "general"),
                "source": "forecastbench",
                "difficulty": 3,
                "resolution_date": _timestamp(
                    str(row["resolution_date"]),
                    end=True,
                ),
                "estimated_start_time": _timestamp(
                    str(row["forecast_due_date"])
                ),
                "ground_truth": answers[task_id]["ground_truth"],
                "context": _context(row),
                "resolution_criteria": row.get("resolution_criteria"),
                "options": ["No", "Yes"],
                "created_at": _timestamp(str(row["forecast_due_date"])),
                "metadata": {
                    "finance": _hgf_metadata(row),
                    "forecastbench": {
                        "task_id": task_id,
                        "source": row.get("source"),
                        "source_question_id": row.get("source_question_id"),
                        "question_class": row.get("question_type"),
                        "origin": row.get("origin"),
                        "url": row.get("url"),
                    },
                    "benchmark_private": {
                        "answer_joined_for_scoring_only": True,
                    },
                },
            }
        )
    return converted, id_map


def _articles(row: dict[str, Any], question_id: str) -> list[dict[str, str]]:
    cutoff = datetime.fromisoformat(
        _timestamp(str(row["forecast_due_date"])).replace("Z", "+00:00")
    )
    fallback = cutoff - timedelta(minutes=1)
    articles = [
        {
            "id": f"art_{question_id}_question",
            "title": "ForecastBench public question and resolution criteria",
            "source": "ForecastBench",
            "published_date": fallback.isoformat(),
            "content": (
                str(row.get("question_text") or "")
                + "\n"
                + str(row.get("resolution_criteria") or "")
            ),
        }
    ]
    background = "\n\n".join(
        str(row.get(key) or "").strip()
        for key in ("source_intro", "background")
        if str(row.get(key) or "").strip()
    )
    if background:
        articles.append(
            {
                "id": f"art_{question_id}_background",
                "title": f"{row.get('source')} source definition and background",
                "source": str(row.get("source") or "ForecastBench"),
                "published_date": (fallback - timedelta(minutes=1)).isoformat(),
                "content": background,
            }
        )
    snapshot = "\n".join(
        f"{key}: {row.get(key)}"
        for key in (
            "freeze_datetime",
            "freeze_datetime_value",
            "freeze_datetime_value_explanation",
        )
        if row.get(key) not in (None, "", "N/A")
    )
    if snapshot:
        try:
            published = datetime.fromisoformat(
                str(row["freeze_datetime"]).replace("Z", "+00:00")
            )
        except (KeyError, ValueError):
            published = fallback - timedelta(minutes=2)
        if published.tzinfo is None:
            published = published.replace(tzinfo=timezone.utc)
        published = min(published, fallback - timedelta(minutes=2))
        articles.append(
            {
                "id": f"art_{question_id}_snapshot",
                "title": "Forecast-time source snapshot",
                "source": str(row.get("source") or "ForecastBench"),
                "published_date": published.isoformat(),
                "content": snapshot,
            }
        )
    market = {
        key: value
        for key, value in row.items()
        if key.startswith("market_info_") and value not in (None, "", "N/A")
    }
    if market:
        articles.append(
            {
                "id": f"art_{question_id}_market",
                "title": "Forecast-time market metadata",
                "source": str(row.get("source") or "ForecastBench"),
                "published_date": (fallback - timedelta(minutes=3)).isoformat(),
                "content": json.dumps(market, ensure_ascii=False),
            }
        )
    return articles


def _write_evidence(
    path: Path,
    question_id: str,
    articles: list[dict[str, str]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as connection:
        connection.execute("DROP TABLE IF EXISTS articles")
        connection.execute(
            """
            CREATE TABLE articles (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                source TEXT NOT NULL,
                published_date TEXT NOT NULL,
                content TEXT,
                collected_for_question_id TEXT NOT NULL
            )
            """
        )
        connection.executemany(
            "INSERT INTO articles VALUES (?, ?, ?, ?, ?, ?)",
            [
                (
                    item["id"],
                    item["title"],
                    item["source"],
                    item["published_date"],
                    item["content"],
                    question_id,
                )
                for item in articles
            ],
        )


def _manifest(
    dag_dir: Path,
    run_dir: Path,
    memory: list[dict[str, Any]],
) -> Path:
    entries = []
    for row in memory:
        question_id = str(row["id"])
        graph = dag_dir / "memory_dags" / "graphs" / f"{question_id}.json"
        blueprint = dag_dir / "memory_refinements" / f"{question_id}.json"
        if not graph.is_file() or not blueprint.is_file():
            raise FileNotFoundError(f"missing graph/blueprint: {question_id}")
        entries.append(
            {
                "question_id": question_id,
                "graph_path": str(graph.resolve()),
                "guidance_path": str(blueprint.resolve()),
            }
        )
    path = run_dir / "memory_bank" / "manifest.json"
    _write_json(
        path,
        {
            "schema_version": "forecastbench_hgf_memory_bank_v1",
            "count": len(entries),
            "entries": entries,
        },
    )
    return path


def _finalize_exemplars(run_dir: Path) -> dict[str, Any]:
    questions_dir = run_dir / "questions"
    memory_questions = {
        str(item.id): item
        for item in read_questions(questions_dir / "memory_questions.jsonl")
    }
    tests = read_questions(questions_dir / "test_questions.jsonl")
    graphs, blueprints = load_final_memory_bank(
        run_dir / "memory_bank" / "manifest.json",
        memory_questions,
    )
    del graphs
    exemplar_dir = run_dir / "exemplars" / "memory"
    exemplars = {}
    for path in exemplar_dir.glob("*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload.get("worked_exemplar"), dict):
            exemplars[str(payload["memory_question_id"])] = payload[
                "worked_exemplar"
            ]

    retrieval = {}
    for question in tests:
        cutoff, _ = resolve_forecast_cutoff(question)
        _, candidates = _resolve_evidence(
            run_dir / "evidence",
            question,
            cutoff,
            80,
        )
        evidence = _rerank_current_evidence(question, candidates, limit=20)
        selected = select_relevant_blueprints(
            blueprints,
            memory_questions,
            question,
            limit=1,
            evidence=evidence,
        )
        if not selected:
            raise ValueError(f"no usable memory for {question.id}")
        retrieval[str(question.id)] = str(selected[0]["question_id"])

    required_ids = sorted(set(retrieval.values()))
    required_path = run_dir / "exemplars" / "required_memory_ids.txt"
    required_path.parent.mkdir(parents=True, exist_ok=True)
    required_path.write_text("\n".join(required_ids) + "\n", encoding="utf-8")
    _write_json(run_dir / "exemplars" / "retrieval_map.json", retrieval)
    missing = sorted(set(required_ids) - set(exemplars))
    if missing:
        return {
            "status": "needs_memory_exemplars",
            "required_unique_memory_count": len(required_ids),
            "missing_memory_exemplar_count": len(missing),
            "required_ids_path": str(required_path.resolve()),
        }

    cases = run_dir / "exemplars" / "cases"
    for question_id, memory_id in retrieval.items():
        _write_json(
            cases / f"{question_id}.json",
            {
                "schema_version": "forecastbench_fixed_exemplar_case_v1",
                "status": "success",
                "question_id": question_id,
                "retrieved_memory_question_id": memory_id,
                "worked_exemplar": exemplars[memory_id],
            },
        )
    return {
        "status": "complete",
        "test_case_count": len(retrieval),
        "unique_memory_exemplar_count": len(required_ids),
    }


def main() -> None:
    args = _args()
    data_dir = args.data_dir.resolve()
    dag_dir = args.dag_dir.resolve()
    run_dir = args.run_dir.resolve()
    questions_dir = run_dir / "questions"

    public_memory = _read_jsonl(data_dir / "memory_questions.jsonl")
    worldreasoner = _read_jsonl(
        data_dir / "worldreasoner_memory_questions.jsonl"
    )
    public_test = _read_jsonl(data_dir / "test_questions.jsonl")
    answers = {
        str(row["task_id"]): row
        for row in _read_jsonl(data_dir / "test_answers.jsonl")
    }
    memory = _convert_memory(worldreasoner, public_memory)
    tests, id_map = _convert_test(public_test, answers)
    _write_jsonl(questions_dir / "memory_questions.jsonl", memory)
    _write_jsonl(questions_dir / "test_questions.jsonl", tests)
    _write_json(
        questions_dir / "selection.json",
        {
            "selection_rule": "ForecastBench fixed test order",
            "question_ids": [row["id"] for row in tests],
        },
    )
    _write_json(questions_dir / "test_id_map.json", id_map)
    for source, test in zip(public_test, tests, strict=True):
        _write_evidence(
            run_dir / "evidence" / "e1" / f"{test['id']}.sqlite",
            str(test["id"]),
            _articles(source, str(test["id"])),
        )
    manifest = _manifest(dag_dir, run_dir, memory)
    result: dict[str, Any] = {
        "status": "prepared",
        "memory_count": len(memory),
        "test_count": len(tests),
        "evidence_db_count": len(
            list((run_dir / "evidence" / "e1").glob("*.sqlite"))
        ),
        "memory_bank_manifest": str(manifest.resolve()),
    }
    if args.finalize_exemplars:
        result["exemplars"] = _finalize_exemplars(run_dir)
    _write_json(run_dir / "preparation_summary.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
