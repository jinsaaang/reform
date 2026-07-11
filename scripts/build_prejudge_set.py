#!/usr/bin/env python3
"""Build a unified pre-judge dataset from public forecasting datasets.

Outputs:
  data/prejudge/judge_units.jsonl
    One row per item that should be sent to the finance/econ LLM judge.
    This intentionally excludes answers and post-resolution rationales.

  data/prejudge/task_index.jsonl
    One row per forecasting task/snapshot. This keeps answers and dates for
    later evaluation, and links each task to a judge row via judge_uid.

  data/prejudge/manifest.json
    Row counts and source-level summary.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


OUT_DIR = Path("data/prejudge")
RAW_CACHE_DIR = OUT_DIR / "raw_cache"
JUDGE_UNITS_PATH = OUT_DIR / "judge_units.jsonl"
TASK_INDEX_PATH = OUT_DIR / "task_index.jsonl"
UNIFIED_TASKS_PATH = OUT_DIR / "unified_tasks.jsonl"
MANIFEST_PATH = OUT_DIR / "manifest.json"

HF_ROWS_ENDPOINT = "https://datasets-server.huggingface.co/rows"
HF_INFO_ENDPOINT = "https://datasets-server.huggingface.co/info"

DAILY_ORACLE_FILES = {
    "tf": "tf_questions_2020-01-01_2026-07-09.csv",
    "mc": "mc_questions_2020-01-01_2026-07-09.csv",
}


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def normalize_text(value: str | None) -> str:
    return re.sub(r"\s+", " ", (value or "").strip()).lower()


def stable_hash(value: str, length: int = 16) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:length]


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def request_json(url: str, retries: int = 5) -> Any:
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                return json.load(response)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = exc
            sleep_seconds = min(2**attempt, 20)
            log(f"retry {attempt + 1}/{retries} after error: {exc}")
            time.sleep(sleep_seconds)
    raise RuntimeError(f"failed to fetch {url}") from last_error


def request_text(url: str, retries: int = 5) -> str:
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=120) as response:
                return response.read().decode("utf-8", errors="replace")
        except (urllib.error.URLError, TimeoutError) as exc:
            last_error = exc
            sleep_seconds = min(2**attempt, 20)
            log(f"retry {attempt + 1}/{retries} after error: {exc}")
            time.sleep(sleep_seconds)
    raise RuntimeError(f"failed to fetch {url}") from last_error


def download_file(url: str, path: Path, retries: int = 5) -> Path:
    if path.exists() and path.stat().st_size > 0:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            log(f"downloading {url}")
            with urllib.request.urlopen(url, timeout=300) as response, path.open("wb") as handle:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)
            return path
        except (urllib.error.URLError, TimeoutError) as exc:
            last_error = exc
            if path.exists():
                path.unlink()
            sleep_seconds = min(2**attempt, 20)
            log(f"retry {attempt + 1}/{retries} after error: {exc}")
            time.sleep(sleep_seconds)
    raise RuntimeError(f"failed to download {url}") from last_error


def hf_repo_files(dataset: str) -> list[str]:
    payload = request_json(f"https://huggingface.co/api/datasets/{dataset}")
    return [item["rfilename"] for item in payload.get("siblings", [])]


def hf_dataset_info(dataset: str) -> dict[str, Any]:
    query = urllib.parse.urlencode({"dataset": dataset})
    return request_json(f"{HF_INFO_ENDPOINT}?{query}")["dataset_info"]["default"]


def hf_rows(dataset: str, split: str, config: str = "default", page_size: int = 100) -> Iterable[dict[str, Any]]:
    offset = 0
    total: int | None = None
    while True:
        query = urllib.parse.urlencode(
            {
                "dataset": dataset,
                "config": config,
                "split": split,
                "offset": offset,
                "length": page_size,
            }
        )
        payload = request_json(f"{HF_ROWS_ENDPOINT}?{query}")
        rows = payload.get("rows", [])
        if total is None:
            total = payload.get("num_rows_total")
            log(f"fetching {dataset}/{split}: {total} rows")
        if not rows:
            break
        for item in rows:
            yield item["row"]
        offset += len(rows)
        if total is not None and offset >= total:
            break


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def write_unified_tasks(
    path: Path,
    task_rows: list[dict[str, Any]],
    judge_units: list[dict[str, Any]],
) -> int:
    judge_by_uid = {row["judge_uid"]: row for row in judge_units}
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for task in task_rows:
            judge = judge_by_uid[task["judge_uid"]]
            row = {
                "task_uid": task["task_uid"],
                "judge_uid": task["judge_uid"],
                "source_dataset": task["source_dataset"],
                "source_id": task["source_id"],
                "source_split": task["source_split"],
                "question": task["question"],
                "question_type": task["question_type"],
                "choices": judge.get("choices"),
                "background": judge.get("background"),
                "resolution_criteria": judge.get("resolution_criteria"),
                "forecast_date": task["forecast_date"],
                "resolution_date": task["resolution_date"],
                "answer": task["answer"],
                "answer_type": task["answer_type"],
                "source_url": task["source_url"],
                "raw_category": task["raw_category"],
                "representative_task_uid": judge.get("representative_task_uid"),
                "task_count_for_judge_uid": judge.get("task_count"),
                "extra": task["extra"],
                "is_finance_econ": None,
                "confidence": None,
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def clean_value(value: Any) -> Any:
    if value is None:
        return None
    try:
        # pandas.NA/NaN support without importing pandas at module import time.
        if value != value:
            return None
    except TypeError:
        pass
    return value


def clean_text(value: Any) -> str | None:
    value = clean_value(value)
    if value is None:
        return None
    return str(value)


def base_judge_unit(
    *,
    judge_uid: str,
    source_dataset: str,
    source_id: str,
    source_split: str | None,
    question: str,
    question_type: str,
    choices: dict[str, str] | None,
    background: str | None,
    resolution_criteria: str | None,
    forecast_date: str | None,
    raw_category: str | None,
    source_url: str | None,
    representative_task_uid: str,
    task_count: int = 1,
) -> dict[str, Any]:
    return {
        "judge_uid": judge_uid,
        "source_dataset": source_dataset,
        "source_id": source_id,
        "source_split": source_split,
        "question": question,
        "question_type": question_type,
        "choices": choices,
        "background": background,
        "resolution_criteria": resolution_criteria,
        "forecast_date": forecast_date,
        "raw_category": raw_category,
        "source_url": source_url,
        "representative_task_uid": representative_task_uid,
        "task_count": task_count,
    }


def base_task_row(
    *,
    task_uid: str,
    judge_uid: str,
    source_dataset: str,
    source_id: str,
    source_split: str | None,
    question: str,
    question_type: str,
    forecast_date: str | None,
    resolution_date: str | None,
    answer: Any,
    answer_type: str | None,
    source_url: str | None,
    raw_category: str | None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "task_uid": task_uid,
        "judge_uid": judge_uid,
        "source_dataset": source_dataset,
        "source_id": source_id,
        "source_split": source_split,
        "question": question,
        "question_type": question_type,
        "forecast_date": forecast_date,
        "resolution_date": resolution_date,
        "answer": answer,
        "answer_type": answer_type,
        "source_url": source_url,
        "raw_category": raw_category,
        "extra": extra or {},
    }


def build_btf2() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    import pandas as pd

    url = "https://huggingface.co/datasets/BTF-2/BTF-2/resolve/main/btf2_questions_and_forecasts.parquet"
    parquet_path = download_file(url, RAW_CACHE_DIR / "btf2" / "btf2_questions_and_forecasts.parquet")
    df = pd.read_parquet(
        parquet_path,
        columns=[
            "question_id",
            "question",
            "resolution_criteria",
            "background",
            "present_date",
            "resolution",
            "sota_forecast_probability",
        ],
    )
    judge_units: list[dict[str, Any]] = []
    task_rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        source_id = clean_text(row["question_id"]) or ""
        task_uid = f"btf2:{source_id}"
        judge_uid = task_uid
        question = clean_text(row.get("question")) or ""
        judge_units.append(
            base_judge_unit(
                judge_uid=judge_uid,
                source_dataset="btf2",
                source_id=source_id,
                source_split="test",
                question=question,
                question_type="binary",
                choices=None,
                background=clean_text(row.get("background")),
                resolution_criteria=clean_text(row.get("resolution_criteria")),
                forecast_date=clean_text(row.get("present_date")),
                raw_category=None,
                source_url=None,
                representative_task_uid=task_uid,
            )
        )
        task_rows.append(
            base_task_row(
                task_uid=task_uid,
                judge_uid=judge_uid,
                source_dataset="btf2",
                source_id=source_id,
                source_split="test",
                question=question,
                question_type="binary",
                forecast_date=clean_text(row.get("present_date")),
                resolution_date=None,
                answer=clean_value(row.get("resolution")),
                answer_type="yes_no",
                source_url=None,
                raw_category=None,
                extra={
                    "sota_forecast_probability": clean_value(row.get("sota_forecast_probability")),
                },
            )
        )
    return judge_units, task_rows, {"judge_units": len(judge_units), "task_rows": len(task_rows)}


def build_openforesight() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    import pandas as pd

    repo_files = hf_repo_files("nikhilchandak/OpenForesight")
    parquet_files = sorted(path for path in repo_files if path.startswith("data/") and path.endswith(".parquet"))
    judge_units: list[dict[str, Any]] = []
    task_rows: list[dict[str, Any]] = []
    per_split: dict[str, int] = {}
    columns = [
        "qid",
        "question_title",
        "background",
        "resolution_criteria",
        "answer_type",
        "answer",
        "url",
        "article_publish_date",
        "news_source",
        "resolution_date",
        "question_start_date",
    ]
    for repo_path in parquet_files:
        filename = repo_path.rsplit("/", 1)[-1]
        split = filename.split("-000", 1)[0]
        url = f"https://huggingface.co/datasets/nikhilchandak/OpenForesight/resolve/main/{repo_path}"
        parquet_path = download_file(url, RAW_CACHE_DIR / "openforesight" / filename)
        df = pd.read_parquet(parquet_path, columns=columns)
        split_count = len(df)
        log(f"processing OpenForesight/{split}: {split_count} rows")
        for _, row in df.iterrows():
            source_id = clean_text(row["qid"]) or ""
            task_uid = f"openforesight:{split}:{source_id}"
            judge_uid = task_uid
            question = clean_text(row.get("question_title")) or ""
            judge_units.append(
                base_judge_unit(
                    judge_uid=judge_uid,
                    source_dataset="openforesight",
                    source_id=source_id,
                    source_split=split,
                    question=question,
                    question_type="open_ended",
                    choices=None,
                    background=clean_text(row.get("background")),
                    resolution_criteria=clean_text(row.get("resolution_criteria")),
                    forecast_date=clean_text(row.get("question_start_date")),
                    raw_category=None,
                    source_url=clean_text(row.get("url")),
                    representative_task_uid=task_uid,
                )
            )
            task_rows.append(
                base_task_row(
                    task_uid=task_uid,
                    judge_uid=judge_uid,
                    source_dataset="openforesight",
                    source_id=source_id,
                    source_split=split,
                    question=question,
                    question_type="open_ended",
                    forecast_date=clean_text(row.get("question_start_date")),
                    resolution_date=clean_text(row.get("resolution_date")),
                    answer=clean_value(row.get("answer")),
                    answer_type=clean_text(row.get("answer_type")),
                    source_url=clean_text(row.get("url")),
                    raw_category=None,
                    extra={
                        "news_source": clean_text(row.get("news_source")),
                        "article_publish_date": clean_text(row.get("article_publish_date")),
                    },
                )
            )
        per_split[split] = per_split.get(split, 0) + split_count
    return judge_units, task_rows, {
        "judge_units": len(judge_units),
        "task_rows": len(task_rows),
        "splits": per_split,
    }


def build_daily_oracle() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    judge_units: list[dict[str, Any]] = []
    task_rows: list[dict[str, Any]] = []
    per_kind: dict[str, int] = {}
    for kind, filename in DAILY_ORACLE_FILES.items():
        url = f"https://huggingface.co/datasets/agentic-learning-ai-lab/daily-oracle/resolve/main/{filename}"
        content = request_text(url)
        reader = csv.DictReader(io.StringIO(content))
        count = 0
        for row in reader:
            if row.get("category") != "Economics & Business":
                continue
            key_material = "|".join([kind, row.get("date") or "", row.get("question") or "", row.get("url") or ""])
            source_id = stable_hash(key_material)
            task_uid = f"daily_oracle:{kind}:{source_id}"
            judge_uid = task_uid
            question = row.get("question") or ""
            choices = None
            question_type = "binary" if kind == "tf" else "multiple_choice"
            answer_type = "yes_no" if kind == "tf" else "choice"
            if kind == "mc":
                choices = {
                    "a": row.get("choice_a") or "",
                    "b": row.get("choice_b") or "",
                    "c": row.get("choice_c") or "",
                    "d": row.get("choice_d") or "",
                }
            judge_units.append(
                base_judge_unit(
                    judge_uid=judge_uid,
                    source_dataset="daily_oracle",
                    source_id=source_id,
                    source_split=kind,
                    question=question,
                    question_type=question_type,
                    choices=choices,
                    background=None,
                    resolution_criteria=None,
                    forecast_date=row.get("date"),
                    raw_category=row.get("category"),
                    source_url=row.get("url"),
                    representative_task_uid=task_uid,
                )
            )
            task_rows.append(
                base_task_row(
                    task_uid=task_uid,
                    judge_uid=judge_uid,
                    source_dataset="daily_oracle",
                    source_id=source_id,
                    source_split=kind,
                    question=question,
                    question_type=question_type,
                    forecast_date=row.get("date"),
                    resolution_date=row.get("date"),
                    answer=row.get("answer"),
                    answer_type=answer_type,
                    source_url=row.get("url"),
                    raw_category=row.get("category"),
                    extra={
                        "source_domain": row.get("source_domain"),
                        "article_selection": row.get("article_selection"),
                        "total_points": row.get("total_points"),
                    },
                )
            )
            count += 1
        per_kind[kind] = count
    return judge_units, task_rows, {
        "judge_units": len(judge_units),
        "task_rows": len(task_rows),
        "files": per_kind,
        "pre_filter": 'category == "Economics & Business"',
    }


def github_contents(path: str) -> list[dict[str, Any]]:
    url = f"https://api.github.com/repos/forecastingresearch/forecastbench-datasets/contents/{path}?ref=main"
    return request_json(url)


def build_forecastbench() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    files = [
        item
        for item in github_contents("datasets/question_sets")
        if item["name"].endswith(".json") and item["name"] != "latest-llm.json"
    ]
    files.sort(key=lambda item: item["name"])

    grouped: dict[str, dict[str, Any]] = {}
    task_rows: list[dict[str, Any]] = []
    snapshot_seen: set[tuple[str, str | None]] = set()
    raw_rows = 0
    duplicate_snapshots = 0
    source_counts: Counter[str] = Counter()

    for item in files:
        payload = request_json(item["download_url"])
        forecast_due_date = payload.get("forecast_due_date")
        for q in payload.get("questions", []):
            raw_rows += 1
            source = q.get("source") or "unknown"
            source_counts[source] += 1
            source_id = json_dumps(q.get("id")) if isinstance(q.get("id"), (dict, list)) else str(q.get("id"))
            freeze_datetime = q.get("freeze_datetime")
            snapshot_key = (source_id, freeze_datetime)
            if snapshot_key in snapshot_seen:
                duplicate_snapshots += 1
                continue
            snapshot_seen.add(snapshot_key)

            question = q.get("question") or ""
            normalized_question = normalize_text(question)
            judge_uid = f"forecastbench:{stable_hash(normalized_question)}"
            task_uid = f"forecastbench:{stable_hash(source_id + '|' + str(freeze_datetime), 24)}"
            group = grouped.setdefault(
                judge_uid,
                {
                    "judge_uid": judge_uid,
                    "source_dataset": "forecastbench",
                    "source_id": stable_hash(normalized_question),
                    "source_split": None,
                    "question": question,
                    "question_type": "forecastbench",
                    "choices": None,
                    "background": q.get("background"),
                    "resolution_criteria": q.get("resolution_criteria"),
                    "forecast_date": freeze_datetime,
                    "raw_category": source,
                    "source_url": q.get("url"),
                    "representative_task_uid": task_uid,
                    "task_count": 0,
                    "_sources": Counter(),
                },
            )
            group["task_count"] += 1
            group["_sources"][source] += 1
            if len(q.get("background") or "") > len(group.get("background") or ""):
                group["background"] = q.get("background")
            if len(q.get("resolution_criteria") or "") > len(group.get("resolution_criteria") or ""):
                group["resolution_criteria"] = q.get("resolution_criteria")
            if not group.get("source_url") and q.get("url"):
                group["source_url"] = q.get("url")

            task_rows.append(
                base_task_row(
                    task_uid=task_uid,
                    judge_uid=judge_uid,
                    source_dataset="forecastbench",
                    source_id=source_id,
                    source_split=item["name"],
                    question=question,
                    question_type="forecastbench",
                    forecast_date=freeze_datetime,
                    resolution_date=q.get("resolution_dates"),
                    answer=None,
                    answer_type=None,
                    source_url=q.get("url"),
                    raw_category=source,
                    extra={
                        "forecast_due_date": forecast_due_date,
                        "freeze_datetime_value": q.get("freeze_datetime_value"),
                        "market_info_open_datetime": q.get("market_info_open_datetime"),
                        "market_info_close_datetime": q.get("market_info_close_datetime"),
                    },
                )
            )

    judge_units: list[dict[str, Any]] = []
    for group in grouped.values():
        sources = dict(group.pop("_sources"))
        group["raw_category"] = ",".join(f"{source}:{count}" for source, count in sorted(sources.items()))
        judge_units.append(group)

    return judge_units, task_rows, {
        "raw_rows": raw_rows,
        "deduped_snapshot_task_rows": len(task_rows),
        "duplicate_snapshots_skipped": duplicate_snapshots,
        "judge_units": len(judge_units),
        "source_counts_raw": dict(source_counts),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    builders = [
        ("btf2", build_btf2),
        ("openforesight", build_openforesight),
        ("daily_oracle", build_daily_oracle),
        ("forecastbench", build_forecastbench),
    ]

    all_judge_units: list[dict[str, Any]] = []
    all_task_rows: list[dict[str, Any]] = []
    manifest: dict[str, Any] = {"sources": {}}

    for name, builder in builders:
        log(f"building {name}")
        judge_units, task_rows, summary = builder()
        all_judge_units.extend(judge_units)
        all_task_rows.extend(task_rows)
        manifest["sources"][name] = summary
        log(f"{name}: {summary}")

    judge_count = write_jsonl(JUDGE_UNITS_PATH, all_judge_units)
    task_count = write_jsonl(TASK_INDEX_PATH, all_task_rows)
    unified_count = write_unified_tasks(UNIFIED_TASKS_PATH, all_task_rows, all_judge_units)

    manifest["total"] = {
        "judge_units": judge_count,
        "task_rows": task_count,
        "unified_task_rows": unified_count,
        "judge_output_columns_to_append": ["is_finance_econ", "confidence"],
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    log(f"wrote {JUDGE_UNITS_PATH} ({judge_count} rows)")
    log(f"wrote {TASK_INDEX_PATH} ({task_count} rows)")
    log(f"wrote {UNIFIED_TASKS_PATH} ({unified_count} rows)")
    log(f"wrote {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
