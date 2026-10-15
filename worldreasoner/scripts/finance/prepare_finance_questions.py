#!/usr/bin/env python3
"""Convert FinFactorBench JSONL records into WorldReasoner questions.

The source files remain immutable. This script writes a lossless, validated
WorldReasoner view and a deterministic ten-question experiment sample.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import date, datetime, time, timezone
from pathlib import Path
from typing import Any, Iterable

from src.domain.models.domain import Domain
from src.domain.models.question import Question, QuestionType


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESEARCH_ROOT = PROJECT_ROOT.parent
DEFAULT_INPUT = RESEARCH_ROOT / "data" / "questions_300_with_seed_sources.jsonl"
DEFAULT_OUTPUT = RESEARCH_ROOT / "data" / "worldreasoner" / "finance_questions_300.jsonl"
DEFAULT_SAMPLE_OUTPUT = (
    RESEARCH_ROOT / "data" / "worldreasoner" / "finance_questions_sample_10.jsonl"
)

DIFFICULTY_MAP = {"market_like": 3, "hard": 4, "very_hard": 5}
DOMAIN_SAMPLE_PLAN = {
    "corporate_earnings": ("binary", "multiclass"),
    "energy_commodities": ("binary", "bucketed_numeric"),
    "macro": ("bucketed_numeric", "bucketed_numeric"),
    "market_fx_credit": ("binary", "multiclass"),
    "monetary_policy": ("bucketed_numeric", "multiclass"),
}
ISO_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _utc_datetime(day: str, *, end_of_day: bool = False) -> datetime:
    parsed = date.fromisoformat(day)
    wall_time = time(23, 59, 59) if end_of_day else time.min
    return datetime.combine(parsed, wall_time, tzinfo=timezone.utc)


def _binary_truth(value: Any) -> bool:
    normalized = str(value).strip().lower()
    if normalized == "yes":
        return True
    if normalized == "no":
        return False
    raise ValueError(f"Unsupported binary ground truth: {value!r}")


def convert_record(record: dict[str, Any]) -> Question:
    """Convert and validate one FinFactorBench record."""
    source_type = record["question_type"]
    question_type = (
        QuestionType.BINARY if source_type == "binary" else QuestionType.MCQ
    )
    ground_truth: Any = record["resolved_outcome"]
    options: list[str] | None = list(record["answer_space"])
    if question_type == QuestionType.BINARY:
        ground_truth = _binary_truth(ground_truth)

    forecast_day = record["forecast_date_options"][0]
    resolution_day = record["resolution_document"]["published_at"]
    temporal_precision_warning = forecast_day == resolution_day

    context_parts = [
        f"Entity: {record['entity']}",
        f"Target period: {record['target_period']}",
        f"Financial area: {record['domain']} / {record['subdomain']}",
        f"Region: {record['region']}",
    ]

    mapped_keys = {
        "question_id",
        "question",
        "question_type",
        "difficulty",
        "forecast_date_options",
        "resolution_document",
        "resolution_rule",
        "resolution_evidence",
        "resolved_outcome",
        "answer_space",
    }
    source_metadata = {
        key: value for key, value in record.items() if key not in mapped_keys
    }
    source_metadata.update(
        {
            "original_domain": record["domain"],
            "original_question_type": source_type,
            "forecast_date_options": record["forecast_date_options"],
            "resolution_document": record["resolution_document"],
            "answer_space": record["answer_space"],
            "resolved_outcome_label": record["resolved_outcome"],
            "temporal_precision_warning": temporal_precision_warning,
        }
    )

    return Question(
        id=record["question_id"],
        question_text=record["question"],
        question_type=question_type,
        domain=Domain.FINANCE,
        source="finfactorbench",
        difficulty=DIFFICULTY_MAP[record["difficulty"]],
        estimated_start_time=_utc_datetime(forecast_day),
        resolution_date=_utc_datetime(resolution_day, end_of_day=True),
        ground_truth=ground_truth,
        options=options,
        context="\n".join(context_parts),
        resolution_criteria=record["resolution_rule"],
        resolution_reasoning=record["resolution_evidence"],
        is_synthetic=False,
        metadata={"finfactorbench": source_metadata},
    )


def _exact_seed_date_count(record: dict[str, Any]) -> int:
    return sum(
        bool(ISO_DATE_RE.fullmatch(source["published_at_or_available_by"]))
        for source in record["pre_resolution_signal_sources"]
    )


def _resolution_gap_days(record: dict[str, Any]) -> int:
    forecast_day = date.fromisoformat(record["forecast_date_options"][0])
    resolution_day = date.fromisoformat(record["resolution_document"]["published_at"])
    return (resolution_day - forecast_day).days


def _sample_rank(record: dict[str, Any]) -> tuple[Any, ...]:
    """Prefer well-sourced, higher-quality records with a usable time window."""
    return (
        -_exact_seed_date_count(record),
        -record["crawl_quality_score"],
        -_resolution_gap_days(record),
        record["question_id"],
    )


def select_sample(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Select two deterministic, type-diverse questions per source domain."""
    selected: list[dict[str, Any]] = []
    used_resolution_docs: set[str] = set()
    used_subdomains: set[tuple[str, str]] = set()

    for domain, requested_types in DOMAIN_SAMPLE_PLAN.items():
        for source_type in requested_types:
            candidates = sorted(
                (
                    record
                    for record in records
                    if record["domain"] == domain
                    and record["question_type"] == source_type
                ),
                key=_sample_rank,
            )
            if not candidates:
                raise ValueError(f"No sample candidate for {domain}/{source_type}")

            unused = [
                record
                for record in candidates
                if record["resolution_doc_id"] not in used_resolution_docs
                and (domain, record["subdomain"]) not in used_subdomains
            ]
            choice = (unused or candidates)[0]
            selected.append(choice)
            used_resolution_docs.add(choice["resolution_doc_id"])
            used_subdomains.add((domain, choice["subdomain"]))

    return selected


def write_jsonl(path: Path, questions: Iterable[Question]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for question in questions:
            data = question.model_dump(
                mode="json", exclude={"created_at", "updated_at"}
            )
            handle.write(
                json.dumps(data, ensure_ascii=False) + "\n"
            )


def build_outputs(
    input_path: Path,
    output_path: Path,
    sample_output_path: Path,
) -> dict[str, Any]:
    records = read_jsonl(input_path)
    if len(records) != 300:
        raise ValueError(f"Expected 300 records, found {len(records)}")
    if len({record["question_id"] for record in records}) != len(records):
        raise ValueError("Duplicate question_id values found")

    questions = [convert_record(record) for record in records]
    sample_records = select_sample(records)
    sample_questions = [convert_record(record) for record in sample_records]

    write_jsonl(output_path, questions)
    write_jsonl(sample_output_path, sample_questions)

    return {
        "input": str(input_path),
        "output": str(output_path),
        "sample_output": str(sample_output_path),
        "question_count": len(questions),
        "sample_count": len(sample_questions),
        "question_types": Counter(q.question_type.value for q in questions),
        "sample_source_domains": Counter(
            q.metadata["finfactorbench"]["original_domain"]
            for q in sample_questions
        ),
        "same_day_precision_warnings": sum(
            q.metadata["finfactorbench"]["temporal_precision_warning"]
            for q in questions
        ),
        "sample_ids": [q.id for q in sample_questions],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sample-output", type=Path, default=DEFAULT_SAMPLE_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_outputs(args.input, args.output, args.sample_output)
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=dict))


if __name__ == "__main__":
    main()
