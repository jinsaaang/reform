"""Paper-aligned blinded reasoning evaluation for Raw DAG versus Full HGF."""

from __future__ import annotations

import argparse
import json
import os
import random
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from dotenv import find_dotenv, load_dotenv
from openai import OpenAI

from hgf.contracts import _target_contract
from hgf.exemplar import _call_with_repair
from hgf.experiment_common import (
    provenance_snapshot,
    read_json,
    utc_now,
    write_json,
)
from hgf.forecast_core import _seed
from hgf.generation import configure_generation
from hgf.package import PACKAGE_ROOT
from hgf.question_io import read_questions


DEFAULT_JUDGE_MODEL = "google/gemini-3.1-flash-lite"
MEASURES = (
    "evidence_coverage",
    "invalid_reasoning",
    "invalid_among_correct",
)
CONDITIONS = ("raw_dag", "full_hgf")
METHOD_CONDITIONS = {
    "direct_dag": "raw_dag",
    "hgf": "full_hgf",
}
INVALID_REASON_KEYS = (
    "unsupported_decisive_claim",
    "forecast_time_violation",
    "selected_outcome_not_justified",
)
PROMPT_VERSION = "hgf_reasoning_judge_paper_v3"
SYSTEM_PROMPT = (
    "You are a blinded forecasting-reasoning evaluator. Never infer or use "
    "the realized answer. Return only schema-conforming JSON."
)
PROMPT_PREAMBLE = (
    "Evaluate two anonymized forecasts. Do not infer method identity, model "
    "identity, or the realized outcome. Judge reasoning quality only from "
    "the public target, forecast cutoff, supplied forecast-time evidence, "
    "written rationale, and selected outcome. Treat only the supplied "
    "evidence records as citable support. Apply the same protocol "
    "independently to A and B."
)
_WRITE_LOCK = threading.Lock()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--forecast-results",
        type=Path,
        action="append",
        required=True,
        help=(
            "A completed ablation or main-table results.json; "
            "repeat for multiple runs."
        ),
    )
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--passes", type=int, default=1)
    parser.add_argument("--workers", type=int, default=30)
    parser.add_argument("--max-tokens", type=int, default=8000)
    parser.add_argument(
        "--reasoning-effort",
        choices=("minimal", "low", "medium", "high", "xhigh", "max"),
        default="medium",
    )
    parser.add_argument("--run-seed", type=int, default=27)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and summarize paired inputs without calling the judge.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/reasoning_judge"),
    )
    return parser.parse_args()


def _judgment_schema() -> dict[str, Any]:
    evidence_item = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "requirement": {"type": "string"},
            "cited_evidence_ids": {
                "type": "array",
                "items": {"type": "string"},
            },
            "supported_at_forecast_time": {"type": "boolean"},
            "rationale": {"type": "string"},
        },
        "required": [
            "requirement",
            "cited_evidence_ids",
            "supported_at_forecast_time",
            "rationale",
        ],
    }
    invalid_reasons = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            key: {"type": "boolean"} for key in INVALID_REASON_KEYS
        },
        "required": list(INVALID_REASON_KEYS),
    }
    forecast = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "evidence_items": {
                "type": "array",
                "minItems": 1,
                "items": evidence_item,
            },
            "invalid_reasoning": {"type": "boolean"},
            "invalid_reasons": invalid_reasons,
            "invalid_reasoning_rationale": {"type": "string"},
        },
        "required": [
            "evidence_items",
            "invalid_reasoning",
            "invalid_reasons",
            "invalid_reasoning_rationale",
        ],
    }
    return {
        "name": PROMPT_VERSION,
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "forecast_a": forecast,
                "forecast_b": forecast,
            },
            "required": ["forecast_a", "forecast_b"],
        },
    }


def _judgment_validator(
    payload: dict[str, Any],
    *,
    allowed_evidence_ids: (
        set[str] | dict[str, set[str]] | None
    ) = None,
) -> tuple[dict[str, float], list[str]]:
    errors: list[str] = []
    for label in ("forecast_a", "forecast_b"):
        judgment = payload.get(label, {})
        evidence_items = judgment.get("evidence_items")
        if not isinstance(evidence_items, list) or not evidence_items:
            errors.append(f"{label}.evidence_items is empty")
            evidence_items = []
        support_values: list[bool] = []
        for index, item in enumerate(evidence_items):
            prefix = f"{label}.evidence_items[{index}]"
            if not str(item.get("requirement") or "").strip():
                errors.append(f"{prefix}.requirement is empty")
            if not str(item.get("rationale") or "").strip():
                errors.append(f"{prefix}.rationale is empty")
            supported = item.get("supported_at_forecast_time")
            if not isinstance(supported, bool):
                errors.append(
                    f"{prefix}.supported_at_forecast_time is not boolean"
                )
            else:
                support_values.append(supported)
            cited_ids = item.get("cited_evidence_ids")
            if not isinstance(cited_ids, list) or any(
                not isinstance(value, str) or not value.strip()
                for value in cited_ids or []
            ):
                errors.append(f"{prefix}.cited_evidence_ids is invalid")
                cited_ids = []
            if supported is True and not cited_ids:
                errors.append(
                    f"{prefix} is supported but cites no evidence ID"
                )
            if allowed_evidence_ids is not None:
                allowed_ids = (
                    allowed_evidence_ids[label]
                    if isinstance(allowed_evidence_ids, dict)
                    else allowed_evidence_ids
                )
                unknown_ids = sorted(set(cited_ids) - allowed_ids)
                if unknown_ids:
                    errors.append(
                        f"{prefix} cites unknown evidence IDs: {unknown_ids}"
                    )
        invalid = judgment.get("invalid_reasoning")
        if not isinstance(invalid, bool):
            errors.append(f"{label}.invalid_reasoning is not boolean")
        reasons = judgment.get("invalid_reasons", {})
        reason_values = []
        for key in INVALID_REASON_KEYS:
            value = reasons.get(key)
            if not isinstance(value, bool):
                errors.append(f"{label}.invalid_reasons.{key} is not boolean")
            else:
                reason_values.append(value)
        unsupported_reason = reasons.get("unsupported_decisive_claim")
        if (
            isinstance(unsupported_reason, bool)
            and len(support_values) == len(evidence_items)
            and unsupported_reason != any(not value for value in support_values)
        ):
            errors.append(
                f"{label}.invalid_reasons.unsupported_decisive_claim must "
                "match unsupported evidence items"
            )
        if (
            isinstance(invalid, bool)
            and len(reason_values) == len(INVALID_REASON_KEYS)
            and invalid != any(reason_values)
        ):
            errors.append(
                f"{label}.invalid_reasoning must equal the OR of "
                "invalid_reasons"
            )
        if not str(
            judgment.get("invalid_reasoning_rationale") or ""
        ).strip():
            errors.append(f"{label}.invalid_reasoning_rationale is empty")
    return {}, errors


def _rubric() -> str:
    return """
Apply the paper's three reasoning measures without assigning 1-5 scores.

1. Evidence coverage
Identify the atomic, non-duplicated factual evidence requirements that the
submitted rationale relies on for its decisive path to the selected outcome.
For each requirement, cite the supplied evidence record IDs that support it.
Mark it supported only when at least one cited record was available at the
forecast cutoff and its text actually supports the requirement. A citation
that is merely related, weaker than the claim, or published after the cutoff
does not count. Evidence coverage is computed later as:
supported required evidence items / all required evidence items.

2. Invalid reasoning
Mark the rationale invalid if at least one decisive claim:
- lacks support in the supplied forecast-time evidence;
- uses information unavailable at the forecast cutoff; or
- does not justify the selected outcome under the exact target contract.
Set the corresponding invalid_reasons flags and make invalid_reasoning equal
their logical OR.

3. Invalid among correct
Do not judge correctness and do not infer the realized outcome. This rate is
computed only after blinding is complete by combining invalid_reasoning with
the separately stored realized outcome.

Judge the written forecast, not hidden model computation. An outcome that
could turn out to be wrong is not by itself invalid reasoning. Evaluate A and
B independently; do not rank them or force a difference.

For each forecast, return:
- evidence_items, each with requirement, cited_evidence_ids,
  supported_at_forecast_time, and rationale;
- invalid_reasoning;
- the three boolean invalid_reasons; and
- invalid_reasoning_rationale, explaining the overall validity decision.
""".strip()


def _judge_prompt(
    *,
    question: Any,
    cutoff: str,
    forecast_a: dict[str, Any],
    forecast_b: dict[str, Any],
) -> str:
    public_question = {
        "question": question.question_text,
        "context": question.context,
        "target_contract": _target_contract(question),
        "forecast_cutoff": cutoff,
    }
    return (
        f"{PROMPT_PREAMBLE}\n\n"
        f"{_rubric()}\n\n"
        f"QUESTION:\n{json.dumps(public_question, ensure_ascii=False)}\n\n"
        f"FORECAST A:\n{json.dumps(forecast_a, ensure_ascii=False)}\n\n"
        f"FORECAST B:\n{json.dumps(forecast_b, ensure_ascii=False)}"
    )


def _read_evidence(row: dict[str, Any]) -> list[dict[str, Any]]:
    if isinstance(row.get("evidence"), list):
        return row["evidence"]
    question_id = str(row.get("question_id") or "")
    evidence_bank = str(row.get("evidence_bank") or "").lower()
    canonical_path = (
        PACKAGE_ROOT
        / "data"
        / "evidence"
        / evidence_bank
        / f"{question_id}.sqlite"
    )
    db_path = (
        canonical_path
        if canonical_path.is_file()
        else Path(str(row["evidence_db"])).resolve()
    )
    ids = [str(value) for value in row.get("evidence_ids", [])]
    if not ids:
        return []
    placeholders = ",".join("?" for _ in ids)
    uri = f"{db_path.as_uri()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        rows = connection.execute(
            f"""
            SELECT id, title, source, published_date, content
            FROM articles
            WHERE id IN ({placeholders})
            """,
            ids,
        ).fetchall()
    by_id = {
        str(raw[0]): {
            "id": str(raw[0]),
            "title": str(raw[1]),
            "source": str(raw[2]),
            "published_date": str(raw[3]),
            "excerpt": " ".join(str(raw[4] or "").split())[:500],
        }
        for raw in rows
    }
    return [by_id[value] for value in ids if value in by_id]


def _selected_outcome(row: dict[str, Any]) -> str:
    probabilities = row.get("probabilities")
    if not isinstance(probabilities, dict) or not probabilities:
        raise ValueError("forecast probabilities are missing")
    options = [
        str(option)
        for option in row.get("options", probabilities.keys())
        if str(option) in probabilities
    ]
    if not options:
        options = [str(option) for option in probabilities]
    return max(options, key=lambda option: float(probabilities[option]))


def _blind_forecast(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "selected_outcome": _selected_outcome(row),
        "reasoning": row.get("reasoning"),
    }


def _condition_for_row(row: dict[str, Any]) -> str | None:
    condition = str(row.get("condition") or "")
    if condition in CONDITIONS:
        return condition
    return METHOD_CONDITIONS.get(str(row.get("method") or ""))


def _paired_rows(
    forecast_paths: list[Path],
) -> list[dict[str, Any]]:
    output = []
    for run_index, path in enumerate(forecast_paths, start=1):
        payload = read_json(path)
        by_key = {}
        for row in payload.get("results", []):
            condition = _condition_for_row(row)
            if row.get("status") != "success" or condition is None:
                continue
            by_key[(str(row.get("question_id")), condition)] = row
        question_ids = sorted(
            question_id
            for question_id, condition in by_key
            if condition == "full_hgf"
            and (question_id, "raw_dag") in by_key
        )
        if len(question_ids) != 100:
            raise ValueError(
                f"{path}: expected 100 paired Raw DAG/Full HGF questions, "
                f"found {len(question_ids)}"
            )
        for question_id in question_ids:
            raw = by_key[(question_id, "raw_dag")]
            full = by_key[(question_id, "full_hgf")]
            output.append(
                {
                    "run_index": run_index,
                    "forecast_results": str(path),
                    "forecaster_model": str(payload.get("model") or ""),
                    "question_id": question_id,
                    "cutoff": str(full.get("cutoff") or raw.get("cutoff")),
                    "evidence": {
                        "raw_dag": _read_evidence(raw),
                        "full_hgf": _read_evidence(full),
                    },
                    "rows": {"raw_dag": raw, "full_hgf": full},
                }
            )
    return output


def _call_judge(
    *,
    client: OpenAI,
    judge_model: str,
    question: Any,
    pair: dict[str, Any],
    judge_pass: int,
    output_dir: Path,
    max_tokens: int,
    random_seed: int,
) -> dict[str, Any]:
    question_id = pair["question_id"]
    output_path = (
        output_dir
        / "cases"
        / f"run_{pair['run_index']}"
        / question_id
        / f"pass_{judge_pass}.json"
    )
    if output_path.is_file():
        cached = read_json(output_path)
        if (
            cached.get("status") == "success"
            and cached.get("prompt_version") == PROMPT_VERSION
            and cached.get("judge_model") == judge_model
        ):
            return cached
    order = list(CONDITIONS)
    random.Random(
        f"{random_seed}:{pair['run_index']}:{question_id}:{judge_pass}"
    ).shuffle(order)
    rows = pair["rows"]
    blinded = {
        condition: {
            **_blind_forecast(rows[condition]),
            "evidence": pair["evidence"][condition],
        }
        for condition in CONDITIONS
    }
    scored, _, usage, seconds, repaired = _call_with_repair(
        client,
        model=judge_model,
        system=SYSTEM_PROMPT,
        prompt=_judge_prompt(
            question=question,
            cutoff=pair["cutoff"],
            forecast_a=blinded[order[0]],
            forecast_b=blinded[order[1]],
        ),
        schema=_judgment_schema(),
        seed=_seed(
            question_id,
            f"judge:run={pair['run_index']}:pass={judge_pass}",
        ),
        max_tokens=max_tokens,
        validator=lambda payload: _judgment_validator(
            payload,
            allowed_evidence_ids={
                "forecast_a": {
                    str(item["id"])
                    for item in pair["evidence"][order[0]]
                    if item.get("id") is not None
                },
                "forecast_b": {
                    str(item["id"])
                    for item in pair["evidence"][order[1]]
                    if item.get("id") is not None
                },
            },
        ),
    )
    labels = {"forecast_a": order[0], "forecast_b": order[1]}
    judgments = {}
    for label, condition in labels.items():
        judgment = scored[label]
        evidence_items = judgment["evidence_items"]
        required_count = len(evidence_items)
        supported_count = sum(
            item["supported_at_forecast_time"] for item in evidence_items
        )
        selected_outcome = _selected_outcome(rows[condition])
        correct = selected_outcome == str(rows[condition].get("ground_truth"))
        judgments[condition] = {
            "evidence_items": evidence_items,
            "required_evidence_items": required_count,
            "supported_evidence_items": supported_count,
            "evidence_coverage": supported_count / required_count,
            "invalid_reasoning": bool(judgment["invalid_reasoning"]),
            "invalid_reasons": judgment["invalid_reasons"],
            "invalid_reasoning_rationale": str(
                judgment["invalid_reasoning_rationale"]
            ),
            "selected_outcome": selected_outcome,
            "correct_after_unblinding": correct,
        }
    result = {
        "schema_version": "hgf_reasoning_judge_case_v3",
        "status": "success",
        "question_id": question_id,
        "run_index": pair["run_index"],
        "judge_pass": judge_pass,
        "judge_model": judge_model,
        "prompt_version": PROMPT_VERSION,
        "blind_order": labels,
        "judgments": judgments,
        "usage": usage,
        "seconds": seconds,
        "parse_retries": int(bool(repaired)),
    }
    with _WRITE_LOCK:
        write_json(output_path, result)
    return result


def _rate(numerator: int, denominator: int) -> dict[str, Any]:
    return {
        "numerator": numerator,
        "denominator": denominator,
        "rate": numerator / denominator if denominator else None,
    }


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "successful_judgments": len(rows),
        "parse_retries": sum(int(row.get("parse_retries") or 0) for row in rows),
        "conditions": {},
    }
    for condition in CONDITIONS:
        condition_rows = [row["judgments"][condition] for row in rows]
        required = sum(
            int(item["required_evidence_items"]) for item in condition_rows
        )
        supported = sum(
            int(item["supported_evidence_items"]) for item in condition_rows
        )
        invalid = sum(
            int(item["invalid_reasoning"]) for item in condition_rows
        )
        correct_rows = [
            item for item in condition_rows if item["correct_after_unblinding"]
        ]
        invalid_correct = sum(
            int(item["invalid_reasoning"]) for item in correct_rows
        )
        summary["conditions"][condition] = {
            "evidence_coverage": _rate(supported, required),
            "invalid_reasoning": _rate(invalid, len(condition_rows)),
            "invalid_among_correct": _rate(
                invalid_correct,
                len(correct_rows),
            ),
        }
    return summary


def main() -> None:
    args = _parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")
    if args.passes < 1:
        raise ValueError("passes must be positive")
    configure_generation(
        reasoning_effort=args.reasoning_effort,
        max_output_tokens=args.max_tokens,
        run_seed=args.run_seed,
    )
    pairs = _paired_rows([path.resolve() for path in args.forecast_results])
    if not pairs:
        raise ValueError("no paired Raw DAG and Full HGF rows")
    forecaster_models = sorted(
        {pair["forecaster_model"] for pair in pairs if pair["forecaster_model"]}
    )
    if args.judge_model in forecaster_models:
        raise ValueError(
            "judge model must differ from every forecast-generating model"
        )
    if args.dry_run:
        print(
            json.dumps(
                {
                    "forecast_results": [
                        str(path.resolve()) for path in args.forecast_results
                    ],
                    "runs": len(args.forecast_results),
                    "paired_questions": len(pairs),
                    "raw_dag_with_evidence": sum(
                        bool(pair["evidence"]["raw_dag"]) for pair in pairs
                    ),
                    "full_hgf_with_evidence": sum(
                        bool(pair["evidence"]["full_hgf"]) for pair in pairs
                    ),
                    "judge_model": args.judge_model,
                    "workers": args.workers,
                    "reasoning_effort": args.reasoning_effort,
                    "max_tokens": args.max_tokens,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    questions = {
        str(question.id): question
        for question in read_questions(
            PACKAGE_ROOT / "data" / "questions" / "test_questions.jsonl"
        )
    }
    output_dir = args.output_dir.resolve()
    write_json(
        output_dir / "protocol.json",
        {
            "schema_version": "hgf_reasoning_judge_protocol_v3",
            "judge_model": args.judge_model,
            "forecaster_models": forecaster_models,
            "passes": args.passes,
            "workers": args.workers,
            "reasoning_effort": args.reasoning_effort,
            "max_tokens": args.max_tokens,
            "prompt_version": PROMPT_VERSION,
            "system_prompt": SYSTEM_PROMPT,
            "user_prompt_preamble": PROMPT_PREAMBLE,
            "rubric": _rubric(),
            "response_schema": _judgment_schema(),
            "blind_fields": ["method", "model", "ground_truth"],
            "measures": list(MEASURES),
            "started_at_utc": utc_now(),
            "provenance": provenance_snapshot(
                root=PACKAGE_ROOT,
                requested_model=args.judge_model,
                run_seed=args.run_seed,
                config_paths=(Path("configs/reasoning_judge.json"),),
                extra={"experiment": "reasoning_judge"},
            ),
        },
    )
    load_dotenv(find_dotenv(usecwd=True))
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        timeout=180,
        max_retries=2,
    )
    tasks = [
        (pair, judge_pass)
        for pair in pairs
        for judge_pass in range(1, args.passes + 1)
    ]
    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                _call_judge,
                client=client,
                judge_model=args.judge_model,
                question=questions[pair["question_id"]],
                pair=pair,
                judge_pass=judge_pass,
                output_dir=output_dir,
                max_tokens=args.max_tokens,
                random_seed=args.run_seed,
            ): (pair, judge_pass)
            for pair, judge_pass in tasks
        }
        for future in as_completed(futures):
            pair, judge_pass = futures[future]
            try:
                row = future.result()
            except Exception as exc:
                row = {
                    "schema_version": "hgf_reasoning_judge_case_v3",
                    "status": "failed",
                    "question_id": pair["question_id"],
                    "run_index": pair["run_index"],
                    "judge_pass": judge_pass,
                    "error": f"{type(exc).__name__}: {exc}",
                }
                write_json(
                    output_dir
                    / "cases"
                    / f"run_{pair['run_index']}"
                    / pair["question_id"]
                    / f"pass_{judge_pass}.failed.json",
                    row,
                )
            rows.append(row)
            successful = sum(row.get("status") == "success" for row in rows)
            print(
                f"PROGRESS {len(rows)}/{len(tasks)} "
                f"success={successful} failed={len(rows)-successful}",
                flush=True,
            )
    successful = [row for row in rows if row.get("status") == "success"]
    summary = _summary(successful)
    summary["expected_judgments"] = len(tasks)
    summary["failed_judgments"] = len(tasks) - len(successful)
    result = {
        "schema_version": "hgf_reasoning_judge_experiment_v3",
        "judge_model": args.judge_model,
        "forecaster_models": forecaster_models,
        "passes": args.passes,
        "summary": summary,
        "results": sorted(
            rows,
            key=lambda row: (
                int(row["run_index"]),
                str(row["question_id"]),
                int(row["judge_pass"]),
            ),
        ),
    }
    write_json(output_dir / "results.json", result)
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
