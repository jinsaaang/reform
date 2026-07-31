#!/usr/bin/env python3
"""Regenerate v22 cutoff-safe worked exemplars from the public DAG bank."""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dotenv import find_dotenv, load_dotenv  # noqa: E402
from openai import OpenAI  # noqa: E402

from exemplar.generator import _distill_exemplar  # noqa: E402
from hgf.forecast_core import _atomic_write  # noqa: E402
from hgf.generation import configure_generation  # noqa: E402
from hgf.memory_bank import (  # noqa: E402
    apply_blueprint_overrides,
    load_final_memory_bank,
)
from hgf.question_io import read_questions  # noqa: E402


DEFAULT_MODEL = "google/gemini-2.5-flash-lite"
DEFAULT_MAX_TOKENS = 2400


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "question_ids",
        nargs="+",
        help="Memory-question IDs to generate.",
    )
    parser.add_argument(
        "--questions",
        type=Path,
        default=ROOT / "data" / "questions" / "memory_questions.jsonl",
    )
    parser.add_argument(
        "--memory-bank-manifest",
        type=Path,
        default=ROOT / "data" / "memory_bank" / "manifest.json",
    )
    parser.add_argument(
        "--blueprint-override-dir",
        type=Path,
        default=None,
        help=(
            "Optional directory of question-ID-named blueprint JSON files. "
            "The canonical graph and evidence payload remain unchanged."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "exemplar" / "generated",
        help="Destination for wrapped fixed-memory exemplar artifacts.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=ROOT / "exemplar" / "cache",
        help="Destination for raw v22 generation cache entries.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--reasoning-effort",
        choices=("minimal", "low", "medium", "high", "xhigh", "max"),
        default=None,
    )
    parser.add_argument(
        "--attempts",
        type=int,
        default=1,
        help=(
            "Maximum generation attempts per memory when the provider "
            "returns invalid or truncated JSON."
        ),
    )
    return parser.parse_args()


def _artifact(
    *,
    memory_id: str,
    worked_exemplar: dict[str, Any],
    model: str,
    max_tokens: int,
) -> dict[str, Any]:
    return {
        "schema_version": "fixed_memory_exemplar_v1",
        "memory_question_id": memory_id,
        "worked_exemplar": worked_exemplar,
        "generation": {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "seed_role": "dag-exemplar",
            "source_logic": "v22 _distill_exemplar",
        },
    }


def main() -> None:
    args = _parse_args()
    if args.max_tokens <= 0:
        raise ValueError("--max-tokens must be positive")
    if args.attempts <= 0:
        raise ValueError("--attempts must be positive")
    if args.workers <= 0 or args.workers > 30:
        raise ValueError("--workers must be between 1 and 30")
    configure_generation(reasoning_effort=args.reasoning_effort)

    load_dotenv(find_dotenv(usecwd=True))
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")

    memory_questions = {
        str(question.id): question
        for question in read_questions(args.questions.resolve())
    }
    requested = list(dict.fromkeys(str(value) for value in args.question_ids))
    missing = sorted(set(requested) - set(memory_questions))
    if missing:
        raise ValueError(f"unknown memory-question IDs: {missing}")

    graphs, blueprints = load_final_memory_bank(
        args.memory_bank_manifest.resolve(),
        memory_questions,
    )
    blueprints = apply_blueprint_overrides(
        blueprints,
        args.blueprint_override_dir,
    )
    graphs_by_id = {
        str(blueprint["question_id"]): graph
        for graph, blueprint in zip(graphs, blueprints, strict=True)
    }
    blueprints_by_id = {
        str(blueprint["question_id"]): blueprint for blueprint in blueprints
    }
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        timeout=180,
        max_retries=2,
    )

    def generate_one(memory_id: str) -> dict[str, Any]:
        for attempt in range(1, args.attempts + 1):
            try:
                exemplar, usage, seconds, cached = _distill_exemplar(
                    client=client,
                    model=args.model,
                    memory_question=memory_questions[memory_id],
                    graph=graphs_by_id[memory_id],
                    blueprint=blueprints_by_id[memory_id],
                    cache_dir=args.cache_dir.resolve(),
                    max_tokens=args.max_tokens,
                )
                break
            except ValueError as exc:
                if attempt == args.attempts:
                    raise
                print(
                    f"RETRY {memory_id} attempt={attempt + 1}/"
                    f"{args.attempts} reason={str(exc)[:160]}",
                    flush=True,
                )
        output_path = args.output_dir.resolve() / f"{memory_id}.json"
        _atomic_write(
            output_path,
            _artifact(
                memory_id=memory_id,
                worked_exemplar=exemplar,
                model=args.model,
                max_tokens=args.max_tokens,
            ),
        )
        return {
            "memory_question_id": memory_id,
            "output": str(output_path),
            "cached": cached,
            "attempts": attempt,
            "seconds": seconds,
            "usage": usage,
        }

    results_by_id: dict[str, dict[str, Any]] = {}
    failures: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(generate_one, memory_id): memory_id
            for memory_id in requested
        }
        for future in as_completed(futures):
            memory_id = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                failures[memory_id] = f"{type(exc).__name__}: {exc}"
                print(
                    f"FAILED {memory_id} reason={failures[memory_id][:200]}",
                    flush=True,
                )
                continue
            results_by_id[str(result["memory_question_id"])] = result
            print(
                f"PROGRESS {len(results_by_id)}/{len(requested)}",
                flush=True,
            )

    print(
        json.dumps(
            {
                "results": [
                    results_by_id[memory_id]
                    for memory_id in requested
                    if memory_id in results_by_id
                ],
                "failures": failures,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
