"""Build and verify the canonical HGF Exemplar bank."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from dotenv import find_dotenv, load_dotenv
from openai import OpenAI

from .exemplar import _exemplar_article_ids
from .exemplar_generator import _distill_exemplar, _validate_exemplar
from .exemplar_selection import load_fixed_exemplar_bank
from .forecast_core import _atomic_write
from .generation import configure_generation
from .package import PACKAGE_ROOT
from .question_io import read_questions, resolve_forecast_cutoff


DEFAULT_MODEL = "google/gemini-2.5-flash-lite"
DEFAULT_MAX_TOKENS = 2400


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _canonical_hash(payload: Any) -> str:
    rendered = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(rendered).hexdigest()


def _load_blueprints(
    blueprint_root: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    manifest = _read(blueprint_root / "manifest.json")
    if manifest.get("blueprint_schema") != "hgf_blueprint_topology_v2":
        raise ValueError("canonical HGF Blueprint schema is missing")
    blueprints: dict[str, dict[str, Any]] = {}
    hashes: dict[str, str] = {}
    for entry in manifest.get("entries", []):
        question_id = str(entry["question_id"])
        path = blueprint_root / str(entry["blueprint_path"])
        payload = _read(path)
        if payload.get("schema_version") != "hgf_blueprint_topology_v2":
            raise ValueError(
                f"non-canonical Blueprint for {question_id}: {path}"
            )
        blueprints[question_id] = payload
        hashes[question_id] = str(entry["blueprint_sha256"])
    if len(blueprints) != int(manifest.get("memory_count") or 0):
        raise ValueError("Blueprint manifest coverage mismatch")
    return blueprints, hashes


def _load_graphs(memory_manifest_path: Path) -> dict[str, dict[str, Any]]:
    manifest = _read(memory_manifest_path)
    return {
        str(entry["question_id"]): _read(
            PACKAGE_ROOT / str(entry["graph_path"])
        )
        for entry in manifest.get("entries", [])
    }


def _exemplar_errors(
    *,
    memory_id: str,
    worked_exemplar: dict[str, Any],
    memory_questions: dict[str, Any],
    graphs: dict[str, dict[str, Any]],
) -> list[str]:
    cutoff, _ = resolve_forecast_cutoff(memory_questions[memory_id])
    allowed_ids = _exemplar_article_ids(graphs[memory_id], cutoff)
    return _validate_exemplar(
        worked_exemplar,
        allowed_article_ids=allowed_ids,
    )


def _memory_wrapper(
    *,
    memory_id: str,
    worked_exemplar: dict[str, Any],
    blueprint_sha256: str,
    model: str,
    max_tokens: int,
    generation_source: str,
) -> dict[str, Any]:
    return {
        "schema_version": "hgf_memory_exemplar_v1",
        "memory_question_id": memory_id,
        "blueprint_sha256": blueprint_sha256,
        "worked_exemplar": worked_exemplar,
        "generation": {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "seed_role": "dag-exemplar",
            "source": generation_source,
        },
    }


def _case_wrapper(
    *,
    question_id: str,
    memory_id: str,
    memory_wrapper: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "fixed_exemplar_case",
        "status": "success",
        "question_id": question_id,
        "retrieved_memory_question_id": memory_id,
        "worked_exemplar": memory_wrapper["worked_exemplar"],
        "hgf_provenance": {
            "blueprint_schema": "hgf_blueprint_topology_v2",
            "blueprint_sha256": memory_wrapper["blueprint_sha256"],
            "memory_exemplar_sha256": _canonical_hash(
                memory_wrapper["worked_exemplar"]
            ),
        },
    }


def _existing_wrappers(output_dir: Path) -> dict[str, dict[str, Any]]:
    wrappers: dict[str, dict[str, Any]] = {}
    for path in (output_dir / "memory").glob("*.json"):
        payload = _read(path)
        memory_id = str(payload.get("memory_question_id") or "")
        if memory_id:
            wrappers[memory_id] = payload
    return wrappers


def _valid_existing_wrapper(
    *,
    wrapper: dict[str, Any],
    memory_id: str,
    blueprint_sha256: str,
    memory_questions: dict[str, Any],
    graphs: dict[str, dict[str, Any]],
) -> bool:
    worked = wrapper.get("worked_exemplar")
    return (
        wrapper.get("schema_version") == "hgf_memory_exemplar_v1"
        and str(wrapper.get("memory_question_id") or "") == memory_id
        and str(wrapper.get("blueprint_sha256") or "")
        == blueprint_sha256
        and isinstance(worked, dict)
        and not _exemplar_errors(
            memory_id=memory_id,
            worked_exemplar=worked,
            memory_questions=memory_questions,
            graphs=graphs,
        )
    )


def _seed_bank(seed_dirs: list[Path]) -> dict[str, dict[str, Any]]:
    return load_fixed_exemplar_bank(seed_dirs)


def _build_manifest(
    *,
    wrappers: dict[str, dict[str, Any]],
    fixed_selection: dict[str, Any],
) -> dict[str, Any]:
    memory_entries = []
    for memory_id in sorted(wrappers):
        wrapper = wrappers[memory_id]
        memory_entries.append(
            {
                "memory_question_id": memory_id,
                "memory_path": f"memory/{memory_id}.json",
                "blueprint_sha256": wrapper["blueprint_sha256"],
                "worked_exemplar_sha256": _canonical_hash(
                    wrapper["worked_exemplar"]
                ),
                "generation_source": wrapper["generation"]["source"],
            }
        )
    case_entries = [
        {
            "question_id": str(entry["question_id"]),
            "memory_question_id": str(entry["memory_question_id"]),
            "case_path": f"cases/{entry['question_id']}.json",
        }
        for entry in fixed_selection.get("entries", [])
    ]
    return {
        "schema_version": "hgf_exemplar_manifest_v1",
        "artifact_name": "HGF",
        "memory_count": len(memory_entries),
        "fixed_case_count": len(case_entries),
        "memory_entries": memory_entries,
        "case_entries": case_entries,
    }


def verify_exemplar_bank(
    *,
    output_dir: Path,
    memory_questions: dict[str, Any],
    graphs: dict[str, dict[str, Any]],
    blueprint_hashes: dict[str, str],
    fixed_selection: dict[str, Any],
) -> dict[str, Any]:
    errors: list[str] = []
    wrappers = _existing_wrappers(output_dir)
    expected_memory_ids = set(memory_questions)
    if set(wrappers) != expected_memory_ids:
        errors.append(
            "memory Exemplar coverage mismatch; "
            f"missing={sorted(expected_memory_ids - set(wrappers))}, "
            f"extra={sorted(set(wrappers) - expected_memory_ids)}"
        )
    for memory_id in sorted(expected_memory_ids & set(wrappers)):
        if not _valid_existing_wrapper(
            wrapper=wrappers[memory_id],
            memory_id=memory_id,
            blueprint_sha256=blueprint_hashes[memory_id],
            memory_questions=memory_questions,
            graphs=graphs,
        ):
            errors.append(f"invalid memory Exemplar: {memory_id}")

    expected_case_ids = {
        str(entry["question_id"])
        for entry in fixed_selection.get("entries", [])
    }
    actual_case_paths = list((output_dir / "cases").glob("*.json"))
    actual_case_ids = {path.stem for path in actual_case_paths}
    if actual_case_ids != expected_case_ids:
        errors.append(
            "fixed Exemplar coverage mismatch; "
            f"missing={sorted(expected_case_ids - actual_case_ids)}, "
            f"extra={sorted(actual_case_ids - expected_case_ids)}"
        )
    for entry in fixed_selection.get("entries", []):
        question_id = str(entry["question_id"])
        memory_id = str(entry["memory_question_id"])
        path = output_dir / "cases" / f"{question_id}.json"
        if not path.is_file() or memory_id not in wrappers:
            continue
        expected = _case_wrapper(
            question_id=question_id,
            memory_id=memory_id,
            memory_wrapper=wrappers[memory_id],
        )
        if _read(path) != expected:
            errors.append(f"fixed Exemplar differs from memory bank: {path}")

    manifest_path = output_dir / "manifest.json"
    expected_manifest = _build_manifest(
        wrappers=wrappers,
        fixed_selection=fixed_selection,
    )
    if not manifest_path.is_file():
        errors.append(f"missing Exemplar manifest: {manifest_path}")
    elif _read(manifest_path) != expected_manifest:
        errors.append("Exemplar manifest differs from bank contents")
    return {
        "status": "pass" if not errors else "fail",
        "errors": errors,
        "memory_count": len(wrappers),
        "fixed_case_count": len(actual_case_ids),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--blueprint-root",
        type=Path,
        default=PACKAGE_ROOT / "artifacts" / "hgf" / "blueprints",
    )
    parser.add_argument(
        "--memory-manifest",
        type=Path,
        default=PACKAGE_ROOT / "data" / "memory_bank" / "manifest.json",
    )
    parser.add_argument(
        "--memory-questions",
        type=Path,
        default=PACKAGE_ROOT
        / "data"
        / "questions"
        / "memory_questions.jsonl",
    )
    parser.add_argument(
        "--fixed-selection",
        type=Path,
        default=PACKAGE_ROOT
        / "data"
        / "memory_bank"
        / "fixed_exemplar_selection.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PACKAGE_ROOT / "artifacts" / "hgf" / "exemplars",
    )
    parser.add_argument(
        "--seed-dir",
        type=Path,
        action="append",
        default=[],
        help="Reuse valid generated Exemplar content from this artifact root.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=PACKAGE_ROOT / "artifacts" / ".cache" / "hgf_exemplars",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument(
        "--reasoning-effort",
        choices=("minimal", "low", "medium", "high", "xhigh", "max"),
        default=None,
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify the canonical bank without model calls or writes.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not 1 <= args.workers <= 30:
        raise ValueError("--workers must be between 1 and 30")
    if args.attempts <= 0:
        raise ValueError("--attempts must be positive")
    if args.max_tokens <= 0:
        raise ValueError("--max-tokens must be positive")

    blueprint_root = args.blueprint_root.resolve()
    output_dir = args.output_dir.resolve()
    blueprints, blueprint_hashes = _load_blueprints(blueprint_root)
    memory_questions = {
        str(question.id): question
        for question in read_questions(args.memory_questions.resolve())
    }
    graphs = _load_graphs(args.memory_manifest.resolve())
    fixed_selection = _read(args.fixed_selection.resolve())
    if set(blueprints) != set(memory_questions) or set(graphs) != set(
        memory_questions
    ):
        raise ValueError("question, graph, and Blueprint banks do not align")

    if args.check:
        report = verify_exemplar_bank(
            output_dir=output_dir,
            memory_questions=memory_questions,
            graphs=graphs,
            blueprint_hashes=blueprint_hashes,
            fixed_selection=fixed_selection,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        if report["errors"]:
            raise SystemExit(1)
        return

    configure_generation(reasoning_effort=args.reasoning_effort)
    wrappers = _existing_wrappers(output_dir)
    completed: dict[str, dict[str, Any]] = {
        memory_id: wrapper
        for memory_id, wrapper in wrappers.items()
        if memory_id in memory_questions
        and _valid_existing_wrapper(
            wrapper=wrapper,
            memory_id=memory_id,
            blueprint_sha256=blueprint_hashes[memory_id],
            memory_questions=memory_questions,
            graphs=graphs,
        )
    }
    seeds = _seed_bank([path.resolve() for path in args.seed_dir])
    for memory_id in sorted(set(memory_questions) - set(completed)):
        worked = seeds.get(memory_id)
        if not isinstance(worked, dict):
            continue
        errors = _exemplar_errors(
            memory_id=memory_id,
            worked_exemplar=worked,
            memory_questions=memory_questions,
            graphs=graphs,
        )
        if errors:
            continue
        completed[memory_id] = _memory_wrapper(
            memory_id=memory_id,
            worked_exemplar=worked,
            blueprint_sha256=blueprint_hashes[memory_id],
            model=args.model,
            max_tokens=args.max_tokens,
            generation_source="validated_seed",
        )

    pending = sorted(set(memory_questions) - set(completed))
    client: OpenAI | None = None
    if pending:
        load_dotenv(find_dotenv(usecwd=True))
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY is required to generate missing Exemplars"
            )
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            timeout=180,
            max_retries=2,
        )

    def generate_one(memory_id: str) -> tuple[str, dict[str, Any]]:
        assert client is not None
        last_error: Exception | None = None
        for _attempt in range(1, args.attempts + 1):
            try:
                worked, _usage, _seconds, _cached = _distill_exemplar(
                    client=client,
                    model=args.model,
                    memory_question=memory_questions[memory_id],
                    graph=graphs[memory_id],
                    blueprint=blueprints[memory_id],
                    cache_dir=args.cache_dir.resolve(),
                    max_tokens=args.max_tokens,
                )
                errors = _exemplar_errors(
                    memory_id=memory_id,
                    worked_exemplar=worked,
                    memory_questions=memory_questions,
                    graphs=graphs,
                )
                if errors:
                    raise ValueError("; ".join(errors))
                return memory_id, _memory_wrapper(
                    memory_id=memory_id,
                    worked_exemplar=worked,
                    blueprint_sha256=blueprint_hashes[memory_id],
                    model=args.model,
                    max_tokens=args.max_tokens,
                    generation_source="canonical_generator",
                )
            except Exception as exc:  # provider errors are retried per memory
                last_error = exc
        assert last_error is not None
        raise last_error

    failures: dict[str, str] = {}
    if pending:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(generate_one, memory_id): memory_id
                for memory_id in pending
            }
            for future in as_completed(futures):
                memory_id = futures[future]
                try:
                    _, wrapper = future.result()
                except Exception as exc:
                    failures[memory_id] = f"{type(exc).__name__}: {exc}"
                else:
                    completed[memory_id] = wrapper
                    _atomic_write(
                        output_dir / "memory" / f"{memory_id}.json",
                        wrapper,
                    )
                print(
                    f"PROGRESS {len(completed)}/{len(memory_questions)} "
                    f"failed={len(failures)}",
                    flush=True,
                )

    for memory_id, wrapper in completed.items():
        _atomic_write(
            output_dir / "memory" / f"{memory_id}.json",
            wrapper,
        )
    if failures:
        print(json.dumps({"failures": failures}, ensure_ascii=False, indent=2))
        raise SystemExit(1)
    if set(completed) != set(memory_questions):
        raise RuntimeError("Exemplar bank remains incomplete")

    for entry in fixed_selection.get("entries", []):
        question_id = str(entry["question_id"])
        memory_id = str(entry["memory_question_id"])
        _atomic_write(
            output_dir / "cases" / f"{question_id}.json",
            _case_wrapper(
                question_id=question_id,
                memory_id=memory_id,
                memory_wrapper=completed[memory_id],
            ),
        )
    _atomic_write(
        output_dir / "manifest.json",
        _build_manifest(
            wrappers=completed,
            fixed_selection=fixed_selection,
        ),
    )
    report = verify_exemplar_bank(
        output_dir=output_dir,
        memory_questions=memory_questions,
        graphs=graphs,
        blueprint_hashes=blueprint_hashes,
        fixed_selection=fixed_selection,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
