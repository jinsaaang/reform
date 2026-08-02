#!/usr/bin/env python3
"""Run independent single-forecast end-to-end topology HGF."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from statistics import fmean
from typing import Any

from dotenv import find_dotenv, load_dotenv
from openai import OpenAI

from hgf.baselines import _condition_evidence
from hgf.boundary import _call_boundary_mapping
from hgf.contracts import _is_temporally_eligible, _target_contract
from hgf.exemplar import _rerank_current_evidence
from hgf.exemplar_selection import load_fixed_exemplar_bank
from hgf.forecast_core import _atomic_write, _ground_truth_option
from hgf.forecast_safety import score_forecast
from hgf.generation import configure_generation
from hgf.memory_bank import load_hgf_blueprint_bank
from hgf.memory_retrieval import select_relevant_blueprints
from hgf.question_io import family_metadata, read_questions, resolve_forecast_cutoff
from hgf.runner import compile_current_target_operator

from .core import attach_graph_audit, call_procedural_topology_reasoning
from .instantiation import call_graph_instantiation, materialize_current_graph
from .pipeline import (
    add_usage,
    call_current_evidence_ledger,
    compile_topology_memory,
    route_topology_subgraphs,
    select_forecast_evidence,
)


METHOD = "procedural_topology_hgf_exemplar_check"
METHOD_LABEL = "Procedural Topology HGF with Worked Reasoning Check"
_WRITE_LOCK = threading.Lock()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions-dir", type=Path, default=Path("data/questions"))
    parser.add_argument("--evidence-dir", type=Path, default=Path("data/evidence"))
    parser.add_argument(
        "--selection-file",
        type=Path,
        default=Path("data/questions/selection.json"),
    )
    parser.add_argument(
        "--blueprint-root",
        type=Path,
        default=Path("artifacts/hgf/blueprints"),
    )
    parser.add_argument(
        "--exemplar-root",
        type=Path,
        default=Path("artifacts/hgf/exemplars"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/procedural_topology_hgf_medium_seed0"),
    )
    parser.add_argument("--model", default="google/gemini-2.5-flash-lite")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--candidate-evidence-limit", type=int, default=80)
    parser.add_argument("--evidence-limit", type=int, default=20)
    parser.add_argument("--forecast-evidence-limit", type=int, default=14)
    parser.add_argument("--max-dags", type=int, default=3)
    parser.add_argument("--max-paths", type=int, default=3)
    parser.add_argument("--max-checkpoints", type=int, default=12)
    parser.add_argument("--ledger-max-tokens", type=int, default=4000)
    parser.add_argument("--graph-max-tokens", type=int, default=5000)
    parser.add_argument("--reasoning-max-tokens", type=int, default=5000)
    parser.add_argument("--boundary-max-tokens", type=int, default=2400)
    parser.add_argument(
        "--reasoning-effort",
        choices=("minimal", "low", "medium", "high", "xhigh", "max"),
        default="medium",
    )
    parser.add_argument("--max-output-tokens", type=int, default=16000)
    parser.add_argument("--run-seed", type=int, default=0)
    parser.add_argument("--question-ids", nargs="*")
    return parser.parse_args()


def _load_stage(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "e2e_topology_ledger_stage_v1":
        return None
    return payload


def _select_compatible_blueprints(
    *,
    blueprints: list[dict[str, Any]],
    memory_questions: dict[str, Any],
    target_question: Any,
    cutoff: Any,
    evidence: list[dict[str, Any]],
    limit: int,
) -> list[dict[str, Any]]:
    target = family_metadata(target_question)
    eligible = []
    for blueprint in blueprints:
        question_id = str(blueprint.get("question_id") or "")
        memory_question = memory_questions.get(question_id)
        if memory_question is None or not _is_temporally_eligible(
            memory_question, cutoff
        ):
            continue
        memory = family_metadata(memory_question)
        if (
            str(memory.get("family_id") or "")
            != str(target.get("family_id") or "")
            or str(memory.get("target_metric") or "")
            != str(target.get("target_metric") or "")
        ):
            continue
        eligible.append(blueprint)
    selected = select_relevant_blueprints(
        eligible,
        memory_questions,
        target_question,
        limit=limit,
        evidence=evidence,
    )
    if not selected:
        raise ValueError(f"no exact-family topology memory for {target_question.id}")
    return selected


def _metrics(
    probabilities: dict[str, float],
    *,
    prediction: str,
    ground_truth: str,
    options: list[str],
) -> dict[str, float]:
    accuracy, brier = score_forecast(
        probabilities=probabilities,
        explicit_prediction=prediction,
        ground_truth=ground_truth,
        options=options,
    )
    return {
        "accuracy": accuracy,
        "brier": brier,
        "nll": -math.log(max(float(probabilities[ground_truth]), 1e-6)),
        "confidence": max(float(value) for value in probabilities.values()),
    }


def _run_case(
    *,
    client: OpenAI,
    model: str,
    question: Any,
    evidence_dir: Path,
    output_dir: Path,
    memory_questions: dict[str, Any],
    blueprints: list[dict[str, Any]],
    exemplars_by_id: dict[str, dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    case_dir = output_dir / "cases" / str(question.id)
    output_path = case_dir / f"{METHOD}.json"
    failed_path = case_dir / f"{METHOD}.failed.json"
    if output_path.is_file():
        cached = json.loads(output_path.read_text(encoding="utf-8"))
        if cached.get("status") == "success":
            failed_path.unlink(missing_ok=True)
            return cached

    started = time.monotonic()
    cutoff, cutoff_source = resolve_forecast_cutoff(question)
    options = [str(option) for option in question.options or []]
    contract = _target_contract(question)
    target_operator = compile_current_target_operator(contract)
    public_case = {
        "question": question.question_text,
        "context": question.context,
        "target_contract": contract,
        "cutoff": cutoff.isoformat(),
    }
    ground_truth = _ground_truth_option(question)
    db_path, candidates = _condition_evidence(
        evidence_dir,
        question,
        cutoff,
        guided=True,
        limit=args.candidate_evidence_limit,
    )
    evidence = _rerank_current_evidence(
        question, candidates, limit=args.evidence_limit
    )

    ledger_path = case_dir / "stages" / "evidence_ledger.json"
    ledger_stage = _load_stage(ledger_path)
    if ledger_stage is None:
        ledger, ledger_usage, ledger_seconds, ledger_repaired = (
            call_current_evidence_ledger(
                client=client,
                model=model,
                question_id=str(question.id),
                public_case=public_case,
                target_operator=target_operator,
                evidence=evidence,
                max_tokens=args.ledger_max_tokens,
            )
        )
        _atomic_write(
            ledger_path,
            {
                "schema_version": "e2e_topology_ledger_stage_v1",
                "payload": ledger,
                "usage": ledger_usage,
                "seconds": ledger_seconds,
                "repaired": ledger_repaired,
            },
        )
    else:
        ledger = ledger_stage["payload"]
        ledger_usage = {}
        ledger_seconds = 0.0
        ledger_repaired = bool(ledger_stage.get("repaired"))

    selected_blueprints = _select_compatible_blueprints(
        blueprints=blueprints,
        memory_questions=memory_questions,
        target_question=question,
        cutoff=cutoff,
        evidence=evidence,
        limit=args.max_dags,
    )
    full_memory = compile_topology_memory(
        selected_blueprints,
        exemplars_by_id,
    )
    routed_memory = route_topology_subgraphs(
        full_memory,
        ledger,
        max_paths=args.max_paths,
        max_checkpoints=args.max_checkpoints,
    )
    forecast_evidence = select_forecast_evidence(
        evidence,
        ledger,
        limit=args.forecast_evidence_limit,
    )

    instantiated_graph, graph_usage, graph_seconds, graph_repaired = (
        call_graph_instantiation(
            client=client,
            model=model,
            question_id=str(question.id),
            public_case=public_case,
            target_operator=target_operator,
            evidence=forecast_evidence,
            evidence_ledger=ledger,
            routed_memory=routed_memory,
            max_tokens=args.graph_max_tokens,
        )
    )
    current_graph = materialize_current_graph(routed_memory, instantiated_graph)
    reasoning, reasoning_usage, reasoning_seconds, reasoning_repaired = (
        call_procedural_topology_reasoning(
            client=client,
            model=model,
            question_id=str(question.id),
            public_case=public_case,
            target_operator=target_operator,
            evidence=forecast_evidence,
            evidence_ledger=ledger,
            current_graph=current_graph,
            worked_reasoning_checks=routed_memory["worked_reasoning_checks"],
            max_tokens=args.reasoning_max_tokens,
        )
    )
    reasoning = attach_graph_audit(
        reasoning,
        instantiated_graph=instantiated_graph,
        routed_memory=routed_memory,
    )
    forecast, probabilities, boundary_usage, boundary_seconds, boundary_repaired = (
        _call_boundary_mapping(
            client=client,
            model=model,
            question_id=str(question.id),
            public_case=public_case,
            evidence=forecast_evidence,
            evidence_ids={str(item["id"]) for item in forecast_evidence},
            options=options,
            contract=contract,
            reasoning=reasoning,
            seed_role="procedural-topology-boundary-v3",
            max_tokens=args.boundary_max_tokens,
            allow_neutral_fallback=True,
            allow_prospective_anchors=True,
        )
    )
    row = {
        "schema_version": "e2e_topology_hgf_exemplar_check_case_v1",
        "status": "success",
        "method": METHOD,
        "method_label": METHOD_LABEL,
        "question_id": str(question.id),
        "category": family_metadata(question).get("category"),
        "question": question.question_text,
        "options": options,
        "ground_truth": ground_truth,
        "cutoff": cutoff.isoformat(),
        "cutoff_source": cutoff_source,
        "evidence_bank": "E1",
        "evidence_db": str(db_path),
        "candidate_evidence_ids": [str(item["id"]) for item in evidence],
        "forecast_evidence_ids": [str(item["id"]) for item in forecast_evidence],
        "retrieved_memory_question_ids": [
            str(item["question_id"]) for item in selected_blueprints
        ],
        "memory": routed_memory,
        "evidence_ledger": ledger,
        "instantiated_graph": instantiated_graph,
        "current_graph": current_graph,
        "reasoning": reasoning,
        "forecast": forecast,
        "probabilities": probabilities,
        "metrics": _metrics(
            probabilities,
            prediction=str(forecast.get("prediction") or ""),
            ground_truth=ground_truth,
            options=options,
        ),
        "usage": add_usage(
            ledger_usage, graph_usage, reasoning_usage, boundary_usage
        ),
        "seconds": (
            ledger_seconds + graph_seconds + reasoning_seconds + boundary_seconds
        ),
        "repaired": (
            ledger_repaired
            or graph_repaired
            or reasoning_repaired
            or boundary_repaired
        ),
        "elapsed_seconds": time.monotonic() - started,
        "single_probability_call": True,
        "prior_prediction_visible": False,
        "prior_probabilities_visible": False,
        "probability_postprocessing": "none",
        "historical_exemplar": "answer-free worked reasoning check",
    }
    _atomic_write(output_path, row)
    failed_path.unlink(missing_ok=True)
    return row


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    successes = [row for row in rows if row.get("status") == "success"]

    def one(group: list[dict[str, Any]]) -> dict[str, float | int]:
        return {
            "count": len(group),
            "accuracy": fmean(row["metrics"]["accuracy"] for row in group),
            "brier": fmean(row["metrics"]["brier"] for row in group),
            "nll": fmean(row["metrics"]["nll"] for row in group),
        }

    categories = sorted({str(row.get("category")) for row in successes})
    return {
        "overall": one(successes) if successes else {},
        "by_category": {
            category: one(
                [row for row in successes if str(row.get("category")) == category]
            )
            for category in categories
        },
        "success_count": len(successes),
        "failed_count": len(rows) - len(successes),
        "repaired_count": sum(bool(row.get("repaired")) for row in successes),
    }


def main() -> None:
    args = _parse_args()
    configure_generation(
        reasoning_effort=args.reasoning_effort,
        max_output_tokens=args.max_output_tokens,
        run_seed=args.run_seed,
    )
    for variable in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ.setdefault(variable, "1")
    load_dotenv(find_dotenv(usecwd=True))
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")

    questions_dir = args.questions_dir.resolve()
    evidence_dir = args.evidence_dir.resolve()
    output_dir = args.output_dir.resolve()
    questions = {
        str(question.id): question
        for question in read_questions(questions_dir / "test_questions.jsonl")
    }
    memory_questions = {
        str(question.id): question
        for question in read_questions(questions_dir / "memory_questions.jsonl")
    }
    blueprints_by_id = load_hgf_blueprint_bank(
        args.blueprint_root.resolve(), expected_ids=set(memory_questions)
    )
    blueprints = [blueprints_by_id[question_id] for question_id in memory_questions]
    exemplars_by_id = load_fixed_exemplar_bank([args.exemplar_root.resolve()])
    missing_exemplars = sorted(set(memory_questions) - set(exemplars_by_id))
    if missing_exemplars:
        raise ValueError(
            f"fixed exemplar bank is missing {len(missing_exemplars)} memory cases"
        )
    selection_payload = json.loads(
        args.selection_file.resolve().read_text(encoding="utf-8")
    )
    selected_ids = list(selection_payload["question_ids"])
    if args.question_ids:
        requested = set(args.question_ids)
        selected_ids = [value for value in selected_ids if value in requested]
    selected_ids = selected_ids[: args.limit]
    selected = [questions[question_id] for question_id in selected_ids]

    blueprint_manifest = args.blueprint_root.resolve() / "manifest.json"
    exemplar_manifest = args.exemplar_root.resolve() / "manifest.json"
    _atomic_write(
        output_dir / "protocol.json",
        {
            "schema_version": "procedural_topology_hgf_exemplar_check_protocol_v1",
            "method": METHOD,
            "model": args.model,
            "workers": args.workers,
            "generation": {
                "reasoning_effort": args.reasoning_effort,
                "max_output_tokens": args.max_output_tokens,
                "run_seed": args.run_seed,
            },
            "selection_file": str(args.selection_file),
            "question_ids": selected_ids,
            "blueprint_manifest": str(
                args.blueprint_root / "manifest.json"
            ),
            "blueprint_manifest_sha256": hashlib.sha256(
                blueprint_manifest.read_bytes()
            ).hexdigest(),
            "exemplar_manifest": str(exemplar_manifest),
            "exemplar_manifest_sha256": hashlib.sha256(
                exemplar_manifest.read_bytes()
            ).hexdigest(),
            "retrieval": {
                "compatibility": "exact family_id and target_metric",
                "max_dags": args.max_dags,
                "after_current_evidence": True,
            },
            "routing": {
                "max_paths": args.max_paths,
                "max_checkpoints": args.max_checkpoints,
                "topology_rewrite": False,
            },
            "implementation_dependency": {
                "previous_experiment_packages": [],
                "previous_result_files": [],
                "dual_view": False,
            },
            "pipeline_stages": [
                "current_evidence_ledger",
                "exact_subgraph_retrieval",
                "current_graph_instantiation",
                "worked_reasoning_structure_check",
                "flexible_procedural_reasoning",
                "frozen_target_boundary_mapping",
            ],
            "single_probability_call": True,
            "prior_prediction_visible": False,
            "prior_probabilities_visible": False,
            "historical_exemplar": {
                "role": "answer-free reasoning completeness check",
                "shared_across_models": True,
                "excluded_fields": [
                    "prospective_target_estimate",
                    "option_mapping",
                    "forecast_time_evidence",
                ],
            },
            "probability_postprocessing": "none",
            "evidence_bank": "E1",
        },
    )
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        timeout=120,
        max_retries=1,
    )
    started = time.monotonic()
    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {
            executor.submit(
                _run_case,
                client=client,
                model=args.model,
                question=question,
                evidence_dir=evidence_dir,
                output_dir=output_dir,
                memory_questions=memory_questions,
                blueprints=blueprints,
                exemplars_by_id=exemplars_by_id,
                args=args,
            ): question
            for question in selected
        }
        for future in as_completed(futures):
            question = futures[future]
            try:
                row = future.result()
            except Exception as exc:
                row = {
                    "schema_version": "e2e_topology_hgf_exemplar_check_case_v1",
                    "status": "failed",
                    "method": METHOD,
                    "method_label": METHOD_LABEL,
                    "question_id": str(question.id),
                    "category": family_metadata(question).get("category"),
                    "error": f"{type(exc).__name__}: {exc}",
                }
                with _WRITE_LOCK:
                    _atomic_write(
                        output_dir
                        / "cases"
                        / str(question.id)
                        / f"{METHOD}.failed.json",
                        row,
                    )
            rows.append(row)
            successes = sum(item.get("status") == "success" for item in rows)
            print(
                f"PROGRESS {len(rows)}/{len(selected)} "
                f"success={successes} failed={len(rows) - successes}",
                flush=True,
            )
    order = {question_id: index for index, question_id in enumerate(selected_ids)}
    rows.sort(key=lambda row: order.get(str(row["question_id"]), 999999))
    payload = {
        "schema_version": "e2e_topology_hgf_exemplar_check_experiment_v1",
        "model": args.model,
        "workers": args.workers,
        "generation": {
            "reasoning_effort": args.reasoning_effort,
            "max_output_tokens": args.max_output_tokens,
            "run_seed": args.run_seed,
        },
        "selection": {"question_ids": selected_ids},
        "summary": _aggregate(rows),
        "elapsed_seconds": time.monotonic() - started,
        "results": rows,
    }
    _atomic_write(output_dir / "results.json", payload)
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
