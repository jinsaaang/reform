"""HGF component ablations layered on the frozen v27 implementation."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from statistics import fmean
from typing import Any, Callable

from dotenv import find_dotenv, load_dotenv
from openai import OpenAI

from hgf.baselines import _condition_evidence, _method_metrics
from hgf.boundary import (
    _boundary_schema,
    _call_boundary_mapping,
    _validate_boundary_forecast,
)
from hgf.contracts import _target_contract
from hgf.exemplar import (
    _add_usage,
    _call_with_repair,
    _ensure_baseline_reasoning_step,
    _normalize_probability_rows,
    _rerank_current_evidence,
)
from hgf.experiment_common import (
    ABLATION_CONDITIONS,
    provenance_snapshot,
    read_json,
    utc_now,
    write_json,
)
from hgf.forecast_core import _ground_truth_option, _probabilities, _seed
from hgf.generation import configure_generation
from hgf.memory_bank import load_final_memory_bank
from hgf.memory_retrieval import compile_raw_dag_ablation
from hgf.package import PACKAGE_ROOT
from hgf.question_io import (
    family_metadata,
    read_questions,
    resolve_forecast_cutoff,
)
from hgf.runner import (
    _call_dag_expert_reasoning,
    _inject_target_operator_step,
    _load_source_cases,
    _reasoning_schema,
    _wire_expert_memory,
    canonical_semantic_lessons,
    compile_current_target_operator,
    compile_dag_expert_memory,
)


_WRITE_LOCK = threading.Lock()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions-dir", type=Path, default=Path("data/questions"))
    parser.add_argument("--evidence-dir", type=Path, default=Path("data/evidence"))
    parser.add_argument(
        "--memory-bank-manifest",
        type=Path,
        default=Path("data/memory_bank/manifest.json"),
    )
    parser.add_argument(
        "--selection-file",
        type=Path,
        default=Path("data/questions/selection.json"),
    )
    parser.add_argument(
        "--hgf-artifact-root",
        type=Path,
        default=Path("artifacts/hgf"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/paper_ablation_v27"),
    )
    parser.add_argument("--model", default="google/gemini-2.5-flash-lite")
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=ABLATION_CONDITIONS,
        default=list(ABLATION_CONDITIONS),
    )
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--question-ids", nargs="*")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--candidate-evidence-limit", type=int, default=80)
    parser.add_argument("--evidence-limit", type=int, default=20)
    parser.add_argument("--reasoning-max-tokens", type=int, default=2600)
    parser.add_argument("--boundary-max-tokens", type=int, default=1800)
    parser.add_argument(
        "--reasoning-effort",
        choices=("minimal", "low", "medium", "high", "xhigh", "max"),
    )
    parser.add_argument("--max-output-tokens", type=int)
    parser.add_argument("--run-seed", type=int, default=0)
    return parser.parse_args()


def transform_expert_memory(
    expert_memory: dict[str, Any],
    condition: str,
) -> dict[str, Any]:
    """Apply exactly the named structural removal to a copied memory."""
    payload = copy.deepcopy(expert_memory)
    demonstration = payload.get("expert_reasoning_demonstration", {})
    lessons = payload.get("dag_derived_semantic_lessons", {})
    if condition == "without_counterevidence":
        demonstration.pop("counterevidence", None)
        for mechanism in payload.get("mechanism_library", []):
            mechanism.pop("fails_when", None)
        payload.pop("alternative_explanations", None)
        lessons.pop("counterevidence_lesson", None)
    elif condition == "without_target_bridge":
        removed = {
            str(item.get("checkpoint_id"))
            for item in payload.get("causal_checkpoint_library", [])
            if item.get("causal_role") == "target_bridge"
        }
        payload["causal_checkpoint_library"] = [
            item
            for item in payload.get("causal_checkpoint_library", [])
            if str(item.get("checkpoint_id")) not in removed
        ]
        for mechanism in payload.get("mechanism_library", []):
            mechanism["checkpoint_ids"] = [
                value
                for value in mechanism.get("checkpoint_ids", [])
                if str(value) not in removed
            ]
    elif condition == "without_uncertainty":
        demonstration.pop("uncertainty", None)
        lessons.pop("calibration_lesson", None)
    return payload


def _condition_schema(
    options: list[str],
    checkpoint_ids: list[str],
    condition: str,
) -> dict[str, Any]:
    schema = _reasoning_schema(options, checkpoint_ids)
    properties = schema["schema"]["properties"]
    required = schema["schema"]["required"]
    step_type = properties["reasoning_steps"]["items"]["properties"]["step_type"]
    if condition == "without_counterevidence":
        properties.pop("counterevidence", None)
        required.remove("counterevidence")
        step_type["enum"] = [
            value for value in step_type["enum"] if value != "counterevidence"
        ]
    if condition == "without_target_bridge":
        for field in ("target_estimate", "option_mapping"):
            properties.pop(field, None)
            required.remove(field)
        step_type["enum"] = [
            value for value in step_type["enum"] if value != "target_bridge"
        ]
    if condition == "without_uncertainty":
        properties.pop("uncertainty", None)
        required.remove("uncertainty")
    schema["name"] = f"hgf_reasoning_{condition}"
    return schema


def _condition_validator(
    *,
    options: list[str],
    evidence_ids: set[str],
    checkpoint_ids: set[str],
    condition: str,
) -> Callable[[dict[str, Any]], tuple[dict[str, float], list[str]]]:
    def validate(payload: dict[str, Any]) -> tuple[dict[str, float], list[str]]:
        _normalize_probability_rows(payload, options)
        _ensure_baseline_reasoning_step(
            payload,
            source_checkpoint_id="TARGET_CONTRACT",
        )
        errors: list[str] = []
        used = {str(value) for value in payload.get("selected_evidence_ids", [])}
        unknown = used - evidence_ids
        if not used:
            errors.append("selected_evidence_ids is empty")
        if unknown:
            errors.append(f"unknown current evidence IDs {sorted(unknown)}")
        probabilities, probability_errors = _probabilities(payload, options)
        errors.extend(probability_errors)
        if not str(payload.get("target_semantics") or "").strip():
            errors.append("target_semantics is empty")
        assessment = payload.get("evidence_fit", {}).get("assessment")
        if not str(assessment or "").strip():
            errors.append("evidence_fit assessment is empty")

        allowed_sources = checkpoint_ids | {"CURRENT_NEW", "TARGET_CONTRACT"}
        steps = payload.get("reasoning_steps", [])
        used_checkpoints = set()
        for step in steps:
            source_id = str(step.get("source_checkpoint_id") or "")
            if source_id not in allowed_sources:
                errors.append(f"unknown DAG checkpoint mapping {source_id!r}")
            if source_id in checkpoint_ids:
                used_checkpoints.add(source_id)
            cleaned = [
                str(value)
                for value in step.get("evidence_ids", [])
                if str(value) not in allowed_sources
            ]
            step["evidence_ids"] = cleaned
            unknown_step_ids = set(cleaned) - evidence_ids
            if unknown_step_ids:
                errors.append(
                    f"reasoning step uses unknown evidence IDs "
                    f"{sorted(unknown_step_ids)}"
                )
        if checkpoint_ids and not used_checkpoints:
            errors.append("reasoning must instantiate a retrieved DAG node")
        step_types = {str(step.get("step_type")) for step in steps}
        if not step_types & {"driver", "mechanism"}:
            errors.append("reasoning requires a driver or mechanism step")

        if condition == "without_counterevidence":
            if "counterevidence" in payload or "counterevidence" in step_types:
                errors.append("counterevidence was reintroduced")
        elif not str(payload.get("counterevidence") or "").strip():
            errors.append("counterevidence is empty")

        if condition == "without_target_bridge":
            if (
                "target_estimate" in payload
                or "option_mapping" in payload
                or "target_bridge" in step_types
            ):
                errors.append("target bridge was reintroduced")
        else:
            if "target_bridge" not in step_types:
                errors.append("reasoning requires a target_bridge step")
            if not str(payload.get("target_estimate") or "").strip():
                errors.append("target_estimate is empty")

        if condition == "without_uncertainty":
            if "uncertainty" in payload:
                errors.append("uncertainty was reintroduced")
        elif not str(payload.get("uncertainty") or "").strip():
            errors.append("uncertainty is empty")
        return probabilities, errors

    return validate


def _reasoning_prompt(
    *,
    condition: str,
    public_case: dict[str, Any],
    target_operator: dict[str, Any],
    memory_payload: dict[str, Any],
    evidence: list[dict[str, Any]],
) -> str:
    condition_instruction = {
        "raw_dag": (
            "Use the redacted raw DAG directly. Do not reconstruct a worked "
            "exemplar or DAG Expert Memory."
        ),
        "full_hgf": (
            "Use checkpoint requirements, mechanisms, failure conditions, "
            "counterevidence, target bridge, and uncertainty."
        ),
        "without_counterevidence": (
            "The counterevidence path and failure conditions are ablated. Do "
            "not emit a counterevidence field or counterevidence reasoning step."
        ),
        "without_target_bridge": (
            "The target bridge and prospective target estimate are ablated. Do "
            "not emit target_estimate, option_mapping, or target_bridge steps."
        ),
        "without_uncertainty": (
            "The uncertainty component is ablated. Do not emit an uncertainty "
            "field or uncertainty instruction."
        ),
    }[condition]
    return (
        "Forecast the unresolved target using only current cutoff-safe evidence. "
        "Historical DAG material is procedural memory, never current evidence. "
        "Lock the exact target contract, instantiate retrieved DAG nodes with "
        "current evidence, and return schema-conforming JSON.\n\n"
        f"EXPERIMENT CONDITION:\n{condition_instruction}\n\n"
        f"CURRENT CASE:\n{json.dumps(public_case, ensure_ascii=False)}\n\n"
        "DETERMINISTIC TARGET OPERATOR:\n"
        f"{json.dumps(target_operator, ensure_ascii=False)}\n\n"
        f"MEMORY INPUT:\n{json.dumps(memory_payload, ensure_ascii=False)}\n\n"
        f"CURRENT EVIDENCE:\n{json.dumps(evidence, ensure_ascii=False)}"
    )


def _call_condition_reasoning(
    *,
    client: OpenAI,
    model: str,
    question_id: str,
    condition: str,
    public_case: dict[str, Any],
    target_operator: dict[str, Any],
    memory_payload: dict[str, Any],
    checkpoint_ids: list[str],
    evidence: list[dict[str, Any]],
    evidence_ids: set[str],
    options: list[str],
    max_tokens: int,
) -> tuple[dict[str, Any], dict[str, int], float, bool]:
    reasoning, _, usage, seconds, repaired = _call_with_repair(
        client,
        model=model,
        system=(
            "You are a cutoff-safe financial forecaster in a controlled HGF "
            "component-ablation experiment. Return only valid JSON."
        ),
        prompt=_reasoning_prompt(
            condition=condition,
            public_case=public_case,
            target_operator=target_operator,
            memory_payload=memory_payload,
            evidence=evidence,
        ),
        schema=_condition_schema(options, checkpoint_ids, condition),
        seed=_seed(question_id, f"ablation:{condition}:reasoning"),
        max_tokens=max_tokens,
        validator=_condition_validator(
            options=options,
            evidence_ids=evidence_ids,
            checkpoint_ids=set(checkpoint_ids),
            condition=condition,
        ),
    )
    _inject_target_operator_step(reasoning, target_operator)
    return reasoning, usage, seconds, repaired


def _call_boundary_without_uncertainty(
    *,
    client: OpenAI,
    model: str,
    question_id: str,
    public_case: dict[str, Any],
    evidence: list[dict[str, Any]],
    evidence_ids: set[str],
    options: list[str],
    contract: dict[str, Any],
    reasoning: dict[str, Any],
    max_tokens: int,
) -> tuple[
    dict[str, Any],
    dict[str, float],
    dict[str, int],
    float,
    bool,
]:
    schema = _boundary_schema(options, "boundary_only")
    schema["name"] = "boundary_without_uncertainty"
    schema["schema"]["properties"].pop("uncertainty")
    schema["schema"]["required"].remove("uncertainty")
    reasoning_view = {
        key: value
        for key, value in reasoning.items()
        if key not in {"prediction", "option_probabilities", "uncertainty"}
    }
    prompt = (
        "Map the supplied current-case reasoning to the exact target boundaries. "
        "Produce low/central/high target estimates, check every public interval, "
        "make the arithmetically mapped option modal, and allocate probabilities "
        "from estimate-range overlap. The uncertainty component is intentionally "
        "absent in this controlled condition.\n\n"
        f"CURRENT CASE:\n{json.dumps(public_case, ensure_ascii=False)}\n\n"
        f"REASONING:\n{json.dumps(reasoning_view, ensure_ascii=False)}\n\n"
        f"CURRENT EVIDENCE:\n{json.dumps(evidence, ensure_ascii=False)}"
    )

    def validator(
        payload: dict[str, Any],
    ) -> tuple[dict[str, float], list[str]]:
        payload["uncertainty"] = "ABLATION_SENTINEL"
        try:
            return _validate_boundary_forecast(
                payload,
                options=options,
                contract=contract,
                evidence_ids=evidence_ids,
                reasoning_policy="boundary_only",
                validation_policy="recovery",
            )
        finally:
            payload.pop("uncertainty", None)

    return _call_with_repair(
        client,
        model=model,
        system=(
            "You are a boundary mapper for a controlled uncertainty ablation. "
            "Return only schema-conforming JSON."
        ),
        prompt=prompt,
        schema=schema,
        seed=_seed(question_id, "ablation:without_uncertainty:boundary"),
        max_tokens=max_tokens,
        validator=validator,
    )


def _run_case(
    *,
    client: OpenAI,
    model: str,
    condition: str,
    question: Any,
    evidence_dir: Path,
    output_dir: Path,
    graphs_by_id: dict[str, dict[str, Any]],
    blueprints_by_id: dict[str, dict[str, Any]],
    exemplar_cases: dict[str, dict[str, Any]],
    candidate_evidence_limit: int,
    evidence_limit: int,
    reasoning_max_tokens: int,
    boundary_max_tokens: int,
) -> dict[str, Any]:
    output_path = output_dir / "cases" / str(question.id) / f"{condition}.json"
    failed_path = output_path.with_suffix(".failed.json")
    if output_path.is_file():
        cached = read_json(output_path)
        if cached.get("status") == "success":
            failed_path.unlink(missing_ok=True)
            return cached
    started = time.monotonic()
    cutoff, cutoff_source = resolve_forecast_cutoff(question)
    db_path, candidates = _condition_evidence(
        evidence_dir,
        question,
        cutoff,
        guided=True,
        limit=candidate_evidence_limit,
    )
    evidence = _rerank_current_evidence(
        question,
        candidates,
        limit=evidence_limit,
    )
    evidence_ids = {str(item["id"]) for item in evidence}
    options = [str(value) for value in question.options or []]
    contract = _target_contract(question)
    public_case = {
        "question": question.question_text,
        "context": question.context,
        "target_contract": contract,
        "cutoff": cutoff.isoformat(),
    }
    target_operator = compile_current_target_operator(contract)
    fixed_case = exemplar_cases[str(question.id)]
    memory_id = str(fixed_case["retrieved_memory_question_id"])
    blueprint = blueprints_by_id[memory_id]
    graph = graphs_by_id[memory_id]
    canonical_expert_memory: dict[str, Any] | None = None
    if condition == "raw_dag":
        raw_memory = json.loads(compile_raw_dag_ablation([graph]))
        memory_payload = raw_memory
        checkpoint_ids = [
            str(node["id"])
            for example in raw_memory.get("examples", [])
            for node in example.get("factor_nodes", [])
        ]
    else:
        expert_memory = compile_dag_expert_memory(
            source_question_id=memory_id,
            blueprint=blueprint,
            worked_exemplar=fixed_case["worked_exemplar"],
        )
        expert_memory["dag_derived_semantic_lessons"] = (
            canonical_semantic_lessons()
        )
        expert_memory = transform_expert_memory(expert_memory, condition)
        canonical_expert_memory = expert_memory
        checkpoint_ids = [
            str(item["checkpoint_id"])
            for item in expert_memory.get("causal_checkpoint_library", [])
        ]
        memory_payload = _wire_expert_memory(expert_memory)

    if condition == "full_hgf":
        if canonical_expert_memory is None:
            raise AssertionError("canonical Full HGF memory was not built")
        reasoning_result = _call_dag_expert_reasoning(
            client=client,
            model=model,
            question_id=str(question.id),
            public_case=public_case,
            evidence=evidence,
            evidence_ids=evidence_ids,
            options=options,
            expert_memory=canonical_expert_memory,
            target_operator=target_operator,
            max_tokens=reasoning_max_tokens,
            allow_memory_rejection=True,
        )
    else:
        reasoning_result = _call_condition_reasoning(
            client=client,
            model=model,
            question_id=str(question.id),
            condition=condition,
            public_case=public_case,
            target_operator=target_operator,
            memory_payload=memory_payload,
            checkpoint_ids=checkpoint_ids,
            evidence=evidence,
            evidence_ids=evidence_ids,
            options=options,
            max_tokens=reasoning_max_tokens,
        )
    reasoning, reasoning_usage, reasoning_seconds, reasoning_repaired = (
        reasoning_result
    )
    if condition == "without_uncertainty":
        boundary_result = _call_boundary_without_uncertainty(
            client=client,
            model=model,
            question_id=str(question.id),
            public_case=public_case,
            evidence=evidence,
            evidence_ids=evidence_ids,
            options=options,
            contract=contract,
            reasoning=reasoning,
            max_tokens=boundary_max_tokens,
        )
    else:
        boundary_result = _call_boundary_mapping(
            client=client,
            model=model,
            question_id=str(question.id),
            public_case=public_case,
            evidence=evidence,
            evidence_ids=evidence_ids,
            options=options,
            contract=contract,
            reasoning=reasoning,
            seed_role=(
                "boundary_mapping"
                if condition == "full_hgf"
                else f"ablation:{condition}:boundary"
            ),
            max_tokens=boundary_max_tokens,
            allow_neutral_fallback=True,
        )
    (
        forecast,
        probabilities,
        boundary_usage,
        boundary_seconds,
        boundary_repaired,
    ) = boundary_result
    ground_truth = _ground_truth_option(question)
    metrics = _method_metrics(probabilities, ground_truth, options)
    result = {
        "schema_version": "hgf_ablation_case_v1",
        "status": "success",
        "condition": condition,
        "method": condition,
        "model": model,
        "question_id": str(question.id),
        "category": family_metadata(question).get("category"),
        "question": question.question_text,
        "options": options,
        "ground_truth": ground_truth,
        "cutoff": cutoff.isoformat(),
        "cutoff_source": cutoff_source,
        "evidence_bank": "E1",
        "evidence_db": str(db_path),
        "evidence": evidence,
        "evidence_ids": sorted(evidence_ids),
        "retrieved_memory_question_id": memory_id,
        "memory": memory_payload,
        "memory_cached": True,
        "reasoning": reasoning,
        "forecast": forecast,
        "probabilities": probabilities,
        "metrics": metrics,
        "usage": _add_usage(reasoning_usage, boundary_usage),
        "seconds": reasoning_seconds + boundary_seconds,
        "elapsed_seconds": time.monotonic() - started,
        "repaired": reasoning_repaired or boundary_repaired,
    }
    with _WRITE_LOCK:
        write_json(output_path, result)
        failed_path.unlink(missing_ok=True)
    return result


def _summary(
    rows: list[dict[str, Any]],
    conditions: list[str],
    selected_count: int,
    elapsed_seconds: float,
) -> dict[str, Any]:
    output: dict[str, Any] = {
        "selected_questions": selected_count,
        "expected_runs": selected_count * len(conditions),
        "completed_runs": sum(row.get("status") == "success" for row in rows),
        "failed_runs": sum(row.get("status") != "success" for row in rows),
        "elapsed_seconds": elapsed_seconds,
        "overall": {},
    }
    for condition in conditions:
        bucket = [
            row
            for row in rows
            if row.get("status") == "success"
            and row.get("condition") == condition
        ]
        if bucket:
            output["overall"][condition] = {
                "n": len(bucket),
                "accuracy": fmean(row["metrics"]["accuracy"] for row in bucket),
                "brier": fmean(row["metrics"]["brier"] for row in bucket),
                "nll": fmean(row["metrics"]["nll"] for row in bucket),
                "mean_total_tokens": fmean(
                    float(row.get("usage", {}).get("total_tokens") or 0)
                    for row in bucket
                ),
                "mean_seconds": fmean(
                    float(row.get("seconds") or 0) for row in bucket
                ),
            }
    return output


def main() -> None:
    args = _parse_args()
    if args.workers != 4:
        raise ValueError("experiments.md fixes workers at 4")
    configure_generation(
        reasoning_effort=args.reasoning_effort,
        max_output_tokens=args.max_output_tokens,
        run_seed=args.run_seed,
    )
    load_dotenv(find_dotenv(usecwd=True))
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")
    root = PACKAGE_ROOT
    questions_dir = args.questions_dir.resolve()
    evidence_dir = args.evidence_dir.resolve()
    output_dir = args.output_dir.resolve()
    hgf_artifact_root = args.hgf_artifact_root.resolve()
    questions = {
        str(question.id): question
        for question in read_questions(questions_dir / "test_questions.jsonl")
    }
    memory_questions = {
        str(question.id): question
        for question in read_questions(questions_dir / "memory_questions.jsonl")
    }
    graphs, blueprints = load_final_memory_bank(
        args.memory_bank_manifest.resolve(),
        memory_questions,
        hgf_artifact_root=hgf_artifact_root / "blueprints",
    )
    graphs_by_id = {
        str(blueprint["question_id"]): graph
        for graph, blueprint in zip(graphs, blueprints)
    }
    blueprints_by_id = {
        str(blueprint["question_id"]): blueprint for blueprint in blueprints
    }
    exemplar_cases = _load_source_cases(hgf_artifact_root / "exemplars")
    frozen = read_json(args.selection_file.resolve())["question_ids"]
    requested = set(args.question_ids or frozen)
    missing = sorted(requested - set(questions))
    if missing:
        raise ValueError(f"unknown question IDs: {missing}")
    selected_ids = [
        question_id
        for question_id in frozen
        if question_id in requested
    ][: args.limit]
    selected = [questions[question_id] for question_id in selected_ids]
    missing_exemplars = sorted(set(selected_ids) - set(exemplar_cases))
    if missing_exemplars:
        raise ValueError(f"missing fixed exemplars: {missing_exemplars}")

    write_json(
        output_dir / "protocol.json",
        {
            "schema_version": "hgf_ablation_protocol_v1",
            "model": args.model,
            "workers": args.workers,
            "run_seed": args.run_seed,
            "conditions": list(args.conditions),
            "question_ids": selected_ids,
            "evidence_bank": "E1",
            "started_at_utc": utc_now(),
            "provenance": provenance_snapshot(
                root=root,
                requested_model=args.model,
                run_seed=args.run_seed,
                config_paths=(Path("configs/experiments_v27.json"),),
                extra={"experiment": "component_ablation"},
            ),
        },
    )
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        timeout=180,
        max_retries=2,
    )
    started = time.monotonic()
    tasks = [
        (question, condition)
        for question in selected
        for condition in args.conditions
    ]
    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                _run_case,
                client=client,
                model=args.model,
                condition=condition,
                question=question,
                evidence_dir=evidence_dir,
                output_dir=output_dir,
                graphs_by_id=graphs_by_id,
                blueprints_by_id=blueprints_by_id,
                exemplar_cases=exemplar_cases,
                candidate_evidence_limit=args.candidate_evidence_limit,
                evidence_limit=args.evidence_limit,
                reasoning_max_tokens=args.reasoning_max_tokens,
                boundary_max_tokens=args.boundary_max_tokens,
            ): (question, condition)
            for question, condition in tasks
        }
        for future in as_completed(futures):
            question, condition = futures[future]
            try:
                row = future.result()
            except Exception as exc:
                row = {
                    "schema_version": "hgf_ablation_case_v1",
                    "status": "failed",
                    "condition": condition,
                    "method": condition,
                    "question_id": str(question.id),
                    "category": family_metadata(question).get("category"),
                    "error": f"{type(exc).__name__}: {exc}",
                }
                write_json(
                    output_dir
                    / "cases"
                    / str(question.id)
                    / f"{condition}.failed.json",
                    row,
                )
            rows.append(row)
            successful = sum(row.get("status") == "success" for row in rows)
            print(
                f"PROGRESS {len(rows)}/{len(tasks)} "
                f"success={successful} failed={len(rows)-successful}",
                flush=True,
            )
    order = {
        (question_id, condition): (
            question_index * len(args.conditions) + condition_index
        )
        for question_index, question_id in enumerate(selected_ids)
        for condition_index, condition in enumerate(args.conditions)
    }
    rows.sort(
        key=lambda row: order.get(
            (str(row["question_id"]), str(row["condition"])),
            math.inf,
        )
    )
    summary = _summary(
        rows,
        list(args.conditions),
        len(selected),
        time.monotonic() - started,
    )
    write_json(
        output_dir / "results.json",
        {
            "schema_version": "hgf_ablation_experiment_v1",
            "model": args.model,
            "workers": args.workers,
            "run_seed": args.run_seed,
            "conditions": list(args.conditions),
            "selection": {"question_ids": selected_ids},
            "summary": summary,
            "results": rows,
        },
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
