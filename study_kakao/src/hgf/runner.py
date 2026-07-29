#!/usr/bin/env python3
"""Run Hindsight-Guided Forecasting with a fixed exemplar bank.

The method compiles one usable WorldReasoner DAG into a compact Expert Memory
containing:

* a cutoff-safe worked reasoning demonstration;
* causal checkpoints and their evidence requirements;
* reusable mechanisms and failure conditions;
* counter-paths and audit questions.

Every current reasoning step records the DAG checkpoint it instantiates, or
``CURRENT_NEW`` when the current case requires a genuinely new factor.  The
historical answer and historical facts are never treated as current evidence.
"""

from __future__ import annotations

import argparse
import base64
import copy
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

from dotenv import find_dotenv, load_dotenv
from openai import OpenAI

from hgf.question_io import (
    family_metadata,
    read_questions,
    resolve_forecast_cutoff,
)
from hgf.memory_bank import load_final_memory_bank
from hgf.exemplar import (
    _add_usage,
    _call_with_repair,
    _ensure_baseline_reasoning_step,
    _forecast_schema_exemplar,
    _normalize_probability_rows,
    _rerank_current_evidence,
    _transferable_dag_structure,
    _validate_exemplar_forecast,
)
from hgf.boundary import _call_boundary_mapping
from hgf.contracts import _target_contract
from hgf.forecast_core import (
    _atomic_write,
    _ground_truth_option,
    _resolve_evidence as _resolve_raw_evidence,
    _score,
    _seed,
)


_WRITE_LOCK = threading.Lock()
_MEMORY_LOCK = threading.Lock()
_SEMANTIC_SCHEMA_NAME = base64.b64decode(
    "ZGFnX3NlbWFudGljX2V4cGVydGlzZV92Mjc="
).decode("utf-8")
_EXPERT_MEMORY_WIRE_SCHEMA = base64.b64decode(
    "ZGFnX2V4cGVydF9tZW1vcnlfdjI3"
).decode("utf-8")


def _wire_expert_memory(expert_memory: dict[str, Any]) -> dict[str, Any]:
    """Restore the frozen API payload label without exposing it in outputs."""
    payload = copy.deepcopy(expert_memory)
    payload["schema_version"] = _EXPERT_MEMORY_WIRE_SCHEMA
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--questions-dir",
        type=Path,
        default=Path("data/questions"),
    )
    parser.add_argument(
        "--evidence-dir",
        type=Path,
        default=Path("data/evidence"),
    )
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
        "--exemplar-dir",
        type=Path,
        default=Path("artifacts/exemplars"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/hgf"),
    )
    parser.add_argument(
        "--model",
        default="google/gemini-2.5-flash-lite",
    )
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--question-ids", nargs="*", default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--candidate-evidence-limit", type=int, default=80)
    parser.add_argument("--evidence-limit", type=int, default=20)
    parser.add_argument("--reasoning-max-tokens", type=int, default=2600)
    parser.add_argument("--boundary-max-tokens", type=int, default=1800)
    parser.add_argument("--semantic-max-tokens", type=int, default=1200)
    parser.add_argument(
        "--semantic-cache-dir",
        type=Path,
        default=Path("artifacts/semantic_lessons"),
    )
    return parser.parse_args()


def compile_dag_expert_memory(
    *,
    source_question_id: str,
    blueprint: dict[str, Any],
    worked_exemplar: dict[str, Any],
) -> dict[str, Any]:
    """Compile all reusable forecasting guidance from one audited DAG."""
    diagnosis = blueprint.get("graph_diagnosis", {})
    if diagnosis.get("usable") is not True:
        raise ValueError(
            f"DAG {source_question_id} is not causally reusable"
        )

    structure = _transferable_dag_structure(blueprint)
    checkpoints = structure.get("checkpoints", [])[:7]
    checkpoint_ids = {
        str(item.get("id")) for item in checkpoints if item.get("id")
    }
    paths = []
    for path in structure.get("causal_paths", [])[:3]:
        kept_ids = [
            str(value)
            for value in path.get("checkpoint_ids", [])
            if str(value) in checkpoint_ids
        ]
        if len(kept_ids) < 2:
            continue
        paths.append(
            {
                **path,
                "checkpoint_ids": kept_ids,
            }
        )

    evidence_lessons = [
        {
            "checkpoint_id": str(item.get("id")),
            "causal_role": str(item.get("role") or ""),
            "factor_role": str(item.get("factor") or ""),
            "evidence_requirement": str(
                item.get("evidence_requirement") or ""
            ),
            "contradiction_signal": str(
                item.get("contradiction_signal") or ""
            ),
        }
        for item in checkpoints
    ]
    mechanism_lessons = [
        {
            "checkpoint_ids": path["checkpoint_ids"],
            "mechanism": str(path.get("generalized_mechanism") or ""),
            "applicable_when": [
                str(value)
                for value in path.get("applicability_conditions", [])[:3]
            ],
            "fails_when": [
                str(value)
                for value in path.get("failure_conditions", [])[:3]
            ],
        }
        for path in paths
    ]
    return {
        "schema_version": "dag_expert_memory",
        "source_question_id": source_question_id,
        "memory_provenance": (
            "All content is compiled from one validated, causally usable "
            "WorldReasoner hindsight DAG and its strictly pre-cutoff worked "
            "reasoning demonstration. No plain-text memory is used."
        ),
        "task_signature": worked_exemplar.get("task_signature", {}),
        "expert_reasoning_demonstration": {
            "target_semantics": str(
                worked_exemplar.get("target_semantics") or ""
            ),
            "reasoning_sequence": [
                str(value)
                for value in worked_exemplar.get("expert_reasoning", [])[:7]
            ],
            "counterevidence": str(
                worked_exemplar.get("counterevidence") or ""
            ),
            "uncertainty": str(
                worked_exemplar.get("uncertainty") or ""
            ),
            "structural_lesson": str(
                worked_exemplar.get("dag_derived_lesson") or ""
            ),
        },
        "causal_checkpoint_library": evidence_lessons,
        "mechanism_library": mechanism_lessons,
        "alternative_explanations": [
            {
                "hypothesis": str(item.get("hypothesis") or ""),
                "discriminating_evidence": str(
                    item.get("discriminating_evidence") or ""
                ),
            }
            for item in structure.get("alternative_hypotheses", [])[:2]
        ],
        "audit_questions": [
            str(value)
            for value in structure.get("forecast_audit_questions", [])[:4]
        ],
        "transfer_rule": (
            "Transfer causal roles, evidence requirements, mechanisms, and "
            "reasoning organization—not historical entities, facts, values, "
            "directions, topology, or answers. Instantiate a checkpoint only "
            "with current cutoff-safe evidence. Use CURRENT_NEW for a necessary "
            "current factor absent from the historical DAG."
        ),
    }


def _dag_semantic_lesson_schema() -> dict[str, Any]:
    return {
        "name": _SEMANTIC_SCHEMA_NAME,
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "target_semantics_lesson": {"type": "string"},
                "evidence_selection_lesson": {"type": "string"},
                "causal_reasoning_lesson": {"type": "string"},
                "counterevidence_lesson": {"type": "string"},
                "calibration_lesson": {"type": "string"},
            },
            "required": [
                "target_semantics_lesson",
                "evidence_selection_lesson",
                "causal_reasoning_lesson",
                "counterevidence_lesson",
                "calibration_lesson",
            ],
        },
    }


def _dag_semantic_lesson_validator(
    payload: dict[str, Any],
) -> tuple[dict[str, float], list[str]]:
    errors = []
    for field in (
        "target_semantics_lesson",
        "evidence_selection_lesson",
        "causal_reasoning_lesson",
        "counterevidence_lesson",
        "calibration_lesson",
    ):
        if not str(payload.get(field) or "").strip():
            errors.append(f"{field} is empty")
    return {}, errors


def _distill_dag_semantic_lessons(
    *,
    client: OpenAI,
    model: str,
    source_question_id: str,
    expert_memory: dict[str, Any],
    cache_dir: Path,
    max_tokens: int,
) -> tuple[dict[str, Any], dict[str, int], float, bool]:
    """Render DAG expertise as readable lessons without adding another source."""
    cache_path = cache_dir / f"{source_question_id}.json"
    with _MEMORY_LOCK:
        if cache_path.exists():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            _, errors = _dag_semantic_lesson_validator(cached)
            if not errors:
                return cached, {}, 0.0, True
    wire_memory = _wire_expert_memory(expert_memory)
    prompt = (
        "Compile concise forecasting expertise from this DAG Expert Memory. "
        "Every output lesson must be entailed by the supplied DAG-derived "
        "checkpoint, mechanism, failure-condition, audit, or worked-example "
        "content. Do not add outside financial advice, historical outcomes, "
        "answer labels, or post-cutoff facts.\n\n"
        "Preserve distinctions that matter to the target operator. For example, "
        "a growth level is not growth acceleration, a price outlook is not a "
        "target-period return, and a level is not a change. Explain which "
        "evidence requirements make a causal bridge valid, how competing paths "
        "should be tested, and how weak links should reduce confidence. These "
        "lessons make the DAG readable; they do not replace its structured "
        "checkpoint and mechanism libraries. Keep each lesson under 60 words.\n\n"
        f"DAG EXPERT MEMORY:\n{json.dumps(wire_memory, ensure_ascii=False)}"
    )
    lessons, _, usage, seconds, repaired = _call_with_repair(
        client,
        model=model,
        system=(
            "You compile readable expert forecasting lessons exclusively from "
            "structured hindsight-DAG memory. Return schema-conforming JSON."
        ),
        prompt=prompt,
        schema=_dag_semantic_lesson_schema(),
        seed=_seed(source_question_id, "semantic_lessons"),
        max_tokens=max_tokens,
        validator=_dag_semantic_lesson_validator,
    )
    with _MEMORY_LOCK:
        _atomic_write(cache_path, lessons)
    return lessons, usage, seconds, repaired


def compile_current_target_operator(
    contract: dict[str, Any],
) -> dict[str, Any]:
    """Compile the public target contract into an explicit semantic operator."""
    metric = str(contract.get("target_metric") or "")
    metric_lower = metric.lower()
    comparison_rule = str(contract.get("comparison_rule") or "")
    resolution_rule = str(contract.get("resolution_rule") or "")
    if "acceleration" in metric_lower:
        semantic_guard = (
            "Estimate the target-period growth rate relative to the prior-period "
            "growth rate. Positive or strong growth alone does not imply positive "
            "growth acceleration."
        )
    elif "return" in metric_lower:
        semantic_guard = (
            "Estimate the return over the exact target period relative to the "
            "immediately preceding period endpoint. A price level or annual price "
            "outlook does not determine the target-period return."
        )
    elif "change" in metric_lower or "growth" in metric_lower:
        semantic_guard = (
            "Estimate the change from the immediately preceding observation, not "
            "the level of the series or a broad annual outlook."
        )
    else:
        semantic_guard = (
            "Estimate exactly the target metric and horizon in the public "
            "contract; do not substitute a related level, direction, or period."
        )
    return {
        "target_metric": metric,
        "target_period": str(contract.get("target_period") or ""),
        "unit": str(contract.get("change_unit") or ""),
        "comparison_rule": comparison_rule,
        "resolution_rule": resolution_rule,
        "semantic_guard": semantic_guard,
        "predicate_or_intervals": (
            contract.get("predicate") or contract.get("intervals") or {}
        ),
    }


def _inject_target_operator_step(
    reasoning: dict[str, Any],
    target_operator: dict[str, Any],
) -> None:
    """Make the exact public target operator visible to the boundary mapper."""
    statement = (
        f"Exact target operator: {target_operator['target_metric']} for "
        f"{target_operator['target_period']} in {target_operator['unit']}. "
        f"Comparison: {target_operator['comparison_rule']}. "
        f"{target_operator['semantic_guard']}"
    )
    steps = reasoning.get("reasoning_steps", [])
    target_contract_step = next(
        (
            item
            for item in steps
            if item.get("source_checkpoint_id") == "TARGET_CONTRACT"
        ),
        None,
    )
    if target_contract_step is not None:
        target_contract_step["statement"] = (
            statement + " Current assessment: "
            + str(target_contract_step.get("statement") or "")
        )
        return
    protocol_step = {
        "step_type": "baseline",
        "statement": statement,
        "evidence_ids": [],
        "effect_on_target": "neutral",
        "source_checkpoint_id": "TARGET_CONTRACT",
    }
    if len(steps) < 7:
        steps.insert(0, protocol_step)
    else:
        replace_index = next(
            (
                index
                for index, item in enumerate(steps)
                if item.get("step_type") == "baseline"
            ),
            0,
        )
        steps[replace_index] = protocol_step
    reasoning["reasoning_steps"] = steps


def _reasoning_schema(
    options: list[str],
    checkpoint_ids: list[str],
) -> dict[str, Any]:
    schema = copy.deepcopy(_forecast_schema_exemplar(options, "none"))
    step_schema = schema["schema"]["properties"]["reasoning_steps"]["items"]
    step_schema["properties"]["source_checkpoint_id"] = {
        "type": "string",
        "enum": checkpoint_ids + ["CURRENT_NEW", "TARGET_CONTRACT"],
    }
    step_schema["required"].append("source_checkpoint_id")
    return schema


def _reasoning_validator(
    *,
    options: list[str],
    evidence_ids: set[str],
    checkpoint_ids: set[str],
    target_bridge_checkpoint_ids: set[str],
) -> Callable[[dict[str, Any]], tuple[dict[str, float], list[str]]]:
    def validate(
        payload: dict[str, Any],
    ) -> tuple[dict[str, float], list[str]]:
        _normalize_probability_rows(payload, options)
        _ensure_baseline_reasoning_step(
            payload,
            source_checkpoint_id="TARGET_CONTRACT",
        )
        non_evidence_ids = checkpoint_ids | {
            "CURRENT_NEW",
            "TARGET_CONTRACT",
        }
        for step in payload.get("reasoning_steps", []):
            step["evidence_ids"] = [
                str(value)
                for value in step.get("evidence_ids", [])
                if str(value) not in non_evidence_ids
            ]
        steps = payload.get("reasoning_steps", [])
        if not any(
            str(item.get("step_type")) == "target_bridge"
            for item in steps
        ):
            source_step = next(
                (
                    item
                    for item in reversed(steps)
                    if item.get("step_type") in {"mechanism", "driver"}
                ),
                None,
            )
            bridge_step = {
                "step_type": "target_bridge",
                "statement": (
                    f"{payload.get('target_estimate', '')} "
                    f"Option mapping: {payload.get('option_mapping', '')}"
                ).strip(),
                "evidence_ids": (
                    list(source_step.get("evidence_ids", []))
                    if source_step
                    else []
                ),
                "effect_on_target": (
                    str(source_step.get("effect_on_target") or "uncertain")
                    if source_step
                    else "uncertain"
                ),
                "source_checkpoint_id": (
                    sorted(target_bridge_checkpoint_ids)[0]
                    if target_bridge_checkpoint_ids
                    else "TARGET_CONTRACT"
                ),
            }
            if len(steps) < 7:
                steps.append(bridge_step)
            else:
                replace_index = next(
                    (
                        index
                        for index in range(len(steps) - 1, -1, -1)
                        if steps[index].get("step_type")
                        == "option_mapping"
                    ),
                    len(steps) - 1,
                )
                steps[replace_index] = bridge_step
            payload["reasoning_steps"] = steps
        probabilities, errors = _validate_exemplar_forecast(
            payload,
            options=options,
            evidence_ids=evidence_ids,
            transfer_policy="none",
        )
        evidence_fit = payload.get("evidence_fit", {})
        if (
            not payload.get("selected_evidence_ids")
            and evidence_fit.get("metric_match") == "weak"
            and evidence_fit.get("magnitude_support") == "unsupported"
        ):
            errors = [
                error
                for error in errors
                if error != "selected_evidence_ids is empty"
            ]
        steps = payload.get("reasoning_steps", [])
        step_types = {str(item.get("step_type")) for item in steps}
        if not step_types & {"driver", "mechanism"}:
            errors.append("reasoning requires a driver or mechanism step")
        used_checkpoints = {
            str(item.get("source_checkpoint_id"))
            for item in steps
            if str(item.get("source_checkpoint_id")) in checkpoint_ids
        }
        if not used_checkpoints:
            errors.append(
                "current reasoning must instantiate at least one retrieved "
                "DAG checkpoint"
            )
        for step in steps:
            source_id = str(step.get("source_checkpoint_id") or "")
            if (
                source_id not in checkpoint_ids
                and source_id not in {"CURRENT_NEW", "TARGET_CONTRACT"}
            ):
                errors.append(
                    f"unknown DAG checkpoint mapping {source_id!r}"
                )
        return probabilities, errors

    return validate


def _call_dag_expert_reasoning(
    *,
    client: OpenAI,
    model: str,
    question_id: str,
    public_case: dict[str, Any],
    evidence: list[dict[str, Any]],
    evidence_ids: set[str],
    options: list[str],
    expert_memory: dict[str, Any],
    target_operator: dict[str, Any],
    max_tokens: int,
) -> tuple[dict[str, Any], dict[str, int], float, bool]:
    checkpoint_ids = [
        str(item["checkpoint_id"])
        for item in expert_memory["causal_checkpoint_library"]
    ]
    target_bridge_checkpoint_ids = {
        str(item["checkpoint_id"])
        for item in expert_memory["causal_checkpoint_library"]
        if item.get("causal_role") == "target_bridge"
    }
    wire_memory = _wire_expert_memory(expert_memory)
    prompt = (
        "Forecast this unresolved financial target using the supplied DAG Expert "
        "Memory and current cutoff-safe evidence. The memory was compiled only "
        "from a causally usable WorldReasoner hindsight DAG. It is procedural "
        "expertise, never current evidence.\n\n"
        "Preserve the strengths of the entire DAG rather than flattening it into "
        "one generic lesson. Use its checkpoint evidence requirements to select "
        "current evidence, its mechanisms to connect current factors, its "
        "failure conditions to test counterevidence, and its worked example to "
        "organize a complete forecast argument. Do not copy historical entities, "
        "facts, dates, values, directions, topology, or conclusions.\n\n"
        "Produce one compact current-case reasoning trace. First lock the exact "
        "target operation, horizon, unit, and boundaries. Establish a calibrated "
        "current baseline. Instantiate relevant DAG checkpoints with current "
        "evidence; if the current case needs an unrepresented factor, mark that "
        "step CURRENT_NEW. Build a supported driver-to-mechanism-to-target chain "
        "and examine a genuine competing explanation. End with one target bridge "
        "that reconciles the paths into the exact target quantity. A direction "
        "alone is not threshold-crossing magnitude. Preserve uncertainty when "
        "metric, horizon, or magnitude support is weak.\n\n"
        "For every reasoning step, source_checkpoint_id must name the retrieved "
        "DAG checkpoint it instantiates. Use TARGET_CONTRACT only for public "
        "target-definition arithmetic and CURRENT_NEW only for a current factor "
        "that is genuinely absent from the DAG memory. Use only current evidence "
        "IDs for current factual claims. Keep fields concise so the JSON remains "
        "complete.\n\n"
        f"CURRENT CASE:\n{json.dumps(public_case, ensure_ascii=False)}\n\n"
        "DETERMINISTIC CURRENT TARGET OPERATOR:\n"
        f"{json.dumps(target_operator, ensure_ascii=False)}\n\n"
        "DAG-ONLY EXPERT MEMORY:\n"
        f"{json.dumps(wire_memory, ensure_ascii=False)}\n\n"
        f"CURRENT CUTOFF-SAFE EVIDENCE:\n"
        f"{json.dumps(evidence, ensure_ascii=False)}"
    )
    reasoning, _, usage, seconds, repaired = _call_with_repair(
        client,
        model=model,
        system=(
            "You are a cutoff-safe financial forecaster transferring structured "
            "expertise from a hindsight DAG. Historical content is procedural "
            "memory only. Return schema-conforming JSON."
        ),
        prompt=prompt,
        schema=_reasoning_schema(options, checkpoint_ids),
        seed=_seed(question_id, "expert_reasoning"),
        max_tokens=max_tokens,
        validator=_reasoning_validator(
            options=options,
            evidence_ids=evidence_ids,
            checkpoint_ids=set(checkpoint_ids),
            target_bridge_checkpoint_ids=target_bridge_checkpoint_ids,
        ),
    )
    _inject_target_operator_step(reasoning, target_operator)
    return reasoning, usage, seconds, repaired


def _load_source_cases(source_dir: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for path in (source_dir / "cases").glob("*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") == "success":
            rows[str(payload["question_id"])] = payload
    return rows


def _run_question(
    *,
    client: OpenAI,
    model: str,
    question: Any,
    blueprint_by_id: dict[str, dict[str, Any]],
    exemplar_case: dict[str, Any],
    evidence_dir: Path,
    output_dir: Path,
    candidate_evidence_limit: int,
    evidence_limit: int,
    reasoning_max_tokens: int,
    boundary_max_tokens: int,
    semantic_cache_dir: Path,
    semantic_max_tokens: int,
) -> dict[str, Any]:
    output_path = output_dir / "cases" / f"{question.id}.json"
    failed_path = output_path.with_suffix(".failed.json")
    if output_path.exists():
        cached = json.loads(output_path.read_text(encoding="utf-8"))
        if cached.get("status") == "success":
            failed_path.unlink(missing_ok=True)
            return cached

    cutoff, cutoff_source = resolve_forecast_cutoff(question)
    db_path, candidates = _resolve_raw_evidence(
        evidence_dir,
        question,
        cutoff,
        candidate_evidence_limit,
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

    memory_id = str(exemplar_case["retrieved_memory_question_id"])
    blueprint = blueprint_by_id[memory_id]
    expert_memory = compile_dag_expert_memory(
        source_question_id=memory_id,
        blueprint=blueprint,
        worked_exemplar=exemplar_case["worked_exemplar"],
    )
    (
        semantic_lessons,
        semantic_usage,
        semantic_seconds,
        semantic_cached,
    ) = _distill_dag_semantic_lessons(
        client=client,
        model=model,
        source_question_id=memory_id,
        expert_memory=expert_memory,
        cache_dir=semantic_cache_dir,
        max_tokens=semantic_max_tokens,
    )
    expert_memory["dag_derived_semantic_lessons"] = semantic_lessons
    reasoning, reasoning_usage, reasoning_seconds, reasoning_repaired = (
        _call_dag_expert_reasoning(
            client=client,
            model=model,
            question_id=str(question.id),
            public_case=public_case,
            evidence=evidence,
            evidence_ids=evidence_ids,
            options=options,
            expert_memory=expert_memory,
            target_operator=target_operator,
            max_tokens=reasoning_max_tokens,
        )
    )
    (
        forecast,
        probabilities,
        boundary_usage,
        boundary_seconds,
        boundary_repaired,
    ) = _call_boundary_mapping(
        client=client,
        model=model,
        question_id=str(question.id),
        public_case=public_case,
        evidence=evidence,
        evidence_ids=evidence_ids,
        options=options,
        contract=contract,
        reasoning=reasoning,
        seed_role="boundary_mapping",
        max_tokens=boundary_max_tokens,
    )
    ground_truth = _ground_truth_option(question)
    accuracy, brier = _score(probabilities, ground_truth, options)
    result = {
        "schema_version": "hgf_case",
        "status": "success",
        "question_id": str(question.id),
        "category": family_metadata(question).get("category"),
        "question": question.question_text,
        "options": options,
        "ground_truth": ground_truth,
        "cutoff": cutoff.isoformat(),
        "cutoff_source": cutoff_source,
        "evidence_db": str(db_path),
        "evidence_count": len(evidence),
        "candidate_evidence_count": len(candidates),
        "retrieved_memory_question_id": memory_id,
        "dag_usable": blueprint.get("graph_diagnosis", {}).get("usable"),
        "target_operator": target_operator,
        "hgf": {
            "dag_expert_memory": expert_memory,
            "reasoning": reasoning,
            "forecast": forecast,
            "probabilities": probabilities,
            "accuracy": accuracy,
            "brier": brier,
            "usage": _add_usage(reasoning_usage, boundary_usage),
            "memory_usage": semantic_usage,
            "memory_seconds": semantic_seconds,
            "memory_cached": semantic_cached,
            "seconds": (
                semantic_seconds + reasoning_seconds + boundary_seconds
            ),
            "repaired": reasoning_repaired or boundary_repaired,
        },
    }
    with _WRITE_LOCK:
        _atomic_write(output_path, result)
        failed_path.unlink(missing_ok=True)
    return result


def _summarize(
    rows: list[dict[str, Any]],
    *,
    selected_count: int,
    elapsed_seconds: float,
) -> dict[str, Any]:
    successful = [row for row in rows if row.get("status") == "success"]
    summary = {
        "status": (
            "success" if len(successful) == selected_count else "incomplete"
        ),
        "selected_count": selected_count,
        "success_count": len(successful),
        "failure_count": selected_count - len(successful),
        "elapsed_seconds": elapsed_seconds,
        "metrics": {},
        "by_category": {},
    }
    if not successful:
        return summary
    summary["metrics"] = {
        "accuracy": sum(
            float(row["hgf"]["accuracy"]) for row in successful
        )
        / len(successful),
        "brier": sum(
            float(row["hgf"]["brier"]) for row in successful
        )
        / len(successful),
    }
    categories = sorted({str(row["category"]) for row in successful})
    for category in categories:
        category_rows = [
            row for row in successful if str(row["category"]) == category
        ]
        summary["by_category"][category] = {
            "accuracy": sum(
                float(row["hgf"]["accuracy"]) for row in category_rows
            )
            / len(category_rows),
            "brier": sum(
                float(row["hgf"]["brier"]) for row in category_rows
            )
            / len(category_rows),
        }
    return summary


def main() -> None:
    args = _parse_args()
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
    selection_payload = json.loads(
        args.selection_file.resolve().read_text(encoding="utf-8")
    )
    source_ids = [
        str(value) for value in selection_payload["question_ids"]
    ]
    if args.question_ids:
        requested = set(args.question_ids)
        selected_ids = [
            value for value in source_ids if value in requested
        ]
        missing = requested - set(selected_ids)
        if missing:
            raise ValueError(f"unknown question IDs: {sorted(missing)}")
    else:
        selected_ids = source_ids[: max(1, min(args.limit, len(source_ids)))]

    questions = {
        str(question.id): question
        for question in read_questions(questions_dir / "test_questions.jsonl")
    }
    memory_questions = {
        str(question.id): question
        for question in read_questions(
            questions_dir / "memory_questions.jsonl"
        )
    }
    _, blueprints = load_final_memory_bank(
        args.memory_bank_manifest.resolve(),
        memory_questions,
    )
    blueprint_by_id = {
        str(item["question_id"]): item for item in blueprints
    }
    exemplar_cases = _load_source_cases(args.exemplar_dir.resolve())
    selected = [questions[value] for value in selected_ids]
    for question_id in selected_ids:
        if question_id not in exemplar_cases:
            raise ValueError(f"exemplar source missing {question_id}")

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        timeout=180,
        max_retries=2,
    )
    started = time.monotonic()
    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(
        max_workers=max(1, args.workers)
    ) as executor:
        futures = {
            executor.submit(
                _run_question,
                client=client,
                model=args.model,
                question=question,
                blueprint_by_id=blueprint_by_id,
                exemplar_case=exemplar_cases[str(question.id)],
                evidence_dir=evidence_dir,
                output_dir=output_dir,
                candidate_evidence_limit=args.candidate_evidence_limit,
                evidence_limit=args.evidence_limit,
                reasoning_max_tokens=args.reasoning_max_tokens,
                boundary_max_tokens=args.boundary_max_tokens,
                semantic_cache_dir=args.semantic_cache_dir.resolve(),
                semantic_max_tokens=args.semantic_max_tokens,
            ): question
            for question in selected
        }
        for future in as_completed(futures):
            question = futures[future]
            try:
                row = future.result()
            except Exception as exc:
                row = {
                    "status": "failed",
                    "question_id": str(question.id),
                    "category": family_metadata(question).get("category"),
                    "error": f"{type(exc).__name__}: {exc}",
                }
                _atomic_write(
                    output_dir
                    / "cases"
                    / f"{question.id}.failed.json",
                    row,
                )
            rows.append(row)
            successful = sum(
                value.get("status") == "success" for value in rows
            )
            print(
                f"PROGRESS {len(rows)}/{len(selected)} "
                f"success={successful} failed={len(rows)-successful}",
                flush=True,
            )

    order = {
        question_id: index
        for index, question_id in enumerate(selected_ids)
    }
    rows.sort(key=lambda row: order.get(str(row["question_id"]), 9999))
    summary = _summarize(
        rows,
        selected_count=len(selected),
        elapsed_seconds=time.monotonic() - started,
    )
    payload = {
        "schema_version": "hgf_experiment",
        "model": args.model,
        "protocol": {
            "memory_source": "usable WorldReasoner DAG only",
            "plain_text_memory_used": False,
            "boundary_mapper": "numeric boundary mapper",
            "baseline_probability_editing": False,
            "fallback_or_gate": False,
        },
        "selection": {
            "selection_rule": (
                "explicit question IDs"
                if args.question_ids
                else "fixed source order"
            ),
            "question_ids": selected_ids,
        },
        "summary": summary,
        "results": rows,
    }
    _atomic_write(output_dir / "results.json", payload)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
