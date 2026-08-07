#!/usr/bin/env python3
"""Run the controlled paper-method comparison on frozen cutoff-safe evidence.

The six baselines share the same model, target contract, output validator,
question IDs, and probability scorer.  Forecast-memory methods share the
question-only evidence bank (E0).  Factor Memory uses the factor-guided
evidence bank (E1).  Every method is checkpointed independently.
"""

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

from hgf.boundary import _call_boundary_mapping
from hgf.contracts import _is_temporally_eligible, _target_contract
from hgf.evidence_store import _direct_evidence_pack
from hgf.exemplar import (
    _add_usage,
    _call_with_repair,
    _ensure_baseline_reasoning_step,
    _forecast_schema_exemplar,
    _normalize_probability_rows,
    _rerank_current_evidence,
    _validate_exemplar_forecast,
)
from hgf.forecast_core import (
    _atomic_write,
    _call_json,
    _compile_single_dag_plan,
    _forecast_schema,
    _ground_truth_option,
    _normalize_single_dag_plan,
    _seed,
    _single_dag_plan_schema,
    _validate_forecast,
    _validate_graph,
    _validate_single_dag_plan,
)
from hgf.forecast_safety import score_forecast
from hgf.generation import configure_generation
from hgf.memory_bank import (
    load_factor_blueprint_bank,
    load_graph_bank,
)
from hgf.memory_retrieval import (
    compile_hgf_search_memory,
    compile_raw_dag_ablation,
    select_relevant_blueprints,
)
from hgf.question_io import (
    family_metadata,
    read_questions,
    resolve_forecast_cutoff,
)
from hgf.repair_resilience import neutral_reasoning_payload
from hgf.text_memory import _distill_text_memory

METHODS = (
    "search_only",
    "factor_memory",
    "case_memory",
    "text_memory",
    "direct_dag",
    "prospective_dag",
)
_FACTOR_MEMORY_WIRE_VIEW = "hgf_search_cards_v1"


METHOD_LABELS = {
    "search_only": "Search-only Agent",
    "factor_memory": "Factor-Memory Agent",
    "case_memory": "Case-Memory Agent",
    "text_memory": "Text-Memory Agent",
    "direct_dag": "Direct DAG Agent",
    "prospective_dag": "Prospective DAG Agent",
}

METHOD_REFERENCES = {
    "search_only": ["AutoCast++", "Human-level Forecasting"],
    "factor_memory": ["ExpeL", "AutoCast++"],
    "case_memory": ["A-Mem"],
    "text_memory": ["ExpeL"],
    "direct_dag": ["WorldReasoner"],
    "prospective_dag": ["WorldReasoner Search-Enabled Graph"],
}

_WRITE_LOCK = threading.Lock()


def _parse_args(
    *,
    default_methods: tuple[str, ...] = METHODS,
    default_output_dir: Path = Path("runs/main_table"),
) -> argparse.Namespace:
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
        "--output-dir",
        type=Path,
        default=default_output_dir,
    )
    parser.add_argument(
        "--selection-file",
        type=Path,
        default=Path("data/questions/selection.json"),
    )
    parser.add_argument(
        "--model",
        default="google/gemini-2.5-flash-lite",
    )
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--per-category", type=int, default=20)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=METHODS,
        default=default_methods,
    )
    parser.add_argument("--question-ids", nargs="*", default=None)
    parser.add_argument("--question-ids-file", type=Path)
    parser.add_argument("--candidate-evidence-limit", type=int, default=80)
    parser.add_argument("--evidence-limit", type=int, default=20)
    parser.add_argument("--reasoning-max-tokens", type=int, default=2600)
    parser.add_argument("--boundary-max-tokens", type=int, default=1800)
    parser.add_argument("--graph-max-tokens", type=int, default=2600)
    parser.add_argument("--semantic-max-tokens", type=int, default=1200)
    parser.add_argument(
        "--reasoning-effort",
        choices=("minimal", "low", "medium", "high", "xhigh", "max"),
    )
    parser.add_argument("--max-output-tokens", type=int)
    parser.add_argument("--run-seed", type=int, default=0)
    return parser.parse_args()


def _condition_evidence(
    evidence_dir: Path,
    question: Any,
    cutoff: Any,
    *,
    guided: bool,
    limit: int,
) -> tuple[Path, list[dict[str, Any]]]:
    """Load the frozen E0 or E1 database for one question."""
    question_id = str(question.id)
    bank = "e1" if guided else "e0"
    path = (evidence_dir / bank / f"{question_id}.sqlite").resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"missing {bank.upper()} evidence DB for {question_id}"
        )
    evidence = _direct_evidence_pack(
        path,
        question_id,
        cutoff,
        limit=limit,
    )
    return path, evidence


def _eligible_retrieval(
    *,
    blueprints: list[dict[str, Any]],
    memory_questions: dict[str, Any],
    target_question: Any,
    cutoff: Any,
    evidence: list[dict[str, Any]],
) -> dict[str, Any]:
    eligible = [
        blueprint
        for blueprint in blueprints
        if (
            str(blueprint.get("question_id")) in memory_questions
            and _is_temporally_eligible(
                memory_questions[str(blueprint["question_id"])],
                cutoff,
            )
        )
    ]
    selected = select_relevant_blueprints(
        eligible,
        memory_questions,
        target_question,
        limit=1,
        evidence=evidence,
    )
    if not selected:
        raise ValueError(
            f"no cutoff-eligible usable memory for {target_question.id}"
        )
    return selected[0]


def _case_memory(
    *,
    memory_question: Any,
    memory_graph: dict[str, Any],
) -> dict[str, Any]:
    """Represent one past episode without graph structure or graph lessons."""
    historical_cutoff, _ = resolve_forecast_cutoff(memory_question)
    articles = [
        {
            "title": item.get("title"),
            "published_date": item.get("published_date"),
            "source": item.get("source"),
            "snippet": str(
                item.get("snippet") or item.get("content") or ""
            )[:700],
        }
        for item in memory_graph.get("evidence", {}).get("articles", [])
        if str(item.get("published_date") or "")[:10]
        < historical_cutoff.date().isoformat()
    ][:12]
    return {
        "memory_type": "resolved_episode",
        "question": memory_question.question_text,
        "context": memory_question.context,
        "target_contract": _target_contract(memory_question),
        "historical_cutoff": historical_cutoff.isoformat(),
        "resolved_option": str(memory_question.ground_truth),
        "historical_forecast_time_evidence": articles,
        "instruction": (
            "Use this only as an analogous resolved episode. Historical facts "
            "and its resolved option are never evidence for the current target."
        ),
    }


def _call_memory_reasoning(
    *,
    client: OpenAI,
    model: str,
    question_id: str,
    public_case: dict[str, Any],
    evidence: list[dict[str, Any]],
    options: list[str],
    memory_type: str,
    memory: Any | None,
    max_tokens: int,
    allow_neutral_fallback: bool = False,
) -> tuple[dict[str, Any], dict[str, int], float, bool]:
    evidence_ids = {str(item["id"]) for item in evidence}
    wire_memory = memory
    if isinstance(memory, dict) and memory.get("view") == "hgf_search_cards":
        wire_memory = {**memory, "view": _FACTOR_MEMORY_WIRE_VIEW}
    memory_instruction = {
        "none": (
            "No historical memory is available. Reason only from the current "
            "question and evidence."
        ),
        "factor": (
            "Factor search cards are supplied as query-expansion and coverage "
            "hints. Use them to identify current drivers and counterevidence to "
            "evaluate, but do not treat the cards, historical outcomes, entities, "
            "dates, values, or causal edges as current evidence."
        ),
        "case": (
            "A retrieved resolved episode is supplied. Use it as an analogy, "
            "but never treat its answer, facts, entity, dates, or values as "
            "current evidence."
        ),
        "text": (
            "Outcome-redacted textual experience is supplied. Apply its general "
            "search and reasoning lessons without assuming the old mechanism "
            "holds now."
        ),
        "raw_dag": (
            "A redacted hindsight DAG is supplied directly. It is a historical "
            "structure, not current evidence. Decide which roles and relations "
            "are supported now; do not copy its topology or direction blindly."
        ),
    }[memory_type]
    prompt = (
        "Produce a compact, auditable forecast for the unresolved financial "
        "target. Lock the exact metric, horizon, unit, and option boundaries. "
        "Establish a current baseline, identify supported drivers and genuine "
        "counterevidence, connect them through a mechanism to the exact target, "
        "and preserve uncertainty when magnitude support is weak. Do not confuse "
        "level with change, growth with acceleration, or direction with boundary "
        "crossing. Use only current evidence IDs for current factual claims. "
        f"{memory_instruction}\n\n"
        f"CURRENT CASE:\n{json.dumps(public_case, ensure_ascii=False)}\n\n"
        f"MEMORY:\n{json.dumps(wire_memory, ensure_ascii=False)}\n\n"
        f"CURRENT EVIDENCE:\n{json.dumps(evidence, ensure_ascii=False)}"
    )
    def validator(
        payload: dict[str, Any],
    ) -> tuple[dict[str, float], list[str]]:
        _normalize_probability_rows(payload, options)
        _ensure_baseline_reasoning_step(payload)
        raw_selected = payload.get("selected_evidence_ids") or []
        if not isinstance(raw_selected, list):
            raw_selected = []
        payload["selected_evidence_ids"] = [
            str(value)
            for value in raw_selected
            if str(value) in evidence_ids
        ]
        for step in payload.get("reasoning_steps", []):
            step["evidence_ids"] = [
                str(value)
                for value in (step.get("evidence_ids") or [])
                if str(value) in evidence_ids
            ]
        if not payload["selected_evidence_ids"]:
            payload["selected_evidence_ids"] = sorted(
                {
                    str(value)
                    for step in payload.get("reasoning_steps", [])
                    for value in step.get("evidence_ids", [])
                    if str(value) in evidence_ids
                }
            )
        steps = payload.get("reasoning_steps", [])
        if not any(
            step.get("step_type") == "target_bridge" for step in steps
        ):
            source = next(
                (
                    step
                    for step in reversed(steps)
                    if step.get("step_type") in {"mechanism", "driver"}
                ),
                None,
            )
            bridge = {
                "step_type": "target_bridge",
                "statement": (
                    f"{payload.get('target_estimate', '')} "
                    f"{payload.get('option_mapping', '')}"
                ).strip(),
                "evidence_ids": (
                    list(source.get("evidence_ids", [])) if source else []
                ),
                "effect_on_target": (
                    str(source.get("effect_on_target") or "uncertain")
                    if source
                    else "uncertain"
                ),
            }
            if len(steps) < 7:
                steps.append(bridge)
            elif steps:
                steps[-1] = bridge
            payload["reasoning_steps"] = steps
        probabilities, errors = _validate_exemplar_forecast(
            payload,
            options=options,
            evidence_ids=evidence_ids,
            transfer_policy="none",
        )
        evidence_fit = payload.get("evidence_fit", {})
        if not isinstance(evidence_fit, dict):
            evidence_fit = {"assessment": str(evidence_fit)}
            payload["evidence_fit"] = evidence_fit
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
        return probabilities, errors
    reasoning, _, usage, seconds, repaired = _call_with_repair(
        client,
        model=model,
        system=(
            "You are a cutoff-safe financial forecaster. Memory, when present, "
            "is process guidance only. Return schema-conforming JSON."
        ),
        prompt=prompt,
        schema=_forecast_schema_exemplar(options, "none"),
        seed=_seed(question_id, f"paper-{memory_type}-reasoning"),
        max_tokens=max_tokens,
        validator=validator,
        fallback_factory=(
            lambda _current, _errors: neutral_reasoning_payload(
                options=options,
                target_semantics=(
                    str(public_case.get("question") or "")
                    + " "
                    + json.dumps(
                        public_case.get("target_contract") or {},
                        ensure_ascii=False,
                    )
                ),
                include_checkpoint_mapping=False,
            )
        )
        if allow_neutral_fallback
        else None,
    )
    return reasoning, usage, seconds, repaired


def _validate_no_memory_plan(
    plan: dict[str, Any],
    evidence_ids: set[str],
) -> list[str]:
    errors = _validate_single_dag_plan(
        plan,
        evidence_ids=evidence_ids,
        checkpoint_ids=set(),
    )
    return [
        error
        for error in errors
        if error
        not in {
            "no retrieved checkpoint instantiated",
            "factor[0] has no evidence",
        }
    ]


def _prospective_dag_forecast(
    *,
    client: OpenAI,
    model: str,
    question_id: str,
    public_case: dict[str, Any],
    evidence: list[dict[str, Any]],
    options: list[str],
    contract: dict[str, Any],
    max_tokens: int,
) -> tuple[
    dict[str, Any],
    dict[str, float],
    dict[str, Any],
    dict[str, int],
    float,
    bool,
]:
    evidence_ids = {str(item["id"]) for item in evidence}
    graph_prompt = (
        "Build one prospective DAG for this unresolved target using only current "
        "cutoff-safe evidence. Do not use historical memory and do not forecast "
        "an option in this stage. Return exactly one prior-period baseline, two "
        "current drivers, one countervailing factor, one synthesis mechanism, "
        "and one target bridge. Set every checkpoint_id to NONE. Cite current "
        "evidence IDs for factual nodes. The target bridge must encode the exact "
        "metric and boundaries without selecting an answer.\n\n"
        f"CURRENT CASE:\n{json.dumps(public_case, ensure_ascii=False)}\n\n"
        f"CURRENT EVIDENCE:\n{json.dumps(evidence, ensure_ascii=False)}"
    )
    plan, graph_usage, graph_seconds = _call_json(
        client,
        model=model,
        system=(
            "You build one evidence-grounded prospective DAG and never select "
            "the forecast answer in the graph-construction stage."
        ),
        prompt=graph_prompt,
        schema=_single_dag_plan_schema(),
        seed=_seed(question_id, "paper-prospective-dag"),
        max_tokens=max_tokens,
    )
    plan, corrections = _normalize_single_dag_plan(
        plan,
        {"checkpoints": []},
    )
    if plan.get("baseline", {}).get("temporal_relation") != (
        "historical_baseline"
    ):
        plan.setdefault("baseline", {})["temporal_relation"] = (
            "historical_baseline"
        )
        corrections.append("baseline temporal relation normalized")
    errors = _validate_no_memory_plan(plan, evidence_ids)
    if errors:
        repair_prompt = (
            f"{graph_prompt}\n\nRepair the following DAG plan once. Fix every "
            "validation error without changing the required node counts. Every "
            "baseline, driver, countervailing factor, and synthesis node must "
            "cite at least one valid CURRENT evidence ID. Every checkpoint_id "
            "must remain NONE. Do not select a forecast answer.\n\n"
            f"ERRORS:\n{json.dumps(errors)}\n\n"
            f"INVALID PLAN:\n{json.dumps(plan, ensure_ascii=False)}"
        )
        repaired_plan, repair_usage, repair_seconds = _call_json(
            client,
            model=model,
            system=(
                "You repair one current-evidence prospective DAG. Return only "
                "schema-conforming JSON."
            ),
            prompt=repair_prompt,
            schema=_single_dag_plan_schema(),
            seed=_seed(question_id, "paper-prospective-dag-repair"),
            max_tokens=max_tokens,
        )
        plan, repair_corrections = _normalize_single_dag_plan(
            repaired_plan,
            {"checkpoints": []},
        )
        if plan.get("baseline", {}).get("temporal_relation") != (
            "historical_baseline"
        ):
            plan.setdefault("baseline", {})["temporal_relation"] = (
                "historical_baseline"
            )
            repair_corrections.append(
                "baseline temporal relation normalized"
            )
        corrections.extend(repair_corrections)
        graph_usage = _add_usage(graph_usage, repair_usage)
        graph_seconds += repair_seconds
        errors = _validate_no_memory_plan(plan, evidence_ids)
        if errors:
            raise ValueError(
                "prospective DAG plan invalid after repair: "
                + "; ".join(errors)
            )
    graph = _compile_single_dag_plan(
        plan,
        question_text=public_case["question"],
        checkpoint_ids=[],
    )
    graph_errors = [
        error
        for error in _validate_graph(
            graph,
            evidence_ids=evidence_ids,
            checkpoint_ids=set(),
            options=set(options),
        )
        if error
        not in {
            "no memory checkpoint instantiated",
            "BASE has no current evidence",
        }
    ]
    if graph_errors:
        raise ValueError(
            "prospective DAG invalid: " + "; ".join(graph_errors)
        )
    linked_ids = {
        str(article_id)
        for node in graph.get("nodes", [])
        for article_id in node.get("evidence_article_ids", [])
    }
    linked_evidence = [
        item for item in evidence if str(item["id"]) in linked_ids
    ]
    forecast_prompt = (
        "Forecast only from the validated current-case DAG, its linked current "
        "evidence, and the public target contract. Trace the main and "
        "countervailing paths to a target estimate before mapping it once to the "
        "options. Preserve uncertainty when the graph lacks exact metric or "
        "magnitude support.\n\n"
        f"TARGET CONTRACT:\n{json.dumps(contract, ensure_ascii=False)}\n\n"
        f"PROSPECTIVE DAG:\n{json.dumps(graph, ensure_ascii=False)}\n\n"
        "LINKED CURRENT EVIDENCE:\n"
        f"{json.dumps(linked_evidence, ensure_ascii=False)}"
    )
    (
        forecast,
        probabilities,
        forecast_usage,
        forecast_seconds,
        forecast_repaired,
    ) = _call_with_repair(
        client,
        model=model,
        system=(
            "You are a graph-grounded cutoff-safe financial forecaster. Return "
            "schema-conforming JSON."
        ),
        prompt=forecast_prompt,
        schema=_forecast_schema(options, graph_arm=True),
        seed=_seed(question_id, "paper-prospective-dag-forecast"),
        max_tokens=max(1000, max_tokens // 2),
        validator=lambda payload: _validate_forecast(
            payload,
            options=options,
            allowed_ids={str(item["id"]) for item in graph["nodes"]},
            graph_arm=True,
            evidence_ids=linked_ids,
            graph=graph,
        ),
    )
    return (
        forecast,
        probabilities,
        {
            "plan": plan,
            "graph": graph,
            "corrections": corrections,
            "forecast_repaired": forecast_repaired,
        },
        _add_usage(graph_usage, forecast_usage),
        graph_seconds + forecast_seconds,
        forecast_repaired,
    )


def _method_metrics(
    probabilities: dict[str, float],
    ground_truth: str,
    options: list[str],
    *,
    explicit_prediction: str | None = None,
) -> dict[str, float]:
    accuracy, brier = score_forecast(
        probabilities=probabilities,
        explicit_prediction=explicit_prediction,
        ground_truth=ground_truth,
        options=options,
    )
    truth_probability = max(float(probabilities[ground_truth]), 1e-6)
    return {
        "accuracy": accuracy,
        "brier": brier,
        "nll": -math.log(truth_probability),
        "confidence": max(float(value) for value in probabilities.values()),
    }


def _run_method(
    *,
    client: OpenAI,
    method: str,
    model: str,
    question: Any,
    evidence_dir: Path,
    output_dir: Path,
    memory_questions: dict[str, Any],
    graphs_by_id: dict[str, dict[str, Any]],
    factor_blueprints: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    output_path = output_dir / "cases" / str(question.id) / f"{method}.json"
    failed_path = output_path.with_suffix(".failed.json")
    if output_path.exists():
        cached = json.loads(output_path.read_text(encoding="utf-8"))
        if cached.get("status") == "success":
            failed_path.unlink(missing_ok=True)
            return cached

    started = time.monotonic()
    cutoff, cutoff_source = resolve_forecast_cutoff(question)
    options = [str(option) for option in question.options or []]
    contract = _target_contract(question)
    public_case = {
        "question": question.question_text,
        "context": question.context,
        "target_contract": contract,
        "cutoff": cutoff.isoformat(),
    }
    ground_truth = _ground_truth_option(question)

    guided = method == "factor_memory"
    evidence_bank = "E1" if guided else "E0"
    db_path, candidates = _condition_evidence(
        evidence_dir,
        question,
        cutoff,
        guided=guided,
        limit=args.candidate_evidence_limit,
    )
    evidence = _rerank_current_evidence(
        question,
        candidates,
        limit=args.evidence_limit,
    )
    evidence_db_payload: Any = str(db_path)
    evidence_ids = {str(item["id"]) for item in evidence}

    blueprint = _eligible_retrieval(
        blueprints=factor_blueprints,
        memory_questions=memory_questions,
        target_question=question,
        cutoff=cutoff,
        evidence=evidence,
    )
    memory_id = str(blueprint["question_id"])
    memory_question = memory_questions[memory_id]
    memory_graph = graphs_by_id[memory_id]
    memory_payload: Any | None = None
    memory_type = "none"
    memory_usage: dict[str, int] = {}
    memory_seconds = 0.0
    memory_cached = True
    if method == "factor_memory":
        memory_type = "factor"
        memory_payload = json.loads(
            compile_hgf_search_memory([blueprint])
        )
    elif method == "case_memory":
        memory_type = "case"
        memory_payload = _case_memory(
            memory_question=memory_question,
            memory_graph=memory_graph,
        )
    elif method == "text_memory":
        memory_type = "text"
        distilled = _distill_text_memory(
            client=client,
            model=model,
            memory_question=memory_question,
            memory_graph=memory_graph,
            cache_dir=output_dir / "memory_cache" / "text",
            max_tokens=args.semantic_max_tokens,
        )
        memory_payload = distilled["memory"]
        memory_usage = distilled.get("usage", {})
        memory_seconds = float(distilled.get("seconds") or 0)
        memory_cached = bool(distilled.get("cached"))
    elif method == "direct_dag":
        memory_type = "raw_dag"
        memory_payload = json.loads(
            compile_raw_dag_ablation([memory_graph])
        )

    if method == "prospective_dag":
        (
            forecast,
            probabilities,
            reasoning_artifact,
            usage,
            seconds,
            repaired,
        ) = _prospective_dag_forecast(
            client=client,
            model=model,
            question_id=str(question.id),
            public_case=public_case,
            evidence=evidence,
            options=options,
            contract=contract,
            max_tokens=args.graph_max_tokens,
        )
        reasoning = reasoning_artifact
    else:
        (
            reasoning,
            reasoning_usage,
            reasoning_seconds,
            reasoning_repaired,
        ) = _call_memory_reasoning(
            client=client,
            model=model,
            question_id=str(question.id),
            public_case=public_case,
            evidence=evidence,
            options=options,
            memory_type=memory_type,
            memory=memory_payload,
            max_tokens=args.reasoning_max_tokens,
            allow_neutral_fallback=True,
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
            seed_role=f"paper-{method}-boundary",
            max_tokens=args.boundary_max_tokens,
            allow_neutral_fallback=True,
        )
        usage = _add_usage(
            memory_usage,
            reasoning_usage,
            boundary_usage,
        )
        seconds = (
            memory_seconds + reasoning_seconds + boundary_seconds
        )
        repaired = reasoning_repaired or boundary_repaired

    metrics = _method_metrics(
        probabilities,
        ground_truth,
        options,
        explicit_prediction=str(forecast.get("prediction") or ""),
    )
    result = {
        "schema_version": "paper_method_case",
        "status": "success",
        "method": method,
        "method_label": METHOD_LABELS[method],
        "references": METHOD_REFERENCES[method],
        "question_id": str(question.id),
        "category": family_metadata(question).get("category"),
        "question": question.question_text,
        "options": options,
        "ground_truth": ground_truth,
        "cutoff": cutoff.isoformat(),
        "cutoff_source": cutoff_source,
        "evidence_bank": evidence_bank,
        "evidence_db": evidence_db_payload,
        "evidence_count": len(evidence),
        "evidence_ids": sorted(evidence_ids),
        "retrieved_memory_question_id": (
            memory_id
            if method not in {"search_only", "prospective_dag"}
            else None
        ),
        "memory": memory_payload,
        "memory_cached": memory_cached,
        "reasoning": reasoning,
        "forecast": forecast,
        "probabilities": probabilities,
        "metrics": metrics,
        "usage": usage,
        "seconds": seconds,
        "elapsed_seconds": time.monotonic() - started,
        "repaired": repaired,
    }
    with _WRITE_LOCK:
        _atomic_write(output_path, result)
        failed_path.unlink(missing_ok=True)
    return result


def _ece(rows: list[dict[str, Any]], bins: int = 5) -> float:
    if not rows:
        return math.nan
    total = 0.0
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        bucket = [
            row
            for row in rows
            if (
                lower <= row["metrics"]["confidence"] < upper
                or (
                    index == bins - 1
                    and row["metrics"]["confidence"] == 1.0
                )
            )
        ]
        if not bucket:
            continue
        total += len(bucket) / len(rows) * abs(
            fmean(row["metrics"]["confidence"] for row in bucket)
            - fmean(row["metrics"]["accuracy"] for row in bucket)
        )
    return total


def _metric_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n": len(rows),
        "accuracy": fmean(row["metrics"]["accuracy"] for row in rows),
        "brier": fmean(row["metrics"]["brier"] for row in rows),
        "nll": fmean(row["metrics"]["nll"] for row in rows),
        "ece_5bin": _ece(rows),
        "mean_total_tokens": fmean(
            float(row.get("usage", {}).get("total_tokens") or 0)
            for row in rows
        ),
        "mean_seconds": fmean(float(row.get("seconds") or 0) for row in rows),
    }


def _summarize(
    rows: list[dict[str, Any]],
    *,
    methods: list[str],
    selected_count: int,
    elapsed_seconds: float,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "selected_questions": selected_count,
        "expected_runs": selected_count * len(methods),
        "completed_runs": sum(row.get("status") == "success" for row in rows),
        "failed_runs": sum(row.get("status") != "success" for row in rows),
        "elapsed_seconds": elapsed_seconds,
        "overall": {},
        "by_category": {},
    }
    for method in methods:
        method_rows = [
            row
            for row in rows
            if row.get("status") == "success"
            and row.get("method") == method
        ]
        if method_rows:
            summary["overall"][method] = _metric_summary(method_rows)
    categories = sorted(
        {
            str(row.get("category"))
            for row in rows
            if row.get("status") == "success"
        }
    )
    for category in categories:
        summary["by_category"][category] = {}
        for method in methods:
            category_rows = [
                row
                for row in rows
                if row.get("status") == "success"
                and row.get("method") == method
                and str(row.get("category")) == category
            ]
            if category_rows:
                summary["by_category"][category][method] = (
                    _metric_summary(category_rows)
                )
    return summary


def main(
    *,
    default_methods: tuple[str, ...] = METHODS,
    default_output_dir: Path = Path("runs/main_table"),
) -> None:
    args = _parse_args(
        default_methods=default_methods,
        default_output_dir=default_output_dir,
    )
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
    bundle_env = Path(__file__).resolve().parents[1] / ".env"
    load_dotenv(
        bundle_env if bundle_env.exists() else find_dotenv(usecwd=True)
    )
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")

    questions_dir = args.questions_dir.resolve()
    evidence_dir = args.evidence_dir.resolve()
    output_dir = args.output_dir.resolve()
    question_list = read_questions(questions_dir / "test_questions.jsonl")
    questions = {str(question.id): question for question in question_list}
    memory_list = read_questions(questions_dir / "memory_questions.jsonl")
    memory_questions = {
        str(question.id): question for question in memory_list
    }
    graphs_by_id = load_graph_bank(
        args.memory_bank_manifest.resolve(),
        memory_questions,
    )
    factor_artifact_root = (
        Path("artifacts/baselines/factor_memory").resolve()
    )
    factor_blueprints_by_id = load_factor_blueprint_bank(
        factor_artifact_root,
        expected_ids=set(memory_questions),
    )
    factor_blueprints = [
        factor_blueprints_by_id[question_id]
        for question_id in memory_questions
    ]
    factor_manifest_path = factor_artifact_root / "manifest.json"
    frozen_selection = json.loads(
        args.selection_file.resolve().read_text(encoding="utf-8")
    )["question_ids"]
    if args.question_ids:
        missing = sorted(set(args.question_ids) - set(questions))
        if missing:
            raise ValueError(f"unknown question IDs: {missing}")
        selected = [
            questions[question_id]
            for question_id in frozen_selection
            if question_id in set(args.question_ids)
        ][: args.limit]
        selection_rule = "explicit IDs in fixed order"
    elif args.question_ids_file:
        requested = {
            line.strip()
            for line in args.question_ids_file.read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        }
        missing = sorted(requested - set(questions))
        if missing:
            raise ValueError(f"unknown question IDs: {missing}")
        selected = [
            questions[question_id]
            for question_id in frozen_selection
            if question_id in requested
        ][: args.limit]
        selection_rule = "ID file in fixed order"
    else:
        selected = [
            questions[question_id]
            for question_id in frozen_selection[: args.limit]
        ]
        selection_rule = "fixed 100-question selection"
    selected_ids = [str(question.id) for question in selected]
    _atomic_write(
        output_dir / "protocol.json",
        {
            "schema_version": "paper_method_protocol",
            "model": args.model,
            "workers": args.workers,
            "generation": {
                "reasoning_effort": args.reasoning_effort,
                "max_output_tokens": args.max_output_tokens,
                "run_seed": args.run_seed,
            },
            "baseline_configuration": {
                "factor_memory_manifest": str(factor_manifest_path),
                "factor_memory_manifest_sha256": hashlib.sha256(
                    factor_manifest_path.read_bytes()
                ).hexdigest(),
                "factor_memory_frozen": True,
            },
            "methods": list(args.methods),
            "method_labels": METHOD_LABELS,
            "method_references": METHOD_REFERENCES,
            "selection_rule": selection_rule,
            "question_ids": selected_ids,
            "categories": [
                family_metadata(question).get("category")
                for question in selected
            ],
            "evidence_contract": {
                "E0": [
                    "search_only",
                    "case_memory",
                    "text_memory",
                    "direct_dag",
                    "prospective_dag",
                ],
                "E1": ["factor_memory"],
                "cutoff_checked_on_every_article": True,
            },
            "shared": {
                "target_contract": True,
                "model": True,
                "probability_scorer": True,
                "boundary_mapper_for_non_graph_methods": True,
                "baseline_probability_editing": False,
            },
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
        (question, method)
        for question in selected
        for method in args.methods
    ]
    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(
        max_workers=max(1, args.workers)
    ) as executor:
        futures = {
            executor.submit(
                _run_method,
                client=client,
                method=method,
                model=args.model,
                question=question,
                evidence_dir=evidence_dir,
                output_dir=output_dir,
                memory_questions=memory_questions,
                graphs_by_id=graphs_by_id,
                factor_blueprints=factor_blueprints,
                args=args,
            ): (question, method)
            for question, method in tasks
        }
        for future in as_completed(futures):
            question, method = futures[future]
            try:
                row = future.result()
            except Exception as exc:
                row = {
                    "schema_version": "paper_method_case",
                    "status": "failed",
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "question_id": str(question.id),
                    "category": family_metadata(question).get("category"),
                    "error": f"{type(exc).__name__}: {exc}",
                }
                _atomic_write(
                    output_dir
                    / "cases"
                    / str(question.id)
                    / f"{method}.failed.json",
                    row,
                )
            rows.append(row)
            completed = sum(
                item.get("status") == "success" for item in rows
            )
            print(
                f"PROGRESS {len(rows)}/{len(tasks)} "
                f"success={completed} failed={len(rows)-completed}",
                flush=True,
            )
    order = {
        (question_id, method): (
            question_index * len(args.methods) + method_index
        )
        for question_index, question_id in enumerate(selected_ids)
        for method_index, method in enumerate(args.methods)
    }
    rows.sort(
        key=lambda row: order.get(
            (str(row["question_id"]), str(row["method"])),
            999999,
        )
    )
    summary = _summarize(
        rows,
        methods=list(args.methods),
        selected_count=len(selected),
        elapsed_seconds=time.monotonic() - started,
    )
    payload = {
        "schema_version": "paper_method_experiment",
        "model": args.model,
        "workers": args.workers,
        "generation": {
            "reasoning_effort": args.reasoning_effort,
            "max_output_tokens": args.max_output_tokens,
            "run_seed": args.run_seed,
        },
        "methods": list(args.methods),
        "selection": {
            "selection_rule": selection_rule,
            "question_ids": selected_ids,
        },
        "summary": summary,
        "results": rows,
    }
    _atomic_write(output_dir / "results.json", payload)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
